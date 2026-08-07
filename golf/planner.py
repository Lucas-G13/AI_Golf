"""Filling in the parts of the swing that cannot be scripted by hand.

`golf.swing` writes down the trunk and the lead arm.  This module turns that
into a complete, achievable trajectory:

* `plan_legs`     -- the pelvis turn, converted into leg angles that keep both
                     feet on their pins
* `plan_lead_arm` -- the follow-through, solved by clubhead position
* `plan_trail_arm`-- the trail arm, which is never scripted because both hands
                     on one shaft is a closed loop
* `tune_swing`    -- a few dry swings that adjust the plan until the fast part
                     of the arc is at ball height
* `place_ball`    -- one more dry swing that puts the ball where the club
                     actually goes

All of the planning runs on a scratch `MjData`, so the live simulation is never
disturbed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from .equipment import BALL_CENTRE_HEIGHT, BALL_RADIUS
from .joints import JOINT_NAMES
from .swing import Phase
from .util import DEG, RAD, quat_axis

_LEG_JOINTS = [j for j in JOINT_NAMES
               if j.startswith(("hip_", "knee_", "ankle_"))]


class SwingPlanner:
    """Mixin: everything that turns the script into a swing that works."""

    # ---- scratch state -----------------------------------------------------
    @contextmanager
    def _scratch(self):
        """Run a block against a throwaway MjData so planning never disturbs
        the live simulation state."""
        live = self.data
        self.data = mujoco.MjData(self.model)
        self.tracker.data = self.data
        try:
            yield
        finally:
            self.data = live
            self.tracker.data = live

    def _apply_root(self, phase: str) -> None:
        """Put the free root where `plan_legs` decided it should be for this
        keyframe, so the arm solvers see the same pelvis the legs produced."""
        q = self._root_plan.get(phase)
        if q is not None:
            self.data.qpos[self.root_qadr:self.root_qadr + 7] = q

    # ---- planning ----------------------------------------------------------
    def plan_legs(self, script: List[Phase]) -> None:
        """Turn each keyframe's pelvis rotation into leg angles.

        A golfer with both feet planted turns the pelvis by twisting the whole
        lower body; the hips, knees and ankles all move together and no single
        joint 'is' the turn.  Scripting hip rotation directly does not work --
        the abductors and knees hold their address angles and simply block it,
        and the pelvis ends up turning 5 deg instead of 40.  So the pelvis is
        rotated to where it should be and the legs are solved to keep the feet
        on their pins.
        """
        self._root_plan: Dict[str, np.ndarray] = {}
        if self.base == "pinned":
            return
        pins = dict(self._pins or {})
        r = self.root_qadr
        q_address = self.qpos_address[r + 3:r + 7].copy()

        with self._scratch():
            for ph in script:
                self.data.qpos[:] = self.qpos_address
                self._set_pose(ph.pose)
                q = np.zeros(4)
                mujoco.mju_mulQuat(
                    q, quat_axis((0, 0, 1), ph.pelvis_turn * DEG), q_address)
                self.data.qpos[r + 3:r + 7] = q
                mujoco.mj_forward(self.model, self.data)

                if pins:
                    res = self._ik_multi(list(pins.items()), _LEG_JOINTS,
                                         rest_pull=0.15, free_root=True)
                    if self.verbose and res > 0.02:
                        print(f"  [warn] '{ph.name}': feet {res * 100:.1f} cm "
                              f"off their pins at {ph.pelvis_turn:+.0f} deg of "
                              f"pelvis turn")
                for j in _LEG_JOINTS:
                    ph.pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD
                self._root_plan[ph.name] = self.data.qpos[r:r + 7].copy()

    def plan_lead_arm(self, script: List[Phase], dz: float = 0.0) -> None:
        """Solve the lead arm where the club has to be in a particular place.

        Only the follow-through is solved this way.  Aiming the *impact* pose
        at the ball as well looks right on paper, but the solver gets there by
        pulling the hands out over the ball until the shaft hangs straight
        down -- which puts the clubhead almost on the axis the body is rotating
        about, so it arrives doing 1 m/s.  Impact instead reuses the address
        arm geometry, where the club is properly extended out to the ball, and
        the body turn swings it through.

        The wrist is deliberately not solved for either: given the freedom it
        keeps 80 deg of lag angle and delivers the clubhead to the ball with
        the club still thrown across the arm, which is not a golf swing.
        """
        L = self.anthro.lead
        joints = [f"shoulder_{L}_abd", f"shoulder_{L}_flex",
                  f"shoulder_{L}_rot", f"elbow_{L}_flex"]
        ball = self.address_clubhead + np.array([0.0, 0.0, dz])
        aim = np.array([1.0, 0.0, 0.0])          # target line
        goals = {"extension": ball + 0.85 * aim + np.array([0.0, 0.0, 0.60])}

        with self._scratch():
            for ph in script:
                if ph.name not in goals:
                    continue
                self.data.qpos[:] = self.qpos_address
                self._set_pose(ph.pose)
                self._apply_root(ph.name)
                mujoco.mj_forward(self.model, self.data)
                res = self._ik("clubhead", goals[ph.name], joints,
                               rest_pull=0.05)
                if self.verbose and res > 0.02:
                    print(f"  [warn] '{ph.name}': clubhead {res * 100:.1f} cm "
                          f"off its target")
                for j in joints:
                    ph.pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD

    def plan_trail_arm(self, script: List[Phase]) -> Dict[str, float]:
        """Fill in the trail-arm angles of every keyframe by IK.

        Each keyframe is warm-started from the previous one.  Solving them
        independently lets the IK hop between elbow-up and elbow-down branches,
        and interpolating across such a hop swings the trail arm through a wild
        arc that flings the club -- the trail arm is strong enough to throw the
        whole swing off.

        Returns the residual per keyframe (metres): how far the trail hand ends
        up from the grip.  Where the pose is out of that arm's reach the
        `trail_grip` equality has to stretch to make up the difference.
        """
        T = self.anthro.trail
        trail = [f"shoulder_{T}_abd", f"shoulder_{T}_flex", f"shoulder_{T}_rot",
                 f"elbow_{T}_flex", f"elbow_{T}_pro", f"wrist_{T}_flex",
                 f"wrist_{T}_dev"]
        residuals: Dict[str, float] = {}

        with self._scratch():
            self.data.qpos[:] = self.qpos_address
            for ph in script:
                previous = {j: float(self.data.qpos[self.tracker.qadr[j]]) * RAD
                            for j in trail}
                self.data.qpos[:] = self.qpos_address
                self._set_pose({**ph.pose, **previous})
                self._apply_root(ph.name)
                mujoco.mj_forward(self.model, self.data)
                residuals[ph.name] = self._solve_trail_arm()
                for j in trail:
                    ph.pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD
        return residuals

    def report_grip_reach(self, residuals: Dict[str, float],
                          tol: float = 0.03) -> None:
        """Say which keyframes the trail arm cannot quite reach."""
        strained = {n: r for n, r in residuals.items() if r > tol}
        if strained and self.verbose:
            worst = ", ".join(f"{n} {r * 100:.0f} cm"
                              for n, r in sorted(strained.items(),
                                                 key=lambda kv: -kv[1]))
            print(f"  trail hand strains against the grip at: {worst}")

    # ---- calibration -------------------------------------------------------
    def _downswing_window(self) -> Tuple[float, float]:
        times = {ph.name: ph.time for ph in self.controller.phases}
        return times["transition"], times["extension"]

    def _dry_swing(self):
        """Reset with the ball parked out of the way, ready to be stepped."""
        self.reset(place_ball=False)
        self._set_ball(np.array([0.0, 8.0, BALL_RADIUS]))

    def measure_arc(self) -> Tuple[float, float, float]:
        """Swing once and report where the clubhead is at its fastest on the
        way down, as (height, time, speed).

        Deliberately not the lowest point of the arc: the club is slowest there
        -- the pelvis has run out of rotation by then -- and a ball sitting at
        the very bottom gets a limp strike.  What matters for impact is the
        height of the *fast* part of the arc.
        """
        self._dry_swing()
        t0, t1 = self._downswing_window()
        best = (0.0, 0.0, 0.0)                   # speed, height, time
        for i in range(int(self.controller.duration / self.timestep)):
            t = i * self.timestep
            self.controller.apply(self.data, t)
            mujoco.mj_step(self.model, self.data)
            if not (t0 <= t <= t1):
                continue
            v = self.tracker.site_velocity("clubhead")
            speed = float(np.linalg.norm(v))
            if v[0] > 2.0 and speed > best[0]:
                z = float(self.tracker.positions()[
                    self.tracker.index["clubhead"]][2])
                best = (speed, z, t)
        return best[1], best[2], best[0]

    def tune_swing(self, script: List[Phase], iters: int = 6,
                   tol: float = 0.015, authority: float = 25.0) -> None:
        """Raise or lower the hands until the fast part of the arc is at ball
        height.

        How the arc really sits depends on the golfer's proportions, the club,
        the tempo and how far the torque-limited servos lag, so it is not
        something that can sensibly be written into the keyframes by hand.
        Measuring an actual swing and correcting converges in a few tries.
        """
        L = self.anthro.lead
        key = f"shoulder_{L}_flex"
        poses = {ph.name: ph.pose for ph in script}
        total = 0.0
        z = t = speed = 0.0

        for _ in range(iters):
            z, t, speed = self.measure_arc()
            err = z - BALL_CENTRE_HEIGHT
            if abs(err) < tol:
                break
            # ~1 deg of lead shoulder flexion raises the hands, and with them
            # the whole arc, by a bit over a centimetre
            delta = float(np.clip(-err / 0.012, -20.0, 20.0))
            # Keep the correction to something a golfer could plausibly do:
            # left unbounded it just jams the lead arm down at its stop.
            delta = float(np.clip(total + delta, -authority, authority)) - total
            if abs(delta) < 1e-6:
                break
            total += delta
            for name, weight in (("impact", 1.0), ("through", 1.0),
                                 ("extension", 0.5)):
                poses[name][key] = self._clamp_deg(
                    key, poses[name][key] + weight * delta)
            self.grip_residuals = self.plan_trail_arm(script)

        if self.verbose:
            print(f"  swing tuned (hands {total:+.0f} deg): fastest point of "
                  f"the arc is {z * 1000:.0f} mm up at t={t:.3f} s, "
                  f"clubhead {speed:.1f} m/s")

    def place_ball(self) -> None:
        """Swing once more and put the ball where the clubface actually passes
        through, fastest and closest to tee height.

        This is how a golfer picks ball position in the first place -- by where
        the club goes -- rather than assuming it returns exactly to address.
        """
        self._dry_swing()
        face_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "clubface")
        t0, t1 = self._downswing_window()
        best = None

        for i in range(int(self.controller.duration / self.timestep)):
            t = i * self.timestep
            self.controller.apply(self.data, t)
            mujoco.mj_step(self.model, self.data)
            if not (t0 <= t <= t1):
                continue
            vel = self.tracker.site_velocity("clubhead")
            if vel[0] < 2.0:            # must be travelling down the target line
                continue
            speed = float(np.linalg.norm(vel))
            pos = self.data.site_xpos[face_sid].copy()
            # prefer the fastest sample within a ball's width of tee height,
            # and fall back to the closest one if it never gets there
            err = abs(pos[2] - BALL_CENTRE_HEIGHT)
            score = (max(err - BALL_RADIUS, 0.0), -speed)
            if best is None or score < best[0]:
                normal = self.data.site_xmat[face_sid].reshape(3, 3)[:, 2].copy()
                best = (score, err, pos, normal, speed, t)

        if best is None:
            if self.verbose:
                print("  [warn] no impact point found; ball left at address")
            return

        _score, err, pos, normal, speed, t = best
        self.ball_pos = pos + BALL_RADIUS * normal
        self.ball_pos[2] = max(self.ball_pos[2], BALL_CENTRE_HEIGHT)
        if self.verbose:
            print(f"  ball placed on the arc: t={t:.3f}s clubhead "
                  f"{speed:.1f} m/s  (arc is {err * 1000:.0f} mm from tee "
                  f"height there)")

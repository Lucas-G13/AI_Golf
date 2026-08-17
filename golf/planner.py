"""Filling in the parts of the swing that cannot be scripted by hand.

`golf.swing` writes down the trunk and the lead arm.  This module turns that
into a complete, achievable trajectory:

* `plan_legs`     -- the pelvis turn, converted into leg angles that keep both
                     feet on their pins
* `plan_lead_arm` -- the downswing and follow-through, solved by clubhead
                     position against the swing plane
* `plan_trail_arm`-- the trail arm, which is never scripted because both hands
                     on one shaft is a closed loop
* `tune_swing`    -- a dry swing that checks the club still reaches the ball,
                     and moves the hands if it does not
* `place_ball`    -- one more dry swing that puts the ball where the club
                     actually goes

All of the planning runs on a scratch `MjData`, so the live simulation is never
disturbed.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

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

    def swing_plane(self, dz: float = 0.0):
        """The circle the clubhead should travel on, built from the address pose.

        Hand-written keyframes gave a plane that descends 40-47 deg all the way
        into the ball and crosses it from outside: a chop.  Rather than guess
        joint angles, define the arc and let IK find the pose.

        The circle is centred on the golfer's own swing centre (the base of the
        neck) with the radius they actually have at address, so every point on
        it is reachable by construction.  It is oriented so the tangent at the
        ball runs down the target line -- which makes the path neutral and the
        attack level exactly there, and in-to-out before it.
        """
        centre = self.tracker.positions()[self.tracker.index["thorax"]].copy()
        ball = self.address_clubhead + np.array([0.0, 0.0, dz])
        to_ball = ball - centre
        radius = float(np.linalg.norm(to_ball))
        d_hat = to_ball / radius
        aim = np.array([1.0, 0.0, 0.0])
        tangent = aim - float(aim @ d_hat) * d_hat
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        return centre, radius, d_hat, tangent

    def on_plane(self, phi_deg: float, dz: float = 0.0) -> np.ndarray:
        """Clubhead position at `phi` degrees around the arc; 0 is the ball,
        negative is before it (still coming down), positive is past it."""
        centre, radius, d_hat, tangent = self.swing_plane(dz)
        phi = math.radians(phi_deg)
        return centre + radius * (math.cos(phi) * d_hat +
                                  math.sin(phi) * tangent)

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
        # The trunk is in the solve, not just the arm.  Moving the clubhead
        # onto the plane needs it 45 cm further inside than the arm alone can
        # reach -- a golfer gets there by turning and side-bending, not by
        # pulling the arm across.
        joints = [f"shoulder_{L}_abd", f"shoulder_{L}_flex",
                  f"shoulder_{L}_rot", f"elbow_{L}_flex",
                  "thorax_rot", "thorax_bend", "lumbar_rot", "lumbar_bend"]
        # Walk the clubhead around the swing plane.  Impact sits a few degrees
        # *past* the bottom so the club is already ascending -- a teed driver
        # wants a positive attack angle, and the low point of a real swing is
        # behind the ball.
        # Impact is deliberately absent: its lead-arm angles stay at their
        # address values, which is what keeps the face square.  Solving impact
        # by position too lets the solver reach the point with the face open,
        # and the smash factor falls away.  The neighbouring phases fix the
        # plane; the address geometry fixes the face.
        goals = {name: self.on_plane(phi, dz) for name, phi in
                 (("delivery", -38.0), ("through", 38.0), ("extension", 72.0))}

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

    def _best_impact(self) -> Optional[Dict[str, Any]]:
        """Swing once and return the point on the arc this swing would strike
        the ball from, or None if the club never comes down the target line.

        One scan, one definition of where impact happens, shared by the tuner
        and by ball placement.  They used to disagree, and that was the whole
        problem: the tuner drove the *fastest* point of the arc to ball height
        while placement picked a point trading height against speed and attack
        angle, so the tuner spent its authority moving a point that was never
        going to be struck.  It also cannot be answered without swinging --
        "where does the arc cross ball height" always has an answer, because
        the clubhead sweeps from the top of the backswing to the ground and
        crosses everything on the way.
        """
        self._dry_swing()
        face_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "clubface")
        t0, t1 = self._downswing_window()
        best: Optional[Dict[str, Any]] = None

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
            attack = math.degrees(math.atan2(
                vel[2], max(float(np.linalg.norm(vel[:2])), 1e-9)))
            # Prefer the fastest sample within a ball's width of tee height --
            # but only from the part of the arc where the club has stopped
            # descending.  A teed driver is struck on the way up; taking the
            # fastest point regardless put the ball 45 cm behind the low point
            # and cost 16 deg of attack angle.
            err = abs(float(pos[2]) - BALL_CENTRE_HEIGHT)
            # All three traded against each other in one number, rather than
            # gating on height first.  A hard "within a ball radius" gate is
            # fine until the arc misses that band where the club is fast and
            # clips it somewhere slow in the follow-through -- then the gate
            # takes the slow one on principle and the ball is teed 90 cm down
            # the target line.  Continuous, the 25 mm miss simply costs what it
            # is worth and the fast point still wins.
            #
            # 500/m prices a centimetre of height error at ~5 m/s of clubhead
            # speed; 0.25 prices 10 deg of attack error at about 2.5 m/s.  The
            # attack term stops a hard preference for "not descending" walking
            # the ball round to the far side of the arc, where the club is
            # climbing at 42 deg and has slowed to 10 m/s.
            score = (500.0 * max(err - BALL_RADIUS, 0.0)
                     - speed + 0.25 * abs(attack - 3.0))
            if best is None or score < best["score"]:
                normal = self.data.site_xmat[face_sid].reshape(3, 3)[:, 2].copy()
                best = {"score": score, "err": err, "height": float(pos[2]),
                        "pos": pos, "normal": normal, "speed": speed,
                        "attack": attack, "time": t}
        return best

    def tune_swing(self, script: List[Phase], iters: int = 6,
                   tol: float = BALL_RADIUS, authority: float = 25.0) -> None:
        """Raise or lower the hands until the club can reach the ball at all.

        A backstop, not the main event -- and it used to be the main event,
        which is what went wrong.  `plan_lead_arm` now builds the arc through
        the ball geometrically, so the height is right by construction and
        there is normally nothing here to fix: the loop measures once and
        exits.  What it still catches is the geometry failing to survive the
        dynamics, where the servos lag so far behind the plan that the club
        never gets down to the ball.

        The correction is deliberately weak, and `tol` is a ball radius rather
        than something tighter, because shoving `shoulder_flex` drags the
        clubhead off the plane that was just solved for.  Run to saturation it
        costs precisely what the plane bought: attack angle +2 deg -> -15, club
        path 7 deg -> 12, and the clubhead through the floor past impact.
        Worth doing only when the alternative is missing the ball entirely.
        """
        L = self.anthro.lead
        key = f"shoulder_{L}_flex"
        poses = {ph.name: ph.pose for ph in script}
        total = 0.0
        hit = None

        for _ in range(iters):
            hit = self._best_impact()
            if hit is None:
                break
            # Same test `place_ball` applies: inside a ball radius the ball can
            # sit on the arc and be struck cleanly, so leave the plan alone.
            err = hit["height"] - BALL_CENTRE_HEIGHT
            if abs(err) <= tol:
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
            if hit is None:
                print("  [warn] the club never comes down the target line")
            else:
                how = "as planned" if total == 0.0 else f"hands {total:+.0f} deg"
                print(f"  arc strikes {hit['height'] * 1000:.0f} mm up at "
                      f"t={hit['time']:.3f} s, clubhead {hit['speed']:.1f} m/s, "
                      f"attack {hit['attack']:+.1f} deg ({how})")

    def place_ball(self) -> None:
        """Swing once more and put the ball where the clubface actually passes
        through, fastest and closest to tee height.

        This is how a golfer picks ball position in the first place -- by where
        the club goes -- rather than assuming it returns exactly to address.
        """
        hit = self._best_impact()
        if hit is None:
            if self.verbose:
                print("  [warn] no impact point found; ball left at address")
            return

        self.ball_pos = hit["pos"] + BALL_RADIUS * hit["normal"]
        # The tee holds the ball at one height, so that is the height it goes
        # to -- not the height the arc happens to pass at.  This used to be a
        # `max`, which let the ball be teed up to 2 cm above where the tee can
        # actually hold it; it then fell onto the tee over the first tenth of a
        # second.  Nothing measured changed, because it had always settled long
        # before impact, but the start of every swing had the ball dropping.
        self.ball_pos[2] = BALL_CENTRE_HEIGHT
        if self.verbose:
            print(f"  ball placed on the arc: t={hit['time']:.3f}s clubhead "
                  f"{hit['speed']:.1f} m/s  (arc is {hit['err'] * 1000:.0f} mm "
                  f"from tee height there)")

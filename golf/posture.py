"""Getting the golfer to address.

The address position cannot simply be written down as joint angles: whether the
club soles at the ball and whether the trail hand can even reach the grip both
depend on the golfer's proportions and the club's length.  So the nominal
angles are a starting point, and this module solves the rest against the model.
"""

from __future__ import annotations

import math
from typing import Dict

import mujoco
import numpy as np

from .equipment import BALL_CENTRE_HEIGHT, BALL_FORWARD
from .joints import ADDRESS
from .util import DEG


class AddressSolver:
    """Mixin: everything that happens before the swing starts."""

    def _solve_address(self, warn: bool = True) -> None:
        """Put the golfer in a self-consistent address position.

        1. apply the nominal address angles
        2. drop the model until the feet touch the ground
        3. square the shoulders to the target line
        4. adjust the lead wrist/forearm until the club soles at the ball
        5. IK the trail arm onto the grip so both hands are on the club
        """
        mujoco.mj_resetData(self.model, self.data)
        self._set_pose(ADDRESS)
        mujoco.mj_forward(self.model, self.data)

        if self.base != "pinned":
            lowest = min(
                self.data.geom_xpos[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_g")][2]
                - self.anthro.ankle_h / 2 for s in ("l", "r"))
            self.data.qpos[self.root_qadr + 2] += -lowest + 0.001
            mujoco.mj_forward(self.model, self.data)

        self._square_shoulders()

        L = self.anthro.lead
        res = 1.0
        # Sole the club, then check the trail hand can actually get on the
        # grip.  If it cannot, pull the hands in toward the middle of the
        # stance -- which is what a golfer does -- and try again.
        for adduct in range(0, 26, 2):
            self.data.qpos[self.tracker.qadr[f"shoulder_{L}_abd"]] = \
                (ADDRESS[f"shoulder_{L}_abd"] - adduct) * DEG
            head = self.tracker.positions()[self.tracker.index["clubhead"]]
            # Sole the club at ball height, a little forward of centre in the
            # stance.  How far the ball ends up from the golfer is left free:
            # it is whatever the arms and the club reach.
            target = np.array([BALL_FORWARD, head[1], BALL_CENTRE_HEIGHT])
            self._ik("clubhead", target,
                     [f"wrist_{L}_dev", f"wrist_{L}_flex", f"elbow_{L}_pro"],
                     mask=(1, 0, 1))
            res = self._solve_trail_arm()
            if res < 5e-3:
                break
        self.address_grip_residual = res
        if warn and self.verbose and res > 5e-3:
            print(f"  [warn] trail hand is {res * 100:.1f} cm off the grip")

    def _square_shoulders(self) -> None:
        """Side-bending a torso that is already tilted forward also yaws it,
        which leaves the shoulders closed by ~10 deg.  Take that back out with
        thorax rotation so the golfer actually aims at the target."""
        for _ in range(12):
            err = self.tracker.metrics()["shoulder_turn"]
            if abs(err) < 0.2:
                break
            adr = self.tracker.qadr["thorax_rot"]
            self.data.qpos[adr] = self._clamp("thorax_rot",
                                              self.data.qpos[adr] - err * DEG)
            mujoco.mj_forward(self.model, self.data)

    def _solve_trail_arm(self) -> float:
        """Put the trail hand on the club.  Both hands on one shaft is a closed
        kinematic loop, so the trail arm is never scripted -- it is solved here
        and held together during the swing by the `trail_grip` equality."""
        T = self.anthro.trail
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                "grip_trail")
        joints = [f"shoulder_{T}_abd", f"shoulder_{T}_flex",
                  f"shoulder_{T}_rot", f"elbow_{T}_flex", f"elbow_{T}_pro",
                  f"wrist_{T}_flex", f"wrist_{T}_dev"]
        return self._ik(f"palm_{T}", self.data.site_xpos[sid].copy(), joints)

    def _foot_pin_targets(self) -> Dict[str, np.ndarray]:
        """World positions of the sites pinned to the ground.

        Heel and toe of each foot, so the feet are genuinely planted.  Pinning
        only the ankle leaves the foot free to spin about the vertical, and the
        leg's rotation then goes into the foot instead of turning the pelvis;
        the freedom the turning pelvis needs comes from `ankle_*_rot` instead.
        """
        names = [f"{p}_{s}" for s in ("l", "r") for p in ("heel", "toe")]
        out = {}
        for name in names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            out[name] = self.data.site_xpos[sid].copy()
        return out

    def head_alignment(self) -> np.ndarray:
        """Orientation the clubhead should have within the club body.

        The head must not simply inherit the shaft's frame.  The shaft comes
        into a real head at the lie angle, so a head welded square to the shaft
        ends up tipped ~40 deg with the sole in the air and the mass hanging off
        the wrong side.  What we actually want is a head that is *level and
        square to the target at address*, which is world-axis-aligned -- so the
        rotation it needs relative to the club is just the inverse of the
        club's own orientation at address.

        Depends on the address pose, which in turn depends on where the head
        is, so `GolfSwingSim` iterates this a couple of times.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "club")
        R_club = self.data.xmat[bid].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(R_club.T).ravel())
        return q

    def face_heading(self) -> float:
        """How far the clubface points away from square at address, degrees.
        Positive is open (aiming right of the target for a right-hander)."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "clubface")
        n = self.data.site_xmat[sid].reshape(3, 3)[:, 2]   # site z = face normal
        return math.degrees(math.atan2(n[1], n[0])) * self.anthro.lead_sign

    def face_loft(self) -> float:
        """Effective loft of the face at address, degrees above horizontal."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "clubface")
        n = self.data.site_xmat[sid].reshape(3, 3)[:, 2]
        return math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))

    def _settle(self, seconds: float) -> None:
        """Hold the solved address pose (which includes the IK'd trail arm) and
        let gravity and the grip constraint find the real equilibrium before
        anything is measured off the pose."""
        for n, v in self.address_angles.items():
            self.data.ctrl[self.actuator_id(n)] = v * DEG
        for _ in range(int(seconds / self.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _take_out_droop(self, iters: int = 4, tol: float = 0.008) -> None:
        """The servos sag a few degrees under load, which drops the clubhead
        below the ball.  Ask for a little more wrist until the club actually
        soles where it should once everything has settled."""
        key = f"wrist_{self.anthro.lead}_dev"
        for _ in range(iters):
            z = float(self.tracker.positions()[
                self.tracker.index["clubhead"]][2])
            err = z - BALL_CENTRE_HEIGHT
            if abs(err) < tol:
                break
            self.address_angles[key] = self._clamp_deg(
                key, self.address_angles[key] - err / 0.018)
            self._settle(0.25)

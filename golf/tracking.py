"""Reading the body: where every joint is, and what the swing is doing.

`SwingTracker` is the measurement layer, and the piece an RL policy will
observe through.  It owns no state of its own -- it just reads `MjData` -- so
it is cheap to call every step.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import mujoco
import numpy as np

from .anthropometry import Anthropometry
from .joints import JOINT_NAMES
from .landmarks import TRACKED, parent_map
from .util import RAD


class SwingTracker:
    """Every major joint centre, in whichever frame you need it.

    All of the position methods return an (N, 3) array whose rows follow
    `TRACKED`, so `tracker.index["wrist_l"]` indexes any of them.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 anthro: Anthropometry):
        self.model, self.data, self.anthro = model, data, anthro
        self.names: Tuple[str, ...] = TRACKED
        self.index = {n: i for i, n in enumerate(TRACKED)}
        self.parent = parent_map(anthro.lead)
        self.sid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                    for n in TRACKED}
        missing = [n for n, i in self.sid.items() if i < 0]
        if missing:
            raise RuntimeError(f"model is missing sites: {missing}")
        self.jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in JOINT_NAMES}
        self.qadr = {n: model.jnt_qposadr[i] for n, i in self.jid.items()}
        self.dadr = {n: model.jnt_dofadr[i] for n, i in self.jid.items()}
        self.pelvis_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lead = anthro.lead
        self.trail = anthro.trail
        self.target = np.array([1.0, 0.0, 0.0])   # target line, world frame

    # ---- positions --------------------------------------------------------
    def positions(self) -> np.ndarray:
        """World-frame position of every tracked landmark, shape (N, 3)."""
        return np.array([self.data.site_xpos[self.sid[n]] for n in TRACKED])

    def pelvis_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """(origin, rotation) of the golfer's pelvis in the world."""
        o = self.data.site_xpos[self.sid["pelvis"]].copy()
        R = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
        return o, R

    def heading_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Origin at the pelvis, x = the direction the golfer faces flattened
        into the horizontal plane, z = world up.  Yaw-only, so heights stay
        heights however far the golfer is bent over."""
        o, R = self.pelvis_frame()
        fwd = R[:, 0].copy()
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        if n < 1e-6:                       # bent double: fall back to the spine
            fwd = -R[:, 2].copy()
            fwd[2] = 0.0
            n = max(np.linalg.norm(fwd), 1e-9)
        fwd /= n
        left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
        return o, np.column_stack([fwd, left, np.array([0.0, 0.0, 1.0])])

    # ---- relative ---------------------------------------------------------
    def egocentric(self, gravity_aligned: bool = True) -> np.ndarray:
        """Landmarks relative to the golfer: origin at the pelvis, axes along
        forward / left / up.  Invariant to where on the range the golfer stands
        and which way they aim, which is what an RL policy wants to see."""
        o, R = (self.heading_frame() if gravity_aligned else self.pelvis_frame())
        return (self.positions() - o) @ R

    def parent_relative(self) -> np.ndarray:
        """Each landmark minus its parent in the kinematic chain -- i.e. the
        segment vectors (world frame).  Chain roots give a zero row."""
        p = self.positions()
        out = np.zeros_like(p)
        for i, n in enumerate(TRACKED):
            par = self.parent[n]
            if par is not None:
                out[i] = p[i] - p[self.index[par]]
        return out

    def pairwise_offsets(self) -> np.ndarray:
        """(N, N, 3): row i, col j = position(j) - position(i)."""
        p = self.positions()
        return p[None, :, :] - p[:, None, :]

    def pairwise_distances(self) -> np.ndarray:
        """(N, N) euclidean distances between every pair of landmarks."""
        return np.linalg.norm(self.pairwise_offsets(), axis=-1)

    # ---- velocities -------------------------------------------------------
    def site_velocity(self, name: str) -> np.ndarray:
        """Linear velocity of a landmark, world frame."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.sid[name], res, 0)
        return res[3:6].copy()

    def clubhead_speed(self) -> float:
        return float(np.linalg.norm(self.site_velocity("clubhead")))

    def ball_speed(self) -> float:
        return float(np.linalg.norm(self.site_velocity("ball")))

    # ---- joint angles -----------------------------------------------------
    def joint_angles(self, degrees: bool = True) -> Dict[str, float]:
        f = RAD if degrees else 1.0
        return {n: float(self.data.qpos[self.qadr[n]]) * f for n in JOINT_NAMES}

    def joint_velocities(self, degrees: bool = True) -> Dict[str, float]:
        f = RAD if degrees else 1.0
        return {n: float(self.data.qvel[self.dadr[n]]) * f for n in JOINT_NAMES}

    # ---- golf-specific ----------------------------------------------------
    @staticmethod
    def _heading(v: np.ndarray) -> float:
        """Angle of a vector's horizontal projection, degrees, +ve toward +y."""
        return math.degrees(math.atan2(v[1], v[0]))

    def _turn(self, trail_site: str, lead_site: str) -> float:
        """Rotation of a body segment about the vertical.  0 = square to the
        target line, negative = turned away from the target (backswing)."""
        p = self.positions()
        v = p[self.index[lead_site]] - p[self.index[trail_site]]
        lead_is_left = self.anthro.lead_sign > 0
        # the line from trail to lead landmark points down the target line at 0
        h = self._heading(v if lead_is_left else -v)
        return ((h + 180.0) % 360.0) - 180.0

    def metrics(self) -> Dict[str, float]:
        """The numbers a coach or a launch monitor would quote."""
        p = self.positions()
        idx = self.index
        lead, trail = self.lead, self.trail

        hip_turn = self._turn(f"hip_{trail}", f"hip_{lead}")
        sh_turn = self._turn(f"shoulder_{trail}", f"shoulder_{lead}")

        spine = p[idx["neck"]] - p[idx["pelvis"]]
        spine_tilt = math.degrees(math.acos(
            np.clip(spine[2] / max(np.linalg.norm(spine), 1e-9), -1, 1)))
        # forward/side split of that tilt, in the golfer's own frame
        _, R = self.heading_frame()
        spine_local = spine @ R
        forward_tilt = math.degrees(math.atan2(spine_local[0], spine_local[2]))
        side_tilt = math.degrees(math.atan2(spine_local[1], spine_local[2]))

        arm = p[idx[f"wrist_{lead}"]] - p[idx[f"shoulder_{lead}"]]
        shaft = p[idx["clubhead"]] - p[idx["grip"]]
        cock = math.degrees(math.acos(np.clip(
            np.dot(arm, shaft) / max(np.linalg.norm(arm) * np.linalg.norm(shaft),
                                     1e-9), -1, 1)))

        chv = self.site_velocity("clubhead")
        speed = float(np.linalg.norm(chv))
        attack = math.degrees(math.atan2(chv[2], max(np.linalg.norm(chv[:2]),
                                                     1e-9)))
        path = math.degrees(math.atan2(chv[1], chv[0])) if speed > 0.5 else 0.0

        return {
            "t": float(self.data.time),
            "hip_turn": hip_turn,
            "shoulder_turn": sh_turn,
            "x_factor": sh_turn - hip_turn,
            "spine_tilt": spine_tilt,
            "forward_tilt": forward_tilt,
            "side_tilt": side_tilt,
            "lead_wrist_cock": cock,
            "clubhead_speed": speed,
            "clubhead_height": float(p[idx["clubhead"]][2]),
            "hand_speed": float(np.linalg.norm(
                self.site_velocity(f"hand_{lead}"))),
            "attack_angle": attack,
            "club_path": path,
            "ball_speed": self.ball_speed(),
        }

    # ---- RL observation ---------------------------------------------------
    def observation(self) -> np.ndarray:
        """Flat observation vector: joint angles and velocities, every landmark
        in the egocentric frame, the segment vectors, and club/ball velocity."""
        q = np.array([self.data.qpos[self.qadr[n]] for n in JOINT_NAMES])
        qd = np.array([self.data.qvel[self.dadr[n]] for n in JOINT_NAMES])
        ego = self.egocentric().ravel()
        seg = self.parent_relative().ravel()
        _, R = self.heading_frame()
        club_v = self.site_velocity("clubhead") @ R
        ball_v = self.site_velocity("ball") @ R
        return np.concatenate([q, qd, ego, seg, club_v, ball_v])

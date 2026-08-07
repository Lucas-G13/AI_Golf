"""Inverse kinematics against the live MuJoCo model.

Damped least squares with a null-space bias, mixed into `GolfSwingSim`.  It
operates on `self.data` directly, which is what lets the callers solve on a
scratch `MjData` without the solver knowing.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import mujoco
import numpy as np

from .util import RAD


class IKSolver:
    """Mixin: `_ik` / `_ik_multi` and the joint-limit helpers they need."""

    def _clamp(self, name: str, v: float) -> float:
        """Clamp a joint value (radians) to its range."""
        lo, hi = self.model.jnt_range[self.tracker.jid[name]]
        return min(max(v, lo), hi)

    def _clamp_deg(self, joint: str, deg: float) -> float:
        lo, hi = self.model.jnt_range[self.tracker.jid[joint]] * RAD
        return float(min(max(deg, lo), hi))

    def _ik(self, site: str, target: np.ndarray, joints: Sequence[str],
            mask: Sequence[float] = (1, 1, 1), iters: int = 200,
            step: float = 0.5, damping: float = 1e-3,
            rest_pull: float = 0.25, max_step: float = 0.12) -> float:
        """Drive one site to one target.  Returns the residual in metres."""
        return self._ik_multi([(site, target)], joints, mask, iters, step,
                              damping, rest_pull, max_step)

    def _ik_multi(self, goals: Sequence[Tuple[str, np.ndarray]],
                  joints: Sequence[str], mask: Sequence[float] = (1, 1, 1),
                  iters: int = 200, step: float = 0.5, damping: float = 1e-3,
                  rest_pull: float = 0.25, max_step: float = 0.12,
                  free_root: bool = False) -> float:
        """Drive several sites to several targets at once.

        The null-space term pulls the unused freedom back toward the pose the
        solver started from.  Without it a 7-DOF arm happily reaches the grip
        with the shoulder abducted 170 deg -- geometrically valid, anatomically
        absurd, and it flings the club as soon as the next keyframe unwinds it.

        `free_root` adds the root body's three translation DOFs to the solve,
        which is what lets the leg planner hold both feet on their pins while
        the pelvis is rotated to a prescribed angle.
        """
        sids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                for s, _ in goals]
        dofs = [self.tracker.dadr[j] for j in joints]
        adrs = [self.tracker.qadr[j] for j in joints]
        if free_root:
            r, d = self.root_qadr, self.root_dadr
            dofs = [d, d + 1, d + 2] + dofs
            adrs = [r, r + 1, r + 2] + adrs

        w = np.tile(np.asarray(mask, dtype=float), len(goals))
        rest = self.data.qpos[adrs].copy()
        jacp = np.zeros((3, self.model.nv))
        eye = np.eye(len(adrs))
        m = len(w)
        err = np.zeros(m)
        # a None name means "no joint limit applies" -- the free root's DOFs
        names = ([None] * 3 + list(joints)) if free_root else list(joints)

        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            err = np.concatenate([t - self.data.site_xpos[s]
                                  for s, (_n, t) in zip(sids, goals)]) * w
            if np.linalg.norm(err) < 1e-5:
                break
            rows = []
            for s in sids:
                mujoco.mj_jacSite(self.model, self.data, jacp, None, s)
                rows.append(jacp[:, dofs])
            J = np.vstack(rows) * w[:, None]
            Jinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(m))
            q = self.data.qpos[adrs]
            dq = Jinv @ err + (eye - Jinv @ J) @ (rest_pull * (rest - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            for name, adr, d in zip(names, adrs, dq):
                v = self.data.qpos[adr] + step * d
                self.data.qpos[adr] = v if name is None else self._clamp(name, v)

        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(err))

"""The scripted reference swing: keyframes, and how they are interpolated.

This is a hand-written reference trajectory, not a good golf swing -- see the
package docstring.  It exists so the model can be exercised and measured; the
RL agent replaces it.

Only the trunk and the lead arm are written down here.  The legs, the trail arm
and the exact impact geometry are *solved* against the model in `golf.planner`,
because they depend on the golfer's proportions and on how far the torque
limited servos lag.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

import mujoco

from .anthropometry import Anthropometry
from .joints import JOINT_NAMES
from .util import DEG, hermite_basis


class Phase(NamedTuple):
    """One keyframe of the swing."""
    name: str
    time: float                   # seconds from the start of the swing
    pose: Dict[str, float]        # joint -> target angle, degrees
    pelvis_turn: float            # whole-body rotation about the spine, deg,
                                  # positive toward the target


def keyframe_pose(base: Dict[str, float], lumbar_turn: float = 0.0,
                  thorax_turn: float = 0.0,
                  **overrides: float) -> Dict[str, float]:
    """Build a keyframe as deltas from the settled address pose.

    Turns are joint-relative, so the shoulder turn seen from outside is
    pelvis + lumbar + thorax.  The pelvis turn is not set here: it is a
    whole-body rotation and becomes leg angles in `SwingPlanner.plan_legs`.
    """
    p = dict(base)
    p["lumbar_rot"] = base["lumbar_rot"] + lumbar_turn
    p["thorax_rot"] = base["thorax_rot"] + thorax_turn
    p.update(overrides)
    return p


def swing_script(anthro: Anthropometry, base: Dict[str, float],
                 tempo: float = 1.0) -> List[Phase]:
    """The eight keyframes of the swing, timed like a real one: ~0.8 s to the
    top and ~0.3 s down, with the pelvis already opening while the arms are
    still finishing the backswing -- the kinematic sequence that produces the
    X-factor stretch."""
    L = anthro.lead

    def arm(flex, abd, rot, elbow, dev, wflex=None):
        d = {f"shoulder_{L}_flex": flex, f"shoulder_{L}_abd": abd,
             f"shoulder_{L}_rot": rot, f"elbow_{L}_flex": elbow,
             f"wrist_{L}_dev": dev}
        if wflex is not None:
            d[f"wrist_{L}_flex"] = wflex
        return d

    s_flex = base[f"shoulder_{L}_flex"]
    s_abd = base[f"shoulder_{L}_abd"]
    e_flex = base[f"elbow_{L}_flex"]
    w_dev = base[f"wrist_{L}_dev"]

    def P(**kw):
        return keyframe_pose(base, **kw)

    phases = [
        Phase("address", 0.00,
              P(**arm(s_flex, s_abd, 0, e_flex, w_dev)), 0),
        Phase("takeaway", 0.32,
              P(lumbar_turn=-4, thorax_turn=-22, thorax_bend=-6,
                **arm(s_flex + 4, s_abd - 12, 10, e_flex + 4, w_dev + 12)), -10),
        Phase("top", 0.82,
              P(lumbar_turn=-12, thorax_turn=-46,
                thorax_bend=-2, lumbar_bend=0, neck_rot=-25,
                **arm(s_flex + 34, s_abd - 34, 40, e_flex + 12, 52)), -42),
        # the pelvis has already started back toward the target while the
        # shoulders are still at the top: the kinematic sequence
        Phase("transition", 0.90,
              P(lumbar_turn=-13, thorax_turn=-48,
                thorax_bend=-10, neck_rot=-22,
                **arm(s_flex + 30, s_abd - 32, 38, e_flex + 12, 52)), -30),
        # The shoulder turn is the sum of all three turns, so at impact the
        # hips are 42 deg open but the chest only ~17 -- the negative X-factor
        # a real golfer has at impact.  The lead arm comes back to its address
        # angles, so the club is extended out at the ball again and the body
        # rotation is what delivers it.
        Phase("impact", 1.14,
              P(lumbar_turn=-5, thorax_turn=-20,
                thorax_bend=-13, lumbar_bend=-9, neck_rot=-10,
                **arm(s_flex, s_abd, 0, e_flex, w_dev - 4)), 42),
        # keeps driving through the bottom of the arc instead of lifting away
        Phase("through", 1.28,
              P(lumbar_turn=0, thorax_turn=-4,
                thorax_bend=-12, lumbar_bend=-8, neck_rot=-4,
                **arm(s_flex + 2, s_abd + 6, -20, e_flex, w_dev - 10)), 60),
        Phase("extension", 1.46,
              P(lumbar_turn=6, thorax_turn=6,
                thorax_bend=-10, neck_rot=8,
                **arm(s_flex + 6, s_abd + 10, -45, e_flex + 6, w_dev - 22)), 74),
        Phase("finish", 1.95,
              P(lumbar_turn=12, thorax_turn=35,
                thorax_bend=8, thorax_flex=-10, neck_rot=25,
                **arm(s_flex + 22, s_abd + 16, -25, 35, w_dev - 10)), 76),
    ]
    return [Phase(p.name, p.time * tempo, p.pose,
                  p.pelvis_turn * anthro.lead_sign) for p in phases]


class SwingController:
    """Interpolates the keyframes and writes position-servo targets."""

    def __init__(self, phases: List[Phase], model: mujoco.MjModel,
                 lead: float = 0.0):
        self.phases = phases
        self.act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         f"act_{n}") for n in JOINT_NAMES}
        self.duration = phases[-1].time
        # Torque-limited servos run behind their reference, which at swing
        # speed puts the club ~0.4 m above the ball when the script says
        # impact.  Feeding them the reference this far ahead of the clock
        # cancels most of that phase lag.
        self.lead = lead

    def phase_at(self, t: float) -> str:
        name = self.phases[0].name
        for ph in self.phases:
            if t >= ph.time:
                name = ph.name
        return name

    def targets(self, t: float) -> Dict[str, float]:
        """Catmull-Rom through the keyframes.

        Not a smoothstep between neighbours: that has zero slope at every
        keyframe, so the reference comes to a dead stop at impact and the
        clubhead arrives at the ball doing 3 m/s.  Catmull-Rom tangents let the
        swing pass *through* its keyframes at speed.
        """
        ph = self.phases
        n = len(ph)
        if t <= ph[0].time:
            return dict(ph[0].pose)
        if t >= ph[-1].time:
            return dict(ph[-1].pose)

        i = next(j for j in range(n - 1) if ph[j].time <= t <= ph[j + 1].time)
        t0, t1 = ph[i].time, ph[i + 1].time
        h = max(t1 - t0, 1e-9)
        b0, bm0, b1, bm1 = hermite_basis((t - t0) / h)
        p0, p1 = ph[i].pose, ph[i + 1].pose
        prev = ph[i - 1] if i > 0 else None
        nxt = ph[i + 2] if i + 2 < n else None

        out = {}
        for k in p0:
            m0 = 0.0 if prev is None else \
                (p1[k] - prev.pose[k]) / (t1 - prev.time) * h
            m1 = 0.0 if nxt is None else \
                (nxt.pose[k] - p0[k]) / (nxt.time - t0) * h
            out[k] = b0 * p0[k] + bm0 * m0 + b1 * p1[k] + bm1 * m1
        return out

    def apply(self, data: mujoco.MjData, t: float) -> None:
        for name, val in self.targets(t + self.lead).items():
            data.ctrl[self.act[name]] = val * DEG

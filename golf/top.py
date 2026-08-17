"""The top of the backswing: one state, shared by both halves of the swing.

Training the backswing and the downswing separately only means anything if
they agree on where one ends and the other begins.  That agreement is this
module: a single `TopState` that the downswing starts from and the backswing is
paid for reaching.  Change it and both stages have to be retrained.

The top is not hand-written.  `search_top` optimises the top keyframe against
the thing the downswing is actually paid for -- carry and straightness through
the launch model -- so the pose is chosen for the swing it has to produce
rather than for looking like a golf swing.  Ten parameters: how far the pelvis,
lumbar and thorax have turned, how the trunk is bent, and where the lead arm
is.  Everything else at the top (the legs, the trail arm) is solved by the
planner, as it always was.

What is deliberately *not* searched is the timing.  The top stays at t=0.82 s
because moving it rescales the whole script, and tempo is a separate question
from posture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .joints import JOINT_NAMES
from .swing import Phase, SwingController, keyframe_pose, swing_script

#: The searchable top parameters, in the order they are packed into a vector.
#: `shoulder_flex`, `shoulder_abd` and `elbow_flex` are deltas from the address
#: pose (that is how `swing.swing_script` writes them); `shoulder_rot` and
#: `wrist_dev` are absolute joint angles.  All degrees.
TOP_PARAMS: Tuple[str, ...] = (
    "pelvis_turn", "lumbar_turn", "thorax_turn", "thorax_bend", "lumbar_bend",
    "shoulder_flex", "shoulder_abd", "shoulder_rot", "elbow_flex", "wrist_dev")

#: The hand-written top from `swing.swing_script`, and the search's starting
#: point.  Keeping it as the mean of the first generation matters: a top drawn
#: from nothing rarely gets the club anywhere near the ball, and a search whose
#: whole first generation misses has no gradient to follow either.
DEFAULT_TOP: Dict[str, float] = dict(
    pelvis_turn=-42.0, lumbar_turn=-12.0, thorax_turn=-46.0, thorax_bend=-2.0,
    lumbar_bend=0.0, shoulder_flex=34.0, shoulder_abd=-34.0, shoulder_rot=40.0,
    elbow_flex=12.0, wrist_dev=52.0)

#: Bounds, in the same units.  These are the anatomy talking, not taste: the
#: lumbar spine gives +/-15 deg of axial rotation and the thorax +/-50, so a
#: search allowed past them just proposes tops the servos cannot hold.
TOP_BOUNDS: Dict[str, Tuple[float, float]] = dict(
    pelvis_turn=(-65.0, -20.0), lumbar_turn=(-15.0, 0.0),
    thorax_turn=(-50.0, -20.0), thorax_bend=(-30.0, 10.0),
    lumbar_bend=(-20.0, 20.0), shoulder_flex=(0.0, 60.0),
    shoulder_abd=(-54.0, -5.0), shoulder_rot=(0.0, 80.0),
    elbow_flex=(0.0, 40.0), wrist_dev=(20.0, 55.0))

#: Neck rotation at the top is held at the scripted value.  It moves the head,
#: which moves nothing that matters to the strike, and searching it only adds a
#: dimension for the sampler to waste evaluations on.
TOP_NECK_ROT = -25.0


@dataclass
class TopState:
    """Where the backswing hands over to the downswing.

    `qpos`/`qvel` are the full simulator state, which is what the downswing
    resets to.  `angles` and `clubhead` are the same state expressed as things
    the backswing can be scored against: joint angles in degrees, and where the
    clubhead ended up.  The clubhead is carried separately because it is a
    1.1 m lever -- a couple of degrees at the shoulder that barely registers in
    the joint-angle error is 10 cm of club.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    angles: np.ndarray                      # degrees, in JOINT_NAMES order
    clubhead: np.ndarray                    # world position, m
    params: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    info: Dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, qpos=self.qpos, qvel=self.qvel, angles=self.angles,
                 clubhead=self.clubhead,
                 params=np.array([self.params[k] for k in TOP_PARAMS]),
                 meta=np.array(json.dumps({"score": self.score,
                                           "info": self.info})))

    @classmethod
    def load(cls, path: Path) -> "TopState":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(qpos=z["qpos"], qvel=z["qvel"], angles=z["angles"],
                   clubhead=z["clubhead"],
                   params=dict(zip(TOP_PARAMS, z["params"].tolist())),
                   score=meta["score"], info=meta["info"])

    def describe(self) -> str:
        p = self.params
        return (f"    pelvis {p['pelvis_turn']:+.0f}  lumbar {p['lumbar_turn']:+.0f}"
                f"  thorax {p['thorax_turn']:+.0f}  (shoulder turn "
                f"{p['pelvis_turn'] + p['lumbar_turn'] + p['thorax_turn']:+.0f})\n"
                f"    trunk bend: thorax {p['thorax_bend']:+.0f}  lumbar "
                f"{p['lumbar_bend']:+.0f}\n"
                f"    lead arm: flex {p['shoulder_flex']:+.0f}  abd "
                f"{p['shoulder_abd']:+.0f}  rot {p['shoulder_rot']:+.0f}  elbow "
                f"{p['elbow_flex']:+.0f}  wrist {p['wrist_dev']:+.0f}")


# ---------------------------------------------------------------------------
# Building a top into the script
# ---------------------------------------------------------------------------

def top_pose(sim: Any, params: Dict[str, float]) -> Dict[str, float]:
    """The top keyframe's joint angles for these parameters, clamped to the
    golfer's actual joint limits."""
    base = sim.address_angles
    L = sim.anthro.lead
    pose = keyframe_pose(
        base,
        lumbar_turn=params["lumbar_turn"], thorax_turn=params["thorax_turn"],
        thorax_bend=params["thorax_bend"], lumbar_bend=params["lumbar_bend"],
        neck_rot=TOP_NECK_ROT,
        **{f"shoulder_{L}_flex": base[f"shoulder_{L}_flex"] + params["shoulder_flex"],
           f"shoulder_{L}_abd": base[f"shoulder_{L}_abd"] + params["shoulder_abd"],
           f"shoulder_{L}_rot": params["shoulder_rot"],
           f"elbow_{L}_flex": base[f"elbow_{L}_flex"] + params["elbow_flex"],
           f"wrist_{L}_dev": params["wrist_dev"]})
    return {k: sim._clamp_deg(k, v) for k, v in pose.items()}


def apply_top(sim: Any, params: Dict[str, float]) -> None:
    """Re-plan the whole swing around a different top.

    The script is rebuilt from `swing_script` rather than edited in place, and
    that is the whole point.  Editing in place means each call starts from
    whatever the *last* call solved -- the planners warm-start from the pose
    they are handed -- so a search evaluating hundreds of candidates would
    score each one through the accumulated drift of every candidate before it,
    and the same top would score differently depending on when it was tried.
    Rebuilding makes a candidate's score depend on the candidate alone.

    Everything downstream of the top is re-solved: the legs and the trail arm
    because both are solved *at* the top, the lead arm because
    `plan_trail_arm` warm-starts off it, and the ball because a different top
    puts the club through a different arc -- which is what a golfer adjusting
    their ball position is doing.
    """
    # Back to address first.  `plan_lead_arm` builds its swing plane from the
    # live state, and the last thing to touch it was a dry swing that left the
    # golfer standing in the finish -- plan from there and the plane is centred
    # on a thorax that has already turned through 125 deg.
    sim.reset()

    script: List[Phase] = swing_script(sim.anthro, sim.address_angles, sim.tempo)
    i = next(k for k, ph in enumerate(script) if ph.name == "top")
    script[i] = script[i]._replace(
        pose=top_pose(sim, params),
        pelvis_turn=params["pelvis_turn"] * sim.anthro.lead_sign)

    sim.plan_legs(script)
    sim.plan_lead_arm(script)
    sim.grip_residuals = sim.plan_trail_arm(script)
    sim.controller = SwingController(script, sim.model,
                                     lead=sim.lead * sim.tempo)
    sim.tune_swing(script)
    sim.place_ball()
    sim.reset()


def capture_top(sim: Any, params: Dict[str, float], score: float = 0.0,
                info: Optional[Dict[str, float]] = None) -> TopState:
    """Roll the reference to the top and freeze the state it actually reaches.

    The *realised* state, not the keyframe: torque-limited servos run behind
    their targets, so the pose the script asks for at the top and the pose the
    golfer is actually in are different, and it is the second one the downswing
    has to start from.  Taking the keyframe instead would hand the downswing a
    state no backswing ever produces.
    """
    t_top = {p.name: p.time for p in sim.controller.phases}["top"]
    sim.reset()
    while sim.data.time < t_top:
        sim.step()
    tracker = sim.tracker
    angles = np.array([tracker.joint_angles()[n] for n in JOINT_NAMES])
    return TopState(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(),
                    angles=angles,
                    clubhead=tracker.positions()[tracker.index["clubhead"]].copy(),
                    params=dict(params), score=score, info=dict(info or {}))


# ---------------------------------------------------------------------------
# Scoring a candidate
# ---------------------------------------------------------------------------

def shot_score(info: Dict[str, Any], w: Any) -> float:
    """How far and how straight, in the same currency the downswing is paid in.

    Deliberately the terminal shot terms of `RewardWeights` and nothing else:
    if the search optimised something the agent is not paid for, the top it
    picked would be the wrong one for the policy that has to swing from it.

    A miss scores below every strike and is ranked by how close it came, so the
    early generations -- where most candidates whiff -- still have a direction
    to move in.
    """
    if not info.get("contact"):
        return -1.0 - float(info.get("miss", 1.0))
    return (w.carry * float(info["carry"]) / w.carry_ref
            - w.offline * abs(float(info["offline"])) / w.offline_ref
            - w.spin_axis * min(abs(float(info["spin_axis"])) / w.axis_ref,
                                w.axis_cap)
            - w.spin_rate * min(
                max(float(info["backspin"]) - w.spin_target, 0.0) / w.spin_ref,
                w.spin_cap))


#: Action noise the candidate tops are stress-tested against, and how many
#: rollouts at it.  Sized at PPO's starting sigma (`log_std_init=-1.0` gives
#: 0.37) because that is the perturbation the reference has to survive on the
#: first day of training.
ROBUST_SIGMA = 0.25
ROBUST_ROLLOUTS = 3


def _rollout(env: Any, sigma: float, rng: Any) -> Dict[str, Any]:
    env.reset()
    n = env.action_space.shape[0]
    info: Dict[str, Any] = {}
    done = False
    while not done:
        a = (np.zeros(n) if sigma <= 0.0
             else rng.standard_normal(n) * sigma).astype(np.float32)
        _obs, _r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    return info


def evaluate_top(sim: Any, env: Any, params: Dict[str, float], w: Any,
                 rng: Any = None, robust: bool = True
                 ) -> Tuple[float, Dict[str, Any]]:
    """Plan this top and score it -- executed perfectly *and* executed badly.

    Scoring the clean swing alone is what went wrong the first time.  It picks
    a top that is a knife edge: the winner carried 53 m open-loop and fell to
    7.6 m/s of clubhead speed under sigma 0.10 of action noise and 2.0 m/s
    under 0.20, where the hand-written top it beat held 15-17 m/s throughout.
    A reference only a perfect executor can follow is useless to a policy,
    which perturbs it constantly -- it made the trained backswing worse than
    doing nothing, collapsed a warm start, and stalled a from-scratch run at
    2 m/s.

    So a candidate is judged on the mean of one clean rollout and several noisy
    ones.  A top has to be good *and* survive being swung imperfectly.
    """
    apply_top(sim, params)
    rng = rng if rng is not None else np.random.default_rng(0)
    clean = _rollout(env, 0.0, rng)
    if not robust:
        return shot_score(clean, w), clean
    scores = [shot_score(clean, w)]
    for _ in range(ROBUST_ROLLOUTS):
        scores.append(shot_score(_rollout(env, ROBUST_SIGMA, rng), w))
    return float(np.mean(scores)), clean


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _pack(params: Dict[str, float]) -> np.ndarray:
    return np.array([params[k] for k in TOP_PARAMS], dtype=float)


def _unpack(x: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(TOP_PARAMS, x)}


def _clip(x: np.ndarray) -> np.ndarray:
    lo = np.array([TOP_BOUNDS[k][0] for k in TOP_PARAMS])
    hi = np.array([TOP_BOUNDS[k][1] for k in TOP_PARAMS])
    return np.clip(x, lo, hi)


def search_top(sim: Any, env: Any, w: Any, generations: int = 12,
               population: int = 24, elite: int = 6, seed: int = 0,
               verbose: bool = True) -> TopState:
    """Cross-entropy search over the top keyframe.

    CEM rather than anything cleverer for two reasons.  The objective is noisy
    only through the physics (the scripted swing is deterministic, so it is
    actually not noisy at all), and each evaluation costs a full planning pass
    plus three dry swings -- a few hundred of them is the entire budget.  CEM
    gets a usable answer in that many; it is also thirty lines and adds no
    dependency, which matters in a repo that has kept to mujoco and numpy.

    The elite fraction is kept wide (6 of 24).  Narrower collapses the
    distribution onto the first decent top it finds, which here means the
    scripted one it started from.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([TOP_BOUNDS[k][0] for k in TOP_PARAMS])
    hi = np.array([TOP_BOUNDS[k][1] for k in TOP_PARAMS])
    mean = _pack(DEFAULT_TOP)
    std = 0.25 * (hi - lo)

    best_x, best_score, best_info = mean.copy(), -np.inf, {}
    for gen in range(generations):
        # Keep the incumbent in the population.  Without it a generation that
        # samples badly can walk the mean away from a top that was working.
        samples = [best_x.copy() if gen else mean.copy()]
        samples += [_clip(mean + std * rng.standard_normal(len(mean)))
                    for _ in range(population - 1)]

        scored = []
        for x in samples:
            # One rng for the whole search, so every candidate in a generation
            # meets different noise.  Fixing the seed per candidate would let
            # the search overfit one particular sequence of perturbations,
            # which is the same mistake as scoring the clean swing only, one
            # level down.
            score, info = evaluate_top(sim, env, _unpack(x), w, rng=rng)
            scored.append((score, x, info))
            if score > best_score:
                best_score, best_x, best_info = score, x.copy(), info

        scored.sort(key=lambda s: -s[0])
        top_k = np.array([x for _s, x, _i in scored[:elite]])
        mean = top_k.mean(axis=0)
        # Floor the spread so the search cannot converge to a point and stop
        # exploring while there are still generations left to spend.
        std = np.maximum(top_k.std(axis=0), 0.05 * (hi - lo))

        if verbose:
            hit = best_info.get("contact", 0.0)
            carry = best_info.get("carry", 0.0) if hit else 0.0
            axis = best_info.get("spin_axis", 0.0) if hit else 0.0
            print(f"  gen {gen + 1:2d}/{generations}  best {best_score:+.3f}"
                  f"  carry {carry:5.1f} m  axis {axis:+5.1f} deg"
                  f"  (mean spread {std.mean():.1f} deg)")

    params = _unpack(best_x)
    if verbose:
        print(f"\n  best top scored {best_score:+.3f}")
    apply_top(sim, params)
    return capture_top(sim, params, score=float(best_score), info={
        k: float(v) for k, v in best_info.items()
        if isinstance(v, (int, float))})

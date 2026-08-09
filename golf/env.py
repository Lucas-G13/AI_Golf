"""Gymnasium environment: swing the club, hit the ball, get paid for the shot.

Two decisions shape everything here.

**Actions are residuals on the scripted swing.**  A random policy over 36 joint
targets flails; it would never contact the ball, the reward would be flat zero,
and nothing would learn.  Instead the action perturbs the reference swing that
`golf.swing` already provides, so an action of zero reproduces a swing that
does make contact.  There is a gradient to climb from the very first episode.
Set `residual_scale` higher to loosen the leash, or `use_reference=False` to
learn from nothing and find out how bad the exploration problem really is.

**The strike is measured, not simulated.**  Clubhead/ball contact is switched
off, so the club sweeps through unimpeded and we record its state at closest
approach -- exactly what a launch monitor does.  `golf.launch` turns that into
carry and curve.  MuJoCo's rigid contact would both perturb the swing with a
wrong impulse and give a ball speed half of what it should be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .anthropometry import Anthropometry
from .equipment import Club
from .launch import Impact, fly, strike
from .joints import JOINT_NAMES
from .sim import GolfSwingSim
from .util import DEG, RAD


@dataclass
class RewardWeights:
    """What the agent is paid for.

    Contact is shaped so near misses score: without it the reward is a delta
    function in a 36-dimensional space and PPO never finds it.  But the shaped
    term has to stay *small*, because carry is already gated on clean contact
    further down -- so paying a large separate bonus for touching the ball just
    invites the agent to stop there.

    Two things about the balance, learned the hard way.  A first run at
    contact=2.0 / carry=3.0 / torque=0.02 taught the agent to halve its
    clubhead speed: contact went 36% -> 97% while carry *fell* from 11.6 m to
    5.8 m, and reward rose the whole time.  Both mistakes were mine:

    * the per-step costs are charged on every one of ~120 steps, so torque and
      action_rate at 0.02 could total 4.8 over an episode -- while the carry
      actually being achieved paid 0.07.  Swinging hard was priced at 35x what
      the result was worth.
    * contact + centred paid up to 3.0 against carry's 0.07, so 98% of the
      reward was available without hitting the ball anywhere.

    The numbers below are chosen so a good drive (45 m/s, 200 m, 15 m offline)
    scores ~12 and the slow-swing degenerate optimum scores ~2.
    """

    # --- terminal: the shot ---
    contact: float = 1.5        # shaped near-miss term (exploration scaffold)
    centred: float = 0.5        # hitting the middle of the face
    carry: float = 12.0         # per CARRY_REF metres -- this should dominate
    offline: float = 2.5        # per OFFLINE_REF metres of miss
    # Dense credit for clubhead speed, which was the scaffold that got the
    # agent swinging hard early on.  Halved now that it does: paid on every
    # swing whether or not it connects, it was most of what a complete whiff
    # scored, which narrows the gap between hitting the ball and missing it.
    speed: float = 0.8

    # Penalising *where the ball finishes* alone is satisfiable by aiming left
    # and slicing back, which is what the first trained policy did: offline
    # near zero, but a 46 deg spin axis and a smash factor of 1.15 against a
    # possible 1.48.  That oblique strike is what caps distance.  This term
    # prices the curve itself, so the only way to score is to actually deliver
    # the face square to the path.
    spin_axis: float = 3.0      # per AXIS_REF degrees of spin-axis tilt

    # --- per step: charged ~120 times, so these are 1/10 of what they look ---
    torque: float = 0.002       # effort
    action_rate: float = 0.002  # smoothness
    posture: float = 0.05       # only bites once the spine folds past 55 deg

    #: Scales that turn metres into O(1) reward.
    carry_ref: float = 250.0
    offline_ref: float = 50.0
    miss_scale: float = 0.10    # a 10 cm miss still scores ~0.37 on contact
    face_scale: float = 0.035   # ~1 face-width of centredness
    axis_ref: float = 60.0      # degrees of spin axis per unit penalty
    #: Cap on the spin-axis penalty, in units of `axis_ref`.  It has to be
    #: capped: uncapped, a wild enough strike scores worse than never touching
    #: the ball, and the agent learns to avoid it entirely.
    axis_cap: float = 1.25


@dataclass
class EpisodeSpec:
    """Where the swing starts and stops."""

    start: str = "address"      # any phase name, e.g. "transition"
    end: str = "impact"
    follow_through: float = 0.06  # seconds simulated past the end phase
    control_hz: float = 100.0


class GolfEnv(gym.Env):
    """One swing per episode."""

    metadata = {"render_modes": []}

    def __init__(self, episode: Optional[EpisodeSpec] = None,
                 weights: Optional[RewardWeights] = None,
                 residual_scale: float = 0.35,
                 use_reference: bool = True,
                 anthro: Optional[Anthropometry] = None,
                 club: Optional[Club] = None,
                 sim: Optional[GolfSwingSim] = None,
                 seed: Optional[int] = None,
                 **sim_kwargs: Any):
        super().__init__()
        self.spec_ = episode or EpisodeSpec()
        self.w = weights or RewardWeights()
        self.residual_scale = residual_scale
        self.use_reference = use_reference

        sim_kwargs.setdefault("verbose", False)
        # Twice the timestep the viewer uses.  The small step existed to stop
        # the clubhead tunnelling through the ball, and impact is analytic
        # here, so it buys nothing -- measured identical to 0.2 ms over a whole
        # swing, for half the cost.  (0.6 ms is *not* safe: it drifts 63 cm.)
        sim_kwargs.setdefault("timestep", 4e-4)
        self.sim = sim or GolfSwingSim(anthro, club, **sim_kwargs)
        self.tracker = self.sim.tracker
        self.lead_sign = self.sim.anthro.lead_sign

        self._disable_ball_contact()
        self._setup_timing()
        self._cache_ids()

        n_act = len(JOINT_NAMES)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self._prev_action = np.zeros(n_act, np.float32)
        self._closest: Optional[Tuple[float, Impact]] = None
        self._steps = 0
        obs = self._observe()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape,
                                            np.float32)
        self._rng = np.random.default_rng(seed)

    # ---- setup -------------------------------------------------------------
    def _disable_ball_contact(self) -> None:
        """The launch model owns impact; MuJoCo's contact would only get in the
        way.  The ball keeps its site so we can still measure against it."""
        self._head_contype = int(self.sim.model.geom_contype[self.sim._head_gid])
        self._head_conaff = int(
            self.sim.model.geom_conaffinity[self.sim._head_gid])
        self.sim.model.geom_conaffinity[self.sim._head_gid] = 0
        self.sim.model.geom_contype[self.sim._head_gid] = 0

    def enable_ball_contact(self) -> None:
        """Put the ball back in the club's way, for watching a swing.

        Training does not want this -- MuJoCo's impulse is wrong and it
        perturbs the club -- but a swing that passes straight through the ball
        is a strange thing to watch.  The reported numbers still come from the
        launch model either way.
        """
        self.sim.model.geom_contype[self.sim._head_gid] = self._head_contype
        self.sim.model.geom_conaffinity[self.sim._head_gid] = self._head_conaff

    def _setup_timing(self) -> None:
        phases = {p.name: p.time for p in self.sim.controller.phases}
        s = self.spec_
        self.t_start = phases[s.start]
        self.t_end = phases[s.end] + s.follow_through
        self.frame_skip = max(1, round(1.0 / (s.control_hz * self.sim.timestep)))
        self.dt = self.frame_skip * self.sim.timestep
        self.max_steps = int(round((self.t_end - self.t_start) / self.dt))

        # A mid-swing start has to begin from a real body state, not from
        # address.  Roll the reference forward once here and snapshot it --
        # re-simulating those 0.9 s on every reset would cost more than the
        # episode itself.
        self._snapshot = None
        if self.t_start > 0.0:
            self.sim.reset()
            while self.sim.data.time < self.t_start:
                self.sim.step()
            self._snapshot = (self.sim.data.qpos.copy(),
                              self.sim.data.qvel.copy(),
                              float(self.sim.data.time))

    def _cache_ids(self) -> None:
        m = self.sim.model
        self.face_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                          "clubface")
        self.head_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                          "club_head")
        self.ball_sid = self.tracker.sid["ball"]
        self.act_ids = np.array([self.sim.actuator_id(n) for n in JOINT_NAMES])
        self.ctrl_lo = m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[self.act_ids, 1].copy()
        self.force_limit = np.maximum(
            np.abs(m.actuator_forcerange[self.act_ids, 1]), 1e-6)

    # ---- observation -------------------------------------------------------
    def _observe(self) -> np.ndarray:
        phase = np.clip((self.sim.data.time - self.t_start) /
                        max(self.t_end - self.t_start, 1e-9), 0.0, 1.0)
        return np.concatenate([
            self.tracker.observation(),
            [phase, self._closest[0] if self._closest else 1.0],
            self._prev_action,
        ]).astype(np.float32)

    # ---- impact measurement ------------------------------------------------
    def _face_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(face centre, normal, [toe axis, crown axis]) in world coordinates."""
        d = self.sim.data
        centre = d.site_xpos[self.face_sid].copy()
        normal = d.site_xmat[self.face_sid].reshape(3, 3)[:, 2].copy()
        R = d.xmat[self.head_bid].reshape(3, 3)
        toe = -self.lead_sign * R[:, 1]
        crown = R[:, 2]
        return centre, normal, np.stack([toe, crown])

    def _track_impact(self) -> None:
        """Record the closest the face ever came to the ball.

        Runs at physics-step resolution, but only once the club is in the
        neighbourhood: at swing speed the clubhead covers 18 cm between control
        steps, so sampling at the control rate would miss the ball entirely.
        """
        centre, normal, axes = self._face_frame()
        ball = self.sim.data.site_xpos[self.ball_sid]
        delta = ball - centre
        vel = self.tracker.site_velocity("clubhead")

        # The club covers ~7 mm per physics step, which would put that much
        # noise on the strike offset.  Slide along the step analytically to the
        # real closest approach instead, so the measurement does not depend on
        # the timestep: the ball sits still and the face passes it at `vel`.
        speed_sq = float(vel @ vel)
        if speed_sq > 1e-9:
            tau = float(np.clip((delta @ vel) / speed_sq,
                                -self.sim.timestep, self.sim.timestep))
            delta = delta - vel * tau

        dist = float(np.linalg.norm(delta))
        if self._closest is not None and dist >= self._closest[0]:
            return
        self._closest = (dist, Impact(club_velocity=vel.copy(),
                                      face_normal=normal,
                                      face_offset=axes @ delta,
                                      miss_distance=dist))

    # ---- gym ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.sim.reset()
        if self._snapshot is not None:
            qpos, qvel, t = self._snapshot
            self.sim.data.qpos[:] = qpos
            self.sim.data.qvel[:] = qvel
            self.sim.data.time = t
            mujoco.mj_forward(self.sim.model, self.sim.data)

        self._prev_action = np.zeros(len(JOINT_NAMES), np.float32)
        self._closest = None
        self._steps = 0
        self._t0 = self.sim.data.time
        return self._observe(), {}

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        t = self.sim.data.time

        ref = np.zeros(len(JOINT_NAMES))
        if self.use_reference:
            targets = self.sim.controller.targets(t + self.sim.controller.lead)
            ref = np.array([targets[n] for n in JOINT_NAMES]) * DEG
        ctrl = np.clip(ref + action * self.residual_scale,
                       self.ctrl_lo, self.ctrl_hi)
        self.sim.data.ctrl[self.act_ids] = ctrl

        # Only chase the ball once the club is nearby -- the distance check is
        # a couple of microseconds, the full impact measurement is not.
        head = self.tracker.positions()[self.tracker.index["clubhead"]]
        ball = self.sim.data.site_xpos[self.ball_sid]
        near = np.linalg.norm(head - ball) < 0.35

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.sim.model, self.sim.data)
            if near:
                self._track_impact()
        if not near:
            self._track_impact()

        self._steps += 1
        truncated = False
        terminated = self._steps >= self.max_steps
        blown_up = not np.isfinite(self.sim.data.qpos).all()
        if blown_up:
            terminated = True

        reward, info = self._reward(action, terminal=terminated,
                                    blown_up=blown_up)
        self._prev_action = action
        return self._observe(), reward, terminated, truncated, info

    # ---- reward ------------------------------------------------------------
    def _reward(self, action: np.ndarray, terminal: bool,
                blown_up: bool) -> Tuple[float, Dict]:
        w = self.w
        d = self.sim.data
        info: Dict[str, Any] = {}

        # --- per-step costs ---
        force = d.actuator_force[self.act_ids] / self.force_limit
        reward = -w.torque * float(np.mean(force ** 2))
        reward -= w.action_rate * float(np.mean((action - self._prev_action) ** 2))

        # Cheap stand-in for the posture metric: how far the spine has folded
        # past address.  `metrics()` would give the same thing but costs 84 us,
        # which is 5% of a control step for a number used once.
        idx = self.tracker.index
        pos = self.tracker.positions()
        spine = pos[idx["neck"]] - pos[idx["pelvis"]]
        tilt = np.degrees(np.arctan2(np.linalg.norm(spine[:2]),
                                     max(spine[2], 1e-9)))
        reward -= w.posture * max(0.0, tilt - 55.0) / 45.0

        if blown_up:
            return reward - 10.0, {"blown_up": True}
        if not terminal:
            return reward, info

        # --- the shot ---
        if self._closest is None:
            return reward, {"contact": 0.0}
        miss, impact = self._closest

        # shaped: a near miss is worth something, so there is a gradient to
        # follow long before the agent can actually strike the ball
        reward += w.contact * float(np.exp(-(miss / w.miss_scale) ** 2))
        reward += w.speed * min(impact.clubhead_speed / 50.0, 1.5)

        info.update(clubhead_speed=impact.clubhead_speed, miss=miss,
                    on_face=impact.on_face,
                    face_offset=impact.face_offset.tolist())

        if impact.on_face and miss < 0.06:
            launch = strike(impact, self.sim.club)
            flight = fly(launch)
            offcentre = float(np.linalg.norm(impact.face_offset))
            reward += w.centred * float(np.exp(-(offcentre / w.face_scale) ** 2))
            reward += w.carry * flight.carry / w.carry_ref
            reward -= w.offline * abs(flight.lateral) / w.offline_ref
            reward -= w.spin_axis * min(abs(launch.spin_axis) / w.axis_ref,
                                        w.axis_cap)
            info.update(contact=1.0, ball_speed=launch.speed,
                        smash=launch.smash, launch_angle=launch.launch_angle,
                        backspin=launch.backspin_rpm,
                        spin_axis=launch.spin_axis, carry=flight.carry,
                        offline=flight.lateral)
        else:
            info["contact"] = 0.0
        return reward, info


def make_env(rank: int = 0, seed: int = 0, **kwargs: Any):
    """Factory for SubprocVecEnv.  Each worker builds its own model."""
    def _init() -> GolfEnv:
        env = GolfEnv(seed=seed + rank, **kwargs)
        return env
    return _init

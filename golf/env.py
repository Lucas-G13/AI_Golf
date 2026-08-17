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
from .top import TopState, apply_top
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

    # Backspin is charged because `golf.launch` does not charge it.  A real
    # ball at 4600 rpm balloons and drags; the flight model caps lift
    # (LIFT_CAP) but holds DRAG_COEFF constant, so high spin there is close to
    # free -- and the first trained downswing duly delivered 4640 rpm, 7-iron
    # spin off a driver, because nothing said otherwise.  This term stands in
    # for the drag the flight model is missing, not for a preference about
    # what a swing should look like.
    #
    # One-sided, because low spin is already priced: less lift, less carry,
    # and the carry term notices.  Only the high side is invisible.
    spin_rate: float = 2.5      # per SPIN_REF rpm above SPIN_TARGET

    # A longer backswing, paid as how far the clubhead sits from the ball
    # through the top of the swing -- the *mean* over a window around it, not
    # the peak.  Three measures were wrong before this one:
    #
    #   arc length      accumulated just as well by waggling at address
    #   shoulder turn   scores a deep coil with the arms collapsed, which is a
    #                   short arc wearing a long coil's clothes
    #   peak distance   a max over the episode, and a max is biased upward by
    #                   noise: exploration at sigma 0.37 alone took it from the
    #                   reference's 2.35 m to 3.0 m, saturating the term before
    #                   training started and paying for flailing, not for
    #                   structure
    #
    # A mean over ~30 control steps at the top has none of those failures. It
    # still captures both halves of a long swing, the body turning and the arms
    # staying wide, because both hold the clubhead further out for longer.
    #
    # A scaffold, like `speed`, not an end: if a longer backswing really does
    # buy distance then `carry` already pays for it, and this only helps the
    # agent find it. Capped at 2.5 against carry's ~11.7 at 244 m, so it can
    # tilt the swing but never buy a wild one.
    backswing: float = 2.0

    # --- terminal: the top of the backswing (objective="top") ---
    # Paid for arriving at `TopState`, which is the same state the downswing
    # was trained to start from.  Scaled so a perfect arrival is worth ~10,
    # against ~12 for a good drive: neither stage is worth obviously more than
    # the other, because a swing needs both halves.
    #
    # Pose and club are separate terms because they fail differently.  The club
    # is a 1.1 m lever, so two degrees at the shoulder that barely move the
    # joint-angle RMS are 10 cm of clubhead -- score the angles alone and the
    # agent is free to arrive with the club somewhere else entirely.  Score the
    # club alone and it can get there in a posture the downswing cannot use.
    #: Weighted by what the handover error actually costs the shot, which was
    #: measured rather than guessed.  Injecting error into the canonical top
    #: and swinging the trained downswing from it:
    #:
    #:     pose 2.3 deg alone      29/30 contact, 239 m   (against 246 m clean)
    #:     velocity 2.0 rad/s alone 18/30 contact, 190 m, axis +23 deg
    #:
    #: Velocity is roughly eight times more expensive than pose, and the first
    #: version of these weights had it five times *cheaper* -- top_still 2.0 at
    #: vel_ref 4.0 priced 2.1 rad/s at about 1.0, against 10.0 available for
    #: pose and club.  The backswing did exactly what it was paid to: bought
    #: 0.2 cm of clubhead accuracy (3.3 -> 3.1 cm, better than the reference)
    #: and spent 2 rad/s of momentum on it, which cost the downswing 35 m of
    #: carry and twelve strikes out of thirty.  It trained to be worse than
    #: doing nothing.
    top_pose: float = 3.0       # joint angles matching the handover
    top_club: float = 3.0       # where the clubhead actually is
    top_still: float = 8.0      # arriving with the right joint velocities
    pose_scale: float = 12.0    # degrees RMS
    club_scale: float = 0.25    # metres of clubhead error
    #: The handover carries velocity, not just position: the reference already
    #: has the pelvis unwinding while the arms finish, and a backswing rewarded
    #: only for *pose* would happily arrive dead still and hand the downswing a
    #: standing start with none of the stretch it was trained to release.
    #: 2.0 rad/s, capped at 1.5 of it, so the penalty saturates at 3.0 -- above
    #: where a trained backswing lands (2.1) and far above the reference (0.2),
    #: which keeps gradient across the whole range that matters.
    vel_ref: float = 2.0        # rad/s RMS
    vel_cap: float = 1.5

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

    #: A tour driver spins 2400-2700 rpm.  Nothing is charged below this.
    spin_target: float = 2600.0
    #: rpm above `spin_target` per unit penalty.  Chosen the same way
    #: `axis_ref` was, and for the same reason: the policy sits at ~4640 rpm,
    #: which is 0.82 of the way to the cap, so there is gradient exactly where
    #: the agent currently lives.  A tighter reference would saturate above
    #: ~3500 rpm and the 4640 the swing actually produces would be a flat
    #: region with nothing to descend.
    spin_ref: float = 2500.0
    #: Metres of mean reach that pay nothing, sized against the reference the
    #: full swing actually trains on: 2.461 m clean, 2.331 m under training
    #: noise (sigma 0.37 -- noise shortens the average rather than lengthening
    #: it, which is the whole reason this is a mean and not a peak).
    #:
    #: `reach_base` sits just under the *noisy* figure so the term is live from
    #: the first episode rather than in a dead zone, and the agent is paid
    #: first for holding a proper backswing together and then for extending it.
    #: `reach_ref` is 0.40 m of range against 1.73 m of arm-plus-club radius,
    #: saturating at 2.80 m.  Sized wrong once already: against the searched
    #: top's 2.09 m these were 1.90/2.30, which the scripted top's 2.46 m
    #: reference walks straight past, leaving the term pinned at its cap with
    #: nothing left to ask for.
    reach_base: float = 2.30
    reach_ref: float = 2.70
    reach_cap: float = 1.25

    #: Capped for the reason `axis_cap` is: uncapped, one wild high-spin strike
    #: scores below never touching the ball at all.  1.0 rather than 1.25, and
    #: the cap is what was tightened rather than `spin_rate`, deliberately: it
    #: bites at 5100 rpm, above where the swing lives, so the gradient at 4640
    #: is untouched while the worst case stays bounded.  At 1.25 a wild strike
    #: (40 deg axis, 6000 rpm, 120 m) scored +1.32 against +1.27 for a 10 cm
    #: miss -- 0.05 of margin between "hit it badly" and "do not hit it".
    spin_cap: float = 1.0


@dataclass
class EpisodeSpec:
    """Where the swing starts and stops."""

    start: str = "address"      # any phase name, e.g. "transition"
    end: str = "impact"
    follow_through: float = 0.06  # seconds simulated past the end phase
    control_hz: float = 100.0
    #: Gaussian noise on the starting state, in degrees and rad/s.  Both stages
    #: want it, for opposite reasons.  The downswing starts from a fixed
    #: `TopState` and would otherwise see one initial state for its entire
    #: training, which makes it brittle exactly where it has to be robust --
    #: the backswing handing over will land *near* the top, never on it.  The
    #: backswing starts from address, where a zero residual already reproduces
    #: the reference that defined the top, so without noise its task is to
    #: replay a trajectory it is handed for free.
    start_jitter: float = 0.0
    start_vel_jitter: float = 0.0


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
                 top: Optional["TopState"] = None,
                 objective: str = "shot",
                 seed: Optional[int] = None,
                 **sim_kwargs: Any):
        super().__init__()
        self.spec_ = episode or EpisodeSpec()
        self.w = weights or RewardWeights()
        self.residual_scale = residual_scale
        self.use_reference = use_reference
        if objective not in ("shot", "top"):
            raise ValueError(f"objective must be 'shot' or 'top', not {objective!r}")
        if objective == "top" and top is None:
            raise ValueError("objective='top' needs the TopState to aim at")
        self.objective = objective
        self.top = top
        #: Only the shot objective needs the ball chased at physics resolution.
        self._measure_impact = objective == "shot"

        sim_kwargs.setdefault("verbose", False)
        # Twice the timestep the viewer uses.  The small step existed to stop
        # the clubhead tunnelling through the ball, and impact is analytic
        # here, so it buys nothing -- measured identical to 0.2 ms over a whole
        # swing, for half the cost.  (0.6 ms is *not* safe: it drifts 63 cm.)
        sim_kwargs.setdefault("timestep", 4e-4)
        self.sim = sim or GolfSwingSim(anthro, club, **sim_kwargs)
        self.tracker = self.sim.tracker
        self.lead_sign = self.sim.anthro.lead_sign

        # The searched top is not just a state to start from -- it is a
        # different swing.  Re-plan the reference around it so the residual the
        # policy learns is applied to the trajectory the top actually implies,
        # and so the ball sits where *this* swing's arc goes.  Skip it and the
        # downswing would start from the handover state but immediately be
        # dragged toward the keyframes of a swing built for a different top.
        if self.top is not None:
            apply_top(self.sim, self.top.params)

        self.disable_ball_contact()
        self._setup_timing()
        self._cache_ids()

        n_act = len(JOINT_NAMES)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self._prev_action = np.zeros(n_act, np.float32)
        self._closest: Optional[Tuple[float, Impact]] = None
        self._reach_sum, self._reach_n = 0.0, 0
        self._steps = 0
        obs = self._observe()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape,
                                            np.float32)
        self._rng = np.random.default_rng(seed)

    # ---- setup -------------------------------------------------------------
    def disable_ball_contact(self) -> None:
        """The launch model owns impact; MuJoCo's contact would only get in the
        way.  The ball keeps its site so we can still measure against it.

        The original flags are cached on the *simulator*, not on self: the
        split swing puts two environments over one sim, and the second one to
        be built would otherwise save the zeros the first had already written
        and "restore" them, leaving the club passing through the ball forever.
        """
        m, gid = self.sim.model, self.sim._head_gid
        if not hasattr(self.sim, "_head_contact"):
            self.sim._head_contact = (int(m.geom_contype[gid]),
                                      int(m.geom_conaffinity[gid]))
        m.geom_conaffinity[gid] = 0
        m.geom_contype[gid] = 0

    def mean_reach(self) -> float:
        """How far the clubhead sat from the ball through the top, in metres.

        Zero if the episode never covered the window -- a downswing-only
        episode starting at the top catches the tail of it, a backswing-only
        episode the front, and both are fine; only a full swing sees all of it.
        """
        return self._reach_sum / self._reach_n if self._reach_n else 0.0

    def club_clear_of_ball(self, margin: float = 0.30) -> bool:
        """Is the clubhead far enough from the ball to make the ball solid?

        Wanted because the two overlap *at address*.  `place_ball` tees the
        ball where the clubface passes at impact, and a swing that reaches
        further at impact than at address -- which the searched top does --
        leaves the ball sitting inside the clubhead's toe by about 16 mm at
        the start.  Turn contact on there and MuJoCo resolves the penetration
        by firing the ball off at 9 m/s before the swing has moved.

        Moving the ball is not the fix: the downswing policy was trained
        against that ball position, and moving it moves the target it learned.
        """
        head = self.tracker.positions()[self.tracker.index["clubhead"]]
        ball = self.sim.data.site_xpos[self.ball_sid]
        return bool(np.linalg.norm(head - ball) > margin)

    def enable_ball_contact(self) -> None:
        """Put the ball back in the club's way, for watching a swing.

        Training does not want this -- MuJoCo's impulse is wrong and it
        perturbs the club -- but a swing that passes straight through the ball
        is a strange thing to watch.  The reported numbers still come from the
        launch model either way.

        Callers watching from address should gate this on
        `club_clear_of_ball()` rather than switching it on at t=0.
        """
        m, gid = self.sim.model, self.sim._head_gid
        m.geom_contype[gid], m.geom_conaffinity[gid] = self.sim._head_contact

    def _setup_timing(self) -> None:
        phases = {p.name: p.time for p in self.sim.controller.phases}
        s = self.spec_
        self.t_start = phases[s.start]
        self.t_end = phases[s.end] + s.follow_through
        # The window the backswing term is measured over: the top of the swing,
        # from a little before the `top` keyframe to a little past `transition`
        # (where the clubhead actually reaches its furthest, the body having
        # started down while the arms are still going back).  Named phases
        # rather than absolute times, so it follows `--tempo`.
        self._reach_lo = phases["top"] - 0.15
        self._reach_hi = phases["transition"] + 0.05
        self.frame_skip = max(1, round(1.0 / (s.control_hz * self.sim.timestep)))
        self.dt = self.frame_skip * self.sim.timestep
        self.max_steps = int(round((self.t_end - self.t_start) / self.dt))

        # A mid-swing start has to begin from a real body state, not from
        # address.  Roll the reference forward once here and snapshot it --
        # re-simulating those 0.9 s on every reset would cost more than the
        # episode itself.  An explicit `TopState` replaces that snapshot
        # outright: it *is* the handover, and rolling the reference forward
        # would only reproduce whatever top the reference happens to reach.
        self._snapshot = None
        if self.top is not None and self.spec_.start == "top":
            self._snapshot = (self.top.qpos.copy(), self.top.qvel.copy(),
                              self.t_start)
        elif self.t_start > 0.0:
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
        # qpos/qvel addresses of the actuated joints, for jittering the start
        # and for scoring the handover.  Not the same indices: the golfer's
        # root and the ball's freejoint sit in qpos too.
        self.q_ids = np.array([self.tracker.qadr[n] for n in JOINT_NAMES])
        self.dof_ids = np.array([self.tracker.dadr[n] for n in JOINT_NAMES])
        jids = [self.tracker.jid[n] for n in JOINT_NAMES]
        self.jnt_lo = m.jnt_range[jids, 0].copy()
        self.jnt_hi = m.jnt_range[jids, 1].copy()
        self._jitters = bool(self.spec_.start_jitter or
                             self.spec_.start_vel_jitter)

    def _jitter_start(self) -> None:
        """Perturb the starting state.  See `EpisodeSpec.start_jitter`.

        Only the actuated joints move.  Jittering the free root as well would
        put the golfer's pelvis somewhere their feet cannot follow -- the feet
        are pinned, so the legs would spend the first few steps fighting the
        equality constraint instead of swinging.
        """
        if not self._jitters:
            return
        d = self.sim.data
        if self.spec_.start_jitter:
            q = d.qpos[self.q_ids] + self._rng.normal(
                0.0, self.spec_.start_jitter * DEG, len(self.q_ids))
            d.qpos[self.q_ids] = np.clip(q, self.jnt_lo, self.jnt_hi)
        if self.spec_.start_vel_jitter:
            d.qvel[self.dof_ids] += self._rng.normal(
                0.0, self.spec_.start_vel_jitter, len(self.dof_ids))

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
        """`options={"keep_state": True}` starts from whatever state the
        simulator is already in, instead of the episode's own starting state.

        That is how the two halves are played as one swing: the backswing
        episode ends with the golfer at the top, and the downswing episode
        picks up from exactly that body rather than teleporting to the
        canonical handover.  Which is the whole test -- if the backswing did
        not really arrive, the downswing has to cope with where it did.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not (options or {}).get("keep_state"):
            self.sim.reset()
            if self._snapshot is not None:
                qpos, qvel, t = self._snapshot
                self.sim.data.qpos[:] = qpos
                self.sim.data.qvel[:] = qvel
                self.sim.data.time = t
            self._jitter_start()
            if self._snapshot is not None or self._jitters:
                mujoco.mj_forward(self.sim.model, self.sim.data)

        self._prev_action = np.zeros(len(JOINT_NAMES), np.float32)
        self._closest = None
        self._reach_sum, self._reach_n = 0.0, 0
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

        # How far the club sits from the ball through the top, averaged over
        # the window rather than maxed over the episode -- see `backswing` in
        # RewardWeights for why the max was wrong.
        head = self.tracker.positions()[self.tracker.index["clubhead"]]
        ball = self.sim.data.site_xpos[self.ball_sid]
        if self._reach_lo <= t <= self._reach_hi:
            self._reach_sum += float(np.linalg.norm(head - ball))
            self._reach_n += 1
        reach = float(np.linalg.norm(head - ball))

        # Only chase the ball once the club is nearby -- the distance check is
        # a couple of microseconds, the full impact measurement is not.  The
        # backswing never goes near it and is not scored on it either.
        near = self._measure_impact and reach < 0.35

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.sim.model, self.sim.data)
            if near:
                self._track_impact()
        if self._measure_impact and not near:
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

        if self.objective == "top":
            return self._top_reward(reward)

        # --- the shot ---
        if self._closest is None:
            return reward, {"contact": 0.0}
        miss, impact = self._closest

        # shaped: a near miss is worth something, so there is a gradient to
        # follow long before the agent can actually strike the ball
        reward += w.contact * float(np.exp(-(miss / w.miss_scale) ** 2))
        reward += w.speed * min(impact.clubhead_speed / 50.0, 1.5)

        info.update(clubhead_speed=impact.clubhead_speed, miss=miss,
                    on_face=impact.on_face, reach=self.mean_reach(),
                    face_offset=impact.face_offset.tolist())

        if impact.on_face and miss < 0.06:
            launch = strike(impact, self.sim.club)
            flight = fly(launch)
            offcentre = float(np.linalg.norm(impact.face_offset))
            reward += w.centred * float(np.exp(-(offcentre / w.face_scale) ** 2))
            # Gated on the strike, alongside carry, and not paid like `speed`
            # is.  Paid unconditionally it is a bounty for whiffing with style:
            # a 30 cm miss off a long backswing scored +3.22 against +1.56 for
            # a 10 cm miss off a normal one, and below some genuinely bad
            # strikes -- the agent would have been paid to stop hitting the
            # ball.  A long backswing is only worth something if the club still
            # arrives, which is also true on a golf course.
            reward += w.backswing * min(
                max(self.mean_reach() - w.reach_base, 0.0)
                / max(w.reach_ref - w.reach_base, 1e-9), w.reach_cap)
            reward += w.carry * flight.carry / w.carry_ref
            reward -= w.offline * abs(flight.lateral) / w.offline_ref
            reward -= w.spin_axis * min(abs(launch.spin_axis) / w.axis_ref,
                                        w.axis_cap)
            reward -= w.spin_rate * min(
                max(launch.backspin_rpm - w.spin_target, 0.0) / w.spin_ref,
                w.spin_cap)
            info.update(contact=1.0, ball_speed=launch.speed,
                        smash=launch.smash, launch_angle=launch.launch_angle,
                        backspin=launch.backspin_rpm,
                        spin_axis=launch.spin_axis, carry=flight.carry,
                        offline=flight.lateral)
        else:
            info["contact"] = 0.0
        return reward, info

    def _top_reward(self, reward: float) -> Tuple[float, Dict]:
        """Score the backswing on the state it hands over.

        Three ways of asking the same question, because each alone is
        satisfiable without the others: the joint angles say the body is in the
        right shape, the clubhead says the club actually went with it, and the
        velocities say the golfer arrived under control and still moving the
        way the downswing expects.
        """
        top = self.top
        d = self.sim.data
        angles = d.qpos[self.q_ids] * RAD
        ang_err = float(np.sqrt(np.mean((angles - top.angles) ** 2)))
        club = self.tracker.positions()[self.tracker.index["clubhead"]]
        club_err = float(np.linalg.norm(club - top.clubhead))
        vel_err = float(np.sqrt(np.mean(
            (d.qvel[self.dof_ids] - top.qvel[self.dof_ids]) ** 2)))

        w = self.w
        reward += w.top_pose * float(np.exp(-(ang_err / w.pose_scale) ** 2))
        reward += w.top_club * float(np.exp(-(club_err / w.club_scale) ** 2))
        reward -= w.top_still * min(vel_err / w.vel_ref, w.vel_cap)
        return reward, {"top_angle_err": ang_err, "top_club_err": club_err,
                        "top_vel_err": vel_err,
                        # "Did this episode arrive?", in the sense the
                        # downswing cares about.  The velocity bound is the
                        # point: without it this read 100% arrived while the
                        # composed swing was losing 35 m and half its strikes,
                        # because 8 deg and 20 cm are both far looser than the
                        # shot's actual tolerance and velocity went unchecked.
                        "top_reached": float(ang_err < 5.0 and club_err < 0.10
                                             and vel_err < 0.5)}


def make_env(rank: int = 0, seed: int = 0, **kwargs: Any):
    """Factory for SubprocVecEnv.  Each worker builds its own model."""
    def _init() -> GolfEnv:
        env = GolfEnv(seed=seed + rank, **kwargs)
        return env
    return _init

"""Train a golf swing with PPO.

    python train.py                      # full swing, 10 workers
    python train.py --start transition   # downswing only, much faster
    python train.py --steps 20_000_000
    python train.py --resume runs/golf/final.zip

Or train the two halves separately, either side of a fixed handover:

    python find_top.py                                    # choose the top
    python train.py --stage downswing --out runs/downswing
    python train.py --stage backswing --out runs/backswing

The two stages are only separable because they agree on one state -- the
`TopState` in `runs/top.npz`, which the downswing starts from and the backswing
is paid for reaching.  Both stages load the same file and re-plan the reference
swing around it, so "the top" means the same thing on both sides.  Change the
top and both policies are stale.

Each worker builds its own MuJoCo model (a few seconds), then the swing runs
entirely in that process.  The bottleneck is PPO itself, not the physics -- see
the note in README.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# keep MuJoCo and BLAS single-threaded: we get our parallelism from workers,
# and letting each of them spawn 12 threads makes everything slower
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback)
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecNormalize)

from golf.env import EpisodeSpec, GolfEnv, RewardWeights


KEYS = ("contact", "miss", "clubhead_speed", "ball_speed", "carry", "offline",
        "smash", "launch_angle", "spin_axis", "backspin", "reach")

#: What a backswing episode reports instead.  `top_reached` is the one to watch:
#: the errors say how far off the handover was, this says whether it landed
#: close enough for the downswing to have been trained on something like it.
TOP_KEYS = ("top_reached", "top_angle_err", "top_club_err", "top_vel_err")


class EpisodeStats(BaseCallback):
    """Log what the swings are actually doing, not just the reward.

    Flushes on a fixed interval rather than once a key has N samples: the
    interesting keys -- carry, ball speed, spin -- only exist on episodes that
    made contact, so a fixed sample count would report them tens of thousands
    of steps later than the rest, exactly when you most want to see them.

    `gate` is the key that marks a terminal episode worth recording, and it
    differs by stage: a backswing episode has no `miss` because it never goes
    near the ball.
    """

    def __init__(self, keys: tuple = KEYS, gate: str = "miss",
                 prefix: str = "shot", flush_every: int = 4000,
                 min_samples: int = 10):
        super().__init__()
        self.keys = keys
        self.gate = gate
        self.prefix = prefix
        self.flush_every = flush_every
        self.min_samples = min_samples
        self._buf: dict[str, list[float]] = {}
        self._since = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if self.gate not in info:
                continue
            for key in self.keys:
                if key in info:
                    self._buf.setdefault(key, []).append(float(info[key]))
        self._since += self.training_env.num_envs
        if self._since >= self.flush_every:
            self._since = 0
            for key, vals in self._buf.items():
                if len(vals) >= self.min_samples:
                    self.logger.record(f"{self.prefix}/{key}", float(np.mean(vals)))
                    self._buf[key] = []
        return True


def stage_setup(args):
    """(episode spec, objective, TopState) for the stage being trained.

    The jitter defaults differ because the two stages are brittle in opposite
    places.  The downswing would otherwise see one single starting state for
    its whole run and learn a trajectory rather than a controller -- and the
    backswing that eventually feeds it will land near the top, never exactly
    on it.  The backswing would otherwise start from address, where a zero
    residual already replays the reference that *defined* the top, so it would
    be handed full marks for doing nothing.
    """
    # The split stages load the searched top; the full swing does not, unless
    # asked with --use-top.
    #
    # It is the better reference only if executed exactly.  Measured over 24
    # rollouts per cell with matched noise, clubhead speed against action
    # sigma:
    #
    #                sigma 0     0.1        0.2        0.37
    #     searched   20.8 m/s   10.2       5.7        2.0     (contact 24/24)
    #     scripted   15.0 m/s   15.6       13.0       11.0    (contact 3/24)
    #
    # A searched top keeps contact and loses all its speed; the scripted one
    # keeps speed and loses contact.  For the *downswing* that does not matter,
    # because it starts at the top from an exact state with no backswing noise
    # ahead of it -- which is how it reached 244 m.  A full swing has 82 steps
    # of its own exploration noise before it ever gets there, so it would begin
    # training at 2 m/s, and speed early is the scaffold the whole thing is
    # built on.  The full-swing runs that worked started from the scripted top.
    top = None
    if args.stage != "full" or args.use_top:
        from golf.top import TopState
        top = TopState.load(args.top)

    if args.stage == "downswing":
        jitter = 1.5 if args.jitter is None else args.jitter
        vel = 0.15 if args.vel_jitter is None else args.vel_jitter
        return (EpisodeSpec(start="top", end="impact",
                            control_hz=args.control_hz,
                            start_jitter=jitter, start_vel_jitter=vel),
                "shot", top)

    if args.stage == "backswing":
        # 6 deg / 0.5 rad/s, and it barely matters: measured, the position
        # servos reject even 15 deg and 2 rad/s over the 0.82 s to the top,
        # arriving 0.79 deg / 12.6 cm out instead of 0.36 deg / 3.3 cm.  What
        # actually makes this stage learnable is its own exploration noise --
        # random actions at sigma 0.25 miss the top by 25 cm and PPO starts at
        # 0.37 -- so the task is to stop wrecking a backswing that already
        # works, not to discover one.  The jitter is kept because robustness at
        # the handover is free here, not because it creates the problem.
        jitter = 6.0 if args.jitter is None else args.jitter
        vel = 0.5 if args.vel_jitter is None else args.vel_jitter
        return (EpisodeSpec(start="address", end="top", follow_through=0.0,
                            control_hz=args.control_hz,
                            start_jitter=jitter, start_vel_jitter=vel),
                "top", top)

    return (EpisodeSpec(start=args.start, control_hz=args.control_hz,
                        start_jitter=args.jitter or 0.0,
                        start_vel_jitter=args.vel_jitter or 0.0),
            "shot", top)


def build_env(args, n_envs: int):
    spec, objective, top = stage_setup(args)
    weights = RewardWeights()

    def factory(rank: int):
        def _init():
            return GolfEnv(episode=spec, weights=weights,
                           residual_scale=args.residual,
                           use_reference=not args.no_reference,
                           top=top, objective=objective,
                           seed=args.seed + rank, base=args.base)
        return _init

    venv = SubprocVecEnv([factory(i) for i in range(n_envs)], start_method="spawn")
    venv = VecMonitor(venv)

    # A 260-dim observation mixing radians, metres and m/s badly needs this.
    # On resume, carry the old statistics over: starting them from scratch
    # feeds the policy observations on a scale it has never seen, and it swings
    # badly for the ~150k steps the running means take to converge.
    if args.resume:
        stats = Path(args.resume).parent / "vecnormalize.pkl"
        if stats.exists():
            venv = VecNormalize.load(str(stats), venv)
            venv.training = True
            print(f"  reusing observation statistics from {stats}")
            return venv
        print("  [warn] no vecnormalize.pkl beside the checkpoint -- the "
              "policy will swing badly until fresh statistics converge")
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0,
                        gamma=args.gamma)


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO golf swing")
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--stage", default="full",
                    choices=["full", "downswing", "backswing"],
                    help="full swing, or one half either side of the handover "
                         "in --top (see find_top.py)")
    ap.add_argument("--top", default="runs/top.npz",
                    help="the TopState both halves agree on")
    ap.add_argument("--use-top", action="store_true",
                    help="build the full swing's reference around --top too. "
                         "Off by default: a searched top is worth 5 m/s more "
                         "executed perfectly and 9 m/s less executed with "
                         "exploration noise on it (see stage_setup)")
    ap.add_argument("--jitter", type=float, default=None,
                    help="degrees of noise on the starting pose "
                         "(stage-dependent default)")
    ap.add_argument("--vel-jitter", type=float, default=None,
                    help="rad/s of noise on the starting velocity")
    ap.add_argument("--start", default="address",
                    help="phase the episode starts from: address | transition "
                         "(--stage overrides this)")
    ap.add_argument("--base", default="feet", choices=["feet", "free", "pinned"])
    ap.add_argument("--control-hz", type=float, default=100.0)
    ap.add_argument("--residual", type=float, default=0.35,
                    help="radians of authority the policy has over the "
                         "reference swing")
    ap.add_argument("--no-reference", action="store_true",
                    help="learn from scratch instead of perturbing the "
                         "scripted swing (much harder)")
    # Note: shrinking the policy's output layer to compensate for a bigger
    # --residual seems like the careful thing to do and is actively harmful.
    # It is exactly neutral on unsaturated joints, so the only ones it changes
    # are those already pinned at the limit -- which then jump up to 2x on
    # their own.  Measured: 0/25 contact, against 25/25 for resuming naively.
    # Uniform scaling preserves the coordination; graded scaling destroys it.
    ap.add_argument("--gamma", type=float, default=0.997)
    # 3e-4 with 10 epochs over a 2816-sample rollout drives approx_kl to 0.3+
    # and pins clip_fraction at 0.67 -- the actor lurches, the entropy freezes
    # and the reward flatlines while the critic sits happily at 0.85 explained
    # variance.  Gentler updates on a bigger rollout, with target_kl as a hard
    # stop, is what this 36-dimensional action space wants.
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/golf")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    needs_top = args.stage != "full" or args.use_top
    if needs_top and not Path(args.top).exists():
        raise SystemExit(
            f"no top at {args.top}. The split stages have to agree on one, so "
            f"make it first:\n    python find_top.py")
    if args.stage != "full":
        print(f"stage: {args.stage}, handing over at {args.top}")
    else:
        print(f"stage: full swing, reference built around "
              f"{args.top if args.use_top else 'the hand-written top'}")

    print(f"building {args.envs} workers ...")
    venv = build_env(args, args.envs)

    # tensorboard is optional -- nice for watching a multi-hour run, but not
    # worth failing the training script over
    try:
        import tensorboard  # noqa: F401
        tb_log = str(out / "tb")
    except ImportError:
        tb_log = None
        print("  (pip install tensorboard for live curves)")

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
        print(f"resumed from {args.resume}")
    else:
        model = PPO("MlpPolicy", venv, learning_rate=args.lr,
                    n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.001, vf_coef=0.5,
                    target_kl=args.target_kl,
                    max_grad_norm=0.5, seed=args.seed, device="cpu",
                    # Start quiet.  SB3's default sigma of 1.0 sends every
                    # joint through the full residual range from step one,
                    # which destroys the reference swing the agent is supposed
                    # to be improving -- contact rate starts near 2% instead of
                    # near 100%.  exp(-1) ~ 0.37 keeps early swings recognisable
                    # and PPO widens the distribution itself if it pays.
                    policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
                    tensorboard_log=tb_log, verbose=1)

    stats = (EpisodeStats(TOP_KEYS, gate="top_angle_err", prefix="top")
             if args.stage == "backswing" else EpisodeStats())
    callbacks = [stats,
                 CheckpointCallback(save_freq=max(1, 500_000 // args.envs),
                                    save_path=str(out), name_prefix="ppo",
                                    # Without this the checkpoints hold the
                                    # policy but not the observation
                                    # normalisation it was trained against,
                                    # which makes them useless on their own --
                                    # the stats would otherwise only be written
                                    # by the finally block on a clean exit.
                                    save_vecnormalize=True)]
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    reset_num_timesteps=args.resume is None)
    finally:
        model.save(out / "final")
        venv.save(str(out / "vecnormalize.pkl"))
        venv.close()
        print(f"saved {out / 'final.zip'}")


if __name__ == "__main__":
    main()

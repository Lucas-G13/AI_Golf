"""Train a golf swing with PPO.

    python train.py                      # full swing, 10 workers
    python train.py --start transition   # downswing only, much faster
    python train.py --steps 20_000_000
    python train.py --resume runs/golf/final.zip

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
        "smash", "launch_angle", "spin_axis")


class ShotStats(BaseCallback):
    """Log what the swings are actually doing, not just the reward.

    Flushes on a fixed interval rather than once a key has N samples: the
    interesting keys -- carry, ball speed, spin -- only exist on episodes that
    made contact, so a fixed sample count would report them tens of thousands
    of steps later than the rest, exactly when you most want to see them.
    """

    def __init__(self, flush_every: int = 4000, min_samples: int = 10):
        super().__init__()
        self.flush_every = flush_every
        self.min_samples = min_samples
        self._buf: dict[str, list[float]] = {}
        self._since = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "miss" not in info:
                continue
            for key in KEYS:
                if key in info:
                    self._buf.setdefault(key, []).append(float(info[key]))
        self._since += self.training_env.num_envs
        if self._since >= self.flush_every:
            self._since = 0
            for key, vals in self._buf.items():
                if len(vals) >= self.min_samples:
                    self.logger.record(f"shot/{key}", float(np.mean(vals)))
                    self._buf[key] = []
        return True


def build_env(args, n_envs: int):
    spec = EpisodeSpec(start=args.start, control_hz=args.control_hz)
    weights = RewardWeights()

    def factory(rank: int):
        def _init():
            return GolfEnv(episode=spec, weights=weights,
                           residual_scale=args.residual,
                           use_reference=not args.no_reference,
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
    ap.add_argument("--start", default="address",
                    help="phase the episode starts from: address | transition")
    ap.add_argument("--base", default="feet", choices=["feet", "free", "pinned"])
    ap.add_argument("--control-hz", type=float, default=100.0)
    ap.add_argument("--residual", type=float, default=0.35,
                    help="radians of authority the policy has over the "
                         "reference swing")
    ap.add_argument("--no-reference", action="store_true",
                    help="learn from scratch instead of perturbing the "
                         "scripted swing (much harder)")
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

    callbacks = [ShotStats(),
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

"""Loading a trained swing and playing it back.

The one thing that is easy to get wrong here: a policy trained under
`VecNormalize` expects *normalised* observations.  The observation is 260
dimensions mixing radians, metres and metres per second, so handing the policy
raw values produces a swing that looks superficially plausible and is nothing
like the one that was trained.  `TrainedSwing` keeps the saved statistics with
the policy so the two cannot be separated by accident.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .env import GolfEnv


class TrainedSwing:
    """A trained policy, with the observation statistics it was trained under."""

    def __init__(self, run: Path, env: GolfEnv):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        run = Path(run)
        model_path = run / "final.zip" if run.is_dir() else run
        if not model_path.exists():
            raise FileNotFoundError(
                f"no policy at {model_path}. Train one with:\n"
                f"    python train.py --out {run}")
        self.model = PPO.load(model_path, device="cpu")
        self.path = model_path

        stats = (run if run.is_dir() else run.parent) / "vecnormalize.pkl"
        self.norm = None
        if stats.exists():
            holder = DummyVecEnv([lambda: env])
            self.norm = VecNormalize.load(str(stats), holder)
            self.norm.training = False
            self.norm.norm_reward = False
        else:
            print(f"  [warn] {stats.name} is missing -- the policy will be fed "
                  f"raw observations and will not swing the way it was trained")

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.norm is not None:
            obs = self.norm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action

    #: Play the distribution's mean action.  Worth trying both: PPO optimises
    #: the *stochastic* policy, so the mean action is not always its best one,
    #: and here it launches the ball noticeably higher than the training
    #: average did.
    deterministic: bool = True

    def __str__(self) -> str:
        return f"trained policy from {self.path}"


def scripted_swing(env: GolfEnv) -> Callable[[np.ndarray], np.ndarray]:
    """The reference swing: a zero residual reproduces it exactly."""
    zero = np.zeros(env.action_space.shape, np.float32)
    return lambda _obs: zero


def run_episode(env: GolfEnv, act: Callable, keep_state: bool = False) -> Dict:
    obs, _ = env.reset(options={"keep_state": True} if keep_state else None)
    info: Dict = {}
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        done = terminated or truncated
    return info


def run_composed(back_env: GolfEnv, back_act: Callable,
                 down_env: GolfEnv, down_act: Callable) -> Dict:
    """Swing the backswing policy to the top, then the downswing policy on.

    Two environments over one simulator, rather than one environment that
    switches policies halfway.  They have to be separate because each policy
    reads a `phase` input that runs 0 -> 1 across *its own* episode and was
    normalised against its own statistics; run them both inside a single
    address-to-impact episode and each sees a phase signal it never trained
    against.  The handover is `keep_state`: the downswing starts from the body
    the backswing actually delivered, not from the canonical top.

    The returned info is the shot, with the backswing's handover error folded
    in so a bad join is visible in the same place as a bad strike.
    """
    top = run_episode(back_env, back_act)
    shot = run_episode(down_env, down_act, keep_state=True)
    return {**shot, **{f"handover_{k.removeprefix('top_')}": v
                       for k, v in top.items() if k.startswith("top_")}}


def report(env: GolfEnv, act: Callable, episodes: int = 20,
           rows: Optional[List[Dict]] = None) -> None:
    """Average the shot over several swings and print it like a launch monitor."""
    if rows is None:
        rows = [run_episode(env, act) for _ in range(episodes)]
    hits = [r for r in rows if r.get("contact")]
    rule = "=" * 74
    print()
    print(rule)
    print(f"SHOT  ({len(hits)}/{len(rows)} made contact)")
    print(rule)
    if not hits:
        misses = [r["miss"] for r in rows if "miss" in r]
        if misses:
            print(f"  closest approach {np.mean(misses) * 100:.1f} cm")
        return

    def mean(key: str) -> float:
        vals = [r[key] for r in hits if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    def spread(key: str) -> float:
        vals = [r[key] for r in hits if key in r]
        return float(np.std(vals)) if vals else float("nan")

    print(f"  clubhead speed  {mean('clubhead_speed'):6.1f} m/s   "
          f"({mean('clubhead_speed') * 2.23694:5.1f} mph)")
    print(f"  ball speed      {mean('ball_speed'):6.1f} m/s   "
          f"smash {mean('smash'):.2f}")
    print(f"  launch angle    {mean('launch_angle'):6.1f} deg")
    print(f"  spin axis       {mean('spin_axis'):+6.1f} deg   "
          f"(0 = dead straight)")
    print(f"  strike          {mean('miss') * 100:6.1f} cm from centre")
    print(f"  carry           {mean('carry'):6.1f} m     "
          f"+/- {spread('carry'):.1f}")
    print(f"  offline         {mean('offline'):+6.1f} m     "
          f"+/- {spread('offline'):.1f}")

    if any("handover_angle_err" in r for r in rows):
        reached = float(np.mean([r.get("handover_reached", 0.0) for r in rows]))
        print(f"\n  handover        {reached * 100:.0f}% arrived at the top   "
              f"({np.mean([r['handover_angle_err'] for r in rows]):.1f} deg "
              f"RMS, {np.mean([r['handover_club_err'] for r in rows]) * 100:.0f}"
              f" cm of club)")


#: GLFW key codes the viewer hands to `key_callback`.
_KEY_SPACE, _KEY_ENTER, _KEY_BACKSPACE, _KEY_R = 32, 257, 259, ord("R")


def watch(env: GolfEnv, act: Callable, slowmo: float = 0.25,
          loop: bool = False, flight: float = 1.2) -> None:
    """Play the swing in the interactive viewer."""
    watch_stages([(env, act)], slowmo=slowmo, loop=loop, flight=flight)


def watch_stages(stages: List[Tuple[GolfEnv, Callable]],
                 slowmo: float = 0.25, loop: bool = False,
                 flight: float = 1.2) -> None:
    """Play consecutive episodes as one continuous swing in the viewer.

    Swings once and then holds the finish, so you can orbit around it, rather
    than restarting on a loop.  Space, Enter, Backspace or R swings again;
    `loop=True` restores the old repeat-forever behaviour.

    Several stages are how the split swing is watched: each hands over with
    `keep_state`, so the clock and the body run straight through the top
    instead of resetting, and what you see is one swing rather than two.
    """
    import mujoco.viewer

    # The ball is made solid part-way into the swing rather than up front --
    # see `GolfEnv.club_clear_of_ball`.  Training runs with contact off so the
    # analytic launch model owns impact, but for watching it you want to see
    # the ball actually leave.
    sim = stages[0][0].sim
    replay = {"go": True}

    def on_key(keycode: int) -> None:
        if keycode in (_KEY_SPACE, _KEY_ENTER, _KEY_BACKSPACE, _KEY_R):
            replay["go"] = True

    with mujoco.viewer.launch_passive(sim.model, sim.data,
                                      key_callback=on_key) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation = 135, -12
        viewer.cam.distance = 4.2
        viewer.cam.lookat[:] = sim.tracker.positions()[
            sim.tracker.index["pelvis"]]

        while viewer.is_running():
            if not replay["go"]:
                # Idle on the finished swing: keep the window responsive but
                # stop advancing physics.
                viewer.sync()
                time.sleep(0.02)
                continue
            replay["go"] = loop

            # Off again for the replay: the club is back at address, back on
            # top of the ball.
            stages[0][0].disable_ball_contact()
            armed = False
            wall0 = time.time()
            info: Dict = {}
            for i, (env, act) in enumerate(stages):
                obs, _ = env.reset(
                    options={"keep_state": True} if i else None)
                done = False
                while viewer.is_running() and not done:
                    obs, _, terminated, truncated, info = env.step(act(obs))
                    done = terminated or truncated
                    if not armed and env.club_clear_of_ball():
                        env.enable_ball_contact()
                        armed = True
                    _pace(viewer, sim, wall0, slowmo)

            # Let the ball actually get somewhere before freezing the frame.
            end = sim.data.time + flight
            while viewer.is_running() and sim.data.time < end:
                mujoco.mj_step(sim.model, sim.data)
                _pace(viewer, sim, wall0, slowmo)

            _print_shot(info)
            if not loop:
                print("  [space] swing again, [Esc] quit")


def _pace(viewer, sim, wall0: float, slowmo: float) -> None:
    """Hold the sim to wall-clock time, played back in slow motion."""
    lag = wall0 + sim.data.time / max(slowmo, 1e-3) - time.time()
    if lag > 0.001:
        viewer.sync()
        time.sleep(min(lag, 0.02))


def _print_shot(info: Dict) -> None:
    if info.get("contact"):
        print(f"  {info['clubhead_speed']:.1f} m/s -> "
              f"{info['ball_speed']:.1f} m/s ball, "
              f"{info['carry']:.0f} m carry, "
              f"{info['offline']:+.0f} m offline")
    else:
        print(f"  missed by {info.get('miss', float('nan')) * 100:.0f} cm")


def contact_sheet(env: GolfEnv, act: Callable, path: str) -> None:
    """Write a strip of the swing's phases to a PNG."""
    contact_sheet_stages([(env, act)], path)


def contact_sheet_stages(stages: List[Tuple[GolfEnv, Callable]],
                         path: str) -> None:
    """Write a strip of the swing's phases to a PNG, across all stages."""
    import mujoco
    from PIL import Image

    stages[0][0].disable_ball_contact()
    armed = False
    sim = stages[0][0].sim
    times = {p.name: p.time for p in sim.controller.phases}
    renderer = mujoco.Renderer(sim.model, height=520, width=420)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.2, 0.3, 1.0]
    cam.distance, cam.elevation, cam.azimuth = 4.0, -8, 90

    frames: List[np.ndarray] = []
    seen: set = set()
    for i, (env, act) in enumerate(stages):
        obs, _ = env.reset(options={"keep_state": True} if i else None)
        done = False
        while not done:
            for name, t in times.items():
                if name not in seen and sim.data.time >= t:
                    renderer.update_scene(sim.data, cam)
                    frames.append(renderer.render().copy())
                    seen.add(name)
            obs, _, terminated, truncated, _ = env.step(act(obs))
            done = terminated or truncated
            if not armed and env.club_clear_of_ball():
                env.enable_ball_contact()
                armed = True

    Image.fromarray(np.concatenate(frames, axis=1)).save(path)
    print(f"  wrote {path} ({len(frames)} frames)")

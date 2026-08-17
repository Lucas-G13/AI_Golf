"""Find the top of the backswing that both training stages hand over at.

    python find_top.py                    # search, write runs/top.npz
    python find_top.py --generations 24   # look harder

The downswing is trained starting from this state and the backswing is trained
to reach it, so it has to be chosen before either.  It is scored by what the
downswing is paid for -- carry and straightness through the launch model -- and
not by whether it looks like a golf swing.

Re-run this and both policies are stale: they were trained against a handover
that has moved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from golf import Anthropometry, Club, GolfSwingSim
from golf.env import EpisodeSpec, GolfEnv, RewardWeights
from golf.top import DEFAULT_TOP, evaluate_top, search_top


def shot_line(tag: str, score: float, info: dict) -> str:
    if not info.get("contact"):
        return (f"  {tag:12s} score {score:+.3f}   no contact, closest "
                f"{info.get('miss', float('nan')) * 100:.1f} cm")
    return (f"  {tag:12s} score {score:+.3f}   {info['clubhead_speed']:.1f} m/s"
            f" -> {info['ball_speed']:.1f} m/s ball, smash {info['smash']:.2f},"
            f" carry {info['carry']:.1f} m, axis {info['spin_axis']:+.1f} deg,"
            f" spin {info['backspin']:.0f} rpm,"
            f" offline {info['offline']:+.1f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description="search for the top of the swing")
    ap.add_argument("--out", default="runs/top.npz")
    ap.add_argument("--generations", type=int, default=16)
    ap.add_argument("--population", type=int, default=28)
    ap.add_argument("--elite", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="feet", choices=["feet", "free", "pinned"])
    ap.add_argument("--height", type=float, default=1.78)
    ap.add_argument("--mass", type=float, default=78.0)
    args = ap.parse_args()

    print("building model ...")
    anthro = Anthropometry(height=args.height, mass=args.mass)
    # The training timestep, not the viewer's.  A top that only works at 0.2 ms
    # is not a top the agent can use.
    sim = GolfSwingSim(anthro, Club(), base=args.base, timestep=4e-4,
                       verbose=False)
    env = GolfEnv(sim=sim, episode=EpisodeSpec(start="address"))
    w = RewardWeights()

    base_score, base_info = evaluate_top(sim, env, DEFAULT_TOP, w)
    print("\nthe scripted top, for reference:")
    print(shot_line("scripted", base_score, base_info))
    print("  (score is the mean of a clean swing and 3 at sigma 0.25 -- a top "
          "has to survive being swung imperfectly)")

    print(f"\nsearching ({args.generations} x {args.population} = "
          f"{args.generations * args.population} swings) ...")
    top = search_top(sim, env, w, generations=args.generations,
                     population=args.population, elite=args.elite,
                     seed=args.seed)

    final_score, final_info = evaluate_top(sim, env, top.params, w)
    print(shot_line("searched", final_score, final_info))
    print("\nthe top it chose:")
    print(top.describe())

    scripted_turn = (DEFAULT_TOP["pelvis_turn"] + DEFAULT_TOP["lumbar_turn"]
                     + DEFAULT_TOP["thorax_turn"])
    found_turn = (top.params["pelvis_turn"] + top.params["lumbar_turn"]
                  + top.params["thorax_turn"])
    print(f"\n  shoulder turn {scripted_turn:+.0f} -> {found_turn:+.0f} deg")
    print(f"  clubhead at the top: {np.round(top.clubhead, 3)}")

    out = Path(args.out)
    top.save(out)
    print(f"\nwrote {out}  --  train against it with:")
    print(f"    python train.py --stage downswing --top {out}")
    print(f"    python train.py --stage backswing --top {out}")


if __name__ == "__main__":
    main()

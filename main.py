"""Command-line front end for the golf swing model.

Watch the scripted reference swing:

    python main.py                  # 3D viewer, quarter-speed swing
    python main.py --report         # headless, print the kinematic report
    python main.py --csv swing.csv  # log every tracked joint every 2 ms
    python main.py --xml golf.xml   # dump the generated MJCF and exit

Watch a swing the agent learned:

    python main.py --policy runs/golf
    python main.py --policy runs/golf --report --episodes 50
    python main.py --policy runs/golf --frames swing.png

The model itself lives in the `golf` package -- start with `golf/__init__.py`.
"""

from __future__ import annotations

import argparse

import mujoco

from golf import (Anthropometry, Club, GolfSwingSim, print_model_summary,
                  print_report, run_swing, view_swing)
from golf.env import EpisodeSpec, GolfEnv
from golf.policy import (TrainedSwing, contact_sheet, report, scripted_swing,
                         watch)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MuJoCo human golf swing")

    golfer = ap.add_argument_group("golfer and equipment")
    golfer.add_argument("--height", type=float, default=1.78,
                        help="golfer height (m)")
    golfer.add_argument("--mass", type=float, default=78.0,
                        help="golfer mass (kg)")
    golfer.add_argument("--hand", choices=["right", "left"], default="right")
    golfer.add_argument("--club-length", type=float, default=1.14)

    physics = ap.add_argument_group("physics")
    physics.add_argument("--base", choices=["feet", "free", "pinned"],
                         default="feet",
                         help="feet: pelvis free but feet planted (default); "
                              "free: full balance physics; pinned: pelvis welded")
    physics.add_argument("--timestep", type=float, default=2e-4)
    physics.add_argument("--strength", type=float, default=1.0,
                         help="scale on every joint torque limit")

    swing = ap.add_argument_group("swing")
    swing.add_argument("--tempo", type=float, default=1.0,
                       help="stretch the whole swing in time (1.5 = slower)")
    swing.add_argument("--lead", type=float, default=0.07,
                       help="how far ahead of the clock the servos are fed "
                            "their reference, to cancel tracking lag (s)")
    swing.add_argument("--no-calibrate", action="store_true",
                       help="skip the tuning swings and ball placement")

    learned = ap.add_argument_group("trained policy")
    learned.add_argument("--policy", default=None,
                         help="play a swing trained by train.py, e.g. runs/golf")
    learned.add_argument("--backswing", default=None,
                         help="policy for address->top; with --downswing, "
                              "plays the two halves as one swing.  'reference' "
                              "uses the scripted backswing, which is the best "
                              "one there is -- see README")
    learned.add_argument("--downswing", default=None,
                         help="policy for top->impact.  On its own, plays the "
                              "downswing from the canonical top")
    learned.add_argument("--top", default="runs/top.npz",
                         help="the handover state the halves agree on")
    learned.add_argument("--episodes", type=int, default=20,
                         help="swings to average over with --policy --report")
    learned.add_argument("--frames", default=None,
                         help="write a strip of the swing's phases to this PNG")
    learned.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of playing the mean one")

    out = ap.add_argument_group("output")
    out.add_argument("--slowmo", type=float, default=0.25,
                     help="viewer playback rate (0.25 = quarter speed)")
    out.add_argument("--loop", action="store_true",
                     help="repeat the swing forever instead of holding the "
                          "finish until you press space")
    out.add_argument("--no-view", action="store_true", help="run headless")
    out.add_argument("--report", action="store_true",
                     help="print the kinematic report (implies --no-view)")
    out.add_argument("--csv", type=str, default=None,
                     help="write every tracked landmark to this CSV")
    out.add_argument("--xml", type=str, default=None,
                     help="dump the generated MJCF here and exit")
    return ap.parse_args()


def play_learned(args, anthro: Anthropometry, club: Club) -> None:
    """Watch, measure or photograph a swing the agent learned."""
    env = GolfEnv(episode=EpisodeSpec(start="address"), anthro=anthro,
                  club=club, base=args.base)
    if args.policy == "reference":
        act, what = scripted_swing(env), "scripted reference swing"
    else:
        swing = TrainedSwing(args.policy, env)
        swing.deterministic = not args.stochastic
        act, what = swing.act, str(swing)
    print(f"playing: {what}")

    if args.frames:
        contact_sheet(env, act, args.frames)
    elif args.report or args.no_view:
        report(env, act, episodes=args.episodes)
    else:
        print("\nviewer -- drag to orbit, scroll to zoom, space to swing "
              "again, Esc to quit")
        watch(env, act, slowmo=args.slowmo, loop=args.loop)


def play_split(args, anthro: Anthropometry, club: Club) -> None:
    """The separately trained halves, joined at the top.

    One simulator, two environments over it -- see `policy.run_composed` for
    why they cannot be a single episode.  With only `--downswing` the backswing
    is skipped entirely and the swing starts from the canonical handover, which
    is what the downswing was trained against and so measures it in isolation.
    """
    from golf.policy import (contact_sheet_stages, report, run_composed,
                             run_episode, scripted_swing, watch_stages)
    from golf.top import TopState

    top = TopState.load(args.top)
    sim = GolfSwingSim(anthro, club, base=args.base, timestep=4e-4,
                       verbose=False)
    # Built before the backswing env, and it matters: the first environment
    # over a sim is the one that caches the ball's real contact flags.
    down_env = GolfEnv(sim=sim, episode=EpisodeSpec(start="top", end="impact"),
                       top=top, objective="shot")
    down = TrainedSwing(args.downswing, down_env)
    down.deterministic = not args.stochastic

    stages = []
    what = ""
    if args.backswing:
        back_env = GolfEnv(sim=sim, top=top, objective="top",
                           episode=EpisodeSpec(start="address", end="top",
                                               follow_through=0.0))
        if args.backswing == "reference":
            act, what = scripted_swing(back_env), "the scripted backswing"
        else:
            back = TrainedSwing(args.backswing, back_env)
            back.deterministic = not args.stochastic
            act, what = back.act, str(back)
        stages.append((back_env, act))
        what += "\n     then "
    else:
        what = "from the canonical top, "
    stages.append((down_env, down.act))
    print(f"playing: {what}{down}")

    if args.frames:
        contact_sheet_stages(stages, args.frames)
    elif args.report or args.no_view:
        if len(stages) == 1:
            rows = [run_episode(down_env, down.act)
                    for _ in range(args.episodes)]
        else:
            rows = [run_composed(stages[0][0], stages[0][1], down_env,
                                 down.act) for _ in range(args.episodes)]
        report(down_env, down.act, rows=rows)
    else:
        print("\nviewer -- drag to orbit, scroll to zoom, space to swing "
              "again, Esc to quit")
        watch_stages(stages, slowmo=args.slowmo, loop=args.loop)


def main() -> None:
    args = parse_args()

    anthro = Anthropometry(height=args.height, mass=args.mass,
                           handedness=args.hand)
    club = Club(length=args.club_length)

    if args.downswing:
        print("building model ...")
        play_split(args, anthro, club)
        return

    if args.policy:
        print("building model ...")
        play_learned(args, anthro, club)
        return

    print("building model ...")
    sim = GolfSwingSim(anthro, club, base=args.base, timestep=args.timestep,
                       strength=args.strength, tempo=args.tempo,
                       lead=args.lead, calibrate=not args.no_calibrate)

    if args.xml:
        with open(args.xml, "w") as fh:
            fh.write(sim.xml)
        print(f"wrote {args.xml}")
        return

    print_model_summary(sim)

    if args.report or args.csv or args.no_view:
        rows, names, impact = run_swing(sim, csv_path=args.csv)
        sim.reset()
        mujoco.mj_forward(sim.model, sim.data)
        print_report(sim, rows, names, impact)
        if args.csv:
            print(f"\n  wrote {args.csv}")
    else:
        print("\nviewer -- drag to orbit, scroll to zoom, space to swing "
              "again, Esc to quit")
        view_swing(sim, slowmo=args.slowmo, loop=args.loop)


if __name__ == "__main__":
    main()

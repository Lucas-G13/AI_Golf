"""Command-line front end for the golf swing model.

    python main.py                  # 3D viewer, quarter-speed swing
    python main.py --report         # headless, print the kinematic report
    python main.py --csv swing.csv  # log every tracked joint every 2 ms
    python main.py --xml golf.xml   # dump the generated MJCF and exit

The model itself lives in the `golf` package -- start with `golf/__init__.py`.
"""

from __future__ import annotations

import argparse

import mujoco

from golf import (Anthropometry, Club, GolfSwingSim, print_model_summary,
                  print_report, run_swing, view_swing)


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

    out = ap.add_argument_group("output")
    out.add_argument("--slowmo", type=float, default=0.25,
                     help="viewer playback rate (0.25 = quarter speed)")
    out.add_argument("--no-view", action="store_true", help="run headless")
    out.add_argument("--report", action="store_true",
                     help="print the kinematic report (implies --no-view)")
    out.add_argument("--csv", type=str, default=None,
                     help="write every tracked landmark to this CSV")
    out.add_argument("--xml", type=str, default=None,
                     help="dump the generated MJCF here and exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    anthro = Anthropometry(height=args.height, mass=args.mass,
                           handedness=args.hand)
    club = Club(length=args.club_length)

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
        print("\nopening viewer -- drag to orbit, scroll to zoom, Esc to quit")
        view_swing(sim, slowmo=args.slowmo)


if __name__ == "__main__":
    main()

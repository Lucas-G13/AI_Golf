"""Running a swing and showing what happened: CSV logging, tables, viewer."""

from __future__ import annotations

import csv
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .joints import JOINT_NAMES
from .landmarks import TRACKED
from .sim import GolfSwingSim

_RULE = "=" * 74
_CSV_METRICS = ("hip_turn", "shoulder_turn", "x_factor", "lead_wrist_cock",
                "side_tilt", "clubhead_speed", "ball_speed")


# ===========================================================================
# Running
# ===========================================================================

def run_swing(sim: GolfSwingSim, csv_path: Optional[str] = None
              ) -> Tuple[List[Dict[str, float]], List[str], Dict[str, float]]:
    """Run one swing.  Returns (per-phase metrics, phase names, impact)."""
    sim.reset()
    tr = sim.tracker
    phases = sim.controller.phases
    rows: List[Dict[str, float]] = []
    names: List[str] = []
    next_phase = 0

    fh = writer = None
    if csv_path:
        fh = open(csv_path, "w", newline="")
        writer = csv.writer(fh)
        header = ["t", "phase"]
        header += [f"{n}_{ax}" for n in TRACKED for ax in "xyz"]
        header += [f"q_{n}" for n in JOINT_NAMES]
        header += list(_CSV_METRICS)
        writer.writerow(header)

    impact = {"clubhead_speed": 0.0, "ball_speed": 0.0, "t": 0.0,
              "attack_angle": 0.0, "club_path": 0.0, "hit": False}
    struck = False
    n_steps = int((sim.duration + 0.25) / sim.timestep)
    log_every = max(1, int(0.002 / sim.timestep))

    for i in range(n_steps):
        t = sim.data.time
        if next_phase < len(phases) and t >= phases[next_phase].time:
            rows.append(tr.metrics())
            names.append(phases[next_phase].name)
            next_phase += 1

        if writer and i % log_every == 0:
            m = tr.metrics()
            row = [f"{t:.5f}", sim.controller.phase_at(t)]
            row += [f"{v:.5f}" for v in tr.positions().ravel()]
            row += [f"{v:.5f}" for v in tr.joint_angles().values()]
            row += [f"{m[k]:.4f}" for k in _CSV_METRICS]
            writer.writerow(row)

        # Impact is read off the actual clubhead/ball contact: clubhead speed
        # at first touch, ball speed once it has separated.
        if not struck and sim.ball_contact():
            struck = True
            m = tr.metrics()
            impact = {"clubhead_speed": m["clubhead_speed"], "ball_speed": 0.0,
                      "t": t, "hit": True, "attack_angle": m["attack_angle"],
                      "club_path": m["club_path"]}
        if struck:
            impact["ball_speed"] = max(impact["ball_speed"], tr.ball_speed())

        sim.step()

    if fh:
        fh.close()
    return rows, names, impact


#: GLFW key codes the viewer hands to `key_callback`.
_REPLAY_KEYS = (32, 257, 259, ord("R"))     # space, enter, backspace, R


def view_swing(sim: GolfSwingSim, slowmo: float = 0.25,
               loop: bool = False) -> None:
    """Open the interactive viewer and play the swing in slow motion.

    Plays once and holds the finish so you can look around it.  Space, Enter,
    Backspace or R swings again; `loop=True` repeats forever.
    """
    import mujoco.viewer

    replay = {"go": True}

    def on_key(keycode: int) -> None:
        if keycode in _REPLAY_KEYS:
            replay["go"] = True

    with mujoco.viewer.launch_passive(sim.model, sim.data,
                                      key_callback=on_key) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        viewer.cam.distance = 4.2
        viewer.cam.lookat[:] = sim.tracker.positions()[
            sim.tracker.index["pelvis"]]
        wall0 = time.time()

        while viewer.is_running():
            if not replay["go"]:
                # idle on the finished swing without advancing physics
                viewer.sync()
                time.sleep(0.02)
                continue
            replay["go"] = loop
            sim.reset()
            wall0 = time.time()

            while viewer.is_running() and sim.data.time <= sim.duration + 0.6:
                sim.step()
                # sync at ~60 Hz of wall clock, played back in slow motion
                lag = wall0 + sim.data.time / max(slowmo, 1e-3) - time.time()
                if lag > 0.001:
                    viewer.sync()
                    time.sleep(min(lag, 0.02))
            if not loop:
                print("  [space] swing again, [Esc] quit")


# ===========================================================================
# Printing
# ===========================================================================

def print_model_summary(sim: GolfSwingSim) -> None:
    m = sim.model
    print(_RULE)
    print("MODEL")
    print(_RULE)
    print(f"  golfer            {sim.anthro.height:.2f} m, "
          f"{sim.anthro.mass:.0f} kg, {sim.anthro.handedness}-handed")
    print(f"  club              {sim.club.length:.2f} m, "
          f"{sim.club.total_mass * 1000:.0f} g, {sim.club.loft_deg:.1f} deg loft")
    print(f"  base              {sim.base}")
    print(f"  bodies/dof/act    {m.nbody} / {m.nv} / {m.nu}   "
          f"(timestep {m.opt.timestep * 1e3:.2f} ms)")
    print(f"  tracked joints    {len(TRACKED)} landmarks, "
          f"{len(JOINT_NAMES)} actuated joints")


def print_pose_table(sim: GolfSwingSim, title: str) -> None:
    tr = sim.tracker
    pos = tr.positions()
    ego = tr.egocentric()
    seg = tr.parent_relative()
    print()
    print(_RULE)
    print(f"{title}  (t = {sim.data.time:.3f} s)")
    print(_RULE)
    print(f"{'landmark':<12}{'world x':>9}{'y':>8}{'z':>8}"
          f"{'| golfer f/l/u':>28}{'| from parent':>16}")
    for i, n in enumerate(TRACKED):
        p, e = pos[i], ego[i]
        print(f"{n:<12}{p[0]:9.3f}{p[1]:8.3f}{p[2]:8.3f}   "
              f"{e[0]:8.3f}{e[1]:8.3f}{e[2]:8.3f}   "
              f"{tr.parent[n] or '-':>10}{np.linalg.norm(seg[i]):7.3f}")


def print_metrics(rows: List[Dict[str, float]], phases: List[str]) -> None:
    print()
    print(_RULE)
    print("SWING KINEMATICS")
    print(_RULE)
    print(f"{'phase':<11}{'t':>6}{'hip':>8}{'shldr':>9}{'X-fac':>8}"
          f"{'cock':>7}{'tilt':>8}{'club m/s':>9}{'head z':>9}")
    for ph, r in zip(phases, rows):
        print(f"{ph:<11}{r['t']:>6.3f}{r['hip_turn']:>8.1f}"
              f"{r['shoulder_turn']:>9.1f}{r['x_factor']:>8.1f}"
              f"{r['lead_wrist_cock']:>7.1f}{r['side_tilt']:>8.1f}"
              f"{r['clubhead_speed']:>9.2f}{r['clubhead_height']:>9.3f}")


def print_distance_matrix(sim: GolfSwingSim, subset: Sequence[str]) -> None:
    tr = sim.tracker
    D = tr.pairwise_distances()
    idx = [tr.index[n] for n in subset]
    print()
    print(_RULE)
    print("PAIRWISE LANDMARK DISTANCES (m) -- SwingTracker.pairwise_distances()")
    print(_RULE)
    print(" " * 11 + "".join(f"{n[:9]:>10}" for n in subset))
    for n, i in zip(subset, idx):
        print(f"{n:<11}" + "".join(f"{D[i, j]:10.3f}" for j in idx))


def print_impact(impact: Dict[str, float]) -> None:
    print()
    print(_RULE)
    print("IMPACT")
    print(_RULE)
    if not impact["hit"]:
        print("  the club missed the ball")
        return
    mph = 2.23694
    smash = impact["ball_speed"] / max(impact["clubhead_speed"], 1e-6)
    print(f"  contact at t = {impact['t']:.3f} s")
    print(f"  clubhead speed  {impact['clubhead_speed']:6.1f} m/s  "
          f"({impact['clubhead_speed'] * mph:5.1f} mph)")
    print(f"  ball speed      {impact['ball_speed']:6.1f} m/s  "
          f"({impact['ball_speed'] * mph:5.1f} mph)")
    print(f"  smash factor    {smash:6.2f}")
    print(f"  attack angle    {impact['attack_angle']:6.1f} deg")
    print(f"  club path       {impact['club_path']:6.1f} deg")


def print_report(sim: GolfSwingSim, rows: List[Dict[str, float]],
                 names: List[str], impact: Dict[str, float]) -> None:
    """The whole kinematic report, from a swing that has already been run."""
    lead = sim.anthro.lead
    print_pose_table(sim, "ADDRESS POSITION")
    print_metrics(rows, names)
    print_distance_matrix(sim, ["pelvis", "thorax", f"shoulder_{lead}",
                                f"wrist_{lead}", "clubhead", "ball"])
    print_impact(impact)

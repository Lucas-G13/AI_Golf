"""RL_Golf -- a MuJoCo model of a human golf swing.

Stage 1 of the project: get the *mechanics* right.  This package builds a
full-body humanoid golfer (36 actuated DOF) holding a driver with both hands,
standing at a teed ball, and drives it through a scripted swing so the
kinematics can be inspected.  Everything the RL agent will eventually need to
observe is exposed by `SwingTracker`: every major joint centre, in world
coordinates, in an egocentric frame, relative to its parent in the kinematic
chain, and as a full pairwise offset/distance tensor.

What the model is
-----------------
* segment lengths and masses scaled from height and body mass (Winter 2009)
* joints with human ranges of motion and physiological torque limits, driven by
  PD position servos: hips/knees/ankles (3+1+3 per leg), lumbar and thoracic
  spine (3 each), neck (2), and shoulder/elbow/wrist (3+2+2 per arm)
* the club is rigidly gripped by the lead hand with a built-in grip angle, and
  the trail hand is held on the shaft by an equality constraint -- so the arms
  are a genuine closed kinematic loop, as they are in a real swing
* driver, ball and tee at legal masses and dimensions; the feet are pinned to
  the ground at heel and toe so the golfer cannot topple

What the scripted swing is *not*
--------------------------------
It is a hand-written reference trajectory, not a good golf swing.  It reaches
about 18 m/s (40 mph) of clubhead speed, roughly a third of a competent driver.
The limit is structural: with both feet planted the pelvis runs out of rotation
about 40 deg past square, so the body stops turning right when the clubhead
needs it most, and an open-loop PD script cannot exploit the ground reaction or
the whip of the release.  Finding a swing that does is the point of the RL
agent -- this package exists to make the body, the club and the measurement
right first.

Conventions
-----------
World:   +x = target line (where the ball is meant to go), +z = up
Golfer:  local +x = the direction they face, +y = their left, +z = up the spine
Joint angles are radians internally, degrees everywhere a human reads them.
Positive = anatomical positive (flexion, abduction, internal rotation); for the
trunk's bend and rotation, positive = toward the target, either handedness.

Layout
------
    anthropometry.py  how big the golfer is
    equipment.py      the club, ball and tee
    joints.py         every joint, its limits and strength; the address pose
    landmarks.py      what gets measured
    model.py          generates the MJCF
    tracking.py       SwingTracker -- reading the body
    swing.py          the scripted keyframes and their interpolation
    ik.py             inverse kinematics against the live model
    posture.py        getting the golfer to address
    planner.py        turning the script into a swing that works
    sim.py            GolfSwingSim -- ties it together
    report.py         running a swing, logging it, showing it
"""

from .anthropometry import Anthropometry
from .equipment import (BALL_CENTRE_HEIGHT, BALL_FORWARD, BALL_MASS,
                        BALL_RADIUS, TEE_HEIGHT, Club)
from .joints import ADDRESS, ADDRESS_LEAN, JOINT_NAMES, all_joints
from .landmarks import TRACKED, parent_map
from .model import build_model_xml
from .report import (print_distance_matrix, print_impact, print_metrics,
                     print_model_summary, print_pose_table, print_report,
                     run_swing, view_swing)
from .sim import GolfSwingSim
from .swing import Phase, SwingController, swing_script
from .tracking import SwingTracker

__all__ = [
    "Anthropometry", "Club", "GolfSwingSim", "SwingTracker", "SwingController",
    "Phase", "swing_script", "build_model_xml", "all_joints", "parent_map",
    "run_swing", "view_swing", "print_report", "print_model_summary",
    "print_pose_table", "print_metrics", "print_distance_matrix",
    "print_impact",
    "TRACKED", "JOINT_NAMES", "ADDRESS", "ADDRESS_LEAN",
    "BALL_RADIUS", "BALL_MASS", "TEE_HEIGHT", "BALL_FORWARD",
    "BALL_CENTRE_HEIGHT",
]

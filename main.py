"""
RL_Golf -- a MuJoCo model of a human golf swing.

Stage 1 of the project: get the *mechanics* right.  This file builds a
full-body humanoid golfer (36 actuated DOF) holding a driver with both hands,
standing at a teed ball, and drives it through a scripted swing so the
kinematics can be inspected.  Everything the RL agent will eventually need to
observe is exposed by `SwingTracker`: every major joint centre, expressed in
world coordinates, in an egocentric frame, relative to its parent in the
kinematic chain, and as a full pairwise offset/distance tensor.

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
about 18 m/s (40 mph) of clubhead speed, roughly a third of a competent
driver.  The limit is structural: with both feet planted the pelvis runs out of
rotation about 40 deg past square, so the body stops turning right when the
clubhead needs it most, and an open-loop PD script cannot exploit the ground
reaction or the whip of the release.  Finding a swing that does is the point of
the RL agent -- this file exists to make the body, the club and the measurement
right first.

Conventions
-----------
World:   +x = target line (where the ball is meant to go)
         +z = up
Golfer:  local +x = the direction the golfer faces (down at the ball)
         local +y = the golfer's left  (= +x world for a right-handed player)
         local +z = up along the spine
Joint angles are in radians internally, degrees everywhere a human reads them.
Positive = anatomical positive (flexion, abduction, internal rotation), and for
the axial joints positive = rotation *toward the target* (the downswing
direction) for both handednesses.

Model construction is a two-pass process because a few things can only be
measured after the model exists:
  pass 1  build -> solve the address pose (IK) -> measure the clubface heading
                   and the foot pin locations
  pass 2  rebuild with the face squared to the target and the feet pinned,
          re-solve the address pose and settle under gravity, plan the swing
          (legs from the pelvis turn, arms from where the club has to be),
          then run a few dry swings to tune the arc height and drop the ball
          where the clubhead really passes through it.

Run
---
    python main.py                 # 3D viewer, slow-motion scripted swing
    python main.py --report        # no viewer, print the kinematic report
    python main.py --csv swing.csv # log every tracked joint every 2 ms
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

DEG = math.pi / 180.0
RAD = 180.0 / math.pi


# ===========================================================================
# 1.  Body dimensions
# ===========================================================================

@dataclass(frozen=True)
class Anthropometry:
    """Segment lengths and masses from Winter (2009), as fractions of standing
    height H and body mass M.  Everything downstream scales off these, so a
    different golfer is one line of code."""

    height: float = 1.78          # m
    mass: float = 78.0            # kg
    handedness: str = "right"     # "right" -> lead side is the left side

    # --- lengths -----------------------------------------------------------
    @property
    def foot_len(self) -> float: return 0.152 * self.height
    @property
    def ankle_h(self) -> float: return 0.039 * self.height
    @property
    def shank(self) -> float: return 0.246 * self.height
    @property
    def thigh(self) -> float: return 0.245 * self.height
    @property
    def hip_h(self) -> float: return 0.530 * self.height     # hip joint centre
    @property
    def hip_sep(self) -> float: return 0.191 * self.height   # between hip centres
    @property
    def trunk(self) -> float: return 0.288 * self.height     # hip -> shoulder
    @property
    def biacromial(self) -> float: return 0.259 * self.height
    @property
    def shoulder_sep(self) -> float:
        """Glenohumeral joint centres, which sit medial and inferior to the
        acromions -- using the full biacromial breadth here would put the arms
        ~4 cm too far apart and neither hand would reach the grip."""
        return 0.84 * self.biacromial
    @property
    def upperarm(self) -> float: return 0.186 * self.height
    @property
    def forearm(self) -> float: return 0.146 * self.height
    @property
    def hand_len(self) -> float: return 0.108 * self.height

    # --- masses ------------------------------------------------------------
    @property
    def m_foot(self) -> float: return 0.0145 * self.mass
    @property
    def m_shank(self) -> float: return 0.0465 * self.mass
    @property
    def m_thigh(self) -> float: return 0.1000 * self.mass
    @property
    def m_pelvis(self) -> float: return 0.1420 * self.mass
    @property
    def m_abdomen(self) -> float: return 0.1390 * self.mass
    @property
    def m_thorax(self) -> float: return 0.2160 * self.mass
    @property
    def m_head(self) -> float: return 0.0810 * self.mass
    @property
    def m_upperarm(self) -> float: return 0.0280 * self.mass
    @property
    def m_forearm(self) -> float: return 0.0160 * self.mass
    @property
    def m_hand(self) -> float: return 0.0060 * self.mass

    # --- handedness --------------------------------------------------------
    @property
    def lead(self) -> str: return "l" if self.handedness == "right" else "r"
    @property
    def trail(self) -> str: return "r" if self.handedness == "right" else "l"
    @property
    def lead_sign(self) -> float:
        """+1 if the lead side is the golfer's left (right-handed player)."""
        return 1.0 if self.handedness == "right" else -1.0


@dataclass(frozen=True)
class Club:
    """A driver.  `grip_angle_deg` is the angle built into the grip between the
    hand's long axis and the shaft -- real wrists only deviate ~20 deg, the rest
    of the 80-90 deg "wrist cock" seen at the top comes from how the shaft sits
    diagonally across the palm."""

    length: float = 1.14           # butt end -> sole, 45 in
    head_mass: float = 0.200       # kg
    shaft_mass: float = 0.065
    grip_mass: float = 0.050
    loft_deg: float = 10.5
    grip_angle_deg: float = 35.0
    lead_hand_drop: float = 0.085  # lead hand this far below the butt
    trail_hand_drop: float = 0.155  # trail hand below that (overlap grip)


BALL_RADIUS = 0.02135              # m, R&A/USGA minimum diameter 42.67 mm
BALL_MASS = 0.0459                 # kg, maximum legal mass
TEE_HEIGHT = 0.030                 # ball sits this high before the radius
BALL_FORWARD = 0.10                # driver ball position, forward of centre (m)


# ===========================================================================
# 2.  Joint definitions
# ===========================================================================
# (name, axis, range_lo_deg, range_hi_deg, kp, force_limit_Nm, damping)
# Axes are written in the *zero pose* frame (standing upright, arms hanging,
# palms facing the thighs), which is each body's own frame -- so e.g. a wrist
# axis rotates correctly with forearm pronation because pronation is a joint
# further up the same chain.

JointSpec = Tuple[str, str, float, float, float, float, float]


def _side_joints(side: str) -> List[JointSpec]:
    """Leg + arm joints for one side.  s = +1 for the left, -1 for the right,
    so that a positive joint value means the same anatomical motion on both."""
    s = 1.0 if side == "l" else -1.0
    return [
        # ---- leg ----------------------------------------------------------
        (f"hip_{side}_flex",  "0 -1 0",   -25, 120, 6000, 220, 6.0),
        (f"hip_{side}_abd",   f"{s} 0 0",  -30,  45, 5000, 160, 6.0),
        (f"hip_{side}_rot",   f"0 0 {-s}", -45,  45, 3000,  90, 4.0),
        (f"knee_{side}",      "0 1 0",       0, 150, 5000, 200, 6.0),
        (f"ankle_{side}_dorsi", "0 -1 0",  -50,  30, 4000, 160, 4.0),
        (f"ankle_{side}_roll",  f"{s} 0 0", -20,  20, 2000,  60, 3.0),
        # shank rotation over the foot (tibial + subtalar, plus a little sole
        # slip).  Without it a golfer with both feet planted has no way to turn
        # the pelvis except by sinking into the ground.
        (f"ankle_{side}_rot",   f"0 0 {-s}", -30,  30, 1500,  70, 3.0),
        # ---- arm ----------------------------------------------------------
        (f"shoulder_{side}_abd",  f"{s} 0 0", -60, 170, 1600, 110, 2.0),
        (f"shoulder_{side}_flex", "0 -1 0",   -70, 170, 1600, 110, 2.0),
        (f"shoulder_{side}_rot",  f"0 0 {-s}", -90,  90,  700,  60, 1.5),
        # a few degrees of hyperextension is real, and it matters here: with
        # both elbows locked at exactly 0 the two arms and the shaft form a
        # rigid closed loop and the club cannot move at all
        (f"elbow_{side}_flex",    "0 -1 0",     -8, 150,  900,  80, 1.5),
        (f"elbow_{side}_pro",     f"0 0 {-s}", -85,  85,  300,  25, 0.8),
        (f"wrist_{side}_flex",    f"{-s} 0 0", -70,  80,  400,  45, 0.5),
        # radial deviation range is extended past anatomical ROM: together with
        # the grip angle it represents the golf "cock/hinge" of the wrist.  Its
        # torque limit lumps in the forearm and grip, which is what actually
        # holds and then releases the club's lag angle.
        (f"wrist_{side}_dev",     "0 -1 0",    -25,  55,  700,  70, 0.5),
    ]


def _spine_joints(lead_sign: float = 1.0) -> List[JointSpec]:
    """Trunk joints.  Lateral bend and axial rotation are signed relative to
    the *target*, not the body: positive = bending/turning toward the target,
    which is the downswing direction for a golfer of either handedness."""
    k = lead_sign
    return [
        ("lumbar_flex", "0 -1 0",   -25,  50, 6000, 250, 6.0),
        ("lumbar_bend", f"{-k} 0 0", -28,  28, 6000, 220, 6.0),
        # the lumbar spine only gives ~13 deg of axial rotation each way
        ("lumbar_rot",  f"0 0 {k}",  -15,  15, 4000, 140, 4.0),
        ("thorax_flex", "0 -1 0",   -22,  35, 5000, 220, 5.0),
        ("thorax_bend", f"{-k} 0 0", -30,  30, 5000, 200, 5.0),
        ("thorax_rot",  f"0 0 {k}",  -50,  50, 4000, 160, 4.0),
        ("neck_flex",   "0 -1 0",   -45,  45,  400,  35, 1.0),
        ("neck_rot",    f"0 0 {k}",  -70,  70,  400,  35, 1.0),
    ]


def all_joints(lead_sign: float = 1.0) -> List[JointSpec]:
    return _spine_joints(lead_sign) + _side_joints("l") + _side_joints("r")


JOINT_NAMES: List[str] = [j[0] for j in all_joints()]


# ---------------------------------------------------------------------------
# Address pose (degrees).  The pelvis itself leans forward by ADDRESS_LEAN; the
# hips are extended by the same amount so the thighs come back to vertical.
# ---------------------------------------------------------------------------

ADDRESS_LEAN = 32.0

ADDRESS: Dict[str, float] = {
    "lumbar_flex": 2, "lumbar_bend": -6, "lumbar_rot": 0,
    # tilted away from the target: this is what drops the trail shoulder below
    # the lead one, and it is the only reason the trail hand can reach a grip
    # that sits below the lead hand
    "thorax_flex": 2, "thorax_bend": -11, "thorax_rot": 0,
    "neck_flex": 20, "neck_rot": 0,
}
for _s in ("l", "r"):
    ADDRESS.update({
        # The pelvis is tilted forward by ADDRESS_LEAN, so ADDRESS_LEAN degrees
        # of hip flexion would put the thighs vertical.  The extra 14 deg sits
        # the hips back behind the ankles: bending over the ball without it
        # puts the centre of mass past the toes and the golfer slowly topples
        # forward over the course of the swing.
        f"hip_{_s}_flex": ADDRESS_LEAN + 14,
        f"hip_{_s}_abd": 6,
        f"hip_{_s}_rot": 0,
        f"knee_{_s}": 16,
        f"ankle_{_s}_dorsi": 2,
        f"ankle_{_s}_roll": 0,
        f"ankle_{_s}_rot": 0,
        f"shoulder_{_s}_abd": -6,
        # likewise the arms hang vertically from a torso that is tilted over
        f"shoulder_{_s}_flex": ADDRESS_LEAN - 2,
        f"shoulder_{_s}_rot": 0,
        f"elbow_{_s}_flex": 6,
        f"elbow_{_s}_pro": 20,
        f"wrist_{_s}_flex": 0,
        f"wrist_{_s}_dev": 8,
    })


# ===========================================================================
# 3.  Tracked landmarks
# ===========================================================================
# Site names of every major joint centre, plus the club and ball.  PARENT gives
# the kinematic chain used for the parent-relative segment vectors.

TRACKED: Tuple[str, ...] = (
    "pelvis", "lumbar", "thorax", "neck", "head",
    "hip_l", "knee_l", "ankle_l", "toe_l",
    "hip_r", "knee_r", "ankle_r", "toe_r",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "grip", "clubhead", "ball",
)

PARENT: Dict[str, Optional[str]] = {
    "pelvis": None,
    "lumbar": "pelvis", "thorax": "lumbar", "neck": "thorax", "head": "neck",
    "hip_l": "pelvis", "knee_l": "hip_l", "ankle_l": "knee_l", "toe_l": "ankle_l",
    "hip_r": "pelvis", "knee_r": "hip_r", "ankle_r": "knee_r", "toe_r": "ankle_r",
    "shoulder_l": "thorax", "elbow_l": "shoulder_l",
    "wrist_l": "elbow_l", "hand_l": "wrist_l",
    "shoulder_r": "thorax", "elbow_r": "shoulder_r",
    "wrist_r": "elbow_r", "hand_r": "wrist_r",
    "grip": None,          # filled in at build time (child of the lead hand)
    "clubhead": "grip",
    "ball": None,
}


# ===========================================================================
# 4.  Model builder
# ===========================================================================

def _q_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def _fmt(v: Sequence[float]) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def build_model_xml(
    anthro: Anthropometry,
    club: Club,
    base: str = "feet",
    timestep: float = 2e-4,
    face_offset_deg: float = 0.0,
    foot_pins: Optional[Dict[str, np.ndarray]] = None,
    strength: float = 1.0,
    trail_arm_gain: float = 0.12,
) -> str:
    """Emit the MuJoCo XML for golfer + club + ball + tee.

    base: "feet"   pelvis free, both feet pinned to the ground (default -- the
                   golfer cannot topple, which is what you want while the swing
                   itself is being studied)
          "free"   pelvis free, nothing pinned: full balance physics
          "pinned" pelvis welded to the world
    """
    a, c = anthro, club
    H = a.height
    lead, trail = a.lead, a.trail

    # ---- root placement ---------------------------------------------------
    # golfer faces -y (right handed) so that their left, and the target, is +x
    yaw = -90.0 * DEG * a.lead_sign
    q_yaw = _q_axis((0, 0, 1), yaw)
    # rotate about the pelvis' own left axis so the spine tips forward, over
    # the ball; the hips are extended by the same amount in ADDRESS so the
    # thighs come back to vertical
    q_lean = _q_axis((0, 1, 0), ADDRESS_LEAN * DEG)
    q_root = np.zeros(4)
    mujoco.mju_mulQuat(q_root, q_yaw, q_lean)
    root_pos = np.array([0.0, 0.62 * a.lead_sign, a.hip_h])

    # ---- trunk landmark heights above the hip joint centre ----------------
    z_lumbar = 0.07 * H
    z_thorax = 0.16 * H
    z_shoulder = a.trunk                    # 0.288 H
    z_neck = 0.28 * H
    z_headc = 0.36 * H

    def joint_xml(spec: JointSpec) -> str:
        name, axis, lo, hi, _kp, _fr, damp = spec
        return (f'<joint name="{name}" type="hinge" axis="{axis}" '
                f'range="{lo} {hi}" damping="{damp}" armature="0.02"/>')

    jmap = {j[0]: j for j in all_joints(a.lead_sign)}

    def joints(*names: str) -> str:
        return "\n        ".join(joint_xml(jmap[n]) for n in names)

    # ---- club geometry in the lead hand -----------------------------------
    g = c.grip_angle_deg * DEG
    palm = np.array([0.0, 0.0, -0.35 * a.hand_len])       # grip point in hand
    shaft_up = np.array([-math.sin(g), 0.0, math.cos(g)])  # club +z in hand frame
    club_pos = palm + c.lead_hand_drop * shaft_up
    q_club = _q_axis((0, -1, 0), g)

    loft = c.loft_deg * DEG
    face_normal = np.array([math.cos(loft), 0.0, math.sin(loft)])
    q_face = _q_axis((0, 0, 1), face_offset_deg * DEG)
    head_ctr = np.array([0.0, 0.045, -0.020])             # toe-ward of the tip
    face_ctr = head_ctr + np.array([0.020, 0.0, 0.0])

    # ---- legs -------------------------------------------------------------
    def leg(side: str) -> str:
        s = 1.0 if side == "l" else -1.0
        return f"""
      <body name="thigh_{side}" pos="0 {s * a.hip_sep / 2:.4f} 0">
        {joints(f"hip_{side}_flex", f"hip_{side}_abd", f"hip_{side}_rot")}
        <site name="hip_{side}" pos="0 0 0" class="landmark"/>
        <geom name="thigh_{side}_g" type="capsule" size="0.060"
              fromto="0 0 0 0 0 {-a.thigh:.4f}" mass="{a.m_thigh:.4f}"/>
        <body name="shank_{side}" pos="0 0 {-a.thigh:.4f}">
          {joints(f"knee_{side}")}
          <site name="knee_{side}" pos="0 0 0" class="landmark"/>
          <geom name="shank_{side}_g" type="capsule" size="0.046"
                fromto="0 0 0 0 0 {-a.shank:.4f}" mass="{a.m_shank:.4f}"/>
          <body name="foot_{side}" pos="0 0 {-a.shank:.4f}">
            {joints(f"ankle_{side}_dorsi", f"ankle_{side}_roll",
                    f"ankle_{side}_rot")}
            <site name="ankle_{side}" pos="0 0 0" class="landmark"/>
            <site name="toe_{side}" pos="{0.62 * a.foot_len:.4f} 0 {-a.ankle_h:.4f}"
                  class="landmark"/>
            <site name="heel_{side}" pos="{-0.32 * a.foot_len:.4f} 0 {-a.ankle_h:.4f}"
                  class="landmark"/>
            <geom name="foot_{side}_g" type="box" class="ground_contact"
                  size="{a.foot_len / 2:.4f} 0.045 {a.ankle_h / 2:.4f}"
                  pos="{0.15 * a.foot_len:.4f} 0 {-a.ankle_h / 2:.4f}"
                  mass="{a.m_foot:.4f}" friction="1.2 0.02 0.001"/>
          </body>
        </body>
      </body>"""

    # ---- arms -------------------------------------------------------------
    def arm(side: str) -> str:
        s = 1.0 if side == "l" else -1.0
        holds_club = (side == lead)
        club_body = f"""
          <body name="club" pos="{_fmt(club_pos)}" quat="{_fmt(q_club)}">
            <site name="grip" pos="0 0 -0.02" class="landmark"/>
            <site name="grip_lead" pos="0 0 {-c.lead_hand_drop:.4f}" class="grip"/>
            <site name="grip_trail" pos="0 0 {-c.trail_hand_drop:.4f}" class="grip"/>
            <geom name="grip_g" type="capsule" size="0.014"
                  fromto="0 0 0.01 0 0 -0.26" mass="{c.grip_mass:.4f}"
                  rgba="0.15 0.15 0.17 1"/>
            <geom name="shaft_g" type="capsule" size="0.0055"
                  fromto="0 0 -0.26 0 0 {-c.length:.4f}" mass="{c.shaft_mass:.4f}"
                  rgba="0.75 0.76 0.8 1"/>
            <body name="club_head" pos="0 0 {-c.length:.4f}" quat="{_fmt(q_face)}">
              <geom name="club_head_g" type="box" size="0.020 0.050 0.030"
                    pos="{_fmt(head_ctr)}" euler="0 {-c.loft_deg} 0"
                    mass="{c.head_mass:.4f}" class="strike"
                    rgba="0.2 0.2 0.24 1"/>
              <site name="clubhead" pos="{_fmt(head_ctr)}" class="landmark"/>
              <site name="clubface" pos="{_fmt(face_ctr)}"
                    zaxis="{_fmt(face_normal)}" class="grip"/>
            </body>
          </body>""" if holds_club else ""

        return f"""
        <body name="upperarm_{side}" pos="0 {s * a.shoulder_sep / 2:.4f} {z_shoulder - z_thorax - 0.02:.4f}">
          {joints(f"shoulder_{side}_abd", f"shoulder_{side}_flex", f"shoulder_{side}_rot")}
          <site name="shoulder_{side}" pos="0 0 0" class="landmark"/>
          <geom name="upperarm_{side}_g" type="capsule" size="0.044"
                fromto="0 0 0 0 0 {-a.upperarm:.4f}" mass="{a.m_upperarm:.4f}"/>
          <body name="forearm_{side}" pos="0 0 {-a.upperarm:.4f}">
            {joints(f"elbow_{side}_flex", f"elbow_{side}_pro")}
            <site name="elbow_{side}" pos="0 0 0" class="landmark"/>
            <geom name="forearm_{side}_g" type="capsule" size="0.038"
                  fromto="0 0 0 0 0 {-a.forearm:.4f}" mass="{a.m_forearm:.4f}"/>
            <body name="hand_{side}" pos="0 0 {-a.forearm:.4f}">
              {joints(f"wrist_{side}_flex", f"wrist_{side}_dev")}
              <site name="wrist_{side}" pos="0 0 0" class="landmark"/>
              <site name="hand_{side}" pos="{_fmt(palm)}" class="landmark"/>
              <site name="palm_{side}" pos="{_fmt(palm)}" class="grip"/>
              <geom name="hand_{side}_g" type="capsule" size="0.035"
                    fromto="0 0 -0.01 0 0 {-0.62 * a.hand_len:.4f}"
                    mass="{a.m_hand:.4f}"/>{club_body}
            </body>
          </body>
        </body>"""

    # ---- actuators --------------------------------------------------------
    # Both hands are on one shaft, so the arms form a closed loop.  If both are
    # driven stiffly they fight each other through it and can lock the club
    # solid, so the trail arm is left compliant and is carried by the grip
    # constraint -- the lead arm and the body do the work.
    acts = []
    trail_arm = tuple(f"{p}_{trail}" for p in ("shoulder", "elbow", "wrist"))
    for name, _axis, lo, hi, kp, frc, _d in all_joints(a.lead_sign):
        soft = trail_arm_gain if name.startswith(trail_arm) else 1.0
        acts.append(
            f'<position name="act_{name}" joint="{name}" kp="{kp * soft:.1f}" '
            f'dampratio="1" ctrlrange="{lo * DEG:.4f} {hi * DEG:.4f}" '
            f'forcerange="{-frc * strength * max(soft, 0.5):.1f} '
            f'{frc * strength * max(soft, 0.5):.1f}"/>')

    # ---- equality constraints --------------------------------------------
    eqs = [f'<connect name="trail_grip" site1="palm_{trail}" site2="grip_trail" '
           f'solref="0.004 1" solimp="0.94 0.995 0.01"/>']
    pin_sites = []
    if base == "feet" and foot_pins:
        for key, pos in foot_pins.items():
            pin_sites.append(f'<site name="pin_{key}" pos="{_fmt(pos)}" '
                             f'class="pin"/>')
            eqs.append(f'<connect name="pin_{key}" site1="{key}" '
                       f'site2="pin_{key}" solref="0.004 1" '
                       f'solimp="0.99 0.9999 0.001"/>')

    root_joint = '<freejoint name="root"/>' if base != "pinned" else ""

    return f"""
<mujoco model="rl_golf">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="80" tolerance="1e-10"
          cone="elliptic" impratio="5"/>
  <size njmax="600" nconmax="200"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <!-- body segments do not self-collide: keeps the sim fast and the arms
         are never meant to swing through the torso anyway -->
    <geom contype="0" conaffinity="0" rgba="0.75 0.68 0.6 1"
          solref="0.004 1" friction="0.8 0.01 0.001"/>
    <site size="0.012" rgba="0.9 0.3 0.2 1" group="3"/>
    <default class="landmark">
      <site size="0.016" rgba="0.95 0.35 0.15 0.9" group="3"/>
    </default>
    <default class="grip">
      <site size="0.010" rgba="0.2 0.7 0.95 0.6" group="4"/>
    </default>
    <default class="pin">
      <site size="0.012" rgba="0.3 0.9 0.4 0.5" group="4"/>
    </default>
    <default class="ground_contact">
      <geom contype="1" conaffinity="1"/>
    </default>
    <default class="strike">
      <!-- the clubhead collides with the ball only.  Turf interaction is a
           whole model of its own, and without this a club that bottoms out a
           centimetre low stops dead in the ground instead of taking a divot -->
      <geom contype="8" conaffinity="2" solref="0.0006 1"
            solimp="0.95 0.99 0.001" friction="0.4 0.01 0.001"/>
    </default>
  </default>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.5 0.68 0.86" rgb2="0.85 0.9 0.95" width="256" height="256"/>
    <texture name="turf" type="2d" builtin="checker" rgb1="0.22 0.4 0.2"
             rgb2="0.19 0.35 0.17" width="512" height="512"/>
    <material name="turf" texture="turf" texrepeat="20 20" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="sun" pos="1 -2 4" dir="-0.2 0.4 -1" directional="true"
           diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <geom name="ground" type="plane" size="40 40 0.1" material="turf"
          contype="1" conaffinity="3" friction="1.2 0.02 0.001"/>
    {"".join(pin_sites)}

    <body name="tee" pos="0 0 0">
      <geom name="tee_g" type="cylinder" size="0.006 {TEE_HEIGHT / 2:.4f}"
            pos="0 0 {TEE_HEIGHT / 2:.4f}" mass="0.003"
            contype="4" conaffinity="2" rgba="0.95 0.95 0.5 1"/>
    </body>

    <body name="ball" pos="0 0 {TEE_HEIGHT + BALL_RADIUS:.4f}">
      <freejoint name="ball_free"/>
      <site name="ball" pos="0 0 0" class="landmark"/>
      <geom name="ball_g" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}"
            contype="2" conaffinity="7" rgba="0.97 0.97 0.97 1"
            solref="0.0006 1" solimp="0.95 0.99 0.001"
            friction="0.35 0.005 0.0001"/>
    </body>

    <body name="pelvis" pos="{_fmt(root_pos)}" quat="{_fmt(q_root)}">
      {root_joint}
      <site name="pelvis" pos="0 0 0" class="landmark"/>
      <geom name="pelvis_g" type="capsule" size="0.085"
            fromto="0 {-a.hip_sep / 2:.4f} 0 0 {a.hip_sep / 2:.4f} 0"
            mass="{a.m_pelvis:.4f}"/>
      {leg("l")}
      {leg("r")}

      <body name="abdomen" pos="0 0 {z_lumbar:.4f}">
        {joints("lumbar_flex", "lumbar_bend", "lumbar_rot")}
        <site name="lumbar" pos="0 0 0" class="landmark"/>
        <geom name="abdomen_g" type="capsule" size="0.092"
              fromto="0 0 0 0 0 {z_thorax - z_lumbar:.4f}"
              mass="{a.m_abdomen:.4f}"/>

        <body name="thorax" pos="0 0 {z_thorax - z_lumbar:.4f}">
          {joints("thorax_flex", "thorax_bend", "thorax_rot")}
          <site name="thorax" pos="0 0 0" class="landmark"/>
          <geom name="thorax_g" type="capsule" size="0.105"
                fromto="0 0 0.01 0 0 {z_neck - z_thorax:.4f}"
                mass="{a.m_thorax:.4f}"/>

          <body name="head" pos="0 0 {z_neck - z_thorax:.4f}">
            {joints("neck_flex", "neck_rot")}
            <site name="neck" pos="0 0 0" class="landmark"/>
            <site name="head" pos="0 0 {z_headc - z_neck:.4f}" class="landmark"/>
            <geom name="neck_g" type="capsule" size="0.045"
                  fromto="0 0 0 0 0 {0.4 * (z_headc - z_neck):.4f}" mass="1.0"/>
            <geom name="head_g" type="sphere" size="0.098"
                  pos="0 0 {z_headc - z_neck:.4f}"
                  mass="{a.m_head - 1.0:.4f}"/>
          </body>
{arm("l")}
{arm("r")}
        </body>
      </body>
    </body>
  </worldbody>

  <equality>
    {"".join(eqs)}
  </equality>

  <actuator>
    {"".join(acts)}
  </actuator>
</mujoco>
"""


# ===========================================================================
# 5.  Kinematic tracking
# ===========================================================================

class SwingTracker:
    """Everything the RL agent needs to see about where the body is.

    All of the `*_positions` methods return an (N, 3) array whose rows follow
    `TRACKED`, so `tracker.index["wrist_l"]` indexes any of them.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 anthro: Anthropometry):
        self.model, self.data, self.anthro = model, data, anthro
        self.names: Tuple[str, ...] = TRACKED
        self.index = {n: i for i, n in enumerate(TRACKED)}
        self.sid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                    for n in TRACKED}
        missing = [n for n, i in self.sid.items() if i < 0]
        if missing:
            raise RuntimeError(f"model is missing sites: {missing}")
        self.jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in JOINT_NAMES}
        self.qadr = {n: model.jnt_qposadr[i] for n, i in self.jid.items()}
        self.dadr = {n: model.jnt_dofadr[i] for n, i in self.jid.items()}
        self.pelvis_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lead = anthro.lead
        self.trail = anthro.trail
        # target direction in world coordinates
        self.target = np.array([1.0, 0.0, 0.0])

    # ---- raw ---------------------------------------------------------------
    def positions(self) -> np.ndarray:
        """World-frame position of every tracked landmark, shape (N, 3)."""
        return np.array([self.data.site_xpos[self.sid[n]] for n in TRACKED])

    def pelvis_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """(origin, rotation) of the golfer's pelvis in the world."""
        o = self.data.site_xpos[self.sid["pelvis"]].copy()
        R = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
        return o, R

    # ---- relative ----------------------------------------------------------
    def heading_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Origin at the pelvis, x = the direction the golfer faces flattened
        into the horizontal plane, z = world up.  Yaw-only, so heights stay
        heights however far the golfer is bent over."""
        o, R = self.pelvis_frame()
        fwd = R[:, 0].copy()
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        if n < 1e-6:                       # bent double: fall back to the spine
            fwd = -R[:, 2].copy()
            fwd[2] = 0.0
            n = max(np.linalg.norm(fwd), 1e-9)
        fwd /= n
        left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
        return o, np.column_stack([fwd, left, np.array([0.0, 0.0, 1.0])])

    def egocentric(self, gravity_aligned: bool = True) -> np.ndarray:
        """Landmarks relative to the golfer: origin at the pelvis, axes along
        forward / left / up.  Invariant to where on the range the golfer stands
        and which way they aim, which is what an RL policy wants to see."""
        o, R = (self.heading_frame() if gravity_aligned else self.pelvis_frame())
        return (self.positions() - o) @ R

    def parent_relative(self) -> np.ndarray:
        """Each landmark minus its parent in the kinematic chain -- i.e. the
        segment vectors (world frame).  Chain roots give a zero row."""
        p = self.positions()
        out = np.zeros_like(p)
        for i, n in enumerate(TRACKED):
            par = PARENT[n]
            if par is not None:
                out[i] = p[i] - p[self.index[par]]
        return out

    def pairwise_offsets(self) -> np.ndarray:
        """(N, N, 3): row i, col j = position(j) - position(i)."""
        p = self.positions()
        return p[None, :, :] - p[:, None, :]

    def pairwise_distances(self) -> np.ndarray:
        """(N, N) euclidean distances between every pair of landmarks."""
        return np.linalg.norm(self.pairwise_offsets(), axis=-1)

    # ---- velocities --------------------------------------------------------
    def site_velocity(self, name: str) -> np.ndarray:
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.sid[name], res, 0)
        return res[3:6].copy()          # linear part, world frame

    def clubhead_speed(self) -> float:
        return float(np.linalg.norm(self.site_velocity("clubhead")))

    def ball_speed(self) -> float:
        return float(np.linalg.norm(self.site_velocity("ball")))

    # ---- joint angles ------------------------------------------------------
    def joint_angles(self, degrees: bool = True) -> Dict[str, float]:
        f = RAD if degrees else 1.0
        return {n: float(self.data.qpos[self.qadr[n]]) * f for n in JOINT_NAMES}

    def joint_velocities(self, degrees: bool = True) -> Dict[str, float]:
        f = RAD if degrees else 1.0
        return {n: float(self.data.qvel[self.dadr[n]]) * f for n in JOINT_NAMES}

    # ---- golf-specific -----------------------------------------------------
    @staticmethod
    def _heading(v: np.ndarray) -> float:
        """Angle of a vector's horizontal projection, degrees, +ve toward +y."""
        return math.degrees(math.atan2(v[1], v[0]))

    def _turn(self, right_site: str, left_site: str) -> float:
        """Rotation of a body segment about the vertical, 0 = square to the
        target line, negative = turned away from the target (backswing)."""
        p = self.positions()
        v = p[self.index[left_site]] - p[self.index[right_site]]
        lead_is_left = self.anthro.lead_sign > 0
        # the line from trail to lead landmark points down the target line at 0
        h = self._heading(v if lead_is_left else -v)
        return ((h + 180.0) % 360.0) - 180.0

    def metrics(self) -> Dict[str, float]:
        p = self.positions()
        idx = self.index
        lead, trail = self.lead, self.trail

        hip_turn = self._turn(f"hip_{trail}", f"hip_{lead}")
        sh_turn = self._turn(f"shoulder_{trail}", f"shoulder_{lead}")

        spine = p[idx["neck"]] - p[idx["pelvis"]]
        spine_tilt = math.degrees(math.acos(
            np.clip(spine[2] / max(np.linalg.norm(spine), 1e-9), -1, 1)))
        # forward/side split of that tilt, in the golfer's own frame
        _, R = self.heading_frame()
        spine_local = spine @ R
        forward_tilt = math.degrees(math.atan2(spine_local[0], spine_local[2]))
        side_tilt = math.degrees(math.atan2(spine_local[1], spine_local[2]))

        arm = p[idx[f"wrist_{lead}"]] - p[idx[f"shoulder_{lead}"]]
        shaft = p[idx["clubhead"]] - p[idx["grip"]]
        cock = math.degrees(math.acos(np.clip(
            np.dot(arm, shaft) / max(np.linalg.norm(arm) * np.linalg.norm(shaft),
                                     1e-9), -1, 1)))

        chv = self.site_velocity("clubhead")
        speed = float(np.linalg.norm(chv))
        attack = math.degrees(math.atan2(chv[2], max(np.linalg.norm(chv[:2]),
                                                     1e-9)))
        path = math.degrees(math.atan2(chv[1], chv[0])) if speed > 0.5 else 0.0

        return {
            "t": float(self.data.time),
            "hip_turn": hip_turn,
            "shoulder_turn": sh_turn,
            "x_factor": sh_turn - hip_turn,
            "spine_tilt": spine_tilt,
            "forward_tilt": forward_tilt,
            "side_tilt": side_tilt,
            "lead_wrist_cock": cock,
            "clubhead_speed": speed,
            "clubhead_height": float(p[idx["clubhead"]][2]),
            "hand_speed": float(np.linalg.norm(
                self.site_velocity(f"hand_{lead}"))),
            "attack_angle": attack,
            "club_path": path,
            "ball_speed": self.ball_speed(),
        }

    # ---- RL observation ----------------------------------------------------
    def observation(self) -> np.ndarray:
        """Flat observation vector: joint angles + velocities, every landmark
        in the egocentric frame, the segment vectors, and club/ball state."""
        q = np.array([self.data.qpos[self.qadr[n]] for n in JOINT_NAMES])
        qd = np.array([self.data.qvel[self.dadr[n]] for n in JOINT_NAMES])
        ego = self.egocentric().ravel()
        seg = self.parent_relative().ravel()
        _, R = self.heading_frame()
        club_v = self.site_velocity("clubhead") @ R
        ball_v = self.site_velocity("ball") @ R
        return np.concatenate([q, qd, ego, seg, club_v, ball_v])


# ===========================================================================
# 6.  Scripted swing
# ===========================================================================

def _pose(base: Dict[str, float], lead_sign: float,
          lumbar_turn: float = 0.0, thorax_turn: float = 0.0,
          **overrides: float) -> Dict[str, float]:
    """Build a keyframe as deltas from the settled address pose.

    Turns are joint-relative, so the shoulder turn seen from outside is
    pelvis + lumbar + thorax.  The pelvis turn itself is not set here: it is a
    whole-body rotation and gets converted into leg angles by
    GolfSwingSim._plan_legs.
    """
    p = dict(base)
    p["lumbar_rot"] = base["lumbar_rot"] + lumbar_turn
    p["thorax_rot"] = base["thorax_rot"] + thorax_turn
    p.update(overrides)
    return p


def swing_script(anthro: Anthropometry, base: Dict[str, float],
                 tempo: float = 1.0
                 ) -> List[Tuple[str, float, Dict[str, float], float]]:
    """(phase name, time in seconds, joint targets in degrees, pelvis turn).

    Timings follow a real swing: ~0.8 s to the top and ~0.3 s down, with the
    pelvis already opening while the arms are still finishing the backswing --
    the kinematic sequence that produces the X-factor stretch.  Only the lead
    arm is scripted; the trail arm is solved afterwards so both hands stay on
    the club (see GolfSwingSim._plan_trail_arm).
    """
    L = anthro.lead
    k = anthro.lead_sign

    def arm(flex, abd, rot, elbow, dev, wflex=None):
        d = {f"shoulder_{L}_flex": flex, f"shoulder_{L}_abd": abd,
             f"shoulder_{L}_rot": rot, f"elbow_{L}_flex": elbow,
             f"wrist_{L}_dev": dev}
        if wflex is not None:
            d[f"wrist_{L}_flex"] = wflex
        return d

    s_flex = base[f"shoulder_{L}_flex"]
    s_abd = base[f"shoulder_{L}_abd"]
    e_flex = base[f"elbow_{L}_flex"]
    w_dev = base[f"wrist_{L}_dev"]

    def P(**kw):
        return _pose(base, k, **kw)

    phases = [
        ("address", 0.00, P(**arm(s_flex, s_abd, 0, e_flex, w_dev)), 0),
        ("takeaway", 0.32, P(
            lumbar_turn=-4, thorax_turn=-22, thorax_bend=-6,
            **arm(s_flex + 4, s_abd - 12, 10, e_flex + 4, w_dev + 12)), -10),
        ("top", 0.82, P(
            lumbar_turn=-12, thorax_turn=-46,
            thorax_bend=-2, lumbar_bend=0, neck_rot=-25,
            **arm(s_flex + 34, s_abd - 34, 40, e_flex + 12, 52)), -42),
        # the pelvis has already started back toward the target while the
        # shoulders are still at the top: the kinematic sequence
        ("transition", 0.90, P(
            lumbar_turn=-13, thorax_turn=-48,
            thorax_bend=-10, neck_rot=-22,
            **arm(s_flex + 30, s_abd - 32, 38, e_flex + 12, 52)), -30),
        # Turns are joint-relative, so the shoulder turn is the sum of all
        # three: at impact the hips are 40 deg open but the chest only ~15,
        # which is the negative X-factor a real golfer has at impact.  The lead
        # arm comes back to its address angles, so the club is extended out at
        # the ball again and the body rotation is what delivers it.
        ("impact", 1.14, P(
            lumbar_turn=-5, thorax_turn=-20,
            thorax_bend=-13, lumbar_bend=-9, neck_rot=-10,
            **arm(s_flex, s_abd, 0, e_flex, w_dev - 4)), 42),
        # keeps driving through the bottom of the arc instead of lifting away
        ("through", 1.28, P(
            lumbar_turn=0, thorax_turn=-4,
            thorax_bend=-12, lumbar_bend=-8, neck_rot=-4,
            **arm(s_flex + 2, s_abd + 6, -20, e_flex, w_dev - 10)), 60),
        ("extension", 1.46, P(
            lumbar_turn=6, thorax_turn=6,
            thorax_bend=-10, neck_rot=8,
            **arm(s_flex + 6, s_abd + 10, -45, e_flex + 6, w_dev - 22)), 74),
        ("finish", 1.95, P(
            lumbar_turn=12, thorax_turn=35,
            thorax_bend=8, thorax_flex=-10, neck_rot=25,
            **arm(s_flex + 22, s_abd + 16, -25, 35, w_dev - 10)), 76),
    ]
    return [(n, t * tempo, p, turn * anthro.lead_sign)
            for n, t, p, turn in phases]


def _hermite(u: float) -> Tuple[float, float, float, float]:
    """Cubic Hermite basis (position form) at u in [0, 1]."""
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1,      # p0
            u3 - 2 * u2 + u,          # m0
            -2 * u3 + 3 * u2,         # p1
            u3 - u2)                  # m1


class SwingController:
    """Interpolates the scripted keyframes and writes position-servo targets."""

    def __init__(self, phases: List[Tuple[str, float, Dict[str, float]]],
                 model: mujoco.MjModel, lead: float = 0.0):
        self.phases = phases
        self.act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         f"act_{n}") for n in JOINT_NAMES}
        self.duration = phases[-1][1]
        # torque-limited servos run behind their reference, which at swing
        # speed puts the club ~0.4 m above the ball when the script says
        # impact.  Feeding them the reference this far ahead of the clock
        # cancels most of that phase lag.
        self.lead = lead

    def phase_at(self, t: float) -> str:
        name = self.phases[0][0]
        for entry in self.phases:
            if t >= entry[1]:
                name = entry[0]
        return name

    def targets(self, t: float) -> Dict[str, float]:
        """Catmull-Rom through the keyframes.

        Not a smoothstep between neighbours: that has zero slope at every
        keyframe, so the reference comes to a dead stop at impact and the
        clubhead arrives at the ball doing 3 m/s.  Catmull-Rom tangents let the
        swing pass *through* its keyframes at speed.
        """
        ph = self.phases
        n = len(ph)
        if t <= ph[0][1]:
            return dict(ph[0][2])
        if t >= ph[-1][1]:
            return dict(ph[-1][2])
        i = 0
        for j in range(n - 1):
            if ph[j][1] <= t <= ph[j + 1][1]:
                i = j
                break
        t0, t1 = ph[i][1], ph[i + 1][1]
        h = max(t1 - t0, 1e-9)
        b0, bm0, b1, bm1 = _hermite((t - t0) / h)
        p0, p1 = ph[i][2], ph[i + 1][2]
        prev = ph[i - 1] if i > 0 else None
        nxt = ph[i + 2] if i + 2 < n else None
        out = {}
        for k in p0:
            m0 = 0.0 if prev is None else \
                (p1[k] - prev[2][k]) / (t1 - prev[1]) * h
            m1 = 0.0 if nxt is None else \
                (nxt[2][k] - p0[k]) / (nxt[1] - t0) * h
            out[k] = b0 * p0[k] + bm0 * m0 + b1 * p1[k] + bm1 * m1
        return out

    def apply(self, data: mujoco.MjData, t: float) -> None:
        for name, val in self.targets(t + self.lead).items():
            data.ctrl[self.act[name]] = val * DEG


# ===========================================================================
# 7.  Simulation
# ===========================================================================

class GolfSwingSim:
    """Builds the model, finds a physically consistent address position, and
    runs the scripted swing."""

    def __init__(self, anthro: Optional[Anthropometry] = None,
                 club: Optional[Club] = None, base: str = "feet",
                 timestep: float = 2e-4, calibrate: bool = True,
                 strength: float = 1.0, tempo: float = 1.0,
                 lead: float = 0.07, verbose: bool = True):
        self.anthro = anthro or Anthropometry()
        self.club = club or Club()
        self.base = base
        self.timestep = timestep
        self.verbose = verbose
        self._strength = strength

        # ---- pass 1: provisional model, used only to measure ---------------
        self._load(build_model_xml(self.anthro, self.club, base, timestep,
                                   strength=strength))
        self._solve_address()
        pins = self._foot_pin_targets() if base == "feet" else None
        self._pins = pins
        face_err = self._face_heading_error()

        # ---- pass 2: face squared to the target, feet pinned ---------------
        self._load(build_model_xml(self.anthro, self.club, base, timestep,
                                   face_offset_deg=-face_err, foot_pins=pins,
                                   strength=strength))
        self._solve_address()
        # the servo targets are the *solved* angles and the stored state is the
        # pose those targets settle into -- so the swing starts from a genuine
        # equilibrium instead of twitching on the first step
        self.address_angles = self.tracker.joint_angles()
        self._settle(0.5)
        # the servos sag a few degrees under load, which drops the clubhead
        # below the ball.  Ask for a little more wrist until the club actually
        # soles where it should once everything has settled.
        key = f"wrist_{self.anthro.lead}_dev"
        for _ in range(4):
            err = float(self.tracker.positions()[
                self.tracker.index["clubhead"]][2]) - (TEE_HEIGHT + BALL_RADIUS)
            if abs(err) < 0.008:
                break
            self.address_angles[key] = self._clamp_deg(
                key, self.address_angles[key] - err / 0.018)
            self._settle(0.25)
        self.qpos_address = self.data.qpos.copy()
        self.address_clubhead = self.tracker.positions()[
            self.tracker.index["clubhead"]].copy()

        script = swing_script(self.anthro, self.address_angles, tempo)
        self._plan_legs(script)
        self._plan_lead_arm(script)
        self._plan_trail_arm(script)
        self.controller = SwingController(script, self.model,
                                          lead=lead * tempo)
        self.ball_pos = self.tracker.positions()[self.tracker.index["ball"]].copy()
        if calibrate:
            self._tune_release(script)
            self._calibrate_ball()
        self.reset()

    # ---- construction ------------------------------------------------------
    def _load(self, xml: str) -> None:
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.tracker = SwingTracker(self.model, self.data, self.anthro)
        self.ball_jadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        # the ball's freejoint is declared before the golfer's, so the root is
        # *not* at qpos[0:7] -- always look it up
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = int(self.model.jnt_qposadr[rid]) if rid >= 0 else -1
        self.root_dadr = int(self.model.jnt_dofadr[rid]) if rid >= 0 else -1
        self.tee_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "tee")
        self._ball_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "ball_g")
        self._head_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "club_head_g")
        PARENT["grip"] = f"hand_{self.anthro.lead}"

    def _set_pose(self, angles: Dict[str, float]) -> None:
        for n, v in angles.items():
            self.data.qpos[self.tracker.qadr[n]] = v * DEG

    def _clamp(self, name: str, v: float) -> float:
        jid = self.tracker.jid[name]
        lo, hi = self.model.jnt_range[jid]
        return min(max(v, lo), hi)

    def _ik(self, site: str, target: np.ndarray, joints: Sequence[str],
            mask: Sequence[float] = (1, 1, 1), iters: int = 200,
            step: float = 0.5, damping: float = 1e-3,
            rest_pull: float = 0.25, max_step: float = 0.12) -> float:
        """Damped least-squares IK on a subset of joints, respecting limits.

        The null-space term pulls the unused freedom back toward the pose the
        solver started from.  Without it a 7-DOF arm happily reaches the grip
        with the shoulder abducted 170 deg -- geometrically valid, anatomically
        absurd, and it flings the club as soon as the next keyframe unwinds it.
        """
        return self._ik_multi([(site, target)], joints, mask, iters, step,
                              damping, rest_pull, max_step)

    def _ik_multi(self, goals: Sequence[Tuple[str, np.ndarray]],
                  joints: Sequence[str], mask: Sequence[float] = (1, 1, 1),
                  iters: int = 200, step: float = 0.5, damping: float = 1e-3,
                  rest_pull: float = 0.25, max_step: float = 0.12,
                  free_root: bool = False) -> float:
        """As `_ik`, but drives several sites to several targets at once.

        `free_root` adds the root body's three translation DOFs to the solve,
        which is what lets the leg planner hold both feet on their pins while
        the pelvis is rotated to a prescribed angle.
        """
        sids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                for s, _ in goals]
        dofs = [self.tracker.dadr[j] for j in joints]
        adrs = [self.tracker.qadr[j] for j in joints]
        if free_root:
            r, d = self.root_qadr, self.root_dadr
            dofs = [d, d + 1, d + 2] + dofs
            adrs = [r, r + 1, r + 2] + adrs
        nj = len(adrs)
        w = np.tile(np.asarray(mask, dtype=float), len(goals))
        rest = self.data.qpos[adrs].copy()
        jacp = np.zeros((3, self.model.nv))
        eye = np.eye(nj)
        m = len(w)
        err = np.zeros(m)
        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            err = np.concatenate([t - self.data.site_xpos[s]
                                  for s, (_n, t) in zip(sids, goals)]) * w
            if np.linalg.norm(err) < 1e-5:
                break
            rows = []
            for s in sids:
                mujoco.mj_jacSite(self.model, self.data, jacp, None, s)
                rows.append(jacp[:, dofs])
            J = np.vstack(rows) * w[:, None]
            Jinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(m))
            q = self.data.qpos[adrs]
            dq = Jinv @ err + (eye - Jinv @ J) @ (rest_pull * (rest - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            names = ([None] * 3 + list(joints)) if free_root else list(joints)
            for j, adr, d in zip(names, adrs, dq):
                v = self.data.qpos[adr] + step * d
                self.data.qpos[adr] = v if j is None else self._clamp(j, v)
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(err))

    def _solve_address(self) -> None:
        """Put the golfer in a self-consistent address position.

        1. apply the nominal address angles
        2. drop the model until the feet touch the ground
        3. square the shoulders to the target line
        4. adjust the lead wrist/forearm until the club soles at the ball
        5. IK the trail arm onto the grip so both hands are on the club
        """
        mujoco.mj_resetData(self.model, self.data)
        self._set_pose(ADDRESS)
        mujoco.mj_forward(self.model, self.data)

        if self.base != "pinned":
            lo = min(
                self.data.geom_xpos[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_g")][2]
                - self.anthro.ankle_h / 2 for s in ("l", "r"))
            self.data.qpos[self.root_qadr + 2] += -lo + 0.001
            mujoco.mj_forward(self.model, self.data)

        L = self.anthro.lead
        # 3. side-bending a torso that is already tilted forward also yaws it,
        # which leaves the shoulders closed by ~10 deg.  Take that back out
        # with thorax rotation so the golfer actually aims at the target.
        for _ in range(12):
            err = self.tracker.metrics()["shoulder_turn"]
            if abs(err) < 0.2:
                break
            adr = self.tracker.qadr["thorax_rot"]
            self.data.qpos[adr] = self._clamp("thorax_rot",
                                              self.data.qpos[adr] - err * DEG)
            mujoco.mj_forward(self.model, self.data)

        # 4+5. sole the club, then check the trail hand can actually get on the
        # grip -- if it cannot, pull the hands in toward the middle of the
        # stance (which is what a golfer does) and try again
        for adduct in range(0, 26, 2):
            self.data.qpos[self.tracker.qadr[f"shoulder_{L}_abd"]] = \
                (ADDRESS[f"shoulder_{L}_abd"] - adduct) * DEG
            head = self.tracker.positions()[self.tracker.index["clubhead"]]
            # sole the club at ball height, a little forward of centre in the
            # stance; how far the ball ends up from the golfer is left free,
            # it is whatever the arms and club reach
            target = np.array([BALL_FORWARD, head[1], TEE_HEIGHT + BALL_RADIUS])
            self._ik("clubhead", target,
                     [f"wrist_{L}_dev", f"wrist_{L}_flex", f"elbow_{L}_pro"],
                     mask=(1, 0, 1))
            res = self._solve_trail_arm()
            if res < 5e-3:
                break
        if self.verbose and res > 5e-3:
            print(f"  [warn] trail hand is {res * 100:.1f} cm off the grip")

    def _solve_trail_arm(self) -> float:
        """Put the trail hand on the club.  Both hands on one shaft is a closed
        kinematic loop, so the trail arm is never scripted -- it is solved here
        and held together during the swing by the `trail_grip` equality."""
        T = self.anthro.trail
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                "grip_trail")
        joints = [f"shoulder_{T}_abd", f"shoulder_{T}_flex",
                  f"shoulder_{T}_rot", f"elbow_{T}_flex", f"elbow_{T}_pro",
                  f"wrist_{T}_flex", f"wrist_{T}_dev"]
        return self._ik(f"palm_{T}", self.data.site_xpos[sid].copy(), joints)

    def _apply_root(self, phase: str) -> None:
        """Put the free root where `_plan_legs` decided it should be for this
        keyframe, so the arm solvers see the same pelvis the legs produced."""
        q = getattr(self, "_root_plan", {}).get(phase)
        if q is not None:
            self.data.qpos[self.root_qadr:self.root_qadr + 7] = q

    def _plan_legs(self, script) -> None:
        """Turn each keyframe's pelvis rotation into leg angles.

        A golfer with both feet planted turns the pelvis by twisting the whole
        lower body; the hips, knees and ankles all have to move together and no
        single joint 'is' the turn.  Scripting hip rotation directly does not
        work -- the abductors and knees hold their address angles and simply
        block it, and the pelvis ends up turning 5 deg instead of 40.  So the
        pelvis is rotated to where it should be and the legs are solved to keep
        the feet on their pins.
        """
        if self.base == "pinned":
            return
        self._root_plan: Dict[str, np.ndarray] = {}
        live = self.data
        self.data = mujoco.MjData(self.model)
        self.tracker.data = self.data
        legs = [j for j in JOINT_NAMES
                if j.startswith(("hip_", "knee_", "ankle_"))]
        pins = dict(self._pins or {})
        r = self.root_qadr
        q_addr = self.qpos_address[r + 3:r + 7].copy()
        try:
            for name, _t, pose, turn in script:
                self.data.qpos[:] = self.qpos_address
                self._set_pose(pose)
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, _q_axis((0, 0, 1), turn * DEG), q_addr)
                self.data.qpos[r + 3:r + 7] = q
                mujoco.mj_forward(self.model, self.data)
                if pins:
                    res = self._ik_multi(list(pins.items()), legs,
                                         rest_pull=0.15, free_root=True)
                    if self.verbose and res > 0.02:
                        print(f"  [warn] '{name}': feet {res * 100:.1f} cm off "
                              f"their pins at {turn:+.0f} deg of pelvis turn")
                for j in legs:
                    pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD
                # the arm planners need the same pelvis pose this turn implies
                self._root_plan[name] = self.data.qpos[r:r + 7].copy()
        finally:
            self.data = live
            self.tracker.data = live

    def _plan_lead_arm(self, script, dz: float = 0.0) -> None:
        """Solve the lead arm at impact and just past it so the clubhead really
        does arrive back at the ball.

        Scripting the arm by hand does not work: the torso is 15-40 deg open
        through impact and carries the club around with it, so the arm has to
        be correspondingly further across the chest.  Asking for the position
        and letting IK find the angles is both easier and more honest.
        """
        live = self.data
        self.data = mujoco.MjData(self.model)
        self.tracker.data = self.data
        L = self.anthro.lead
        # the wrist is deliberately *not* solved for: given the freedom it
        # keeps 80 deg of lag angle and delivers the clubhead to the ball with
        # the club still thrown across the arm, which is not a golf swing.  The
        # keyframes set the release and the shoulder/elbow do the aiming.
        joints = [f"shoulder_{L}_abd", f"shoulder_{L}_flex", f"shoulder_{L}_rot",
                  f"elbow_{L}_flex"]
        ball = self.address_clubhead + np.array([0.0, 0.0, dz])
        aim = np.array([1.0, 0.0, 0.0])         # target line
        # Only the follow-through is solved by position.  Aiming the *impact*
        # pose at the ball as well looks right on paper but the solver gets
        # there by pulling the hands out over the ball until the shaft hangs
        # straight down -- which puts the clubhead almost on the axis the body
        # is rotating about, so it arrives at the ball doing 1 m/s.  Impact
        # instead reuses the address arm geometry, where the club is properly
        # extended out to the ball, and the body turn swings it through.
        goals = {"extension": ball + 0.85 * aim + np.array([0.0, 0.0, 0.60])}
        try:
            for name, _t, pose, _turn in script:
                if name not in goals:
                    continue
                self.data.qpos[:] = self.qpos_address
                self._set_pose(pose)
                self._apply_root(name)
                mujoco.mj_forward(self.model, self.data)
                res = self._ik("clubhead", goals[name], joints, rest_pull=0.05)
                if self.verbose and res > 0.02:
                    print(f"  [warn] '{name}': clubhead {res * 100:.1f} cm off "
                          f"its target")
                for j in joints:
                    pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD
        finally:
            self.data = live
            self.tracker.data = live

    def _plan_trail_arm(self, script, warn: bool = True) -> None:
        """Fill in the trail-arm angles of every keyframe by IK, on a scratch
        copy of the state so the live simulation is untouched.

        Each keyframe is warm-started from the previous one.  Solving them
        independently lets the IK hop between elbow-up/elbow-down branches, and
        interpolating across such a hop swings the trail arm through a wild arc
        that flings the club -- the trail arm is strong enough to throw the
        whole swing off.
        """
        live = self.data
        self.data = mujoco.MjData(self.model)
        self.tracker.data = self.data
        T = self.anthro.trail
        trail = [f"shoulder_{T}_abd", f"shoulder_{T}_flex", f"shoulder_{T}_rot",
                 f"elbow_{T}_flex", f"elbow_{T}_pro", f"wrist_{T}_flex",
                 f"wrist_{T}_dev"]
        try:
            self.data.qpos[:] = self.qpos_address
            for name, _t, pose, _turn in script:
                prev = {j: float(self.data.qpos[self.tracker.qadr[j]]) * RAD
                        for j in trail}
                self.data.qpos[:] = self.qpos_address
                self._set_pose({**pose, **prev})
                self._apply_root(name)
                mujoco.mj_forward(self.model, self.data)
                res = self._solve_trail_arm()
                if warn and self.verbose and res > 0.02:
                    print(f"  [warn] '{name}': trail hand {res * 100:.1f} cm "
                          f"off the grip")
                for j in trail:
                    pose[j] = float(self.data.qpos[self.tracker.qadr[j]]) * RAD
        finally:
            self.data = live
            self.tracker.data = live

    def _foot_pin_targets(self) -> Dict[str, np.ndarray]:
        """World positions of the sites pinned to the ground.

        Heel and toe of each foot, so the feet are genuinely planted -- pinning
        only the ankle leaves the foot free to spin about the vertical, and the
        leg's rotation then goes into the foot instead of turning the pelvis.
        The freedom the turning pelvis needs comes from `ankle_*_rot` instead.
        """
        names = [f"{p}_{s}" for s in ("l", "r") for p in ("heel", "toe")]
        out = {}
        for name in names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            out[name] = self.data.site_xpos[sid].copy()
        return out

    def _face_heading_error(self) -> float:
        """How far the clubface points away from square, in degrees."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "clubface")
        n = self.data.site_xmat[sid].reshape(3, 3)[:, 2]     # site z = face normal
        return math.degrees(math.atan2(n[1], n[0]))

    def _settle(self, seconds: float) -> None:
        """Hold the solved address pose (which includes the IK'd trail arm) and
        let gravity and the grip constraint find the real equilibrium before
        anything is measured off the pose."""
        for n, v in self.address_angles.items():
            self.data.ctrl[self.controller_act(n)] = v * DEG
        for _ in range(int(seconds / self.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def controller_act(self, joint: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                 f"act_{joint}")

    # ---- calibration -------------------------------------------------------
    def _swing_arc(self) -> Tuple[float, float, float]:
        """Run one swing with the ball out of the way and report where the
        clubhead is at its fastest on the way down.

        Not the lowest point of the arc: the club is slowest there (the pelvis
        has run out of rotation by then), and a ball sitting at the very bottom
        gets a limp strike.  What matters for impact is the height of the
        *fast* part of the arc.
        """
        self.reset(place_ball=False)
        self._set_ball(np.array([0.0, 8.0, BALL_RADIUS]))
        times = {e[0]: e[1] for e in self.controller.phases}
        t0, t1 = times["transition"], times["extension"]
        best = (0.0, 0.0, 0.0)                  # speed, z, t
        for i in range(int(self.controller.duration / self.timestep)):
            t = i * self.timestep
            self.controller.apply(self.data, t)
            mujoco.mj_step(self.model, self.data)
            if not (t0 <= t <= t1):
                continue
            v = self.tracker.site_velocity("clubhead")
            speed = float(np.linalg.norm(v))
            if v[0] > 2.0 and speed > best[0]:
                z = float(self.tracker.positions()[
                    self.tracker.index["clubhead"]][2])
                best = (speed, z, t)
        return best[1], best[2], best[0]

    def _tune_release(self, script, iters: int = 6, tol: float = 0.015) -> None:
        """Move the planned impact height until the *actual* swing bottoms out
        at ball height.

        How deep the arc really bottoms depends on the golfer's proportions,
        the club, the tempo and how far the torque-limited servos lag, so it is
        not something that can sensibly be written into the keyframes by hand.
        Aiming the plan lower or higher and re-measuring converges in a few
        swings, and unlike holding the wrists back it does not cost any speed.
        """
        L = self.anthro.lead
        key = f"shoulder_{L}_flex"
        phases = {e[0]: e[2] for e in self.controller.phases}
        ideal = TEE_HEIGHT + BALL_RADIUS
        total = 0.0
        z = t = speed = 0.0
        for _ in range(iters):
            z, t, speed = self._swing_arc()
            err = z - ideal
            if abs(err) < tol:
                break
            # ~1 deg of lead shoulder flexion raises the hands, and with them
            # the whole arc, by a bit over a centimetre
            delta = float(np.clip(-err / 0.012, -20.0, 20.0))
            # keep the correction to something a golfer could plausibly do:
            # left unbounded it just jams the lead arm down at its stop
            delta = float(np.clip(total + delta, -25.0, 25.0)) - total
            total += delta
            if abs(delta) < 1e-6:
                break
            for name, w in (("impact", 1.0), ("through", 1.0),
                            ("extension", 0.5)):
                phases[name][key] = self._clamp_deg(key,
                                                    phases[name][key] + w * delta)
            self._plan_trail_arm(script, warn=False)
        if self.verbose:
            print(f"  swing tuned (hands {total:+.0f} deg): fastest point of "
                  f"the arc is {z * 1000:.0f} mm up at t={t:.3f} s, "
                  f"clubhead {speed:.1f} m/s")

    def _clamp_deg(self, joint: str, deg: float) -> float:
        lo, hi = self.model.jnt_range[self.tracker.jid[joint]] * RAD
        return float(min(max(deg, lo), hi))

    def _calibrate_ball(self) -> None:
        """Swing once with the ball out of the way and put it where the
        clubface actually passes through the bottom of the arc, travelling
        fastest -- which is how a golfer picks ball position in the first
        place, rather than assuming the club returns exactly to address."""
        self.reset(place_ball=False)
        self._set_ball(np.array([0.0, 8.0, BALL_RADIUS]))
        face_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "clubface")
        times = {e[0]: e[1] for e in self.controller.phases}
        t0, t1 = times["transition"], times["extension"]
        ideal = TEE_HEIGHT + BALL_RADIUS
        best = None
        for i in range(int(self.controller.duration / self.timestep)):
            t = i * self.timestep
            self.controller.apply(self.data, t)
            mujoco.mj_step(self.model, self.data)
            if not (t0 <= t <= t1):
                continue
            vel = self.tracker.site_velocity("clubhead")
            speed = float(np.linalg.norm(vel))
            if vel[0] < 2.0:              # must be travelling down the target line
                continue
            pos = self.data.site_xpos[face_sid].copy()
            # prefer the fastest sample that is within a ball's width of tee
            # height, and fall back to the closest one if it never gets there
            err = abs(pos[2] - ideal)
            score = (max(err - BALL_RADIUS, 0.0), -speed)
            if best is None or score < best[0]:
                normal = self.data.site_xmat[face_sid].reshape(3, 3)[:, 2].copy()
                best = (score, err, pos, normal, speed, t)
        if best is None:
            if self.verbose:
                print("  [warn] no impact point found; ball left at address")
            return
        _score, err, pos, normal, speed, t = best
        self.ball_pos = pos + BALL_RADIUS * normal
        self.ball_pos[2] = max(self.ball_pos[2], TEE_HEIGHT + BALL_RADIUS)
        if self.verbose:
            print(f"  ball placed at the bottom of the arc: t={t:.3f}s "
                  f"clubhead {speed:.1f} m/s  (arc bottom {err * 1000:.0f} mm "
                  f"from tee height)")

    def _set_ball(self, pos: np.ndarray) -> None:
        a = self.ball_jadr
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.model.body_pos[self.tee_bid] = np.array([pos[0], pos[1], 0.0])

    def ball_contact(self) -> bool:
        """True while the clubhead is touching the ball."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._ball_gid in pair and self._head_gid in pair:
                return True
        return False

    # ---- running -----------------------------------------------------------
    def reset(self, place_ball: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos_address
        self.data.qvel[:] = 0.0
        if place_ball:
            self._set_ball(self.ball_pos)
        self.controller.apply(self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def step(self, t: Optional[float] = None) -> None:
        self.controller.apply(self.data, self.data.time if t is None else t)
        mujoco.mj_step(self.model, self.data)

    @property
    def duration(self) -> float:
        return self.controller.duration


# ===========================================================================
# 8.  Reporting
# ===========================================================================

def print_model_summary(sim: GolfSwingSim) -> None:
    m = sim.model
    print("=" * 74)
    print("MODEL")
    print("=" * 74)
    print(f"  golfer            {sim.anthro.height:.2f} m, {sim.anthro.mass:.0f} kg, "
          f"{sim.anthro.handedness}-handed")
    print(f"  club              {sim.club.length:.2f} m, "
          f"{(sim.club.head_mass + sim.club.shaft_mass + sim.club.grip_mass) * 1000:.0f} g, "
          f"{sim.club.loft_deg:.1f} deg loft")
    print(f"  base              {sim.base}")
    print(f"  bodies/dof/act    {m.nbody} / {m.nv} / {m.nu}   "
          f"(timestep {m.opt.timestep * 1e3:.2f} ms)")
    print(f"  tracked joints    {len(TRACKED)} landmarks, "
          f"{len(JOINT_NAMES)} actuated joints")


def print_pose_table(sim: GolfSwingSim, title: str) -> None:
    tr = sim.tracker
    ego = tr.egocentric()
    seg = tr.parent_relative()
    print()
    print("=" * 74)
    print(f"{title}  (t = {sim.data.time:.3f} s)")
    print("=" * 74)
    print(f"{'landmark':<12}{'world x':>9}{'y':>8}{'z':>8}"
          f"{'| pelvis-frame f/l/u':>28}{'| from parent':>16}")
    for i, n in enumerate(TRACKED):
        p = tr.positions()[i]
        e = ego[i]
        s = np.linalg.norm(seg[i])
        par = PARENT[n] or "-"
        print(f"{n:<12}{p[0]:9.3f}{p[1]:8.3f}{p[2]:8.3f}   "
              f"{e[0]:8.3f}{e[1]:8.3f}{e[2]:8.3f}   "
              f"{par:>10}{s:7.3f}")


def print_metrics(rows: List[Dict[str, float]], phases: List[str]) -> None:
    print()
    print("=" * 74)
    print("SWING KINEMATICS")
    print("=" * 74)
    hdr = (f"{'phase':<11}{'t':>6}{'hip':>8}{'shldr':>9}{'X-fac':>8}"
           f"{'cock':>7}{'tilt':>8}{'club m/s':>9}{'head z':>9}")
    print(hdr)
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
    print("=" * 74)
    print("PAIRWISE LANDMARK DISTANCES (m) -- SwingTracker.pairwise_distances()")
    print("=" * 74)
    print(" " * 11 + "".join(f"{n[:9]:>10}" for n in subset))
    for n, i in zip(subset, idx):
        print(f"{n:<11}" + "".join(f"{D[i, j]:10.3f}" for j in idx))


# ===========================================================================
# 9.  Entry point
# ===========================================================================

def run_swing(sim: GolfSwingSim, csv_path: Optional[str] = None,
              collect_phases: bool = True) -> Tuple[List[Dict[str, float]],
                                                    List[str], Dict[str, float]]:
    """Run one swing, return the per-phase metrics and the impact summary."""
    sim.reset()
    tr = sim.tracker
    phases = sim.controller.phases
    rows: List[Dict[str, float]] = []
    names: List[str] = []
    next_phase = 0

    writer = None
    fh = None
    if csv_path:
        import csv as _csv
        fh = open(csv_path, "w", newline="")
        writer = _csv.writer(fh)
        header = ["t", "phase"]
        header += [f"{n}_{ax}" for n in TRACKED for ax in "xyz"]
        header += [f"q_{n}" for n in JOINT_NAMES]
        header += ["hip_turn", "shoulder_turn", "x_factor", "lead_wrist_cock",
                   "side_tilt", "clubhead_speed", "ball_speed"]
        writer.writerow(header)

    best_impact = {"clubhead_speed": 0.0, "ball_speed": 0.0, "t": 0.0,
                   "attack_angle": 0.0, "club_path": 0.0, "hit": False}
    struck = False
    n_steps = int((sim.duration + 0.25) / sim.timestep)
    log_every = max(1, int(0.002 / sim.timestep))

    for i in range(n_steps):
        t = sim.data.time
        if collect_phases and next_phase < len(phases) and t >= phases[next_phase][1]:
            rows.append(tr.metrics())
            names.append(phases[next_phase][0])
            next_phase += 1
        if writer and i % log_every == 0:
            m = tr.metrics()
            row = [f"{t:.5f}", sim.controller.phase_at(t)]
            row += [f"{v:.5f}" for v in tr.positions().ravel()]
            row += [f"{v:.5f}" for v in tr.joint_angles().values()]
            row += [f"{m[k]:.4f}" for k in
                    ("hip_turn", "shoulder_turn", "x_factor", "lead_wrist_cock",
                     "side_tilt", "clubhead_speed", "ball_speed")]
            writer.writerow(row)
        # impact is read off the actual clubhead/ball contact: clubhead speed
        # at first touch, ball speed once it has separated
        if not struck and sim.ball_contact():
            struck = True
            m = tr.metrics()
            best_impact = {"clubhead_speed": m["clubhead_speed"],
                           "ball_speed": 0.0, "t": t, "hit": True,
                           "attack_angle": m["attack_angle"],
                           "club_path": m["club_path"]}
        if struck:
            best_impact["ball_speed"] = max(best_impact["ball_speed"],
                                            tr.ball_speed())
        sim.step()

    if fh:
        fh.close()
    return rows, names, best_impact


def view_swing(sim: GolfSwingSim, slowmo: float = 0.25,
               loops: int = 0) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        viewer.cam.distance = 4.2
        viewer.cam.lookat[:] = sim.tracker.positions()[
            sim.tracker.index["pelvis"]]
        n = 0
        sim.reset()
        wall0 = time.time()
        while viewer.is_running():
            if sim.data.time > sim.duration + 0.6:
                n += 1
                if loops and n >= loops:
                    break
                sim.reset()
                wall0 = time.time()
            sim.step()
            # sync at ~60 Hz of wall clock, played back in slow motion
            target_wall = wall0 + sim.data.time / max(slowmo, 1e-3)
            lag = target_wall - time.time()
            if lag > 0.001:
                viewer.sync()
                time.sleep(min(lag, 0.02))


def main() -> None:
    ap = argparse.ArgumentParser(description="MuJoCo human golf swing")
    ap.add_argument("--height", type=float, default=1.78, help="golfer height (m)")
    ap.add_argument("--mass", type=float, default=78.0, help="golfer mass (kg)")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--club-length", type=float, default=1.14)
    ap.add_argument("--base", choices=["feet", "free", "pinned"], default="feet",
                    help="feet: pelvis free but feet planted (default); "
                         "free: full balance physics; pinned: pelvis welded")
    ap.add_argument("--timestep", type=float, default=2e-4)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="scale on every joint torque limit")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="stretch the whole swing in time (1.5 = slower swing)")
    ap.add_argument("--lead", type=float, default=0.07,
                    help="how far ahead of the clock the servos are fed their "
                         "reference, to cancel tracking lag (s)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the tuning swings and ball placement")
    ap.add_argument("--slowmo", type=float, default=0.25,
                    help="viewer playback rate (0.25 = quarter speed)")
    ap.add_argument("--no-view", action="store_true", help="run headless")
    ap.add_argument("--report", action="store_true",
                    help="print the kinematic report (implies --no-view)")
    ap.add_argument("--csv", type=str, default=None,
                    help="write every tracked landmark to this CSV")
    ap.add_argument("--xml", type=str, default=None,
                    help="dump the generated MJCF here and exit")
    args = ap.parse_args()

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
        print_pose_table(sim, "ADDRESS POSITION")
        print_metrics(rows, names)
        print_distance_matrix(sim, ["pelvis", "thorax", f"shoulder_{anthro.lead}",
                                    f"wrist_{anthro.lead}", "clubhead", "ball"])
        print()
        print("=" * 74)
        print("IMPACT")
        print("=" * 74)
        if impact["hit"]:
            smash = impact["ball_speed"] / max(impact["clubhead_speed"], 1e-6)
            print(f"  contact at t = {impact['t']:.3f} s")
            print(f"  clubhead speed  {impact['clubhead_speed']:6.1f} m/s  "
                  f"({impact['clubhead_speed'] * 2.23694:5.1f} mph)")
            print(f"  ball speed      {impact['ball_speed']:6.1f} m/s  "
                  f"({impact['ball_speed'] * 2.23694:5.1f} mph)")
            print(f"  smash factor    {smash:6.2f}")
            print(f"  attack angle    {impact['attack_angle']:6.1f} deg")
            print(f"  club path       {impact['club_path']:6.1f} deg")
        else:
            print("  the club missed the ball")
        if args.csv:
            print(f"\n  wrote {args.csv}")
    else:
        print("\nopening viewer -- drag to orbit, scroll to zoom, Esc to quit")
        view_swing(sim, slowmo=args.slowmo)


if __name__ == "__main__":
    main()

"""Generates the MJCF for golfer + club + ball + tee.

Everything is emitted from `Anthropometry` and `Club`, so there is no static
XML file to keep in sync -- change the golfer's height and the model rebuilds
around it.

World frame: +x is the target line, +z is up.  The golfer's own frame has +x
facing the ball, +y to their left and +z up the spine; the root body carries
the rotation between the two, including the forward lean at address.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import mujoco
import numpy as np

from .anthropometry import Anthropometry
from .equipment import (BALL_MASS, BALL_RADIUS, TEE_HEIGHT, Club)
from .joints import ADDRESS_LEAN, JointSpec, all_joints
from .util import DEG, fmt_vec, quat_axis


def build_model_xml(
    anthro: Anthropometry,
    club: Club,
    base: str = "feet",
    timestep: float = 2e-4,
    head_quat: Optional[np.ndarray] = None,
    foot_pins: Optional[Dict[str, np.ndarray]] = None,
    strength: float = 1.0,
    trail_arm_gain: float = 0.12,
) -> str:
    """Emit the MuJoCo XML for the whole scene.

    base: "feet"   pelvis free, both feet pinned to the ground (default -- the
                   golfer cannot topple, which is what you want while the swing
                   itself is being studied)
          "free"   pelvis free, nothing pinned: full balance physics
          "pinned" pelvis welded to the world

    `head_quat` and `foot_pins` are measured off a first, provisional build of
    this same model -- see `GolfSwingSim.__init__`.
    """
    a, c = anthro, club
    H = a.height
    lead, trail = a.lead, a.trail

    # ---- root placement ---------------------------------------------------
    # golfer faces -y (right handed) so that their left, and the target, is +x
    yaw = -90.0 * DEG * a.lead_sign
    q_yaw = quat_axis((0, 0, 1), yaw)
    # rotate about the pelvis' own left axis so the spine tips forward, over
    # the ball; the hips are extended by the same amount in ADDRESS so the
    # thighs come back to vertical
    q_lean = quat_axis((0, 1, 0), ADDRESS_LEAN * DEG)
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
    palm = np.array([0.0, 0.0, -0.35 * a.hand_len])        # grip point in hand
    shaft_up = np.array([-math.sin(g), 0.0, math.cos(g)])  # club +z in hand frame
    club_pos = palm + c.lead_hand_drop * shaft_up
    q_club = quat_axis((0, -1, 0), g)

    # ---- clubhead ---------------------------------------------------------
    # The head does NOT inherit the shaft's frame.  A real head sits level on
    # the turf with the shaft entering the heel at the lie angle, so its frame
    # is set from the outside: `head_quat` is measured at address such that the
    # head ends up square to the target and level (x = target line, z = up).
    # Left as identity on the first build, then solved -- see
    # `AddressSolver.head_alignment`.
    q_head = (np.array([1.0, 0.0, 0.0, 0.0]) if head_quat is None
              else np.asarray(head_quat, dtype=float))
    loft = c.loft_deg * DEG
    face_normal = np.array([math.cos(loft), 0.0, math.sin(loft)])
    # heel is at the shaft tip; the head extends toward the toe, away from the
    # golfer, and hangs below the tip
    head_ctr = np.array([0.0, -a.lead_sign * 0.045, -0.025])
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

    # ---- arms (the lead one carries the club) -----------------------------
    def arm(side: str) -> str:
        s = 1.0 if side == "l" else -1.0
        holds_club = (side == lead)
        club_body = f"""
          <body name="club" pos="{fmt_vec(club_pos)}" quat="{fmt_vec(q_club)}">
            <site name="grip" pos="0 0 -0.02" class="landmark"/>
            <site name="grip_lead" pos="0 0 {-c.lead_hand_drop:.4f}" class="grip"/>
            <site name="grip_trail" pos="0 0 {-c.trail_hand_drop:.4f}" class="grip"/>
            <geom name="grip_g" type="capsule" size="0.014"
                  fromto="0 0 0.01 0 0 -0.26" mass="{c.grip_mass:.4f}"
                  rgba="0.15 0.15 0.17 1"/>
            <geom name="shaft_g" type="capsule" size="0.0055"
                  fromto="0 0 -0.26 0 0 {-c.length:.4f}" mass="{c.shaft_mass:.4f}"
                  rgba="0.75 0.76 0.8 1"/>
            <body name="club_head" pos="0 0 {-c.length:.4f}" quat="{fmt_vec(q_head)}">
              <geom name="club_head_g" type="box" size="0.020 0.050 0.030"
                    pos="{fmt_vec(head_ctr)}" euler="0 {-c.loft_deg} 0"
                    mass="{c.head_mass:.4f}" class="strike"
                    rgba="0.2 0.2 0.24 1"/>
              <site name="clubhead" pos="{fmt_vec(head_ctr)}" class="landmark"/>
              <site name="clubface" pos="{fmt_vec(face_ctr)}"
                    zaxis="{fmt_vec(face_normal)}" class="grip"/>
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
              <site name="hand_{side}" pos="{fmt_vec(palm)}" class="landmark"/>
              <site name="palm_{side}" pos="{fmt_vec(palm)}" class="grip"/>
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
            pin_sites.append(f'<site name="pin_{key}" pos="{fmt_vec(pos)}" '
                             f'class="pin"/>')
            eqs.append(f'<connect name="pin_{key}" site1="{key}" '
                       f'site2="pin_{key}" solref="0.004 1" '
                       f'solimp="0.99 0.9999 0.001"/>')

    root_joint = '<freejoint name="root"/>' if base != "pinned" else ""

    return f"""
<mujoco model="rl_golf">
  <compiler angle="degree" autolimits="true"/>
  <!-- Pyramidal friction, not elliptic: elliptic costs 2.4x per step and the
       swing is identical to three significant figures, because the feet are
       held by equality constraints rather than by friction.  Newton converges
       in under one iteration here, so the iteration cap is not binding. -->
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="30" tolerance="1e-8"
          cone="pyramidal" impratio="3"/>
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

    <body name="pelvis" pos="{fmt_vec(root_pos)}" quat="{fmt_vec(q_root)}">
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

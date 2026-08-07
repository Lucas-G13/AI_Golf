"""Every joint in the golfer, and the pose they start from.

A `JointSpec` is (name, axis, range_lo_deg, range_hi_deg, kp, torque_limit_Nm,
damping).  Axes are written in the *zero pose* frame -- standing upright, arms
hanging, palms facing the thighs -- which is each body's own frame.  That means
a wrist axis rotates correctly with forearm pronation, because pronation is a
joint further up the same chain.

Sign convention: positive is the anatomical positive (flexion, abduction,
internal rotation).  For the trunk's lateral bend and axial rotation, positive
is *toward the target* -- the downswing direction -- for a golfer of either
handedness, which is why those axes take `lead_sign`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

JointSpec = Tuple[str, str, float, float, float, float, float]


def side_joints(side: str) -> List[JointSpec]:
    """Leg + arm joints for one side.

    s = +1 for the left, -1 for the right, so a positive joint value means the
    same anatomical motion on both sides.
    """
    s = 1.0 if side == "l" else -1.0
    return [
        # ---- leg ----------------------------------------------------------
        (f"hip_{side}_flex",  "0 -1 0",   -25, 120, 6000, 220, 6.0),
        (f"hip_{side}_abd",   f"{s} 0 0",  -30,  45, 5000, 160, 6.0),
        (f"hip_{side}_rot",   f"0 0 {-s}", -45,  45, 3000,  90, 4.0),
        (f"knee_{side}",      "0 1 0",       0, 150, 5000, 200, 6.0),
        (f"ankle_{side}_dorsi", "0 -1 0",  -50,  30, 4000, 160, 4.0),
        (f"ankle_{side}_roll",  f"{s} 0 0", -20,  20, 2000,  60, 3.0),
        # Shank rotation over the foot (tibial + subtalar, plus a little sole
        # slip).  Without it a golfer with both feet planted has no way to turn
        # the pelvis except by sinking into the ground.
        (f"ankle_{side}_rot",   f"0 0 {-s}", -30,  30, 1500,  70, 3.0),
        # ---- arm ----------------------------------------------------------
        (f"shoulder_{side}_abd",  f"{s} 0 0", -60, 170, 1600, 110, 2.0),
        (f"shoulder_{side}_flex", "0 -1 0",   -70, 170, 1600, 110, 2.0),
        (f"shoulder_{side}_rot",  f"0 0 {-s}", -90,  90,  700,  60, 1.5),
        # A few degrees of hyperextension is real, and it matters here: with
        # both elbows locked at exactly 0 the two arms and the shaft form a
        # rigid closed loop and the club cannot move at all.
        (f"elbow_{side}_flex",    "0 -1 0",     -8, 150,  900,  80, 1.5),
        (f"elbow_{side}_pro",     f"0 0 {-s}", -85,  85,  300,  25, 0.8),
        (f"wrist_{side}_flex",    f"{-s} 0 0", -70,  80,  400,  45, 0.5),
        # Radial deviation runs past anatomical ROM: together with the club's
        # grip angle it stands in for the golf "cock/hinge" of the wrist.  Its
        # torque limit lumps in the forearm and grip, which is what actually
        # holds and then releases the club's lag angle.
        (f"wrist_{side}_dev",     "0 -1 0",    -25,  55,  700,  70, 0.5),
    ]


def spine_joints(lead_sign: float = 1.0) -> List[JointSpec]:
    """Trunk and neck joints, signed relative to the target (see module doc)."""
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
    return spine_joints(lead_sign) + side_joints("l") + side_joints("r")


#: Joint names, in model order.  Independent of handedness.
JOINT_NAMES: List[str] = [j[0] for j in all_joints()]


# ---------------------------------------------------------------------------
# Address pose (degrees)
# ---------------------------------------------------------------------------

#: How far the pelvis itself is tilted forward over the ball.  The model's root
#: body carries this rotation; the hips then take it back out (see below).
ADDRESS_LEAN = 32.0

ADDRESS: Dict[str, float] = {
    "lumbar_flex": 2, "lumbar_bend": -6, "lumbar_rot": 0,
    # Tilted away from the target.  This is what drops the trail shoulder below
    # the lead one, and it is the only reason the trail hand can reach a grip
    # that sits below the lead hand.
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

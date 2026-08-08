"""From clubhead to landing: impact, launch conditions, and ball flight.

This replaces the rigid-body ball contact as the source of the reward.  MuJoCo
can tell us where the clubhead is and how fast it is going, which is all a
launch monitor measures; everything after that is better computed than
simulated.  Two reasons:

* MuJoCo's contact gives a smash factor around 0.8 where a real driver gives
  1.48.  Its ball speed, and therefore any distance derived from it, is wrong.
* Sidespin comes from gear effect -- the head twisting about its own centre of
  mass on an off-centre strike.  A rigid box striking a sphere produces none of
  it, so "how straight was that shot" is not even representable.

The impact model here is an oblique rigid-body collision with a gripping
(non-slipping) contact.  That is enough to reproduce, from first principles:

    smash factor      (1 + e) * m_eff / (m_eff + m_ball)  ~ 1.48 when centred
    the D-plane       the ball starts between the face normal and the club
                      path, because the impulse has a normal and a tangential
                      part -- no hand-tuned 85/15 rule needed
    spin from loft    the tangential impulse rolls the ball up the face
    off-centre loss   an off-centre strike rotates the head, which lowers the
                      effective mass it presents to the ball
    gear effect       that same head rotation spins the ball the other way

Calibrated, not derived: the coefficient of restitution, the head's moments of
inertia, the gear ratio, and the aerodynamic coefficients.  Each is a named
constant below with the benchmark it was set against.  See `benchmarks()`.

Known limits.  Ball speed, smash factor, spin rate and carry all land within a
few percent of launch-monitor figures, but *curvature runs strong*: a shot with
a 20 deg spin axis finishes 50-60 m offline where TrackMan would say 35-45.
Rank ordering and sign are right, magnitude is roughly 1.3x.  That is fine for
shaping a reward -- straighter still scores better -- but do not read the
offline numbers as yardages.  Also assumes the ball grips the face, which holds
for a driver and gets shakier as loft increases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .equipment import BALL_CENTRE_HEIGHT, BALL_MASS, BALL_RADIUS, Club

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Coefficient of restitution.  The USGA limit is 0.830 and every driver worth
#: hitting sits on it.  Real COR falls slightly with impact speed; this does
#: not model that.
COR = 0.830

#: Driver head moments of inertia about its own centre of mass (kg m^2).
#: MOI_VERTICAL resists twisting on a heel/toe miss (USGA limit 5900 g cm^2);
#: MOI_HORIZONTAL resists it on a high/low miss, and is always the smaller.
MOI_VERTICAL = 5.2e-4
MOI_HORIZONTAL = 3.2e-4

#: Gear ratio between head rotation and the spin it imparts.  Physically this
#: is roughly (CoM-to-face distance) / (ball radius).  Set so a 15 mm toe
#: strike at driver speed produces ~900 rpm of draw spin.
GEAR_RATIO = 1.0

#: Ball inertia factor: a uniform sphere has I = (2/5) m r^2, so a gripping
#: oblique impact leaves the ball with 2/7 of the tangential velocity and
#: (5/7) v_t / r of spin.  Both fall out of the same calculation.
_BALL_TANGENTIAL = 2.0 / 7.0
_BALL_SPIN = 5.0 / 7.0

#: Half-extent of a driver face, heel-to-toe and crown-to-sole (m).  Used to
#: decide whether the face actually covered the ball.
FACE_HALF_WIDTH = 0.050
FACE_HALF_HEIGHT = 0.030

# --- aerodynamics ----------------------------------------------------------
AIR_DENSITY = 1.225           # kg/m^3, sea level
DRAG_COEFF = 0.25             # golf ball at Re ~ 1.5e5
#: Lift coefficient per unit spin ratio S = omega*r/v, capped.  A golf ball at
#: S ~ 0.1 has Cl ~ 0.2, saturating around 0.3 -- calibrated so the benchmark
#: tour drive (73 m/s ball, 2400 rpm) carries ~250 m.
LIFT_PER_SPIN = 2.2
LIFT_CAP = 0.33
SPIN_DECAY_S = 25.0           # spin falls off with this time constant
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Impact:
    """What the simulator measured at the moment the club passed the ball."""

    club_velocity: np.ndarray      # world frame, m/s
    face_normal: np.ndarray        # world frame, unit, pointing down-target
    #: Where on the face the ball sat, in face coordinates (m):
    #: [0] toward the toe, [1] toward the crown.  (0, 0) is the sweet spot.
    face_offset: np.ndarray
    #: Closest approach of the ball's centre to the face plane centre (m).
    #: Zero for a dead-centre strike.
    miss_distance: float = 0.0

    @property
    def clubhead_speed(self) -> float:
        return float(np.linalg.norm(self.club_velocity))

    @property
    def on_face(self) -> bool:
        """Did the face actually cover the ball?"""
        return (abs(self.face_offset[0]) < FACE_HALF_WIDTH + BALL_RADIUS and
                abs(self.face_offset[1]) < FACE_HALF_HEIGHT + BALL_RADIUS)


@dataclass(frozen=True)
class Launch:
    """Ball state leaving the face -- what a launch monitor reports."""

    velocity: np.ndarray           # world frame, m/s
    spin: np.ndarray               # world frame, rad/s (vector along spin axis)
    smash: float

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def launch_angle(self) -> float:
        """Degrees above horizontal."""
        v = self.velocity
        return math.degrees(math.atan2(v[2], max(np.linalg.norm(v[:2]), 1e-9)))

    @property
    def azimuth(self) -> float:
        """Start direction, degrees; positive is right of the target line."""
        return math.degrees(math.atan2(-self.velocity[1], self.velocity[0]))

    def _spin_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(lift axis, curve axis) for the spin.

        Only the part of the spin perpendicular to the direction of travel does
        anything -- the component along it is rifle spin and produces no force
        at all -- so the spin splits into exactly these two useful pieces.
        """
        v = self.velocity
        speed = np.linalg.norm(v)
        horiz = np.array([v[0], v[1], 0.0])
        h = np.linalg.norm(horiz)
        if speed < 1e-9 or h < 1e-9:
            return None
        lift = np.cross(horiz / h, np.array([0.0, 0.0, 1.0]))
        return lift, np.cross(v / speed, lift)

    @property
    def backspin_rpm(self) -> float:
        """Component of spin that lifts the ball."""
        frame = self._spin_frame()
        if frame is None:
            return 0.0
        return float(np.dot(self.spin, frame[0])) * 60.0 / (2 * math.pi)

    @property
    def sidespin_rpm(self) -> float:
        """Component of spin that curves it.  Positive bends the shot right."""
        frame = self._spin_frame()
        if frame is None:
            return 0.0
        return float(np.dot(self.spin, frame[1])) * 60.0 / (2 * math.pi)

    @property
    def spin_axis(self) -> float:
        """Tilt of the spin axis from horizontal, degrees.  Positive tilts the
        shot right (a fade for a right-hander); this is what curves the ball."""
        return math.degrees(math.atan2(self.sidespin_rpm,
                                       max(abs(self.backspin_rpm), 1e-6)))


@dataclass(frozen=True)
class Flight:
    """Where it finished."""

    carry: float                   # metres down the target line
    lateral: float                 # metres right of the target line (+ = right)
    apex: float                    # metres
    hang_time: float               # seconds
    landing_speed: float           # m/s

    @property
    def offline(self) -> float:
        return abs(self.lateral)


# ---------------------------------------------------------------------------
# Impact
# ---------------------------------------------------------------------------

def _face_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the face's tangent axes: (toe direction, crown direction).

    The toe axis is horizontal -- a clubface can be open or shut, but the
    heel-toe axis stays level -- and the crown axis completes the frame.
    """
    up = np.array([0.0, 0.0, 1.0])
    toe = np.cross(normal, up)
    n = np.linalg.norm(toe)
    if n < 1e-9:                       # face pointing straight up or down
        toe = np.array([0.0, -1.0, 0.0])
    else:
        toe = toe / n
    crown = np.cross(toe, normal)
    return toe, crown / max(np.linalg.norm(crown), 1e-9)


def strike(impact: Impact, club: Optional[Club] = None) -> Launch:
    """Oblique impact of a gripping ball on a moving clubface."""
    club = club or Club()
    n = impact.face_normal / max(np.linalg.norm(impact.face_normal), 1e-9)
    v = impact.club_velocity

    v_n = float(np.dot(v, n))          # closing speed, along the face normal
    v_t = v - v_n * n                  # face sliding across the ball
    if v_n <= 0.0:                     # face moving away: no strike
        return Launch(np.zeros(3), np.zeros(3), 0.0)

    # --- effective mass ---------------------------------------------------
    # An off-centre strike spins the head instead of driving the ball, so the
    # head presents less mass to the ball the further out you catch it.
    dy, dz = float(impact.face_offset[0]), float(impact.face_offset[1])
    m_head = club.head_mass
    twist = m_head * (dy ** 2 / MOI_VERTICAL + dz ** 2 / MOI_HORIZONTAL)
    m_eff = m_head / (1.0 + twist)

    # --- normal impulse: this is the ball speed ---------------------------
    ball_n = (1.0 + COR) * m_eff / (m_eff + BALL_MASS) * v_n
    impulse = BALL_MASS * ball_n

    # --- tangential impulse: this is the spin -----------------------------
    # The ball grips the face, so friction drags it up the face until the
    # contact point stops slipping: it keeps 2/7 of the tangential speed and
    # spins at 5/7 v_t / r.
    ball_t = _BALL_TANGENTIAL * v_t
    spin_from_slip = np.zeros(3)
    speed_t = float(np.linalg.norm(v_t))
    if speed_t > 1e-9:
        # The face slides down across the ball, so the ball rolls *up* it: the
        # spin axis is t x n, which for a lofted face is backspin.
        axis = np.cross(v_t / speed_t, n)
        spin_from_slip = axis * (_BALL_SPIN * speed_t / BALL_RADIUS)

    # --- gear effect ------------------------------------------------------
    # The same off-centre impulse twists the head, and the ball, gripping the
    # face, is geared by that rotation in the opposite sense.  A toe strike
    # opens the head and draws the ball back.
    spin_gear = np.zeros(3)
    offset = math.hypot(dy, dz)
    if offset > 1e-6:
        toe, crown = _face_frame(n)
        d_vec = toe * dy + crown * dz
        moi = (MOI_VERTICAL * (dy / offset) ** 2 +
               MOI_HORIZONTAL * (dz / offset) ** 2)
        head_spin = impulse * offset / moi                # rad/s of head twist
        # The ball pushes the head *back*, so the head turns about -(d x n):
        # a toe strike opens the face.  The ball is geared the opposite way,
        # which puts it back on +(d x n) -- a toe strike draws.
        spin_gear = GEAR_RATIO * head_spin * np.cross(d_vec / offset, n)

    velocity = ball_n * n + ball_t
    smash = float(np.linalg.norm(velocity)) / max(impact.clubhead_speed, 1e-9)
    return Launch(velocity, spin_from_slip + spin_gear, smash)


# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------

def fly(launch: Launch, dt: float = 0.01, max_time: float = 12.0) -> Flight:
    """Integrate the ball to the ground under gravity, drag and Magnus lift."""
    area = math.pi * BALL_RADIUS ** 2
    k = 0.5 * AIR_DENSITY * area / BALL_MASS

    # Start on the tee, not at z = 0.  A ball driven downwards has to be able
    # to reach the ground and stop: with the old zero start the "is it below
    # the ground yet" test could never fire on the first step, and a shot
    # smashed into the turf flew underground for the full 12 s and reported a
    # 250 m carry off a 20 m/s ball.
    p = np.array([0.0, 0.0, BALL_CENTRE_HEIGHT])
    v = launch.velocity.copy()
    w = launch.spin.copy()
    if launch.speed < 1e-6:
        return Flight(0.0, 0.0, 0.0, 0.0, 0.0)

    def accel(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.array([0.0, 0.0, -GRAVITY])
        spin_ratio = float(np.linalg.norm(w)) * BALL_RADIUS / speed
        cl = min(LIFT_PER_SPIN * spin_ratio, LIFT_CAP)
        drag = -k * DRAG_COEFF * speed * v
        lift = np.zeros(3)
        wn = float(np.linalg.norm(w))
        if wn > 1e-6:
            lift = k * cl * speed * np.cross(w / wn, v)
        return drag + lift + np.array([0.0, 0.0, -GRAVITY])

    apex = 0.0
    t = 0.0
    while t < max_time:
        # RK2 is plenty at dt = 10 ms for a 6 s flight
        a1 = accel(v, w)
        v_mid = v + 0.5 * dt * a1
        a2 = accel(v_mid, w)
        p_next = p + dt * (v + 0.5 * dt * a1)
        v = v + dt * a2
        w = w * math.exp(-dt / SPIN_DECAY_S)
        t += dt
        apex = max(apex, p_next[2])
        if p_next[2] <= 0.0:
            # land: interpolate to the ground crossing
            f = p[2] / max(p[2] - p_next[2], 1e-9)
            p = p + f * (p_next - p)
            break
        p = p_next

    return Flight(carry=float(p[0]), lateral=float(-p[1]), apex=float(apex),
                  hang_time=float(t), landing_speed=float(np.linalg.norm(v)))


def shot(impact: Impact, club: Optional[Club] = None) -> Tuple[Launch, Flight]:
    """Impact through to landing."""
    launch = strike(impact, club)
    return launch, fly(launch)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def square_impact(speed: float, attack_deg: float = 0.0,
                  face_deg: float = 0.0, path_deg: float = 0.0,
                  loft_deg: float = 10.5,
                  offset: Tuple[float, float] = (0.0, 0.0)) -> Impact:
    """Build an `Impact` from launch-monitor style numbers, for testing.

    Angles are degrees; `face_deg` and `path_deg` are positive to the right of
    the target line, `attack_deg` positive for hitting up on the ball.
    """
    d2r = math.pi / 180.0
    # club path: horizontal direction plus attack angle
    ca, sa = math.cos(attack_deg * d2r), math.sin(attack_deg * d2r)
    cp, sp = math.cos(path_deg * d2r), math.sin(path_deg * d2r)
    direction = np.array([ca * cp, -ca * sp, sa])
    # face: aimed face_deg right of target, lofted back by loft_deg
    cf, sf = math.cos(face_deg * d2r), math.sin(face_deg * d2r)
    cl, sl = math.cos(loft_deg * d2r), math.sin(loft_deg * d2r)
    normal = np.array([cl * cf, -cl * sf, sl])
    return Impact(club_velocity=direction * speed, face_normal=normal,
                  face_offset=np.array(offset, dtype=float))


def benchmarks() -> None:
    """Print the reference shots the constants above were calibrated against."""
    rows = [
        ("tour driver", square_impact(50.0, attack_deg=2.0)),
        ("good amateur", square_impact(45.0, attack_deg=0.0)),
        ("slower swing", square_impact(38.0, attack_deg=-1.0)),
        ("15 mm toe", square_impact(45.0, offset=(0.015, 0.0))),
        ("15 mm heel", square_impact(45.0, offset=(-0.015, 0.0))),
        ("face 4 open", square_impact(45.0, face_deg=4.0)),
        ("out-to-in 5", square_impact(45.0, face_deg=0.0, path_deg=-5.0)),
    ]
    print(f"{'shot':<15}{'club':>6}{'ball':>7}{'smash':>7}{'launch':>8}"
          f"{'spin':>8}{'axis':>7}{'carry':>7}{'offline':>8}")
    for name, imp in rows:
        launch, flight = shot(imp)
        print(f"{name:<15}{imp.clubhead_speed:6.1f}{launch.speed:7.1f}"
              f"{launch.smash:7.2f}{launch.launch_angle:8.1f}"
              f"{launch.backspin_rpm:8.0f}{launch.spin_axis:7.1f}"
              f"{flight.carry:7.1f}{flight.lateral:8.1f}")


if __name__ == "__main__":
    benchmarks()

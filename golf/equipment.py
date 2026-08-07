"""The club, the ball and the tee."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Club:
    """A driver.

    `grip_angle_deg` is the angle built into the grip between the hand's long
    axis and the shaft.  Real wrists only deviate ~20 deg; the rest of the
    80-90 deg "wrist cock" seen at the top of a swing comes from the shaft
    sitting diagonally across the palm, and this is how that is modelled.
    """

    length: float = 1.14            # butt end -> sole, 45 in
    head_mass: float = 0.200        # kg
    shaft_mass: float = 0.065
    grip_mass: float = 0.050
    loft_deg: float = 10.5
    grip_angle_deg: float = 35.0
    lead_hand_drop: float = 0.085   # lead hand this far below the butt
    trail_hand_drop: float = 0.155  # trail hand below that (overlap grip)

    @property
    def total_mass(self) -> float:
        return self.head_mass + self.shaft_mass + self.grip_mass


BALL_RADIUS = 0.02135   # m, R&A/USGA minimum diameter 42.67 mm
BALL_MASS = 0.0459      # kg, maximum legal mass
TEE_HEIGHT = 0.030      # ball sits this high before adding the radius
BALL_FORWARD = 0.10     # driver ball position, forward of stance centre (m)

#: Height of the ball's centre when it is teed up.
BALL_CENTRE_HEIGHT = TEE_HEIGHT + BALL_RADIUS

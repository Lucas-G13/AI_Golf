"""How big the golfer is.

Segment lengths and masses come from Winter (2009) as fractions of standing
height H and body mass M.  Everything downstream -- the MJCF, the address pose,
the swing plan -- scales off this one object, so a different golfer is a single
constructor argument.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anthropometry:
    height: float = 1.78          # m
    mass: float = 78.0            # kg
    handedness: str = "right"     # "right" -> lead side is the left side

    # ---- lengths ----------------------------------------------------------
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

    # ---- masses -----------------------------------------------------------
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

    # ---- handedness -------------------------------------------------------
    @property
    def lead(self) -> str:
        """'l' or 'r' -- the side nearer the target, which grips the top of
        the club and whose arm stays straight through the swing."""
        return "l" if self.handedness == "right" else "r"

    @property
    def trail(self) -> str:
        return "r" if self.handedness == "right" else "l"

    @property
    def lead_sign(self) -> float:
        """+1 if the lead side is the golfer's left (right-handed player).

        Multiplying a body-relative quantity by this converts it to a
        target-relative one, which is how the trunk joints and the swing script
        stay handedness-agnostic.
        """
        return 1.0 if self.handedness == "right" else -1.0

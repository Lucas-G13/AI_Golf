"""What gets measured.

`TRACKED` is the contract between the model builder (which creates a site of
each name) and `SwingTracker` (which reads them).  Adding a landmark means
adding a site in `golf.model` and a name here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

#: Every major joint centre, plus the club and the ball.  Order is fixed: it is
#: the row order of every array `SwingTracker` returns.
TRACKED: Tuple[str, ...] = (
    "pelvis", "lumbar", "thorax", "neck", "head",
    "hip_l", "knee_l", "ankle_l", "toe_l",
    "hip_r", "knee_r", "ankle_r", "toe_r",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "grip", "clubhead", "ball",
)

_CHAIN: Dict[str, Optional[str]] = {
    "pelvis": None,
    "lumbar": "pelvis", "thorax": "lumbar", "neck": "thorax", "head": "neck",
    "hip_l": "pelvis", "knee_l": "hip_l", "ankle_l": "knee_l", "toe_l": "ankle_l",
    "hip_r": "pelvis", "knee_r": "hip_r", "ankle_r": "knee_r", "toe_r": "ankle_r",
    "shoulder_l": "thorax", "elbow_l": "shoulder_l",
    "wrist_l": "elbow_l", "hand_l": "wrist_l",
    "shoulder_r": "thorax", "elbow_r": "shoulder_r",
    "wrist_r": "elbow_r", "hand_r": "wrist_r",
    "grip": None,          # depends on handedness -- see parent_map
    "clubhead": "grip",
    "ball": None,
}


def parent_map(lead: str) -> Dict[str, Optional[str]]:
    """Parent of each landmark in the kinematic chain, for the segment vectors.

    `lead` is "l" or "r": the club hangs off whichever hand grips the top of
    it, so that one link cannot be written down until handedness is known.
    """
    chain = dict(_CHAIN)
    chain["grip"] = f"hand_{lead}"
    return chain

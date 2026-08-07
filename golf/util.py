"""Small shared helpers: angle units, quaternions, XML formatting, splines."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

DEG = math.pi / 180.0
RAD = 180.0 / math.pi


def quat_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation of `angle` rad about `axis`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def fmt_vec(v: Sequence[float]) -> str:
    """Format a vector the way MJCF attributes want it."""
    return " ".join(f"{x:.6g}" for x in v)


def hermite_basis(u: float) -> Tuple[float, float, float, float]:
    """Cubic Hermite basis (position form) at u in [0, 1] -> (p0, m0, p1, m1)."""
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1,      # p0
            u3 - 2 * u2 + u,          # m0
            -2 * u3 + 3 * u2,         # p1
            u3 - u2)                  # m1

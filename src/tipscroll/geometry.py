from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


Point3 = Sequence[float]


def joint_angle_degrees(point_a: Point3, vertex: Point3, point_c: Point3) -> float:
    """Return the smaller 3D angle A-vertex-C in degrees."""

    a = np.asarray(point_a, dtype=float)
    b = np.asarray(vertex, dtype=float)
    c = np.asarray(point_c, dtype=float)
    first = a - b
    second = c - b
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return math.nan
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def index_finger_angles(points: Sequence[Point3]) -> tuple[float, float]:
    """Calculate PIP and DIP angles from MCP/PIP/DIP/TIP points."""

    if len(points) != 4:
        raise ValueError("Expected MCP, PIP, DIP and TIP points")
    mcp, pip, dip, tip = points
    return (
        joint_angle_degrees(mcp, pip, dip),
        joint_angle_degrees(pip, dip, tip),
    )


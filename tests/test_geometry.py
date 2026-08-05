from __future__ import annotations

import math

import numpy as np
import pytest

from tipscroll.geometry import index_finger_angles, joint_angle_degrees


def _rotate_z(point: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    angle = math.radians(degrees)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = rotation @ np.asarray(point)
    return tuple(float(value) for value in result)


@pytest.mark.parametrize("rotation", [20.0, 35.0, 50.0, 65.0])
def test_straight_finger_angles_are_rotation_invariant(rotation: float) -> None:
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    rotated = [_rotate_z(point, rotation) for point in points]

    pip, dip = index_finger_angles(rotated)

    assert pip == pytest.approx(180.0)
    assert dip == pytest.approx(180.0)


def test_bent_finger_has_small_pip_angle() -> None:
    pip, dip = index_finger_angles(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0)]
    )

    assert pip == pytest.approx(90.0)
    assert dip == pytest.approx(180.0)


def test_zero_length_joint_returns_nan() -> None:
    assert math.isnan(joint_angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0)))


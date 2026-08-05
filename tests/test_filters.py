from __future__ import annotations

import math

import pytest

from tipscroll.filters import OneEuroFilter, PassThroughFilter, smoothing_alpha


def _run(filter_: OneEuroFilter, samples: list[float], dt: float) -> list[float]:
    return [filter_.update(value, dt) for value in samples]


def test_alpha_is_frame_rate_independent() -> None:
    # Two frames at 60Hz must smooth as much as one frame at 30Hz.
    fast = smoothing_alpha(2.0, 1.0 / 60.0)
    slow = smoothing_alpha(2.0, 1.0 / 30.0)

    remaining_after_two_fast = (1.0 - fast) ** 2
    remaining_after_one_slow = 1.0 - slow

    assert remaining_after_two_fast == pytest.approx(remaining_after_one_slow, rel=1e-12)


def test_alpha_rejects_non_positive_arguments() -> None:
    with pytest.raises(ValueError):
        smoothing_alpha(0.0, 0.01)
    with pytest.raises(ValueError):
        smoothing_alpha(1.0, 0.0)


def test_first_sample_passes_through_unchanged() -> None:
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=4.0, derivative_cutoff_hz=1.0)

    assert filter_.update(0.5, 1.0 / 30.0) == 0.5
    assert filter_.value == 0.5


def test_resting_jitter_is_attenuated() -> None:
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=4.0, derivative_cutoff_hz=1.0)
    dt = 1.0 / 30.0
    jitter = [0.5 + 0.004 * math.sin(index * 2.4) for index in range(120)]

    output = _run(filter_, jitter, dt)

    raw_span = max(jitter[30:]) - min(jitter[30:])
    filtered_span = max(output[30:]) - min(output[30:])
    assert filtered_span < raw_span * 0.25


def test_deliberate_motion_keeps_low_lag() -> None:
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=4.0, derivative_cutoff_hz=1.0)
    dt = 1.0 / 30.0
    # A steady 1.0 unit/second sweep, the speed of an intentional move.
    ramp = [0.5 + index * dt for index in range(30)]

    output = _run(filter_, ramp, dt)

    lag = ramp[-1] - output[-1]
    assert lag < 0.05


def test_speed_adaptive_cutoff_beats_a_fixed_low_cutoff() -> None:
    dt = 1.0 / 30.0
    ramp = [0.5 + index * dt for index in range(30)]
    adaptive = OneEuroFilter(min_cutoff_hz=1.0, beta=4.0, derivative_cutoff_hz=1.0)
    fixed = OneEuroFilter(min_cutoff_hz=1.0, beta=0.0, derivative_cutoff_hz=1.0)

    adaptive_lag = ramp[-1] - _run(adaptive, ramp, dt)[-1]
    fixed_lag = ramp[-1] - _run(fixed, ramp, dt)[-1]

    assert adaptive_lag < fixed_lag


def test_reset_clears_history() -> None:
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=4.0, derivative_cutoff_hz=1.0)
    _run(filter_, [0.5] * 10, 1.0 / 30.0)

    filter_.reset(0.8)

    assert filter_.value == 0.8
    assert filter_.update(0.9, 1.0 / 30.0) < 0.9


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff_hz=0.0, beta=1.0, derivative_cutoff_hz=1.0)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff_hz=1.0, beta=-1.0, derivative_cutoff_hz=1.0)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff_hz=1.0, beta=1.0, derivative_cutoff_hz=0.0)


def test_pass_through_filter_is_transparent() -> None:
    filter_ = PassThroughFilter()

    assert filter_.update(0.42, 1.0 / 30.0) == 0.42
    assert filter_.value == 0.42
    filter_.reset(None)
    assert filter_.value is None

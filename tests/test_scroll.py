from __future__ import annotations

import pytest

from tipscroll.scroll import WHEEL_DELTA, RecordingScrollSink, ScrollAccumulator


def test_fractional_scroll_accumulates_without_loss() -> None:
    accumulator = ScrollAccumulator(WHEEL_DELTA)

    assert accumulator.advance(2.5, 0.2) == 0
    assert accumulator.advance(2.5, 0.2) == WHEEL_DELTA
    assert accumulator.value == pytest.approx(0.0)


def test_negative_scroll_accumulates_symmetrically() -> None:
    accumulator = ScrollAccumulator(WHEEL_DELTA)

    assert accumulator.advance(-2.5, 0.2) == 0
    assert accumulator.advance(-2.5, 0.2) == -WHEEL_DELTA
    assert accumulator.value == pytest.approx(0.0)


def test_reset_discards_fractional_remainder() -> None:
    accumulator = ScrollAccumulator(WHEEL_DELTA)
    accumulator.advance(3.0, 0.2)

    accumulator.reset()

    assert accumulator.value == 0.0


def test_high_resolution_emits_sub_notch_deltas() -> None:
    accumulator = ScrollAccumulator(1)

    # A rate that owes far less than one notch per tick still moves every tick.
    assert accumulator.advance(2.5, 1.0 / 60.0) == 5
    assert accumulator.advance(-2.5, 1.0 / 60.0) == -5


def test_low_rate_stalls_without_high_resolution() -> None:
    # The rate produced just past the stop threshold; one notch is owed only
    # every few seconds, so whole-notch output looks frozen.
    slow_rate = 0.4
    tick = 1.0 / 60.0
    legacy = ScrollAccumulator(WHEEL_DELTA)
    high_resolution = ScrollAccumulator(1)

    legacy_events = sum(1 for _ in range(60) if legacy.advance(slow_rate, tick))
    high_resolution_events = sum(1 for _ in range(60) if high_resolution.advance(slow_rate, tick))

    assert legacy_events == 0
    assert high_resolution_events >= 40


def test_quantum_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ScrollAccumulator(0)


def test_emitted_deltas_sum_to_the_commanded_distance() -> None:
    accumulator = ScrollAccumulator(1)
    tick = 1.0 / 60.0

    total = sum(accumulator.advance(6.0, tick) for _ in range(60))

    # One second at 6 notches per second, minus whatever is still owed.
    assert total + round(accumulator.value) == 6 * WHEEL_DELTA


def test_recording_sink_never_injects_os_input() -> None:
    sink = RecordingScrollSink()
    sink.emit_wheel_delta(240)
    sink.emit_wheel_delta(-30)

    assert sink.events == [240, -30]

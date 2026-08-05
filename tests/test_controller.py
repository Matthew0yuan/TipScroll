from __future__ import annotations

from dataclasses import replace

import pytest

from tipscroll.config import AppConfig
from tipscroll.controller import TipScrollController
from tipscroll.domain import AppState, StopReason, TipObservation


def _observation(
    timestamp_ms: int,
    y: float = 0.5,
    *,
    x: float = 0.5,
    visible: bool = True,
    pip: float = 175.0,
    dip: float = 175.0,
) -> TipObservation:
    return TipObservation(
        timestamp_ms=timestamp_ms,
        frame_id=timestamp_ms,
        visible=visible,
        tip_x=x if visible else None,
        tip_y=y if visible else None,
        pip_angle_deg=pip if visible else None,
        dip_angle_deg=dip if visible else None,
    )


def _active_controller(**changes: object) -> TipScrollController:
    config = replace(AppConfig(), smoothing_enabled=False, **changes)
    controller = TipScrollController(config)
    for timestamp in (0, 50, 100, 150, 200):
        snapshot = controller.process(_observation(timestamp))
    assert snapshot.state is AppState.ACTIVE
    return controller


def test_arming_requires_continuous_200ms_stability() -> None:
    controller = TipScrollController(AppConfig())

    for timestamp in (0, 50, 100, 150):
        snapshot = controller.process(_observation(timestamp))
        assert snapshot.state is AppState.ARMING

    snapshot = controller.process(_observation(200))
    assert snapshot.state is AppState.ACTIVE
    assert snapshot.anchor_y == pytest.approx(0.5)


def test_motion_resets_arming_timer() -> None:
    controller = TipScrollController(AppConfig())
    controller.process(_observation(0, 0.50))
    controller.process(_observation(100, 0.56))

    for timestamp in (150, 200, 250):
        assert controller.process(_observation(timestamp, 0.56)).state is AppState.ARMING
    assert controller.process(_observation(300, 0.56)).state is AppState.ACTIVE


def _time_to_arm(
    ambiguous: frozenset[int] = frozenset(),
    *,
    interval_ms: int = 33,
    frames: int = 60,
) -> int | None:
    """Feed a steady stream and return when the dwell first completes."""

    controller = TipScrollController(AppConfig())
    for index in range(frames):
        timestamp = index * interval_ms
        pip = 145.0 if index in ambiguous else 175.0
        snapshot = controller.process(_observation(timestamp, pip=pip))
        if snapshot.state is AppState.ACTIVE:
            return timestamp
    return None


def test_single_frame_pose_dropout_costs_one_frame_not_a_new_dwell() -> None:
    # Landmark noise makes one-frame ambiguous poses common. Before the grace
    # period each one cost a fresh 0.2 second dwell, which is what made arming
    # feel impossible.
    clean = _time_to_arm()
    dropped = _time_to_arm(frozenset({3}))

    assert clean is not None and dropped is not None
    assert dropped - clean == 33
    assert dropped < clean + AppConfig().arm_duration_ms


def test_pose_dropout_extends_the_dwell_by_its_own_duration() -> None:
    clean = _time_to_arm()
    dropped = _time_to_arm(frozenset({3, 4, 5}))

    assert clean is not None and dropped is not None
    # Three untrusted frames span 99ms, so the dwell moves out by about that.
    assert 90 <= dropped - clean <= 132


def test_dropout_holds_progress_rather_than_zeroing_it() -> None:
    controller = TipScrollController(AppConfig())
    for timestamp in (0, 33, 66):
        assert controller.process(_observation(timestamp)).state is AppState.ARMING
    progress_before = controller.snapshot().arming_progress
    assert progress_before > 0.0

    held = controller.process(_observation(99, pip=145.0))

    assert held.state is AppState.ARMING
    assert held.arming_progress == progress_before


def test_sustained_ambiguous_pose_still_stops_the_dwell() -> None:
    controller = TipScrollController(AppConfig())
    controller.process(_observation(0))

    for timestamp in (33, 66, 99, 132, 165, 198, 231, 264):
        snapshot = controller.process(_observation(timestamp, pip=145.0))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.POSE_AMBIGUOUS


def test_transient_fault_stops_output_at_once_but_keeps_the_anchor() -> None:
    controller = _active_controller(commit_delay_ms=0)
    assert controller.process(_observation(250, 0.60)).committed_rate != 0.0
    anchor = controller.snapshot().anchor_y

    held = controller.process(_observation(300, 0.60, pip=145.0))

    assert held.state is AppState.HOLD
    assert held.stop_reason is StopReason.POSE_AMBIGUOUS
    assert held.desired_rate == 0.0
    assert held.committed_rate == 0.0
    assert held.anchor_y == anchor


def test_recovered_fault_resumes_without_a_new_dwell() -> None:
    controller = _active_controller(commit_delay_ms=0)
    anchor = controller.snapshot().anchor_y
    assert controller.process(_observation(250, 0.60, pip=145.0)).state is AppState.HOLD

    resumed = controller.process(_observation(283, 0.60))

    assert resumed.state is AppState.ACTIVE
    assert resumed.anchor_y == anchor


def test_resumed_session_requires_the_hysteresis_to_retrigger() -> None:
    controller = _active_controller(commit_delay_ms=0)
    controller.process(_observation(250, 0.60, pip=145.0))

    # Back inside the neutral band: recovering must not replay a stale rate.
    resumed = controller.process(_observation(283, 0.505))

    assert resumed.state is AppState.ACTIVE
    assert not resumed.scrolling
    assert resumed.committed_rate == 0.0


def test_fault_outlasting_the_grace_clears_the_anchor() -> None:
    controller = _active_controller(commit_delay_ms=0)
    grace = AppConfig().hold_grace_ms

    for timestamp in range(250, 250 + grace + 100, 33):
        snapshot = controller.process(_observation(timestamp, 0.60, pip=145.0))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.POSE_AMBIGUOUS
    assert snapshot.anchor_y is None


def test_hold_never_emits_while_waiting() -> None:
    controller = _active_controller(commit_delay_ms=0)
    controller.process(_observation(250, 0.70))

    for timestamp in (283, 316, 349):
        snapshot = controller.process(_observation(timestamp, 0.70, pip=145.0))
        assert snapshot.state is AppState.HOLD
        assert snapshot.desired_rate == 0.0
        assert snapshot.committed_rate == 0.0


def test_curling_during_the_dwell_stops_without_grace() -> None:
    controller = TipScrollController(AppConfig())
    controller.process(_observation(0))
    controller.process(_observation(33))

    snapshot = controller.process(_observation(66, pip=130.0))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.FINGER_CURLED


def test_repeated_dropouts_cannot_extend_the_dwell_without_bound() -> None:
    # Every other frame ambiguous: the dwell must still complete, capped by the
    # grace budget rather than being pushed out forever.
    config = AppConfig()
    armed_at = _time_to_arm(frozenset(range(1, 60, 2)))

    assert armed_at is not None
    assert armed_at <= config.arm_duration_ms + config.arm_pose_grace_ms + 66


def test_outside_activation_zone_does_not_arm() -> None:
    controller = TipScrollController(AppConfig())

    snapshot = controller.process(_observation(0, 0.10))

    assert snapshot.state is AppState.IDLE
    assert snapshot.stop_reason is StopReason.OUTSIDE_ARM_ZONE


def test_rate_is_monotonic_and_direction_is_correct() -> None:
    controller = _active_controller(commit_delay_ms=0)

    slow_up = controller.process(_observation(250, 0.46)).desired_rate
    fast_up = controller.process(_observation(300, 0.35)).desired_rate
    controller.process(_observation(350, visible=False))
    for timestamp in (400, 450, 500, 550, 600):
        controller.process(_observation(timestamp, 0.5))
    slow_down = controller.process(_observation(650, 0.54)).desired_rate
    fast_down = controller.process(_observation(700, 0.65)).desired_rate

    assert 0.0 < slow_up < fast_up
    assert fast_down < slow_down < 0.0
    assert fast_up <= AppConfig().max_scroll_notches_per_second
    assert abs(fast_down) <= AppConfig().max_scroll_notches_per_second


def test_stop_zone_hysteresis_prevents_boundary_chatter() -> None:
    controller = _active_controller(commit_delay_ms=0)

    assert controller.process(_observation(250, 0.54)).scrolling
    assert controller.process(_observation(300, 0.525)).scrolling
    stopped = controller.process(_observation(350, 0.519))

    assert not stopped.scrolling
    assert stopped.desired_rate == 0.0


def test_commit_buffer_delays_then_commits_rate() -> None:
    controller = _active_controller(commit_delay_ms=50)

    first = controller.process(_observation(250, 0.60))
    second = controller.process(_observation(280, 0.60))
    committed = controller.process(_observation(300, 0.60))

    assert first.desired_rate < 0.0
    assert first.committed_rate == 0.0
    assert second.committed_rate == 0.0
    assert committed.committed_rate < 0.0


@pytest.mark.parametrize(
    ("pip", "dip", "reason"),
    [
        (130.0, 175.0, StopReason.FINGER_CURLED),
        (145.0, 175.0, StopReason.POSE_AMBIGUOUS),
        (175.0, 115.0, StopReason.FINGER_CURLED),
        (175.0, 130.0, StopReason.POSE_AMBIGUOUS),
    ],
)
def test_invalid_pose_stops_and_clears_output(
    pip: float, dip: float, reason: StopReason
) -> None:
    controller = _active_controller(commit_delay_ms=0)
    assert controller.process(_observation(250, 0.60)).committed_rate != 0.0

    snapshot = controller.process(_observation(300, 0.60, pip=pip, dip=dip))

    assert snapshot.state is not AppState.ACTIVE
    assert snapshot.stop_reason is reason
    assert snapshot.desired_rate == 0.0
    assert snapshot.committed_rate == 0.0


def test_ipad_angle_dip_value_can_arm_when_pip_is_straight() -> None:
    controller = TipScrollController(AppConfig())

    for timestamp in (0, 50, 100, 150, 200):
        snapshot = controller.process(_observation(timestamp, pip=170.0, dip=150.0))

    assert snapshot.state is AppState.ACTIVE


def test_tip_loss_stops_output_immediately() -> None:
    controller = _active_controller(commit_delay_ms=0)
    assert controller.process(_observation(250, 0.60)).committed_rate != 0.0

    snapshot = controller.process(_observation(300, visible=False))

    assert snapshot.state is AppState.HOLD
    assert snapshot.stop_reason is StopReason.TIP_LOST
    assert snapshot.desired_rate == 0.0
    assert snapshot.committed_rate == 0.0


def test_stale_watchdog_stops_output_at_once() -> None:
    controller = _active_controller()

    snapshot = controller.watchdog(281)

    assert snapshot.state is AppState.HOLD
    assert snapshot.stop_reason is StopReason.STALE_RESULT
    assert snapshot.committed_rate == 0.0


def test_stale_watchdog_clears_the_anchor_once_the_gap_is_real() -> None:
    controller = _active_controller()

    controller.watchdog(281)
    snapshot = controller.watchdog(281 + AppConfig().hold_grace_ms + 1)

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.STALE_RESULT
    assert snapshot.anchor_y is None
    assert snapshot.committed_rate == 0.0


def test_already_stale_observation_is_rejected_immediately() -> None:
    controller = _active_controller()

    snapshot = controller.process(
        TipObservation(
            timestamp_ms=250,
            frame_id=250,
            visible=True,
            tip_x=0.5,
            tip_y=0.6,
            pip_angle_deg=175.0,
            dip_angle_deg=175.0,
            result_age_ms=81.0,
        )
    )

    assert snapshot.state is AppState.HOLD
    assert snapshot.stop_reason is StopReason.STALE_RESULT
    assert snapshot.committed_rate == 0.0


def test_track_jump_stops_controller() -> None:
    controller = _active_controller()

    snapshot = controller.process(_observation(250, 0.90, x=0.90))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.TRACK_JUMP


def test_timestamp_must_increase() -> None:
    controller = _active_controller()

    snapshot = controller.process(_observation(200, 0.5))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.TIMESTAMP_ERROR


def test_safe_stop_requires_a_new_arming_dwell() -> None:
    controller = _active_controller()
    # A loss that outlasts the grace is a real loss, not a dropped frame.
    for timestamp in range(250, 250 + AppConfig().hold_grace_ms + 100, 33):
        lost = controller.process(_observation(timestamp, visible=False))
    assert lost.state is AppState.SAFE_STOP

    start = timestamp + 50
    for offset in (0, 50, 100, 150):
        assert (
            controller.process(_observation(start + offset, 0.55)).state is AppState.ARMING
        )
    snapshot = controller.process(_observation(start + 200, 0.55))

    assert snapshot.state is AppState.ACTIVE
    assert snapshot.anchor_y == pytest.approx(0.55)


def test_user_abort_is_latched_against_late_camera_callbacks() -> None:
    controller = _active_controller()
    controller.emergency_stop(StopReason.USER_ABORT)

    for timestamp in (250, 300, 350, 400, 450, 500):
        snapshot = controller.process(_observation(timestamp, 0.55))

    assert snapshot.state is AppState.SAFE_STOP
    assert snapshot.stop_reason is StopReason.USER_ABORT
    assert snapshot.committed_rate == 0.0


def _arm_smoothed(frame_interval_ms: int) -> tuple[TipScrollController, int]:
    controller = TipScrollController(replace(AppConfig(), commit_delay_ms=0))
    timestamp = 0
    while timestamp <= 400:
        snapshot = controller.process(_observation(timestamp))
        timestamp += frame_interval_ms
    assert snapshot.state is AppState.ACTIVE
    return controller, timestamp


def test_smoothing_tracks_the_same_path_at_any_frame_rate() -> None:
    # The same 0.5 units/second sweep sampled at 30fps and at 60fps must land
    # on the same offset; a fixed EMA alpha would smooth twice as hard at 60fps.
    offsets = []
    for frame_interval_ms in (33, 16):
        controller, timestamp = _arm_smoothed(frame_interval_ms)
        start_ms = timestamp
        while timestamp <= start_ms + 600:
            tip_y = 0.5 + 0.5 * (timestamp - start_ms) / 1000.0
            snapshot = controller.process(_observation(timestamp, tip_y))
            timestamp += frame_interval_ms
        offsets.append(snapshot.offset)

    assert offsets[0] == pytest.approx(offsets[1], abs=0.005)


def test_smoothing_absorbs_a_single_frame_landmark_spike() -> None:
    smoothed, timestamp = _arm_smoothed(33)
    raw = TipScrollController(replace(AppConfig(), commit_delay_ms=0, smoothing_enabled=False))
    for raw_timestamp in range(0, 401, 33):
        raw.process(_observation(raw_timestamp))

    spike = 0.5 + 0.05
    smoothed_snapshot = smoothed.process(_observation(timestamp, spike))
    raw_snapshot = raw.process(_observation(429, spike))

    assert not smoothed_snapshot.scrolling
    assert smoothed_snapshot.desired_rate == 0.0
    assert raw_snapshot.scrolling


def test_five_minutes_of_stationary_synthetic_input_never_scrolls() -> None:
    controller = TipScrollController(replace(AppConfig(), smoothing_enabled=False, commit_delay_ms=0))

    for timestamp in range(0, 300_001, 33):
        snapshot = controller.process(_observation(timestamp, 0.5))
        assert snapshot.desired_rate == 0.0
        assert snapshot.committed_rate == 0.0

    assert snapshot.state is AppState.ACTIVE

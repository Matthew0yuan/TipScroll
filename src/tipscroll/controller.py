from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass

from .config import AppConfig
from .domain import AppState, ControllerSnapshot, StopReason, TipObservation
from .filters import OneEuroFilter, PassThroughFilter

# A result gap longer than this is treated as a restart of the filter clock
# rather than a genuine sample interval. The freshness watchdog normally stops
# the session well before this, so the clamp only guards arithmetic.
_MAX_FILTER_DT_SECONDS = 0.25
_MIN_FILTER_DT_SECONDS = 0.001


@dataclass(frozen=True, slots=True)
class _PendingRate:
    timestamp_ms: int
    rate: float


def _build_filter(config: AppConfig) -> OneEuroFilter | PassThroughFilter:
    if not config.smoothing_enabled:
        return PassThroughFilter()
    return OneEuroFilter(
        min_cutoff_hz=config.smoothing_min_cutoff_hz,
        beta=config.smoothing_beta,
        derivative_cutoff_hz=config.smoothing_derivative_cutoff_hz,
    )


class TipScrollController:
    """Thread-safe state machine and rate controller.

    Only the fingertip Y coordinate contributes to scroll rate. Index-finger
    joint angles are used solely as an activation and release gate.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._state = AppState.IDLE
        self._stop_reason = StopReason.NONE
        self._last_result_timestamp_ms: int | None = None
        self._last_frame_id: int | None = None
        self._last_tip_x: float | None = None
        self._last_tip_y: float | None = None
        self._raw_tip_x: float | None = None
        self._raw_tip_y: float | None = None
        self._filter = _build_filter(config)
        self._filter_timestamp_ms: int | None = None
        self._pip_angle_deg: float | None = None
        self._dip_angle_deg: float | None = None
        self._anchor_y: float | None = None
        self._offset = 0.0
        self._desired_rate = 0.0
        self._committed_rate = 0.0
        self._scrolling = False
        self._pending_rates: deque[_PendingRate] = deque()
        self._arming_started_ms: int | None = None
        self._arming_samples: list[float] = []
        self._arming_progress = 0.0
        self._arming_hold_started_ms: int | None = None
        self._arming_hold_ms = 0
        self._hold_started_ms: int | None = None
        self._result_age_ms = 0.0
        self._abort_latched = False

    def process(self, observation: TipObservation) -> ControllerSnapshot:
        with self._lock:
            self._last_frame_id = observation.frame_id
            self._raw_tip_x = observation.tip_x
            self._raw_tip_y = observation.tip_y
            self._pip_angle_deg = observation.pip_angle_deg
            self._dip_angle_deg = observation.dip_angle_deg
            self._result_age_ms = max(0.0, observation.result_age_ms)

            if self._abort_latched:
                self._safe_stop(self._stop_reason)
                return self._snapshot_unlocked()

            if (
                self._last_result_timestamp_ms is not None
                and observation.timestamp_ms <= self._last_result_timestamp_ms
            ):
                self._safe_stop(StopReason.TIMESTAMP_ERROR)
                return self._snapshot_unlocked()

            self._last_result_timestamp_ms = observation.timestamp_ms

            if observation.result_age_ms > self.config.stale_result_ms:
                self._transient_fault(StopReason.STALE_RESULT, observation.timestamp_ms)
                return self._snapshot_unlocked()

            if not observation.visible:
                self._transient_fault(StopReason.TIP_LOST, observation.timestamp_ms)
                return self._snapshot_unlocked()

            if not self._valid_coordinate(observation.tip_x) or not self._valid_coordinate(
                observation.tip_y
            ):
                self._transient_fault(StopReason.POSE_AMBIGUOUS, observation.timestamp_ms)
                return self._snapshot_unlocked()

            tip_x = float(observation.tip_x)
            tip_y = float(observation.tip_y)
            if self._is_track_jump(tip_x, tip_y):
                self._safe_stop(StopReason.TRACK_JUMP)
                return self._snapshot_unlocked()

            self._last_tip_x = tip_x
            self._last_tip_y = tip_y

            pose_reason = self._pose_stop_reason(
                observation.pip_angle_deg, observation.dip_angle_deg
            )
            if pose_reason is not None:
                if pose_reason is StopReason.FINGER_CURLED:
                    self._safe_stop(pose_reason)
                else:
                    self._transient_fault(pose_reason, observation.timestamp_ms)
                return self._snapshot_unlocked()
            self._end_arming_hold(observation.timestamp_ms)

            if self._state is AppState.HOLD:
                self._resume_from_hold(tip_y, observation.timestamp_ms)

            if self._state is AppState.ACTIVE:
                self._update_active(tip_y, observation.timestamp_ms)
            else:
                self._update_arming(tip_y, observation.timestamp_ms)

            return self._snapshot_unlocked()

    def watchdog(self, now_ms: int) -> ControllerSnapshot:
        with self._lock:
            if self._last_result_timestamp_ms is None:
                self._result_age_ms = 0.0
                return self._snapshot_unlocked()

            self._result_age_ms = max(0.0, float(now_ms - self._last_result_timestamp_ms))
            if self._result_age_ms > self.config.stale_result_ms and self._state in (
                AppState.ARMING,
                AppState.ACTIVE,
                AppState.HOLD,
            ):
                self._transient_fault(StopReason.STALE_RESULT, now_ms)
            return self._snapshot_unlocked()

    def emergency_stop(self, reason: StopReason = StopReason.USER_ABORT) -> ControllerSnapshot:
        with self._lock:
            if reason in (StopReason.USER_ABORT, StopReason.OUTPUT_ERROR):
                self._abort_latched = True
            self._safe_stop(reason)
            return self._snapshot_unlocked()

    def snapshot(self, now_ms: int | None = None) -> ControllerSnapshot:
        if now_ms is not None:
            return self.watchdog(now_ms)
        with self._lock:
            return self._snapshot_unlocked()

    @staticmethod
    def _valid_coordinate(value: float | None) -> bool:
        return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0

    def _is_track_jump(self, tip_x: float, tip_y: float) -> bool:
        if self._last_tip_x is None or self._last_tip_y is None:
            return False
        distance = math.hypot(tip_x - self._last_tip_x, tip_y - self._last_tip_y)
        return distance > self.config.max_track_jump

    def _pose_stop_reason(
        self, pip_angle_deg: float | None, dip_angle_deg: float | None
    ) -> StopReason | None:
        if (
            pip_angle_deg is None
            or dip_angle_deg is None
            or not math.isfinite(pip_angle_deg)
            or not math.isfinite(dip_angle_deg)
        ):
            return StopReason.POSE_AMBIGUOUS
        if (
            pip_angle_deg <= self.config.pip_curled_angle_deg
            or dip_angle_deg <= self.config.dip_curled_angle_deg
        ):
            return StopReason.FINGER_CURLED
        if (
            pip_angle_deg < self.config.pip_extended_angle_deg
            or dip_angle_deg < self.config.dip_extended_angle_deg
        ):
            return StopReason.POSE_AMBIGUOUS
        return None

    def _transient_fault(self, reason: StopReason, timestamp_ms: int) -> None:
        """Stop output immediately, but do not assume the session is over.

        Output stopping and the anchor being discarded are separate concerns.
        The first is a safety requirement and always happens at once. The
        second buys no safety once nothing is being emitted, and for a rate
        controller it is expensive: the anchor is the whole interaction, so
        dropping it on a one-frame landmark dropout forces a fresh dwell and
        reads as the controller constantly repositioning itself.

        A fault that persists past the grace period is treated as real and
        clears the anchor. A deliberate curl and a track jump never come here.
        """

        if self._state is AppState.ACTIVE:
            self._enter_hold(reason, timestamp_ms)
            return
        if self._state is AppState.HOLD:
            self._stop_reason = reason
            started = self._hold_started_ms
            if started is None or timestamp_ms - started > self.config.hold_grace_ms:
                self._safe_stop(reason)
            return
        if self._state is AppState.ARMING:
            if self._arming_hold_started_ms is None:
                self._arming_hold_started_ms = timestamp_ms
            elif timestamp_ms - self._arming_hold_started_ms > self.config.arm_pose_grace_ms:
                self._safe_stop(reason)
            return
        self._safe_stop(reason)

    def _enter_hold(self, reason: StopReason, timestamp_ms: int) -> None:
        self._state = AppState.HOLD
        self._stop_reason = reason
        self._hold_started_ms = timestamp_ms
        self._desired_rate = 0.0
        self._committed_rate = 0.0
        self._scrolling = False
        self._pending_rates.clear()

    def _resume_from_hold(self, tip_y: float, timestamp_ms: int) -> None:
        if self._anchor_y is None:
            self._safe_stop(StopReason.POSE_AMBIGUOUS)
            return
        self._state = AppState.ACTIVE
        self._stop_reason = StopReason.NONE
        self._hold_started_ms = None
        # The tip may have moved while it was untrusted, so restart the filter
        # where it actually is rather than synthesising a spike from the stale
        # value, and make the hysteresis re-trigger before output resumes.
        self._filter.reset(tip_y)
        self._filter_timestamp_ms = timestamp_ms
        self._scrolling = False
        self._pending_rates.clear()

    def _end_arming_hold(self, timestamp_ms: int) -> None:
        if self._arming_hold_started_ms is None:
            return
        held_ms = timestamp_ms - self._arming_hold_started_ms
        self._arming_hold_started_ms = None
        self._arming_hold_ms = min(
            self.config.arm_pose_grace_ms, self._arming_hold_ms + held_ms
        )

    def _clear_arming_hold(self) -> None:
        self._arming_hold_started_ms = None
        self._arming_hold_ms = 0

    def _update_arming(self, tip_y: float, timestamp_ms: int) -> None:
        if not self.config.arm_zone_min_y <= tip_y <= self.config.arm_zone_max_y:
            self._state = AppState.IDLE
            self._stop_reason = StopReason.OUTSIDE_ARM_ZONE
            self._reset_motion(keep_last_tip=True)
            return

        # The dwell is filtered too. Measuring stillness on the raw landmark
        # conflates hand drift with landmark noise, and noise dominates: it
        # forces a threshold loose enough to also accept real drift, which is
        # exactly the slop the anchor then inherits.
        filtered_tip_y = self._filter.update(tip_y, self._filter_dt_seconds(timestamp_ms))
        self._filter_timestamp_ms = timestamp_ms

        if self._state is not AppState.ARMING or self._arming_started_ms is None:
            self._state = AppState.ARMING
            self._stop_reason = StopReason.NONE
            self._arming_started_ms = timestamp_ms
            self._arming_samples = [filtered_tip_y]
            self._arming_progress = 0.0
            self._clear_arming_hold()
            return

        self._arming_samples.append(filtered_tip_y)
        if max(self._arming_samples) - min(self._arming_samples) > self.config.arm_max_y_span:
            self._arming_started_ms = timestamp_ms
            self._arming_samples = [filtered_tip_y]
            self._arming_progress = 0.0
            self._clear_arming_hold()
            return

        # Frames skipped during a pose dropout are not credited to the dwell,
        # so the deadline moves out by however long the pose was untrusted.
        required_ms = self.config.arm_duration_ms + self._arming_hold_ms
        elapsed_ms = timestamp_ms - self._arming_started_ms
        self._arming_progress = min(1.0, elapsed_ms / required_ms)
        if elapsed_ms < required_ms:
            return

        # The filter has been tracking throughout the dwell, so it carries into
        # the active session unchanged; only the anchor is new.
        anchor = float(statistics.median(self._arming_samples))
        self._state = AppState.ACTIVE
        self._stop_reason = StopReason.NONE
        self._anchor_y = anchor
        self._offset = 0.0
        self._desired_rate = 0.0
        self._committed_rate = 0.0
        self._scrolling = False
        self._pending_rates.clear()
        self._arming_started_ms = None
        self._arming_samples.clear()
        self._arming_progress = 1.0
        self._clear_arming_hold()

    def _update_active(self, tip_y: float, timestamp_ms: int) -> None:
        if self._anchor_y is None:
            self._safe_stop(StopReason.POSE_AMBIGUOUS)
            return

        filtered_tip_y = self._filter.update(tip_y, self._filter_dt_seconds(timestamp_ms))
        self._filter_timestamp_ms = timestamp_ms

        self._offset = filtered_tip_y - self._anchor_y
        magnitude = abs(self._offset)
        if self._scrolling:
            if magnitude < self.config.stop_enter_offset:
                self._scrolling = False
        elif magnitude > self.config.stop_exit_offset:
            self._scrolling = True

        self._desired_rate = self._map_rate(self._offset) if self._scrolling else 0.0
        self._pending_rates.append(_PendingRate(timestamp_ms, self._desired_rate))
        while (
            self._pending_rates
            and timestamp_ms - self._pending_rates[0].timestamp_ms
            >= self.config.commit_delay_ms
        ):
            self._committed_rate = self._pending_rates.popleft().rate

    def _filter_dt_seconds(self, timestamp_ms: int) -> float:
        if self._filter_timestamp_ms is None:
            return 1.0 / self.config.camera_fps
        dt = (timestamp_ms - self._filter_timestamp_ms) / 1000.0
        return min(_MAX_FILTER_DT_SECONDS, max(_MIN_FILTER_DT_SECONDS, dt))

    def _map_rate(self, offset: float) -> float:
        magnitude = abs(offset)
        normalized = (magnitude - self.config.stop_enter_offset) / (
            self.config.fast_offset - self.config.stop_enter_offset
        )
        normalized = max(0.0, min(1.0, normalized))
        rate = self.config.max_scroll_notches_per_second * (
            normalized**self.config.rate_exponent
        )
        if offset > 0.0:
            return -rate
        if offset < 0.0:
            return rate
        return 0.0

    def _safe_stop(self, reason: StopReason) -> None:
        self._state = AppState.SAFE_STOP
        self._stop_reason = reason
        self._reset_motion(keep_last_tip=False)

    def _reset_motion(self, *, keep_last_tip: bool) -> None:
        self._anchor_y = None
        self._filter.reset(None)
        self._filter_timestamp_ms = None
        self._offset = 0.0
        self._desired_rate = 0.0
        self._committed_rate = 0.0
        self._scrolling = False
        self._pending_rates.clear()
        self._arming_started_ms = None
        self._arming_samples.clear()
        self._arming_progress = 0.0
        self._clear_arming_hold()
        self._hold_started_ms = None
        if not keep_last_tip:
            self._last_tip_x = None
            self._last_tip_y = None

    def _snapshot_unlocked(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self._state,
            stop_reason=self._stop_reason,
            timestamp_ms=self._last_result_timestamp_ms,
            frame_id=self._last_frame_id,
            raw_tip_x=self._raw_tip_x,
            raw_tip_y=self._raw_tip_y,
            filtered_tip_y=self._filter.value,
            pip_angle_deg=self._pip_angle_deg,
            dip_angle_deg=self._dip_angle_deg,
            anchor_y=self._anchor_y,
            offset=self._offset,
            desired_rate=self._desired_rate,
            committed_rate=self._committed_rate,
            scrolling=self._scrolling,
            arming_progress=self._arming_progress,
            result_age_ms=self._result_age_ms,
        )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()
    ARMING = auto()
    ACTIVE = auto()
    HOLD = auto()
    SAFE_STOP = auto()


class StopReason(Enum):
    NONE = auto()
    TIP_LOST = auto()
    FINGER_CURLED = auto()
    POSE_AMBIGUOUS = auto()
    STALE_RESULT = auto()
    TRACK_JUMP = auto()
    TIMESTAMP_ERROR = auto()
    OUTSIDE_ARM_ZONE = auto()
    USER_ABORT = auto()
    OUTPUT_ERROR = auto()


@dataclass(frozen=True, slots=True)
class TipObservation:
    timestamp_ms: int
    frame_id: int
    visible: bool
    tip_x: float | None = None
    tip_y: float | None = None
    pip_angle_deg: float | None = None
    dip_angle_deg: float | None = None
    result_age_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    state: AppState
    stop_reason: StopReason
    timestamp_ms: int | None
    frame_id: int | None
    raw_tip_x: float | None
    raw_tip_y: float | None
    filtered_tip_y: float | None
    pip_angle_deg: float | None
    dip_angle_deg: float | None
    anchor_y: float | None
    offset: float
    desired_rate: float
    committed_rate: float
    scrolling: bool
    arming_progress: float
    result_age_ms: float


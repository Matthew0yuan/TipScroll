from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    """All runtime thresholds for the first TipScroll prototype."""

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_fourcc: str = "MJPG"
    mirror_input: bool = True

    hand_detection_confidence: float = 0.6
    hand_presence_confidence: float = 0.6
    hand_tracking_confidence: float = 0.6

    arm_duration_ms: int = 200
    arm_max_y_span: float = 0.015
    arm_zone_min_y: float = 0.15
    arm_zone_max_y: float = 0.85
    arm_pose_grace_ms: int = 200

    pip_extended_angle_deg: float = 150.0
    pip_curled_angle_deg: float = 140.0
    dip_extended_angle_deg: float = 140.0
    dip_curled_angle_deg: float = 120.0
    smoothing_enabled: bool = True
    smoothing_min_cutoff_hz: float = 1.0
    smoothing_beta: float = 4.0
    smoothing_derivative_cutoff_hz: float = 1.0

    stop_enter_offset: float = 0.020
    stop_exit_offset: float = 0.030
    fast_offset: float = 0.15
    rate_exponent: float = 1.5
    max_scroll_notches_per_second: float = 18.0

    commit_delay_ms: int = 50
    hold_grace_ms: int = 200
    stale_result_ms: int = 80
    max_track_jump: float = 0.25
    scroll_output_hz: int = 60
    high_resolution_wheel: bool = True

    ui_update_ms: int = 16
    debug_update_ms: int = 33
    safe_stop_fade_ms: int = 350

    def __post_init__(self) -> None:
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("Camera dimensions and FPS must be positive")
        if self.camera_fourcc and len(self.camera_fourcc) != 4:
            raise ValueError("camera_fourcc must be a 4-character code or empty")
        if self.smoothing_min_cutoff_hz <= 0.0 or self.smoothing_derivative_cutoff_hz <= 0.0:
            raise ValueError("Smoothing cutoffs must be positive")
        if self.smoothing_beta < 0.0:
            raise ValueError("smoothing_beta cannot be negative")
        if not 0.0 <= self.arm_zone_min_y < self.arm_zone_max_y <= 1.0:
            raise ValueError("Activation zone must be within normalized image bounds")
        if not 0.0 <= self.stop_enter_offset < self.stop_exit_offset < self.fast_offset:
            raise ValueError("Scroll zones must be ordered: enter < exit < fast")
        if self.pip_curled_angle_deg >= self.pip_extended_angle_deg:
            raise ValueError("PIP curled threshold must be below its extended threshold")
        if self.dip_curled_angle_deg >= self.dip_extended_angle_deg:
            raise ValueError("DIP curled threshold must be below its extended threshold")
        if self.commit_delay_ms < 0 or self.stale_result_ms <= 0:
            raise ValueError("Timing thresholds must be non-negative")
        if self.arm_pose_grace_ms < 0 or self.hold_grace_ms < 0:
            raise ValueError("Grace periods cannot be negative")

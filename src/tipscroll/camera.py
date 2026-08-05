from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import AppConfig
from .controller import TipScrollController
from .domain import ControllerSnapshot, StopReason, TipObservation
from .geometry import index_finger_angles


@dataclass(frozen=True, slots=True)
class DebugFrame:
    rgb: np.ndarray
    observation: TipObservation
    snapshot: ControllerSnapshot


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """What the driver actually granted, which is often not what was asked."""

    width: int
    height: int
    fps_reported: float
    fourcc: str

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps_reported:.0f}fps "
            f"({self.fourcc or 'unknown'})"
        )


def _decode_fourcc(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    return "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip()


def _encode_fourcc(code: str) -> int:
    if len(code) != 4:
        raise ValueError("A FOURCC must be exactly four characters")
    return (
        ord(code[0])
        | (ord(code[1]) << 8)
        | (ord(code[2]) << 16)
        | (ord(code[3]) << 24)
    )


def _read_camera_info(capture: cv2.VideoCapture) -> CameraInfo:
    return CameraInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps_reported=float(capture.get(cv2.CAP_PROP_FPS)),
        fourcc=_decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
    )


class MeasuredRate:
    """Rolling frame rate of whatever loop calls :meth:`tick`."""

    def __init__(self, window: int = 60) -> None:
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=window)
        self._previous: float | None = None

    def tick(self, now: float) -> None:
        with self._lock:
            if self._previous is not None:
                delta = now - self._previous
                if delta > 0.0:
                    self._samples.append(delta)
            self._previous = now

    def read(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            mean = sum(self._samples) / len(self._samples)
        return 1.0 / mean if mean > 0.0 else 0.0


class LatestDebugFrame:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: DebugFrame | None = None
        self._error: str | None = None

    def update(self, frame: DebugFrame) -> None:
        with self._lock:
            self._frame = frame

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def read(self) -> tuple[DebugFrame | None, str | None]:
        with self._lock:
            return self._frame, self._error


class CameraPipeline:
    """Latest-frame camera capture feeding MediaPipe Live Stream mode."""

    def __init__(
        self,
        config: AppConfig,
        model_path: Path,
        controller: TipScrollController,
        debug_frame: LatestDebugFrame,
        on_snapshot: Callable[[ControllerSnapshot], None] | None = None,
    ) -> None:
        self._config = config
        self._model_path = model_path
        self._controller = controller
        self._debug_frame = debug_frame
        self._on_snapshot = on_snapshot
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._landmarker: mp.tasks.vision.HandLandmarker | None = None
        self._last_submitted_timestamp_ms = -1
        self.info: CameraInfo | None = None
        self.capture_rate = MeasuredRate()
        self.result_rate = MeasuredRate()

    def start(self) -> CameraInfo:
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self._model_path}. "
                "Run: python scripts/download_model.py"
            )
        self._capture = self._open_capture()
        self.info = _read_camera_info(self._capture)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self._config.hand_detection_confidence,
            min_hand_presence_confidence=self._config.hand_presence_confidence,
            min_tracking_confidence=self._config.hand_tracking_confidence,
            result_callback=self._handle_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="tipscroll-camera", daemon=True)
        self._thread.start()
        return self.info

    def _open_capture(self) -> cv2.VideoCapture:
        config = self._config
        capture = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open camera index {config.camera_index}")
        # The pixel format has to be negotiated before the frame size. Left at
        # the default the driver usually picks an uncompressed format whose
        # bandwidth caps 720p near 10fps, which silently triples input latency.
        if config.camera_fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, _encode_fourcc(config.camera_fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _capture_loop(self) -> None:
        capture = self._capture
        if capture is None:
            return

        while not self._stop_event.is_set():
            ok, bgr = capture.read()
            if not ok:
                self._debug_frame.set_error("Camera frame read failed")
                self._stop_event.wait(0.01)
                continue
            self.capture_rate.tick(time.monotonic())
            if self._config.mirror_input:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            if timestamp_ms <= self._last_submitted_timestamp_ms:
                timestamp_ms = self._last_submitted_timestamp_ms + 1
            self._last_submitted_timestamp_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            try:
                if self._landmarker is not None:
                    self._landmarker.detect_async(image, timestamp_ms)
            except Exception as exc:
                self._debug_frame.set_error(f"MediaPipe input failed: {exc}")
                self._controller.emergency_stop(StopReason.TIMESTAMP_ERROR)
                self._stop_event.wait(0.01)

    def _handle_result(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        try:
            callback_ms = time.monotonic_ns() // 1_000_000
            result_age_ms = max(0.0, float(callback_ms - timestamp_ms))
            self.result_rate.tick(time.monotonic())
            if not result.hand_landmarks:
                observation = TipObservation(
                    timestamp_ms=timestamp_ms,
                    frame_id=timestamp_ms,
                    visible=False,
                    result_age_ms=result_age_ms,
                )
            else:
                normalized = result.hand_landmarks[0]
                points_source = (
                    result.hand_world_landmarks[0]
                    if result.hand_world_landmarks
                    else normalized
                )
                points = [
                    (
                        float(points_source[index].x),
                        float(points_source[index].y),
                        float(points_source[index].z),
                    )
                    for index in (5, 6, 7, 8)
                ]
                pip_angle, dip_angle = index_finger_angles(points)
                if not math.isfinite(pip_angle) or not math.isfinite(dip_angle):
                    pip_angle = dip_angle = math.nan
                tip = normalized[8]
                observation = TipObservation(
                    timestamp_ms=timestamp_ms,
                    frame_id=timestamp_ms,
                    visible=True,
                    tip_x=float(tip.x),
                    tip_y=float(tip.y),
                    pip_angle_deg=pip_angle,
                    dip_angle_deg=dip_angle,
                    result_age_ms=result_age_ms,
                )

            snapshot = self._controller.process(observation)
            rgb = np.asarray(output_image.numpy_view()).copy()
            self._debug_frame.update(DebugFrame(rgb, observation, snapshot))
            if self._on_snapshot is not None:
                self._on_snapshot(snapshot)
        except Exception as exc:
            self._debug_frame.set_error(f"MediaPipe result processing failed: {exc}")
            self._controller.emergency_stop(StopReason.POSE_AMBIGUOUS)


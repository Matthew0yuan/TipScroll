from __future__ import annotations

import csv
import queue
import threading
from datetime import datetime
from pathlib import Path

from .domain import ControllerSnapshot


_FIELDS = [
    "event_type",
    "timestamp_ms",
    "frame_id",
    "state",
    "stop_reason",
    "raw_tip_x",
    "raw_tip_y",
    "filtered_tip_y",
    "pip_angle_deg",
    "dip_angle_deg",
    "anchor_y",
    "offset",
    "desired_rate",
    "committed_rate",
    "scrolling",
    "arming_progress",
    "result_age_ms",
    "emitted_wheel_delta",
]


class SessionCsvLogger:
    """Non-blocking, best-effort CSV session logger."""

    def __init__(self, log_directory: Path) -> None:
        self._directory = log_directory
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=10_000)
        self._thread: threading.Thread | None = None
        self.path: Path | None = None
        self.dropped_rows = 0

    def start(self) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = self._directory / f"session-{stamp}.csv"
        self._thread = threading.Thread(target=self._run, name="tipscroll-log", daemon=True)
        self._thread.start()
        return self.path

    def log_snapshot(
        self,
        event_type: str,
        snapshot: ControllerSnapshot,
        emitted_wheel_delta: int = 0,
    ) -> None:
        row: dict[str, object] = {
            "event_type": event_type,
            "timestamp_ms": snapshot.timestamp_ms,
            "frame_id": snapshot.frame_id,
            "state": snapshot.state.name,
            "stop_reason": snapshot.stop_reason.name,
            "raw_tip_x": snapshot.raw_tip_x,
            "raw_tip_y": snapshot.raw_tip_y,
            "filtered_tip_y": snapshot.filtered_tip_y,
            "pip_angle_deg": snapshot.pip_angle_deg,
            "dip_angle_deg": snapshot.dip_angle_deg,
            "anchor_y": snapshot.anchor_y,
            "offset": snapshot.offset,
            "desired_rate": snapshot.desired_rate,
            "committed_rate": snapshot.committed_rate,
            "scrolling": snapshot.scrolling,
            "arming_progress": snapshot.arming_progress,
            "result_age_ms": snapshot.result_age_ms,
            "emitted_wheel_delta": emitted_wheel_delta,
        }
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped_rows += 1

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Make space for the shutdown marker without blocking the caller.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        if self.path is None:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            rows_since_flush = 0
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)
                rows_since_flush += 1
                if rows_since_flush >= 30:
                    handle.flush()
                    rows_since_flush = 0
            handle.flush()


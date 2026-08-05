from __future__ import annotations

import ctypes
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from ctypes import wintypes

from .controller import TipScrollController
from .domain import AppState, ControllerSnapshot, StopReason


WHEEL_DELTA = 120
INPUT_MOUSE = 0
MOUSEEVENTF_WHEEL = 0x0800


class ScrollSink(ABC):
    @abstractmethod
    def emit_wheel_delta(self, delta: int) -> None:
        """Emit a signed vertical wheel delta, where one notch is WHEEL_DELTA."""


class NullScrollSink(ScrollSink):
    def emit_wheel_delta(self, delta: int) -> None:
        del delta


class RecordingScrollSink(ScrollSink):
    def __init__(self) -> None:
        self.events: list[int] = []

    def emit_wheel_delta(self, delta: int) -> None:
        self.events.append(delta)


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("payload",)
    _fields_ = [("type", wintypes.DWORD), ("payload", _InputUnion)]


class WindowsScrollSink(ScrollSink):
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("WindowsScrollSink is only available on Windows")
        self._send_input = ctypes.windll.user32.SendInput
        self._send_input.argtypes = (wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int)
        self._send_input.restype = wintypes.UINT

    def emit_wheel_delta(self, delta: int) -> None:
        if delta == 0:
            return
        wheel_data = delta & 0xFFFFFFFF
        event = _Input(
            type=INPUT_MOUSE,
            mi=_MouseInput(
                dx=0,
                dy=0,
                mouseData=wheel_data,
                dwFlags=MOUSEEVENTF_WHEEL,
                time=0,
                dwExtraInfo=0,
            ),
        )
        sent = self._send_input(1, ctypes.byref(event), ctypes.sizeof(_Input))
        if sent != 1:
            raise ctypes.WinError()


class ScrollAccumulator:
    """Convert a notch-per-second rate into whole wheel units.

    ``quantum`` is the smallest delta the sink may emit. High resolution wheel
    consumers accumulate any multiple of one unit, which keeps low rates smooth
    instead of stalling until a full notch is owed; legacy consumers that drop
    sub-notch deltas need ``quantum=WHEEL_DELTA``.
    """

    def __init__(self, quantum: int = WHEEL_DELTA) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        self.quantum = quantum
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def advance(self, rate: float, elapsed_seconds: float) -> int:
        if elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds cannot be negative")
        self.value += rate * elapsed_seconds * WHEEL_DELTA
        steps = math.trunc(self.value / self.quantum)
        delta = steps * self.quantum
        self.value -= delta
        return delta


class ScrollWorker:
    def __init__(
        self,
        controller: TipScrollController,
        sink: ScrollSink,
        output_hz: int,
        on_emit: Callable[[int, ControllerSnapshot], None] | None = None,
        high_resolution: bool = True,
    ) -> None:
        self._controller = controller
        self._sink = sink
        self._period = 1.0 / output_hz
        self._on_emit = on_emit
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._accumulator = ScrollAccumulator(1 if high_resolution else WHEEL_DELTA)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="tipscroll-output", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._accumulator.reset()

    def _run(self) -> None:
        previous = time.monotonic()
        while not self._stop_event.is_set():
            started = time.monotonic()
            elapsed = min(0.25, max(0.0, started - previous))
            previous = started
            now_ms = time.monotonic_ns() // 1_000_000
            snapshot = self._controller.watchdog(now_ms)

            if snapshot.state is not AppState.ACTIVE or snapshot.committed_rate == 0.0:
                self._accumulator.reset()
            else:
                delta = self._accumulator.advance(snapshot.committed_rate, elapsed)
                if delta:
                    try:
                        self._sink.emit_wheel_delta(delta)
                    except Exception:
                        self._controller.emergency_stop(StopReason.OUTPUT_ERROR)
                        self._accumulator.reset()
                    else:
                        if self._on_emit is not None:
                            self._on_emit(delta, snapshot)

            wait_for = self._period - (time.monotonic() - started)
            self._stop_event.wait(max(0.0, wait_for))


from __future__ import annotations

import ctypes
import os
import threading
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from .camera import CameraPipeline, LatestDebugFrame
from .config import AppConfig
from .controller import TipScrollController
from .domain import AppState, ControllerSnapshot


_TRANSPARENT = "#010203"


class TipScrollUI:
    def __init__(
        self,
        controller: TipScrollController,
        config: AppConfig,
        debug_frame: LatestDebugFrame,
        shutdown_event: threading.Event,
        debug: bool,
        camera: CameraPipeline | None = None,
    ) -> None:
        self._controller = controller
        self._config = config
        self._debug_frame = debug_frame
        self._shutdown_event = shutdown_event
        self._debug = debug
        self._camera = camera
        self._root = tk.Tk()
        self._root.title("TipScroll")
        self._root.overrideredirect(True)
        self._root.configure(bg=_TRANSPARENT)
        self._root.attributes("-topmost", True)
        self._root.wm_attributes("-transparentcolor", _TRANSPARENT)

        width, height = 84, 190
        x = self._root.winfo_screenwidth() - width - 24
        y = (self._root.winfo_screenheight() - height) // 2
        self._root.geometry(f"{width}x{height}+{x}+{y}")
        self._indicator = tk.Canvas(
            self._root,
            width=width,
            height=height,
            bg=_TRANSPARENT,
            highlightthickness=0,
        )
        self._indicator.pack(fill="both", expand=True)

        self._debug_window: tk.Toplevel | None = None
        self._debug_label: tk.Label | None = None
        self._debug_photo: ImageTk.PhotoImage | None = None
        self._previous_state = AppState.IDLE
        self._safe_stop_started: float | None = None
        if debug:
            self._create_debug_window()

        self._root.after(0, self._make_click_through)
        self._root.after(self._config.ui_update_ms, self._update_indicator)
        if debug:
            self._root.after(self._config.debug_update_ms, self._update_debug)

    def run(self) -> None:
        self._root.mainloop()

    def _create_debug_window(self) -> None:
        self._debug_window = tk.Toplevel(self._root)
        self._debug_window.title("TipScroll Debug")
        self._debug_window.geometry("960x600")
        self._debug_window.configure(bg="#17191d")
        self._debug_window.protocol("WM_DELETE_WINDOW", self._request_shutdown)
        self._debug_label = tk.Label(
            self._debug_window,
            text="Waiting for MediaPipe results…",
            bg="#17191d",
            fg="white",
        )
        self._debug_label.pack(fill="both", expand=True)

    def _request_shutdown(self) -> None:
        self._shutdown_event.set()

    def _make_click_through(self) -> None:
        if os.name != "nt":
            return
        hwnd = self._root.winfo_id()
        get_style = ctypes.windll.user32.GetWindowLongW
        set_style = ctypes.windll.user32.SetWindowLongW
        extended_style_index = -20
        layered = 0x00080000
        transparent = 0x00000020
        tool_window = 0x00000080
        style = get_style(hwnd, extended_style_index)
        set_style(hwnd, extended_style_index, style | layered | transparent | tool_window)

    def _update_indicator(self) -> None:
        if self._shutdown_event.is_set():
            try:
                self._root.destroy()
            except tk.TclError:
                pass
            return

        snapshot = self._controller.snapshot(time.monotonic_ns() // 1_000_000)
        self._draw_indicator(snapshot)
        self._root.after(self._config.ui_update_ms, self._update_indicator)

    def _draw_indicator(self, snapshot: ControllerSnapshot) -> None:
        canvas = self._indicator
        canvas.delete("all")
        center_x, center_y = 42, 95
        radius = 10

        if snapshot.state is AppState.SAFE_STOP and self._previous_state is not AppState.SAFE_STOP:
            self._safe_stop_started = time.monotonic()
        elif snapshot.state is not AppState.SAFE_STOP:
            self._safe_stop_started = None
        self._previous_state = snapshot.state

        if snapshot.state is AppState.IDLE:
            return
        if snapshot.state is AppState.ARMING:
            color = "#f5b642"
            canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=color,
                outline="",
            )
            extent = max(4.0, snapshot.arming_progress * 356.0)
            canvas.create_arc(
                center_x - 17,
                center_y - 17,
                center_x + 17,
                center_y + 17,
                start=90,
                extent=-extent,
                style="arc",
                outline="#fff0b5",
                width=3,
            )
            return
        if snapshot.state is AppState.SAFE_STOP:
            elapsed_ms = (
                0.0
                if self._safe_stop_started is None
                else (time.monotonic() - self._safe_stop_started) * 1000.0
            )
            if elapsed_ms >= self._config.safe_stop_fade_ms:
                return
            fade = 1.0 - elapsed_ms / self._config.safe_stop_fade_ms
            color = _blend_hex("#8b7355", _TRANSPARENT, 1.0 - fade)
            canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=color,
                outline="",
            )
            return

        if snapshot.state is AppState.HOLD:
            # Anchor kept, output stopped: a hollow ring rather than a filled dot.
            canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline="#55c8ff",
                width=3,
            )
            return

        color = "#55c8ff" if snapshot.committed_rate == 0.0 else "#56e39f"
        canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="",
        )
        rate_ratio = min(
            1.0,
            abs(snapshot.committed_rate) / self._config.max_scroll_notches_per_second,
        )
        if rate_ratio > 0.0:
            length = 18 + 54 * rate_ratio
            direction = -1 if snapshot.committed_rate > 0.0 else 1
            end_y = center_y + direction * length
            canvas.create_line(
                center_x,
                center_y + direction * 12,
                center_x,
                end_y,
                fill=color,
                width=4,
                capstyle=tk.ROUND,
            )

    def _update_debug(self) -> None:
        if self._shutdown_event.is_set() or self._debug_window is None:
            return
        frame, error = self._debug_frame.read()
        if frame is not None and self._debug_label is not None:
            rgb = frame.rgb.copy()
            snapshot = self._controller.snapshot(time.monotonic_ns() // 1_000_000)
            self._annotate(rgb, snapshot, error)
            image = Image.fromarray(rgb)
            image.thumbnail((940, 540), Image.Resampling.LANCZOS)
            self._debug_photo = ImageTk.PhotoImage(image=image)
            self._debug_label.configure(image=self._debug_photo, text="")
        elif error is not None and self._debug_label is not None:
            self._debug_label.configure(text=error, image="")
        self._root.after(self._config.debug_update_ms, self._update_debug)

    def _annotate(
        self,
        rgb: object,
        snapshot: ControllerSnapshot,
        error: str | None,
    ) -> None:
        height, width = rgb.shape[:2]  # type: ignore[union-attr]
        if snapshot.raw_tip_x is not None and snapshot.raw_tip_y is not None:
            x = int(snapshot.raw_tip_x * width)
            y = int(snapshot.raw_tip_y * height)
            cv2.circle(rgb, (x, y), 9, (255, 80, 80), -1)  # type: ignore[arg-type]
        if snapshot.anchor_y is not None:
            anchor_px = int(snapshot.anchor_y * height)
            cv2.line(rgb, (0, anchor_px), (width, anchor_px), (80, 220, 255), 2)  # type: ignore[arg-type]
            for offset, color in (
                (self._config.stop_enter_offset, (80, 255, 120)),
                (self._config.stop_exit_offset, (255, 210, 70)),
                (self._config.fast_offset, (255, 90, 90)),
            ):
                for sign in (-1, 1):
                    y = int((snapshot.anchor_y + sign * offset) * height)
                    if 0 <= y < height:
                        cv2.line(rgb, (0, y), (width, y), color, 1)  # type: ignore[arg-type]

        lines = [
            f"state={snapshot.state.name} reason={snapshot.stop_reason.name}",
            f"pip={_fmt(snapshot.pip_angle_deg)} dip={_fmt(snapshot.dip_angle_deg)} age={snapshot.result_age_ms:.0f}ms",
            f"offset={snapshot.offset:+.4f} desired={snapshot.desired_rate:+.2f} committed={snapshot.committed_rate:+.2f}",
        ]
        if self._camera is not None:
            info = self._camera.info
            lines.append(
                f"cam={info.describe() if info else 'n/a'} "
                f"capture={self._camera.capture_rate.read():.1f}fps "
                f"results={self._camera.result_rate.read():.1f}fps"
            )
        lines.append("Ctrl+Alt+Esc: emergency stop and exit")
        if error:
            lines.append(f"ERROR: {error}")
        for index, text in enumerate(lines):
            cv2.putText(  # type: ignore[arg-type]
                rgb,
                text,
                (16, 28 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _blend_hex(start: str, end: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(a + (b - a) * amount) for a, b in zip(start_rgb, end_rgb))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Callable
from ctypes import wintypes


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
VK_ESCAPE = 0x1B
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class GlobalAbortHotkey:
    """Register Ctrl+Alt+Escape without suppressing normal keyboard input."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self.error: str | None = None

    def start(self) -> None:
        if os.name != "nt" or (self._thread is not None and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run, name="tipscroll-hotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if os.name == "nt" and self._thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._thread_id = None

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        hotkey_id = 1
        if not user32.RegisterHotKey(None, hotkey_id, MOD_CONTROL | MOD_ALT, VK_ESCAPE):
            self.error = "Ctrl+Alt+Esc could not be registered"
            return
        try:
            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY and message.wParam == hotkey_id:
                    self._callback()
        finally:
            user32.UnregisterHotKey(None, hotkey_id)


from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .camera import CameraPipeline, LatestDebugFrame
from .config import AppConfig
from .controller import TipScrollController
from .domain import StopReason
from .hotkey import GlobalAbortHotkey
from .scroll import NullScrollSink, ScrollWorker, WindowsScrollSink
from .session_log import SessionCsvLogger
from .ui import TipScrollUI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control vertical scrolling with one fingertip")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--debug", action="store_true", help="Show the camera diagnostics window")
    parser.add_argument(
        "--no-scroll",
        action="store_true",
        help="Run recognition and rate control without injecting wheel input",
    )
    parser.add_argument(
        "--legacy-wheel",
        action="store_true",
        help="Emit whole 120-unit notches for applications that ignore sub-notch wheel deltas",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "hand_landmarker.task"
    config = replace(
        AppConfig(),
        camera_index=args.camera,
        high_resolution_wheel=not args.legacy_wheel,
    )
    shutdown_event = threading.Event()
    controller = TipScrollController(config)
    debug_frame = LatestDebugFrame()
    logger = SessionCsvLogger(project_root / "logs")
    log_path = logger.start()

    sink = NullScrollSink() if args.no_scroll else WindowsScrollSink()
    scroll_worker = ScrollWorker(
        controller,
        sink,
        config.scroll_output_hz,
        on_emit=lambda delta, snapshot: logger.log_snapshot(
            "scroll", snapshot, emitted_wheel_delta=delta
        ),
        high_resolution=config.high_resolution_wheel,
    )
    camera = CameraPipeline(
        config,
        model_path,
        controller,
        debug_frame,
        on_snapshot=lambda snapshot: logger.log_snapshot("observation", snapshot),
    )

    def request_shutdown() -> None:
        controller.emergency_stop(StopReason.USER_ABORT)
        shutdown_event.set()

    hotkey = GlobalAbortHotkey(request_shutdown)
    print(f"TipScroll log: {log_path}")
    print("Emergency stop: Ctrl+Alt+Esc")
    try:
        scroll_worker.start()
        camera_info = camera.start()
        print(f"Camera negotiated: {camera_info.describe()}")
        if camera_info.fourcc != config.camera_fourcc:
            print(
                f"Warning: requested {config.camera_fourcc} but the driver reports "
                f"{camera_info.fourcc or 'unknown'}; frame rate may be reduced",
                file=sys.stderr,
            )
        hotkey.start()
        ui = TipScrollUI(controller, config, debug_frame, shutdown_event, args.debug, camera)
        ui.run()
        return 0
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"TipScroll failed to start: {exc}", file=sys.stderr)
        return 1
    finally:
        request_shutdown()
        scroll_worker.stop()
        camera.stop()
        hotkey.stop()
        logger.log_snapshot("shutdown", controller.snapshot())
        logger.stop()


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))

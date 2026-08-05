from __future__ import annotations

import csv

from tipscroll.config import AppConfig
from tipscroll.controller import TipScrollController
from tipscroll.session_log import SessionCsvLogger


def test_session_logger_writes_snapshot(tmp_path) -> None:
    controller = TipScrollController(AppConfig())
    logger = SessionCsvLogger(tmp_path)
    path = logger.start()
    logger.log_snapshot("test", controller.snapshot(), emitted_wheel_delta=240)
    logger.stop()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["event_type"] == "test"
    assert rows[0]["emitted_wheel_delta"] == "240"


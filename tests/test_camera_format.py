from __future__ import annotations

import pytest

from tipscroll.camera import CameraInfo, MeasuredRate, _decode_fourcc, _encode_fourcc


def test_fourcc_round_trips() -> None:
    for code in ("MJPG", "YUY2", "NV12"):
        assert _decode_fourcc(_encode_fourcc(code)) == code


def test_fourcc_matches_the_opencv_encoding() -> None:
    cv2 = pytest.importorskip("cv2")

    assert _encode_fourcc("MJPG") == int(cv2.VideoWriter_fourcc(*"MJPG"))


def test_decode_tolerates_an_unreported_format() -> None:
    assert _decode_fourcc(0.0) == ""


def test_encode_rejects_a_malformed_code() -> None:
    with pytest.raises(ValueError):
        _encode_fourcc("MJP")


def test_camera_info_describes_what_the_driver_granted() -> None:
    info = CameraInfo(width=640, height=480, fps_reported=30.0, fourcc="MJPG")

    assert info.describe() == "640x480 @ 30fps (MJPG)"


def test_measured_rate_reports_zero_before_two_ticks() -> None:
    rate = MeasuredRate()

    assert rate.read() == 0.0
    rate.tick(1.0)
    assert rate.read() == 0.0


def test_measured_rate_averages_the_observed_interval() -> None:
    rate = MeasuredRate()

    for index in range(11):
        rate.tick(index * 0.02)

    assert rate.read() == pytest.approx(50.0)

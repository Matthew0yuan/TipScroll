from __future__ import annotations

import math


def smoothing_alpha(cutoff_hz: float, dt_seconds: float) -> float:
    """Return the frame-rate independent alpha of a first-order low pass.

    A fixed alpha changes meaning whenever the camera frame rate changes, so
    every filter here derives alpha from the elapsed time and a cutoff instead.

    This uses the exact exponential discretization rather than the ``1/(1 +
    tau/dt)`` approximation quoted in the One Euro paper: only the exponential
    form composes exactly, so two frames at 60Hz smooth precisely as much as
    one frame at 30Hz. The approximation drifts a few percent at the frame
    intervals a webcam actually delivers.
    """

    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive")
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 - math.exp(-dt_seconds / tau)


class OneEuroFilter:
    """One Euro filter over a normalized coordinate.

    The cutoff rises with the smoothed speed of the signal, so a resting
    fingertip is filtered hard enough to hide landmark jitter while a
    deliberate move keeps almost no lag. ``beta`` is expressed in hertz per
    normalized-height unit per second, not the pixel-scale values quoted in the
    original paper.
    """

    def __init__(
        self,
        min_cutoff_hz: float,
        beta: float,
        derivative_cutoff_hz: float,
    ) -> None:
        if min_cutoff_hz <= 0.0 or derivative_cutoff_hz <= 0.0:
            raise ValueError("Filter cutoffs must be positive")
        if beta < 0.0:
            raise ValueError("beta cannot be negative")
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._derivative_cutoff_hz = derivative_cutoff_hz
        self._value: float | None = None
        self._derivative = 0.0

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value
        self._derivative = 0.0

    def update(self, value: float, dt_seconds: float) -> float:
        if self._value is None:
            self._value = value
            self._derivative = 0.0
            return value

        derivative = (value - self._value) / dt_seconds
        derivative_alpha = smoothing_alpha(self._derivative_cutoff_hz, dt_seconds)
        self._derivative += derivative_alpha * (derivative - self._derivative)

        cutoff_hz = self._min_cutoff_hz + self._beta * abs(self._derivative)
        alpha = smoothing_alpha(cutoff_hz, dt_seconds)
        self._value += alpha * (value - self._value)
        return self._value


class PassThroughFilter:
    """Filter shaped stand-in used when smoothing is disabled."""

    def __init__(self) -> None:
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self, value: float | None = None) -> None:
        self._value = value

    def update(self, value: float, dt_seconds: float) -> float:
        del dt_seconds
        self._value = value
        return value

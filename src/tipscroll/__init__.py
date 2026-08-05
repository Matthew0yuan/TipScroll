"""TipScroll: safe vertical scrolling controlled by one fingertip."""

from .config import AppConfig
from .controller import TipScrollController
from .domain import AppState, ControllerSnapshot, StopReason, TipObservation

__all__ = [
    "AppConfig",
    "AppState",
    "ControllerSnapshot",
    "StopReason",
    "TipObservation",
    "TipScrollController",
]

__version__ = "0.1.0"


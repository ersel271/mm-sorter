# tests/helpers/events_helpers.py

from config.constants import ColourID
from utils.events import VisionEvent
from utils.metrics import RunningMetrics

def make_event(**overrides) -> VisionEvent:
    defaults = {
        "ts_wall": "2026-03-14T12:00:00.000Z",
        "ts_mono": 1234.567,
        "object_id": 1,
        "class_id": 1,
        "class_name": "red",
        "confidence": 0.92,
        "low_confidence": False,
        "centroid_x": 400,
        "centroid_y": 200,
        "area": 1800.0,
        "sat_mean": 130.0,
        "highlight_ratio": 0.04,
        "hue_peak_width": 18,
        "texture_variance": 11.0,
        "circularity": 0.87,
        "aspect_ratio": 1.10,
        "solidity": 0.96,
        "frame_ms": 22.5,
    }
    defaults.update(overrides)
    return VisionEvent(**defaults)

def make_metrics() -> RunningMetrics:
    """build a RunningMetrics instance with a few events across different colour classes."""
    m = RunningMetrics()
    for colour_id, low_conf in [
        (ColourID.RED,    False),
        (ColourID.GREEN,  False),
        (ColourID.BLUE,   False),
        (ColourID.NON_MM, True),
    ]:
        m.update(make_event(class_id=int(colour_id), low_confidence=low_conf))
    return m

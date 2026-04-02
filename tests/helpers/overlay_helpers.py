# tests/helpers/overlay_helpers.py

from dataclasses import replace

import numpy as np

from src.ui import Overlay
from tests.helpers.features_helpers import make_decision, make_features, make_preprocess_result

NOT_FOUND = replace(
    make_preprocess_result(),
    found=False,
    contour=None,
    centroid=None,
    bbox=None,
    area=0.0,
)

def render_overlay(overlay: Overlay, found: bool = False, **kwargs) -> np.ndarray | None:
    """call render with sensible defaults; override via kwargs."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = make_preprocess_result() if found else NOT_FOUND
    features = make_features() if found else None
    decision = make_decision() if found else None
    return overlay.render(
        frame,
        result,
        kwargs.get("features", features),
        kwargs.get("decision", decision),
        uart_sent=kwargs.get("uart_sent", 0),
        uart_dropped=kwargs.get("uart_dropped", 0),
        uart_connected=kwargs.get("uart_connected", False),
    )

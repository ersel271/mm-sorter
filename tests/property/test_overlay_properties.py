# tests/property/test_overlay_properties.py

import tempfile
from pathlib import Path

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st

from src.ui import Overlay
from src.ui.panel import PANEL_W
from config import Config
from config.constants import ColourID
from utils.metrics import RunningMetrics
from tests.helpers.overlay_helpers import NOT_FOUND
from tests.helpers.config_helpers import write_config
from tests.helpers.features_helpers import make_decision, make_preprocess_result

# all ColourID values produce a valid uint8 three-channel ndarray
@given(st.sampled_from(list(ColourID)))
@pytest.mark.property
def test_all_colour_ids_produce_valid_ndarray(colour_id: ColourID) -> None:
    ov = Overlay(Config(), RunningMetrics())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = ov.render(frame, make_preprocess_result(), None, make_decision(label=colour_id))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8

# output always has exactly three channels regardless of colour label
@given(st.sampled_from(list(ColourID)))
@pytest.mark.property
def test_output_always_has_three_channels(colour_id: ColourID) -> None:
    ov = Overlay(Config(), RunningMetrics())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = ov.render(frame, make_preprocess_result(), None, make_decision(label=colour_id))
    assert out is not None
    assert out.ndim == 3
    assert out.shape[2] == 3

# found=False never causes a crash regardless of uart_connected value
@given(st.booleans())
@pytest.mark.property
def test_found_false_never_crashes(uart_connected: bool) -> None:
    ov = Overlay(Config(), RunningMetrics())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = ov.render(
        frame, NOT_FOUND, None, None,
        uart_connected=uart_connected,
    )
    assert isinstance(out, np.ndarray)

# any boolean value for uart_connected renders without error
@given(st.booleans())
@pytest.mark.property
def test_uart_connected_any_bool_does_not_crash(uart_connected: bool) -> None:
    ov = Overlay(Config(), RunningMetrics())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = ov.render(
        frame, make_preprocess_result(), None, make_decision(),
        uart_connected=uart_connected,
    )
    assert out is not None

# output height matches input scaled by the configured factor; width includes sidebar
@given(st.floats(min_value=0.1, max_value=3.0))
@settings(max_examples=20)
@pytest.mark.property
def test_output_shape_matches_scale(scale: float) -> None:
    data = Config().as_dict()
    data["system"]["display_scale"] = scale
    with tempfile.TemporaryDirectory() as d:
        ov = Overlay(Config(write_config(data, Path(d))), RunningMetrics())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = ov.render(frame, NOT_FOUND, None, None)
    assert out is not None
    assert out.shape[0] == int(100 * scale)
    assert out.shape[1] == int(100 * scale) + PANEL_W

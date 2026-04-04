# tests/property/test_overlay_properties.py

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.ui import Overlay
from src.ui.panel import PANEL_W
from config.constants import ColourID
from utils.metrics import RunningMetrics
from tests.helpers.overlay_helpers import NOT_FOUND
from tests.helpers.config_helpers import make_config
from tests.helpers.vision_helpers import make_decision, make_preprocess_result

_ov = Overlay(make_config(), RunningMetrics())


@pytest.mark.regression
class TestOverlayProperties:
    """property tests for Overlay render output invariants"""

    # all ColourID values produce a valid uint8 three-channel ndarray
    @given(st.sampled_from(list(ColourID)))
    def test_all_colour_ids_produce_valid_ndarray(self, colour_id: ColourID) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _ov.render(frame, make_preprocess_result(), None, make_decision(label=colour_id))
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8

    # output always has exactly three channels regardless of colour label
    @given(st.sampled_from(list(ColourID)))
    def test_output_always_has_three_channels(self, colour_id: ColourID) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _ov.render(frame, make_preprocess_result(), None, make_decision(label=colour_id))
        assert out is not None
        assert out.ndim == 3
        assert out.shape[2] == 3

    # found=False never causes a crash regardless of uart_connected value
    @given(st.booleans())
    def test_found_false_never_crashes(self, uart_connected: bool) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _ov.render(
            frame, NOT_FOUND, None, None,
            uart_connected=uart_connected,
        )
        assert isinstance(out, np.ndarray)

    # any boolean value for uart_connected renders without error
    @given(st.booleans())
    def test_uart_connected_any_bool_does_not_crash(self, uart_connected: bool) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _ov.render(
            frame, make_preprocess_result(), None, make_decision(),
            uart_connected=uart_connected,
        )
        assert out is not None

    # output height matches input scaled by the configured factor; width includes sidebar
    @given(st.floats(min_value=0.1, max_value=3.0))
    @settings(max_examples=20, deadline=None)
    def test_output_shape_matches_scale(self, scale: float) -> None:
        ov = Overlay(make_config(system={"display_scale": scale}), RunningMetrics())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = ov.render(frame, NOT_FOUND, None, None)
        assert out is not None
        assert out.shape[0] == int(100 * scale)
        assert out.shape[1] == int(100 * scale) + PANEL_W

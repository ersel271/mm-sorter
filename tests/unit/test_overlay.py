# tests/unit/test_overlay.py

import time

import pytest
import numpy as np

from src.ui import Overlay
from src.ui.panel import PANEL_W
from config import Config
from config.constants import ColourID
from utils.metrics import RunningMetrics
from tests.helpers.features_helpers import make_decision
from tests.helpers.overlay_helpers import NOT_FOUND, render_overlay

@pytest.mark.smoke
@pytest.mark.unit
class TestOverlayInit:
    """verify Overlay construction and initial state."""

    def test_instantiates_without_error(self, overlay: Overlay) -> None:
        assert overlay is not None

    def test_fps_zero_before_firstrender_overlay(self, overlay: Overlay) -> None:
        assert overlay.fps == 0.0

    def test_fps_zero_after_exactly_onerender_overlay(self, overlay: Overlay) -> None:
        render_overlay(overlay)
        assert overlay.fps == 0.0

@pytest.mark.unit
class TestRenderDisabled:
    """verify render returns None when display is disabled."""

    def test_returns_none_when_disabled(self, overlay_disabled: Overlay) -> None:
        assert render_overlay(overlay_disabled) is None

    def test_returns_none_consistently(self, overlay_disabled: Overlay) -> None:
        for _ in range(5):
            assert render_overlay(overlay_disabled) is None

@pytest.mark.unit
class TestRenderOutput:
    """verify render produces valid annotated frames."""

    def test_returns_ndarray_when_enabled(self, overlay: Overlay) -> None:
        assert isinstance(render_overlay(overlay), np.ndarray)

    def test_output_has_three_channels(self, overlay: Overlay) -> None:
        out = render_overlay(overlay)
        assert out is not None
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_output_dtype_uint8(self, overlay: Overlay) -> None:
        out = render_overlay(overlay)
        assert out is not None
        assert out.dtype == np.uint8

    def test_found_true_does_not_crash(self, overlay: Overlay) -> None:
        assert render_overlay(overlay, found=True) is not None

    def test_found_false_does_not_crash(self, overlay: Overlay) -> None:
        assert render_overlay(overlay, found=False) is not None

    def test_decision_none_does_not_crash(self, overlay: Overlay) -> None:
        assert render_overlay(overlay, found=True, decision=None) is not None

    def test_features_none_does_not_crash(self, overlay: Overlay) -> None:
        assert render_overlay(overlay, found=True, features=None) is not None

    def test_uart_connected_true_does_not_crash(self, overlay: Overlay) -> None:
        assert render_overlay(overlay, uart_connected=True) is not None

    def test_does_not_mutate_input_frame(self, overlay: Overlay) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        overlay.render(frame, NOT_FOUND, None, None)
        assert np.array_equal(frame, original)

    def test_all_colour_idsrender_overlay_without_crash(self, overlay: Overlay) -> None:
        for colour_id in ColourID:
            dec = make_decision(label=colour_id)
            assert render_overlay(overlay, found=True, decision=dec) is not None

    def test_roi_smaller_than_frame_does_not_crash(self, overlay: Overlay) -> None:
        # roi (100x100) smaller than frame (200x200) triggers _draw_roi_boundary drawing path
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = NOT_FOUND
        out = overlay.render(frame, result, None, None)
        assert isinstance(out, np.ndarray)

@pytest.mark.unit
class TestDebugMode:
    """verify debug toggle controls annotation rendering."""

    def test_debug_true_by_default(self, overlay: Overlay) -> None:
        assert overlay.debug is True

    def test_toggle_debug_switches_to_false(self, overlay: Overlay) -> None:
        overlay.toggle_debug()
        assert overlay.debug is False

    def test_toggle_debug_twice_restores_true(self, overlay: Overlay) -> None:
        overlay.toggle_debug()
        overlay.toggle_debug()
        assert overlay.debug is True

    def test_debug_false_still_returns_ndarray(self, overlay: Overlay) -> None:
        overlay.toggle_debug()
        assert isinstance(render_overlay(overlay), np.ndarray)

    def test_debug_false_output_dtype_uint8(self, overlay: Overlay) -> None:
        overlay.toggle_debug()
        out = render_overlay(overlay)
        assert out is not None
        assert out.dtype == np.uint8

@pytest.mark.unit
class TestRenderScale:
    """verify output dimensions follow the configured scale factor."""

    def test_scale_half_halves_output_dimensions(self, overlay_half_scale: Overlay) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = overlay_half_scale.render(frame, NOT_FOUND, None, None)
        assert out is not None
        assert out.shape[:2] == (50, 50 + PANEL_W)

    def test_scale_one_preserves_dimensions(self, overlay: Overlay) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = overlay.render(frame, NOT_FOUND, None, None)
        assert out is not None
        assert out.shape[:2] == (100, 100 + PANEL_W)

@pytest.mark.unit
class TestFPS:
    """verify fps property reflects render call frequency."""

    def test_fps_zero_before_anyrender_overlay(self, overlay: Overlay) -> None:
        assert overlay.fps == 0.0

    def test_fps_positive_after_multiplerender_overlays(self, overlay: Overlay) -> None:
        for _ in range(3):
            render_overlay(overlay)
            time.sleep(0.01)
        assert overlay.fps > 0.0

    def test_fps_always_nonnegative(self, overlay: Overlay) -> None:
        for _ in range(5):
            render_overlay(overlay)
        assert overlay.fps >= 0.0

@pytest.mark.unit
class TestSidebarModes:
    """verify sidebar mode cycling and all three panel render paths."""

    def test_default_sidebar_mode_is_zero(self, overlay: Overlay) -> None:
        assert overlay._sidebar_mode == 0

    def test_toggle_sidebar_advances_mode(self, overlay: Overlay) -> None:
        overlay.toggle_sidebar()
        assert overlay._sidebar_mode == 1

    def test_toggle_sidebar_wraps_to_zero(self, overlay: Overlay) -> None:
        for _ in range(3):
            overlay.toggle_sidebar()
        assert overlay._sidebar_mode == 0

    def test_sidebar_mode_1_renders_without_crash(self, overlay: Overlay) -> None:
        overlay.toggle_sidebar()
        assert render_overlay(overlay) is not None

    def test_sidebar_mode_2_renders_without_crash(self, overlay: Overlay) -> None:
        overlay.toggle_sidebar()
        overlay.toggle_sidebar()
        assert render_overlay(overlay) is not None

@pytest.mark.unit
class TestLogStripToggle:
    """verify show_log toggle and log strip render path."""

    def test_show_log_false_by_default(self, overlay: Overlay) -> None:
        assert overlay._show_log is False

    def test_toggle_log_enables_strip(self, overlay: Overlay) -> None:
        overlay.toggle_log()
        assert overlay._show_log is True

    def test_toggle_log_twice_restores_false(self, overlay: Overlay) -> None:
        overlay.toggle_log()
        overlay.toggle_log()
        assert overlay._show_log is False

    def test_show_log_true_renders_taller_output(self, overlay: Overlay) -> None:
        out_no_log = render_overlay(overlay)
        overlay.toggle_log()
        out_with_log = render_overlay(overlay)
        assert out_no_log is not None and out_with_log is not None
        assert out_with_log.shape[0] > out_no_log.shape[0]

@pytest.mark.unit
class TestContextManager:
    """verify context manager protocol does not raise."""

    def test_context_manager_does_not_crash(self) -> None:
        with Overlay(Config(), RunningMetrics()) as ov:
            assert ov is not None

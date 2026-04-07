# tests/unit/test_keybinds.py

import pytest

from src.ui import Overlay, handle_key

@pytest.mark.smoke
class TestHandleKeyQuit:
    """verify q key signals application quit."""

    def test_q_returns_true(self, overlay: Overlay) -> None:
        assert handle_key(ord("q"), overlay) is True

    def test_q_with_high_bits_returns_true(self, overlay: Overlay) -> None:
        assert handle_key(ord("q") | 0xFFFF00, overlay) is True

@pytest.mark.smoke
class TestHandleKeyDebug:
    """verify d key toggles debug overlay and returns False."""

    def test_d_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(ord("d"), overlay) is False

    def test_d_toggles_debug_off(self, overlay: Overlay) -> None:
        assert overlay.debug is True
        handle_key(ord("d"), overlay)
        assert overlay.debug is False

    def test_d_toggles_debug_back_on(self, overlay: Overlay) -> None:
        handle_key(ord("d"), overlay)
        handle_key(ord("d"), overlay)
        assert overlay.debug is True

@pytest.mark.smoke
class TestHandleKeySidebar:
    """verify t key cycles sidebar mode and returns False."""

    def test_t_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(ord("t"), overlay) is False

    def test_t_advances_sidebar_mode(self, overlay: Overlay) -> None:
        initial = overlay._sidebar_mode
        handle_key(ord("t"), overlay)
        assert overlay._sidebar_mode == (initial + 1) % 3

    def test_t_three_times_returns_to_initial(self, overlay: Overlay) -> None:
        initial = overlay._sidebar_mode
        for _ in range(3):
            handle_key(ord("t"), overlay)
        assert overlay._sidebar_mode == initial

@pytest.mark.smoke
class TestHandleKeyLog:
    """verify l key toggles log strip and returns False."""

    def test_l_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(ord("l"), overlay) is False

    def test_l_toggles_show_log_on(self, overlay: Overlay) -> None:
        assert overlay._show_log is False
        handle_key(ord("l"), overlay)
        assert overlay._show_log is True

    def test_l_toggles_show_log_off(self, overlay: Overlay) -> None:
        handle_key(ord("l"), overlay)
        handle_key(ord("l"), overlay)
        assert overlay._show_log is False

@pytest.mark.smoke
class TestHandleKeyFreeze:
    """verify ctrl+f key toggles freeze mode and returns False."""

    def test_ctrl_f_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(ord("f"), overlay) is False

    def test_ctrl_f_toggles_freeze_on(self, overlay: Overlay) -> None:
        assert overlay.frozen is False
        handle_key(ord("f"), overlay)
        assert overlay.frozen is True

    def test_ctrl_f_toggles_freeze_off(self, overlay: Overlay) -> None:
        handle_key(ord("f"), overlay)
        handle_key(ord("f"), overlay)
        assert overlay.frozen is False

@pytest.mark.regression
class TestHandleKeyUnknown:
    """verify unbound keys are silently ignored."""

    def test_unknown_key_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(ord("z"), overlay) is False

    def test_no_key_minus_one_returns_false(self, overlay: Overlay) -> None:
        # cv2.waitKey returns -1 when no key is pressed
        assert handle_key(-1, overlay) is False

    def test_zero_returns_false(self, overlay: Overlay) -> None:
        assert handle_key(0, overlay) is False

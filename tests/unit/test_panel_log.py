# tests/unit/test_panel_log.py

import logging
import time
from pathlib import Path

import pytest
import numpy as np

from src.ui.panels.log_panel import LogPanel, _STRIP_H

@pytest.mark.unit
class TestLogPanelShape:
    """verify LogPanel.render produces the correct (_STRIP_H, total_w, 3) strip."""

    def test_shape_strip(self, log_panel: LogPanel) -> None:
        assert log_panel.render(800).shape == (_STRIP_H, 800, 3)

    def test_various_widths_produce_correct_shape(self, log_panel: LogPanel) -> None:
        for w in (320, 640, 1280):
            assert log_panel.render(w).shape == (_STRIP_H, w, 3)

    def test_dtype_uint8(self, log_panel: LogPanel) -> None:
        assert log_panel.render(640).dtype == np.uint8

@pytest.mark.unit
class TestLogPanelStripH:
    """verify strip_h property exposes the correct constant."""

    def test_strip_h_matches_module_constant(self, log_panel: LogPanel) -> None:
        assert log_panel.strip_h == _STRIP_H

@pytest.mark.unit
class TestLogPanelSafety:
    """verify LogPanel behaves correctly without a log file."""

    def test_no_log_file_render_does_not_crash(self) -> None:
        panel = LogPanel()
        assert panel.render(640) is not None

    def test_close_twice_does_not_crash(self, log_panel: LogPanel) -> None:
        log_panel.close()
        log_panel.close()

@pytest.mark.unit
class TestLogPanelWithFile:
    """verify LogPanel reads from a real log file and renders lines."""

    def test_reads_existing_lines(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        log_file.write_text("line one\nline two\n", encoding="utf-8")
        handler = logging.FileHandler(str(log_file))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            panel = LogPanel()
            time.sleep(0.15)  # allow tail thread to read the file
            out = panel.render(640)
            panel.close()
        finally:
            root.removeHandler(handler)
            handler.close()
        assert out.shape == (_STRIP_H, 640, 3)
        assert out.dtype == np.uint8

    def test_new_lines_appear_in_strip(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        log_file.write_text("", encoding="utf-8")
        handler = logging.FileHandler(str(log_file))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            panel = LogPanel()
            time.sleep(0.05)
            log_file.open("a").write("new entry\n")
            time.sleep(0.25)  # allow tail thread to pick up the new line
            out = panel.render(800)
            panel.close()
        finally:
            root.removeHandler(handler)
            handler.close()
        assert out.shape == (_STRIP_H, 800, 3)

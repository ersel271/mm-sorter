# src/ui/panels/log_panel.py
"""Log tail panel; renders the last N log lines into a full-width strip below the main frame."""

import collections
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from src.ui.panel import FONT, PANEL_SEP_COLOUR, Panel

_MAX_LINES: int = 8
_LINE_H:    int = 16
_STRIP_H:   int = _MAX_LINES * _LINE_H + 10

_BG_COLOUR:   tuple[int, int, int] = ( 12,  12,  12)
_TEXT_COLOUR: tuple[int, int, int] = (130, 140, 130)

log = logging.getLogger(__name__)

class LogPanel(Panel):
    """renders the last N lines of the active log file into a full-width strip."""

    def __init__(self) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=_MAX_LINES)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

        log_path = self._find_log_path()
        if log_path is None:
            log.warning("log panel: no file handler found on root logger")
            self._thread = None
            return

        self._thread = threading.Thread(
            target=self._tail, args=(log_path,), daemon=True, name="log-panel",
        )
        self._thread.start()

    @staticmethod
    def _find_log_path() -> Path | None:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return None

    def _tail(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                existing = fh.read().splitlines()
                with self._lock:
                    for line in existing[-_MAX_LINES:]:
                        if line.strip():
                            self._lines.append(line)

                while not self._stop.is_set():
                    line = fh.readline()
                    if line:
                        stripped = line.rstrip("\n")
                        if stripped.strip():
                            with self._lock:
                                self._lines.append(stripped)
                    else:
                        self._stop.wait(0.1)
        except OSError as exc:
            log.error("log panel read error: %s", exc)

    def _render_lines(self, arr: np.ndarray, lines: list[str], x_offset: int, max_w: int) -> None:
        max_chars = max(10, (max_w - 20) // 7)
        for i, text in enumerate(lines):
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            y = 6 + (i + 1) * _LINE_H
            cv2.putText(arr, text, (x_offset, y), FONT, 0.38, _TEXT_COLOUR, 1, cv2.LINE_AA)

    def render(self, total_w: int) -> np.ndarray:  # type: ignore[override]
        """return a (_STRIP_H, total_w, 3) strip showing the last log lines."""
        strip = np.full((_STRIP_H, total_w, 3), _BG_COLOUR, dtype=np.uint8)
        cv2.line(strip, (0, 0), (total_w - 1, 0), PANEL_SEP_COLOUR, 1)
        with self._lock:
            lines = list(self._lines)
        self._render_lines(strip, lines, 10, total_w)
        return strip

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def strip_h(self) -> int:
        return _STRIP_H

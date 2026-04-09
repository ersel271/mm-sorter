# src/ui/overlay.py
"""
Live OpenCV overlay renderer for the M&M sorting pipeline.

Draws detection, classification, and telemetry annotations onto each
frame. render() returns an annotated ndarray; show() is isolated so
CI can run headlessly without a display.

Usage:
    with Overlay(cfg, metrics) as ov:
        frame_out = ov.render(frame, result, features, decision, metrics)
        if frame_out is not None:
            key = ov.show(frame_out)
"""

import collections
import logging
import time

import cv2
import numpy as np

from config import Config
from config.constants import COLOUR_NAMES, ColourID
from src.vision.features import Features
from src.vision.preprocess import PreprocessResult
from src.vision.rule import Decision
from src.ui.panel import LABEL_COLOURS
from src.ui.panels import FeaturePanel, DecisionPanel, StatsPanel, LogPanel
from utils.metrics import RunningMetrics

log = logging.getLogger(__name__)

_FPS_WINDOW_LEN: int = 30
_HISTORY_LEN: int = 10

# BGR colour constants, tuned for dark backgrounds
_CONTOUR_COLOUR:  tuple[int, int, int] = (  0, 230,  80)
_BBOX_COLOUR:     tuple[int, int, int] = (  0, 190, 210)
_CENTROID_COLOUR: tuple[int, int, int] = ( 60,  60, 240)
_TEXT_COLOUR:     tuple[int, int, int] = (190, 200, 200)
_DIM_COLOUR:      tuple[int, int, int] = (110, 110, 110)
_UART_OK_COLOUR:  tuple[int, int, int] = (  0, 200,  80)
_UART_ERR_COLOUR: tuple[int, int, int] = ( 60,  60, 200)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

class Overlay:
    """
    renders annotation overlays onto BGR frames from the pipeline.
    """

    WINDOW_NAME: str = "M&M Sorter"

    def __init__(self, config: Config, metrics: RunningMetrics) -> None:
        self._cfg = config.system
        self._timestamps: collections.deque[float] = collections.deque(maxlen=_FPS_WINDOW_LEN)
        self._history: collections.deque[tuple[Decision, bool]] = collections.deque(maxlen=_HISTORY_LEN)
        self._window_open: bool = False
        self._debug: bool = True
        self._last_features: Features | None = None
        self._last_decision: Decision | None = None
        self._last_low_conf: bool = False
        self._feature_panel: FeaturePanel = FeaturePanel()
        self._decision_panel: DecisionPanel = DecisionPanel(config)
        self._metrics: RunningMetrics = metrics
        self._stats_panel: StatsPanel = StatsPanel(metrics)
        self._log_panel: LogPanel = LogPanel()
        self._sidebar_mode: int = 0  # 0 = features, 1 = rules, 2 = stats
        self._show_log: bool = False
        self._frozen: bool = False

    @property
    def debug(self) -> bool:
        return self._debug

    def toggle_debug(self) -> None:
        self._debug = not self._debug

    def toggle_sidebar(self) -> None:
        self._sidebar_mode = (self._sidebar_mode + 1) % 3

    def toggle_log(self) -> None:
        self._show_log = not self._show_log

    @property
    def frozen(self) -> bool:
        return self._frozen

    def toggle_freeze(self) -> None:
        self._frozen = not self._frozen

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def render(
        self,
        frame: np.ndarray,
        result: PreprocessResult,
        features: Features | None,
        decision: Decision | None,
        uart_sent: int = 0,
        uart_dropped: int = 0,
        uart_connected: bool = False,
        record: bool = True,
        low_conf: bool = False,
    ) -> np.ndarray | None:
        if not self._cfg.get("display_enabled", True):
            return None
        self._timestamps.append(time.monotonic())
        self._last_low_conf = low_conf
        if decision is not None and record:
            self._history.append((decision, low_conf))
        if features is not None:
            self._last_features = features
        if decision is not None:
            self._last_decision = decision
        if self._frozen:
            features = self._last_features
            decision = self._last_decision
            low_conf = self._last_low_conf

        display = frame.copy()

        # always-on: core detection feedback
        self._draw_bbox(display, result)
        self._draw_label_banner(display, result, decision, low_conf)

        # debug-only: diagnostic overlays
        if self._debug:
            self._draw_roi_boundary(display, frame, result)
            self._draw_contour(display, result)
            self._draw_centroid(display, result)
            self._draw_no_object(display, result)
            self._draw_status(display, uart_sent, uart_dropped, uart_connected)

        h, w = display.shape[:2]
        scale = float(self._cfg.get("display_scale", 1.0))
        if scale != 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        h = display.shape[0]
        if self._sidebar_mode == 0:
            sidebar = self._feature_panel.render(self._last_features, None, h)
        elif self._sidebar_mode == 1:
            sidebar = self._decision_panel.render(self._last_features, self._last_decision, h)
        else:
            sidebar = self._stats_panel.render(None, None, h)
        if self._frozen:
            self._draw_freeze_indicator(display)

        combined = np.hstack([display, sidebar])

        # draw history before log strip so centering uses only the camera frame height
        if self._debug:
            self._draw_history(combined)

        if self._show_log:
            combined = np.vstack([combined, self._log_panel.render(combined.shape[1])])

        return combined

    def show(self, frame: np.ndarray) -> int:
        self._window_open = True
        cv2.imshow(self.WINDOW_NAME, frame)
        return cv2.waitKey(1)

    def close(self) -> None:
        if self._window_open:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_open = False

    def __enter__(self) -> "Overlay":
        return self

    def __exit__(self, *_: object) -> None:
        # log_panel owns a background tail thread; explicit close joins it before teardown
        self._log_panel.close()
        self.close()

    def _draw_roi_boundary(self, display: np.ndarray, frame: np.ndarray, result: PreprocessResult) -> None:
        # no-op when ROI covers the full frame
        if result.roi.shape[:2] == frame.shape[:2]:
            return
        fh, fw = frame.shape[:2]
        rh, rw = result.roi.shape[:2]
        ox = (fw - rw) // 2
        oy = (fh - rh) // 2
        cv2.rectangle(display, (ox, oy), (ox + rw, oy + rh), _DIM_COLOUR, 1)

    def _draw_contour(self, display: np.ndarray, result: PreprocessResult) -> None:
        if not result.found or result.contour is None or result.bbox is None:
            return
        # contour is in ROI-local space. offset shifts it to full-frame coordinates
        lx, ly = cv2.boundingRect(result.contour)[:2]
        cv2.drawContours(
            display,
            [result.contour],
            -1,
            _CONTOUR_COLOUR,
            2,
            offset=(result.bbox[0] - lx, result.bbox[1] - ly),
        )

    def _draw_bbox(self, display: np.ndarray, result: PreprocessResult) -> None:
        if not result.found or result.bbox is None:
            return
        x, y, w, h = result.bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), _BBOX_COLOUR, 2)

    def _draw_centroid(self, display: np.ndarray, result: PreprocessResult) -> None:
        if not result.found or result.centroid is None:
            return
        cv2.circle(display, result.centroid, 5, _CENTROID_COLOUR, -1)

    def _draw_label_banner(
        self, display: np.ndarray, result: PreprocessResult,
        decision: Decision | None, low_conf: bool = False,
    ) -> None:
        if not result.found or result.bbox is None:
            return
        if decision is None:
            text = "---"
            colour = _DIM_COLOUR
        elif low_conf:
            colour_name = COLOUR_NAMES[decision.label].lower()
            text = (
                f"{COLOUR_NAMES[ColourID.NON_MM]}  {decision.confidence:.0%}"
                f"  [low-confidence: {colour_name}]"
            )
            colour = LABEL_COLOURS[ColourID.NON_MM]
        else:
            text = f"{COLOUR_NAMES[decision.label]}  {decision.confidence:.0%}  [{decision.rule}]"
            colour = LABEL_COLOURS[decision.label]
        ty = max(result.bbox[1] - 8, 16)
        cv2.putText(display, text, (result.bbox[0], ty), _FONT, 0.6, colour, 2, cv2.LINE_AA)

    def _draw_no_object(self, display: np.ndarray, result: PreprocessResult) -> None:
        if result.found:
            return
        text = "NO OBJECT"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 1.2, 2)
        fh, fw = display.shape[:2]
        tx = (fw - tw) // 2
        ty = (fh + th) // 2
        cv2.putText(display, text, (tx, ty), _FONT, 1.2, _DIM_COLOUR, 2, cv2.LINE_AA)

    def _draw_status(self, display: np.ndarray, uart_sent: int, uart_dropped: int, uart_connected: bool) -> None:
        _, fw = display.shape[:2]

        # top-right: FPS, frame count, then UART on the line below
        fps_text = f"FPS: {self.fps:.1f}  Frames: {self._metrics.total}"
        (fw1, _), _ = cv2.getTextSize(fps_text, _FONT, 0.55, 1)
        cv2.putText(display, fps_text, (fw - fw1 - 10, 22), _FONT, 0.55, _TEXT_COLOUR, 1, cv2.LINE_AA)

        uart_text = f"UART {uart_sent}/{uart_dropped}"
        (utw, _), _ = cv2.getTextSize(uart_text, _FONT, 0.50, 1)
        uart_label = "OK" if uart_connected else "ERR"
        uart_colour = _UART_OK_COLOUR if uart_connected else _UART_ERR_COLOUR
        (ulw, _), _ = cv2.getTextSize(uart_label, _FONT, 0.50, 1)
        total_uart_w = utw + 8 + ulw
        ux = fw - total_uart_w - 10
        cv2.putText(display, uart_text,  (ux,           42), _FONT, 0.50, _TEXT_COLOUR, 1, cv2.LINE_AA)
        cv2.putText(display, uart_label, (ux + utw + 8, 42), _FONT, 0.50, uart_colour,  1, cv2.LINE_AA)

    def _draw_freeze_indicator(self, display: np.ndarray) -> None:
        text = "FROZEN"
        fh, fw = display.shape[:2]
        tw = cv2.getTextSize(text, _FONT, 0.7, 2)[0][0]
        cv2.putText(display, text, (fw - tw - 10, fh - 10), _FONT, 0.7, _TEXT_COLOUR, 2, cv2.LINE_AA)

    def _draw_history(self, display: np.ndarray) -> None:
        history = list(self._history)
        if not history:
            return
        fh, _   = display.shape[:2]
        bw, bh  = 20, 20
        gap     = 5
        margin  = 6
        n       = len(history)
        total_h = n * bh + (n - 1) * gap
        y0      = (fh - total_h) // 2
        x0      = margin
        for i, (dec, lc) in enumerate(history):
            y = y0 + i * (bh + gap)
            sq_colour = LABEL_COLOURS[ColourID.NON_MM] if lc else LABEL_COLOURS[dec.label]
            cv2.rectangle(display, (x0, y), (x0 + bw, y + bh), sq_colour, -1)
            cv2.rectangle(display, (x0, y), (x0 + bw, y + bh), _DIM_COLOUR, 1)

# src/ui/panels/stats_panel.py
"""Session classification statistics sidebar panel."""

import cv2
import numpy as np

from config.constants import COLOUR_NAMES, ColourID
from src.vision.features import Features
from src.vision.rule import Decision
from src.ui.panel import (
    Panel, PANEL_W, BAR_H, FONT,
    PANEL_SEP_COLOUR, TEXT_COLOUR, DIM_COLOUR, BAR_BG_COLOUR, LABEL_COLOURS,
)
from utils.metrics import RunningMetrics

_LOW_CONF_COLOUR: tuple[int, int, int] = (50, 80, 210)

_ALL_COLOURS: list[ColourID] = [
    ColourID.RED, ColourID.GREEN, ColourID.BLUE,
    ColourID.YELLOW, ColourID.ORANGE, ColourID.BROWN,
    ColourID.NON_MM,
]

_ROW_H: int = 28

class StatsPanel(Panel):
    """renders session-level classification statistics into a fixed-width ndarray."""

    def __init__(self, metrics: RunningMetrics) -> None:
        self._metrics = metrics

    def render(self, features: Features | None, decision: Decision | None, panel_h: int) -> np.ndarray:
        panel = np.full((panel_h, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, panel_h - 1), PANEL_SEP_COLOUR, 1)
        cv2.putText(panel, "[STATS]", (12, 24), FONT, 0.44, TEXT_COLOUR, 1, cv2.LINE_AA)
        cv2.line(panel, (8, 32), (PANEL_W - 8, 32), PANEL_SEP_COLOUR, 1)

        m = self._metrics
        total    = m.total
        low_conf = m.low_confidence

        y = 54
        cv2.putText(panel, f"Total:    {total}",    (12, y), FONT, 0.44, TEXT_COLOUR,     1, cv2.LINE_AA)
        y += 20
        cv2.putText(panel, f"Low Conf: {low_conf}", (12, y), FONT, 0.44, _LOW_CONF_COLOUR, 1, cv2.LINE_AA)
        y += 26

        cv2.line(panel, (8, y), (PANEL_W - 8, y), PANEL_SEP_COLOUR, 1)
        y += 16

        cv2.putText(panel, f"Avg Conf:   {m.mean_confidence:.2f}", (12, y), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, f"Avg Frame:  {m.mean_frame_ms:.1f}ms", (12, y), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)
        y += 24

        cv2.line(panel, (8, y), (PANEL_W - 8, y), PANEL_SEP_COLOUR, 1)
        y += 14

        counts    = [m.class_count(int(c)) for c in _ALL_COLOURS]
        max_count = max(counts) if any(counts) else 1
        bar_x0, bar_x1 = 12, PANEL_W - 12
        bar_w = bar_x1 - bar_x0

        for colour_id, count in zip(_ALL_COLOURS, counts, strict=True):
            name   = COLOUR_NAMES[colour_id]
            colour = LABEL_COLOURS[colour_id]

            cv2.putText(panel, name, (bar_x0, y), FONT, 0.42, colour, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(str(count), FONT, 0.42, 1)
            cv2.putText(panel, str(count), (bar_x1 - tw, y), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)

            bar_y  = y + 4
            fill_w = int((count / max_count) * bar_w)
            cv2.rectangle(panel, (bar_x0, bar_y), (bar_x1, bar_y + BAR_H), BAR_BG_COLOUR, -1)
            if fill_w > 0:
                cv2.rectangle(panel, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + BAR_H), colour, -1)

            y += _ROW_H

        return panel

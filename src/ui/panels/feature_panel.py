# src/ui/panels/feature_panel.py
"""Normalised feature bar chart sidebar panel."""

import cv2
import numpy as np

from src.vision.features import Features
from src.vision.rule import Decision
from src.ui.panel import (
    Panel, PANEL_W, BAR_H, FONT,
    PANEL_SEP_COLOUR, TEXT_COLOUR, DIM_COLOUR, BAR_BG_COLOUR,
)

_BAR_FG_COLOUR: tuple[int, int, int] = (0, 170, 110)

# (label, range_lo, range_hi)
_ROWS: list[tuple[str, float, float]] = [
    ("Saturation",   0.0, 255.0),
    ("Circularity",  0.0,   1.0),
    ("Solidity",     0.0,   1.0),
    ("Aspect Ratio", 1.0,   5.0),
    ("Texture Var",  0.0, 800.0),
    ("Highlight",    0.0,   1.0),
    ("Hue Width",    0.0,  30.0),
]

class FeaturePanel(Panel):
    """renders a normalised feature bar chart into a fixed-width ndarray."""

    def render(self, features: Features | None, decision: Decision | None, panel_h: int) -> np.ndarray:
        panel = np.full((panel_h, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, panel_h - 1), PANEL_SEP_COLOUR, 1)
        cv2.putText(panel, "Detection  [FEATURES]", (12, 24), FONT, 0.44, TEXT_COLOUR, 1, cv2.LINE_AA)
        cv2.line(panel, (8, 32), (PANEL_W - 8, 32), PANEL_SEP_COLOUR, 1)

        if features is None:
            cv2.putText(panel, "no object", (12, 54), FONT, 0.48, DIM_COLOUR, 1, cv2.LINE_AA)
            return panel

        vals: list[float] = [
            features.sat_mean,
            features.circularity,
            features.solidity,
            features.aspect_ratio,
            features.texture_variance,
            features.highlight_ratio,
            float(features.hue_peak_width),
        ]
        n = len(_ROWS)
        y0 = 44
        row_h = max(28, (panel_h - y0 - 10) // n)
        bar_x0, bar_x1 = 12, PANEL_W - 12
        bar_w = bar_x1 - bar_x0

        for i, ((label, lo, hi), val) in enumerate(zip(_ROWS, vals)):
            y = y0 + i * row_h
            val_text = f"{val:.2f}" if hi <= 2.0 else f"{val:.1f}"
            cv2.putText(panel, label, (bar_x0, y), FONT, 0.42, TEXT_COLOUR, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(val_text, FONT, 0.42, 1)
            cv2.putText(panel, val_text, (bar_x1 - tw, y), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)
            bar_y = y + 4
            cv2.rectangle(panel, (bar_x0, bar_y), (bar_x1, bar_y + BAR_H), BAR_BG_COLOUR, -1)
            ratio = max(0.0, min(1.0, (val - lo) / (hi - lo) if hi > lo else 0.0))
            fill_w = int(ratio * bar_w)
            if fill_w > 0:
                cv2.rectangle(panel, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + BAR_H), _BAR_FG_COLOUR, -1)
        return panel

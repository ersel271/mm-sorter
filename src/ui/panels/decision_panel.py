# src/ui/panels/decision_panel.py
"""Per-rule pass/fail breakdown sidebar panel."""

import cv2
import numpy as np

from config import Config
from config.constants import COLOUR_NAMES, ColourID
from src.vision.features import Features
from src.vision.rule import Decision, Priority
from src.ui.panel import (
    Panel, PANEL_W, BAR_H, FONT,
    PANEL_SEP_COLOUR, TEXT_COLOUR, DIM_COLOUR, BAR_BG_COLOUR, LABEL_COLOURS,
)

_PASS_COLOUR: tuple[int, int, int] = (  0, 170, 110)
_FAIL_COLOUR: tuple[int, int, int] = ( 50,  80, 210)

# (label, Features attr, thresholds key, direction)
# "ge": pass when val >= threshold   "le": pass when val <= threshold
_STAGE1: list[tuple[str, str, str, str]] = [
    ("Saturation",  "sat_mean",        "sat_min",       "ge"),
    ("Highlight",   "highlight_ratio", "highlight_max", "le"),
    ("Hue Width",   "hue_peak_width",  "hue_width_min", "ge"),
]
_STAGE2: list[tuple[str, str, str, str]] = [
    ("Circularity", "circularity",      "circularity_min",  "ge"),
    ("Aspect Ratio","aspect_ratio",     "aspect_ratio_max", "le"),
    ("Solidity",    "solidity",         "solidity_min",     "ge"),
    ("Texture Var", "texture_variance", "texture_max",      "le"),
]

_DOT_SIZE:   int = 7
_ROW_H:      int = 36   # two-line rows: label + val/threshold
_STAGE_H:    int = 20   # stage header height
_GAP:        int = 14   # gap between stages

class DecisionPanel(Panel):
    """renders a per-rule pass/fail breakdown into a fixed-width ndarray."""

    def __init__(self, config: Config) -> None:
        self._thresholds = config.thresholds

    def render(self, features: Features | None, decision: Decision | None, panel_h: int) -> np.ndarray:
        panel = np.full((panel_h, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, panel_h - 1), PANEL_SEP_COLOUR, 1)
        cv2.putText(panel, "Detection  [RULES]", (12, 24), FONT, 0.44, TEXT_COLOUR, 1, cv2.LINE_AA)
        cv2.line(panel, (8, 32), (PANEL_W - 8, 32), PANEL_SEP_COLOUR, 1)
        if features is None:
            cv2.putText(panel, "no object", (12, 54), FONT, 0.48, DIM_COLOUR, 1, cv2.LINE_AA)
            return panel

        y = 46
        y = self._draw_stage(panel, "S1  reject", _STAGE1, features, y)
        y += _GAP
        y = self._draw_stage(panel, "S2  shape",  _STAGE2, features, y)
        y += _GAP
        self._draw_colour_result(panel, decision, y, panel_h)

        if decision is not None and decision.label == ColourID.NON_MM:
            self._draw_rejection_footer(panel, decision, panel_h)
        return panel

    def _draw_stage(self, panel: np.ndarray, title: str, checks: list[tuple[str, str, str, str]], features: Features, y: int) -> int:
        all_pass = all(
            float(getattr(features, attr)) >= float(self._thresholds[tkey])
            if direction == "ge"
            else float(getattr(features, attr)) <= float(self._thresholds[tkey])
            for _, attr, tkey, direction in checks
        )
        stage_colour = _PASS_COLOUR if all_pass else _FAIL_COLOUR

        cv2.rectangle(panel, (12, y - 1), (14, y + _STAGE_H - 6), stage_colour, -1)
        cv2.putText(panel, title, (18, y + _STAGE_H - 8), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)
        y += _STAGE_H + 24

        for label, attr, threshold_key, direction in checks:
            val = float(getattr(features, attr))
            threshold = float(self._thresholds[threshold_key])
            passed = val >= threshold if direction == "ge" else val <= threshold
            dot_colour = _PASS_COLOUR if passed else _FAIL_COLOUR
            text_colour = TEXT_COLOUR if passed else _FAIL_COLOUR
            val_colour  = DIM_COLOUR  if passed else _FAIL_COLOUR

            val_s = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            thr_s = f"{threshold:.2f}" if threshold < 10 else f"{threshold:.1f}"
            combined = f"{val_s} / {thr_s}"

            dot_y = y - _DOT_SIZE + 1
            cv2.rectangle(panel, (18, dot_y), (18 + _DOT_SIZE, dot_y + _DOT_SIZE), dot_colour, -1)
            cv2.putText(panel, label, (30, y), FONT, 0.46, text_colour, 1, cv2.LINE_AA)
            cv2.putText(panel, combined, (38, y + 18), FONT, 0.42, val_colour, 1, cv2.LINE_AA)
            y += _ROW_H
        return y

    def _draw_colour_result(self, panel: np.ndarray, decision: Decision | None, y: int, panel_h: int) -> None:
        s3_accent = (
            LABEL_COLOURS[decision.label]
            if decision is not None and decision.label != ColourID.NON_MM
            else DIM_COLOUR
        )
        cv2.rectangle(panel, (12, y - 1), (14, y + _STAGE_H - 6), s3_accent, -1)
        cv2.putText(panel, "S3  colour", (18, y + _STAGE_H - 8), FONT, 0.42, DIM_COLOUR, 1, cv2.LINE_AA)
        y += _STAGE_H + 6

        if decision is None:
            cv2.putText(panel, "not reached", (18, y), FONT, 0.44, DIM_COLOUR, 1, cv2.LINE_AA)
            return
        if decision.label == ColourID.NON_MM:
            # rejected before colour stage, priority < 30 means S1/S2 fired
            if decision.priority < Priority.S3:
                cv2.putText(panel, "not reached", (18, y), FONT, 0.44, DIM_COLOUR, 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, decision.rule, (18, y), FONT, 0.44, _FAIL_COLOUR, 1, cv2.LINE_AA)
            return

        bar_colour = LABEL_COLOURS[decision.label]
        name = COLOUR_NAMES[decision.label]
        conf_text = f"{decision.confidence:.2f}"
        (tw, _), _ = cv2.getTextSize(conf_text, FONT, 0.44, 1)
        cv2.putText(panel, conf_text, (PANEL_W - tw - 12, y), FONT, 0.44, DIM_COLOUR, 1, cv2.LINE_AA)

        bar_x0, bar_x1 = 12, PANEL_W - 12
        bar_y = y + 6
        cv2.rectangle(panel, (bar_x0, bar_y), (bar_x1, bar_y + BAR_H + 2), BAR_BG_COLOUR, -1)
        fill_w = int(decision.confidence * (bar_x1 - bar_x0))
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + BAR_H + 2), bar_colour, -1)
        cv2.putText(panel, name, (bar_x0, bar_y + BAR_H + 18), FONT, 0.48, bar_colour, 1, cv2.LINE_AA)

    def _draw_rejection_footer(self, panel: np.ndarray, decision: Decision, panel_h: int) -> None:
        fy = panel_h - 22
        cv2.line(panel, (8, fy - 10), (PANEL_W - 8, fy - 10), PANEL_SEP_COLOUR, 1)
        cv2.putText(panel, f"rejected: {decision.rule}", (12, fy), FONT, 0.42, _FAIL_COLOUR, 1, cv2.LINE_AA)

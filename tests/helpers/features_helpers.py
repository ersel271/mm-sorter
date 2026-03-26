# tests/helpers/features_helpers.py

import cv2
import numpy as np

from config.constants import ColourID
from src.vision import Decision, Features, PreprocessResult

def make_preprocess_result(hue=10, sat=200, val=180, radius=30, size=100) -> PreprocessResult:
    """build a synthetic PreprocessResult with a uniform-colour circle."""
    h, w = size, size
    cx, cy = w // 2, h // 2

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(hsv, (cx, cy), radius, (hue, sat, val), -1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # blur matches the preprocess pipeline and softens the circle edge
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    return PreprocessResult(
        roi=bgr,
        hsv=hsv,
        gray=gray,
        mask=mask,
        contour=contour,
        centroid=(cx, cy),
        bbox=(cx - radius, cy - radius, 2 * radius, 2 * radius),
        area=float(cv2.contourArea(contour)),
        found=True,
    )

def make_decision(
    label: ColourID = ColourID.RED,
    confidence: float = 0.90,
    rule: str = "colour",
    priority: int = 30,
) -> Decision:
    return Decision(label=label, confidence=confidence, rule=rule, priority=priority)

def make_features(
    mask_pixels: int = 2827,
    sat_mean: float = 150.0,
    val_mean: float = 150.0,
    highlight_ratio: float = 0.0,
    hue_hist: np.ndarray | None = None,
    hue_peak_width: int = 20,
    texture_variance: float = 100.0,
    circularity: float = 0.90,
    aspect_ratio: float = 1.0,
    solidity: float = 0.95,
) -> Features:
    """build a Features object with all-passing defaults; override specific fields to trigger rules."""
    if hue_hist is None:
        # neutral green-ish hue, away from red/orange ranges; will be replaced in colour tests
        h = np.zeros(180)
        h[45:56] = 1.0 / 11
        hue_hist = h
    return Features(
        mask_pixels=mask_pixels,
        sat_mean=sat_mean,
        val_mean=val_mean,
        highlight_ratio=highlight_ratio,
        hue_hist=hue_hist,
        hue_peak_width=hue_peak_width,
        texture_variance=texture_variance,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
    )

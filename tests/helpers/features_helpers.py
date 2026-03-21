# tests/helpers/features_helpers.py

import cv2
import numpy as np

from src.vision.preprocess import PreprocessResult

def make_feature_result(hue=10, sat=200, val=180, radius=30, size=100) -> PreprocessResult:
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

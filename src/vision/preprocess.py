# src/vision/preprocess.py
"""
Frame preprocessing for the M&M sorting pipeline.

Takes a raw BGR camera frame, extracts a region of interest, generates
a binary object mask, and locates the largest contour with its centroid
and bounding box.

Usage:
    prep = Preprocessor(cfg)
    result = prep.process(frame)
    if result.found:
        print(result.centroid, result.area)

NOTE: object detection relies on saturation thresholding. this assumes
a matte black background with coloured objects on top. low-saturation
objects (grey, black, white) will not be detected by design.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config

log = logging.getLogger(__name__)

@dataclass
class PreprocessResult:
    """
    output of a single preprocessing pass.

    Coordinate spaces:
        centroid, bbox: full-frame coordinates (always, regardless of roi_enabled)
        roi, hsv, gray, mask, contour: ROI-local coordinates / ROI-space arrays

    when found is False, contour/centroid/bbox fields are None
    and area is 0. roi, hsv, gray, and mask are always populated.
    """
    roi: np.ndarray
    hsv: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    contour: np.ndarray | None
    centroid: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None
    area: float
    found: bool

class Preprocessor:
    """
    stateless frame preprocessor. all parameters are read from config
    at construction time.
    """

    def __init__(self, config: Config):
        self._cfg = config.preprocess

        log.info(
            "preprocessor initialised -- blur=%d, sat_thresh=%d, min_area=%d",
            self._cfg["blur_kernel"],
            self._cfg["sat_threshold"],
            self._cfg["min_area"],
        )

        # tracks previous found state for transition logging
        self._prev_found: bool = False

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """
        run the full preprocessing pipeline on a BGR frame.
        """
        roi, (ox, oy) = self._extract_roi(frame)
        blur_k = self._cfg["blur_kernel"]
        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        mask = self._make_mask(hsv)
        contour, area = self._find_largest_contour(mask)

        if contour is None or area < self._cfg["min_area"]:
            if self._prev_found:
                log.info("object lost")
            self._prev_found = False
            return PreprocessResult(
                roi=roi, hsv=hsv, gray=gray, mask=mask,
                contour=None, centroid=None, bbox=None,
                area=0.0, found=False,
            )

        cx, cy = self._compute_centroid(contour)
        centroid = (cx + ox, cy + oy)
        bx, by, bw, bh = cv2.boundingRect(contour)
        bbox = (bx + ox, by + oy, bw, bh)

        if not self._prev_found:
            log.info("object found -- area=%.0f centroid=(%d, %d)", area, *centroid)
        log.debug("preprocess -- area=%.0f centroid=(%d, %d) bbox=%s", area, *centroid, bbox)
        self._prev_found = True

        return PreprocessResult(
            roi=roi, hsv=hsv, gray=gray, mask=mask,
            contour=contour, centroid=centroid, bbox=bbox,
            area=area, found=True,
        )

    def _extract_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        crop the centre region of the frame if roi is enabled.
        returns (roi, (x_offset, y_offset)) so callers can translate
        ROI-local coordinates back to full-frame coordinates.
        """
        if not self._cfg.get("roi_enabled", False):
            return frame, (0, 0)

        h, w = frame.shape[:2]
        frac = self._cfg.get("roi_fraction", 0.9)
        dw = int(w * (1 - frac) / 2)
        dh = int(h * (1 - frac) / 2)
        return frame[dh:h - dh, dw:w - dw].copy(), (dw, dh)

    def _make_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        generate a binary mask by thresholding the saturation channel
        and applying morphological cleanup.
        """
        sat = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, self._cfg["sat_threshold"], 255, cv2.THRESH_BINARY)

        morph_k = self._cfg["morph_kernel"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        mask = cv2.erode(mask, kernel, iterations=self._cfg.get("morph_erode_iter", 1))
        mask = cv2.dilate(mask, kernel, iterations=self._cfg.get("morph_dilate_iter", 2))

        return mask

    @staticmethod
    def _find_largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        find the largest contour by area. returns (contour, area)
        or (None, 0.0) if no contours exist.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        largest = max(contours, key=cv2.contourArea)
        return largest, cv2.contourArea(largest)

    @staticmethod
    def _compute_centroid(contour: np.ndarray) -> tuple[int, int]:
        """
        compute the centroid of a contour using image moments.
        falls back to the first contour point if the area is zero.
        """
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            return (cx, cy)

        # degenerate contour, use the first point
        pt = contour[0][0]
        return (int(pt[0]), int(pt[1]))

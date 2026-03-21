# src/vision/features.py
"""
Feature extraction for the M&M sorting pipeline.

Computes geometric, colour, and texture descriptors from a
PreprocessResult and returns them as a structured Features object.

Usage:
    extractor = FeatureExtractor(cfg)
    features = extractor.extract(result)
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config
from src.vision.preprocess import PreprocessResult

log = logging.getLogger(__name__)

@dataclass
class Features:
    mask_pixels: int
    sat_mean: float
    highlight_ratio: float
    hue_hist: np.ndarray
    hue_peak_width: int
    texture_variance: float
    circularity: float
    aspect_ratio: float
    solidity: float

class FeatureExtractor:
    """
    extracts quantitative measurements from a PreprocessResult.
    does not perform classification or decision making.
    """

    def __init__(self, config: Config):
        self._cfg = config.features
        
        log.info(
            "feature extractor initialised -- hue_bins=%d, highlight_value=%d",
            self._cfg["hue_bins"],
            self._cfg["highlight_value"],
        )

    def extract(self, result: PreprocessResult) -> Features:
        """
        compute all features from a valid PreprocessResult.
        raises ValueError if result.found is False or mask has no pixels.
        """
        if not result.found:
            raise ValueError("feature extraction requires result.found == True")

        assert result.contour is not None

        mask_pixels = int(np.count_nonzero(result.mask))
        if mask_pixels == 0:
            raise ValueError("mask contains no foreground pixels")

        sat_mean = self._compute_sat_mean(result.hsv, result.mask)
        highlight_ratio = self._compute_highlight_ratio(result.hsv, result.mask)
        hue_hist = self._compute_hue_hist(result.hsv, result.mask)
        hue_peak_width = self._compute_hue_peak_width(hue_hist)
        texture_variance = self._compute_texture_variance(result.gray, result.mask)
        circularity = self._compute_circularity(result.contour)
        aspect_ratio = self._compute_aspect_ratio(result.contour)
        solidity = self._compute_solidity(result.contour)

        log.debug(
            "features -- sat=%.2f highlight=%.3f circ=%.3f ar=%.3f solid=%.3f tex=%.1f peak_w=%d",
            sat_mean, highlight_ratio, circularity, aspect_ratio, solidity,
            texture_variance, hue_peak_width,
        )

        return Features(
            mask_pixels=mask_pixels,
            sat_mean=sat_mean,
            highlight_ratio=highlight_ratio,
            hue_hist=hue_hist,
            hue_peak_width=hue_peak_width,
            texture_variance=texture_variance,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
        )

    def _compute_sat_mean(self, hsv: np.ndarray, mask: np.ndarray) -> float:
        """
        mean saturation of object pixels inside the mask.

        sat_mean = mean(S_mask)

        high values indicate strong colour intensity while low values
        suggest grey or desaturated objects.
        """
        sat = hsv[:, :, 1]
        return float(sat[mask > 0].mean())

    def _compute_highlight_ratio(self, hsv: np.ndarray, mask: np.ndarray) -> float:
        """
        fraction of masked pixels exceeding the brightness threshold.

        highlight_ratio = count(V > highlight_value) / count(mask)

        large ratios indicate specular highlights or highly reflective
        objects rather than matte candy surfaces.
        """
        val = hsv[:, :, 2]
        threshold = self._cfg["highlight_value"]
        masked_val = val[mask > 0]
        return float((masked_val > threshold).sum() / len(masked_val))

    def _compute_hue_hist(self, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        normalised histogram of hue values inside the mask.

        smoothed with a Gaussian kernel of width hue_smooth_sigma.
        uses circular padding to handle the wraparound at hue=0/180.
        useful for colour classification via hue distribution matching.
        """
        bins = self._cfg["hue_bins"]
        hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        hist = hist.flatten().astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total

        sigma = self._cfg["hue_smooth_sigma"]
        if sigma > 0:
            # pad = 4*sigma ensures the kernel (half-width ~3*sigma) fits within the padding
            pad = int(sigma * 4)
            padded = np.concatenate([hist[-pad:], hist, hist[:pad]])
            padded_2d = padded.reshape(1, -1).astype(np.float32)
            smoothed = cv2.GaussianBlur(padded_2d, (0, 0), sigma)
            hist = smoothed.flatten()[pad:pad + bins].astype(np.float64)

        return hist

    def _compute_hue_peak_width(self, hue_hist: np.ndarray) -> int:
        """
        width of the dominant hue cluster in histogram bins.

        extends left and right from the peak bin while values remain
        above hue_peak_ratio * peak_value. uses modulo indexing so
        wraparound colours like red are measured correctly.
        """
        if hue_hist.max() == 0:
            return 0

        ratio = self._cfg["hue_peak_ratio"]
        threshold = hue_hist.max() * ratio
        peak_idx = int(np.argmax(hue_hist))
        bins = len(hue_hist)
        # each direction is capped at half the wheel to prevent double-counting
        max_spread = bins // 2

        left = 1
        while left <= max_spread and hue_hist[(peak_idx - left) % bins] >= threshold:
            left += 1

        right = 1
        while right <= max_spread and hue_hist[(peak_idx + right) % bins] >= threshold:
            right += 1

        return min(left + right - 1, bins)

    def _compute_texture_variance(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """
        laplacian variance of grayscale pixels inside the mask.

        variance = Var(Laplacian(gray)[mask])

        smooth candy surfaces produce low variance while textured
        objects produce higher values.
        """
        assert gray.ndim == 2
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian[mask > 0].var())

    def _compute_circularity(self, contour: np.ndarray) -> float:
        """
        measures how close the contour shape is to a circle.

        circularity = 4 * pi * area / perimeter^2

        values near 1 indicate round shapes. lower values indicate
        irregular or elongated objects.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return float(4 * np.pi * area / (perimeter ** 2))

    def _compute_aspect_ratio(self, contour: np.ndarray) -> float:
        """
        normalised bounding box aspect ratio: max(w,h) / min(w,h).

        always >= 1.0 regardless of object orientation.
        values near 1.0 indicate square-like objects (M&Ms ~1.0–1.35).
        """
        _, _, w, h = cv2.boundingRect(contour)
        return float(max(w, h) / min(w, h))

    def _compute_solidity(self, contour: np.ndarray) -> float:
        """
        ratio of contour area to convex hull area.

        solidity = contour_area / convex_hull_area

        values near 1 indicate compact shapes. irregular or concave
        contours produce lower solidity scores.
        """
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0.0
        return float(area / hull_area)

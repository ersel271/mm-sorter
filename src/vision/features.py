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

        sigma = self._cfg.get("hue_smooth_sigma", 3)
        if sigma > 0:
            # pad = 4*sigma ensures the kernel (half-width ~3*sigma) fits within the padding
            pad = int(sigma * 4)
            padded = np.concatenate([hist[-pad:], hist, hist[:pad]])
            padded_2d = padded.reshape(1, -1).astype(np.float32)
            smoothed = cv2.GaussianBlur(padded_2d, (0, 0), sigma)
            hist = smoothed.flatten()[pad:pad + bins].astype(np.float64)

        return hist

    def _compute_hue_peak_width(self, hue_hist: np.ndarray) -> int:
        pass

    def _compute_texture_variance(self, gray: np.ndarray, mask: np.ndarray) -> float:
        pass

    def _compute_circularity(self, contour: np.ndarray) -> float:
        pass

    def _compute_aspect_ratio(self, contour: np.ndarray) -> float:
        pass

    def _compute_solidity(self, contour: np.ndarray) -> float:
        pass
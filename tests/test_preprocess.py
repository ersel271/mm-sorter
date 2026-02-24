# tests/test_preprocess.py

import cv2
import numpy as np
import pytest

from config import Config
from src.vision.preprocess import Preprocessor, PreprocessResult

@pytest.fixture
def prep() -> Preprocessor:
    return Preprocessor(Config())

def make_frame(width=1920, height=1080) -> np.ndarray:
    """plain black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def draw_circle(frame, centre=None, radius=50, colour_bgr=(0, 0, 255), sat=200):
    """
    draw a filled circle with a given saturation onto a black frame.
    uses HSV to control saturation precisely, then converts back to BGR.
    """
    if centre is None:
        h, w = frame.shape[:2]
        centre = (w // 2, h // 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bgr_ref = np.uint8([[colour_bgr]])
    hsv_ref = cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2HSV)[0][0]

    colour_hsv = (int(hsv_ref[0]), sat, int(hsv_ref[2]))
    cv2.circle(hsv, centre, radius, colour_hsv, -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def draw_saturated_circle(frame, centre=None, radius=50):
    """shorthand: bright circle with high saturation drawn onto existing frame."""
    if centre is None:
        h, w = frame.shape[:2]
        centre = (w // 2, h // 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.circle(hsv, centre, radius, (0, 200, 200), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

@pytest.mark.smoke
@pytest.mark.unit
class TestPreprocessResult:
    """verify PreprocessResult fields for found and not-found cases."""

    def test_found_object(self, prep):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert result.found is True
        assert result.contour is not None
        assert result.centroid is not None
        assert result.bbox is not None
        assert result.area > 0

    def test_empty_frame(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert result.found is False
        assert result.contour is None
        assert result.centroid is None
        assert result.bbox is None
        assert result.area == 0.0

    def test_roi_hsv_gray_mask_always_present(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert result.roi is not None
        assert result.hsv is not None
        assert result.gray is not None
        assert result.mask is not None

@pytest.mark.unit
class TestMask:
    """verify binary mask generation from saturation thresholding."""

    def test_mask_shape_matches_roi(self, prep):
        frame = draw_saturated_circle(make_frame())
        result = prep.process(frame)
        assert result.mask.shape == result.roi.shape[:2]

    def test_mask_is_binary(self, prep):
        frame = draw_saturated_circle(make_frame())
        result = prep.process(frame)
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})

    def test_mask_has_nonzero_for_coloured_object(self, prep):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert np.count_nonzero(result.mask) > 0

    def test_mask_all_zero_for_black_frame(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert np.count_nonzero(result.mask) == 0

@pytest.mark.unit
class TestContour:
    """verify contour detection selects the largest object."""

    def test_largest_contour_selected(self, prep):
        frame = make_frame()
        # small circle
        frame = draw_saturated_circle(frame, centre=(300, 300), radius=20)
        # large circle
        frame = draw_saturated_circle(frame, centre=(960, 540), radius=80)
        result = prep.process(frame)
        assert result.found is True
        cx, cy = result.centroid
        assert abs(cx - 960) < 20
        assert abs(cy - 540) < 20

    def test_object_below_min_area_rejected(self, prep):
        frame = make_frame()
        # tiny circle, area < min_area (500)
        frame = draw_saturated_circle(frame, radius=5)
        result = prep.process(frame)
        assert result.found is False

@pytest.mark.unit
class TestCentroid:
    """verify centroid is computed near the object centre."""

    def test_centroid_near_frame_centre(self, prep):
        centre = (960, 540)
        frame = draw_saturated_circle(make_frame(), centre=centre, radius=60)
        result = prep.process(frame)
        assert result.found is True
        cx, cy = result.centroid
        assert abs(cx - centre[0]) < 15
        assert abs(cy - centre[1]) < 15

    def test_centroid_off_centre(self, prep):
        centre = (400, 200)
        frame = draw_saturated_circle(make_frame(), centre=centre, radius=60)
        result = prep.process(frame)
        assert result.found is True
        cx, cy = result.centroid
        assert abs(cx - centre[0]) < 15
        assert abs(cy - centre[1]) < 15

@pytest.mark.unit
class TestBoundingBox:
    """verify bounding box encloses the detected object."""

    def test_bbox_contains_centroid(self, prep):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert result.found is True
        x, y, w, h = result.bbox
        cx, cy = result.centroid
        assert x <= cx <= x + w
        assert y <= cy <= y + h

    def test_bbox_dimensions_reasonable(self, prep):
        radius = 60
        frame = draw_saturated_circle(make_frame(), radius=radius)
        result = prep.process(frame)
        assert result.found is True
        _, _, w, h = result.bbox
        # bbox should roughly be 2*radius, with some tolerance for
        # blur and morphological operations
        assert radius < w < radius * 3
        assert radius < h < radius * 3

@pytest.mark.unit
class TestROI:
    """verify ROI extraction with and without cropping enabled."""

    def test_roi_disabled_returns_full_frame(self, prep):
        frame = make_frame(1920, 1080)
        result = prep.process(frame)
        assert result.roi.shape[:2] == (1080, 1920)

    def test_roi_enabled_crops_frame(self):
        cfg = Config()
        data = cfg.as_dict()
        data["preprocess"]["roi_enabled"] = True
        data["preprocess"]["roi_fraction"] = 0.5

        import yaml
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f)
            cfg2 = Config(path)

        p = Preprocessor(cfg2)
        frame = make_frame(1920, 1080)
        result = p.process(frame)
        # 50% crop: roughly 540x960
        h, w = result.roi.shape[:2]
        assert h < 1080
        assert w < 1920
        assert abs(h - 540) < 10
        assert abs(w - 960) < 10

@pytest.mark.unit
class TestColourConversions:
    """verify HSV and grayscale conversions match expected shapes and types."""

    def test_hsv_has_three_channels(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert len(result.hsv.shape) == 3
        assert result.hsv.shape[2] == 3

    def test_gray_is_single_channel(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert len(result.gray.shape) == 2

    def test_hsv_dtype_is_uint8(self, prep):
        frame = make_frame()
        result = prep.process(frame)
        assert result.hsv.dtype == np.uint8

@pytest.mark.integration
class TestPreprocessIntegration:
    """verify the full preprocessing pipeline end to end."""

    def test_multiple_objects_finds_largest(self, prep):
        frame = make_frame()
        radii = [20, 40, 80, 30]
        centres = [(200, 200), (600, 400), (960, 540), (1500, 800)]
        for c, r in zip(centres, radii):
            frame = draw_saturated_circle(frame, centre=c, radius=r)
        result = prep.process(frame)
        assert result.found is True
        cx, cy = result.centroid
        assert abs(cx - 960) < 25
        assert abs(cy - 540) < 25

    def test_pipeline_returns_consistent_shapes(self, prep):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        h, w = result.roi.shape[:2]
        assert result.hsv.shape == (h, w, 3)
        assert result.gray.shape == (h, w)
        assert result.mask.shape == (h, w)

    def test_low_saturation_object_not_detected(self, prep):
        frame = make_frame()
        # draw a gray circle (low saturation) directly in BGR
        cv2.circle(frame, (960, 540), 60, (128, 128, 128), -1)
        result = prep.process(frame)
        assert result.found is False

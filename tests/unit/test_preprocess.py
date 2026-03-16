# tests/test_preprocess.py

import cv2
import numpy as np
import pytest

from config import Config
from tests.helpers.config_helpers import write_config
from src.vision.preprocess import Preprocessor, PreprocessResult
from tests.helpers.image_helpers import make_frame, draw_circle, draw_saturated_circle

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

    def test_roi_enabled_centroid_and_bbox_in_full_frame_coordinates(self, tmp_path):
        frame_w, frame_h = 1920, 1080
        roi_fraction = 0.9
        # deterministic offset from the same formula used in _extract_roi
        ox = int(frame_w * (1 - roi_fraction) / 2)   # 96
        oy = int(frame_h * (1 - roi_fraction) / 2)   # 54

        cfg = Config()
        data = cfg.as_dict()
        data["preprocess"]["roi_enabled"] = True
        data["preprocess"]["roi_fraction"] = roi_fraction
        path = write_config(data, tmp_path)
        p = Preprocessor(Config(path))

        # object placed well off-centre so ROI-local and full-frame coords diverge clearly
        centre = (400, 300)
        roi_local = (centre[0] - ox, centre[1] - oy)  # (304, 246) without fix
        frame = draw_saturated_circle(make_frame(frame_w, frame_h), centre=centre, radius=60)
        result = p.process(frame)

        assert result.found is True

        cx, cy = result.centroid
        # positive: centroid must be near the full-frame position
        assert abs(cx - centre[0]) < 5, f"centroid x={cx} not near full-frame {centre[0]}"
        assert abs(cy - centre[1]) < 5, f"centroid y={cy} not near full-frame {centre[1]}"
        # negative: centroid must NOT be near the ROI-local position (catches regression)
        assert abs(cx - roi_local[0]) > 50, f"centroid x={cx} looks like ROI-local {roi_local[0]}"
        assert abs(cy - roi_local[1]) > 50, f"centroid y={cy} looks like ROI-local {roi_local[1]}"

        # bbox origin must also be in full-frame coordinates, not ROI-local
        radius = 60
        x, y, w, h = result.bbox
        assert abs(x - (centre[0] - radius)) < 20, f"bbox x={x} not near global {centre[0] - radius}"
        assert abs(y - (centre[1] - radius)) < 20, f"bbox y={y} not near global {centre[1] - radius}"
        # centroid must fall inside the bbox (sanity check)
        assert x <= cx <= x + w
        assert y <= cy <= y + h

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

# tests/integration/test_preprocess_integration.py

import cv2
import pytest

from tests.helpers.image_helpers import make_frame, draw_saturated_circle

@pytest.mark.smoke
@pytest.mark.regression
class TestPreprocessIntegration:
    """verify the full preprocessing pipeline end to end."""

    def test_multiple_objects_finds_largest(self, prep):
        frame = make_frame()
        radii = [20, 40, 80, 30]
        centres = [(200, 200), (600, 400), (960, 540), (1500, 800)]
        for c, r in zip(centres, radii, strict=False):
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
        cv2.circle(frame, (960, 540), 60, (128, 128, 128), -1)
        result = prep.process(frame)
        assert result.found is False

# tests/property/test_preprocess_properties.py

import cv2
import pytest
import numpy as np

from hypothesis import given, strategies as st, assume

from src.vision.preprocess import Preprocessor
from tests.helpers.config_helpers import make_config

frame_w = 1920
frame_h = 1080

_prep = Preprocessor(make_config())


def make_frame():
    return np.zeros((frame_h, frame_w, 3), dtype=np.uint8)


@pytest.mark.regression
class TestPreprocessProperties:
    """property tests for Preprocessor spatial output invariants"""

    # detected centroid must always lie within the reported bounding box
    @given(
        st.integers(min_value=50, max_value=frame_w - 50),
        st.integers(min_value=50, max_value=frame_h - 50),
        st.integers(min_value=20, max_value=120),
    )
    def test_centroid_inside_bbox(self, cx, cy, r):
        frame = make_frame()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.circle(hsv, (cx, cy), r, (0, 200, 200), -1)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = _prep.process(frame)
        assume(result.found)

        x, y, w, h = result.bbox
        c_x, c_y = result.centroid
        assert x <= c_x <= x + w
        assert y <= c_y <= y + h

    # roi, hsv, gray and mask outputs must share identical spatial dimensions
    @given(
        st.integers(min_value=50, max_value=frame_w - 50),
        st.integers(min_value=50, max_value=frame_h - 50),
        st.integers(min_value=20, max_value=120),
    )
    def test_pipeline_outputs_same_spatial_dimensions(self, cx, cy, r):
        frame = make_frame()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.circle(hsv, (cx, cy), r, (0, 200, 200), -1)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = _prep.process(frame)
        assume(result.found)

        h, w = result.roi.shape[:2]
        assert result.hsv.shape[:2] == (h, w)
        assert result.gray.shape[:2] == (h, w)
        assert result.mask.shape[:2] == (h, w)

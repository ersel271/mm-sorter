# tests/property/test_preprocess_properties.py

import pytest
import numpy as np
import cv2

from hypothesis import given, strategies as st, assume

from src.vision.preprocess import Preprocessor
from config import Config

frame_w = 1920
frame_h = 1080

def make_frame():
    return np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

# detected centroid must always lie within the reported bounding box
@given(
    st.integers(min_value=50, max_value=frame_w - 50),
    st.integers(min_value=50, max_value=frame_h - 50),
    st.integers(min_value=20, max_value=120),
)
@pytest.mark.property
def test_centroid_inside_bbox(cx, cy, r):
    prep = Preprocessor(Config())
    frame = make_frame()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.circle(hsv, (cx, cy), r, (0, 200, 200), -1)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = prep.process(frame)
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
@pytest.mark.property
def test_pipeline_outputs_same_spatial_dimensions(cx, cy, r):
    prep = Preprocessor(Config())
    frame = make_frame()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.circle(hsv, (cx, cy), r, (0, 200, 200), -1)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = prep.process(frame)
    assume(result.found)

    h, w = result.roi.shape[:2]
    assert result.hsv.shape[:2] == (h, w)
    assert result.gray.shape[:2] == (h, w)
    assert result.mask.shape[:2] == (h, w)

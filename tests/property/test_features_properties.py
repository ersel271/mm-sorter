# tests/property/test_features_properties.py

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st

from src.vision.features import FeatureExtractor
from tests.helpers.config_helpers import make_config
from tests.helpers.vision_helpers import make_preprocess_result

_extractor = FeatureExtractor(make_config())

@pytest.mark.regression
class TestFeaturesProperties:
    """property tests for FeatureExtractor output invariants"""

    # sat_mean is bounded by the uint8 saturation channel range [0, 255]
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
    )
    def test_sat_mean_in_valid_range(self, hue, sat, val):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val))
        assert 0.0 <= features.sat_mean <= 255.0

    # val_mean is bounded by the uint8 value channel range [0, 255]
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=0, max_value=255),
    )
    def test_val_mean_in_valid_range(self, hue, sat, val):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val))
        assert 0.0 <= features.val_mean <= 255.0

    # highlight_ratio is always a fraction between 0 and 1
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=255),
    )
    def test_highlight_ratio_in_valid_range(self, hue, sat, val):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val))
        assert 0.0 <= features.highlight_ratio <= 1.0

    # normalised hue histogram must sum to approximately one
    @given(
        st.integers(min_value=5, max_value=174),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
    )
    def test_hue_hist_sum_near_one(self, hue, sat, val):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val))
        assert abs(features.hue_hist.sum() - 1.0) < 0.02

    # all hue histogram bins must be non-negative
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
    )
    def test_hue_hist_all_nonnegative(self, hue, sat, val):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val))
        assert np.all(features.hue_hist >= 0)

    # circularity of a discrete circle contour must lie in [0, 1]
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=10, max_value=35),
    )
    def test_circularity_in_valid_range(self, hue, sat, val, radius):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val, radius=radius))
        assert 0.0 <= features.circularity <= 1.0

    # solidity of a circle is always in [0, 1]
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=10, max_value=35),
    )
    def test_solidity_in_valid_range(self, hue, sat, val, radius):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val, radius=radius))
        assert 0.0 <= features.solidity <= 1.0

    # normalised aspect ratio must always be >= 1.0 for any contour orientation
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=10, max_value=35),
    )
    def test_aspect_ratio_never_below_one(self, hue, sat, val, radius):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val, radius=radius))
        assert features.aspect_ratio >= 1.0

    # Laplacian variance is always non-negative
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=10, max_value=35),
    )
    def test_texture_variance_nonnegative(self, hue, sat, val, radius):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val, radius=radius))
        assert features.texture_variance >= 0.0

    # hue_peak_width must fall within valid histogram bounds for any input
    @given(
        st.integers(min_value=0, max_value=179),
        st.integers(min_value=80, max_value=255),
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=10, max_value=35),
    )
    def test_hue_peak_width_in_bounds(self, hue, sat, val, radius):
        features = _extractor.extract(make_preprocess_result(hue=hue, sat=sat, val=val, radius=radius))
        assert 1 <= features.hue_peak_width <= 180

    # hue_peak_width for a boundary hue must match a mid-range hue within tolerance
    @given(st.integers(min_value=15, max_value=30))
    @settings(max_examples=20)
    def test_hue_peak_width_consistent_at_boundary(self, radius):
        f_boundary = _extractor.extract(make_preprocess_result(hue=0, radius=radius))
        f_midrange = _extractor.extract(make_preprocess_result(hue=90, radius=radius))
        assert abs(f_boundary.hue_peak_width - f_midrange.hue_peak_width) <= 5

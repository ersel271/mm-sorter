# tests/property/test_classifier_properties.py

import pytest
from hypothesis import given, settings, strategies as st

from config.constants import ColourID
from src.vision import Classifier
from tests.helpers.config_helpers import make_config
from tests.helpers.vision_helpers import make_features

_classifier = Classifier(make_config())


@pytest.mark.regression
class TestClassifierProperties:
    """property tests for Classifier output invariants"""

    # sat_mean below sat_min must always result in NON_MM regardless of other fields
    @given(st.floats(min_value=0.0, max_value=59.0))
    def test_low_sat_always_rejected(self, sat_mean):
        # all other fields pass their respective rules so LowSaturationRule is the sole trigger
        f = make_features(sat_mean=sat_mean)
        d = _classifier.classify(f)
        assert d.label == ColourID.NON_MM

    # circularity below circularity_min must always result in NON_MM
    @given(st.floats(min_value=0.0, max_value=0.74))
    def test_low_circularity_always_rejected(self, circularity):
        # sat_mean=150 ensures no stage-1 rule fires; shape rule is the sole trigger
        f = make_features(circularity=circularity, sat_mean=150.0)
        d = _classifier.classify(f)
        assert d.label == ColourID.NON_MM

    # if a stage-1 rule fires the winning priority must be <= 10
    @given(st.floats(min_value=0.0, max_value=59.0))
    def test_stage1_rule_fires_priority_invariant(self, sat_mean):
        f = make_features(sat_mean=sat_mean)
        d = _classifier.classify(f)
        # LowSaturationRule always fires here (sat_mean < sat_min=60)
        assert d.priority <= 10

    # confidence must always lie in [0.0, 1.0] for any input
    @given(
        st.floats(min_value=0.0, max_value=255.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=1, max_value=180),
        st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=200)
    def test_confidence_always_in_unit_range(self, sat_mean, highlight_ratio, circularity, solidity, hue_peak_width, texture_variance):
        f = make_features(
            sat_mean=sat_mean,
            highlight_ratio=highlight_ratio,
            circularity=circularity,
            solidity=solidity,
            hue_peak_width=hue_peak_width,
            texture_variance=texture_variance,
        )
        d = _classifier.classify(f)
        assert 0.0 <= d.confidence <= 1.0

    # result label is never None -- classifier always returns a concrete ColourID
    @given(
        st.floats(min_value=0.0, max_value=255.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_label_never_none(self, sat_mean, circularity, solidity):
        f = make_features(sat_mean=sat_mean, circularity=circularity, solidity=solidity)
        d = _classifier.classify(f)
        assert d.label is not None

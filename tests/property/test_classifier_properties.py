# tests/property/test_classifier_properties.py

import pytest
from hypothesis import given, settings, strategies as st

from config import Config
from config.constants import ColourID
from src.vision import Classifier
from tests.helpers.features_helpers import make_features

# sat_mean below sat_min must always result in NON_MM regardless of other fields
@given(st.floats(min_value=0.0, max_value=59.0))
@pytest.mark.property
def test_low_sat_always_rejected(sat_mean):
    # all other fields pass their respective rules so LowSaturationRule is the sole trigger
    f = make_features(sat_mean=sat_mean)
    d = Classifier(Config()).classify(f)
    assert d.label == ColourID.NON_MM

# circularity below circularity_min must always result in NON_MM
@given(st.floats(min_value=0.0, max_value=0.74))
@pytest.mark.property
def test_low_circularity_always_rejected(circularity):
    # sat_mean=150 ensures no stage-1 rule fires; shape rule is the sole trigger
    f = make_features(circularity=circularity, sat_mean=150.0)
    d = Classifier(Config()).classify(f)
    assert d.label == ColourID.NON_MM

# if a stage-1 rule fires the winning priority must be <= 10
@given(st.floats(min_value=0.0, max_value=59.0))
@pytest.mark.property
def test_stage1_rule_fires_priority_invariant(sat_mean):
    f = make_features(sat_mean=sat_mean)
    d = Classifier(Config()).classify(f)
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
@pytest.mark.property
def test_confidence_always_in_unit_range(sat_mean, highlight_ratio, circularity, solidity, hue_peak_width, texture_variance):
    f = make_features(
        sat_mean=sat_mean,
        highlight_ratio=highlight_ratio,
        circularity=circularity,
        solidity=solidity,
        hue_peak_width=hue_peak_width,
        texture_variance=texture_variance,
    )
    d = Classifier(Config()).classify(f)
    assert 0.0 <= d.confidence <= 1.0

# result label is never None — classifier always returns a concrete ColourID
@given(
    st.floats(min_value=0.0, max_value=255.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
@pytest.mark.property
def test_label_never_none(sat_mean, circularity, solidity):
    f = make_features(sat_mean=sat_mean, circularity=circularity, solidity=solidity)
    d = Classifier(Config()).classify(f)
    assert d.label is not None

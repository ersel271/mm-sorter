# tests/unit/test_panel_feature.py

import pytest
import numpy as np

from src.ui.panels import FeaturePanel
from src.ui.panel import PANEL_W
from tests.helpers.features_helpers import make_features

@pytest.mark.unit
class TestFeaturePanelShape:
    """verify FeaturePanel output matches the (panel_h, PANEL_W, 3) contract."""

    def test_shape_no_features(self, feature_panel: FeaturePanel) -> None:
        out = feature_panel.render(None, None, 400)
        assert out.shape == (400, PANEL_W, 3)

    def test_shape_with_features(self, feature_panel: FeaturePanel) -> None:
        out = feature_panel.render(make_features(), None, 400)
        assert out.shape == (400, PANEL_W, 3)

    def test_various_heights_produce_correct_shape(self, feature_panel: FeaturePanel) -> None:
        for h in (100, 300, 480, 720):
            out = feature_panel.render(None, None, h)
            assert out.shape == (h, PANEL_W, 3)

@pytest.mark.unit
class TestFeaturePanelDtype:
    """verify FeaturePanel always returns a uint8 array."""

    def test_dtype_no_features(self, feature_panel: FeaturePanel) -> None:
        assert feature_panel.render(None, None, 300).dtype == np.uint8

    def test_dtype_with_features(self, feature_panel: FeaturePanel) -> None:
        assert feature_panel.render(make_features(), None, 300).dtype == np.uint8

@pytest.mark.unit
class TestFeaturePanelSafety:
    """verify FeaturePanel does not crash on edge-case inputs."""

    def test_features_none_does_not_crash(self, feature_panel: FeaturePanel) -> None:
        assert feature_panel.render(None, None, 300) is not None

    def test_features_with_zero_values_does_not_crash(self, feature_panel: FeaturePanel) -> None:
        f = make_features(sat_mean=0.0, circularity=0.0, solidity=0.0,
                          aspect_ratio=1.0, texture_variance=0.0,
                          highlight_ratio=0.0, hue_peak_width=0)
        assert feature_panel.render(f, None, 300) is not None

    def test_features_exceeding_range_does_not_crash(self, feature_panel: FeaturePanel) -> None:
        # values above defined ranges must not cause exceptions; bars are clamped
        f = make_features(sat_mean=9999.0, texture_variance=9999.0, aspect_ratio=99.0)
        assert feature_panel.render(f, None, 300) is not None

    def test_panel_not_entirely_black_when_features_present(self, feature_panel: FeaturePanel) -> None:
        out = feature_panel.render(make_features(), None, 300)
        assert out.max() > 0

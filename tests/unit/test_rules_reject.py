# tests/unit/test_rules_reject.py

import pytest

from config import Config
from config.constants import ColourID
from src.vision.rules.reject import HighHighlightRule, LowSaturationRule, NarrowHuePeakRule
from tests.helpers.features_helpers import make_features

@pytest.mark.unit
class TestLowSaturationRule:
    """verify low-saturation foreign object rejection."""

    def test_fires_below_threshold(self):
        rule = LowSaturationRule(Config())
        d = rule.apply(make_features(sat_mean=10.0))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = LowSaturationRule(cfg)
        threshold = cfg.thresholds["sat_min"]
        d = rule.apply(make_features(sat_mean=float(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self):
        rule = LowSaturationRule(Config())
        d = rule.apply(make_features(sat_mean=200.0))
        assert d is None

    def test_confidence_at_zero_saturation(self):
        rule = LowSaturationRule(Config())
        d = rule.apply(make_features(sat_mean=0.0))
        assert d.confidence == pytest.approx(1.0)

    def test_confidence_formula_midpoint(self):
        cfg = Config()
        rule = LowSaturationRule(cfg)
        threshold = float(cfg.thresholds["sat_min"])
        # sat_mean = threshold / 2 → confidence = 1 - 0.5 = 0.5
        d = rule.apply(make_features(sat_mean=threshold / 2))
        assert d.confidence == pytest.approx(0.5, abs=0.01)

    def test_rule_name_and_priority(self):
        assert LowSaturationRule.name == "low_saturation"
        assert LowSaturationRule.priority == 10

@pytest.mark.unit
class TestHighHighlightRule:
    """verify specular highlight foreign object rejection."""

    def test_fires_above_threshold(self):
        rule = HighHighlightRule(Config())
        d = rule.apply(make_features(highlight_ratio=0.99))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = HighHighlightRule(cfg)
        threshold = cfg.thresholds["highlight_max"]
        d = rule.apply(make_features(highlight_ratio=float(threshold)))
        assert d is None

    def test_does_not_fire_below_threshold(self):
        rule = HighHighlightRule(Config())
        d = rule.apply(make_features(highlight_ratio=0.0))
        assert d is None

    def test_confidence_capped_at_one(self):
        rule = HighHighlightRule(Config())
        d = rule.apply(make_features(highlight_ratio=1.0))
        assert d.confidence <= 1.0

    def test_confidence_positive_when_firing(self):
        rule = HighHighlightRule(Config())
        d = rule.apply(make_features(highlight_ratio=0.99))
        assert d.confidence > 0.0

    def test_rule_name_and_priority(self):
        assert HighHighlightRule.name == "high_highlight"
        assert HighHighlightRule.priority == 10

@pytest.mark.unit
class TestNarrowHuePeakRule:
    """verify narrow hue peak foreign object rejection."""

    def test_fires_below_threshold(self):
        rule = NarrowHuePeakRule(Config())
        d = rule.apply(make_features(hue_peak_width=1))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = NarrowHuePeakRule(cfg)
        threshold = cfg.thresholds["hue_width_min"]
        d = rule.apply(make_features(hue_peak_width=int(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self):
        rule = NarrowHuePeakRule(Config())
        d = rule.apply(make_features(hue_peak_width=40))
        assert d is None

    def test_confidence_at_width_one(self):
        cfg = Config()
        rule = NarrowHuePeakRule(cfg)
        threshold = float(cfg.thresholds["hue_width_min"])
        d = rule.apply(make_features(hue_peak_width=1))
        expected = min(1.0 - 1.0 / threshold, 1.0)
        assert d.confidence == pytest.approx(expected, abs=0.01)

    def test_rule_name_and_priority(self):
        assert NarrowHuePeakRule.name == "narrow_hue_peak"
        assert NarrowHuePeakRule.priority == 10

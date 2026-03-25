# tests/unit/test_rules_shape.py

import pytest

from config import Config
from config.constants import ColourID
from src.vision.rules.shape import BadAspectRatioRule, HighTextureRule, LowCircularityRule, LowSolidityRule
from tests.helpers.features_helpers import make_features

@pytest.mark.unit
class TestLowCircularityRule:
    """verify non-circular shape rejection."""

    def test_fires_below_threshold(self):
        rule = LowCircularityRule(Config())
        d = rule.apply(make_features(circularity=0.30))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = LowCircularityRule(cfg)
        threshold = cfg.thresholds["circularity_min"]
        d = rule.apply(make_features(circularity=float(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self):
        rule = LowCircularityRule(Config())
        d = rule.apply(make_features(circularity=0.95))
        assert d is None

    def test_confidence_formula(self):
        cfg = Config()
        rule = LowCircularityRule(cfg)
        threshold = float(cfg.thresholds["circularity_min"])
        circ = threshold / 2
        d = rule.apply(make_features(circularity=circ))
        expected = min(1.0 - circ / threshold, 1.0)
        assert d.confidence == pytest.approx(expected, abs=0.01)

    def test_rule_name_and_priority(self):
        assert LowCircularityRule.name == "low_circularity"
        assert LowCircularityRule.priority == 20

@pytest.mark.unit
class TestBadAspectRatioRule:
    """verify elongated shape rejection."""

    def test_fires_above_threshold(self):
        rule = BadAspectRatioRule(Config())
        d = rule.apply(make_features(aspect_ratio=3.0))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = BadAspectRatioRule(cfg)
        threshold = cfg.thresholds["aspect_ratio_max"]
        d = rule.apply(make_features(aspect_ratio=float(threshold)))
        assert d is None

    def test_does_not_fire_below_threshold(self):
        rule = BadAspectRatioRule(Config())
        d = rule.apply(make_features(aspect_ratio=1.0))
        assert d is None

    def test_confidence_positive_when_firing(self):
        rule = BadAspectRatioRule(Config())
        d = rule.apply(make_features(aspect_ratio=3.0))
        assert d.confidence > 0.0

    def test_confidence_capped_at_one(self):
        rule = BadAspectRatioRule(Config())
        d = rule.apply(make_features(aspect_ratio=99.0))
        assert d.confidence <= 1.0

    def test_rule_name_and_priority(self):
        assert BadAspectRatioRule.name == "bad_aspect_ratio"
        assert BadAspectRatioRule.priority == 20

@pytest.mark.unit
class TestLowSolidityRule:
    """verify non-convex shape rejection."""

    def test_fires_below_threshold(self):
        rule = LowSolidityRule(Config())
        d = rule.apply(make_features(solidity=0.50))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = LowSolidityRule(cfg)
        threshold = cfg.thresholds["solidity_min"]
        d = rule.apply(make_features(solidity=float(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self):
        rule = LowSolidityRule(Config())
        d = rule.apply(make_features(solidity=0.98))
        assert d is None

    def test_confidence_formula(self):
        cfg = Config()
        rule = LowSolidityRule(cfg)
        threshold = float(cfg.thresholds["solidity_min"])
        sol = threshold / 2
        d = rule.apply(make_features(solidity=sol))
        expected = min(1.0 - sol / threshold, 1.0)
        assert d.confidence == pytest.approx(expected, abs=0.01)

    def test_rule_name_and_priority(self):
        assert LowSolidityRule.name == "low_solidity"
        assert LowSolidityRule.priority == 20

@pytest.mark.unit
class TestHighTextureRule:
    """verify high-texture surface rejection (placed at stage 2 due to M&M imprint)."""

    def test_fires_above_threshold(self):
        rule = HighTextureRule(Config())
        d = rule.apply(make_features(texture_variance=9000.0))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self):
        cfg = Config()
        rule = HighTextureRule(cfg)
        threshold = cfg.thresholds["texture_max"]
        d = rule.apply(make_features(texture_variance=float(threshold)))
        assert d is None

    def test_does_not_fire_below_threshold(self):
        rule = HighTextureRule(Config())
        d = rule.apply(make_features(texture_variance=50.0))
        assert d is None

    def test_confidence_positive_when_firing(self):
        rule = HighTextureRule(Config())
        d = rule.apply(make_features(texture_variance=9000.0))
        assert d.confidence > 0.0

    def test_confidence_capped_at_one(self):
        rule = HighTextureRule(Config())
        d = rule.apply(make_features(texture_variance=1e9))
        assert d.confidence <= 1.0

    def test_rule_name_and_priority(self):
        assert HighTextureRule.name == "high_texture"
        assert HighTextureRule.priority == 20

# tests/unit/test_rules_shape.py

import pytest

from config.constants import ColourID
from src.vision.rules.shape import BadAspectRatioRule, HighTextureRule, LowCircularityRule, LowSolidityRule
from tests.helpers.vision_helpers import make_features

@pytest.mark.regression
class TestLowCircularityRule:
    """verify non-circular shape rejection."""

    def test_fires_below_threshold(self, default_cfg):
        rule = LowCircularityRule(default_cfg)
        d = rule.apply(make_features(circularity=0.30))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self, default_cfg):
        rule = LowCircularityRule(default_cfg)
        threshold = default_cfg.thresholds["circularity_min"]
        d = rule.apply(make_features(circularity=float(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self, default_cfg):
        rule = LowCircularityRule(default_cfg)
        d = rule.apply(make_features(circularity=0.95))
        assert d is None

    def test_confidence_formula(self, default_cfg):
        rule = LowCircularityRule(default_cfg)
        threshold = float(default_cfg.thresholds["circularity_min"])
        circ = threshold / 2
        d = rule.apply(make_features(circularity=circ))
        expected = min((1.0 - circ / threshold) ** 0.5, 1.0)
        assert d.confidence == pytest.approx(expected, abs=0.01)

    def test_rule_name_and_priority(self):
        assert LowCircularityRule.name == "low_circularity"
        assert LowCircularityRule.priority == 20

@pytest.mark.regression
class TestBadAspectRatioRule:
    """verify elongated shape rejection."""

    def test_fires_above_threshold(self, default_cfg):
        rule = BadAspectRatioRule(default_cfg)
        d = rule.apply(make_features(aspect_ratio=3.0))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self, default_cfg):
        rule = BadAspectRatioRule(default_cfg)
        threshold = default_cfg.thresholds["aspect_ratio_max"]
        d = rule.apply(make_features(aspect_ratio=float(threshold)))
        assert d is None

    def test_does_not_fire_below_threshold(self, default_cfg):
        rule = BadAspectRatioRule(default_cfg)
        d = rule.apply(make_features(aspect_ratio=1.0))
        assert d is None

    def test_confidence_positive_when_firing(self, default_cfg):
        rule = BadAspectRatioRule(default_cfg)
        d = rule.apply(make_features(aspect_ratio=3.0))
        assert d.confidence > 0.0

    def test_confidence_capped_at_one(self, default_cfg):
        rule = BadAspectRatioRule(default_cfg)
        d = rule.apply(make_features(aspect_ratio=99.0))
        assert d.confidence <= 1.0

    def test_rule_name_and_priority(self):
        assert BadAspectRatioRule.name == "bad_aspect_ratio"
        assert BadAspectRatioRule.priority == 20

@pytest.mark.regression
class TestLowSolidityRule:
    """verify non-convex shape rejection."""

    def test_fires_below_threshold(self, default_cfg):
        rule = LowSolidityRule(default_cfg)
        d = rule.apply(make_features(solidity=0.50))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self, default_cfg):
        rule = LowSolidityRule(default_cfg)
        threshold = default_cfg.thresholds["solidity_min"]
        d = rule.apply(make_features(solidity=float(threshold)))
        assert d is None

    def test_does_not_fire_above_threshold(self, default_cfg):
        rule = LowSolidityRule(default_cfg)
        d = rule.apply(make_features(solidity=0.98))
        assert d is None

    def test_confidence_formula(self, default_cfg):
        rule = LowSolidityRule(default_cfg)
        threshold = float(default_cfg.thresholds["solidity_min"])
        sol = threshold / 2
        d = rule.apply(make_features(solidity=sol))
        expected = min((1.0 - sol / threshold) ** 0.5, 1.0)
        assert d.confidence == pytest.approx(expected, abs=0.01)

    def test_rule_name_and_priority(self):
        assert LowSolidityRule.name == "low_solidity"
        assert LowSolidityRule.priority == 20

@pytest.mark.regression
class TestHighTextureRule:
    """verify high-texture surface rejection (placed at stage 2 due to M&M imprint)."""

    def test_fires_above_threshold(self, default_cfg):
        rule = HighTextureRule(default_cfg)
        d = rule.apply(make_features(texture_variance=9000.0))
        assert d.label == ColourID.NON_MM

    def test_does_not_fire_at_threshold(self, default_cfg):
        rule = HighTextureRule(default_cfg)
        threshold = default_cfg.thresholds["texture_max"]
        d = rule.apply(make_features(texture_variance=float(threshold)))
        assert d is None

    def test_does_not_fire_below_threshold(self, default_cfg):
        rule = HighTextureRule(default_cfg)
        d = rule.apply(make_features(texture_variance=50.0))
        assert d is None

    def test_confidence_positive_when_firing(self, default_cfg):
        rule = HighTextureRule(default_cfg)
        d = rule.apply(make_features(texture_variance=9000.0))
        assert d.confidence > 0.0

    def test_confidence_capped_at_one(self, default_cfg):
        rule = HighTextureRule(default_cfg)
        d = rule.apply(make_features(texture_variance=1e9))
        assert d.confidence <= 1.0

    def test_rule_name_and_priority(self):
        assert HighTextureRule.name == "high_texture"
        assert HighTextureRule.priority == 20

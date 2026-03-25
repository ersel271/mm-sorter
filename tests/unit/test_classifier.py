# tests/unit/test_classifier.py

import pytest

from config import Config
from config.constants import ColourID
from src.vision import Classifier
from src.vision.rule import Decision, Rule, _RULE_REGISTRY
from tests.helpers.features_helpers import make_features

class _AlwaysRejectRule(Rule):
    name = "mock_reject"
    priority = 10

    def apply(self, f):
        return Decision(ColourID.NON_MM, 0.75, self.name, self.priority)

class _AlwaysColourRule(Rule):
    name = "mock_colour"
    priority = 30

    def apply(self, f):
        return Decision(ColourID.RED, 0.95, self.name, self.priority)

class _NeverFireRule(Rule):
    name = "mock_no_fire"
    priority = 5

    def apply(self, f):
        return None

@pytest.mark.smoke
@pytest.mark.unit
class TestClassifierFallback:
    """verify safe fallback when no rules fire."""

    def test_empty_rule_list_returns_non_mm(self):
        c = Classifier(Config(), rules=[])
        d = c.classify(make_features())
        assert d.label == ColourID.NON_MM

    def test_fallback_rule_name_is_none(self):
        c = Classifier(Config(), rules=[])
        d = c.classify(make_features())
        assert d.rule == "none"

    def test_fallback_priority_is_sentinel(self):
        c = Classifier(Config(), rules=[])
        d = c.classify(make_features())
        assert d.priority == 999

    def test_never_fire_rule_still_returns_fallback(self):
        cfg = Config()
        c = Classifier(cfg, rules=[_NeverFireRule(cfg)])
        d = c.classify(make_features())
        assert d.label == ColourID.NON_MM
        assert d.rule == "none"

@pytest.mark.unit
class TestClassifierSelection:
    """verify priority-then-confidence winner selection."""

    def test_lower_priority_number_wins_over_higher_confidence(self):
        # reject(p=10, c=0.75) beats colour(p=30, c=0.95)
        cfg = Config()
        c = Classifier(cfg, rules=[_AlwaysRejectRule(cfg), _AlwaysColourRule(cfg)])
        d = c.classify(make_features())
        assert d.label == ColourID.NON_MM
        assert d.rule == "mock_reject"

    def test_single_firing_rule_is_returned(self):
        cfg = Config()
        c = Classifier(cfg, rules=[_AlwaysRejectRule(cfg)])
        d = c.classify(make_features())
        assert d.label == ColourID.NON_MM
        assert d.confidence == pytest.approx(0.75)

    def test_no_fire_rule_does_not_influence_result(self):
        cfg = Config()
        c = Classifier(cfg, rules=[_NeverFireRule(cfg), _AlwaysColourRule(cfg)])
        d = c.classify(make_features())
        assert d.rule == "mock_colour"

    def test_same_priority_higher_confidence_wins(self):
        cfg = Config()

        class _LowConf(Rule):
            name = "low_conf"
            priority = 10
            def apply(self, f):
                return Decision(ColourID.NON_MM, 0.20, self.name, self.priority)

        class _HighConf(Rule):
            name = "high_conf"
            priority = 10
            def apply(self, f):
                return Decision(ColourID.NON_MM, 0.80, self.name, self.priority)

        c = Classifier(cfg, rules=[_LowConf(cfg), _HighConf(cfg)])
        d = c.classify(make_features())
        assert d.rule == "high_conf"

@pytest.mark.unit
class TestClassifierInjection:
    """verify injection path isolation from global registry."""

    def test_injected_rules_only_run(self):
        cfg = Config()
        registry_before = list(_RULE_REGISTRY)
        c = Classifier(cfg, rules=[_AlwaysRejectRule(cfg)])
        d = c.classify(make_features())
        assert d.rule == "mock_reject"
        assert list(_RULE_REGISTRY) == registry_before

    def test_registry_unchanged_after_injection(self):
        registry_before = list(_RULE_REGISTRY)
        cfg = Config()
        Classifier(cfg, rules=[_AlwaysColourRule(cfg)])
        assert list(_RULE_REGISTRY) == registry_before

@pytest.mark.unit
class TestClassifierRulesProperty:
    """verify the rules accessor."""

    def test_rules_returns_list(self):
        cfg = Config()
        c = Classifier(cfg, rules=[_AlwaysRejectRule(cfg), _AlwaysColourRule(cfg)])
        assert isinstance(c.rules, list)

    def test_rules_sorted_by_priority(self):
        cfg = Config()
        # pass colour first (p=30) then reject (p=10), should come out sorted
        c = Classifier(cfg, rules=[_AlwaysColourRule(cfg), _AlwaysRejectRule(cfg)])
        priorities = [r.priority for r in c.rules]
        assert priorities == sorted(priorities)

    def test_rules_is_a_copy(self):
        cfg = Config()
        c = Classifier(cfg, rules=[_AlwaysRejectRule(cfg)])
        c.rules.clear()
        assert len(c.rules) == 1

# tests/unit/test_rules_colour.py

import copy

import numpy as np
import pytest

from config import Config
from config.constants import ColourID
from src.vision.rules.colour import ColourClassificationRule
from tests.helpers.config_helpers import write_config
from tests.helpers.vision_helpers import make_features

# isolated two-colour config used throughout; ranges here are test fixtures, not calibrated values
_COLOUR_CFG = {
    "red": {"h": [[0, 10]], "s": [100, 255], "v": [50, 255]},
    "green": {"h": [[35, 85]], "s": [80, 255], "v": [40, 255]},
}

def _cfg(full_data, tmp_path, colours=None):
    data = copy.deepcopy(full_data)
    data["colours"] = colours if colours is not None else _COLOUR_CFG
    return Config(write_config(data, tmp_path))

def _red_hist() -> np.ndarray:
    # all mass in red hue range [0, 10] — conf_red = 1.0
    h = np.zeros(180)
    h[0:11] = 1.0 / 11
    return h

@pytest.mark.smoke
@pytest.mark.regression
class TestColourClassificationRule:
    """verify hue matching, s/v gating, and ambiguity detection."""

    def test_clear_colour_match_returns_label(self, full_data, tmp_path):
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        d = rule.apply(make_features(sat_mean=150.0, val_mean=150.0, hue_hist=_red_hist()))
        assert d.label == ColourID.RED
        assert d.confidence > 0.5

    def test_sv_gate_fail_returns_no_fire(self, full_data, tmp_path):
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        # sat_mean=50 is below red s_min=100 and green s_min=80, both gates fail
        d = rule.apply(make_features(sat_mean=50.0, val_mean=150.0, hue_hist=_red_hist()))
        assert d is None

    def test_low_confidence_returns_no_fire(self, full_data, tmp_path):
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        # gate passes but hue mass is tiny, conf well below colour_confidence_min=0.15
        sparse = np.zeros(180)
        sparse[0:11] = 0.005 / 11
        d = rule.apply(make_features(sat_mean=150.0, val_mean=150.0, hue_hist=sparse))
        assert d is None

    def test_ambiguity_returns_non_mm(self, full_data, tmp_path):
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        # conf_red=0.20, conf_green=0.15 — gap=0.05 < epsilon=0.10, ambiguous
        ambig = np.zeros(180)
        ambig[0:11] = 0.20 / 11
        ambig[35:86] = 0.15 / 51
        d = rule.apply(make_features(sat_mean=150.0, val_mean=150.0, hue_hist=ambig))
        assert d.label == ColourID.NON_MM
        assert d.rule == "ambiguous_colour"

    def test_clear_winner_is_not_ambiguous(self, full_data, tmp_path):
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        # conf_red=0.90, conf_green=0.05, gap=0.85 >> epsilon=0.10
        dominant = np.zeros(180)
        dominant[0:11] = 0.90 / 11
        dominant[35:86] = 0.05 / 51
        d = rule.apply(make_features(sat_mean=150.0, val_mean=150.0, hue_hist=dominant))
        assert d.label == ColourID.RED

    def test_val_gate_fail_returns_no_fire(self, full_data, tmp_path):
        # red v=[50, 255], green v=[40, 255], val_mean=20 fails both
        cfg = _cfg(full_data, tmp_path)
        rule = ColourClassificationRule(cfg)
        d = rule.apply(make_features(sat_mean=150.0, val_mean=20.0, hue_hist=_red_hist()))
        assert d is None

    def test_rule_name_and_priority(self):
        assert ColourClassificationRule.name == "colour_match"
        assert ColourClassificationRule.priority == 30

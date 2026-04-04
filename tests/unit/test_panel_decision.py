# tests/unit/test_panel_decision.py

import numpy as np

from config.constants import ColourID
from src.ui.panels import DecisionPanel
from src.ui.panel import PANEL_W
from src.vision.rule import Priority
from tests.helpers.vision_helpers import make_features, make_decision

class TestDecisionPanelShape:
    """verify DecisionPanel output matches the (panel_h, PANEL_W, 3) contract."""

    def test_shape_no_features_no_decision(self, decision_panel: DecisionPanel) -> None:
        assert decision_panel.render(None, None, 400).shape == (400, PANEL_W, 3)

    def test_shape_with_features_and_decision(self, decision_panel: DecisionPanel) -> None:
        out = decision_panel.render(make_features(), make_decision(), 400)
        assert out.shape == (400, PANEL_W, 3)

    def test_various_heights_produce_correct_shape(self, decision_panel: DecisionPanel) -> None:
        for h in (200, 400, 600, 800):
            assert decision_panel.render(None, None, h).shape == (h, PANEL_W, 3)

class TestDecisionPanelDtype:
    """verify DecisionPanel always returns a uint8 array."""

    def test_dtype_no_features(self, decision_panel: DecisionPanel) -> None:
        assert decision_panel.render(None, None, 300).dtype == np.uint8

    def test_dtype_with_full_inputs(self, decision_panel: DecisionPanel) -> None:
        assert decision_panel.render(make_features(), make_decision(), 300).dtype == np.uint8

class TestDecisionPanelColours:
    """verify DecisionPanel handles all ColourID values without crashing."""

    def test_all_colour_ids_do_not_crash(self, decision_panel: DecisionPanel) -> None:
        for colour_id in ColourID:
            dec = make_decision(label=colour_id)
            assert decision_panel.render(make_features(), dec, 400) is not None

class TestDecisionPanelRejectionLogic:
    """verify rejection footer and stage display paths are exercised."""

    def test_s1_rejection_does_not_crash(self, decision_panel: DecisionPanel) -> None:
        dec = make_decision(label=ColourID.NON_MM, rule="low_saturation", priority=Priority.S1)
        assert decision_panel.render(make_features(), dec, 500) is not None

    def test_s2_rejection_does_not_crash(self, decision_panel: DecisionPanel) -> None:
        dec = make_decision(label=ColourID.NON_MM, rule="low_circularity", priority=Priority.S2)
        assert decision_panel.render(make_features(), dec, 500) is not None

    def test_s3_rejection_does_not_crash(self, decision_panel: DecisionPanel) -> None:
        dec = make_decision(label=ColourID.NON_MM, rule="ambiguous_colour", priority=Priority.S3)
        assert decision_panel.render(make_features(), dec, 500) is not None

    def test_panel_not_entirely_black_with_rejection(self, decision_panel: DecisionPanel) -> None:
        dec = make_decision(label=ColourID.NON_MM, rule="low_saturation", priority=Priority.S1)
        out = decision_panel.render(make_features(), dec, 400)
        assert out.max() > 0

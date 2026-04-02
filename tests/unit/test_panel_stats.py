# tests/unit/test_panel_stats.py

import pytest
import numpy as np

from src.ui.panels import StatsPanel
from src.ui.panel import PANEL_W
from utils.metrics import RunningMetrics

@pytest.mark.unit
class TestStatsPanelShape:
    """verify StatsPanel output matches the (panel_h, PANEL_W, 3) contract."""

    def test_shape_empty_metrics(self) -> None:
        out = StatsPanel(RunningMetrics()).render(None, None, 400)
        assert out.shape == (400, PANEL_W, 3)

    def test_shape_populated_metrics(self, stats_panel: StatsPanel) -> None:
        assert stats_panel.render(None, None, 400).shape == (400, PANEL_W, 3)

    def test_various_heights_produce_correct_shape(self, stats_panel: StatsPanel) -> None:
        for h in (200, 400, 600, 800):
            assert stats_panel.render(None, None, h).shape == (h, PANEL_W, 3)

@pytest.mark.unit
class TestStatsPanelDtype:
    """verify StatsPanel always returns a uint8 array."""

    def test_dtype_empty_metrics(self) -> None:
        assert StatsPanel(RunningMetrics()).render(None, None, 300).dtype == np.uint8

    def test_dtype_populated_metrics(self, stats_panel: StatsPanel) -> None:
        assert stats_panel.render(None, None, 300).dtype == np.uint8

@pytest.mark.unit
class TestStatsPanelSafety:
    """verify StatsPanel does not crash on edge-case inputs."""

    def test_zero_total_does_not_crash(self) -> None:
        assert StatsPanel(RunningMetrics()).render(None, None, 400) is not None

    def test_populated_metrics_does_not_crash(self, stats_panel: StatsPanel) -> None:
        assert stats_panel.render(None, None, 400) is not None

    def test_panel_not_entirely_black_when_populated(self, stats_panel: StatsPanel) -> None:
        out = stats_panel.render(None, None, 400)
        assert out.max() > 0

    def test_features_and_decision_args_are_ignored(self, stats_panel: StatsPanel) -> None:
        # StatsPanel does not use features or decision, passing them should not change shape
        from tests.helpers.features_helpers import make_features, make_decision
        out = stats_panel.render(make_features(), make_decision(), 300)
        assert out.shape == (300, PANEL_W, 3)

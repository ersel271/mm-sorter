# tests/fixtures/panel_fixtures.py

import pytest

from src.ui.panels import FeaturePanel, DecisionPanel, StatsPanel
from src.ui.panels.log_panel import LogPanel
from tests.helpers.events_helpers import make_metrics

@pytest.fixture
def feature_panel() -> FeaturePanel:
    return FeaturePanel()

@pytest.fixture
def decision_panel(default_cfg) -> DecisionPanel:
    return DecisionPanel(default_cfg)

@pytest.fixture
def stats_panel() -> StatsPanel:
    return StatsPanel(make_metrics())

@pytest.fixture
def log_panel() -> LogPanel:
    return LogPanel()

# tests/fixtures/overlay_fixtures.py

import pytest

from src.ui import Overlay
from utils.metrics import RunningMetrics
from tests.helpers.config_helpers import make_config

@pytest.fixture
def overlay(default_cfg) -> Overlay:
    return Overlay(default_cfg, RunningMetrics())

@pytest.fixture
def overlay_disabled() -> Overlay:
    return Overlay(make_config(system={"display_enabled": False}), RunningMetrics())

@pytest.fixture
def overlay_half_scale() -> Overlay:
    return Overlay(make_config(system={"display_scale": 0.5}), RunningMetrics())

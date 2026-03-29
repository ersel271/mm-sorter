# tests/fixtures/overlay_fixtures.py

import pytest
from pathlib import Path

from config import Config
from src.ui import Overlay
from tests.helpers.config_helpers import write_config

@pytest.fixture
def overlay() -> Overlay:
    return Overlay(Config())

@pytest.fixture
def overlay_disabled(default_cfg: Config, tmp_path: Path) -> Overlay:
    data = default_cfg.as_dict()
    data["system"]["display_enabled"] = False
    return Overlay(Config(write_config(data, tmp_path)))

@pytest.fixture
def overlay_half_scale(default_cfg: Config, tmp_path: Path) -> Overlay:
    data = default_cfg.as_dict()
    data["system"]["display_scale"] = 0.5
    return Overlay(Config(write_config(data, tmp_path)))
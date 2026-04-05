# tests/fixtures/config_fixtures.py

import yaml
import pytest

from config import Config
from tests.helpers.config_helpers import FULL_DATA, MINIMAL_DATA

@pytest.fixture
def full_data() -> dict:
    return FULL_DATA

@pytest.fixture
def minimal_data() -> dict:
    return MINIMAL_DATA

@pytest.fixture
def default_cfg(full_data, tmp_path) -> Config:
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(full_data, f)
    return Config(cfg_path)

@pytest.fixture
def tmp_cfg(full_data, tmp_path) -> Config:
    import copy
    data = copy.deepcopy(full_data)
    data["system"]["log_dir"] = str(tmp_path / "logs")
    data["system"]["event_dir"] = str(tmp_path / "events")

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(data, f)

    return Config(cfg_path)

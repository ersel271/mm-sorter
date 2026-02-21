# tests/conftest.py

import copy
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import serial
import yaml

from config import Config
from utils import log as log_module

@pytest.fixture
def default_cfg() -> Config:
    return Config()

@pytest.fixture
def valid_data() -> dict:
    return {
        "camera": {
            "device": 2,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "format": "MJPG",
            "autofocus": False,
            "focus": 200,
            "auto_exposure": 3,
            "exposure": 157,
            "auto_wb": True,
            "wb_temperature": 4600,
            "power_line_frequency": 1,
        },
        "preprocess": {
            "roi_enabled": False,
            "roi_fraction": 0.9,
            "blur_kernel": 5,
            "sat_threshold": 40,
            "morph_kernel": 5,
            "morph_erode_iter": 1,
            "morph_dilate_iter": 2,
            "min_area": 500,
        },
        "features": {
            "hue_bins": 180,
            "hue_smooth_sigma": 3,
            "highlight_value": 240,
        },
        "thresholds": {
            "sat_min": 60,
            "highlight_max": 0.20,
            "hue_width_min": 8,
            "texture_max": 500.0,
            "circularity_min": 0.75,
            "aspect_ratio_max": 1.35,
            "solidity_min": 0.90,
            "colour_confidence_min": 0.15,
        },
        "colours": {
            "red": {
                "h": [[0, 10], [170, 180]],
                "s": [100, 255],
                "v": [50, 255],
            },
        },
        "uart": {
            "port": "/dev/ttyUSB0",
            "baud": 115200,
            "timeout": 0.1,
        },
        "system": {
            "log_dir": "data/logs",
            "sample_dir": "data/samples",
            "log_queue_size": 256,
            "display_enabled": True,
            "display_scale": 1.0,
        },
    }

@pytest.fixture
def mock_port() -> MagicMock:
    port = MagicMock(spec=serial.Serial)
    port.write = MagicMock(return_value=None)
    port.readline = MagicMock(return_value=b"")
    port.close = MagicMock(return_value=None)
    port.timeout = 0.1
    return port

@pytest.fixture
def tmp_cfg(tmp_path) -> Config:
    cfg = Config()
    data = cfg.as_dict()
    data["system"]["log_dir"] = str(tmp_path / "logs")
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(data, f)
    return Config(cfg_path)

def write_config(data: dict, directory: Path) -> Path:
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path

def sample_fields(**overrides) -> dict:
    base = {"id": 42, "class": 3, "conf": 0.91, "x": 960, "y": 540}
    base.update(overrides)
    return base

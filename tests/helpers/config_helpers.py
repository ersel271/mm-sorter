# tests/helpers/config_helpers.py

import copy
import tempfile
from pathlib import Path

import yaml

from config import Config

FULL_DATA: dict = {
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
        "hue_peak_ratio": 0.15,
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
        "colour_ambiguity_epsilon": 0.10,
        "decision_min": 0.5,
    },
    "colours": {
        "red":    {"h": [[0, 10], [170, 180]], "s": [100, 255], "v": [50, 255]},
        "green":  {"h": [[35, 85]],            "s": [80, 255],  "v": [40, 255]},
        "blue":   {"h": [[100, 130]],          "s": [100, 255], "v": [40, 255]},
        "yellow": {"h": [[20, 35]],            "s": [100, 255], "v": [100, 255]},
        "orange": {"h": [[10, 20]],            "s": [120, 255], "v": [100, 255]},
        "brown":  {"h": [[10, 25]],            "s": [40, 120],  "v": [30, 150]},
    },
    "uart": {
        "port": "/dev/ttyUSB0",
        "baud": 115200,
        "timeout": 0.1,
    },
    "system": {
        "log_dir": "data/logs",
        "event_dir": "data/events",
        "plot_dir": "data/plots",
        "log_queue_size": 256,
        "display_enabled": True,
        "display_scale": 1.0,
    },
}

MINIMAL_DATA: dict = {
    "camera": {
        "device": 2,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "format": "MJPG",
        "autofocus": False,
        "focus": 200,
    },
    "preprocess": {
        "blur_kernel": 5,
        "sat_threshold": 40,
        "morph_kernel": 5,
        "min_area": 500,
    },
    "features": {
        "hue_bins": 180,
        "hue_smooth_sigma": 3,
        "highlight_value": 240,
        "hue_peak_ratio": 0.15,
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
        "colour_ambiguity_epsilon": 0.10,
        "decision_min": 0.5,
    },
    "colours": {
        "red": {"h": [[0, 10], [170, 180]], "s": [100, 255], "v": [50, 255]},
    },
    "uart": {
        "port": "/dev/ttyUSB0",
        "baud": 115200,
    },
    "system": {
        "log_dir": "data/logs",
        "event_dir": "data/events",
        "plot_dir": "data/plots",
        "log_queue_size": 256,
    },
}

def make_config(**section_overrides):
    """Build a Config from FULL_DATA with optional section-level overrides.

    Each keyword argument is a section name mapped to a dict of keys to
    merge into that section.  Only the provided keys change; the rest of
    FULL_DATA is preserved.

    Usage:
        make_config()
        make_config(camera={"autofocus": True})
        make_config(camera={"auto_exposure": 1, "exposure": 200})
    """
    data = copy.deepcopy(FULL_DATA)
    for section, values in section_overrides.items():
        data[section] = {**data[section], **values}

    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    return Config(path)

def write_config(data: dict, directory: Path) -> Path:
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path

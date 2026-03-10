# tests/helpers/config_helpers.py

from pathlib import Path
import yaml

def write_config(data: dict, directory: Path) -> Path:
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path

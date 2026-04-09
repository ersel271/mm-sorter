# config/config.py
"""
YAML configuration loader for the M&M sorter system.

Loads config.yaml and provides dict-based access to all parameters.
Immutable after loading.

Usage:
    cfg = Config()
    cfg = Config("path/to/config.yaml")

    device = cfg.camera["device"]          # required field
    timeout = cfg.uart.get("timeout", 0.1) # optional field with default
"""

import logging
from pathlib import Path

import yaml

from config.validate import validate, ConfigError  # ConfigError re-exported for __init__.py

log = logging.getLogger(__name__)


class Config:
    """
    immutable configuration accessor loaded from a YAML file.
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).parent / "config.yaml"
        self.path = Path(path)
        self._data = self._load(self.path)
        validate(self._data)

        self.camera: dict = self._data["camera"]
        self.preprocess: dict = self._data["preprocess"]
        self.features: dict = self._data["features"]
        self.thresholds: dict = self._data["thresholds"]
        self.colours: dict = self._data["colours"]
        self.uart: dict = self._data["uart"]
        self.system: dict = self._data["system"]

        log.info("config loaded from %s (%d colours)", self.path, len(self.colours))

    def colour_names(self) -> list[str]:
        """
        return list of configured colour names.
        """
        return list(self.colours.keys())

    def as_dict(self) -> dict:
        """
        return the full configuration as a deep-copied plain dict.
        """
        import copy
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(path={self.path})"

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")

        log.debug("loading config from %s", path)

        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigError(f"YAML parse error in {path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

        return data


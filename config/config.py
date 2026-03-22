# config/config.py
"""
YAML configuration loader and validator for the M&M sorter system.

Loads config.yaml, validates required sections and fields, and provides
dict-based access to all parameters. Immutable after loading.

Usage:
    cfg = Config()
    cfg = Config("path/to/config.yaml")
    
    device = cfg.camera["device"]          # required field
    timeout = cfg.uart.get("timeout", 0.1) # optional field with default
"""

import logging
from pathlib import Path

import yaml

from config.constants import COLOUR_IDS

log = logging.getLogger(__name__)

# sections that every valid configuration file must include
_REQUIRED_SECTIONS = (
    "camera", "preprocess", "features", "thresholds", "colours", "uart", "system",
)

# fields within sections that every valid configuration must include.
# use value None if type checking is unnecessary for a field
_REQUIRED_FIELDS: dict[str, dict[str, type | None]] = {
    "camera": {
        "device": int,
        "width": int,
        "height": int,
        "fps": int,
        "format": str,
        "autofocus": bool,
        "focus": int,
    },
    "preprocess": {
        "blur_kernel": int,
        "sat_threshold": int,
        "morph_kernel": int,
        "min_area": int,
    },
    "features": {
        "hue_bins": int,
        "highlight_value": int,
        "hue_smooth_sigma": (int, float),
        "hue_peak_ratio": (int, float),
    },
    "thresholds": {
        "sat_min": (int, float),
        "highlight_max": (int, float),
        "hue_width_min": (int, float),
        "texture_max": (int, float),
        "circularity_min": (int, float),
        "aspect_ratio_max": (int, float),
        "solidity_min": (int, float),
        "colour_confidence_min": (int, float),
    },
    "colours": {},  # validated separately due to dynamic colour names
    "uart": {
        "port": str,
        "baud": int,
    },
    "system": {
        "log_dir": str,
        "event_dir": str,
        "sample_dir": str,
        "log_queue_size": int,
    },
}

# required fields for each colour entry in the colours section.
# h is a list of [min, max] hue pairs (supports wraparound colours like red).
# s and v are [min, max] pairs for saturation and value channels
_COLOUR_FIELDS = ("h", "s", "v")

class ConfigError(Exception):
    """
    raised when configuration is invalid or missing required fields.
    """

class Config:
    """
    immutable configuration accessor loaded from a YAML file.
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).parent / "config.yaml"
        self.path = Path(path)
        self._data = self._load(self.path)
        self._validate(self._data)

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

    @staticmethod
    def _validate(data: dict) -> None:
        # required sections
        for section in _REQUIRED_SECTIONS:
            if section not in data:
                raise ConfigError(f"missing required section: '{section}'")
            if not isinstance(data[section], dict):
                raise ConfigError(
                    f"section '{section}' must be a mapping, got {type(data[section]).__name__}"
                )

        # required fields and type checks per section
        for section, fields in _REQUIRED_FIELDS.items():
            for field, expected_type in fields.items():
                if field not in data[section]:
                    raise ConfigError(f"missing required field: '{section}.{field}'")
                if expected_type is not None:
                    value = data[section][field]
                    if not isinstance(value, expected_type):
                        raise ConfigError(
                            f"field '{section}.{field}' must be {expected_type}, "
                            f"got {type(value).__name__} ({value!r})"
                        )

        # colour entries
        colours = data["colours"]
        if len(colours) == 0:
            raise ConfigError("at least one colour must be defined in 'colours' section")

        _valid = sorted(n for n in COLOUR_IDS if n != "non-m&m")
        for name, cfg in colours.items():
            if name.lower() not in COLOUR_IDS or name.lower() == "non-m&m":
                raise ConfigError(
                    f"colour '{name}' is not a recognised colour — "
                    f"valid names: {', '.join(_valid)}"
                )
            if not isinstance(cfg, dict):
                raise ConfigError(f"colour '{name}' must be a mapping")

            for field in _COLOUR_FIELDS:
                if field not in cfg:
                    raise ConfigError(f"colour '{name}' missing required field: '{field}'")

            # h: list of [min, max] hue pairs
            h = cfg["h"]
            if not isinstance(h, list) or len(h) == 0:
                raise ConfigError(f"colour '{name}.h' must be a non-empty list")

            for i, rng in enumerate(h):
                _validate_pair(rng, f"colour '{name}'.h[{i}]")

            # s and v: each a [min, max] pair
            _validate_pair(cfg["s"], f"colour '{name}'.s")
            _validate_pair(cfg["v"], f"colour '{name}'.v")

        # camera numeric ranges.
        # resolution and fps are only checked for basic sanity here;
        # whether the camera actually supports them is verified at open() time
        cam = data["camera"]
        if cam["width"] < 1 or cam["height"] < 1:
            raise ConfigError("camera width and height must be positive")
        if cam["fps"] < 1:
            raise ConfigError("camera fps must be positive")
        if not (1 <= cam["focus"] <= 1023):
            raise ConfigError("camera focus must be in range 1--1023")

        pre = data["preprocess"]
        if pre["blur_kernel"] % 2 == 0:
            raise ConfigError("preprocess.blur_kernel must be odd")
        if pre["morph_kernel"] < 1:
            raise ConfigError("preprocess.morph_kernel must be positive")

        thresh = data["thresholds"]
        if not (0.0 <= thresh["highlight_max"] <= 1.0):
            raise ConfigError("thresholds.highlight_max must be in range 0.0--1.0")
        if not (0.0 <= thresh["circularity_min"] <= 1.0):
            raise ConfigError("thresholds.circularity_min must be in range 0.0--1.0")
        if not (0.0 <= thresh["solidity_min"] <= 1.0):
            raise ConfigError("thresholds.solidity_min must be in range 0.0--1.0")
        if not (0.0 <= thresh["colour_confidence_min"] <= 1.0):
            raise ConfigError("thresholds.colour_confidence_min must be in range 0.0--1.0")

        uart = data["uart"]
        if uart["baud"] < 9600:
            raise ConfigError("uart.baud must be at least 9600")

def _validate_pair(pair: object, label: str) -> None:
    """
    validate a [min, max] numeric pair.
    """
    if not isinstance(pair, list) or len(pair) != 2:
        raise ConfigError(f"{label} must be a [min, max] pair")
    lo, hi = pair
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        raise ConfigError(f"{label} values must be numbers")
    if lo > hi:
        raise ConfigError(f"{label}: min ({lo}) > max ({hi})")

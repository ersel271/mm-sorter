# config/validate.py
"""
Validation logic for the M&M sorter configuration.

Validates required sections, required fields, optional field types, colour
entries, and numeric range constraints. Raises ConfigError on any violation.

Usage:
    validate(data) 
"""

from config.constants import COLOUR_IDS

# sections that every valid configuration file must include
_REQUIRED_SECTIONS = (
    "camera", "preprocess", "features", "thresholds", "colours", "uart", "system",
)

# fields within sections that every valid configuration must include.
# use value None if type checking is unnecessary for a field
_REQUIRED_FIELDS: dict[str, dict[str, type | tuple[type, ...] | None]] = {
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
        "colour_ambiguity_epsilon": (int, float),
        "decision_min": (int, float),
    },
    "colours": {},  # validated separately due to dynamic colour names
    "uart": {
        "port": str,
        "baud": int,
    },
    "system": {
        "log_dir": str,
        "event_dir": str,
        "plot_dir": str,
        "log_queue_size": int,
    },
}

# optional fields and their expected types
_OPTIONAL_FIELDS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "camera": {
        "auto_exposure": int,
        "exposure": (int, float),
        "auto_wb": bool,
        "wb_temperature": (int, float),
        "power_line_frequency": int,
        "saturation": int,
        "gamma": int,
    },
    "preprocess": {
        "roi_enabled": bool,
        "roi_fraction": (int, float),
        "morph_erode_iter": int,
        "morph_dilate_iter": int,
        "sat_min_dark": int,
        "sat_max_dark": int,
        "val_min_dark": int,
        "val_max_dark": int,
        "sec_morph_erode_iter": int,
        "sec_min_area": int,
        "sec_morph_dilate_iter": int,
    },
    "uart": {
        "timeout": (int, float),
    },
    "system": {
        "found_frames_min": int,
        "display_enabled": bool,
        "display_scale": (int, float),
    },
}

# required fields for each colour entry in the colours section.
# h is a list of [min, max] hue pairs (supports wraparound colours like red).
# s and v are [min, max] pairs for saturation and value channels
_COLOUR_FIELDS = ("h", "s", "v")

class ConfigError(Exception):
    """raised when configuration is invalid or missing required fields."""

def validate(data: dict) -> None:
    _validate_sections(data)
    _validate_fields(data)
    _validate_optional_fields(data)
    _validate_colours(data["colours"])
    _validate_ranges(data)

def _validate_sections(data: dict) -> None:
    for section in _REQUIRED_SECTIONS:
        if section not in data:
            raise ConfigError(f"missing required section: '{section}'")
        if not isinstance(data[section], dict):
            raise ConfigError(
                f"section '{section}' must be a mapping, got {type(data[section]).__name__}"
            )

def _validate_fields(data: dict) -> None:
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

def _validate_optional_fields(data: dict) -> None:
    for section, fields in _OPTIONAL_FIELDS.items():
        sec_data = data.get(section, {})
        for field, expected_type in fields.items():
            if field in sec_data:
                value = sec_data[field]
                if not isinstance(value, expected_type):
                    raise ConfigError(
                        f"field '{section}.{field}' must be {expected_type}, "
                        f"got {type(value).__name__} ({value!r})"
                    )

def _validate_colours(colours: dict) -> None:
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

def _validate_pair(pair: object, label: str) -> None:
    """validate a [min, max] numeric pair."""
    if not isinstance(pair, list) or len(pair) != 2:
        raise ConfigError(f"{label} must be a [min, max] pair")
    lo, hi = pair
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        raise ConfigError(f"{label} values must be numbers")
    if lo > hi:
        raise ConfigError(f"{label}: min ({lo}) > max ({hi})")

def _validate_ranges(data: dict) -> None:
    _validate_ranges_camera(data["camera"])
    _validate_ranges_preprocess(data["preprocess"])
    _validate_ranges_features(data["features"])
    _validate_ranges_thresholds(data["thresholds"])
    _validate_ranges_uart(data["uart"])
    _validate_ranges_system(data["system"])

def _validate_ranges_camera(cam: dict) -> None:
    if cam["device"] < 0:
        raise ConfigError("camera.device must be >= 0")
    if cam["format"] not in {"MJPG", "YUYV"}:
        raise ConfigError(f"camera.format must be MJPG or YUYV, got {cam['format']!r}")
    if not (1 <= cam["focus"] <= 1023):
        raise ConfigError("camera focus must be in range 1--1023")
    if cam["fps"] < 1:
        raise ConfigError("camera fps must be positive")
    if cam["width"] < 1 or cam["height"] < 1:
        raise ConfigError("camera width and height must be positive")
    if "auto_exposure" in cam and cam["auto_exposure"] not in {1, 3}:
        raise ConfigError("camera.auto_exposure must be 1 (manual) or 3 (aperture priority)")
    if "exposure" in cam and cam["exposure"] <= 0:
        raise ConfigError("camera.exposure must be positive")
    if "wb_temperature" in cam and not (1000 <= cam["wb_temperature"] <= 10000):
        raise ConfigError("camera.wb_temperature must be in range 1000--10000")
    if "power_line_frequency" in cam and cam["power_line_frequency"] not in {0, 1, 2}:
        raise ConfigError("camera.power_line_frequency must be 0, 1, or 2")
    if "saturation" in cam and not (0 <= cam["saturation"] <= 128):
        raise ConfigError("camera.saturation must be in range 0--128")
    if "gamma" in cam and not (0 <= cam["gamma"] <= 500):
        raise ConfigError("camera.gamma must be in range 0--500")

def _validate_ranges_preprocess(pre: dict) -> None:  # noqa: CCR001
    if pre["blur_kernel"] % 2 == 0:
        raise ConfigError("preprocess.blur_kernel must be odd")
    if pre["morph_kernel"] < 1:
        raise ConfigError("preprocess.morph_kernel must be positive")
    if "roi_fraction" in pre and not (0.0 < pre["roi_fraction"] <= 1.0):
        raise ConfigError("preprocess.roi_fraction must be in range (0.0, 1.0]")
    if "morph_erode_iter" in pre and pre["morph_erode_iter"] < 0:
        raise ConfigError("preprocess.morph_erode_iter must be >= 0")
    if "morph_dilate_iter" in pre and pre["morph_dilate_iter"] < 0:
        raise ConfigError("preprocess.morph_dilate_iter must be >= 0")
    if "sat_min_dark" in pre and not (0 <= pre["sat_min_dark"] <= 255):
        raise ConfigError("preprocess.sat_min_dark must be in range 0--255")
    if "sat_max_dark" in pre and not (0 <= pre["sat_max_dark"] <= 255):
        raise ConfigError("preprocess.sat_max_dark must be in range 0--255")
    if "val_min_dark" in pre and not (0 <= pre["val_min_dark"] <= 255):
        raise ConfigError("preprocess.val_min_dark must be in range 0--255")
    if "val_max_dark" in pre and not (0 <= pre["val_max_dark"] <= 255):
        raise ConfigError("preprocess.val_max_dark must be in range 0--255")
    if "sec_morph_erode_iter" in pre and pre["sec_morph_erode_iter"] < 0:
        raise ConfigError("preprocess.sec_morph_erode_iter must be >= 0")
    if "sec_min_area" in pre and pre["sec_min_area"] < 0:
        raise ConfigError("preprocess.sec_min_area must be >= 0")
    if "sec_morph_dilate_iter" in pre and pre["sec_morph_dilate_iter"] < 0:
        raise ConfigError("preprocess.sec_morph_dilate_iter must be >= 0")

def _validate_ranges_features(feat: dict) -> None:
    if feat["hue_bins"] < 1:
        raise ConfigError("features.hue_bins must be positive")
    if not (0 <= feat["highlight_value"] <= 255):
        raise ConfigError("features.highlight_value must be in range 0--255")
    if feat["hue_smooth_sigma"] < 0:
        raise ConfigError("features.hue_smooth_sigma must be >= 0")
    if not (0.0 <= feat["hue_peak_ratio"] <= 1.0):
        raise ConfigError("features.hue_peak_ratio must be in range 0.0--1.0")

def _validate_ranges_thresholds(thresh: dict) -> None:
    if thresh["sat_min"] <= 0:
        raise ConfigError("thresholds.sat_min must be positive")
    if not (0.0 <= thresh["highlight_max"] <= 1.0):
        raise ConfigError("thresholds.highlight_max must be in range 0.0--1.0")
    if thresh["hue_width_min"] <= 0:
        raise ConfigError("thresholds.hue_width_min must be positive")
    if thresh["texture_max"] <= 0:
        raise ConfigError("thresholds.texture_max must be positive")
    if not (0.0 <= thresh["circularity_min"] <= 1.0):
        raise ConfigError("thresholds.circularity_min must be in range 0.0--1.0")
    if thresh["aspect_ratio_max"] < 1.0:
        raise ConfigError("thresholds.aspect_ratio_max must be >= 1.0")
    if not (0.0 <= thresh["solidity_min"] <= 1.0):
        raise ConfigError("thresholds.solidity_min must be in range 0.0--1.0")
    if not (0.0 <= thresh["colour_confidence_min"] <= 1.0):
        raise ConfigError("thresholds.colour_confidence_min must be in range 0.0--1.0")
    if not (0.0 <= thresh["colour_ambiguity_epsilon"] <= 1.0):
        raise ConfigError("thresholds.colour_ambiguity_epsilon must be in range 0.0--1.0")
    if not (0.0 <= thresh["decision_min"] <= 1.0):
        raise ConfigError("thresholds.decision_min must be in range 0.0--1.0")

def _validate_ranges_uart(uart: dict) -> None:
    if uart["baud"] < 9600:
        raise ConfigError("uart.baud must be at least 9600")
    if "timeout" in uart and uart["timeout"] < 0:
        raise ConfigError("uart.timeout must be >= 0")

def _validate_ranges_system(sys_: dict) -> None:
    if sys_["log_queue_size"] < 1:
        raise ConfigError("system.log_queue_size must be positive")
    if "found_frames_min" in sys_ and sys_["found_frames_min"] < 1:
        raise ConfigError("system.found_frames_min must be >= 1")
    if "display_scale" in sys_ and sys_["display_scale"] <= 0:
        raise ConfigError("system.display_scale must be positive")

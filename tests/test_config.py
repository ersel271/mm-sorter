# tests/test_config.py

import copy

import pytest

from config import Config, ColourID, COLOUR_NAMES
from config.config import ConfigError
from tests.helpers.config_helpers import write_config

@pytest.mark.unit
class TestConstants:
    """verify ColourID enum values and COLOUR_NAMES mapping consistency."""

    def test_colour_id_values(self):
        assert ColourID.NON_MM == 0
        assert ColourID.RED == 1
        assert ColourID.GREEN == 2
        assert ColourID.BLUE == 3
        assert ColourID.YELLOW == 4
        assert ColourID.ORANGE == 5
        assert ColourID.BROWN == 6

    def test_colour_id_count(self):
        assert len(ColourID) == 7

    def test_colour_names_keys_match_enum(self):
        for cid in ColourID:
            assert cid in COLOUR_NAMES

    def test_colour_names_values_are_strings(self):
        for name in COLOUR_NAMES.values():
            assert isinstance(name, str)
            assert len(name) > 0

@pytest.mark.smoke
@pytest.mark.unit
class TestConfigLoading:
    """verify config loads from default and custom paths, rejects bad input."""

    def test_default_config_loads(self, default_cfg):
        assert default_cfg is not None
        assert default_cfg.path.name == "config.yaml"

    def test_custom_path_loads(self, valid_data, tmp_path):
        path = write_config(valid_data, tmp_path)
        cfg = Config(path)
        assert cfg.camera["device"] == 2

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            Config("/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        bad = tmp_path / "config.yaml"
        bad.write_text("{{{{invalid yaml: [")
        with pytest.raises(ConfigError, match="YAML parse error"):
            Config(bad)

    def test_non_dict_root_raises(self, tmp_path):
        bad = tmp_path / "config.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigError, match="root must be a mapping"):
            Config(bad)

    def test_repr(self, default_cfg):
        r = repr(default_cfg)
        assert "Config(" in r
        assert "config.yaml" in r

@pytest.mark.unit
class TestConfigAccess:
    """verify dict-based access to each config section."""

    def test_camera_section(self, default_cfg):
        cam = default_cfg.camera
        assert isinstance(cam, dict)
        assert cam["device"] == 2
        assert cam["width"] == 1920
        assert cam["height"] == 1080

    def test_thresholds_section(self, default_cfg):
        t = default_cfg.thresholds
        assert t["sat_min"] == 60
        assert t["highlight_max"] == 0.20
        assert t["circularity_min"] == 0.75

    def test_colours_section(self, default_cfg):
        c = default_cfg.colours
        assert "red" in c
        assert "blue" in c
        assert "brown" in c
        assert len(c) == 6

    def test_colour_sv_format(self, default_cfg):
        for name, colour in default_cfg.colours.items():
            assert isinstance(colour["s"], list) and len(colour["s"]) == 2
            assert isinstance(colour["v"], list) and len(colour["v"]) == 2
            assert colour["s"][0] <= colour["s"][1]
            assert colour["v"][0] <= colour["v"][1]

    def test_uart_section(self, default_cfg):
        u = default_cfg.uart
        assert u["baud"] == 115200
        assert u["port"] == "/dev/ttyUSB0"

    def test_system_section(self, default_cfg):
        s = default_cfg.system
        assert s["log_dir"] == "data/logs"

@pytest.mark.unit
class TestConfigColours:
    """verify colour_names() returns all configured colours."""

    def test_colour_names_returns_all(self, default_cfg):
        names = default_cfg.colour_names()
        assert set(names) == {"red", "green", "blue", "yellow", "orange", "brown"}

@pytest.mark.unit
class TestConfigAsDict:
    """verify as_dict() returns a complete deep copy."""

    def test_returns_dict(self, default_cfg):
        d = default_cfg.as_dict()
        assert isinstance(d, dict)
        assert "camera" in d

    def test_is_deep_copy(self, default_cfg):
        d = default_cfg.as_dict()
        d["camera"]["device"] = 999
        assert default_cfg.camera["device"] == 2

@pytest.mark.unit
class TestValidationMissingSections:
    """verify ConfigError is raised when any required section is missing."""

    @pytest.mark.parametrize("section", [
        "camera", "preprocess", "features", "thresholds", "colours", "uart", "system",
    ])
    def test_missing_section_raises(self, valid_data, tmp_path, section):
        data = copy.deepcopy(valid_data)
        del data[section]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=f"missing required section.*{section}"):
            Config(path)

@pytest.mark.unit
class TestValidationMissingFields:
    """verify ConfigError is raised when required fields are missing."""

    @pytest.mark.parametrize("section,field", [
        ("camera", "device"),
        ("camera", "width"),
        ("camera", "focus"),
        ("preprocess", "blur_kernel"),
        ("preprocess", "min_area"),
        ("features", "hue_bins"),
        ("thresholds", "sat_min"),
        ("thresholds", "circularity_min"),
        ("uart", "port"),
        ("uart", "baud"),
        ("system", "log_dir"),
    ])
    def test_missing_field_raises(self, valid_data, tmp_path, section, field):
        data = copy.deepcopy(valid_data)
        del data[section][field]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=f"missing required field.*{section}.{field}"):
            Config(path)

@pytest.mark.unit
class TestValidationTypes:
    """verify ConfigError is raised on type mismatches."""

    def test_camera_device_must_be_int(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["camera"]["device"] = "two"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="camera.device"):
            Config(path)

    def test_camera_autofocus_must_be_bool(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["camera"]["autofocus"] = "yes"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="camera.autofocus"):
            Config(path)

    def test_uart_port_must_be_str(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["uart"]["port"] = 12345
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="uart.port"):
            Config(path)


@pytest.mark.unit
class TestValidationRanges:
    """verify ConfigError is raised when numeric values are out of range."""

    def test_focus_below_range(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["camera"]["focus"] = 0
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="focus"):
            Config(path)

    def test_focus_above_range(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["camera"]["focus"] = 1024
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="focus"):
            Config(path)

    def test_blur_kernel_even_raises(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["preprocess"]["blur_kernel"] = 4
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="blur_kernel must be odd"):
            Config(path)

    def test_highlight_max_above_1(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["thresholds"]["highlight_max"] = 1.5
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="highlight_max"):
            Config(path)

    def test_circularity_min_negative(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["thresholds"]["circularity_min"] = -0.1
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="circularity_min"):
            Config(path)

    def test_baud_too_low(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["uart"]["baud"] = 300
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="baud"):
            Config(path)


@pytest.mark.unit
class TestValidationColours:
    """verify ConfigError is raised on malformed colour entries."""

    def test_empty_colours_raises(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"] = {}
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="at least one colour"):
            Config(path)

    def test_colour_missing_h(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        del data["colours"]["red"]["h"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="red.*h"):
            Config(path)

    def test_colour_missing_s(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        del data["colours"]["red"]["s"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="red.*s"):
            Config(path)

    def test_colour_missing_v(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        del data["colours"]["red"]["v"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="red.*v"):
            Config(path)

    def test_h_empty_list(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["h"] = []
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="h.*non-empty"):
            Config(path)

    def test_h_bad_pair(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["h"] = [[10, 20, 30]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="min, max.*pair"):
            Config(path)

    def test_h_min_greater_than_max(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["h"] = [[50, 10]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="min.*>.*max"):
            Config(path)

    def test_h_non_numeric(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["h"] = [["a", "b"]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="must be numbers"):
            Config(path)

    def test_s_bad_pair(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["s"] = [100]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="min, max.*pair"):
            Config(path)

    def test_v_min_greater_than_max(self, valid_data, tmp_path):
        data = copy.deepcopy(valid_data)
        data["colours"]["red"]["v"] = [255, 50]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="min.*>.*max"):
            Config(path)


@pytest.mark.integration
class TestConfigIntegration:
    """verify the default config.yaml is internally consistent."""

    def test_all_sections_present(self, default_cfg):
        assert default_cfg.camera is not None
        assert default_cfg.preprocess is not None
        assert default_cfg.features is not None
        assert default_cfg.thresholds is not None
        assert default_cfg.colours is not None
        assert default_cfg.uart is not None
        assert default_cfg.system is not None

    def test_default_has_six_colours(self, default_cfg):
        assert len(default_cfg.colours) == 6

    def test_all_colours_have_required_fields(self, default_cfg):
        for name, colour in default_cfg.colours.items():
            assert "h" in colour, f"{name} missing h"
            assert "s" in colour, f"{name} missing s"
            assert "v" in colour, f"{name} missing v"

    def test_threshold_values_are_sane(self, default_cfg):
        t = default_cfg.thresholds
        assert 0 < t["sat_min"] < 255
        assert 0 < t["highlight_max"] < 1.0
        assert 0 < t["circularity_min"] < 1.0
        assert 0 < t["solidity_min"] < 1.0
        assert t["aspect_ratio_max"] > 1.0

    def test_camera_resolution_standard(self, default_cfg):
        cam = default_cfg.camera
        assert cam["width"] * cam["height"] > 0
        ratio = cam["width"] / cam["height"]
        assert abs(ratio - 16 / 9) < 0.01

    def test_dict_access_works(self, default_cfg):
        assert default_cfg.camera["device"] == 2
        assert default_cfg.uart["baud"] == 115200
        assert default_cfg.thresholds["sat_min"] == 60
        assert default_cfg.colours["red"]["s"] == [100, 255]

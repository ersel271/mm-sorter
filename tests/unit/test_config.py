# tests/unit/test_config.py

import pytest

from config import Config, ColourID, COLOUR_NAMES
from config.validate import ConfigError
from tests.helpers.config_helpers import write_config

@pytest.mark.smoke
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
class TestConfigLoading:
    """verify config loads from default and custom paths, rejects bad input."""

    def test_default_config_loads(self, default_cfg):
        assert default_cfg is not None
        assert default_cfg.path.name == "config.yaml"

    def test_custom_path_loads(self, full_data, tmp_path):
        path = write_config(full_data, tmp_path)
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

@pytest.mark.smoke
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
        for _name, colour in default_cfg.colours.items():
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

@pytest.mark.smoke
class TestConfigColours:
    """verify colour_names() returns all configured colours."""

    def test_colour_names_returns_all(self, default_cfg):
        names = default_cfg.colour_names()
        assert set(names) == {"red", "green", "blue", "yellow", "orange", "brown"}

@pytest.mark.smoke
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


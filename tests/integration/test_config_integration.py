# tests/integration/test_config_integration.py

import pytest

@pytest.mark.smoke
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

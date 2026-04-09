# tests/unit/test_config_validation.py

import copy

import pytest

from config import Config
from config.validate import ConfigError
from tests.helpers.config_helpers import write_config, make_config

@pytest.mark.regression
class TestValidationMissingSections:
    """verify ConfigError is raised when any required section is missing."""

    @pytest.mark.parametrize("section", [
        "camera", "preprocess", "features", "thresholds", "colours", "uart", "system",
    ])
    def test_missing_section_raises(self, full_data, tmp_path, section):
        data = copy.deepcopy(full_data)
        del data[section]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=f"missing required section.*{section}"):
            Config(path)

    def test_section_not_a_dict_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"] = "not_a_dict"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="must be a mapping"):
            Config(path)

@pytest.mark.regression
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
    def test_missing_field_raises(self, full_data, tmp_path, section, field):
        data = copy.deepcopy(full_data)
        del data[section][field]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=f"missing required field.*{section}.{field}"):
            Config(path)

@pytest.mark.regression
class TestValidationTypes:
    """verify ConfigError is raised on type mismatches for required fields."""

    def test_camera_device_must_be_int(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"]["device"] = "two"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"camera\.device"):
            Config(path)

    def test_camera_autofocus_must_be_bool(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"]["autofocus"] = "yes"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"camera\.autofocus"):
            Config(path)

    def test_uart_port_must_be_str(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["uart"]["port"] = 12345
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"uart\.port"):
            Config(path)

@pytest.mark.regression
class TestValidationRanges:
    """verify ConfigError is raised when numeric values are out of range."""

    def test_focus_below_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"]["focus"] = 0
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="focus"):
            Config(path)

    def test_focus_above_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"]["focus"] = 1024
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="focus"):
            Config(path)

    def test_blur_kernel_even_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["preprocess"]["blur_kernel"] = 4
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="blur_kernel must be odd"):
            Config(path)

    def test_highlight_max_above_1(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["thresholds"]["highlight_max"] = 1.5
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="highlight_max"):
            Config(path)

    def test_circularity_min_negative(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["thresholds"]["circularity_min"] = -0.1
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="circularity_min"):
            Config(path)

    def test_fps_below_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["camera"]["fps"] = 0
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="fps must be positive"):
            Config(path)

    def test_morph_kernel_below_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["preprocess"]["morph_kernel"] = 0
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="morph_kernel must be positive"):
            Config(path)

    def test_solidity_min_out_of_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["thresholds"]["solidity_min"] = 1.5
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="solidity_min"):
            Config(path)

    def test_colour_confidence_min_out_of_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["thresholds"]["colour_confidence_min"] = -0.1
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="colour_confidence_min"):
            Config(path)

    def test_decision_min_out_of_range(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["thresholds"]["decision_min"] = 1.5
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="decision_min"):
            Config(path)

    def test_decision_min_missing_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        del data["thresholds"]["decision_min"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="decision_min"):
            Config(path)

    def test_baud_too_low(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["uart"]["baud"] = 300
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="baud"):
            Config(path)

@pytest.mark.regression
class TestValidationColours:
    """verify ConfigError is raised on malformed colour entries."""

    def test_empty_colours_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"] = {}
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="at least one colour"):
            Config(path)

    def test_colour_missing_h(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        del data["colours"]["red"]["h"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"red.*h"):
            Config(path)

    def test_colour_missing_s(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        del data["colours"]["red"]["s"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"red.*s"):
            Config(path)

    def test_colour_missing_v(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        del data["colours"]["red"]["v"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"red.*v"):
            Config(path)

    def test_h_empty_list(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["h"] = []
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"h.*non-empty"):
            Config(path)

    def test_h_bad_pair(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["h"] = [[10, 20, 30]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"min, max.*pair"):
            Config(path)

    def test_h_min_greater_than_max(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["h"] = [[50, 10]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"min.*>.*max"):
            Config(path)

    def test_h_non_numeric(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["h"] = [["a", "b"]]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="must be numbers"):
            Config(path)

    def test_s_bad_pair(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["s"] = [100]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"min, max.*pair"):
            Config(path)

    def test_v_min_greater_than_max(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"]["v"] = [255, 50]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match=r"min.*>.*max"):
            Config(path)

    def test_unknown_colour_name_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["purple"] = data["colours"]["red"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="not a recognised colour"):
            Config(path)

    def test_non_mm_as_colour_name_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["non-m&m"] = data["colours"]["red"]
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="not a recognised colour"):
            Config(path)

    def test_colour_entry_not_a_dict_raises(self, full_data, tmp_path):
        data = copy.deepcopy(full_data)
        data["colours"]["red"] = "not_a_dict"
        path = write_config(data, tmp_path)
        with pytest.raises(ConfigError, match="must be a mapping"):
            Config(path)

@pytest.mark.regression
class TestValidationOptionalFields:
    """verify ConfigError is raised on optional field type and range violations."""

    def test_auto_exposure_wrong_type(self):
        with pytest.raises(ConfigError, match=r"camera\.auto_exposure"):
            make_config(camera={"auto_exposure": "auto"})

    def test_auto_wb_wrong_type(self):
        with pytest.raises(ConfigError, match=r"camera\.auto_wb"):
            make_config(camera={"auto_wb": "yes"})

    def test_roi_fraction_wrong_type(self):
        with pytest.raises(ConfigError, match=r"preprocess\.roi_fraction"):
            make_config(preprocess={"roi_fraction": "half"})

    def test_timeout_wrong_type(self):
        with pytest.raises(ConfigError, match=r"uart\.timeout"):
            make_config(uart={"timeout": "1s"})

    def test_display_scale_wrong_type(self):
        with pytest.raises(ConfigError, match=r"system\.display_scale"):
            make_config(system={"display_scale": "large"})

    def test_auto_exposure_invalid_value(self):
        with pytest.raises(ConfigError, match="auto_exposure"):
            make_config(camera={"auto_exposure": 2})

    def test_power_line_frequency_invalid_value(self):
        with pytest.raises(ConfigError, match="power_line_frequency"):
            make_config(camera={"power_line_frequency": 3})

    def test_exposure_zero(self):
        with pytest.raises(ConfigError, match="exposure"):
            make_config(camera={"exposure": 0})

    def test_wb_temperature_below_range(self):
        with pytest.raises(ConfigError, match="wb_temperature"):
            make_config(camera={"wb_temperature": 500})

    def test_wb_temperature_above_range(self):
        with pytest.raises(ConfigError, match="wb_temperature"):
            make_config(camera={"wb_temperature": 11000})

    def test_roi_fraction_zero(self):
        with pytest.raises(ConfigError, match="roi_fraction"):
            make_config(preprocess={"roi_fraction": 0.0})

    def test_roi_fraction_above_one(self):
        with pytest.raises(ConfigError, match="roi_fraction"):
            make_config(preprocess={"roi_fraction": 1.1})

    def test_morph_erode_iter_negative(self):
        with pytest.raises(ConfigError, match="morph_erode_iter"):
            make_config(preprocess={"morph_erode_iter": -1})

    def test_sat_min_dark_above_range(self):
        with pytest.raises(ConfigError, match="sat_min_dark"):
            make_config(preprocess={"sat_min_dark": 256})

    def test_val_max_dark_negative(self):
        with pytest.raises(ConfigError, match="val_max_dark"):
            make_config(preprocess={"val_max_dark": -1})

    def test_timeout_negative(self):
        with pytest.raises(ConfigError, match="timeout"):
            make_config(uart={"timeout": -1})

    def test_found_frames_min_zero(self):
        with pytest.raises(ConfigError, match="found_frames_min"):
            make_config(system={"found_frames_min": 0})

    def test_display_scale_zero(self):
        with pytest.raises(ConfigError, match="display_scale"):
            make_config(system={"display_scale": 0})

    def test_camera_device_negative(self):
        with pytest.raises(ConfigError, match="device"):
            make_config(camera={"device": -1})

    def test_camera_format_invalid(self):
        with pytest.raises(ConfigError, match="format"):
            make_config(camera={"format": "RAW"})

    def test_hue_bins_zero(self):
        with pytest.raises(ConfigError, match="hue_bins"):
            make_config(features={"hue_bins": 0})

    def test_highlight_value_above_range(self):
        with pytest.raises(ConfigError, match="highlight_value"):
            make_config(features={"highlight_value": 256})

    def test_hue_peak_ratio_above_one(self):
        with pytest.raises(ConfigError, match="hue_peak_ratio"):
            make_config(features={"hue_peak_ratio": 1.5})

    def test_aspect_ratio_max_below_one(self):
        with pytest.raises(ConfigError, match="aspect_ratio_max"):
            make_config(thresholds={"aspect_ratio_max": 0.9})

    def test_log_queue_size_zero(self):
        with pytest.raises(ConfigError, match="log_queue_size"):
            make_config(system={"log_queue_size": 0})

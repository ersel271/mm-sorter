# tests/unit/test_camera.py

import copy
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import pytest
import numpy as np

from config import Config
from src.io import Camera
from tests.helpers.config_helpers import FULL_DATA, make_config

@pytest.fixture(autouse=True)
def _stub_subprocess():
    with patch("src.io.camera.subprocess.run", side_effect=FileNotFoundError):
        yield

@pytest.mark.smoke
class TestCameraInit:
    """verify Camera initialises without opening a device."""

    def test_not_open_after_init(self, default_cfg):
        cam = Camera(default_cfg)
        assert cam.is_open is False

    def test_read_before_open_returns_false(self, default_cfg):
        cam = Camera(default_cfg)
        ok, frame = cam.read()
        assert ok is False
        assert frame is None

@pytest.mark.smoke
@pytest.mark.regression
class TestCameraOpenMock:
    """verify open() applies config settings via cv2.VideoCapture."""

    def test_open_success(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_cap.set.return_value = True
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            assert cam.open() is True
            assert cam.is_open is True

    def test_open_failure(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            assert cam.open() is False
            assert cam.is_open is False

    def test_open_sets_resolution(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()

            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls[cv2.CAP_PROP_FRAME_WIDTH] == 1920
            assert set_calls[cv2.CAP_PROP_FRAME_HEIGHT] == 1080

    def test_open_disables_autofocus_when_config_says_so(self, default_cfg):
        assert default_cfg.camera["autofocus"] is False
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()

            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls[cv2.CAP_PROP_AUTOFOCUS] == 0
            assert set_calls[cv2.CAP_PROP_FOCUS] == default_cfg.camera["focus"]

    def test_open_applies_autofocus_enabled(self):
        cfg = make_config(camera={"autofocus": True})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls[cv2.CAP_PROP_AUTOFOCUS] == 1

    def test_open_applies_manual_exposure(self):
        cfg = make_config(camera={"auto_exposure": 1, "exposure": 200})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls.get(cv2.CAP_PROP_EXPOSURE) == 200

    def test_open_applies_manual_white_balance(self):
        cfg = make_config(camera={"auto_wb": False, "wb_temperature": 5000})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls.get(cv2.CAP_PROP_WB_TEMPERATURE) == 5000

    def test_open_applies_saturation(self):
        cfg = make_config(camera={"saturation": 90})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls.get(cv2.CAP_PROP_SATURATION) == 90

    def test_open_skips_saturation_when_absent(self):
        data = copy.deepcopy(FULL_DATA)
        del data["camera"]["saturation"]
        tmp = Path(tempfile.mkdtemp()) / "config.yaml"
        with open(tmp, "w") as f:
            yaml.dump(data, f)
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(Config(tmp))
            cam.open()
            set_props = [call[0][0] for call in mock_cap.set.call_args_list]
            assert cv2.CAP_PROP_SATURATION not in set_props

    def test_open_applies_gamma(self):
        cfg = make_config(camera={"gamma": 85})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            set_calls = {call[0][0]: call[0][1] for call in mock_cap.set.call_args_list}
            assert set_calls.get(cv2.CAP_PROP_GAMMA) == 85

    def test_open_skips_gamma_when_absent(self):
        data = copy.deepcopy(FULL_DATA)
        del data["camera"]["gamma"]
        tmp = Path(tempfile.mkdtemp()) / "config.yaml"
        with open(tmp, "w") as f:
            yaml.dump(data, f)
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(Config(tmp))
            cam.open()
            set_props = [call[0][0] for call in mock_cap.set.call_args_list]
            assert cv2.CAP_PROP_GAMMA not in set_props

@pytest.mark.smoke
@pytest.mark.regression
class TestCameraReadMock:
    """verify read() returns frames or failure from the underlying capture."""

    def test_read_success(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            ok, frame = cam.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (1080, 1920, 3)

    def test_read_failure(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_cap.read.return_value = (False, None)
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            ok, frame = cam.read()
            assert ok is False
            assert frame is None

class TestCameraSetters:
    """verify focus, exposure, and white balance setters."""

    @pytest.fixture
    def open_cam(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_cap.set.return_value = True
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            yield cam, mock_cap

    def test_set_focus(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_focus(500) is True

    def test_set_autofocus(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_autofocus(True) is True

    def test_set_exposure_auto(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_exposure(3) is True

    def test_set_exposure_manual(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_exposure(1, value=200) is True

    def test_set_white_balance_auto(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_white_balance(auto=True) is True

    def test_set_white_balance_manual(self, open_cam):
        cam, _mock_cap = open_cam
        assert cam.set_white_balance(auto=False, temperature=5000) is True

    def test_setters_return_false_when_not_open(self, default_cfg):
        cam = Camera(default_cfg)
        assert cam.set_focus(500) is False
        assert cam.set_autofocus(True) is False
        assert cam.set_exposure(3) is False
        assert cam.set_white_balance(auto=True) is False

    def test_set_focus_failure(self, open_cam):
        cam, mock_cap = open_cam
        mock_cap.set.side_effect = lambda prop, val: prop != cv2.CAP_PROP_FOCUS
        assert cam.set_focus(500) is False

    def test_set_white_balance_manual_no_temperature(self, open_cam):
        cam, mock_cap = open_cam
        mock_cap.set.reset_mock()
        result = cam.set_white_balance(auto=False, temperature=None)
        assert result is True
        set_props = [call[0][0] for call in mock_cap.set.call_args_list]
        assert cv2.CAP_PROP_WB_TEMPERATURE not in set_props

class TestCameraPowerLineFreq:
    """verify _apply_power_line_freq() dispatches correctly via v4l2-ctl."""

    def _open_cam(self, cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap
            cam = Camera(cfg)
            cam.open()
            return cam

    def test_applied_on_linux(self):
        cfg = make_config(camera={"power_line_frequency": 1})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "linux"), \
             patch("src.io.camera.subprocess.run") as mock_run:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            mock_probe = MagicMock()
            mock_probe.stdout = "power_line_frequency 0x00980918 (menu): value=1"
            mock_run.side_effect = [mock_probe, MagicMock()]

            cam = Camera(cfg)
            cam.open()

            assert mock_run.call_count == 2
            set_call_args = mock_run.call_args_list[1][0][0]
            assert "--set-ctrl=power_line_frequency=1" in set_call_args

    def test_skipped_when_zero(self):
        cfg = make_config(camera={"power_line_frequency": 0})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "linux"), \
             patch("src.io.camera.subprocess.run") as mock_run:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(cfg)
            cam.open()

            mock_run.assert_not_called()

    def test_skipped_on_non_linux(self):
        cfg = make_config(camera={"power_line_frequency": 1})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "win32"), \
             patch("src.io.camera.subprocess.run") as mock_run:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(cfg)
            cam.open()

            mock_run.assert_not_called()

    def test_skipped_when_not_supported(self):
        cfg = make_config(camera={"power_line_frequency": 1})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "linux"), \
             patch("src.io.camera.subprocess.run") as mock_run:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            mock_probe = MagicMock()
            mock_probe.stdout = "brightness 0x00980900 (int): value=128"
            mock_run.side_effect = [mock_probe]

            cam = Camera(cfg)
            cam.open()

            assert mock_run.call_count == 1

    def test_v4l2_not_found(self):
        cfg = make_config(camera={"power_line_frequency": 1})
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "linux"), \
             patch("src.io.camera.subprocess.run", side_effect=FileNotFoundError):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(cfg)
            assert cam.open() is True

    def test_v4l2_error(self):
        import subprocess as _subprocess
        cfg = make_config(camera={"power_line_frequency": 1})
        err = _subprocess.CalledProcessError(1, "v4l2-ctl")
        err.stderr = "operation not permitted"
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc, \
             patch("src.io.camera.sys.platform", "linux"), \
             patch("src.io.camera.subprocess.run", side_effect=err):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(cfg)
            assert cam.open() is True

@pytest.mark.smoke
class TestCameraRelease:
    """verify release() cleans up the capture device."""

    def test_release(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            cam.release()
            assert cam.is_open is False
            mock_cap.release.assert_called_once()

    def test_release_when_not_open(self, default_cfg):
        cam = Camera(default_cfg)
        cam.release()
        assert cam.is_open is False

@pytest.mark.smoke
class TestCameraProperties:
    """verify get_properties() reads values from the driver."""

    def test_get_properties_when_not_open(self, default_cfg):
        cam = Camera(default_cfg)
        assert cam.get_properties() == {}

    def test_get_properties_returns_dict(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            props = cam.get_properties()
            assert "width" in props
            assert "height" in props
            assert "fps" in props
            assert "format" in props
            assert "focus" in props

    def test_get_properties_includes_saturation_and_gamma(self, default_cfg):
        with patch("src.io.camera.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0
            mock_vc.return_value = mock_cap

            cam = Camera(default_cfg)
            cam.open()
            props = cam.get_properties()
            assert "saturation" in props
            assert "gamma" in props

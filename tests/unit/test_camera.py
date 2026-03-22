# tests/test_camera.py

from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.io import Camera

@pytest.mark.unit
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

@pytest.mark.unit
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
            import cv2
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
            import cv2
            assert set_calls[cv2.CAP_PROP_AUTOFOCUS] == 0
            assert set_calls[cv2.CAP_PROP_FOCUS] == default_cfg.camera["focus"]

@pytest.mark.unit
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

@pytest.mark.unit
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
        cam, mock_cap = open_cam
        assert cam.set_focus(500) is True

    def test_set_autofocus(self, open_cam):
        cam, mock_cap = open_cam
        assert cam.set_autofocus(True) is True

    def test_set_exposure_auto(self, open_cam):
        cam, mock_cap = open_cam
        assert cam.set_exposure(3) is True

    def test_set_exposure_manual(self, open_cam):
        cam, mock_cap = open_cam
        assert cam.set_exposure(1, value=200) is True

    def test_set_white_balance_auto(self, open_cam):
        cam, mock_cap = open_cam
        assert cam.set_white_balance(auto=True) is True

    def test_set_white_balance_manual(self, open_cam):
        cam, mock_cap = open_cam
        assert cam.set_white_balance(auto=False, temperature=5000) is True

    def test_setters_return_false_when_not_open(self, default_cfg):
        cam = Camera(default_cfg)
        assert cam.set_focus(500) is False
        assert cam.set_autofocus(True) is False
        assert cam.set_exposure(3) is False
        assert cam.set_white_balance(auto=True) is False

@pytest.mark.unit
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

@pytest.mark.unit
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

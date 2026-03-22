# tests/hw/test_camera_hw.py

import pytest

from config import Config
from src.io import Camera

@pytest.mark.hw
class TestCameraHardware:
    """requires a physical camera connected at the configured device index."""

    def test_open_and_read(self, default_cfg):
        cam = Camera(default_cfg)
        if not cam.open():
            pytest.skip("camera not available")
        ok, frame = cam.read()
        assert ok is True
        assert frame is not None
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0
        cam.release()

    def test_focus_control(self, default_cfg):
        cam = Camera(default_cfg)
        if not cam.open():
            pytest.skip("camera not available")
        assert cam.set_focus(200) is True
        props = cam.get_properties()
        assert props["focus"] == 200
        cam.release()

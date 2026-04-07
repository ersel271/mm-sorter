# tests/hw/test_camera_hw.py

import sys
import subprocess

import pytest

from src.io import Camera

@pytest.mark.hw
@pytest.mark.camera
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

    def test_power_line_frequency(self, default_cfg):
        if sys.platform != "linux":
            pytest.skip("v4l2-ctl only available on Linux")
        cam = Camera(default_cfg)
        if not cam.open():
            pytest.skip("camera not available")
        device = int(default_cfg.camera["device"])
        result = subprocess.run(
            ["v4l2-ctl", f"--device=/dev/video{device}", "--get-ctrl=power_line_frequency"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            cam.release()
            pytest.skip("power_line_frequency not supported by this camera")
        # output format: "power_line_frequency: 1"
        actual = int(result.stdout.split(":")[1].strip())
        assert actual == default_cfg.camera["power_line_frequency"]
        cam.release()

# tests/integration/test_sort_pipeline.py

import sys
from unittest.mock import MagicMock, patch

import pytest

from sort import pipeline
from tests.helpers.config_helpers import make_config
from tests.helpers.inject_helpers import make_inject_folder

def _pipeline_cfg(tmp_path, found_frames_min=1):
    return make_config(system={
        "display_enabled": False,
        "found_frames_min": found_frames_min,
        "plot_dir": str(tmp_path / "plots"),
        "event_dir": str(tmp_path / "events"),
        "log_dir": str(tmp_path / "logs"),
    })

@pytest.mark.pipeline
@pytest.mark.smoke
@pytest.mark.regression
class TestPipelineFullRun:
    """verify pipeline() runs to completion under controlled stop conditions"""

    def test_exits_cleanly_with_max_objects(self, monkeypatch, tmp_path, sender):
        folder = make_inject_folder(tmp_path, n_images=3)
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "sort.py",
            "--inject-from", str(folder),
            "--max-objects", "1",
            "--log-level", "WARNING",
        ])
        with patch("sort.Config", return_value=cfg):
            with patch("sort.UARTSender", return_value=sender):
                result = pipeline()
        assert result == 0

    def test_exits_cleanly_with_timeout(self, monkeypatch, tmp_path, sender):
        folder = make_inject_folder(tmp_path, n_images=3)
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "sort.py",
            "--inject-from", str(folder),
            "--timeout", "0.3",
            "--log-level", "WARNING",
        ])
        with patch("sort.Config", return_value=cfg):
            with patch("sort.UARTSender", return_value=sender):
                result = pipeline()
        assert result == 0

    def test_camera_open_failure_returns_one(self, monkeypatch, tmp_path, sender):
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sort.py", "--log-level", "WARNING"])
        with patch("sort.Config", return_value=cfg):
            with patch("sort.Camera") as mock_cam_cls:
                mock_cam = MagicMock()
                mock_cam.open.return_value = False
                mock_cam_cls.return_value = mock_cam
                with patch("sort.UARTSender", return_value=sender):
                    result = pipeline()
        assert result == 1

    def test_pipeline_with_ground_truth(self, monkeypatch, tmp_path, sender):
        from tests.helpers.ground_truth_helpers import write_gt
        gt_file = write_gt(tmp_path, ["red"])
        folder = make_inject_folder(tmp_path, n_images=3)
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "sort.py",
            "--inject-from", str(folder),
            "--ground-truth", str(gt_file),
            "--max-objects", "1",
            "--log-level", "WARNING",
        ])
        with patch("sort.Config", return_value=cfg):
            with patch("sort.UARTSender", return_value=sender):
                result = pipeline()
        assert result == 0

# tests/integration/test_sort_pipeline.py

import sys
from unittest.mock import MagicMock, patch

from src.io import build_packet, PCK_START, PCK_END_OK, PCK_END_ERR, PCK_FREEZE_START

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

    def test_start_and_end_ok_packets_sent(self, monkeypatch, tmp_path, sender, mock_port):
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
                pipeline()
        writes = [c.args[0] for c in mock_port.write.call_args_list]
        assert writes[0]  == build_packet(PCK_START)
        assert writes[-1] == build_packet(PCK_END_OK)

    def test_end_err_packet_sent_on_camera_failure(self, monkeypatch, tmp_path, sender, mock_port):
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", ["sort.py", "--log-level", "WARNING"])
        with patch("sort.Config", return_value=cfg):
            with patch("sort.Camera") as mock_cam_cls:
                mock_cam = MagicMock()
                mock_cam.open.return_value = False
                mock_cam_cls.return_value = mock_cam
                with patch("sort.UARTSender", return_value=sender):
                    pipeline()
        writes = [c.args[0] for c in mock_port.write.call_args_list]
        assert writes[0]  == build_packet(PCK_START)
        assert writes[-1] == build_packet(PCK_END_ERR)

    def test_freeze_start_sent_and_quit_while_frozen(self, monkeypatch, tmp_path, sender, mock_port):
        folder = make_inject_folder(tmp_path, n_images=10)
        cfg = _pipeline_cfg(tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "sort.py",
            "--inject-from", str(folder),
            "--log-level", "WARNING",
        ])
        call_count = 0

        def mock_display(frame, result, features, decision, ov, uart, is_record_frame, low_conf=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                ov.toggle_freeze()  # triggers PCK_FREEZE_START on return
                return False
            return True  # quit: normal path (call 2+), or frozen path if already frozen

        with patch("sort.Config", return_value=cfg):
            with patch("sort.UARTSender", return_value=sender):
                with patch("sort.display", side_effect=mock_display):
                    result = pipeline()

        assert result == 0
        writes = [c.args[0] for c in mock_port.write.call_args_list]
        assert build_packet(PCK_FREEZE_START) in writes

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

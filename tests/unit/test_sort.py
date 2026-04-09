# tests/unit/test_sort.py

import sys
import time
from unittest.mock import MagicMock, call, patch

import pytest
import numpy as np

from config.constants import ColourID
from src.io import PCK_END_ERR, PCK_END_OK
from sort import (
    PipelineState,
    StopReason,
    _accumulate,
    _handle_gt,
    _make_uart_payload,
    _process_found_frame,
    _resolve_class,
    _setup_frame_source,
    _setup_gt,
    _setup_plot_worker,
    _submit_plots,
    parse_args,
    acquire,
    classify,
    display,
    extract,
    record,
    teardown,
)
from tests.helpers.config_helpers import make_config
from tests.helpers.vision_helpers import make_decision, make_features, make_preprocess_result
from tests.helpers.ground_truth_helpers import write_gt
from tests.helpers.inject_helpers import make_inject_folder

@pytest.mark.smoke
@pytest.mark.regression
class TestParseArgs:
    """verify CLI argument parsing defaults and overrides"""

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py"])
        args = parse_args()
        assert args.plot is False
        assert args.ground_truth is None
        assert args.inject_from is None
        assert args.max_objects is None
        assert args.timeout is None
        assert args.config is None
        assert args.log_level == "INFO"

    def test_plot_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--plot"])
        assert parse_args().plot is True

    def test_config_path(self, monkeypatch, tmp_path):
        p = str(tmp_path / "cfg.yaml")
        monkeypatch.setattr(sys, "argv", ["sort.py", "--config", p])
        assert parse_args().config == p

    def test_log_level_debug(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--log-level", "DEBUG"])
        assert parse_args().log_level == "DEBUG"

    def test_invalid_log_level_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--log-level", "VERBOSE"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_max_objects_parsed_as_int(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--max-objects", "5"])
        assert parse_args().max_objects == 5

    def test_invalid_max_objects_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--max-objects", "abc"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_timeout_parsed_as_float(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--timeout", "2.5"])
        assert parse_args().timeout == pytest.approx(2.5)

    def test_inject_from_stored(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--inject-from", str(tmp_path)])
        assert parse_args().inject_from == str(tmp_path)

    def test_ground_truth_stored(self, monkeypatch, tmp_path):
        p = str(tmp_path / "gt.txt")
        monkeypatch.setattr(sys, "argv", ["sort.py", "--ground-truth", p])
        assert parse_args().ground_truth == p

@pytest.mark.smoke
class TestPipelineState:
    """verify PipelineState default field values"""

    def test_default_object_id(self):
        assert PipelineState().object_id == 0

    def test_default_frame_num(self):
        assert PipelineState().frame_num == 0

    def test_default_stop_reason(self):
        assert PipelineState().stop_reason == StopReason.USER_QUIT

    def test_default_timeout_is_infinite(self):
        assert PipelineState().timeout_at == float("inf")

    def test_default_low_conf_is_false(self):
        assert PipelineState().low_conf is False

    def test_default_lists_empty(self):
        s = PipelineState()
        assert s.pred_list == []
        assert s.conf_list == []
        assert s.gt_labels == {}

@pytest.mark.smoke
@pytest.mark.regression
class TestSetupFrameSource:
    """verify _setup_frame_source returns iterator or camera based on args"""

    def test_inject_from_returns_iterator(self, monkeypatch, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        monkeypatch.setattr(sys, "argv", ["sort.py", "--inject-from", str(folder)])
        args = parse_args()
        frames_iter, cam = _setup_frame_source(args, make_config(), 1)
        assert frames_iter is not None
        assert cam is None

    def test_inject_from_empty_folder_raises(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(sys, "argv", ["sort.py", "--inject-from", str(empty)])
        args = parse_args()
        with pytest.raises(RuntimeError, match="injection folder is empty"):
            frames, _cam = _setup_frame_source(args, make_config(), 1)
            next(frames)

    def test_no_inject_opens_camera(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py"])
        args = parse_args()
        with patch("sort.Camera") as mock_cam_cls:
            mock_cam = MagicMock()
            mock_cam.open.return_value = True
            mock_cam_cls.return_value = mock_cam
            frames_iter, cam = _setup_frame_source(args, make_config(), 1)
        assert frames_iter is None
        assert cam is mock_cam

    def test_camera_open_failure_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py"])
        args = parse_args()
        with patch("sort.Camera") as mock_cam_cls:
            mock_cam = MagicMock()
            mock_cam.open.return_value = False
            mock_cam_cls.return_value = mock_cam
            with pytest.raises(RuntimeError, match="camera failed to open"):
                _setup_frame_source(args, make_config(), 1)

@pytest.mark.smoke
class TestSetupGT:
    """verify _setup_gt loads labels or returns empty on missing or absent file"""

    def test_no_ground_truth_returns_empty(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py"])
        labels, acc = _setup_gt(parse_args())
        assert labels == {}
        assert len(acc) == 0

    def test_valid_file_loaded(self, monkeypatch, tmp_path):
        gt_file = write_gt(tmp_path, ["red", "green"])
        monkeypatch.setattr(sys, "argv", ["sort.py", "--ground-truth", str(gt_file)])
        labels, _acc = _setup_gt(parse_args())
        assert labels == {1: 1, 2: 2}

    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--ground-truth", str(tmp_path / "missing.txt")])
        labels, _ = _setup_gt(parse_args())
        assert labels == {}

@pytest.mark.smoke
class TestSetupPlotWorker:
    """verify _setup_plot_worker creates or skips the plot worker"""

    def test_no_plot_returns_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sort.py"])
        assert _setup_plot_worker(parse_args(), make_config()) is None

    def test_plot_flag_returns_worker(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["sort.py", "--plot"])
        cfg = make_config(system={"plot_dir": str(tmp_path / "plots")})
        worker = _setup_plot_worker(parse_args(), cfg)
        assert worker is not None
        worker.stop()

@pytest.mark.smoke
class TestMakeUartPayload:
    """verify _make_uart_payload builds the expected dict fields"""

    def test_fields_with_centroid(self):
        result = make_preprocess_result()
        decision = make_decision(confidence=0.88)
        state = PipelineState(object_id=3)
        payload = _make_uart_payload(state, result, decision, int(ColourID.RED))
        assert payload["id"] == 3
        assert payload["class"] == int(ColourID.RED)
        assert payload["conf"] == pytest.approx(0.88)
        assert payload["x"] == result.centroid[0]
        assert payload["y"] == result.centroid[1]

    def test_fields_without_centroid(self):
        mock_result = MagicMock()
        mock_result.centroid = None
        state = PipelineState()
        payload = _make_uart_payload(state, mock_result, make_decision(), 0)
        assert payload["x"] == 0
        assert payload["y"] == 0

@pytest.mark.smoke
class TestAccumulate:
    """verify _accumulate appends prediction and confidence to state lists"""

    def test_appends_prediction(self):
        state = PipelineState()
        _accumulate(state, make_decision(label=ColourID.RED))
        assert state.pred_list == [int(ColourID.RED)]

    def test_appends_confidence(self):
        state = PipelineState()
        _accumulate(state, make_decision(confidence=0.85))
        assert state.conf_list == [pytest.approx(0.85)]

    def test_multiple_calls_grow_lists(self):
        state = PipelineState()
        _accumulate(state, make_decision())
        _accumulate(state, make_decision())
        assert len(state.pred_list) == 2
        assert len(state.conf_list) == 2

@pytest.mark.regression
class TestHandleGT:
    """verify _handle_gt records pairs only when labels are present and match"""

    def test_no_labels_does_nothing(self):
        state = PipelineState()
        _handle_gt(state, make_decision())
        assert len(state.gt_acc) == 0

    def test_matching_label_records_pair(self):
        state = PipelineState(object_id=1, gt_labels={1: 2})
        _handle_gt(state, make_decision(label=ColourID.RED))
        assert len(state.gt_acc) == 1
        assert state.gt_acc.pairs == [(2, int(ColourID.RED))]

    def test_no_matching_object_id_skips(self):
        state = PipelineState(object_id=5, gt_labels={1: 2})
        _handle_gt(state, make_decision())
        assert len(state.gt_acc) == 0

@pytest.mark.smoke
class TestSubmitPlots:
    """verify _submit_plots enqueues or skips based on worker and accumulated data"""

    def test_no_worker_does_nothing(self):
        _submit_plots(PipelineState())  # must not raise

    def test_empty_pred_list_skips(self):
        mock_worker = MagicMock()
        _submit_plots(PipelineState(plot_worker=mock_worker))
        mock_worker.enqueue.assert_not_called()

    def test_with_predictions_enqueues(self):
        mock_worker = MagicMock()
        state = PipelineState(plot_worker=mock_worker, pred_list=[1, 2], conf_list=[0.9, 0.8])
        _submit_plots(state)
        mock_worker.enqueue.assert_called_once()

    def test_with_gt_acc_enqueues_gt_plot(self):
        from utils.ground_truth import GTSession
        mock_worker = MagicMock()
        gt_acc = GTSession()
        gt_acc.record(1, 2, 0.9)
        state = PipelineState(
            plot_worker=mock_worker,
            pred_list=[1],
            conf_list=[0.9],
            gt_labels={1: 1},
            gt_acc=gt_acc,
        )
        _submit_plots(state)
        mock_worker.enqueue.assert_called_once()

@pytest.mark.smoke
@pytest.mark.regression
class TestAcquire:
    """verify acquire returns a frame from iterator or camera, None on camera failure"""

    def test_from_iterator(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = acquire(iter([frame]), None)
        assert result is frame

    def test_from_camera_success(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cam = MagicMock()
        mock_cam.read.return_value = (True, frame)
        assert acquire(None, mock_cam) is frame

    def test_from_camera_failure_returns_none(self):
        mock_cam = MagicMock()
        mock_cam.read.return_value = (False, None)
        assert acquire(None, mock_cam) is None

@pytest.mark.smoke
@pytest.mark.regression
class TestExtract:
    """verify extract wraps feature extraction and returns None on ValueError"""

    def test_returns_features_on_success(self, extractor):
        features = extract(make_preprocess_result(), extractor, PipelineState())
        assert features is not None

    def test_returns_none_on_value_error(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = ValueError("bad contour")
        assert extract(make_preprocess_result(), mock_extractor, PipelineState()) is None

@pytest.mark.smoke
@pytest.mark.regression
class TestClassify:
    """verify classify returns None for missing features and wraps ValueError"""

    def test_none_features_returns_none(self, classifier):
        assert classify(None, classifier, PipelineState()) is None

    def test_returns_decision_on_success(self, extractor, classifier):
        features = extract(make_preprocess_result(), extractor, PipelineState())
        decision = classify(features, classifier, PipelineState())
        assert decision is not None

    def test_returns_none_on_value_error(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = ValueError("bad features")
        assert classify(make_features(), mock_classifier, PipelineState()) is None

@pytest.mark.regression
class TestResolveClass:
    """verify _resolve_class returns label or NON_MM based on low_conf flag"""

    def test_low_conf_false_returns_decision_label(self):
        decision = make_decision(label=ColourID.RED)
        assert _resolve_class(decision, 1, low_conf=False) == int(ColourID.RED)

    def test_low_conf_true_returns_non_mm(self):
        decision = make_decision(label=ColourID.RED)
        assert _resolve_class(decision, 1, low_conf=True) == int(ColourID.NON_MM)

    def test_low_conf_true_non_mm_input_stays_non_mm(self):
        decision = make_decision(label=ColourID.NON_MM)
        assert _resolve_class(decision, 1, low_conf=True) == int(ColourID.NON_MM)

@pytest.mark.smoke
@pytest.mark.regression
class TestRecord:
    """verify record increments counters and returns limit-reached flag"""

    def _call(self, state, max_objects=5):
        return record(
            make_preprocess_result(), make_features(), make_decision(),
            time.monotonic(), MagicMock(), MagicMock(),
            max_objects=max_objects, state=state,
        )

    def test_returns_false_below_max(self):
        state = PipelineState(object_id=1)
        assert self._call(state) is False
        assert state.classified_count == 1

    def test_returns_true_at_max_objects(self):
        state = PipelineState(object_id=1, classified_count=4)
        assert self._call(state, max_objects=5) is True

    def test_no_max_objects_never_stops(self):
        state = PipelineState(object_id=1, classified_count=9999)
        assert self._call(state, max_objects=None) is False

    def test_classified_count_incremented(self):
        state = PipelineState(object_id=1)
        self._call(state)
        assert state.classified_count == 1

@pytest.mark.smoke
class TestTeardown:
    """verify teardown stops all workers and releases hardware resources"""

    def test_stops_plot_worker_when_present(self):
        mock_plot = MagicMock()
        teardown(mock_plot, None, MagicMock(), MagicMock())
        mock_plot.stop.assert_called_once()

    def test_skips_plot_worker_when_none(self):
        teardown(None, None, MagicMock(), MagicMock())  # must not raise

    def test_releases_camera_when_present(self):
        mock_cam = MagicMock()
        teardown(None, mock_cam, MagicMock(), MagicMock())
        mock_cam.release.assert_called_once()

    def test_skips_camera_when_none(self):
        teardown(None, None, MagicMock(), MagicMock())  # must not raise

    def test_closes_uart_and_stops_event_worker(self):
        mock_uart = MagicMock()
        mock_event = MagicMock()
        teardown(None, None, mock_uart, mock_event)
        mock_uart.send.assert_called_once_with(PCK_END_OK)
        mock_uart.close.assert_called_once()
        mock_event.stop.assert_called_once()

    def test_sends_end_ok_before_close(self):
        mock_uart = MagicMock()
        manager = MagicMock()
        manager.attach_mock(mock_uart.send, "send")
        manager.attach_mock(mock_uart.close, "close")
        teardown(None, None, mock_uart, MagicMock())
        manager.assert_has_calls([call.send(PCK_END_OK), call.close()])

    def test_sends_end_err_on_error(self):
        mock_uart = MagicMock()
        teardown(None, None, mock_uart, MagicMock(), error=True)
        mock_uart.send.assert_called_once_with(PCK_END_ERR)

    def test_sends_end_ok_on_clean_exit(self):
        mock_uart = MagicMock()
        teardown(None, None, mock_uart, MagicMock(), error=False)
        mock_uart.send.assert_called_once_with(PCK_END_OK)

    def test_end_sent_even_when_camera_and_plot_absent(self):
        mock_uart = MagicMock()
        teardown(None, None, mock_uart, MagicMock())
        mock_uart.send.assert_called_once_with(PCK_END_OK)

@pytest.mark.smoke
@pytest.mark.regression
class TestProcessFoundFrame:
    """verify _process_found_frame classifies, sets state, and signals stop correctly"""

    def _call(self, state, threshold=0.5, max_objects=None):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = make_features()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = make_decision()
        return _process_found_frame(
            make_preprocess_result(), mock_extractor, mock_classifier,
            MagicMock(), MagicMock(), state,
            found_frames_min=3, threshold=threshold,
            max_objects=max_objects, t0=time.monotonic(),
        )

    def test_increments_found_count(self):
        state = PipelineState()
        self._call(state)
        assert state.found_count == 1

    def test_sets_object_id_on_first_found(self):
        state = PipelineState(object_id=0)
        self._call(state)
        assert state.object_id == 1

    def test_does_not_reset_object_id_on_subsequent_found(self):
        state = PipelineState(object_id=1, found_count=1)
        self._call(state)
        assert state.object_id == 1

    def test_stop_false_when_below_max(self):
        state = PipelineState(object_id=1)
        _, _, _, stop = self._call(state, max_objects=10)
        assert stop is False

    def test_is_record_frame_true_at_found_frames_min(self):
        state = PipelineState(object_id=1, found_count=2)
        _, _, is_record_frame, _ = self._call(state)
        assert is_record_frame is True

    def test_is_record_frame_false_before_min(self):
        state = PipelineState(object_id=1, found_count=0)
        _, _, is_record_frame, _ = self._call(state)
        assert is_record_frame is False

@pytest.mark.smoke
class TestDisplay:
    """verify display() forwards low_conf to overlay render"""

    def test_low_conf_true_forwarded(self):
        ov = MagicMock()
        ov.render.return_value = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        display(frame, make_preprocess_result(), None, make_decision(), ov, MagicMock(), False, low_conf=True)
        assert ov.render.call_args.kwargs["low_conf"] is True

    def test_low_conf_false_forwarded(self):
        ov = MagicMock()
        ov.render.return_value = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        display(frame, make_preprocess_result(), None, make_decision(), ov, MagicMock(), False, low_conf=False)
        assert ov.render.call_args.kwargs["low_conf"] is False

    def test_low_conf_defaults_to_false(self):
        ov = MagicMock()
        ov.render.return_value = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        display(frame, make_preprocess_result(), None, None, ov, MagicMock(), False)
        assert ov.render.call_args.kwargs.get("low_conf", False) is False

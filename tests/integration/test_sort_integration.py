# tests/integration/test_sort_integration.py

import time
from unittest.mock import MagicMock

import pytest

from config.constants import ColourID
from sort import (
    PipelineState,
    _accumulate,
    _handle_gt,
    _submit_plots,
    acquire,
    classify,
    extract,
    record,
)
from utils.ground_truth import GTSession
from utils.inject import build_inject_cycle
from tests.helpers.vision_helpers import make_decision, make_features, make_preprocess_result
from tests.helpers.inject_helpers import make_inject_folder

@pytest.mark.smoke
@pytest.mark.regression
class TestAcquirePreprocessChain:
    """verify acquire and preprocess produce a found result from injected M&M frames"""

    def test_inject_frame_found(self, tmp_path, prep):
        from sort import preprocess
        folder = make_inject_folder(tmp_path, n_images=1)
        frames_iter = build_inject_cycle(str(folder), repeat=1)
        frame = acquire(frames_iter, None)
        result = preprocess(frame, prep)
        assert result.found is True

    def test_gap_frame_not_found(self, tmp_path, prep):
        from sort import preprocess
        folder = make_inject_folder(tmp_path, n_images=1)
        frames_iter = build_inject_cycle(str(folder), repeat=1)
        acquire(frames_iter, None)
        gap = acquire(frames_iter, None)
        result = preprocess(gap, prep)
        assert result.found is False

@pytest.mark.smoke
class TestExtractClassifyChain:
    """verify extract and classify produce a valid decision from a synthetic result"""

    def test_synthetic_result_classifies(self, extractor, classifier):
        result = make_preprocess_result(hue=10, sat=200, val=180)
        state = PipelineState()
        features = extract(result, extractor, state)
        decision = classify(features, classifier, state)
        assert decision is not None
        assert isinstance(decision.label, ColourID)

    def test_decision_confidence_in_range(self, extractor, classifier):
        result = make_preprocess_result(hue=10, sat=200, val=180)
        state = PipelineState()
        features = extract(result, extractor, state)
        decision = classify(features, classifier, state)
        assert 0.0 <= decision.confidence <= 1.0

@pytest.mark.smoke
class TestAccumulateGTChain:
    """verify _accumulate and _handle_gt update state lists and gt_acc together"""

    def test_accumulate_and_handle_gt_consistent(self):
        state = PipelineState(object_id=1, gt_labels={1: 2})
        decision = make_decision(label=ColourID.RED, confidence=0.9)
        _accumulate(state, decision)
        _handle_gt(state, decision)
        assert len(state.pred_list) == 1
        assert len(state.gt_acc) == 1
        assert state.gt_acc.pairs[0] == (2, int(ColourID.RED))

    def test_multiple_objects_accumulated(self):
        state = PipelineState()
        for i in range(5):
            state.object_id = i + 1
            _accumulate(state, make_decision())
        assert len(state.pred_list) == 5
        assert len(state.conf_list) == 5

@pytest.mark.smoke
class TestSubmitPlotsChain:
    """verify accumulated state feeds correctly into _submit_plots"""

    def test_predictions_without_gt_enqueues_plain_plot(self):
        mock_worker = MagicMock()
        state = PipelineState(plot_worker=mock_worker)
        for _ in range(3):
            _accumulate(state, make_decision())
        _submit_plots(state)
        mock_worker.enqueue.assert_called_once()

    def test_predictions_with_gt_enqueues_gt_plot(self):
        mock_worker = MagicMock()
        gt_acc = GTSession()
        gt_acc.record(1, 2, 0.9)
        state = PipelineState(
            plot_worker=mock_worker,
            pred_list=[2],
            conf_list=[0.9],
            gt_labels={1: 1},
            gt_acc=gt_acc,
        )
        _submit_plots(state)
        enqueued = mock_worker.enqueue.call_args[0][0]
        assert enqueued.ground_truth is not None

@pytest.mark.smoke
class TestRecordChain:
    """verify record drives event emission, uart send, GT accumulation, and plot submission"""

    def test_record_increments_classified_and_gt(self):
        gt_acc = GTSession()
        state = PipelineState(object_id=1, gt_labels={1: 1}, gt_acc=gt_acc)
        record(
            make_preprocess_result(), make_features(), make_decision(),
            time.monotonic(), MagicMock(), MagicMock(),
            max_objects=None, state=state,
        )
        assert state.classified_count == 1
        assert len(state.gt_acc) == 1

    def test_record_sends_uart(self):
        mock_uart = MagicMock()
        state = PipelineState(object_id=1)
        record(
            make_preprocess_result(), make_features(), make_decision(),
            time.monotonic(), MagicMock(), mock_uart,
            max_objects=None, state=state,
        )
        mock_uart.send.assert_called_once()

    def test_record_enqueues_event(self):
        mock_worker = MagicMock()
        state = PipelineState(object_id=1)
        record(
            make_preprocess_result(), make_features(), make_decision(),
            time.monotonic(), mock_worker, MagicMock(),
            max_objects=None, state=state,
        )
        mock_worker.enqueue.assert_called_once()

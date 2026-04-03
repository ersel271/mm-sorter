# tests/integration/test_events_integration.py

import json
from pathlib import Path

import pytest

from utils.events import EventQueueWorker
from utils.metrics import RunningMetrics
from tests.helpers.events_helpers import make_event

@pytest.mark.integration
class TestEventPipelineIntegration:
    """verify end-to-end event flow: enqueue -> background write -> JSONL on disk."""

    def test_worker_creates_event_dir(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        worker.stop()
        assert Path(tmp_cfg.system["event_dir"]).exists()

    def test_write_cycle_produces_valid_jsonl(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        for i in range(5):
            worker.enqueue(make_event(object_id=i))
        worker._queue.join()
        worker.stop()
        lines = worker.writer_path.read_text().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "object_id" in obj
            assert "ts_wall" in obj
            assert "ts_mono" in obj

    def test_shutdown_flushes_all_queued_events(self, tmp_cfg):
        # stop() must drain remaining queue items before closing the writer
        worker = EventQueueWorker(tmp_cfg)
        for i in range(20):
            worker.enqueue(make_event(object_id=i))
        worker.stop()
        lines = worker.writer_path.read_text().splitlines()
        assert len(lines) == 20

    def test_event_field_values_preserved_in_file(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        worker.enqueue(make_event(object_id=77, class_name="red", confidence=0.91))
        worker._queue.join()
        worker.stop()
        obj = json.loads(worker.writer_path.read_text().strip())
        assert obj["object_id"] == 77
        assert obj["class_name"] == "red"
        assert abs(obj["confidence"] - 0.91) < 1e-9

    def test_metrics_updated_by_worker_after_stop(self, tmp_cfg):
        m = RunningMetrics()
        worker = EventQueueWorker(tmp_cfg, metrics=m)
        worker.enqueue(make_event(low_confidence=False))
        worker.enqueue(make_event(low_confidence=False))
        worker.enqueue(make_event(low_confidence=True))
        worker.stop()
        assert m.total == 3
        assert m.low_confidence == 1

    def test_metrics_mean_confidence_reflects_events(self, tmp_cfg):
        m = RunningMetrics()
        worker = EventQueueWorker(tmp_cfg, metrics=m)
        worker.enqueue(make_event(confidence=0.80))
        worker.enqueue(make_event(confidence=0.60))
        worker.stop()
        assert abs(m.mean_confidence - 0.70) < 1e-9

    def test_each_written_line_is_independent_json(self, tmp_cfg):
        # each line must parse independently
        worker = EventQueueWorker(tmp_cfg)
        for i in range(3):
            worker.enqueue(make_event(object_id=i))
        worker._queue.join()
        worker.stop()
        for line in worker.writer_path.read_text().splitlines():
            obj = json.loads(line)
            assert isinstance(obj, dict)

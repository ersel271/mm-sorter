# tests/unit/test_events.py

import json
import queue as q
from dataclasses import fields, asdict
from pathlib import Path

import pytest

from utils.events import (
    VisionEvent,
    EventWriter,
    EventQueueWorker,
    check_reserved_fields,
    serialise_event,
)
from tests.helpers.events_helpers import make_event

@pytest.mark.smoke
class TestVisionEventFields:
    """verify event dataclass field names do not collide with logging reserved names"""

    def test_no_reserved_field_collision(self):
        # must fail if a future field name collides with a LogRecord attribute
        check_reserved_fields()

    def test_all_expected_fields_present(self):
        field_names = {f.name for f in fields(VisionEvent)}
        required = {
            "ts_wall", "ts_mono", "object_id", "class_id", "class_name",
            "confidence", "low_confidence", "centroid_x", "centroid_y", "area",
            "sat_mean", "highlight_ratio", "hue_peak_width", "texture_variance",
            "circularity", "aspect_ratio", "solidity", "frame_ms",
        }
        assert required.issubset(field_names)

    def test_check_reserved_fields_raises_on_collision(self, monkeypatch):
        monkeypatch.setattr("utils.events._LOGGING_RESERVED", frozenset({"ts_wall"}))
        with pytest.raises(ValueError, match="ts_wall"):
            check_reserved_fields()

@pytest.mark.smoke
class TestSerialiseEvent:
    """verify JSON serialisation correctness"""

    def test_produces_valid_json(self, sample_event):
        parsed = json.loads(serialise_event(sample_event))
        assert isinstance(parsed, dict)

    def test_no_trailing_newline(self, sample_event):
        assert not serialise_event(sample_event).endswith("\n")

    def test_fields_match_dataclass(self, sample_event):
        parsed = json.loads(serialise_event(sample_event))
        assert parsed["object_id"] == sample_event.object_id
        assert parsed["class_name"] == sample_event.class_name
        assert parsed["confidence"] == pytest.approx(sample_event.confidence)

    def test_asdict_round_trip(self, sample_event):
        reconstructed = VisionEvent(**asdict(sample_event))
        assert reconstructed == sample_event

@pytest.mark.smoke
class TestEventWriter:
    """verify JSONL file creation, append behaviour, and per-event flush"""

    def test_creates_event_dir(self, tmp_path):
        target = tmp_path / "events"
        writer = EventWriter(target)
        assert target.exists()
        writer.close()

    def test_creates_timestamped_jsonl_file(self, event_writer):
        name = event_writer.path.name
        assert name.startswith("vision_")
        assert name.endswith(".jsonl")
        # vision_YYYYMMDD_HHMMSS.jsonl = 7 + 15 + 6 = 28 chars
        assert len(name) == len("vision_YYYYMMDD_HHMMSS.jsonl")

    def test_writes_one_line_per_event(self, event_writer):
        event_writer.write(make_event(object_id=1))
        event_writer.write(make_event(object_id=2))
        event_writer.close()
        lines = event_writer.path.read_text().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, event_writer):
        event_writer.write(make_event())
        event_writer.close()
        for line in event_writer.path.read_text().splitlines():
            assert json.loads(line)

    def test_object_id_preserved_in_file(self, event_writer):
        event_writer.write(make_event(object_id=99))
        event_writer.close()
        parsed = json.loads(event_writer.path.read_text().strip())
        assert parsed["object_id"] == 99

    def test_write_after_close_does_not_raise(self, tmp_path):
        writer = EventWriter(tmp_path)
        writer.close()
        writer.write(make_event())  # must not propagate the OSError

    def test_path_property_is_path_object(self, event_writer):
        assert isinstance(event_writer.path, Path)

@pytest.mark.smoke
@pytest.mark.regression
class TestEventQueueWorker:
    """verify async enqueue, background write, drop-newest policy, and shutdown"""

    def _drain(self, worker: EventQueueWorker) -> None:
        worker._queue.join()

    def test_writes_event_to_file(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        worker.enqueue(make_event(object_id=42))
        self._drain(worker)
        worker.stop()
        lines = worker.writer_path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["object_id"] == 42

    def test_writes_multiple_events(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        for i in range(5):
            worker.enqueue(make_event(object_id=i))
        self._drain(worker)
        worker.stop()
        lines = worker.writer_path.read_text().splitlines()
        assert len(lines) == 5

    def test_drop_newest_when_queue_full(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        # stop the worker thread first so it cannot drain the queue
        worker._stop_event.set()
        worker._thread.join()
        full_q: q.Queue = q.Queue(maxsize=1)
        full_q.put_nowait(make_event(object_id=0))
        worker._queue = full_q
        worker.enqueue(make_event(object_id=1))
        worker._writer.close()
        assert worker.dropped == 1

    def test_stop_flushes_pending_events(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        for i in range(10):
            worker.enqueue(make_event(object_id=i))
        worker.stop()
        lines = worker.writer_path.read_text().splitlines()
        assert len(lines) == 10

    def test_updates_metrics_low_confidence(self, tmp_cfg, metrics):
        worker = EventQueueWorker(tmp_cfg, metrics=metrics)
        worker.enqueue(make_event(low_confidence=True))
        self._drain(worker)
        worker.stop()
        assert metrics.low_confidence == 1

    def test_updates_metrics_normal_confidence(self, tmp_cfg, metrics):
        worker = EventQueueWorker(tmp_cfg, metrics=metrics)
        worker.enqueue(make_event(low_confidence=False))
        self._drain(worker)
        worker.stop()
        assert metrics.low_confidence == 0

    def test_dropped_starts_at_zero(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        worker.stop()
        assert worker.dropped == 0

    def test_writer_path_is_path_object(self, tmp_cfg):
        worker = EventQueueWorker(tmp_cfg)
        worker.stop()
        assert isinstance(worker.writer_path, Path)

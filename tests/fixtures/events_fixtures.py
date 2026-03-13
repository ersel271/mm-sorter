# tests/fixtures/events_fixtures.py

import pytest

from utils.events import EventWriter, EventQueueWorker, VisionEvent
from utils.metrics import RunningMetrics
from tests.helpers.events_helpers import make_event

@pytest.fixture
def sample_event() -> VisionEvent:
    return make_event()

@pytest.fixture
def event_writer(tmp_path) -> EventWriter:
    writer = EventWriter(tmp_path / "events")
    yield writer
    writer.close()

@pytest.fixture
def metrics() -> RunningMetrics:
    return RunningMetrics()

@pytest.fixture
def event_worker(tmp_cfg) -> EventQueueWorker:
    worker = EventQueueWorker(tmp_cfg)
    yield worker
    # stop only if thread is still alive (test may have stopped it already)
    if worker._thread.is_alive():
        worker.stop()

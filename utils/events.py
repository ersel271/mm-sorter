# utils/events.py
"""
Structured classification event types, serialisation, and async writer.

Provides VisionEvent dataclass, EventWriter for JSONL output,
and EventQueueWorker for off-thread event processing.

Usage:
    worker = EventQueueWorker(cfg, metrics)
    worker.enqueue(event)
    worker.stop()
"""

import json
import logging
import queue
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# derive reserved attribute names directly from a real LogRecord instance
# so the set stays correct across Python versions without manual maintenance
reserved = set(logging.LogRecord(None, None, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime",}
_LOGGING_RESERVED = frozenset(reserved)

@dataclass
class VisionEvent:
    # wall-clock ISO-8601 timestamp for log correlation
    ts_wall: str
    # monotonic timestamp for reliable latency measurements
    ts_mono: float
    object_id: int
    class_id: int
    class_name: str
    confidence: float
    decision: str
    centroid_x: int
    centroid_y: int
    area: float
    sat_mean: float
    highlight_ratio: float
    hue_peak_width: float
    texture_variance: float
    circularity: float
    aspect_ratio: float
    solidity: float
    frame_ms: float

def check_reserved_fields() -> None:
    """raise ValueError if any VisionEvent field collides with logging reserved names"""
    from dataclasses import fields
    collisions = {f.name for f in fields(VisionEvent)} & _LOGGING_RESERVED
    if collisions:
        raise ValueError(f"VisionEvent fields collide with logging reserved names: {collisions}")

def serialise_event(event: VisionEvent) -> str:
    """serialise a VisionEvent to a compact JSON string without trailing newline"""
    return json.dumps(asdict(event), separators=(",", ":"))

class EventWriter:
    """
    appends VisionEvent records as JSON Lines to a timestamped .jsonl file.
    each event is flushed immediately to minimise data loss on crash.
    """

    def __init__(self, event_dir: str | Path):
        self._event_dir = Path(event_dir)
        self._event_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._event_dir / f"vision_{timestamp}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        log.info("event writer opened %s", self._path)

    def write(self, event: VisionEvent) -> None:
        """append one event as a JSON line and flush immediately"""
        try:
            self._file.write(serialise_event(event) + "\n")
            self._file.flush()
        except (OSError, ValueError) as exc:
            log.error("event write failed: %s", exc)

    def close(self) -> None:
        """close the underlying file"""
        try:
            self._file.close()
        except OSError as exc:
            log.warning("event writer close error: %s", exc)

    @property
    def path(self) -> Path:
        return self._path

class EventQueueWorker:
    """
    background worker that drains an event queue, writes events to a JSONL
    file, and optionally updates a RunningMetrics instance.

    the real-time path enqueues VisionEvent objects without blocking.
    if the queue is full the newest event is dropped (drop-newest policy).
    stop() is safe to call even when the queue is full.
    """

    def __init__(self, config, metrics=None):
        self._cfg = config.system
        self._writer = EventWriter(self._cfg["event_dir"])
        self._queue: queue.Queue = queue.Queue(maxsize=self._cfg["log_queue_size"])
        self._metrics = metrics
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="event-worker")
        self._thread.start()

    def enqueue(self, event: VisionEvent) -> None:
        """put event on queue without blocking; drops event if queue is full"""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1
            log.warning("event queue full, dropped event (total drops: %d)", self._dropped)

    def stop(self, timeout: float = 5.0) -> None:
        """signal worker to stop, drain remaining events, and close the writer"""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._writer.close()

    @property
    def dropped(self) -> int:
        with self._dropped_lock:
            return self._dropped

    @property
    def writer_path(self) -> Path:
        return self._writer.path

    def _run(self) -> None:
        # continue while stop has not been requested or items still remain
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                self._writer.write(item)
                if self._metrics is not None:
                    self._metrics.update(item)
                self._queue.task_done()
            except queue.Empty:
                continue

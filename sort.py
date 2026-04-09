#!/usr/bin/env python3
# sort.py
"""
Pipeline entry point for the M&M sorter pipeline.
Stages: (1)acquire, (2)preprocess, (3)features, (4)classify, (5)record, (6)display.

Options:
  --plot                save analysis plots to plot_dir after each object
  --ground-truth PATH   plaintext file with ground-truth labels, one label per line.
                        enables GT plots when used with --plot
  --inject-from PATH    inject frames from a folder instead of the camera. cycles 
                        infinitely on its own
  --max-objects N       stop cleanly after N classified objects
  --timeout SECS        hard wall-clock exit after SECS seconds
  --config PATH         path to config YAML (default: config/config.yaml)
  --log-level LEVEL     DEBUG, INFO, WARNING (default: INFO)

Controls:
    d       toggle debug mode
    f       toggle freeze mode
    t       toggle sidebar menus (features, decision breakdown and stats)
    l       toggle log menu
    q       quit the application

Run Modes:
    Live mode uses the camera as the frame source for real-time operation
    Injection mode (--inject-from) replaces the camera with a deterministic image cycle
    Evaluation mode (--ground-truth [+ --plot]) enables labelled performance tracking
    Execution can be bounded via --max-objects or --timeout for controlled runs

Output:
  Events  cfg.system["event_dir"]   one event per line jsonl file
  Plots   cfg.system["plot_dir"]    png files, updated after each object
  Logs    cfg.system["log_dir"]     timestamped log file
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from enum import StrEnum
from typing import Any, cast
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from config import Config
from config.constants import ColourID, COLOUR_NAMES, OBJECT_ID_MAX

from src.ui import Overlay, handle_key
from src.io import Camera, UARTSender, PCK_START, PCK_END_OK, PCK_END_ERR, PCK_FREEZE_START, PCK_FREEZE_END
from src.vision import Classifier, Decision, FeatureExtractor, Features, Preprocessor, PreprocessResult

from utils.log import setup_logger
from utils.metrics import RunningMetrics
from utils.inject import build_inject_cycle
from utils.plot import PlotData, PlotQueueWorker
from utils.ground_truth import GTSession, load_gt
from utils.events import EventQueueWorker, make_event

log = logging.getLogger(__name__)

# constants

class StopReason(StrEnum):
    USER_QUIT = "user_quit"
    KEYBOARD_INTERRUPT = "keyboard_interrupt"
    TIMEOUT = "timeout"
    MAX_OBJECTS = "max_objects"

# pipeline session state

@dataclass
class PipelineState:
    object_id: int = 0
    frame_num: int = 0
    low_conf: bool = False
    found_count: int = 0
    classified_count: int = 0
    stop_reason: StopReason = StopReason.USER_QUIT
    timeout_at: float = float("inf")
    pred_list: list[int] = field(default_factory=list)
    conf_list: list[float] = field(default_factory=list)
    gt_labels: dict[int, int] = field(default_factory=dict)
    gt_acc: GTSession = field(default_factory=GTSession)
    plot_worker: PlotQueueWorker | None = None

# cli

def parse_args() -> argparse.Namespace:
    """parse and return the CLI arguments"""
    p = argparse.ArgumentParser(description="M&M sorter pipeline")
    p.add_argument("--plot", action="store_true", help="save analysis plots to the configured plot_dir after each object")
    p.add_argument("--ground-truth", metavar="PATH", help="plaintext file with ground-truth labels, one label per line. enables GT plots when used with --plot")
    p.add_argument("--inject-from", metavar="PATH", help="inject frames from a folder instead of the camera. cycles infinitely on its own")
    p.add_argument("--max-objects", metavar="N", type=int, help="stop cleanly after N classified objects")
    p.add_argument("--timeout", metavar="SECS", type=float, help="hard wall-clock exit after N seconds")
    p.add_argument("--config", metavar="PATH", help="path to config YAML (default: config/config.yaml)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="logging verbosity (default: INFO)")
    return p.parse_args()

# helpers

def _setup_frame_source(args: argparse.Namespace, cfg: Config, found_frames_min: int) -> tuple[Iterator | None, Camera | None]:
    """open inject cycle or camera; raises RuntimeError on failure"""
    if args.inject_from:
        return build_inject_cycle(args.inject_from, found_frames_min), None
    cam = Camera(cfg)
    if not cam.open():
        raise RuntimeError("camera failed to open")
    return None, cam

def _setup_gt(args: argparse.Namespace) -> tuple[dict[int, int], GTSession]:
    """load GT file if provided; logs and continues on error"""
    gt_labels: dict[int, int] = {}
    if args.ground_truth:
        try:
            gt_labels = load_gt(args.ground_truth)
        except (OSError, ValueError) as exc:
            log.error("could not load ground truth file: %s", exc)
    return gt_labels, GTSession()

def _setup_plot_worker(args: argparse.Namespace, cfg: Config) -> PlotQueueWorker | None:
    """create the background plot worker if --plot is set"""
    if not args.plot:
        return None
    plot_dir = Path(cfg.system["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    return PlotQueueWorker(plot_dir)

def _make_uart_payload(state: PipelineState, result: PreprocessResult, decision: Decision, effective_class: int) -> dict[str, Any]:
    """build the UART payload dict for the current object"""
    cx, cy = result.centroid if result.centroid else (0, 0)
    return {
        "id":    state.object_id,
        "class": effective_class,
        "conf":  decision.confidence,
        "x":     cx,
        "y":     cy,
    }

def _accumulate(state: PipelineState, decision: Decision) -> None:
    """append one classified object into the running plot accumulators"""
    state.pred_list.append(int(decision.label))
    state.conf_list.append(decision.confidence)

def _handle_gt(state: PipelineState, decision: Decision) -> None:
    """record ground-truth pair if a label exists for the current object"""
    if not state.gt_labels:
        return
    true_cls = state.gt_labels.get(state.object_id)
    if true_cls is not None:
        state.gt_acc.record(true_cls, int(decision.label), decision.confidence)

def _submit_plots(state: PipelineState) -> None:
    """build a PlotData snapshot and hand it to the background plot worker"""
    if state.plot_worker is None or not state.pred_list:
        return
    class_names = list(COLOUR_NAMES.values())
    if state.gt_labels and len(state.gt_acc) > 0:
        plot_data = state.gt_acc.to_plot_data(class_names)
    else:
        plot_data = PlotData(
            predictions=list(state.pred_list),
            confidences=list(state.conf_list),
            class_names=class_names,
        )
    state.plot_worker.enqueue(plot_data)

# pipeline stages

def acquire(frames_iter: Iterator | None, cam: Camera | None) -> np.ndarray | None:
    """step 1: read one frame from camera or inject source (None to skip)"""
    if frames_iter is not None:
        return cast(np.ndarray, next(frames_iter))
    if cam is None:
        return None
    ok, raw = cam.read()
    if not ok or raw is None:
        return None
    return raw

def preprocess(frame: np.ndarray, prep: Preprocessor) -> PreprocessResult:
    """step 2: run vision preprocessing on a raw frame"""
    return prep.process(frame)

def extract(result: PreprocessResult, extractor: FeatureExtractor, state: PipelineState) -> Features | None:
    """step 3: compute geometric and colour features (None on error)"""
    try:
        return extractor.extract(result)
    except ValueError as exc:
        log.warning("feature extraction error on frame %d: %s", state.frame_num, exc)
        return None

def classify(features: Features | None, classifier: Classifier, state: PipelineState) -> Decision | None:
    """step 4: apply classifier rules to features (None if no features or error)"""
    if features is None:
        return None
    try:
        return classifier.classify(features)
    except ValueError as exc:
        log.warning("classification error on frame %d: %s", state.frame_num, exc)
        return None

def _resolve_class(decision: Decision, object_id: int, low_conf: bool) -> int:
    """return effective class; logs a warning when low_conf is True"""
    if low_conf:
        log.warning(
            "low confidence %.2f for object %d (%s), overriding to Non-M&M",
            decision.confidence, object_id, COLOUR_NAMES[decision.label],
        )
        return int(ColourID.NON_MM)
    return int(decision.label)

def record(
    result: PreprocessResult,
    features: Features,
    decision: Decision,
    t0: float,
    worker: EventQueueWorker,
    uart: UARTSender,
    max_objects: int | None,
    state: PipelineState,
) -> bool:
    """step 5: emit event, send UART, accumulate plots and GT (True if limit reached)"""
    effective_class = _resolve_class(decision, state.object_id, state.low_conf)
    frame_ms = (time.monotonic() - t0) * 1000
    worker.enqueue(make_event(state.object_id, result, features, decision, frame_ms, t0, effective_class, state.low_conf))
    uart.send(_make_uart_payload(state, result, decision, effective_class))
    state.classified_count += 1
    _accumulate(state, decision)
    _handle_gt(state, decision)
    _submit_plots(state)
    return bool(max_objects and state.classified_count >= max_objects)

def display(
    frame: np.ndarray,
    result: PreprocessResult,
    features: Features | None,
    decision: Decision | None,
    ov: Overlay,
    uart: UARTSender,
    is_record_frame: bool,
    low_conf: bool = False,
) -> bool:
    """step 6: render overlay and show frame (True if user requested quit)"""
    frame_out = ov.render(
        frame, result, features, decision,
        uart_sent=uart.packets_sent,
        uart_dropped=uart.packets_dropped,
        uart_connected=uart.is_open,
        record=is_record_frame,
        low_conf=low_conf,
    )
    if frame_out is not None:
        return handle_key(ov.show(frame_out), ov)
    return False

def teardown(
    plot_worker: PlotQueueWorker | None,
    cam: Camera | None,
    uart: UARTSender,
    worker: EventQueueWorker,
    error: bool = False
) -> None:
    """stop background workers and close hardware resources"""
    if plot_worker is not None:
        plot_worker.stop()
    if cam is not None:
        cam.release()
    uart.send(PCK_END_ERR if error else PCK_END_OK)
    uart.close()
    worker.stop()

# orchestration

def pipeline() -> int:
    args = parse_args()

    # pipeline setup
    cfg = Config(args.config)
    threshold: float = cfg.thresholds["decision_min"]
    setup_logger(cfg, level=getattr(logging, args.log_level))
    log.info("starting M&M sorter -- log_level=%s, decision_min=%.2f", args.log_level, threshold)

    metrics = RunningMetrics()
    worker = EventQueueWorker(cfg, metrics)
    prep = Preprocessor(cfg)
    extractor = FeatureExtractor(cfg)
    classifier = Classifier(cfg)
    uart = UARTSender(cfg)
    uart.open()
    uart.send(PCK_START)

    found_frames_min: int = cfg.system.get("found_frames_min", 3)

    cam = None
    try:
        frames_iter, cam = _setup_frame_source(args, cfg, found_frames_min)
    except RuntimeError as exc:
        log.error("%s", exc)
        teardown(None, None, uart, worker, True)
        return 1

    gt_labels, gt_acc = _setup_gt(args)
    plot_worker = _setup_plot_worker(args, cfg)

    state = PipelineState(
        gt_labels=gt_labels,
        gt_acc=gt_acc,
        plot_worker=plot_worker,
        timeout_at=time.monotonic() + args.timeout if args.timeout else float("inf"),
    )

    log.info("pipeline running -- q: quit")

    # main pipeline loop
    try:
        last_frame: np.ndarray | None = None
        last_result: PreprocessResult | None = None

        with Overlay(cfg, metrics) as ov:
            while True:
                if time.monotonic() >= state.timeout_at:
                    state.stop_reason = StopReason.TIMEOUT
                    break

                t0 = time.monotonic()

                if ov.frozen:
                    if last_frame is not None and last_result is not None:
                        state.frame_num += 1
                        if display(last_frame, last_result, None, None, ov, uart, False):
                            break
                        if not ov.frozen:
                            uart.send(PCK_FREEZE_END)
                    continue

                frame = acquire(frames_iter, cam)
                if frame is None:
                    continue

                result = preprocess(frame, prep)
                last_frame, last_result = frame, result
                is_record_frame = False
                features = decision = None
                state.low_conf = False

                if not result.found:
                    state.found_count = 0
                else:
                    state.found_count += 1
                    is_record_frame = state.found_count == found_frames_min
                    if state.found_count == 1:
                        state.object_id = (state.object_id % OBJECT_ID_MAX) + 1
                    # extract/classify run on every found frame (not just the record frame)
                    # so the overlay can display live features and decision while the object is present
                    features = extract(result, extractor, state)
                    decision = classify(features, classifier, state)
                    state.low_conf = decision.confidence < threshold if decision is not None else False

                    if is_record_frame and features is not None and decision is not None:
                        if record(result, features, decision, t0, worker, uart, args.max_objects, state):
                            state.stop_reason = StopReason.MAX_OBJECTS
                            break

                state.frame_num += 1
                if display(frame, result, features, decision, ov, uart, is_record_frame, low_conf=state.low_conf):
                    break
                if ov.frozen:
                    uart.send(PCK_FREEZE_START)

    # teardown
    except KeyboardInterrupt:
        state.stop_reason = StopReason.KEYBOARD_INTERRUPT
        log.info("interrupted")
    finally:
        teardown(state.plot_worker, cam, uart, worker)

    snap = metrics.snapshot()
    log.info(
        "stopped -- %d frames processed, stop_reason=%s, objects=%d, "
        "low_confidence=%d, mean_confidence=%.3f",
        state.frame_num, state.stop_reason,
        snap["total"], snap["low_confidence"], snap["mean_confidence"],
    )
    return 0

if __name__ == "__main__":
    sys.exit(pipeline())

# utils/plot.py
"""
Plot generation and async saving for M&M sorting session data.

Provides PlotData for passing session data, generate_dashboard for
saving individual plot PNGs to disk, and PlotQueueWorker for
off-thread rendering so the main loop is not blocked.

Usage:
    worker = PlotQueueWorker(cfg.system["plot_dir"])
    worker.enqueue(plot_data)
    worker.stop()
"""

import queue
import logging
import threading
from pathlib import Path
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.ui.panel import LABEL_COLOURS
from utils.metrics import accuracy, confusion_matrix, normalise_confusion_matrix, per_class_metrics

# dark theme for global matplotlib usage
matplotlib.rcParams.update({
    "figure.facecolor": "#1a1a1a",
    "axes.facecolor":   "#222222",
    "axes.edgecolor":   "#444444",
    "grid.color":       "#333333",
    "text.color":       "#cccccc",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
})

# RGB (0-1) per class index, derived from panel.py LABEL_COLOURS (BGR 0-255)
_RGB_BY_ID: list[tuple[float, float, float]] = [
    (LABEL_COLOURS[cid][2] / 255.0,
     LABEL_COLOURS[cid][1] / 255.0,
     LABEL_COLOURS[cid][0] / 255.0)
    for cid in sorted(LABEL_COLOURS, key=lambda c: c.value)
]

# global colour palette
_IVORY = "#FFF5E4"
_BLUSH = "#FFC4C4"
_ROSE  = "#EE6983"
_WINE  = "#850E35"

_CMAP = LinearSegmentedColormap.from_list("rose_depth", ["#1a0a10", _ROSE])

log = logging.getLogger(__name__)

def _colour(class_idx: int) -> tuple[float, float, float]:
    """return RGB (0-1) for a class index"""
    if 0 <= class_idx < len(_RGB_BY_ID):
        return _RGB_BY_ID[class_idx]
    return (0.5, 0.5, 0.5)

@dataclass(frozen=True)
class PlotData:
    """
    snapshot of session statistics passed to the background plot worker.
    produced once per classified object and never modified after construction.
    ground_truth is None in sessions run without a GT file
    """
    predictions: list[int]
    confidences: list[float]
    class_names: list[str]
    ground_truth: list[int] | None = None

# no-GT plots

def _plot_class_distribution(ax: plt.Axes, data: PlotData) -> None:
    counts = Counter(data.predictions)
    values = [counts.get(i, 0) for i in range(len(data.class_names))]
    colors = [_colour(i) for i in range(len(data.class_names))]
    bars = ax.bar(data.class_names, values, color=colors)
    ax.bar_label(bars, fmt="%d", padding=2, color="#aaaaaa", fontsize=8)
    ax.set_title("class distribution")
    ax.yaxis.grid(True, alpha=0.2, zorder=0)
    ax.tick_params(axis="x", rotation=45)

def _plot_confidence_hist(ax: plt.Axes, data: PlotData) -> None:
    ax.hist(data.confidences, bins=20, color=_ROSE, edgecolor="#333333")
    mean_conf = float(np.mean(data.confidences))
    ax.axvline(mean_conf, color="#ffffff", linestyle="--", linewidth=1,
               label=f"mean {mean_conf:.2f}")
    ax.legend(fontsize=8)
    ax.set_title("confidence distribution")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)

def _plot_confidence_by_class(ax: plt.Axes, data: PlotData) -> None:
    groups: dict[int, list[float]] = {}
    for p, c in zip(data.predictions, data.confidences, strict=False):
        groups.setdefault(p, []).append(c)
    present = [i for i in range(len(data.class_names)) if i in groups]
    if not present:
        ax.set_title("confidence by predicted class")
        return
    labels   = [data.class_names[i] for i in present]
    box_data = [groups[i] for i in present]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    medianprops={"color": "#cccccc"},
                    whiskerprops={"color": "#aaaaaa"},
                    capprops={"color": "#aaaaaa"},
                    flierprops={"markeredgecolor": "#aaaaaa"})
    for patch, idx in zip(bp["boxes"], present, strict=False):
        patch.set_facecolor(_colour(idx))
    ax.set_title("confidence by predicted class")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

# GT plots

def _plot_confusion_matrix(ax: plt.Axes, data: PlotData) -> None:
    assert data.ground_truth is not None
    pairs = list(zip(data.ground_truth, data.predictions, strict=False))
    n    = len(data.class_names)
    mat  = confusion_matrix(pairs, n)
    norm = normalise_confusion_matrix(mat)
    ax.imshow(norm, cmap=_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(data.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(data.class_names, fontsize=8)
    for i, tick in enumerate(ax.get_xticklabels()):
        if i > 0:
            tick.set_color(_colour(i))
    for i, tick in enumerate(ax.get_yticklabels()):
        if i > 0:
            tick.set_color(_colour(i))
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("confusion matrix (row-normalised)")
    for i in range(n):
        for j in range(n):
            txt_color = _IVORY if norm[i][j] > 0.5 else "#555555"
            ax.text(j, i, f"{norm[i][j]:.2f}\n({mat[i][j]})", ha="center", va="center",
                    color=txt_color, fontsize=6)

def _plot_per_class_metrics(ax: plt.Axes, data: PlotData) -> None:
    assert data.ground_truth is not None
    pairs = list(zip(data.ground_truth, data.predictions, strict=False))
    n   = len(data.class_names)
    mat = confusion_matrix(pairs, n)
    pc  = per_class_metrics(mat)
    acc = accuracy(mat)
    x     = np.arange(n)
    width = 0.25
    precisions = [m["precision"] for m in pc]
    recalls    = [m["recall"]    for m in pc]
    f1s        = [m["f1"]        for m in pc]
    ax.bar(x - width, precisions, width, label="Precision", color=_BLUSH)
    ax.bar(x,         recalls,    width, label="Recall",    color=_ROSE)
    ax.bar(x + width, f1s,        width, label="F1",        color=_WINE)
    ax.set_xticks(x)
    ax.set_xticklabels(data.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"per-class metrics  (accuracy {acc:.1%})")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, alpha=0.2, zorder=0)
    ax.legend(fontsize=8)

def _plot_error_breakdown(ax: plt.Axes, data: PlotData) -> None:
    assert data.ground_truth is not None
    errors = [
        (a, p)
        for a, p in zip(data.ground_truth, data.predictions, strict=False)
        if a != p
    ]
    if not errors:
        ax.set_title("error breakdown")
        ax.text(0.5, 0.5, "no errors", ha="center", va="center",
                transform=ax.transAxes)
        return
    counts  = Counter(errors)
    records = sorted(counts.items(), key=lambda x: -x[1])
    labels  = [f"{data.class_names[a]} -> {data.class_names[p]}" for (a, p), _ in records]
    values  = [c for _, c in records]
    bars = ax.bar(labels, values, color=_ROSE)
    ax.bar_label(bars, fmt="%d", padding=2, color="#aaaaaa", fontsize=8)
    ax.set_title("error breakdown")
    ax.yaxis.grid(True, alpha=0.2, zorder=0)
    ax.tick_params(axis="x", rotation=45)

def _plot_confidence_vs_accuracy(ax: plt.Axes, data: PlotData) -> None:
    assert data.ground_truth is not None
    bins = np.linspace(0, 1, 11)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(10):
        lo, hi  = bins[i], bins[i + 1]
        indices = [j for j, c in enumerate(data.confidences) if lo <= c < hi]
        if indices:
            correct = sum(
                1 for j in indices
                if data.ground_truth[j] == data.predictions[j]
            )
            xs.append((lo + hi) / 2)
            ys.append(correct / len(indices))
    if not xs:
        ax.set_title("confidence vs accuracy (no data)")
        return
    ax.fill_between([0, 1], [0, 1], 0, alpha=0.06, color="#ffffff", zorder=0)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#ffffff", linewidth=0.8, zorder=0)
    ax.plot(xs, ys, color=_ROSE, marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("confidence vs accuracy  (calibration curve)")

# rendering

def _build_fig(fn: Callable[[plt.Axes, PlotData], None], data: PlotData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a1a")
    fn(ax, data)
    fig.tight_layout()
    return fig

def generate_dashboard(data: PlotData, output_dir: Path, ts: str) -> None:
    """save plot PNGs under output_dir, named plot_{ts}_{chart}.png — overwrites on each call"""
    plots = [
        ("class_distribution",  _plot_class_distribution),
        ("confidence_hist",     _plot_confidence_hist),
        ("confidence_by_class", _plot_confidence_by_class),
    ]
    if data.ground_truth is not None:
        plots += [
            ("confusion_matrix",       _plot_confusion_matrix),
            ("per_class_metrics",      _plot_per_class_metrics),
            ("error_breakdown",        _plot_error_breakdown),
            ("confidence_vs_accuracy", _plot_confidence_vs_accuracy),
        ]

    for name, fn in plots:
        fig = _build_fig(fn, data)
        fig.savefig(output_dir / f"plot_{ts}_{name}.png", dpi=120)
        plt.close(fig)

class PlotQueueWorker:
    """
    background worker that drains a plot queue and saves dashboards
    to disk as PNG files.

    the real-time path submits PlotData objects without blocking.
    if the queue is full the newest submission is dropped (drop-newest policy).
    stop() drains remaining items before returning.
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._queue: queue.Queue[PlotData] = queue.Queue(maxsize=1)
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="plot-worker")
        self._thread.start()

    def enqueue(self, data: PlotData) -> None:
        """put plot data on queue without blocking; drops data if queue is full"""
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1
            log.warning("plot queue full, dropped plot data (total drops: %d)", self._dropped)

    def stop(self, timeout: float = 10.0) -> None:
        """signal worker to stop, drain remaining plots, and return"""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    @property
    def dropped(self) -> int:
        with self._dropped_lock:
            return self._dropped

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.5)
                generate_dashboard(data, self._output_dir, self._ts)
                self._queue.task_done()
            except queue.Empty:
                continue

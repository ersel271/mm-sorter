# utils/ground_truth.py
"""
Ground-truth loading and in-session accumulation for labelled evaluation runs.

Usage:
    labels = load_gt("data/gt.txt")
    acc = GTSession()
    acc.record(true_cls, pred_cls, confidence)
    plot_data = acc.to_plot_data(class_names)
"""

import logging

from utils.plot import PlotData
from config.constants import COLOUR_IDS

log = logging.getLogger(__name__)

def load_gt(path: str) -> dict[int, int]:
    """
    load a plaintext GT file and return {object_id: true_class_id}.

    each non-blank, non-comment line maps to an object in arrival order
    (1-indexed). accepted tokens: integer class ID (0-6) or colour name.
    """
    gt: dict[int, int] = {}
    object_id = 0
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            object_id += 1
            try:
                cls_id = int(line)
            except ValueError:
                key = line.lower()
                if key not in COLOUR_IDS:
                    raise ValueError(f"unknown ground-truth token on object {object_id}: {line!r}") from None
                cls_id = COLOUR_IDS[key]
            gt[object_id] = cls_id
    preview = list(gt.items())[:5]
    preview_str = ", ".join(f"{oid}:{cid}" for oid, cid in preview)
    suffix = ", ..." if len(gt) > 5 else ""
    log.info("ground truth loaded: %d labels from %s [%s%s]", len(gt), path, preview_str, suffix)
    return gt

class GTSession:
    """accumulates per-object ground-truth data for post-session plotting"""

    def __init__(self) -> None:
        self.pairs: list[tuple[int, int]] = []
        self.confidences: list[float] = []

    def record(self, true_cls: int, pred_cls: int, confidence: float) -> None:
        self.pairs.append((true_cls, pred_cls))
        self.confidences.append(confidence)

    def to_plot_data(self, class_names: list[str]) -> PlotData:
        return PlotData(
            predictions=[p for _, p in self.pairs],
            confidences=list(self.confidences),
            class_names=class_names,
            ground_truth=[t for t, _ in self.pairs],
        )

    def __len__(self) -> int:
        return len(self.pairs)

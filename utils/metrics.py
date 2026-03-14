# utils/metrics.py
"""
Running classification metrics tracker and offline evaluation utilities.

RunningMetrics accumulates live statistics updated by the background event
worker. Offline helpers derive confusion matrix, precision, recall, and F1
from labelled evaluation pairs.
"""

import logging
from collections import defaultdict

log = logging.getLogger(__name__)

class RunningMetrics:
    """
    accumulates live statistics from VisionEvent objects.
    all state is in-memory; no disk I/O occurs here.
    """

    def __init__(self) -> None:
        self._total = 0
        self._accepted = 0
        self._rejected = 0
        self._per_class: dict[int, int] = defaultdict(int)
        self._confidence_sum = 0.0
        self._frame_ms_sum = 0.0

    def update(self, event) -> None:
        """update counters from a single VisionEvent"""
        self._total += 1
        if event.decision == "ACCEPT":
            self._accepted += 1
        else:
            self._rejected += 1
        self._per_class[event.class_id] += 1
        self._confidence_sum += event.confidence
        self._frame_ms_sum += event.frame_ms

    @property
    def total(self) -> int:
        return self._total

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def rejected(self) -> int:
        return self._rejected

    def class_count(self, class_id: int) -> int:
        """return total predictions for a given class_id"""
        return self._per_class[class_id]

    @property
    def mean_confidence(self) -> float:
        if self._total == 0:
            return 0.0
        return self._confidence_sum / self._total

    @property
    def mean_frame_ms(self) -> float:
        if self._total == 0:
            return 0.0
        return self._frame_ms_sum / self._total

    def snapshot(self) -> dict:
        """return a plain dict snapshot of all current metrics"""
        return {
            "total": self._total,
            "accepted": self._accepted,
            "rejected": self._rejected,
            "mean_confidence": self.mean_confidence,
            "mean_frame_ms": self.mean_frame_ms,
            "per_class": dict(self._per_class),
        }

def confusion_matrix(
    pairs: list[tuple[int, int]], num_classes: int
) -> list[list[int]]:
    """
    compute a confusion matrix from (ground_truth, predicted) pairs.
    returns a num_classes x num_classes matrix where matrix[actual][predicted]
    holds the count; pairs with out-of-range indices are silently skipped.
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for actual, predicted in pairs:
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual][predicted] += 1
    return matrix

def per_class_metrics(matrix: list[list[int]]) -> list[dict]:
    """
    compute per-class precision, recall, and F1 from a confusion matrix.
    returns one dict per class with keys: class_id, precision, recall, f1.
    """
    n = len(matrix)
    results = []
    for c in range(n):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n) if r != c)
        fn = sum(matrix[c][r] for r in range(n) if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results.append(
            {"class_id": c, "precision": precision, "recall": recall, "f1": f1}
        )
    return results

def accuracy(matrix: list[list[int]]) -> float:
    """compute overall accuracy from a confusion matrix"""
    total = sum(matrix[r][c] for r in range(len(matrix)) for c in range(len(matrix[r])))
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return correct / total if total > 0 else 0.0

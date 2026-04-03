# tests/unit/test_metrics.py

import pytest

from tests.helpers.events_helpers import make_event
from utils.metrics import confusion_matrix, per_class_metrics, accuracy, normalise_confusion_matrix

@pytest.mark.unit
class TestRunningMetricsCounters:
    """verify total, accepted, and rejected counters"""

    def test_initial_total_is_zero(self, metrics):
        assert metrics.total == 0

    def test_initial_accepted_is_zero(self, metrics):
        assert metrics.accepted == 0

    def test_initial_rejected_is_zero(self, metrics):
        assert metrics.rejected == 0

    def test_total_increments_on_update(self, metrics):
        metrics.update(make_event())
        assert metrics.total == 1

    def test_accepted_increments_on_accept(self, metrics):
        metrics.update(make_event(decision="ACCEPT"))
        assert metrics.accepted == 1
        assert metrics.rejected == 0

    def test_rejected_increments_on_reject(self, metrics):
        metrics.update(make_event(decision="REJECT"))
        assert metrics.rejected == 1
        assert metrics.accepted == 0

    def test_multiple_updates_accumulate(self, metrics):
        for _ in range(3):
            metrics.update(make_event(decision="ACCEPT"))
        metrics.update(make_event(decision="REJECT"))
        assert metrics.total == 4
        assert metrics.accepted == 3
        assert metrics.rejected == 1

@pytest.mark.unit
class TestRunningMetricsPerClass:
    """verify per-class counters"""

    def test_class_count_zero_for_unseen_class(self, metrics):
        assert metrics.class_count(99) == 0

    def test_class_count_increments(self, metrics):
        metrics.update(make_event(class_id=2))
        metrics.update(make_event(class_id=2))
        metrics.update(make_event(class_id=3))
        assert metrics.class_count(2) == 2
        assert metrics.class_count(3) == 1

@pytest.mark.unit
class TestRunningMetricsAverages:
    """verify mean_confidence and mean_frame_ms"""

    def test_mean_confidence_zero_when_empty(self, metrics):
        assert metrics.mean_confidence == pytest.approx(0.0)

    def test_mean_frame_ms_zero_when_empty(self, metrics):
        assert metrics.mean_frame_ms == pytest.approx(0.0)

    def test_mean_confidence_single_event(self, metrics):
        metrics.update(make_event(confidence=0.80))
        assert metrics.mean_confidence == pytest.approx(0.80)

    def test_mean_confidence_multiple_events(self, metrics):
        metrics.update(make_event(confidence=0.80))
        metrics.update(make_event(confidence=0.60))
        assert metrics.mean_confidence == pytest.approx(0.70)

    def test_mean_frame_ms_multiple_events(self, metrics):
        metrics.update(make_event(frame_ms=20.0))
        metrics.update(make_event(frame_ms=40.0))
        assert metrics.mean_frame_ms == pytest.approx(30.0)

@pytest.mark.unit
class TestRunningMetricsSnapshot:
    """verify snapshot() returns a consistent plain dict"""

    def test_snapshot_is_dict(self, metrics):
        assert isinstance(metrics.snapshot(), dict)

    def test_snapshot_contains_expected_keys(self, metrics):
        snap = metrics.snapshot()
        for key in ("total", "accepted", "rejected", "mean_confidence", "mean_frame_ms", "per_class"):
            assert key in snap

    def test_snapshot_values_match_properties(self, metrics):
        metrics.update(make_event(decision="ACCEPT", confidence=0.75, frame_ms=25.0))
        snap = metrics.snapshot()
        assert snap["total"] == metrics.total
        assert snap["accepted"] == metrics.accepted
        assert snap["mean_confidence"] == pytest.approx(metrics.mean_confidence)

@pytest.mark.unit
class TestConfusionMatrix:
    """verify confusion_matrix builds a correct num_classes x num_classes grid"""

    def test_empty_pairs_produces_zero_matrix(self):
        matrix = confusion_matrix([], num_classes=3)
        assert matrix == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def test_perfect_predictions(self):
        pairs = [(0, 0), (1, 1), (2, 2)]
        matrix = confusion_matrix(pairs, num_classes=3)
        assert matrix[0][0] == 1
        assert matrix[1][1] == 1
        assert matrix[2][2] == 1

    def test_off_diagonal_misprediction(self):
        pairs = [(0, 1)]
        matrix = confusion_matrix(pairs, num_classes=3)
        assert matrix[0][1] == 1
        assert matrix[0][0] == 0

    def test_out_of_range_index_skipped(self):
        pairs = [(5, 0)]
        matrix = confusion_matrix(pairs, num_classes=3)
        assert all(matrix[r][c] == 0 for r in range(3) for c in range(3))

    def test_matrix_dimensions(self):
        matrix = confusion_matrix([], num_classes=4)
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

@pytest.mark.unit
class TestPerClassMetrics:
    """verify per-class precision, recall, and F1 computations"""

    def test_perfect_classifier(self):
        pairs = [(0, 0), (0, 0), (1, 1), (1, 1)]
        matrix = confusion_matrix(pairs, num_classes=2)
        results = per_class_metrics(matrix)
        for r in results:
            assert r["precision"] == pytest.approx(1.0)
            assert r["recall"] == pytest.approx(1.0)
            assert r["f1"] == pytest.approx(1.0)

    def test_zero_tp_gives_zero_f1(self):
        # class 0 always mispredicted as class 1
        pairs = [(0, 1), (0, 1)]
        matrix = confusion_matrix(pairs, num_classes=2)
        results = per_class_metrics(matrix)
        assert results[0]["f1"] == pytest.approx(0.0)

    def test_result_contains_expected_keys(self):
        matrix = confusion_matrix([(0, 0)], num_classes=2)
        for entry in per_class_metrics(matrix):
            for key in ("class_id", "precision", "recall", "f1"):
                assert key in entry

    def test_class_id_matches_index(self):
        matrix = confusion_matrix([], num_classes=3)
        results = per_class_metrics(matrix)
        for i, r in enumerate(results):
            assert r["class_id"] == i

@pytest.mark.unit
class TestAccuracy:
    """verify overall accuracy computation"""

    def test_perfect_accuracy(self):
        pairs = [(0, 0), (1, 1), (2, 2)]
        matrix = confusion_matrix(pairs, num_classes=3)
        assert accuracy(matrix) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        # everything mispredicted
        pairs = [(0, 1), (1, 0)]
        matrix = confusion_matrix(pairs, num_classes=2)
        assert accuracy(matrix) == pytest.approx(0.0)

    def test_partial_accuracy(self):
        pairs = [(0, 0), (0, 1)]
        matrix = confusion_matrix(pairs, num_classes=2)
        assert accuracy(matrix) == pytest.approx(0.5)

    def test_empty_matrix_returns_zero(self):
        assert accuracy([[0, 0], [0, 0]]) == pytest.approx(0.0)

@pytest.mark.unit
class TestNormaliseConfusionMatrix:
    """verify row-normalisation of confusion matrices"""

    def test_perfect_diagonal_normalises_to_ones(self):
        mat = [[3, 0], [0, 2]]
        result = normalise_confusion_matrix(mat)
        assert result[0] == pytest.approx([1.0, 0.0])
        assert result[1] == pytest.approx([0.0, 1.0])

    def test_row_sums_to_one(self):
        mat = [[2, 1], [1, 3]]
        result = normalise_confusion_matrix(mat)
        for row in result:
            assert sum(row) == pytest.approx(1.0)

    def test_zero_row_returns_zeros(self):
        mat = [[0, 0], [1, 1]]
        result = normalise_confusion_matrix(mat)
        assert result[0] == pytest.approx([0.0, 0.0])

    def test_output_shape_matches_input(self):
        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = normalise_confusion_matrix(mat)
        assert len(result) == 3
        assert all(len(row) == 3 for row in result)

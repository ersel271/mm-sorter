# tests/property/test_metrics_properties.py

import pytest
from hypothesis import given, strategies as st

from utils.metrics import RunningMetrics, confusion_matrix, per_class_metrics, accuracy
from tests.helpers.events_helpers import make_event

num_classes = st.integers(min_value=1, max_value=10)
valid_index = st.integers(min_value=0, max_value=9)
valid_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_frame_ms = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
pair = st.tuples(valid_index, valid_index)
pairs = st.lists(pair, max_size=100)
decisions = st.lists(st.sampled_from(["ACCEPT", "REJECT"]), max_size=100)

# confusion matrix cells must never be negative for any input
@given(n=num_classes, pairs=pairs)
@pytest.mark.property
def test_confusion_matrix_never_negative(n, pairs):
    valid = [(a, p) for a, p in pairs if a < n and p < n]
    matrix = confusion_matrix(valid, n)
    for row in matrix:
        for cell in row:
            assert cell >= 0

# confusion matrix dimensions must always equal num_classes x num_classes
@given(n=num_classes, pairs=pairs)
@pytest.mark.property
def test_confusion_matrix_dimensions(n, pairs):
    valid = [(a, p) for a, p in pairs if a < n and p < n]
    matrix = confusion_matrix(valid, n)
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)

# accuracy must always lie within [0.0, 1.0]
@given(n=num_classes, pairs=pairs)
@pytest.mark.property
def test_accuracy_in_unit_interval(n, pairs):
    valid = [(a, p) for a, p in pairs if a < n and p < n]
    matrix = confusion_matrix(valid, n)
    acc = accuracy(matrix)
    assert 0.0 <= acc <= 1.0

# per-class precision, recall, and F1 must all lie within [0.0, 1.0]
@given(n=num_classes, pairs=pairs)
@pytest.mark.property
def test_per_class_metrics_in_unit_interval(n, pairs):
    valid = [(a, p) for a, p in pairs if a < n and p < n]
    matrix = confusion_matrix(valid, n)
    for r in per_class_metrics(matrix):
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0

# total must always equal accepted + rejected regardless of event sequence
@given(decisions=decisions, confidence=valid_confidence, frame_ms=valid_frame_ms)
@pytest.mark.property
def test_total_equals_accepted_plus_rejected(decisions, confidence, frame_ms):
    m = RunningMetrics()
    for d in decisions:
        m.update(make_event(decision=d, confidence=confidence, frame_ms=frame_ms))
    assert m.total == m.accepted + m.rejected

# mean_confidence must stay within [0.0, 1.0] when fed valid confidence values
@given(decisions=decisions.filter(bool), confidence=valid_confidence)
@pytest.mark.property
def test_mean_confidence_in_unit_interval(decisions, confidence):
    m = RunningMetrics()
    for d in decisions:
        m.update(make_event(decision=d, confidence=confidence))
    assert 0.0 <= m.mean_confidence <= 1.0

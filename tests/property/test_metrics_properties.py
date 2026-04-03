# tests/property/test_metrics_properties.py

import pytest
from hypothesis import given, strategies as st

from utils.metrics import RunningMetrics, confusion_matrix, per_class_metrics, accuracy, normalise_confusion_matrix
from tests.helpers.events_helpers import make_event
from tests.helpers.metrics_helpers import square_non_negative_int_matrix

num_classes = st.integers(min_value=1, max_value=10)
non_negative_int = st.integers(min_value=0, max_value=100)
valid_index = st.integers(min_value=0, max_value=9)
valid_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_frame_ms = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
pair = st.tuples(valid_index, valid_index)
pairs = st.lists(pair, max_size=100)
low_conf_flags = st.lists(st.booleans(), max_size=100)

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

# all normalised values must lie within [0.0, 1.0] for any non-negative input
@given(matrix=square_non_negative_int_matrix())
@pytest.mark.property
def test_normalised_values_in_unit_interval(matrix):
    for val in (v for row in normalise_confusion_matrix(matrix) for v in row):
        assert 0.0 <= val <= 1.0

# every non-zero row must sum to exactly 1.0 after normalisation
@given(matrix=square_non_negative_int_matrix())
@pytest.mark.property
def test_non_zero_rows_sum_to_one(matrix):
    for raw_row, norm_row in zip(matrix, normalise_confusion_matrix(matrix), strict=True):
        if sum(raw_row) > 0:
            assert sum(norm_row) == pytest.approx(1.0)

# zero rows must remain all-zero after normalisation
@given(matrix=square_non_negative_int_matrix())
@pytest.mark.property
def test_zero_rows_stay_zero(matrix):
    for raw_row, norm_row in zip(matrix, normalise_confusion_matrix(matrix), strict=True):
        if sum(raw_row) == 0:
            assert all(v == 0.0 for v in norm_row)

# output dimensions must always match input dimensions
@given(matrix=square_non_negative_int_matrix())
@pytest.mark.property
def test_normalised_dimensions_match_input(matrix):
    result = normalise_confusion_matrix(matrix)
    assert len(result) == len(matrix)
    assert all(len(result[i]) == len(matrix[i]) for i in range(len(matrix)))

# low_confidence count must never exceed total regardless of event sequence
@given(flags=low_conf_flags, confidence=valid_confidence, frame_ms=valid_frame_ms)
@pytest.mark.property
def test_low_confidence_never_exceeds_total(flags, confidence, frame_ms):
    m = RunningMetrics()
    for f in flags:
        m.update(make_event(low_confidence=f, confidence=confidence, frame_ms=frame_ms))
    assert m.low_confidence <= m.total

# mean_confidence must stay within [0.0, 1.0] when fed valid confidence values
@given(flags=low_conf_flags.filter(bool), confidence=valid_confidence)
@pytest.mark.property
def test_mean_confidence_in_unit_interval(flags, confidence):
    m = RunningMetrics()
    for f in flags:
        m.update(make_event(low_confidence=f, confidence=confidence))
    assert 0.0 <= m.mean_confidence <= 1.0

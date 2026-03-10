# tests/property/test_config_properties.py

import pytest
from hypothesis import given, strategies as st

from config.config import _validate_pair, ConfigError

numeric = st.floats(
    allow_nan=False,
    allow_infinity=False,
)

# ordered numeric pairs must always be accepted and reversed pairs rejected
@given(numeric, numeric)
@pytest.mark.property
def test_validate_pair_accepts_ordered_values(lo, hi):
    if lo <= hi:
        _validate_pair([lo, hi], "pair")
    else:
        with pytest.raises(ConfigError):
            _validate_pair([lo, hi], "pair")

# validate_pair only accepts lists of length exactly two
@given(st.lists(numeric, min_size=0, max_size=5))
@pytest.mark.property
def test_validate_pair_requires_exact_length(values):
    if len(values) != 2:
        with pytest.raises(ConfigError):
            _validate_pair(values, "pair")
    else:
        lo, hi = values
        if lo <= hi:
            _validate_pair(values, "pair")
        else:
            with pytest.raises(ConfigError):
                _validate_pair(values, "pair")

# non numeric inputs must never be accepted as valid numeric ranges
@given(st.text(), st.text())
@pytest.mark.property
def test_validate_pair_rejects_strings(lo, hi):
    with pytest.raises(ConfigError):
        _validate_pair([lo, hi], "pair")

# tests/property/test_config_properties.py

import pytest
from hypothesis import given, strategies as st

from config.validate import _validate_pair, ConfigError
from tests.helpers.config_helpers import make_config

numeric = st.floats(
    allow_nan=False,
    allow_infinity=False,
)


@pytest.mark.regression
class TestConfigProperties:
    """property tests for config validation invariants"""

    # ordered numeric pairs must always be accepted and reversed pairs rejected
    @given(numeric, numeric)
    def test_validate_pair_accepts_ordered_values(self, lo, hi):
        if lo <= hi:
            _validate_pair([lo, hi], "pair")
        else:
            with pytest.raises(ConfigError):
                _validate_pair([lo, hi], "pair")

    # validate_pair only accepts lists of length exactly two
    @given(st.lists(numeric, min_size=0, max_size=5))
    def test_validate_pair_requires_exact_length(self, values):
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
    def test_validate_pair_rejects_strings(self, lo, hi):
        with pytest.raises(ConfigError):
            _validate_pair([lo, hi], "pair")

    # roi_fraction outside (0.0, 1.0] must always be rejected
    @given(st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: not (0.0 < x <= 1.0)))
    def test_roi_fraction_outside_range_always_rejected(self, value):
        with pytest.raises(ConfigError):
            make_config(preprocess={"roi_fraction": value})

    # auto_exposure values not in {1, 3} must always be rejected
    @given(st.integers().filter(lambda x: x not in {1, 3}))
    def test_auto_exposure_not_in_valid_set_always_rejected(self, value):
        with pytest.raises(ConfigError):
            make_config(camera={"auto_exposure": value})

    # power_line_frequency values not in {0, 1, 2} must always be rejected
    @given(st.integers().filter(lambda x: x not in {0, 1, 2}))
    def test_power_line_frequency_not_in_valid_set_always_rejected(self, value):
        with pytest.raises(ConfigError):
            make_config(camera={"power_line_frequency": value})

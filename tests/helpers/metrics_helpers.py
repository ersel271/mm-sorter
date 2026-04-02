# tests/helpers/metrics_helpers.py

from hypothesis import strategies as st

@st.composite
def square_non_negative_int_matrix(draw, min_n: int = 1, max_n: int = 10) -> list[list[int]]:
    """draw an n x n matrix of non-negative ints"""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    return [draw(st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n)) for _ in range(n)]

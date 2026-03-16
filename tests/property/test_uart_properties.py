# tests/property/test_uart_properties.py

import pytest
from hypothesis import given, strategies as st

from src.io.uart import build_packet
from config.constants import UART_SEPARATOR, UART_TERMINATOR

ascii_key = st.text(
    alphabet=st.characters(min_codepoint=48, max_codepoint=122),
    min_size=1,
    max_size=8,
)

value_strategy = st.one_of(
    st.integers(min_value=0, max_value=100000),
    st.floats(
        min_value=0,
        max_value=100000,
        allow_nan=False,
        allow_infinity=False,
    ),
)

# every serialised packet must end with the configured UART terminator
@given(st.dictionaries(ascii_key, value_strategy, min_size=0, max_size=10))
@pytest.mark.property
def test_packet_ends_with_terminator(fields):
    pkt = build_packet(fields)
    assert pkt.endswith(UART_TERMINATOR.encode("ascii"))

# number of serialised fields must match number of separators in packet
@given(st.dictionaries(ascii_key, value_strategy, min_size=1, max_size=10))
@pytest.mark.property
def test_packet_separator_count(fields):
    pkt = build_packet(fields)
    decoded = pkt.decode("ascii").strip()
    parts = decoded.split(UART_SEPARATOR)
    assert len(parts) == len(fields)

# packets must always be ASCII encodable for UART transmission
@given(st.dictionaries(ascii_key, value_strategy, min_size=0, max_size=10))
@pytest.mark.property
def test_packet_is_ascii(fields):
    pkt = build_packet(fields)
    pkt.decode("ascii")

# serialisation must be deterministic for identical inputs
@given(st.dictionaries(ascii_key, value_strategy, min_size=0, max_size=10))
@pytest.mark.property
def test_packet_deterministic(fields):
    pkt1 = build_packet(fields)
    pkt2 = build_packet(fields)
    assert pkt1 == pkt2

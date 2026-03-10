# tests/fixtures/uart_fixtures.py

import pytest
import serial
from unittest.mock import MagicMock

@pytest.fixture
def mock_port() -> MagicMock:
    port = MagicMock(spec=serial.Serial)
    port.write = MagicMock(return_value=None)
    port.readline = MagicMock(return_value=b"")
    port.close = MagicMock(return_value=None)
    port.timeout = 0.1
    return port

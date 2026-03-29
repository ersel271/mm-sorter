# tests/integration/test_uart_integration.py

import pytest
import serial

from config.constants import UART_SEPARATOR
from tests.helpers.uart_helpers import sample_fields

@pytest.mark.integration
class TestUARTIntegration:
    """verify full send cycles and disconnect recovery."""

    def test_full_classification_cycle(self, sender, mock_port):
        assert sender.send(sample_fields()) is True
        written = mock_port.write.call_args[0][0]
        decoded = written.decode("ascii")
        parts = decoded.strip().split(UART_SEPARATOR)
        assert len(parts) == 6
        assert int(parts[0]) == 42
        assert int(parts[1]) == 3
        assert float(parts[2]) == pytest.approx(0.91)
        assert int(parts[3]) == 1
        assert int(parts[4]) == 960
        assert int(parts[5]) == 540

    def test_pipeline_survives_disconnect(self, sender, mock_port):
        sender.send(sample_fields(id=1))
        assert sender.packets_sent == 1
        mock_port.write.side_effect = serial.SerialException("unplugged")
        sender.send(sample_fields(id=2))
        assert sender.packets_dropped == 1
        sender.send(sample_fields(id=3))
        assert sender.packets_dropped == 2

    def test_extended_packet(self, sender, mock_port):
        fields = sample_fields(area=1500, perimeter=142.5)
        assert sender.send(fields) is True
        written = mock_port.write.call_args[0][0].decode("ascii")
        parts = written.strip().split(UART_SEPARATOR)
        assert len(parts) == 8
        assert parts[6] == "1500"
        assert parts[7] == "142.50"

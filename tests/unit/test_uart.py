# tests/test_uart.py

import pytest
import serial

from src.io import UARTSender, build_packet
from config.constants import UART_SEPARATOR, UART_TERMINATOR
from tests.helpers.uart_helpers import sample_fields

@pytest.fixture
def sender(default_cfg, mock_port) -> UARTSender:
    s = UARTSender(default_cfg)
    s._port = mock_port
    s._is_open = True
    return s

@pytest.mark.smoke
@pytest.mark.unit
class TestBuildPacket:
    """verify packet serialisation format, extensibility, and edge cases."""

    def test_basic_fields(self):
        pkt = build_packet(sample_fields())
        assert pkt == b"42;3;0.91;960;540\n"

    def test_float_formatting(self):
        pkt = build_packet({"val": 0.5})
        assert pkt == b"0.50\n"

    def test_integer_formatting(self):
        pkt = build_packet({"a": 0, "b": 65535})
        assert pkt == b"0;65535\n"

    def test_single_field(self):
        pkt = build_packet({"only": 42})
        assert pkt == b"42\n"

    def test_separator_count(self):
        pkt = build_packet({"a": 1, "b": 2, "c": 3})
        decoded = pkt.decode("ascii")
        assert decoded.count(UART_SEPARATOR) == 2

    def test_terminator_at_end(self):
        pkt = build_packet({"x": 1})
        assert pkt.endswith(UART_TERMINATOR.encode("ascii"))

    def test_empty_fields(self):
        pkt = build_packet({})
        assert pkt == UART_TERMINATOR.encode("ascii")

    def test_extensibility_extra_fields(self):
        fields = sample_fields(area=1234, ts=1.50)
        pkt = build_packet(fields)
        parts = pkt.decode("ascii").strip().split(UART_SEPARATOR)
        assert len(parts) == 7
        assert parts[5] == "1234"
        assert parts[6] == "1.50"

    def test_reduced_fields(self):
        pkt = build_packet({"id": 1, "class": 3})
        assert pkt == b"1;3\n"

@pytest.mark.unit
class TestUARTSenderOpen:
    """verify open() behaviour for available and unavailable ports."""

    def test_open_real_port_failure(self, default_cfg):
        s = UARTSender(default_cfg)
        result = s.open()
        assert result is False
        assert s.is_open is False

    def test_open_already_open(self, sender):
        assert sender.open() is True

@pytest.mark.unit
class TestUARTSenderSend:
    """verify send() writes correct packets and handles failures gracefully."""

    def test_send_writes_to_port(self, sender, mock_port):
        result = sender.send(sample_fields())
        assert result is True
        mock_port.write.assert_called_once_with(b"42;3;0.91;960;540\n")

    def test_send_increments_counter(self, sender):
        sender.send(sample_fields(id=1))
        sender.send(sample_fields(id=2))
        assert sender.packets_sent == 2

    def test_send_when_closed_drops(self, default_cfg):
        s = UARTSender(default_cfg)
        result = s.send(sample_fields())
        assert result is False
        assert s.packets_dropped == 1

    def test_send_serial_exception_drops(self, sender, mock_port):
        mock_port.write.side_effect = serial.SerialException("disconnected")
        result = sender.send(sample_fields())
        assert result is False
        assert sender.packets_dropped == 1

    def test_send_os_error_drops(self, sender, mock_port):
        mock_port.write.side_effect = OSError("device removed")
        result = sender.send(sample_fields())
        assert result is False
        assert sender.packets_dropped == 1

    def test_disconnect_marks_closed(self, sender, mock_port):
        mock_port.write.side_effect = serial.SerialException("gone")
        sender.send(sample_fields())
        assert sender.is_open is False

    def test_multiple_sends(self, sender):
        for i in range(10):
            sender.send(sample_fields(id=i))
        assert sender.packets_sent == 10
        assert sender.packets_dropped == 0

    def test_warning_only_on_first_drop(self, sender, mock_port, caplog):
        mock_port.write.side_effect = serial.SerialException("gone")
        sender.send(sample_fields(id=1))
        sender.send(sample_fields(id=2))
        sender.send(sample_fields(id=3))
        assert sender.packets_dropped == 3
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

@pytest.mark.unit
class TestUARTSenderReceive:
    """verify receive() decoding, timeout handling, and error recovery."""

    def test_receive_returns_line(self, sender, mock_port):
        mock_port.readline.return_value = b"ACK\n"
        result = sender.receive()
        assert result == "ACK"

    def test_receive_strips_whitespace(self, sender, mock_port):
        mock_port.readline.return_value = b"  OK  \r\n"
        result = sender.receive()
        assert result == "OK"

    def test_receive_empty_returns_none(self, sender, mock_port):
        mock_port.readline.return_value = b""
        result = sender.receive()
        assert result is None

    def test_receive_when_closed_returns_none(self, default_cfg):
        s = UARTSender(default_cfg)
        assert s.receive() is None

    def test_receive_with_timeout(self, sender, mock_port):
        mock_port.readline.return_value = b"DATA\n"
        result = sender.receive(timeout=0.5)
        assert result == "DATA"

    def test_receive_exception_marks_closed(self, sender, mock_port):
        mock_port.readline.side_effect = serial.SerialException("error")
        result = sender.receive()
        assert result is None
        assert sender.is_open is False

@pytest.mark.unit
class TestUARTSenderClose:
    """verify close() cleans up state and calls port.close()."""

    def test_close_resets_state(self, sender):
        sender.send(sample_fields())
        sender.close()
        assert sender.is_open is False

    def test_close_calls_port_close(self, sender, mock_port):
        sender.close()
        mock_port.close.assert_called_once()

    def test_close_when_already_closed(self, default_cfg):
        s = UARTSender(default_cfg)
        s.close()
        assert s.is_open is False

    def test_counters_persist_after_close(self, sender):
        sender.send(sample_fields())
        sender.close()
        assert sender.packets_sent == 1

# tests/hw/test_uart_hw.py

import os

import pytest

from src.io import UARTSender
from tests.helpers.config_helpers import make_config

@pytest.mark.hw
@pytest.mark.serial
class TestUARTHardwareVirtual:
    """serial round-trip tests via virtual pty port."""

    def test_send_packet_reaches_other_end(self, virtual_uart):
        master_fd, slave_path = virtual_uart
        sender = UARTSender(make_config(uart={"port": slave_path, "baud": 115200}))
        assert sender.open() is True
        assert sender.send({"id": 1, "class": 2, "conf": 0.9, "decision": 1, "x": 100, "y": 200}) is True
        data = os.read(master_fd, 256)
        assert len(data) > 0
        sender.close()

    def test_receive_reads_from_port(self, virtual_uart):
        master_fd, slave_path = virtual_uart
        sender = UARTSender(make_config(uart={"port": slave_path, "baud": 115200}))
        assert sender.open() is True
        os.write(master_fd, b"hello\n")
        result = sender.receive()
        assert result == "hello"
        sender.close()

    def test_open_and_close_cycle(self, virtual_uart):
        _master_fd, slave_path = virtual_uart
        sender = UARTSender(make_config(uart={"port": slave_path, "baud": 115200}))
        assert sender.open() is True
        assert sender.is_open is True
        sender.close()
        assert sender.is_open is False

    def test_send_then_receive_roundtrip(self, virtual_uart):
        master_fd, slave_path = virtual_uart
        sender = UARTSender(make_config(uart={"port": slave_path, "baud": 115200}))
        assert sender.open() is True
        fields = {"id": 1, "class": 2, "conf": 0.9, "decision": 1, "x": 100, "y": 200}
        assert sender.send(fields) is True
        raw = os.read(master_fd, 256)
        assert len(raw) > 0
        os.write(master_fd, raw)
        result = sender.receive()
        assert result is not None
        sender.close()

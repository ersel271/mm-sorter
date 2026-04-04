# tests/property/test_uart_stateful.py

import pytest
import serial
from unittest.mock import MagicMock

from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.io import UARTSender
from tests.helpers.config_helpers import make_config

class UARTSenderStateMachine(RuleBasedStateMachine):
    """state machine for UARTSender exploring connect, send, and disconnect sequences"""

    def __init__(self):
        super().__init__()
        self.sender = UARTSender(make_config())
        self._model_open: bool = False
        self._model_attempted: int = 0

    # simulate successful connection via mock port
    @rule()
    def open_with_mock(self):
        mock_port = MagicMock()
        mock_port.readline.return_value = b""
        self.sender._port = mock_port
        self.sender._is_open = True
        self._model_open = True

    # simulate disconnection
    @rule()
    def close(self):
        self.sender.close()
        self._model_open = False

    # attempt to send a packet when port may be open or closed
    @rule()
    def send(self):
        fields = {"id": 1, "class": 2, "conf": 0.9, "decision": 1, "x": 100, "y": 200}
        self.sender.send(fields)
        self._model_attempted += 1

    # simulate a write failure mid-send
    @rule()
    def send_raises(self):
        if self.sender._is_open and self.sender._port is not None:
            self.sender._port.write.side_effect = serial.SerialException("injected")
        self.sender.send({"id": 1, "class": 2, "conf": 0.9, "decision": 1, "x": 0, "y": 0})
        self._model_attempted += 1

    # attempt to receive; must not raise regardless of port state
    @rule()
    def receive(self):
        self.sender.receive()

    # invariant: counters must never be negative
    @invariant()
    def counters_non_negative(self):
        assert self.sender.packets_sent >= 0
        assert self.sender.packets_dropped >= 0

    # invariant: every send attempt must be counted exactly once
    @invariant()
    def total_equals_attempted(self):
        total = self.sender.packets_sent + self.sender.packets_dropped
        assert total == self._model_attempted

    # invariant: when open, port must not be None
    @invariant()
    def open_implies_port_not_none(self):
        if self.sender._is_open:
            assert self.sender._port is not None


TestUARTSenderStateMachine = UARTSenderStateMachine.TestCase
TestUARTSenderStateMachine = pytest.mark.regression(TestUARTSenderStateMachine)

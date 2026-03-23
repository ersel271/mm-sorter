# tests/property/test_uart_stateful.py

import pytest
import serial

from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from config import Config
from src.io import UARTSender

class UARTSenderStateMachine(RuleBasedStateMachine):
    """state machine model for UARTSender where hypothesis explores random connect, send and disconnect sequences"""

    def __init__(self):
        super().__init__()

        self.cfg = Config()
        self.sender = UARTSender(self.cfg)

        # counters used as a simple behavioural model
        self.attempted_packets = 0

    # simulate successful connection
    @rule()
    def connect(self):
        self.sender._is_open = True

    # simulate disconnection
    @rule()
    def disconnect(self):
        self.sender._is_open = False

    # attempt to send a packet
    @rule()
    def send_packet(self):

        fields = {
            "id": 1,
            "class": 2,
            "conf": 0.9,
            "x": 100,
            "y": 200,
        }

        try:
            self.sender.send(fields)
        except serial.SerialException:
            pass

        self.attempted_packets += 1

    # invariant: counters must never be negative
    @invariant()
    def counters_non_negative(self):
        assert self.sender.packets_sent >= 0
        assert self.sender.packets_dropped >= 0

    # invariant: total packets tracked must not exceed attempts
    @invariant()
    def counters_consistent(self):
        total = self.sender.packets_sent + self.sender.packets_dropped
        assert total <= self.attempted_packets

TestUARTSenderStateMachine = UARTSenderStateMachine.TestCase
TestUARTSenderStateMachine = pytest.mark.property(TestUARTSenderStateMachine)

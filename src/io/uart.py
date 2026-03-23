# src/io/uart.py
"""
UART serial communication for transmitting classification results.

Packets are ASCII, semicolon-delimited, newline-terminated, and
extensible by adding fields to the dict passed to send().

Usage:
    sender = UARTSender(cfg)
    sender.open()
    sender.send({"id": 1, "class": 3, "conf": 0.91, "x": 960, "y": 540})
    sender.close()
"""

import logging
from typing import Any

import serial

from config import Config
from config.constants import UART_SEPARATOR, UART_TERMINATOR

log = logging.getLogger(__name__)

def build_packet(fields: dict[str, Any]) -> bytes:
    """
    serialise a dict of fields into an ASCII packet.

    field order follows dict insertion order. floats are formatted
    to two decimal places; everything else is converted with str().
    """
    parts = []
    for value in fields.values():
        if isinstance(value, float):
            parts.append(f"{value:.2f}")
        else:
            parts.append(str(value))

    line = UART_SEPARATOR.join(parts) + UART_TERMINATOR
    return line.encode("ascii")

class UARTSender:
    """
    serial transmitter for classification packets.

    if the port is unavailable or disconnects mid-operation, the sender
    logs a warning on the first failure then silently counts subsequent
    drops. the pipeline is never blocked. a summary is logged on close().
    """

    def __init__(self, config: Config):
        self._cfg = config.uart
        self._port: serial.Serial | None = None
        self._is_open: bool = False

        self.packets_sent: int = 0
        self.packets_dropped: int = 0
        self._warned: bool = False
        
        log.info("uart initialised -- port=%s baud=%d", self._cfg["port"], self._cfg["baud"])

    def open(self) -> bool:
        """
        open the serial port. returns True on success, False on failure.
        never raises.
        """
        if self._is_open:
            return True

        try:
            self._port = serial.Serial(
                port=self._cfg["port"],
                baudrate=self._cfg["baud"],
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._cfg.get("timeout", 0.1),
            )
            self._is_open = True
            self._warned = False
            log.info("uart port opened: %s @ %d baud", self._cfg["port"], self._cfg["baud"])
            return True
        except (serial.SerialException, OSError) as e:
            log.warning("uart port open failed: %s", e)
            self._port = None
            self._is_open = False
            return False

    def send(self, fields: dict[str, Any]) -> bool:
        """
        serialise fields into a packet and transmit.
        returns True if sent, False if dropped.
        """
        if not self._is_open or self._port is None:
            self._drop()
            return False

        packet = build_packet(fields)

        try:
            self._port.write(packet)
            self.packets_sent += 1
            return True
        except (serial.SerialException, OSError) as e:
            self._drop(reason=str(e))
            self._is_open = False
            return False

    def receive(self, timeout: float | None = None) -> str | None:
        """
        read a line from the serial port.
        returns decoded line or None.
        """
        if not self._is_open or self._port is None:
            return None

        try:
            prev_timeout = self._port.timeout
            if timeout is not None:
                self._port.timeout = timeout

            raw = self._port.readline()

            if timeout is not None:
                self._port.timeout = prev_timeout

            if raw:
                return str(raw.decode("ascii", errors="replace").strip())
            return None
        except (serial.SerialException, OSError) as e:
            log.warning("uart receive failed: %s", e)
            self._is_open = False
            return None

    def close(self) -> None:
        """
        close the serial port and log a summary of sent/dropped packets.
        """
        if self._port is not None:
            try:
                self._port.close()
            except (serial.SerialException, OSError):
                pass
        self._port = None
        self._is_open = False
        log.info(
            "uart closed -- sent: %d, dropped: %d",
            self.packets_sent, self.packets_dropped,
        )

    @property
    def is_open(self) -> bool:
        return self._is_open

    def _drop(self, reason: str | None = None) -> None:
        """
        increment drop counter. logs a warning on the first drop
        only to avoid flooding during sustained disconnects.
        """
        self.packets_dropped += 1
        if not self._warned:
            msg = "uart packet dropped"
            if reason:
                msg += f": {reason}"
            log.warning(msg)
            self._warned = True

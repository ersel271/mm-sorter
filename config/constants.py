# config/constants.py
"""Fixed enumeration values for the M&M sorter system."""

from enum import IntEnum

class ColourID(IntEnum):
    NON_MM = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    ORANGE = 5
    BROWN = 6

COLOUR_NAMES: dict[int, str] = {
    ColourID.NON_MM: "Non-M&M",
    ColourID.RED: "Red",
    ColourID.GREEN: "Green",
    ColourID.BLUE: "Blue",
    ColourID.YELLOW: "Yellow",
    ColourID.ORANGE: "Orange",
    ColourID.BROWN: "Brown",
}

COLOUR_IDS: dict[str, int] = {v.lower(): k for k, v in COLOUR_NAMES.items()}

NUM_COLOURS = 6

UART_SEPARATOR = ";"
UART_TERMINATOR = "\n"
OBJECT_ID_MAX = 65535
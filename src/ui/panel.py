# src/ui/panel.py
"""Shared constants and abstract base class for UI panels."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from config.constants import ColourID
from src.vision.features import Features
from src.vision.rule import Decision

PANEL_W: int = 230
BAR_H:   int = 10

FONT = cv2.FONT_HERSHEY_SIMPLEX

PANEL_SEP_COLOUR: tuple[int, int, int] = ( 55,  55,  55)
TEXT_COLOUR:      tuple[int, int, int] = (190, 200, 200)
DIM_COLOUR:       tuple[int, int, int] = (110, 110, 110)
BAR_BG_COLOUR:    tuple[int, int, int] = ( 35,  35,  35)

LABEL_COLOURS: dict[ColourID, tuple[int, int, int]] = {
    ColourID.NON_MM: (110, 110, 110),
    ColourID.RED:    ( 40,  50, 220),
    ColourID.GREEN:  ( 10, 200,  60),
    ColourID.BLUE:   (220, 100,  10),
    ColourID.YELLOW: (  0, 210, 230),
    ColourID.ORANGE: (  0, 130, 240),
    ColourID.BROWN:  ( 80, 120, 160),
}

class Panel(ABC):
    """abstract base for UI panels that render into BGR ndarrays.

    sidebar panels follow the (features, decision, panel_h) signature below.
    panels with a different layout may override render with a different signature.
    """

    @abstractmethod
    def render(self, features: Features | None, decision: Decision | None, panel_h: int) -> np.ndarray:
        """return a (panel_h, PANEL_W, 3) BGR ndarray."""
        ...

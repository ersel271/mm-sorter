# src/ui/panel.py
"""Shared constants and abstract base class for sidebar panels."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from config.constants import ColourID
from src.vision.features import Features
from src.vision.rule import Decision

class Panel(ABC):
    """abstract base for UI panels that render into BGR ndarrays.

    sidebar panels follow the (features, decision, panel_h) signature below.
    panels with a different layout may override render with a different signature.
    """

    @abstractmethod
    def render(self, features: Features | None, decision: Decision | None, panel_h: int) -> np.ndarray:
        """return a (panel_h, PANEL_W, 3) BGR ndarray."""
        ...

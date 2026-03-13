# tests/fixtures/vision_fixtures.py

import pytest

from config import Config
from src.vision.preprocess import Preprocessor

@pytest.fixture
def prep() -> Preprocessor:
    return Preprocessor(Config())

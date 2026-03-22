# tests/fixtures/vision_fixtures.py

import pytest

from config import Config
from src.vision.preprocess import Preprocessor
from src.vision.features import FeatureExtractor

@pytest.fixture
def prep() -> Preprocessor:
    return Preprocessor(Config())

@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(Config())

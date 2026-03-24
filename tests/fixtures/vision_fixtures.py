# tests/fixtures/vision_fixtures.py

import pytest

from config import Config
from src.vision import Classifier, FeatureExtractor, Preprocessor

@pytest.fixture
def prep() -> Preprocessor:
    return Preprocessor(Config())

@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(Config())

@pytest.fixture
def classifier() -> Classifier:
    return Classifier(Config())

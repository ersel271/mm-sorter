# tests/fixtures/vision_fixtures.py

import pytest

from src.vision import Classifier, FeatureExtractor, Preprocessor

@pytest.fixture
def prep(default_cfg) -> Preprocessor:
    return Preprocessor(default_cfg)

@pytest.fixture
def extractor(default_cfg) -> FeatureExtractor:
    return FeatureExtractor(default_cfg)

@pytest.fixture
def classifier(default_cfg) -> Classifier:
    return Classifier(default_cfg)

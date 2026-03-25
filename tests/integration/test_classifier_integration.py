# tests/integration/test_classifier_integration.py

import copy

import pytest

from config import Config
from config.constants import ColourID
from src.vision import Classifier
from tests.helpers.config_helpers import write_config
from tests.helpers.features_helpers import make_preprocess_result, make_features
from tests.helpers.image_helpers import draw_saturated_circle, make_frame

@pytest.mark.integration
class TestClassifierPipelineIntegration:
    """verify the full preprocess → features → classify pipeline."""

    def test_full_pipeline_returns_valid_colour_id(self, prep, extractor, classifier):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert result.found is True
        features = extractor.extract(result)
        decision = classifier.classify(features)
        assert isinstance(decision.label, ColourID)

    def test_pipeline_decision_has_valid_confidence(self, prep, extractor, classifier):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)
        decision = classifier.classify(features)
        assert 0.0 <= decision.confidence <= 1.0

    def test_all_config_keys_accessible_without_error(self, classifier):
        # verify no KeyError from any rule reading its threshold
        f = make_features()
        classifier.classify(f)

    def test_lower_sat_threshold_stops_low_sat_rejection(self, valid_data, tmp_path):
        # with default sat_min=60, sat_mean=50 triggers LowSaturationRule
        default_cfg = Config()
        default_classifier = Classifier(default_cfg)
        f = make_features(sat_mean=50.0)
        d_default = default_classifier.classify(f)
        assert d_default.rule == "low_saturation"

        # lower sat_min to 30, sat_mean=50 no longer triggers it
        data = copy.deepcopy(valid_data)
        data["thresholds"]["sat_min"] = 30
        low_cfg = Config(write_config(data, tmp_path))
        low_classifier = Classifier(low_cfg)
        d_low = low_classifier.classify(f)
        assert d_low.rule != "low_saturation"

    def test_synthetic_object_does_not_crash_pipeline(self, extractor, classifier):
        result = make_preprocess_result(hue=10, sat=200, val=180, radius=30)
        features = extractor.extract(result)
        decision = classifier.classify(features)
        assert decision.label is not None

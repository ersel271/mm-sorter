# tests/integration/test_features_integration.py

import numpy as np
import pytest

from src.vision.features import Features
from tests.helpers.vision_helpers import make_preprocess_result
from tests.helpers.image_helpers import draw_saturated_circle, make_frame

@pytest.mark.smoke
@pytest.mark.regression
class TestFeaturesIntegration:
    """verify feature extraction on realistically preprocessed frames."""

    def test_full_pipeline_produces_valid_features(self, prep, extractor):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert result.found is True
        features = extractor.extract(result)
        assert isinstance(features, Features)
        assert features.mask_pixels > 0

    def test_feature_ranges_on_circle_object(self, prep, extractor):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)

        assert 0.0 <= features.circularity <= 1.0
        assert features.circularity > 0.75
        assert 0.0 <= features.solidity <= 1.0
        assert features.solidity > 0.90
        assert 0.8 <= features.aspect_ratio <= 1.2
        assert 0.0 <= features.highlight_ratio <= 1.0
        assert features.sat_mean > 100
        assert 0.0 <= features.val_mean <= 255.0
        assert features.texture_variance >= 0.0
        assert features.hue_peak_width >= 1
        assert len(features.hue_hist) == 180

    def test_different_hues_produce_different_hist_peaks(self, extractor):
        f_green = extractor.extract(make_preprocess_result(hue=60))
        f_blue = extractor.extract(make_preprocess_result(hue=120))
        peak_green = int(np.argmax(f_green.hue_hist))
        peak_blue = int(np.argmax(f_blue.hue_hist))
        # green and blue peaks must be clearly separated
        assert abs(peak_green - peak_blue) > 30

    def test_hue_hist_sum_after_full_preprocess(self, prep, extractor):
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)
        assert abs(features.hue_hist.sum() - 1.0) < 0.01

    def test_sat_mean_reflects_object_colour(self, extractor):
        low_sat = extractor.extract(make_preprocess_result(sat=50))
        high_sat = extractor.extract(make_preprocess_result(sat=220))
        assert high_sat.sat_mean > low_sat.sat_mean

    def test_highlight_ratio_increases_with_brightness(self, extractor):
        dark = extractor.extract(make_preprocess_result(val=150))
        bright = extractor.extract(make_preprocess_result(val=250))
        assert bright.highlight_ratio > dark.highlight_ratio

    def test_val_mean_reflects_object_brightness(self, extractor):
        dark = extractor.extract(make_preprocess_result(val=50))
        bright = extractor.extract(make_preprocess_result(val=210))
        assert bright.val_mean > dark.val_mean

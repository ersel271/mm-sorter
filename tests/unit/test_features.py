# tests/unit/test_features.py

import dataclasses

import numpy as np
import pytest

from config import Config
from src.vision.features import FeatureExtractor, Features
from tests.helpers.config_helpers import write_config
from tests.helpers.features_helpers import make_preprocess_result

@pytest.mark.smoke
@pytest.mark.unit
class TestExtractGuards:
    """verify that extract() enforces its input preconditions."""

    def test_raises_when_not_found(self, extractor):
        result = dataclasses.replace(make_preprocess_result(), found=False)
        with pytest.raises(ValueError, match="found"):
            extractor.extract(result)

    def test_raises_when_mask_empty(self, extractor):
        base = make_preprocess_result()
        result = dataclasses.replace(base, mask=np.zeros_like(base.mask))
        with pytest.raises(ValueError, match="mask"):
            extractor.extract(result)

    def test_raises_when_contour_is_none(self, extractor):
        result = dataclasses.replace(make_preprocess_result(), contour=None)
        with pytest.raises(ValueError, match="contour"):
            extractor.extract(result)

    def test_returns_features_instance(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert isinstance(features, Features)

    def test_mask_pixels_matches_nonzero_count(self, extractor):
        result = make_preprocess_result(radius=30)
        features = extractor.extract(result)
        assert features.mask_pixels == int(np.count_nonzero(result.mask))

@pytest.mark.unit
class TestSatMean:
    """verify mean saturation across masked pixels."""

    def test_high_saturation_object(self, extractor):
        features = extractor.extract(make_preprocess_result(sat=200))
        assert features.sat_mean > 150

    def test_low_saturation_object(self, extractor):
        features = extractor.extract(make_preprocess_result(sat=30))
        assert features.sat_mean < 60

    def test_uniform_sat_matches_value(self, extractor):
        # uniform circle: every masked pixel has sat=180, mean must be exact
        features = extractor.extract(make_preprocess_result(sat=180))
        assert abs(features.sat_mean - 180) < 5

@pytest.mark.unit
class TestValMean:
    """verify mean brightness across masked pixels."""

    def test_high_val_object(self, extractor):
        features = extractor.extract(make_preprocess_result(val=220))
        assert features.val_mean > 170

    def test_low_val_object(self, extractor):
        features = extractor.extract(make_preprocess_result(val=40))
        assert features.val_mean < 70

    def test_uniform_val_matches_value(self, extractor):
        # uniform circle: every masked pixel has val=160, mean must be close
        features = extractor.extract(make_preprocess_result(val=160))
        assert abs(features.val_mean - 160) < 5

@pytest.mark.unit
class TestHighlightRatio:
    """verify fraction of pixels exceeding the brightness threshold."""

    def test_all_below_threshold_returns_zero(self, extractor):
        # val=180 is below highlight_value=240 in default config
        features = extractor.extract(make_preprocess_result(val=180))
        assert features.highlight_ratio == 0.0

    def test_all_above_threshold_returns_one(self, extractor):
        # val=250 is above highlight_value=240
        features = extractor.extract(make_preprocess_result(val=250))
        assert features.highlight_ratio == 1.0

    def test_at_threshold_not_counted(self, extractor):
        # val == highlight_value (240) uses strict >, must return 0.0
        features = extractor.extract(make_preprocess_result(val=240))
        assert features.highlight_ratio == 0.0

@pytest.mark.unit
class TestHueHist:
    """verify hue histogram normalisation, shape, and peak position."""

    def test_length_matches_hue_bins(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert len(features.hue_hist) == 180

    def test_normalised_sum_close_to_one(self, extractor):
        # hue=30 is well away from the 0/179 boundary; no edge effects
        features = extractor.extract(make_preprocess_result(hue=30))
        assert abs(features.hue_hist.sum() - 1.0) < 0.01

    def test_peak_at_correct_hue_bin(self, extractor):
        # uniform colour at hue=30: argmax should land near bin 30
        features = extractor.extract(make_preprocess_result(hue=30))
        peak = int(np.argmax(features.hue_hist))
        assert abs(peak - 30) <= 2

    def test_all_bins_nonnegative(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert np.all(features.hue_hist >= 0)

    def test_no_smoothing_when_sigma_zero(self, tmp_path):
        data = Config().as_dict()
        data["features"]["hue_smooth_sigma"] = 0
        ext = FeatureExtractor(Config(write_config(data, tmp_path)))
        features = ext.extract(make_preprocess_result(hue=30))
        # without smoothing the entire mass is at a single bin
        assert features.hue_hist[30] > 0.95

@pytest.mark.unit
class TestHuePeakWidth:
    """verify dominant hue cluster width including circular wrap-around."""

    def test_zero_histogram_returns_zero(self, extractor):
        assert extractor._compute_hue_peak_width(np.zeros(180)) == 0

    def test_single_bin_returns_one(self, extractor):
        hist = np.zeros(180)
        hist[90] = 1.0
        # no adjacent bins above 15% of 1.0, so width must be 1
        assert extractor._compute_hue_peak_width(hist) == 1

    def test_known_width_without_wrap(self, extractor):
        hist = np.zeros(180)
        # bins 88..92 above threshold (5 bins centred at 90)
        hist[88:93] = 0.5
        hist[90] = 1.0
        width = extractor._compute_hue_peak_width(hist)
        assert width == 5

    def test_peak_width_wraps_at_hue_zero(self, extractor):
        # red at hue=0 must give comparable width to a mid-range hue
        f_boundary = extractor.extract(make_preprocess_result(hue=0))
        f_midrange = extractor.extract(make_preprocess_result(hue=90))
        assert abs(f_boundary.hue_peak_width - f_midrange.hue_peak_width) <= 5

@pytest.mark.unit
class TestTextureVariance:
    """verify Laplacian-based texture measurement."""

    def test_smooth_surface_low_variance(self, extractor):
        # uniform colour circle has minimal edges inside the mask
        features = extractor.extract(make_preprocess_result())
        assert features.texture_variance < 500

    def test_nonnegative(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert features.texture_variance >= 0.0

    def test_raises_on_multichannel_gray(self, extractor):
        result = dataclasses.replace(make_preprocess_result(), gray=np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="2-D"):
            extractor.extract(result)

@pytest.mark.unit
class TestCircularity:
    """verify 4*pi*A/P^2 circularity computation."""

    def test_circle_near_one(self, extractor):
        features = extractor.extract(make_preprocess_result(radius=30))
        assert features.circularity > 0.85

    def test_circularity_bounded(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert 0.0 <= features.circularity <= 1.0

    def test_degenerate_single_point_returns_zero(self, extractor):
        result = dataclasses.replace(make_preprocess_result(), contour=np.array([[[50, 50]]], dtype=np.int32))
        features = extractor.extract(result)
        assert features.circularity == 0.0

@pytest.mark.unit
class TestAspectRatio:
    """verify bounding box width/height ratio."""

    def test_circle_near_one(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert 0.85 <= features.aspect_ratio <= 1.15

    def test_wide_rectangle_above_one(self, extractor):
        # horizontal rectangle 80x20, aspect ratio ~4
        result = dataclasses.replace(
            make_preprocess_result(),
            contour=np.array([[[10, 40]], [[90, 40]], [[90, 60]], [[10, 60]]], dtype=np.int32),
        )
        features = extractor.extract(result)
        assert features.aspect_ratio > 1.5

    def test_tall_rectangle_normalised(self, extractor):
        # vertical rectangle 20x80 -- normalised ratio = 80/20 = 4.0
        result = dataclasses.replace(
            make_preprocess_result(),
            contour=np.array([[[40, 10]], [[60, 10]], [[60, 90]], [[40, 90]]], dtype=np.int32),
        )
        features = extractor.extract(result)
        assert features.aspect_ratio > 1.5

@pytest.mark.unit
class TestSolidity:
    """verify contour area / convex hull area ratio."""

    def test_convex_circle_near_one(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert features.solidity > 0.95

    def test_solidity_bounded(self, extractor):
        features = extractor.extract(make_preprocess_result())
        assert 0.0 <= features.solidity <= 1.0

    def test_degenerate_collinear_contour_returns_zero(self, extractor):
        # collinear points: convex hull is a line, hull area = 0
        result = dataclasses.replace(
            make_preprocess_result(),
            contour=np.array([[[0, 0]], [[50, 0]], [[100, 0]]], dtype=np.int32),
        )
        features = extractor.extract(result)
        assert features.solidity == 0.0

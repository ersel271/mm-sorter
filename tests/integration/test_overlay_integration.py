# tests/integration/test_overlay_integration.py

import numpy as np
import pytest

from src.ui import Overlay
from src.ui.panel import PANEL_W
from utils.metrics import RunningMetrics
from tests.helpers.image_helpers import draw_saturated_circle, make_frame
from tests.helpers.overlay_helpers import NOT_FOUND

@pytest.mark.smoke
class TestOverlayFullPipeline:
    """verify render works end-to-end with real pipeline outputs."""

    def test_full_pipeline_produces_valid_ndarray(self, default_cfg, prep, extractor, classifier) -> None:
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        assert result.found
        features = extractor.extract(result)
        decision = classifier.classify(features)
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, features, decision)
        assert isinstance(out, np.ndarray)

    def test_output_dtype_uint8(self, default_cfg, prep, extractor, classifier) -> None:
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)
        decision = classifier.classify(features)
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, features, decision)
        assert out is not None
        assert out.dtype == np.uint8

    def test_output_preserves_frame_dimensions(self, default_cfg, prep, extractor, classifier) -> None:
        frame = draw_saturated_circle(make_frame(640, 480), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)
        decision = classifier.classify(features)
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, features, decision)
        assert out is not None
        assert out.shape[:2] == (480, 640 + PANEL_W)

    def test_pipeline_with_populated_metrics(self, default_cfg, prep, extractor, classifier) -> None:
        from utils.events import VisionEvent
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        features = extractor.extract(result)
        decision = classifier.classify(features)
        metrics = RunningMetrics()
        event = VisionEvent(
            ts_wall="2024-01-01T00:00:00", ts_mono=0.0,
            object_id=1, class_id=int(decision.label), class_name="Red",
            confidence=decision.confidence, low_confidence=False,
            centroid_x=960, centroid_y=540, area=1000.0,
            sat_mean=180.0, highlight_ratio=0.05, hue_peak_width=20,
            texture_variance=50.0, circularity=0.90,
            aspect_ratio=1.1, solidity=0.95, frame_ms=10.0,
        )
        metrics.update(event)
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, features, decision)
        assert out is not None

@pytest.mark.regression
class TestOverlayNotFoundPipeline:
    """verify render handles not-found frames correctly."""

    def test_black_frame_preprocess_overlay_renders(self, default_cfg, prep) -> None:
        frame = make_frame()
        result = prep.process(frame)
        assert not result.found
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, None, None)
        assert isinstance(out, np.ndarray)

    def test_not_found_result_directly_renders(self, default_cfg) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = NOT_FOUND
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, None, None)
        assert isinstance(out, np.ndarray)

    def test_disabled_overlay_returns_none(self, overlay_disabled) -> None:
        frame = make_frame()
        result = NOT_FOUND
        out = overlay_disabled.render(frame, result, None, None)
        assert out is None

    def test_decision_none_features_none_renders(self, default_cfg, prep) -> None:
        frame = draw_saturated_circle(make_frame(), radius=60)
        result = prep.process(frame)
        ov = Overlay(default_cfg, RunningMetrics())
        out = ov.render(frame, result, None, None)
        assert isinstance(out, np.ndarray)

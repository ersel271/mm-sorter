# tests/integration/test_ground_truth_integration.py

import pytest

from config.constants import COLOUR_NAMES
from utils.plot import PlotData
from utils.ground_truth import GTSession, load_gt
from tests.helpers.ground_truth_helpers import write_gt

@pytest.mark.smoke
class TestGTLoadAndAccumulate:
    """verify load_gt output integrates correctly with GTSession recording and plot conversion"""

    def test_load_then_record_matches(self, tmp_path):
        path = write_gt(tmp_path, ["red", "green", "blue"])
        labels = load_gt(str(path))
        acc = GTSession()
        for true_cls in labels.values():
            acc.record(true_cls=true_cls, pred_cls=true_cls, confidence=0.9)
        assert acc.pairs == [(1, 1), (2, 2), (3, 3)]

    def test_full_workflow_to_plot_data(self, tmp_path):
        path = write_gt(tmp_path, ["red", "green", "1"])
        labels = load_gt(str(path))
        acc = GTSession()
        preds = [2, 3, 4]
        confs = [0.9, 0.8, 0.7]
        for (_oid, true_cls), pred, conf in zip(labels.items(), preds, confs, strict=True):
            acc.record(true_cls=true_cls, pred_cls=pred, confidence=conf)
        class_names = list(COLOUR_NAMES.values())
        result = acc.to_plot_data(class_names)
        assert isinstance(result, PlotData)
        assert result.ground_truth == [1, 2, 1]
        assert result.predictions == preds
        assert result.class_names == class_names

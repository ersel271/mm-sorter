# tests/unit/test_plot.py

import pytest

from pathlib import Path

from config.constants import COLOUR_NAMES
from utils.plot import PlotData, PlotQueueWorker, generate_dashboard

_CLASS_NAMES = [v.lower().replace("&", "") for v in COLOUR_NAMES.values()]

@pytest.mark.smoke
class TestPlotData:
    """verify PlotData dataclass construction"""

    def test_minimal_construction(self):
        data = PlotData(predictions=[0, 1], confidences=[0.9, 0.8],
                        class_names=_CLASS_NAMES)
        assert data.ground_truth is None

    def test_with_ground_truth(self):
        data = PlotData(predictions=[0], confidences=[0.9],
                        class_names=_CLASS_NAMES, ground_truth=[0])
        assert data.ground_truth == [0]

@pytest.mark.visual
class TestGenerateDashboard:
    """verify generate_dashboard saves files to disk"""

    def _make_data(self, with_gt: bool = False) -> PlotData:
        preds = [1, 2, 3, 1, 2]
        confs = [0.9, 0.8, 0.7, 0.85, 0.75]
        gt = preds if with_gt else None
        return PlotData(predictions=preds, confidences=confs,
                        class_names=_CLASS_NAMES, ground_truth=gt)

    def test_no_gt_files_created(self, tmp_path: Path):
        generate_dashboard(self._make_data(), tmp_path, "sess")
        names = {p.name for p in tmp_path.iterdir()}
        assert "plot_sess_class_distribution.png" in names
        assert "plot_sess_confidence_hist.png" in names
        assert "plot_sess_confidence_by_class.png" in names

    def test_gt_files_created(self, tmp_path: Path):
        generate_dashboard(self._make_data(with_gt=True), tmp_path, "sess")
        names = {p.name for p in tmp_path.iterdir()}
        assert "plot_sess_class_distribution.png" in names
        assert "plot_sess_confidence_hist.png" in names
        assert "plot_sess_confidence_by_class.png" in names
        assert "plot_sess_confusion_matrix.png" in names
        assert "plot_sess_per_class_metrics.png" in names
        assert "plot_sess_error_breakdown.png" in names
        assert "plot_sess_confidence_vs_accuracy.png" in names

@pytest.mark.visual
class TestPlotQueueWorker:
    """verify PlotQueueWorker enqueues, drops, and drains correctly"""

    def _make_data(self) -> PlotData:
        return PlotData(
            predictions=[1, 2, 3],
            confidences=[0.9, 0.8, 0.7],
            class_names=_CLASS_NAMES,
        )

    def test_enqueue_saves_file(self, tmp_path: Path):
        worker = PlotQueueWorker(tmp_path)
        worker.enqueue(self._make_data())
        worker.stop()
        assert any(tmp_path.iterdir())

    def test_enqueue_when_full_does_not_raise(self, tmp_path: Path):
        worker = PlotQueueWorker(tmp_path)
        worker.enqueue(self._make_data())
        worker.enqueue(self._make_data())  # queue full, should drop silently
        worker.stop()

    def test_dropped_counter_increments_on_full_queue(self, tmp_path: Path):
        worker = PlotQueueWorker(tmp_path)
        worker.enqueue(self._make_data())
        worker.enqueue(self._make_data())  # dropped
        assert worker.dropped == 1
        worker.stop()

    def test_stop_drains_queue(self, tmp_path: Path):
        worker = PlotQueueWorker(tmp_path)
        worker.enqueue(self._make_data())
        worker.stop()
        assert any(p.suffix == ".png" for p in tmp_path.iterdir())

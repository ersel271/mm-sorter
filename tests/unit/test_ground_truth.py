# tests/unit/test_ground_truth.py

import pytest

from utils.plot import PlotData
from utils.ground_truth import GTSession, load_gt
from tests.helpers.ground_truth_helpers import write_gt

@pytest.mark.smoke
class TestLoadGtTokens:
    """verify load_gt parses integer ids and colour name tokens"""

    def test_integer_ids(self, tmp_path):
        path = write_gt(tmp_path, ["1", "2", "3"])
        assert load_gt(str(path)) == {1: 1, 2: 2, 3: 3}

    def test_colour_names(self, tmp_path):
        path = write_gt(tmp_path, ["red", "green", "blue"])
        assert load_gt(str(path)) == {1: 1, 2: 2, 3: 3}

    def test_case_insensitive(self, tmp_path):
        path = write_gt(tmp_path, ["RED", "Green", "BLUE"])
        assert load_gt(str(path)) == {1: 1, 2: 2, 3: 3}

    def test_mixed_tokens(self, tmp_path):
        path = write_gt(tmp_path, ["1", "green"])
        assert load_gt(str(path)) == {1: 1, 2: 2}

    def test_zero_id(self, tmp_path):
        path = write_gt(tmp_path, ["0"])
        assert load_gt(str(path)) == {1: 0}

class TestLoadGtSkips:
    """verify blank lines and comment lines are not counted as objects"""

    def test_skips_blank_lines(self, tmp_path):
        path = write_gt(tmp_path, ["1", "", "2"])
        assert load_gt(str(path)) == {1: 1, 2: 2}

    def test_skips_comment_lines(self, tmp_path):
        path = write_gt(tmp_path, ["1", "# a comment", "2"])
        assert load_gt(str(path)) == {1: 1, 2: 2}

    def test_blank_and_comment_mixed(self, tmp_path):
        path = write_gt(tmp_path, ["", "# comment", "", "1"])
        assert load_gt(str(path)) == {1: 1}

class TestLoadGtLineMapping:
    """verify object ids are assigned in arrival order starting from 1"""

    def test_one_indexed(self, tmp_path):
        path = write_gt(tmp_path, ["red"])
        result = load_gt(str(path))
        assert 1 in result
        assert 0 not in result

    def test_sequential_ids(self, tmp_path):
        path = write_gt(tmp_path, ["red", "green", "blue"])
        assert list(load_gt(str(path)).keys()) == [1, 2, 3]

@pytest.mark.smoke
@pytest.mark.regression
class TestLoadGtErrors:
    """verify load_gt raises on bad input"""

    def test_unknown_token_raises(self, tmp_path):
        path = write_gt(tmp_path, ["notacolour"])
        with pytest.raises(ValueError, match="unknown ground-truth token"):
            load_gt(str(path))

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(OSError):
            load_gt(str(tmp_path / "missing.txt"))

@pytest.mark.smoke
class TestGTSessionRecord:
    """verify GTSession.record appends pairs and confidences"""

    def test_record_appends_pair(self):
        acc = GTSession()
        acc.record(1, 2, 0.9)
        assert acc.pairs == [(1, 2)]

    def test_record_appends_confidence(self):
        acc = GTSession()
        acc.record(1, 2, 0.9)
        assert acc.confidences == [0.9]

    def test_multiple_records(self):
        acc = GTSession()
        acc.record(1, 2, 0.9)
        acc.record(3, 4, 0.7)
        acc.record(5, 6, 0.5)
        assert len(acc.pairs) == 3
        assert len(acc.confidences) == 3

@pytest.mark.smoke
class TestGTSessionLen:
    """verify __len__ returns the number of recorded pairs"""

    def test_empty_is_zero(self):
        assert len(GTSession()) == 0

    def test_len_matches_records(self):
        acc = GTSession()
        acc.record(1, 1, 0.8)
        acc.record(2, 2, 0.9)
        assert len(acc) == 2

class TestGTSessionToPlotData:
    """verify to_plot_data builds a correctly populated PlotData"""

    def _acc_with_records(self) -> GTSession:
        acc = GTSession()
        acc.record(1, 2, 0.9)
        acc.record(3, 4, 0.7)
        return acc

    def test_returns_plot_data(self):
        acc = self._acc_with_records()
        assert isinstance(acc.to_plot_data(["a", "b"]), PlotData)

    def test_predictions_match(self):
        acc = self._acc_with_records()
        result = acc.to_plot_data(["a"])
        assert result.predictions == [2, 4]

    def test_ground_truth_matches(self):
        acc = self._acc_with_records()
        result = acc.to_plot_data(["a"])
        assert result.ground_truth == [1, 3]

    def test_confidences_match(self):
        acc = self._acc_with_records()
        result = acc.to_plot_data(["a"])
        assert result.confidences == pytest.approx([0.9, 0.7])

    def test_class_names_passed(self):
        acc = self._acc_with_records()
        names = ["x", "y", "z"]
        assert acc.to_plot_data(names).class_names == names

# tests/unit/test_inject.py

import itertools

import pytest
import numpy as np

from utils.inject import build_inject_cycle
from tests.helpers.inject_helpers import make_inject_folder

@pytest.mark.smoke
@pytest.mark.regression
class TestBuildInjectCycleErrors:
    """verify build_inject_cycle raises when the folder has no images"""

    def test_empty_folder_raises(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        gen = build_inject_cycle(str(folder), repeat=1)
        with pytest.raises(RuntimeError, match="injection folder is empty"):
            next(gen)

@pytest.mark.smoke
class TestBuildInjectCycleFrames:
    """verify the iterator yields real image frames followed by black gap frames"""

    def test_yields_image_frames(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=1)
        frame = next(gen)
        assert frame.max() > 0

    def test_gap_frame_is_black(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=1)
        next(gen)
        gap = next(gen)
        assert np.all(gap == 0)

    def test_gap_frame_matches_shape(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=1)
        img = next(gen)
        gap = next(gen)
        assert gap.shape == img.shape

class TestBuildInjectCycleRepeat:
    """verify each image is yielded repeat times then followed by a gap frame"""

    def test_repeat_one(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=1)
        img = next(gen)
        gap = next(gen)
        assert img.max() > 0
        assert np.all(gap == 0)

    def test_repeat_three(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=3)
        frames = list(itertools.islice(gen, 4))
        assert all(f.max() > 0 for f in frames[:3])
        assert np.all(frames[3] == 0)

    def test_repeat_followed_by_gap(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=1)
        gen = build_inject_cycle(str(folder), repeat=2)
        next(gen)
        next(gen)
        gap = next(gen)
        assert np.all(gap == 0)

class TestBuildInjectCycleCycles:
    """verify the iterator cycles through images infinitely"""

    def test_cycles_infinitely(self, tmp_path):
        folder = make_inject_folder(tmp_path, n_images=2)
        gen = build_inject_cycle(str(folder), repeat=1)
        # 2 images * (1 repeat + 1 gap) = 4 frames per cycle, take 3 cycles
        frames = list(itertools.islice(gen, 12))
        assert len(frames) == 12

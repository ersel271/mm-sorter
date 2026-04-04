# utils/inject.py
"""
Frame injection from a folder of images for camera-free pipeline runs.

Usage:
    frames = build_inject_cycle("data/samples", repeat=3)
"""

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# recognised image file extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def build_inject_cycle(folder: str, repeat: int) -> Iterator[np.ndarray]:
    """
    yield frames from a sorted folder, repeating each image `repeat` times
    and inserting a black gap frame between objects to reset the detection counter
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise RuntimeError(f"injection folder does not exist or is not a directory: {folder!r}")

    paths = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"injection folder is empty: {folder!r}")

    log.info("inject-from: %d images in %s, each repeated %d times", len(paths), folder, repeat)
    for path in itertools.cycle(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("could not read image %s, skipping", path)
            continue
        for _ in range(repeat):
            yield frame
        # black gap frame resets found_count so the next image is a new object
        yield np.zeros_like(frame)

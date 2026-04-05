# tests/helpers/inject_helpers.py

from pathlib import Path

import cv2

from tests.helpers.image_helpers import make_demo_frame

def make_inject_folder(tmp_path: Path, n_images: int = 3) -> Path:
    """create a folder of synthetic M&M PNG frames for inject-from tests."""
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(n_images):
        frame = make_demo_frame(colour_idx=i)
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), frame)
    return folder

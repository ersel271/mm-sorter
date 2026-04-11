# tools/cal/_common.py
"""
Shared helpers for tools/cal/ standalone scripts.

Provides camera setup, session state, checkpoint writing, image processing,
logging, and geometry metric helpers.
"""

import datetime
import json
import logging
import os

import cv2
import numpy as np

# log directory relative to this file
_LOG_DIR = os.path.join(os.path.dirname(__file__), "_log")

def _log_path(name: str) -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_LOG_DIR, f"{name}_{stamp}.log")

def get_logger(name: str) -> logging.Logger:
    """return a logger that writes to stderr and a dated file under tools/_log/."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(_log_path(name))
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log

def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """open a MJPG camera at the given resolution and return the capture object."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise OSError(f"cannot open camera device {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def apply_focus(cap: cv2.VideoCapture, value: int) -> None:
    """disable autofocus and set the focus register to value."""
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, value)

def apply_wb_temperature(cap: cv2.VideoCapture, value: int) -> None:
    """disable auto white balance and set the temperature register to value."""
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, value)

def save_checkpoint(path: str, title: str, **fields) -> None:
    """append a titled block to the checkpoint file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# === {title} "
    # pad header with = signs so the timestamp ends around column 52
    pad = max(0, 52 - len(header) - len(ts))
    header = header + "=" * pad + " " + ts
    lines = ["\n" + header]
    for key, val in fields.items():
        lines.append(f"{key + ':':<16}{val}")
    lines.append("")
    block = "\n".join(lines) + "\n"
    with open(path, "a") as f:
        f.write(block)

# session file path relative to the tools/ directory
_SESSION_PATH = os.path.join(os.path.dirname(__file__), ".session.json")

def load_session(path: str | None = None) -> dict:
    """read .session.json and return its contents; return {} if the file is absent."""
    p = path or _SESSION_PATH
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def save_session(updates: dict, path: str | None = None) -> None:
    """merge updates into .session.json, creating the file if necessary."""
    p = path or _SESSION_PATH
    data = load_session(p)
    data.update(updates)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def extract_mask(
    frame: np.ndarray,
    blur_k: int,
    sat_thresh: int,
    morph_k: int,
    erode: int,
    dilate: int,
) -> np.ndarray:
    """
    return a binary mask of saturated foreground pixels.

    pipeline: gaussian blur, BGR to HSV, saturation threshold,
    ellipse morphology (erode then dilate).
    """
    blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.erode(mask, kernel, iterations=erode)
    mask = cv2.dilate(mask, kernel, iterations=dilate)
    return mask

def compute_circularity(area: float, perimeter: float) -> float:
    """
    measures how close the contour shape is to a circle.

    circularity = 4 * pi * area / perimeter**2

    values near 1 indicate round shapes. lower values indicate
    irregular or elongated objects.
    """
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))

def compute_aspect_ratio(w: float, h: float) -> float:
    """
    normalised bounding box aspect ratio: max(w, h) / min(w, h).

    always >= 1.0 regardless of object orientation.
    values near 1.0 indicate square-like objects.
    """
    if min(w, h) == 0:
        return 0.0
    return float(max(w, h) / min(w, h))

def compute_solidity(contour: np.ndarray) -> float:
    """
    ratio of contour area to convex hull area.

    solidity = contour_area / convex_hull_area

    values near 1 indicate compact shapes. irregular or concave
    contours produce lower solidity scores.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area / hull_area)

def compute_texture(crop: np.ndarray, mask: np.ndarray) -> float:
    """
    laplacian variance of greyscale pixels inside the mask.

    variance = Var(Laplacian(grey)[mask])

    smooth candy surfaces produce low variance while textured
    objects produce higher values.
    """
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return float(laplacian[mask > 0].var())

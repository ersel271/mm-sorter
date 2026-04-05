# tests/helpers/image_helpers.py

import cv2
import numpy as np

def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(b), int(g), int(r))

# M&M colours (cv2 HSV: H 0-179, S/V 0-255)
_MM_BGR = [
    _hsv_to_bgr(  0, 220, 180),  # red
    _hsv_to_bgr( 60, 220, 180),  # green
    _hsv_to_bgr(120, 220, 180),  # blue
    _hsv_to_bgr( 30, 220, 200),  # yellow
    _hsv_to_bgr( 15, 230, 200),  # orange
    _hsv_to_bgr( 15,  90,  90),  # brown
]

def _draw_mm_circle(frame: np.ndarray, cx: int, cy: int, bgr: tuple[int, int, int]) -> None:
    r = 50
    cv2.circle(frame, (cx, cy), r, bgr, -1)
    cv2.circle(frame, (cx - r // 4, cy - r // 4), r // 5, (200, 210, 220), -1)
    cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2)

_REJECT_BGR = _hsv_to_bgr(90, 220, 200)

def _draw_ellipse(
    frame: np.ndarray,
    cx: int,
    cy: int,
    axes: tuple[int, int] = (90, 22),
) -> None:
    # elongated ellipse, bad aspect ratio (AR ~4.0 with default axes)
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, _MM_BGR[0], -1)
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (0, 0, 0), 2)

def _draw_noisy_circle(
    frame: np.ndarray,
    cx: int,
    cy: int,
    noise_std: float = 90.0,
) -> None:
    # pixel noise on surface, high texture (Laplacian variance >> 500 with default noise_std)
    r = 50
    _draw_mm_circle(frame, cx, cy, _MM_BGR[0])
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r - 3, 255, -1)
    noise = np.random.normal(0, noise_std, frame.shape).astype(np.int16)
    tmp = frame.astype(np.int16)
    tmp[mask > 0] += noise[mask > 0]
    np.clip(tmp, 0, 255, out=tmp)
    frame[:] = tmp.astype(np.uint8)

def _draw_star(
    frame: np.ndarray,
    cx: int,
    cy: int,
    outer_r: int = 50,
    inner_r: int = 20,
) -> None:
    # 5-pointed star, low circularity (~0.2 with default radii)
    n = 5
    pts = [
        [
            int(cx + (outer_r if i % 2 == 0 else inner_r) * np.cos(np.pi * i / n - np.pi / 2)),
            int(cy + (outer_r if i % 2 == 0 else inner_r) * np.sin(np.pi * i / n - np.pi / 2)),
        ]
        for i in range(2 * n)
    ]
    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(frame, [poly], _REJECT_BGR)
    cv2.polylines(frame, [poly], True, (0, 0, 0), 2)

_NON_MM_DRAWERS = {
    "ellipse": _draw_ellipse,
    "noisy_circle": _draw_noisy_circle,
    "star": _draw_star,
}

def make_non_mm_frame(
    shape: str = "ellipse",
    width: int = 640,
    height: int = 480,
    **kwargs,
) -> np.ndarray:
    """synthetic BGR frame with a non-M&M shape drawn at centre.

    shape selects the rejection reason:
        "ellipse"      - bad aspect ratio,  accepts: axes=(int, int)
        "noisy_circle" - high texture,      accepts: noise_std=float
        "star"         - low circularity,   accepts: outer_r=int, inner_r=int
    """
    if shape not in _NON_MM_DRAWERS:
        raise ValueError(f"unknown shape {shape!r}, choose from {list(_NON_MM_DRAWERS)}")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _NON_MM_DRAWERS[shape](frame, width // 2, height // 2, **kwargs)
    return frame

def make_demo_frame(
    colour_idx: int = 0,
    bgr: tuple[int, int, int] | None = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """synthetic BGR frame with a single M&M circle drawn at centre.

    bgr overrides colour_idx when provided, allowing tests to supply
    explicit colour values independent of the _MM_BGR defaults.
    """
    colour = bgr if bgr is not None else _MM_BGR[colour_idx % len(_MM_BGR)]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_mm_circle(frame, width // 2, height // 2, colour)
    return frame

def make_frame(width=1920, height=1080) -> np.ndarray:
    """plain black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def draw_circle(frame, centre=None, radius=50, colour_bgr=(0, 0, 255), sat=200):
    """
    draw a filled circle with a given saturation onto a black frame.
    uses HSV to control saturation precisely, then converts back to BGR.
    """
    if centre is None:
        h, w = frame.shape[:2]
        centre = (w // 2, h // 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bgr_ref = np.uint8([[colour_bgr]])
    hsv_ref = cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2HSV)[0][0]

    colour_hsv = (int(hsv_ref[0]), sat, int(hsv_ref[2]))
    cv2.circle(hsv, centre, radius, colour_hsv, -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def draw_saturated_circle(frame, centre=None, radius=50):
    """shorthand: bright circle with high saturation drawn onto existing frame."""
    if centre is None:
        h, w = frame.shape[:2]
        centre = (w // 2, h // 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.circle(hsv, centre, radius, (0, 200, 200), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

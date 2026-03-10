# tests/helpers/image_helpers.py

import cv2
import numpy as np

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

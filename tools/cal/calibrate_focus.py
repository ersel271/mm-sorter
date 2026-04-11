#!/usr/bin/env python3
# tools/cal/calibrate_focus.py
"""
Interactive focus calibration tool for the USB camera.

Opens a live camera feed and lets you adjust the manual focus value
using keyboard controls. A laplacian-based sharpness score is shown
on screen to help find the optimal focus point.

The tool is standalone and does not depend on any project modules.
OpenCV and optionally PyYAML are required.

Usage:
    python calibrate_focus.py [--device DEVICE] [--config CONFIG]
                              [--no-session]

Procedure:
    1. Place a printed reference pattern (text, checkerboard, or ruler)
       flat on the inspection surface at the working distance.
    2. Run this tool.
    3. Use f/g for coarse adjustment until the image looks roughly sharp.
    4. Switch to d/h for fine adjustment and watch the sharpness score.
    5. Find the focus value where sharpness peaks, go past it and come
       back to confirm the maximum. The exact number depends on content,
       so focus on finding the peak rather than hitting a specific target.
    6. Press 's' to save.
    7. Copy the output into your configuration.

Controls:
    f / g   coarse focus adjustment (default +-10)
    d / h   fine focus adjustment (default +-1)
    a       trigger autofocus once, then lock and read the value
    s       save current values to checkpoint file and log to terminal
    q       quit

Output:
    Logs the final focus value to the terminal and tools/_log/.
    Appends focus and sharpness to calibration_checkpoint.txt.

NOTE: The camera must support UVC focus control. If set_focus
calls return False, check v4l2-ctl --list-ctrls for your device.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import _common

log = _common.get_logger("calibrate_focus")

# override these if the defaults feel too fast or too slow
COARSE_STEP = 10
FINE_STEP = 1
FOCUS_MIN = 1
FOCUS_MAX = 1023

# camera defaults, override with --config
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

CHECKPOINT_FILE = "calibration_checkpoint.txt"
PREVIEW_SCALE = 0.85

def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def sharpness(frame):
    """laplacian variance of a greyscale frame. higher = sharper"""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    return lap.var()

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def main():
    parser = argparse.ArgumentParser(description="interactive focus calibration")
    parser.add_argument("--device", type=int, default=2, help="camera device index")
    parser.add_argument("--config", default=None, help="path to config yaml (optional)")
    parser.add_argument("--no-session", action="store_true", help="skip session.json read/write")
    args = parser.parse_args()

    device = args.device
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT
    fps = CAMERA_FPS
    focus_default = 200

    if args.config:
        cfg = _load_config(args.config)
        cam = cfg.get("camera", {})
        device = cam.get("device", device)
        width = cam.get("width", width)
        height = cam.get("height", height)
        fps = cam.get("fps", fps)
        focus_default = cam.get("focus", focus_default)

    try:
        cap = _common.open_camera(device, width, height, fps)
    except OSError as e:
        log.info(f"error: {e}")
        return 1

    # disable autofocus, read current focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    focus = int(cap.get(cv2.CAP_PROP_FOCUS))
    if focus < FOCUS_MIN:
        focus = focus_default
    cap.set(cv2.CAP_PROP_FOCUS, focus)

    log.info("focus calibration started")
    log.info(f"  device: {device}")
    log.info(f"  initial focus: {focus}")
    log.info(f"  controls: f/g=coarse(+-{COARSE_STEP})  d/h=fine(+-{FINE_STEP})  a=autofocus  s=save  q=quit")
    log.info("")

    help_text = f"f/g: +-{COARSE_STEP} | d/h: +-{FINE_STEP} | a: autofocus | s: save | q: quit"

    focus_applied = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # apply focus on the first live frame (pre-stream sets are ignored).
        # nudge first so the driver sends a real motor command even if the
        if not focus_applied:
            nudge = focus - 1 if focus > FOCUS_MIN else focus + 1
            cap.set(cv2.CAP_PROP_FOCUS, nudge)
            cap.read()  # let motor reach nudge position before sending target
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            focus_applied = True

        sharp = sharpness(frame)

        # draw overlay
        display = frame.copy()
        cv2.putText(display, f"focus: {focus}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"sharpness: {sharp:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, help_text, (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # crosshair at centre
        cy, cx = display.shape[0] // 2, display.shape[1] // 2
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 1)

        display = cv2.resize(display, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
        cv2.imshow("Focus Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("f"):
            focus = clamp(focus + COARSE_STEP, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)

        elif key == ord("g"):
            focus = clamp(focus - COARSE_STEP, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)

        elif key == ord("d"):
            focus = clamp(focus + FINE_STEP, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)

        elif key == ord("h"):
            focus = clamp(focus - FINE_STEP, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)

        elif key == ord("a"):
            log.info("  autofocus triggered, waiting 2 seconds...")
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            # let autofocus settle
            for _ in range(60):
                cap.read()
                cv2.waitKey(33)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            focus = int(cap.get(cv2.CAP_PROP_FOCUS))
            focus = clamp(focus, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            log.info(f"  autofocus locked at: {focus}")

        elif key == ord("s"):
            _common.save_checkpoint(
                CHECKPOINT_FILE, "focus",
                focus=focus,
                sharpness=f"{sharp:.1f}",
            )
            log.info(f"  saved to {CHECKPOINT_FILE}")
            log.info(f"  focus={focus}  sharpness={sharp:.1f}")

    cap.release()
    cv2.destroyAllWindows()

    if not args.no_session:
        _common.save_session({"focus": focus})
        log.info(f"  session updated: focus={focus}")

    log.info(f"final focus value: {focus}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# tools/cal/calibrate_wb.py
"""
Interactive white balance calibration tool for the USB camera.

Opens a live camera feed and lets you adjust the white balance
temperature using keyboard controls. A centre ROI patch shows the
average BGR values to help verify neutral white balance against
a white reference card.

The tool is standalone and does not depend on any project modules.
OpenCV, NumPy, and optionally PyYAML are required.

Usage:
    python calibrate_wb.py [--device DEVICE] [--config CONFIG]
                           [--no-session]

Procedure:
    1. Place a white reference card (plain white paper works)
       flat on the inspection surface under the LED illumination.
    2. Run this tool.
    3. Press 'w' to trigger auto white balance and wait for it
       to stabilise. The average BGR of the centre patch should
       approach roughly equal values (e.g. 200, 200, 200).
    4. Fine-tune manually with t/y and r/u if needed.
    5. Press 's' to save the temperature value.
    6. Copy the output into your configuration.

Controls:
    w       trigger auto white balance, wait, then lock and read
    t / y   coarse temperature adjustment (default +-100)
    r / u   fine temperature adjustment (default +-25)
    s       save current values to checkpoint file and log to terminal
    q       quit

Output:
    Logs the white balance temperature to the terminal and tools/_log/.
    Appends values to calibration_checkpoint.txt.

NOTE: Not all cameras expose WB_TEMPERATURE via UVC. If the temperature
reads as 0 or does not change, use v4l2-ctl to verify support:
    v4l2-ctl -d /dev/video2 --list-ctrls | grep white_balance
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _common

log = _common.get_logger("calibrate_wb")

# override these if the defaults feel too fast or too slow
COARSE_STEP = 100
FINE_STEP = 25
WB_TEMP_MIN = 2800
WB_TEMP_MAX = 6500

# camera defaults, override with --config
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

ROI_SIZE = 100

CHECKPOINT_FILE = "calibration_checkpoint.txt"
PREVIEW_SCALE = 0.85


def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def centre_roi_stats(frame):
    """average BGR of a square patch at the frame centre."""
    h, w = frame.shape[:2]
    half = ROI_SIZE // 2
    cx, cy = w // 2, h // 2
    roi = frame[cy - half:cy + half, cx - half:cx + half]
    mean = roi.mean(axis=(0, 1))
    return mean, (cx - half, cy - half, cx + half, cy + half)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def main():
    parser = argparse.ArgumentParser(description="interactive white balance calibration")
    parser.add_argument("--device", type=int, default=2, help="camera device index")
    parser.add_argument("--config", default=None, help="path to config yaml (optional)")
    parser.add_argument("--no-session", action="store_true", help="skip session.json read/write")
    args = parser.parse_args()

    device = args.device
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT
    fps = CAMERA_FPS
    wb_default = 4600

    if args.config:
        cfg = _load_config(args.config)
        cam = cfg.get("camera", {})
        device = cam.get("device", device)
        width = cam.get("width", width)
        height = cam.get("height", height)
        fps = cam.get("fps", fps)
        wb_default = cam.get("wb_temperature", wb_default)

    try:
        cap = _common.open_camera(device, width, height, fps)
    except OSError as e:
        log.info(f"error: {e}")
        return 1

    # apply focus from previous session if available
    if not args.no_session:
        session = _common.load_session()
        if "focus" in session:
            _common.apply_focus(cap, session["focus"])
            log.info(f"  session focus applied: {session['focus']}")

    # start with auto WB, read temperature as initial value
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    temp = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
    if temp < WB_TEMP_MIN:
        temp = wb_default
    wb_auto = True

    log.info("white balance calibration started")
    log.info(f"  device: {device}")
    log.info(f"  initial temperature: {temp}K")
    log.info(f"  place a white reference card under the camera")
    log.info(f"  controls: w=auto-lock  t/y=coarse(+-{COARSE_STEP})  r/u=fine(+-{FINE_STEP})  s=save  q=quit")
    log.info("")

    help_text = f"w: auto-lock | t/y: +-{COARSE_STEP} | r/u: +-{FINE_STEP} | s: save | q: quit"

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        avg_bgr, (rx1, ry1, rx2, ry2) = centre_roi_stats(frame)

        # draw overlay
        display = frame.copy()

        mode_str = "AUTO" if wb_auto else "MANUAL"
        cv2.putText(display, f"WB: {mode_str}  temp: {temp}K", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"avg BGR: ({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # colour balance indicator
        spread = max(avg_bgr) - min(avg_bgr)
        if spread < 15:
            balance_str = "BALANCED"
            balance_colour = (0, 255, 0)
        elif spread < 30:
            balance_str = "CLOSE"
            balance_colour = (0, 200, 255)
        else:
            balance_str = "UNBALANCED"
            balance_colour = (0, 0, 255)
        cv2.putText(display, balance_str, (20, 120),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, balance_colour, 2)

        # ROI rectangle
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        cv2.putText(display, help_text, (20, display.shape[0] - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        display = cv2.resize(display, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
        cv2.imshow("White Balance Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("w"):
            log.info("  auto WB triggered, waiting 3 seconds...")
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            wb_auto = True
            for _ in range(90):
                cap.read()
                cv2.waitKey(33)
            temp = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
            temp = clamp(temp, WB_TEMP_MIN, WB_TEMP_MAX)
            _common.apply_wb_temperature(cap, temp)
            wb_auto = False
            log.info(f"  auto WB locked at: {temp}K")

        elif key == ord("t"):
            temp = clamp(temp + COARSE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            _common.apply_wb_temperature(cap, temp)
            wb_auto = False

        elif key == ord("y"):
            temp = clamp(temp - COARSE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            _common.apply_wb_temperature(cap, temp)
            wb_auto = False

        elif key == ord("r"):
            temp = clamp(temp + FINE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            _common.apply_wb_temperature(cap, temp)
            wb_auto = False

        elif key == ord("u"):
            temp = clamp(temp - FINE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            _common.apply_wb_temperature(cap, temp)
            wb_auto = False

        elif key == ord("s"):
            _common.save_checkpoint(
                CHECKPOINT_FILE, "white balance",
                wb_temperature=temp,
                auto_wb="false",
                avg_bgr=f"({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})",
            )
            log.info(f"  saved to {CHECKPOINT_FILE}")
            log.info(f"  wb_temperature={temp}")
            log.info(f"  avg BGR=({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})")

    cap.release()
    cv2.destroyAllWindows()

    if not args.no_session:
        _common.save_session({"wb_temperature": temp})
        log.info(f"  session updated: wb_temperature={temp}")

    log.info(f"final temperature: {temp}K")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
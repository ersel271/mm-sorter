#!/usr/bin/env python3
# tools/calibrate_wb.py
"""
Interactive white balance calibration tool for the USB camera.

Opens a live camera feed and lets you adjust the white balance
temperature using keyboard controls. A centre ROI patch shows the
average BGR values to help verify neutral white balance against
a white reference card.

The tool is standalone and does not depend on any project modules.
Only OpenCV and Numpy are required.

Usage:
    python tools/calibrate_wb.py
    python tools/calibrate_wb.py --device 0

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
    s       save current values to checkpoint file and print to terminal
    q       quit

Output:
    Prints the white balance temperature to the terminal.
    Appends values to calibration_checkpoint.txt in key=value format.

NOTE: Not all cameras expose WB_TEMPERATURE via UVC. If the temperature
reads as 0 or does not change, use v4l2-ctl to verify support:
    v4l2-ctl -d /dev/video2 --list-ctrls | grep white_balance
"""

import argparse
import datetime

import cv2
import numpy as np

# override these if the defaults feel too fast or too slow
COARSE_STEP = 100
FINE_STEP = 25
WB_TEMP_MIN = 2800
WB_TEMP_MAX = 6500

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

ROI_SIZE = 100

CHECKPOINT_FILE = "calibration_checkpoint.txt"

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

def save_checkpoint(temperature, avg_bgr):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n# written by calibrate_wb.py at {ts}\n"
        f"wb_temperature={temperature}\n"
        f"auto_wb=false\n"
        f"avg_bgr=({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})\n"
    )
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(block)
    print(f"\n  saved to {CHECKPOINT_FILE}")
    print(f"  wb_temperature={temperature}")
    print(f"  avg BGR=({avg_bgr[0]:.0f}, {avg_bgr[1]:.0f}, {avg_bgr[2]:.0f})")
    print(f"                        wb_temperature: {temperature}\n")

def main():
    parser = argparse.ArgumentParser(description="interactive white balance calibration")
    parser.add_argument("--device", type=int, default=2, help="camera device index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"error: cannot open camera device {args.device}")
        return 1

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # start with auto WB
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    temp = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
    if temp < WB_TEMP_MIN:
        temp = 4600
    wb_auto = True

    print("white balance calibration started")
    print(f"  device: {args.device}")
    print(f"  initial temperature: {temp}K")
    print(f"  place a white reference card under the camera")
    print(f"  controls: w=auto-lock  t/y=coarse(+-{COARSE_STEP})  r/u=fine(+-{FINE_STEP})  s=save  q=quit")
    print()

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

        cv2.imshow("White Balance Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("w"):
            print("  auto WB triggered, waiting 3 seconds...")
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            wb_auto = True
            for _ in range(90):
                cap.read()
                cv2.waitKey(33)
            temp = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
            temp = clamp(temp, WB_TEMP_MIN, WB_TEMP_MAX)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
            wb_auto = False
            print(f"  auto WB locked at: {temp}K")

        elif key == ord("t"):
            temp = clamp(temp + COARSE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
            wb_auto = False

        elif key == ord("y"):
            temp = clamp(temp - COARSE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
            wb_auto = False

        elif key == ord("r"):
            temp = clamp(temp + FINE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
            wb_auto = False

        elif key == ord("u"):
            temp = clamp(temp - FINE_STEP, WB_TEMP_MIN, WB_TEMP_MAX)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
            wb_auto = False

        elif key == ord("s"):
            save_checkpoint(temp, avg_bgr)

    cap.release()
    cv2.destroyAllWindows()
    print(f"final temperature: {temp}K")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

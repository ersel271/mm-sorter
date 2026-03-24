#!/usr/bin/env python3
# tools/calibrate_focus.py
"""
Interactive focus calibration tool for the USB camera.

Opens a live camera feed and lets you adjust the manual focus value
using keyboard controls. A laplacian-based sharpness score is shown
on screen to help find the optimal focus point.

The tool is standalone and does not depend on any project modules.
Only OpenCV is required.

Usage:
    python tools/calibrate_focus.py [--device DEVICE]

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
    7. Copy the output your configuration.

Controls:
    f / g   coarse focus adjustment (default +-10)
    d / h   fine focus adjustment (default +-1)
    a       trigger autofocus once, then lock and read the value
    s       save current values to checkpoint file and print to terminal
    q       quit

Output:
    Prints the final focus value to the terminal.
    Appends focus and sharpness to calibration_checkpoint.txt

NOTE: The camera must support UVC focus control. If set_focus
calls return False, check v4l2-ctl --list-ctrls for your device.
"""

import argparse
import datetime

import cv2

# override these if the defaults feel too fast or too slow
COARSE_STEP = 10
FINE_STEP = 1
FOCUS_MIN = 1
FOCUS_MAX = 1023

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

CHECKPOINT_FILE = "calibration_checkpoint.txt"

PREVIEW_SCALE = 0.85

def sharpness(frame):
    """laplacian variance of a grayscale frame. higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def save_checkpoint(focus, sharp):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n# written by calibrate_focus.py at {ts}\n"
        f"focus={focus}\n"
        f"sharpness={sharp:.1f}\n"
    )
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(block)
    print(f"\n  saved to {CHECKPOINT_FILE}")
    print(f"  focus={focus}  sharpness={sharp:.1f}")

def main():
    parser = argparse.ArgumentParser(description="interactive focus calibration")
    parser.add_argument("--device", type=int, default=2, help="camera device index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"error: cannot open camera device {args.device}")
        return 1

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # disable autofocus, read current focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    focus = int(cap.get(cv2.CAP_PROP_FOCUS))
    if focus < FOCUS_MIN:
        focus = 200

    cap.set(cv2.CAP_PROP_FOCUS, focus)

    print("focus calibration started")
    print(f"  device: {args.device}")
    print(f"  initial focus: {focus}")
    print(f"  controls: f/g=coarse(+-{COARSE_STEP})  d/h=fine(+-{FINE_STEP})  a=autofocus  s=save  q=quit")
    print()

    help_text = f"f/g: +-{COARSE_STEP} | d/h: +-{FINE_STEP} | a: autofocus | s: save | q: quit"

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

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
            print("  autofocus triggered, waiting 2 seconds...")
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            # let autofocus settle
            for _ in range(60):
                cap.read()
                cv2.waitKey(33)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            focus = int(cap.get(cv2.CAP_PROP_FOCUS))
            focus = clamp(focus, FOCUS_MIN, FOCUS_MAX)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            print(f"  autofocus locked at: {focus}")

        elif key == ord("s"):
            save_checkpoint(focus, sharp)

    cap.release()
    cv2.destroyAllWindows()
    print(f"final focus value: {focus}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

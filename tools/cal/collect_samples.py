#!/usr/bin/env python3
# tools/cal/collect_samples.py
"""
Labelled sample collection tool for the M&M sorter dataset.

Captures full frame images from the configured camera under real sorting
conditions and saves them into class based directories alongside a machine
readable manifest file. Supports automatic geometry extraction via saturation
thresholding or manual bbox drawing via mouse drag on the preview window.

The tool is standalone and does not depend on any project modules.
OpenCV, NumPy, and optionally PyYAML are required.

Usage:
    python collect_samples.py [--device DEVICE] [--output-dir OUTPUT_DIR]
                              [--clear] [--bbox-mode {none,auto,manual}]
                              [--config CONFIG] [--no-session]

Procedure:
    1. Place a single M&M (or negative sample) in view of the camera.
    2. Run this tool.
    3. Press a digit key to select the label (see controls below).
    4. In manual mode, drag the mouse to draw a bbox around the object.
    5. Press 'c' to capture and save the frame.
    6. Repeat for each sample. Press 'q' when done.

Controls:
    0         select label: non_mm
    1         select label: red
    2         select label: green
    3         select label: blue
    4         select label: yellow
    5         select label: orange
    6         select label: brown
    c         capture current frame with selected label
    n         clear current label selection
    r         clear current manual bbox (manual mode only)
    q         quit

    mouse drag  draw manual bbox (manual mode only)

Output:
    Saves full resolution PNG images to <output_dir>/<label>/<label>_<timestamp>.png
    Appends one JSON line per sample to <output_dir>/manifest.jsonl

NOTE: bbox and centroid in the manifest are always in the original full
resolution coordinate system regardless of preview scale. In manual mode the
bbox is drawn on the preview window but converted to full resolution coords
before being written to the manifest.
"""

import datetime
import json
import os
import shutil
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _common

log = _common.get_logger("collect_samples")

# camera defaults, override with --config
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
PREVIEW_SCALE = 0.85

# auto geometry extraction parameters, tuned for matte background + coloured objects
AUTO_BLUR_K = 5
AUTO_SAT_THRESH = 40
AUTO_MORPH_K = 5
AUTO_ERODE_ITER = 1
AUTO_DILATE_ITER = 2
AUTO_MIN_AREA = 500

# dataset constants
LABEL_MAP: dict[str, tuple[int, bool]] = {
    "non_mm": (0, False),
    "red":    (1, True),
    "green":  (2, True),
    "blue":   (3, True),
    "yellow": (4, True),
    "orange": (5, True),
    "brown":  (6, True),
}

KEY_TO_LABEL: dict[int, str] = {
    ord("0"): "non_mm",
    ord("1"): "red",
    ord("2"): "green",
    ord("3"): "blue",
    ord("4"): "yellow",
    ord("5"): "orange",
    ord("6"): "brown",
}

MANIFEST_FILE = "manifest.jsonl"
IMAGE_EXT = ".png"

def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_geometry(frame: np.ndarray) -> tuple[list | None, list | None, float | None]:
    """
    attempt saturation-based contour extraction on a full resolution BGR frame.
    returns (bbox, centroid, area) or (None, None, None) on failure.
    bbox is [x, y, w, h], centroid is [cx, cy], both in original frame coords.
    """
    mask = _common.extract_mask(frame, AUTO_BLUR_K, AUTO_SAT_THRESH, AUTO_MORPH_K, AUTO_ERODE_ITER, AUTO_DILATE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < AUTO_MIN_AREA:
        return None, None, None
    x, y, w, h = cv2.boundingRect(largest)
    m = cv2.moments(largest)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    else:
        pt = largest[0][0]
        cx, cy = int(pt[0]), int(pt[1])
    return [x, y, w, h], [cx, cy], float(area)

def ensure_dirs(output_dir: str) -> None:
    for label in LABEL_MAP:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    manifest = os.path.join(output_dir, MANIFEST_FILE)
    if not os.path.exists(manifest):
        open(manifest, "a", encoding="utf-8").close()

def append_manifest(output_dir: str, entry: dict) -> bool:
    path = os.path.join(output_dir, MANIFEST_FILE)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except OSError as e:
        log.info(f"error: failed to write manifest entry: {e}")
        return False

def bbox_centroid(bbox: list[int]) -> list[int]:
    x, y, w, h = bbox
    return [x + w // 2, y + h // 2]

def preview_bbox_to_full(
    bbox_preview: list[int],
    scale: float,
    frame_w: int,
    frame_h: int,
) -> list[int] | None:
    """
    convert a bbox from preview coordinate space to full resolution.
    clamps to image bounds. returns None if resulting dimensions are non-positive.
    """
    x, y, w, h = bbox_preview
    fx = int(x / scale)
    fy = int(y / scale)
    fw = int(w / scale)
    fh = int(h / scale)
    fx = max(0, min(fx, frame_w - 1))
    fy = max(0, min(fy, frame_h - 1))
    fw = min(fw, frame_w - fx)
    fh = min(fh, frame_h - fy)
    if fw <= 0 or fh <= 0:
        return None
    return [fx, fy, fw, fh]

def handle_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
    """
    mouse callback for manual bbox drawing. only active when bbox_mode is 'manual'.
    updates param dict in place with drawing state and completed bbox coordinates.
    """
    state = param
    if state.get("bbox_mode") != "manual":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["start"] = (x, y)
        state["end"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"]:
            state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        state["end"] = (x, y)
        sx, sy = state["start"]
        ex, ey = state["end"]
        bx = min(sx, ex)
        by = min(sy, ey)
        bw = abs(ex - sx)
        bh = abs(ey - sy)
        if bw > 0 and bh > 0:
            state["bbox_preview"] = [bx, by, bw, bh]
            state["bbox_full"] = preview_bbox_to_full(
                state["bbox_preview"],
                state["scale"],
                state["frame_w"],
                state["frame_h"],
            )
        else:
            state["bbox_preview"] = None
            state["bbox_full"] = None

def save_sample(
    frame: np.ndarray,
    label: str,
    output_dir: str,
    device: int,
    bbox_mode: str,
    manual_bbox: list[int] | None = None,
    manual_centroid: list[int] | None = None,
) -> tuple[bool, str | None]:
    """
    save frame and append manifest entry. returns (True, rel_path) on success.
    geometry fields are populated based on bbox_mode.
    """
    ts = datetime.datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")
    ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%S")

    filename = f"{label}_{ts_str}{IMAGE_EXT}"
    abs_path = os.path.join(output_dir, label, filename)
    rel_path = f"{label}/{filename}"

    ok = cv2.imwrite(abs_path, frame)
    if not ok:
        log.info(f"error: failed to save image to {abs_path}")
        return False, None

    h, w = frame.shape[:2]
    class_id, is_mm = LABEL_MAP[label]

    bbox = None
    centroid = None
    area = None
    bbox_source = "none"

    if bbox_mode == "auto":
        bbox, centroid, area = extract_geometry(frame)
        bbox_source = "auto"
    elif bbox_mode == "manual":
        bbox = manual_bbox
        centroid = manual_centroid
        area = float(manual_bbox[2] * manual_bbox[3]) if manual_bbox else None
        bbox_source = "manual"

    entry = {
        "image_path": rel_path,
        "label_name": label,
        "class_id": class_id,
        "is_mm": is_mm,
        "timestamp": ts_iso,
        "frame_width": w,
        "frame_height": h,
        "device": device,
        "bbox": bbox,
        "centroid": centroid,
        "area": area,
        "bbox_source": bbox_source,
    }
    ok = append_manifest(output_dir, entry)
    if not ok:
        os.remove(abs_path)
        return False, None
    return True, rel_path

def draw_overlay(
    display: np.ndarray,
    label: str | None,
    bbox_mode: str,
    session_counts: dict[str, int],
    bbox: list | None,
    centroid: list | None,
    frame_w: int,
    frame_h: int,
) -> None:
    """
    draw all overlay elements onto the preview frame in place.
    bbox and centroid must already be scaled to preview coordinates.
    """
    h, w = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    label_str = label if label else "none"
    cv2.putText(display, f"label: {label_str}", (20, 36), font, 0.6, (0, 255, 0), 1)
    cv2.putText(display, f"bbox mode: {bbox_mode}", (20, 64), font, 0.6, (0, 255, 0), 1)
    cv2.putText(display, f"res: {frame_w}x{frame_h}", (20, 92), font, 0.6, (0, 255, 0), 1)

    # per label session counts
    y = 130
    cv2.putText(display, "session:", (20, y), font, 0.5, (0, 255, 0), 1)
    y += 22
    for lname, count in session_counts.items():
        cv2.putText(display, f"  {lname}: {count}", (20, y), font, 0.5, (0, 255, 0), 1)
        y += 20

    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
    if centroid is not None:
        cx, cy = centroid
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

    if bbox_mode == "manual":
        cv2.putText(display, "drag: draw bbox | r: clear bbox", (20, h - 32), font, 0.5, (200, 200, 200), 1)
    cv2.putText(display, "0-6: label | c: capture | n: clear | q: quit", (20, h - 12), font, 0.5, (200, 200, 200), 1)

def main() -> int:
    parser = argparse.ArgumentParser(description="M&M sample collection tool")
    parser.add_argument("--device", type=int, default=2, help="camera device index")
    parser.add_argument("--output-dir", default="data/samples", help="dataset root directory")
    parser.add_argument("--clear", action="store_true", help="delete all existing samples before starting")
    parser.add_argument("--bbox-mode", choices=["none", "auto", "manual"], default="auto", help="geometry extraction mode")
    parser.add_argument("--config", default=None, help="path to config yaml (optional)")
    parser.add_argument("--no-session", action="store_true", help="skip session.json read")
    args = parser.parse_args()

    global AUTO_BLUR_K, AUTO_SAT_THRESH, AUTO_MORPH_K, AUTO_ERODE_ITER, AUTO_DILATE_ITER, AUTO_MIN_AREA

    device = args.device
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT
    fps = CAMERA_FPS

    if args.config:
        cfg = _load_config(args.config)
        cam = cfg.get("camera", {})
        device = cam.get("device", device)
        width = cam.get("width", width)
        height = cam.get("height", height)
        fps = cam.get("fps", fps)
        pre = cfg.get("preprocess", {})
        AUTO_BLUR_K = pre.get("blur_kernel", AUTO_BLUR_K)
        AUTO_SAT_THRESH = pre.get("sat_threshold", AUTO_SAT_THRESH)
        AUTO_MORPH_K = pre.get("morph_kernel", AUTO_MORPH_K)
        AUTO_ERODE_ITER = pre.get("morph_erode_iter", AUTO_ERODE_ITER)
        AUTO_DILATE_ITER = pre.get("morph_dilate_iter", AUTO_DILATE_ITER)
        AUTO_MIN_AREA = pre.get("min_area", AUTO_MIN_AREA)

    if args.clear and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        log.info(f"cleared {args.output_dir}")

    ensure_dirs(args.output_dir)

    try:
        cap = _common.open_camera(device, width, height, fps)
    except OSError as e:
        log.info(f"error: {e}")
        return 1

    # apply focus and wb from session, then from config (session takes priority)
    focus_to_apply = None
    wb_to_apply = None
    if args.config:
        cam = _load_config(args.config).get("camera", {})
        focus_to_apply = cam.get("focus")
        wb_to_apply = cam.get("wb_temperature")
    if not args.no_session:
        session = _common.load_session()
        if "focus" in session:
            focus_to_apply = session["focus"]
        if "wb_temperature" in session:
            wb_to_apply = session["wb_temperature"]
    if focus_to_apply is not None:
        _common.apply_focus(cap, focus_to_apply)
        log.info(f"  focus applied: {focus_to_apply}")
    if wb_to_apply is not None:
        _common.apply_wb_temperature(cap, wb_to_apply)
        log.info(f"  wb_temperature applied: {wb_to_apply}")

    log.info("sample collection started")
    log.info(f"  device:    {device}")
    log.info(f"  output:    {args.output_dir}")
    log.info(f"  bbox mode: {args.bbox_mode}")
    log.info(f"  controls:  0-6=label  c=capture  n=clear  r=reset bbox  q=quit")
    log.info("")

    mouse_state: dict = {
        "bbox_mode": args.bbox_mode,
        "scale": PREVIEW_SCALE,
        "frame_w": width,
        "frame_h": height,
        "drawing": False,
        "start": None,
        "end": None,
        "bbox_preview": None,
        "bbox_full": None,
    }

    cv2.namedWindow("Sample Collection")
    cv2.setMouseCallback("Sample Collection", handle_mouse, mouse_state)

    session_counts: dict[str, int] = {label: 0 for label in LABEL_MAP}
    current_label: str | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        fh, fw = frame.shape[:2]
        mouse_state["frame_w"] = fw
        mouse_state["frame_h"] = fh

        # compute geometry on full resolution frame for overlay preview
        preview_bbox = None
        preview_centroid = None
        if args.bbox_mode == "auto":
            bbox, centroid, _ = extract_geometry(frame)
            if bbox is not None:
                sx, sy = PREVIEW_SCALE, PREVIEW_SCALE
                px, py, pw, ph = bbox
                preview_bbox = [int(px * sx), int(py * sy), int(pw * sx), int(ph * sy)]
                pcx, pcy = centroid
                preview_centroid = [int(pcx * sx), int(pcy * sy)]
        elif args.bbox_mode == "manual":
            preview_bbox = mouse_state["bbox_preview"]
            if preview_bbox is not None:
                preview_centroid = bbox_centroid(preview_bbox)

        display = cv2.resize(frame, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
        draw_overlay(display, current_label, args.bbox_mode, session_counts, preview_bbox, preview_centroid, fw, fh)

        # draw active drag rect while mouse button is held
        if args.bbox_mode == "manual" and mouse_state["drawing"]:
            s = mouse_state["start"]
            e = mouse_state["end"]
            if s and e:
                cv2.rectangle(display, s, e, (0, 255, 255), 1)

        cv2.imshow("Sample Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in KEY_TO_LABEL:
            current_label = KEY_TO_LABEL[key]
            log.info(f"  label selected: {current_label}")

        elif key == ord("n"):
            current_label = None
            log.info("  label cleared")

        elif key == ord("r"):
            mouse_state["bbox_preview"] = None
            mouse_state["bbox_full"] = None
            mouse_state["drawing"] = False
            log.info("  manual bbox cleared")

        elif key == ord("c"):
            if current_label is None:
                log.info("warning: no label selected, capture skipped")
                continue
            if args.bbox_mode == "manual" and mouse_state["bbox_full"] is None:
                log.info("warning: no bbox drawn, capture skipped")
                continue
            manual_full = mouse_state["bbox_full"] if args.bbox_mode == "manual" else None
            manual_cent = bbox_centroid(manual_full) if manual_full else None
            saved, rel_path = save_sample(
                frame, current_label, args.output_dir, device, args.bbox_mode,
                manual_bbox=manual_full,
                manual_centroid=manual_cent,
            )
            if saved:
                session_counts[current_label] += 1
                total = session_counts[current_label]
                log.info(f"  saved: {current_label} (session total: {total})")
                log.info(f"  file:  {rel_path}")

    cap.release()
    cv2.destroyAllWindows()

    log.info("")
    log.info("session summary:")
    for label, count in session_counts.items():
        if count > 0:
            log.info(f"  {label}: {count}")

    total = sum(session_counts.values())
    log.info(f"  total captured: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

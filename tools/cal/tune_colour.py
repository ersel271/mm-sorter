#!/usr/bin/env python3
# tools/cal/tune_colours.py
"""
HSV threshold tuning tool for the M&M sorter colour classifier.

Reads labelled sample images collected by tools/collect_samples.py and
computes robust HSV colour ranges for each M&M colour class. The output
is a YAML snippet that can be pasted directly into the colours: section
of config/config.yaml.

The tool is standalone and does not depend on any project modules.
OpenCV, NumPy, and Matplotlib (for --plot) are required.

Usage:
    python tune_colours.py [--input-dir INPUT_DIR] [--manifest MANIFEST]
                           [--output-yaml OUTPUT_YAML] [--plot]

Procedure:
    1. Collect labelled samples with tools/collect_samples.py.
    2. Run this tool against the dataset directory.
    3. Review the terminal summary per colour.
    4. Copy the printed YAML snippet into config/config.yaml under colours:

Output:
    Logs a per-colour summary (sample count, pixel count) to the terminal.
    Emits a colours: YAML snippet to stdout and optionally to --output-yaml.
    With --plot, writes one hue histogram PNG per colour to <input-dir>/plots/.

NOTE: non_mm samples appear in the dataset statistics but are excluded from
threshold generation. Only the six colour classes are emitted in YAML.
"""

import json
import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _common

log = _common.get_logger("tune_colours")

# override these if the defaults feel too aggressive or too loose
HUE_BINS = 180
HUE_SMOOTH_SIGMA = 3
SV_LOW_PERCENTILE = 5
SV_HIGH_PERCENTILE = 95
BBOX_PADDING = 10

# mask extraction parameters
AUTO_BLUR_K = 5
AUTO_SAT_THRESH = 40
AUTO_MORPH_K = 5
AUTO_ERODE_ITER = 1
AUTO_DILATE_ITER = 2
AUTO_MIN_AREA = 500

# bins below this fraction of the histogram peak are treated as background
HUE_REGION_THRESH = 0.15

COLOUR_LABELS = ["red", "green", "blue", "yellow", "orange", "brown"]
MANIFEST_FILE = "manifest.jsonl"
PLOT_DIR = "plots"

def _smooth_circular(arr: np.ndarray, sigma: float) -> np.ndarray:
    # circular gaussian smoothing via symmetric boundary padding
    size = max(3, int(6 * sigma) | 1)
    pad = size // 2
    x = np.arange(size) - pad
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    return np.convolve(padded, k, mode="valid")

def _find_segments(active: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            start = i
            in_seg = True
        elif not a and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(active) - 1))
    return segments

def compute_hue_ranges(hue_values: np.ndarray) -> list[list[int]]:
    """
    derive hue range(s) from raw pixel hue values.
    handles wrap-around for colours like red that straddle hue 0.
    """
    hist = np.bincount(hue_values.astype(np.int32), minlength=HUE_BINS).astype(float)
    hist = _smooth_circular(hist, HUE_SMOOTH_SIGMA)
    thresh = hist.max() * HUE_REGION_THRESH
    active = hist >= thresh
    segments = _find_segments(active)

    if not segments:
        lo = int(np.percentile(hue_values, SV_LOW_PERCENTILE))
        hi = int(np.percentile(hue_values, SV_HIGH_PERCENTILE))
        return [[lo, hi]]

    # wrap-around: significant mass at both ends of the hue circle
    if segments[0][0] == 0 and segments[-1][1] == HUE_BINS - 1:
        return [[0, segments[0][1]], [segments[-1][0], HUE_BINS - 1]]

    largest = max(segments, key=lambda s: s[1] - s[0])
    return [[largest[0], largest[1]]]

def compute_sv_range(values: np.ndarray) -> list[int]:
    lo = int(np.percentile(values, SV_LOW_PERCENTILE))
    hi = int(np.percentile(values, SV_HIGH_PERCENTILE))
    return [lo, hi]

def extract_mask(image: np.ndarray) -> np.ndarray | None:
    """
    saturation-threshold object extraction on a BGR image.
    returns a binary mask covering the largest detected object, or None on failure.
    """
    mask = _common.extract_mask(image, AUTO_BLUR_K, AUTO_SAT_THRESH, AUTO_MORPH_K, AUTO_ERODE_ITER, AUTO_DILATE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < AUTO_MIN_AREA:
        return None
    obj_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(obj_mask, [largest], -1, 255, cv2.FILLED)
    return obj_mask

def collect_pixels_for_sample(
    row: dict,
    input_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    load one sample image and return (H, S, V) pixel arrays for the detected object.
    uses the manifest bbox crop when available; falls back to full frame extraction.
    returns None when the sample cannot be processed.
    """
    img_rel = row.get("image_path")
    if not img_rel:
        log.info("warning: manifest row missing image_path, skipping sample")
        return None

    img_path = os.path.join(input_dir, img_rel)
    image = cv2.imread(img_path)
    if image is None:
        log.info(f"warning: failed to read image: {img_path}")
        return None

    bbox = row.get("bbox")
    crop = image
    if bbox is not None:
        try:
            if len(bbox) != 4:
                raise ValueError(f"expected 4 elements, got {len(bbox)}")
            x, y, bw, bh = [int(v) for v in bbox]
            if bw <= 0 or bh <= 0:
                raise ValueError(f"non-positive bbox dimensions ({bw}x{bh})")
            ih, iw = image.shape[:2]
            x1 = max(0, x - BBOX_PADDING)
            y1 = max(0, y - BBOX_PADDING)
            x2 = min(iw, x + bw + BBOX_PADDING)
            y2 = min(ih, y + bh + BBOX_PADDING)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"bbox crop is empty after clamping to image bounds")
            crop = image[y1:y2, x1:x2]
        except (TypeError, ValueError) as e:
            log.info(f"warning: malformed bbox in {img_rel}: {e}, skipping sample")
            return None

    mask = extract_mask(crop)
    if mask is None and crop is not image:
        # bbox crop failed, retry on full frame
        crop = image
        mask = extract_mask(image)

    if mask is None:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return None

    return pixels[:, 0], pixels[:, 1], pixels[:, 2]

def load_manifest(path: str) -> list[dict]:
    rows = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.info(f"warning: skipping malformed JSON at line {lineno}: {e}")
                skipped += 1
    if skipped:
        log.info(f"warning: skipped {skipped} malformed manifest line(s)")
    return rows

def build_yaml_snippet(results: dict) -> str:
    lines = ["colours:"]
    for label in COLOUR_LABELS:
        if label not in results:
            continue
        r = results[label]
        h_str = "[" + ", ".join(f"[{lo}, {hi}]" for lo, hi in r["h"]) + "]"
        lines.append(f"  {label}:")
        lines.append(f"    h: {h_str}")
        lines.append(f"    s: [{r['s'][0]}, {r['s'][1]}]")
        lines.append(f"    v: [{r['v'][0]}, {r['v'][1]}]")
    return "\n".join(lines)

def save_plots(
    colour_data: dict[str, np.ndarray],
    results: dict,
    plot_dir: str,
) -> None:
    """
    write one hue histogram PNG per colour into plot_dir.
    detected ranges are highlighted with vertical bands.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("error: matplotlib is required for --plot")
        sys.exit(1)

    try:
        os.makedirs(plot_dir, exist_ok=True)
    except OSError as e:
        log.info(f"error: cannot create plot directory {plot_dir}: {e}")
        sys.exit(1)

    for label, hues in colour_data.items():
        hist = np.bincount(hues.astype(np.int32), minlength=HUE_BINS).astype(float)
        hist_smooth = _smooth_circular(hist, HUE_SMOOTH_SIGMA)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(HUE_BINS), hist, color="lightgray", label="raw")
        ax.plot(range(HUE_BINS), hist_smooth, color="black", linewidth=1.2, label="smoothed")

        seen_labels: set[str] = set()
        if label in results:
            for lo, hi in results[label]["h"]:
                range_label = f"range [{lo}, {hi}]"
                ax.axvspan(lo, hi, alpha=0.25, color="red", label=range_label if range_label not in seen_labels else None)
                seen_labels.add(range_label)

        ax.set_title(f"{label} hue distribution")
        ax.set_xlabel("hue (OpenCV 0-179)")
        ax.set_ylabel("pixel count")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"{label}_hue.png")
        try:
            fig.savefig(out_path, dpi=100)
            log.info(f"  plot: {out_path}")
        except OSError as e:
            log.info(f"error: failed to save plot {out_path}: {e}")
        plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser(description="M&M HSV threshold tuning tool")
    parser.add_argument("--input-dir", default="data/samples", help="dataset root directory")
    parser.add_argument("--manifest", default=None, help="path to manifest.jsonl (default: <input-dir>/manifest.jsonl)")
    parser.add_argument("--output-yaml", default=None, help="write YAML snippet to this file")
    parser.add_argument("--plot", action="store_true", help="generate hue histogram plots")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.input_dir, MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        log.info(f"error: manifest not found: {manifest_path}")
        return 1

    rows = load_manifest(manifest_path)
    log.info(f"loaded {len(rows)} manifest entries from {manifest_path}")
    log.info("")

    grouped: dict[str, list[dict]] = {}
    invalid = 0
    for row in rows:
        lbl = row.get("label_name")
        img = row.get("image_path")
        if not lbl or not img:
            log.info(f"warning: skipping row missing label_name or image_path: {row}")
            invalid += 1
            continue
        grouped.setdefault(lbl, []).append(row)

    known = set(COLOUR_LABELS) | {"non_mm"}
    for lbl in grouped:
        if lbl not in known:
            log.info(f"warning: unknown label '{lbl}' in manifest ({len(grouped[lbl])} row(s)), will be ignored")

    if invalid:
        log.info(f"warning: skipped {invalid} invalid manifest row(s)")
        log.info("")

    results: dict = {}
    colour_hues: dict[str, np.ndarray] = {}

    for label in COLOUR_LABELS + ["non_mm"]:
        label_rows = grouped.get(label, [])
        if not label_rows:
            log.info(f"{label}: no samples found")
            continue

        h_all, s_all, v_all = [], [], []
        skipped = 0
        for row in label_rows:
            result = collect_pixels_for_sample(row, args.input_dir)
            if result is None:
                skipped += 1
                continue
            h, s, v = result
            h_all.append(h)
            s_all.append(s)
            v_all.append(v)

        usable = len(label_rows) - skipped
        log.info(f"{label}:")
        log.info(f"  samples: {usable} usable, {skipped} skipped")

        if not h_all:
            log.info("  warning: no usable samples, skipping")
            log.info("")
            continue

        hues = np.concatenate(h_all)
        sats = np.concatenate(s_all)
        vals = np.concatenate(v_all)

        log.info(f"  pixels:  {len(hues)}")

        if label == "non_mm":
            log.info("")
            continue

        h_ranges = compute_hue_ranges(hues)
        s_range = compute_sv_range(sats)
        v_range = compute_sv_range(vals)

        results[label] = {"h": h_ranges, "s": s_range, "v": v_range}
        colour_hues[label] = hues
        log.info("")

    if not results:
        log.info("error: no colour data to emit")
        return 1

    yaml_snippet = build_yaml_snippet(results)
    log.info("-" * 40)
    log.info(yaml_snippet)
    log.info("-" * 40)

    if args.output_yaml:
        try:
            with open(args.output_yaml, "w", encoding="utf-8") as f:
                f.write(yaml_snippet + "\n")
            log.info(f"wrote YAML snippet to {args.output_yaml}")
        except OSError as e:
            log.info(f"error: failed to write {args.output_yaml}: {e}")
            return 1

    if args.plot:
        plot_dir = os.path.join(args.input_dir, PLOT_DIR)
        save_plots(colour_hues, results, plot_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

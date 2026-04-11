#!/usr/bin/env python3
# tools/cal/tune_shape.py
"""
Shape threshold tuning tool for the M&M sorter object validator.

Reads is_mm=true samples from a manifest, extracts contour metrics from each
bbox crop, and recommends threshold values based on the observed distribution.
The output is a YAML snippet for the thresholds: section of config/config.yaml.

The tool is standalone and does not depend on any project modules.
OpenCV, NumPy, and Matplotlib (for --plot) are required.

Usage:
    python tune_shape.py [--input-dir INPUT_DIR] [--manifest MANIFEST]
                         [--plot]

Procedure:
    1. Collect labelled samples with tools/collect_samples.py (auto bbox mode).
    2. Run this tool against the dataset directory.
    3. Review the per-metric distribution summary.
    4. Copy the printed YAML snippet into config/config.yaml under thresholds:

Output:
    Logs per-metric stats (min, max, mean, p05, p95) to the terminal and tools/_log/.
    Emits a thresholds: YAML snippet covering circularity_min, aspect_ratio_max,
    solidity_min, and texture_max.
    With --plot, writes one histogram PNG per metric to <input-dir>/plots/.

NOTE: samples without a bbox in the manifest are skipped. Collect samples with
--bbox-mode auto or manual to ensure bbox fields are populated.
"""

import json
import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _common

log = _common.get_logger("tune_shape")

# mask extraction parameters, must match collect_samples
AUTO_BLUR_K = 5
AUTO_SAT_THRESH = 40
AUTO_MORPH_K = 5
AUTO_ERODE_ITER = 1
AUTO_DILATE_ITER = 2
AUTO_MIN_AREA = 500

# percentile bounds used to derive thresholds from the observed distribution
SHAPE_LOW_PERCENTILE = 5
SHAPE_HIGH_PERCENTILE = 95

BBOX_PADDING = 10
MANIFEST_FILE = "manifest.jsonl"
PLOT_DIR = "plots"

def load_mm_rows(path: str) -> list[dict]:
    """read manifest and return only is_mm=true rows that have a bbox."""
    rows = []
    skipped_json = 0
    skipped_filter = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log.info(f"warning: skipping malformed JSON at line {lineno}: {e}")
                skipped_json += 1
                continue
            if not row.get("is_mm"):
                skipped_filter += 1
                continue
            if row.get("bbox") is None:
                skipped_filter += 1
                continue
            rows.append(row)
    if skipped_json:
        log.info(f"warning: skipped {skipped_json} malformed manifest line(s)")
    return rows, skipped_filter

def _process_sample(row: dict, input_dir: str) -> dict | None:
    """
    load one sample, extract the largest contour from its bbox crop, compute metrics.
    returns a dict with circularity, aspect_ratio, solidity, texture, or None on failure.
    """
    img_rel = row.get("image_path")
    bbox = row.get("bbox")

    img_path = os.path.join(input_dir, img_rel)
    image = cv2.imread(img_path)
    if image is None:
        log.info(f"warning: failed to read image: {img_path}")
        return None

    try:
        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            raise ValueError(f"non-positive bbox dimensions ({bw}x{bh})")
        ih, iw = image.shape[:2]
        x1 = max(0, x - BBOX_PADDING)
        y1 = max(0, y - BBOX_PADDING)
        x2 = min(iw, x + bw + BBOX_PADDING)
        y2 = min(ih, y + bh + BBOX_PADDING)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox crop empty after clamping to image bounds")
        crop = image[y1:y2, x1:x2]
    except (TypeError, ValueError) as e:
        log.info(f"warning: malformed bbox in {img_rel}: {e}")
        return None

    mask = _common.extract_mask(crop, AUTO_BLUR_K, AUTO_SAT_THRESH, AUTO_MORPH_K, AUTO_ERODE_ITER, AUTO_DILATE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < AUTO_MIN_AREA:
        return None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, cw, ch = cv2.boundingRect(contour)

    return {
        "circularity":  _common.compute_circularity(area, perimeter),
        "aspect_ratio": _common.compute_aspect_ratio(cw, ch),
        "solidity":     _common.compute_solidity(contour),
        "texture":      _common.compute_texture(crop, mask),
    }

def save_plots(
    metrics: dict[str, list[float]],
    thresholds: dict[str, float],
    plot_dir: str,
) -> None:
    """
    write one histogram PNG per metric into plot_dir.
    the suggested threshold is highlighted with a vertical line.
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

    # map metric name to (threshold key, is_upper_bound)
    threshold_map = {
        "circularity":  ("circularity_min",  False),
        "aspect_ratio": ("aspect_ratio_max", True),
        "solidity":     ("solidity_min",      False),
        "texture":      ("texture_max",       True),
    }

    for metric, values in metrics.items():
        arr = np.array(values)
        thresh_key, is_upper = threshold_map[metric]
        thresh_val = thresholds[thresh_key]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(arr, bins=30, color="lightgray", edgecolor="grey", label="samples")
        ax.axvline(thresh_val, color="red", linewidth=1.5,
                   label=f"{thresh_key} = {thresh_val:.3f}")

        ax.set_title(f"{metric} distribution")
        ax.set_xlabel(metric)
        ax.set_ylabel("sample count")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"{metric}.png")
        try:
            fig.savefig(out_path, dpi=100)
            log.info(f"  plot: {out_path}")
        except OSError as e:
            log.info(f"error: failed to save plot {out_path}: {e}")
        plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser(description="M&M shape threshold tuning tool")
    parser.add_argument("--input-dir", default="data/samples", help="dataset root directory")
    parser.add_argument("--manifest", default=None, help="path to manifest.jsonl (default: <input-dir>/manifest.jsonl)")
    parser.add_argument("--plot", action="store_true", help="generate per-metric histogram plots")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.input_dir, MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        log.info(f"error: manifest not found: {manifest_path}")
        return 1

    rows, filtered = load_mm_rows(manifest_path)
    log.info(f"loaded {len(rows)} is_mm samples from {manifest_path} ({filtered} filtered)")
    if not rows:
        log.info("error: no usable is_mm samples found")
        return 1
    log.info("")

    metrics: dict[str, list[float]] = {
        "circularity":  [],
        "aspect_ratio": [],
        "solidity":     [],
        "texture":      [],
    }
    skipped = 0

    for row in rows:
        result = _process_sample(row, args.input_dir)
        if result is None:
            skipped += 1
            continue
        for key in metrics:
            metrics[key].append(result[key])

    processed = len(rows) - skipped
    log.info(f"processed: {processed} / {len(rows)} samples ({skipped} skipped)")
    log.info("")

    if processed == 0:
        log.info("error: no samples could be processed")
        return 1

    circ_min  = float(np.percentile(metrics["circularity"],  SHAPE_LOW_PERCENTILE))
    ar_max    = float(np.percentile(metrics["aspect_ratio"], SHAPE_HIGH_PERCENTILE))
    solid_min = float(np.percentile(metrics["solidity"],     SHAPE_LOW_PERCENTILE))
    tex_max   = float(np.percentile(metrics["texture"],      SHAPE_HIGH_PERCENTILE))

    thresholds = {
        "circularity_min":  circ_min,
        "aspect_ratio_max": ar_max,
        "solidity_min":     solid_min,
        "texture_max":      tex_max,
    }

    # per-metric distribution summary
    for metric, thresh_key in [
        ("circularity",  "circularity_min"),
        ("aspect_ratio", "aspect_ratio_max"),
        ("solidity",     "solidity_min"),
        ("texture",      "texture_max"),
    ]:
        arr = np.array(metrics[metric])
        log.info(f"{metric}:")
        log.info(
            f"  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}"
            f"  p{SHAPE_LOW_PERCENTILE:02d}={np.percentile(arr, SHAPE_LOW_PERCENTILE):.3f}"
            f"  p{SHAPE_HIGH_PERCENTILE:02d}={np.percentile(arr, SHAPE_HIGH_PERCENTILE):.3f}"
        )
        log.info("")

    snippet = (
        "thresholds:\n"
        f"  circularity_min:  {circ_min:.2f}\n"
        f"  aspect_ratio_max: {ar_max:.2f}\n"
        f"  solidity_min:     {solid_min:.2f}\n"
        f"  texture_max:      {tex_max:.1f}"
    )
    log.info("-" * 40)
    log.info(snippet)
    log.info("-" * 40)

    if args.plot:
        plot_dir = os.path.join(args.input_dir, PLOT_DIR)
        save_plots(metrics, thresholds, plot_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
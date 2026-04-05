# tests/helpers/ground_truth_helpers.py

from pathlib import Path

def write_gt(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "gt.txt"
    path.write_text("\n".join(lines))
    return path

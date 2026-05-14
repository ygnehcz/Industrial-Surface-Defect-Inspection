"""Inspect the industrial surface defect dataset.

Scans train/ and test/ split directories, reads every ground-truth mask,
and reports per-split statistics: total masks, defective count, normal count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = {
    "train": DATA_DIR / "train",
    "test": DATA_DIR / "test",
}


def inspect_split(split_name: str, split_dir: Path) -> dict[str, int]:
    """Scan a single split directory and count defective / normal samples.

    Raises
    ------
    FileNotFoundError
        If the split directory does not exist.
    RuntimeError
        If a mask file cannot be read.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Collect all *_GT.png files (sorted for deterministic output)
    mask_paths = sorted(split_dir.glob("*_GT.png"))
    if not mask_paths:
        print(f"  [WARN] No *_GT.png masks found in {split_dir}", file=sys.stderr)

    total = 0
    defective = 0

    for mask_path in mask_paths:
        total += 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # Any non-zero pixel => defective
        if np.any(mask):
            defective += 1

    normal = total - defective
    return {"total": total, "defective": defective, "normal": normal}


def main() -> None:
    grand_total = 0
    grand_defective = 0
    grand_normal = 0

    for name, directory in SPLITS.items():
        print(f"[{name}]  {directory}")
        try:
            stats = inspect_split(name, directory)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"  Total masks:       {stats['total']:>6d}")
        print(f"  Defective:         {stats['defective']:>6d}")
        print(f"  Normal:            {stats['normal']:>6d}")
        print()

        grand_total += stats["total"]
        grand_defective += stats["defective"]
        grand_normal += stats["normal"]

    print("=" * 36)
    print(f"[TOTAL]")
    print(f"  Total masks:       {grand_total:>6d}")
    print(f"  Defective:         {grand_defective:>6d}")
    print(f"  Normal:            {grand_normal:>6d}")


if __name__ == "__main__":
    main()

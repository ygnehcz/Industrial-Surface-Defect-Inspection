"""Visualize defective samples as triplets: original, GT mask, overlay.

Usage:
    python scripts/visualize_samples.py --split train --num-samples 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "dataset_samples"

OVERLAY_ALPHA = 0.45          # transparency of defect overlay
DEFECT_COLOR = (1.0, 0.0, 0.0)  # red in [0,1] float for matplotlib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect_defective_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs for all defective samples.

    A sample is defective when its GT mask contains at least one non-zero pixel.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    mask_paths = sorted(split_dir.glob("*_GT.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_GT.png masks found in {split_dir}")

    pairs: list[tuple[Path, Path]] = []

    for mask_path in mask_paths:
        # Derive original image filename: strip "_GT" suffix
        stem = mask_path.stem  # e.g. "10301_GT"
        if not stem.endswith("_GT"):
            continue
        image_stem = stem[:-3]  # remove trailing "_GT"
        image_path = split_dir / f"{image_stem}.png"

        if not image_path.is_file():
            print(f"  [WARN] Missing original image, skipping: {image_path}", file=sys.stderr)
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [WARN] Failed to read mask, skipping: {mask_path}", file=sys.stderr)
            continue

        if np.any(mask):
            pairs.append((image_path, mask_path))

    return pairs


def build_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend the defect mask (red) on top of the original image."""
    overlay = image_rgb.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        OVERLAY_ALPHA * np.array(DEFECT_COLOR) * 255
        + (1 - OVERLAY_ALPHA) * overlay[mask_bool]
    ).astype(np.uint8)
    return overlay


def save_triplet(
    image_path: Path,
    mask_path: Path,
    out_dir: Path,
) -> Path:
    """Generate and save a triplet figure for a single sample.

    Returns the path to the saved figure.
    """
    image_bgr = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = build_overlay(image_rgb, mask)

    stem = image_path.stem  # e.g. "10301"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "GT Mask", "Overlay (Red = Defect)"]
    images = [image_rgb, mask, overlay]
    cmaps = [None, "gray", None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / f"{stem}_triplet.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize defective samples from the surface-defect dataset."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which data split to sample from (default: train).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of defective samples to visualise (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_dir = DATA_DIR / args.split
    out_dir = OUTPUT_DIR / args.split

    # Collect all defective (image, mask) pairs
    try:
        pairs = collect_defective_pairs(split_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if len(pairs) == 0:
        print(f"ERROR: No defective samples found in {split_dir}", file=sys.stderr)
        sys.exit(1)

    if len(pairs) < args.num_samples:
        print(
            f"WARNING: Requested {args.num_samples} samples, "
            f"but only {len(pairs)} defective samples available. "
            f"Showing all {len(pairs)}.",
            file=sys.stderr,
        )
        selected = pairs
    else:
        # Evenly spaced selection to cover the dataset range
        indices = np.linspace(0, len(pairs) - 1, args.num_samples, dtype=int)
        selected = [pairs[i] for i in indices]

    print(f"Split: {args.split}")
    print(f"Defective samples found: {len(pairs)}")
    print(f"Samples to visualise: {len(selected)}")
    print(f"Output: {out_dir}")
    print("-" * 50)

    for img_path, mask_path in selected:
        try:
            saved = save_triplet(img_path, mask_path, out_dir)
            print(f"  {saved}")
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    print("-" * 50)
    print(f"Done. {len(selected)} triplet(s) saved.")


if __name__ == "__main__":
    main()

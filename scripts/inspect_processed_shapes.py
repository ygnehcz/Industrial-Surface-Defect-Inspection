"""Inspect image/mask shapes, channels, and pairing in data/processed/.

Checks every split (train, val, test) for:
  - file counts in images/ and masks/
  - filename 1:1 correspondence
  - spatial size consistency between each image-mask pair
  - channel count distributions

Useful as a sanity check before writing PyTorch Dataset classes.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inspect_split(split_dir: Path, label: str) -> None:
    """Run all checks on a single processed split directory.

    Raises FileNotFoundError on missing directories or count mismatches.
    Raises RuntimeError on read failures or size inconsistencies.
    """
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

    image_names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
    mask_names = sorted(p.name for p in masks_dir.iterdir() if p.is_file())

    print(f"[{label}]")
    print(f"  Image files: {len(image_names)}")
    print(f"  Mask files:  {len(mask_names)}")

    # ---- Count mismatch ----
    if len(image_names) != len(mask_names):
        raise FileNotFoundError(
            f"[{label}] File count mismatch: "
            f"{len(image_names)} images vs {len(mask_names)} masks"
        )

    # ---- Filename 1:1 correspondence ----
    if image_names != mask_names:
        only_images = set(image_names) - set(mask_names)
        only_masks = set(mask_names) - set(image_names)
        msg_parts = [f"[{label}] Filename mismatch:"]
        if only_images:
            msg_parts.append(f"  Only in images/: {sorted(only_images)[:5]}...")
        if only_masks:
            msg_parts.append(f"  Only in masks/:  {sorted(only_masks)[:5]}...")
        raise FileNotFoundError("\n".join(msg_parts))

    # ---- Per-file checks ----
    img_shapes: Counter = Counter()
    msk_shapes: Counter = Counter()
    img_channels: Counter = Counter()
    msk_channels: Counter = Counter()

    for fname in image_names:
        img_path = images_dir / fname
        msk_path = masks_dir / fname

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")

        # Shape in (H, W) format, channels count as 3rd dim when present
        img_h, img_w = img.shape[:2]
        msk_h, msk_w = mask.shape[:2]
        img_c = 1 if img.ndim == 2 else img.shape[2]
        msk_c = 1 if mask.ndim == 2 else mask.shape[2]

        if (img_h, img_w) != (msk_h, msk_w):
            raise RuntimeError(
                f"[{label}] Shape mismatch for {fname}: "
                f"image {img_h}x{img_w}, mask {msk_h}x{msk_w}"
            )

        img_shapes[(img_h, img_w)] += 1
        msk_shapes[(msk_h, msk_w)] += 1
        img_channels[img_c] += 1
        msk_channels[msk_c] += 1

    # ---- Print distributions ----
    def _print_dist(title: str, counter: Counter, suffix: str = "") -> None:
        print(f"  {title}:")
        for key, count in sorted(counter.items()):
            print(f"    {key}{suffix}  ->  {count}")

    _print_dist("Image shape distribution", img_shapes, "")
    _print_dist("Mask shape distribution", msk_shapes, "")
    _print_dist("Image channel distribution", img_channels, "")
    _print_dist("Mask channel distribution", msk_channels, "")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("data/processed/ shape inspection")
    print("=" * 40)
    print()

    all_ok = True

    for split_name in SPLITS:
        split_dir = PROCESSED_DIR / split_name
        try:
            inspect_split(split_dir, split_name)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            print()
            all_ok = False

    if all_ok:
        print("All checks passed.")
    else:
        print("One or more checks failed — see errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

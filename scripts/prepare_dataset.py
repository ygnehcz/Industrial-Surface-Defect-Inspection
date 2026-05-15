"""Prepare the KolektorSDD2 dataset for PyTorch training.

Reads raw data from data/raw/{train,test}, performs a stratified
80/20 train/val split on the official training set, and copies all
files into data/processed/ with the following layout:

    data/processed/
    ├── train/
    │   ├── images/   # original images
    │   └── masks/    # GT masks (renamed: strip "_GT" suffix)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

The official test split is copied as-is (no re-splitting).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VAL_RATIO = 0.2
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect_samples(split_dir: Path) -> list[tuple[Path, Path, bool]]:
    """Scan a raw split directory and return (image_path, mask_path, is_defective).

    Raises FileNotFoundError if the directory is missing or contains no masks.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    mask_paths = sorted(split_dir.glob("*_GT.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_GT.png masks found in {split_dir}")

    samples: list[tuple[Path, Path, bool]] = []

    for mask_path in mask_paths:
        stem = mask_path.stem  # e.g. "10301_GT"
        if not stem.endswith("_GT"):
            continue
        image_stem = stem[:-3]
        image_path = split_dir / f"{image_stem}.png"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing original image: {image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        is_defective = bool(np.any(mask))
        samples.append((image_path, mask_path, is_defective))

    return samples


def stratified_split(
    samples: list[tuple[Path, Path, bool]],
) -> tuple[list, list]:
    """Split samples into train / val, preserving class ratio per stratum.

    Uses a fixed random seed for deterministic output.
    """
    defective = [s for s in samples if s[2]]
    normal = [s for s in samples if not s[2]]

    rng = np.random.default_rng(SEED)

    def _split(group):
        indices = rng.permutation(len(group))
        n_val = max(1, int(len(group) * VAL_RATIO))
        val_idx = set(indices[:n_val].tolist())
        train_list = [s for i, s in enumerate(group) if i not in val_idx]
        val_list = [s for i, s in enumerate(group) if i in val_idx]
        return train_list, val_list

    train_def, val_def = _split(defective)
    train_norm, val_norm = _split(normal)

    train_all = train_def + train_norm
    val_all = val_def + val_norm

    # Shuffle within each set so defective/normal are interleaved
    rng.shuffle(train_all)
    rng.shuffle(val_all)

    return train_all, val_all


def copy_split(
    samples: list[tuple[Path, Path, bool]],
    dst_dir: Path,
    split_label: str,
) -> tuple[int, int]:
    """Copy images and masks to a processed split directory.

    Masks are renamed: "<stem>_GT.png" → "<stem>.png" so they share
    the same filename as the original image.

    Returns (defective_count, normal_count).
    """
    images_dir = dst_dir / "images"
    masks_dir = dst_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    defective = 0
    normal = 0

    for img_path, mask_path, is_defective in samples:
        shutil.copy2(img_path, images_dir / img_path.name)
        # Strip "_GT" from the mask filename
        new_mask_name = mask_path.stem[:-3] + mask_path.suffix  # ".png"
        shutil.copy2(mask_path, masks_dir / new_mask_name)

        if is_defective:
            defective += 1
        else:
            normal += 1

    return defective, normal


def print_split_stats(
    label: str, total: int, defective: int, normal: int
) -> None:
    print(f"  [{label}]")
    print(f"    Total:      {total:>6d}")
    print(f"    Defective:  {defective:>6d}")
    print(f"    Normal:     {normal:>6d}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Safety check: raw directories must exist ----
    raw_train = RAW_DIR / "train"
    raw_test = RAW_DIR / "test"

    if not raw_train.is_dir():
        print(f"ERROR: Raw train directory not found: {raw_train}", file=sys.stderr)
        sys.exit(1)
    if not raw_test.is_dir():
        print(f"ERROR: Raw test directory not found: {raw_test}", file=sys.stderr)
        sys.exit(1)

    # ---- Handle existing processed directory ----
    if PROCESSED_DIR.exists():
        print(f"INFO: Removing existing {PROCESSED_DIR} ...")
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Collect ----
    print("Collecting raw samples ...")
    try:
        train_samples = collect_samples(raw_train)
        test_samples = collect_samples(raw_test)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  raw/train: {len(train_samples)} masks found")
    print(f"  raw/test:  {len(test_samples)} masks found")
    print()

    # ---- Stratified split on train ----
    train_list, val_list = stratified_split(train_samples)

    # ---- Copy ----
    print("Copying files to data/processed/ ...")
    print()

    t_def, t_norm = copy_split(train_list, PROCESSED_DIR / "train", "train")
    v_def, v_norm = copy_split(val_list, PROCESSED_DIR / "val", "val")
    te_def, te_norm = copy_split(test_samples, PROCESSED_DIR / "test", "test")

    # ---- Print summary ----
    print("=" * 40)
    print("Dataset split summary")
    print("=" * 40)
    print_split_stats("train", len(train_list), t_def, t_norm)
    print_split_stats("val", len(val_list), v_def, v_norm)
    print_split_stats("test", len(test_samples), te_def, te_norm)

    print("Done. Output written to", PROCESSED_DIR)


if __name__ == "__main__":
    main()

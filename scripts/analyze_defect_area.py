"""Analyse defect-area distribution across train/test splits of KSDD2.

Computes per-defect pixel count and defect-to-image ratio, prints summary
statistics, and saves histograms to outputs/figures/defect_area_analysis/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = {
    "train": DATA_DIR / "train",
    "test": DATA_DIR / "test",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "defect_area_analysis"

# Histogram bins
BINS = 40


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def collect_defect_stats(split_dir: Path) -> tuple[list[int], list[float]]:
    """Walk a split directory and return (pixel_counts, ratios) for defective masks.

    Raises FileNotFoundError if the directory is missing or contains no masks.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    mask_paths = sorted(split_dir.glob("*_GT.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No *_GT.png masks found in {split_dir}")

    pixel_counts: list[int] = []
    ratios: list[float] = []

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if not np.any(mask):
            continue  # skip normal samples

        total_pixels = mask.size
        defect_pixels = int(np.count_nonzero(mask))
        pixel_counts.append(defect_pixels)
        ratios.append(defect_pixels / total_pixels * 100.0)

    return pixel_counts, ratios


def print_stats(label: str, pixel_counts: list[int], ratios: list[float]) -> None:
    """Print summary statistics for one split (or total)."""
    if not pixel_counts:
        print(f"\n[{label}]\n  No defective samples.\n")
        return

    arr_px = np.array(pixel_counts)
    arr_ratio = np.array(ratios)

    print(f"\n[{label}]  n = {len(pixel_counts)}")
    print(f"  defect_pixels  --  min: {arr_px.min():>10,d}  "
          f"max: {arr_px.max():>10,d}  "
          f"mean: {arr_px.mean():>12,.1f}  "
          f"median: {np.median(arr_px):>10,.1f}")
    print(f"  defect_ratio   --  min: {arr_ratio.min():>9.4f}%  "
          f"max: {arr_ratio.max():>9.4f}%  "
          f"mean: {arr_ratio.mean():>9.4f}%  "
          f"median: {np.median(arr_ratio):>9.4f}%")


def save_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    color: str = "steelblue",
) -> None:
    """Save a single histogram figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=BINS, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -- {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    all_pixels: list[int] = []
    all_ratios: list[float] = []

    # ---- Collect & print per-split stats ----
    for name, directory in SPLITS.items():
        try:
            px, ratio = collect_defect_stats(directory)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print_stats(name, px, ratio)
        all_pixels.extend(px)
        all_ratios.extend(ratio)

    # ---- Combined stats ----
    print_stats("TOTAL", all_pixels, all_ratios)

    if not all_pixels:
        print("ERROR: No defective samples found in any split.", file=sys.stderr)
        sys.exit(1)

    # ---- Histograms ----
    print(f"\nSaving charts to {OUTPUT_DIR}")

    save_histogram(
        data=np.array(all_pixels),
        title="Defect Area Distribution (Pixels)",
        xlabel="Defect Pixels",
        out_path=OUTPUT_DIR / "defect_pixels_hist.png",
    )

    save_histogram(
        data=np.array(all_ratios),
        title="Defect Area Ratio Distribution",
        xlabel="Defect Ratio (%)",
        out_path=OUTPUT_DIR / "defect_ratio_hist.png",
        color="indianred",
    )

    print("Done.")


if __name__ == "__main__":
    main()

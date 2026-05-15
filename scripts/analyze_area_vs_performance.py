"""Analyse defect area versus segmentation performance on validation samples.

Reads the per-sample CSV, groups defective samples by ground-truth defect
size (small / medium / large), prints a summary table, saves a group-level
CSV, and generates scatter plots of area vs. Dice and area vs. Recall.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "outputs/analysis/validation_sample_analysis/val_defective_sample_metrics.csv"
OUTPUT_DIR = Path("outputs/analysis/area_vs_performance")
SUMMARY_CSV = OUTPUT_DIR / "area_group_summary.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load CSV ----
    csv_path = Path(INPUT_CSV)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "filename": row["filename"],
                    "dice": float(row["dice"]),
                    "iou": float(row["iou"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "gt_positive_pixels": int(row["gt_positive_pixels"]),
                    "pred_positive_pixels": int(row["pred_positive_pixels"]),
                }
            )

    n = len(records)
    print(f"Input CSV: {csv_path}")
    print(f"Total defective samples: {n}")

    # ---- Sort by defect area ----
    records.sort(key=lambda r: r["gt_positive_pixels"])

    # ---- Split into three roughly equal groups ----
    group_size = n // 3
    remainder = n % 3
    # Distribute remainder across the first remainder groups
    boundaries = [0]
    for g in range(3):
        extra = 1 if g < remainder else 0
        boundaries.append(boundaries[-1] + group_size + extra)

    groups = {
        "small": records[boundaries[0] : boundaries[1]],
        "medium": records[boundaries[1] : boundaries[2]],
        "large": records[boundaries[2] : boundaries[3]],
    }

    # ---- Group statistics ----
    def stats(samples: list[dict]) -> dict:
        if not samples:
            return {}
        dice_vals = [s["dice"] for s in samples]
        iou_vals = [s["iou"] for s in samples]
        prec_vals = [s["precision"] for s in samples]
        rec_vals = [s["recall"] for s in samples]
        areas = [s["gt_positive_pixels"] for s in samples]
        return {
            "sample_count": len(samples),
            "min_gt_pixels": min(areas),
            "max_gt_pixels": max(areas),
            "mean_gt_pixels": np.mean(areas),
            "mean_dice": np.mean(dice_vals),
            "median_dice": np.median(dice_vals),
            "mean_iou": np.mean(iou_vals),
            "mean_precision": np.mean(prec_vals),
            "mean_recall": np.mean(rec_vals),
        }

    group_stats = {name: stats(samples) for name, samples in groups.items()}

    # ---- Print summary table ----
    print()
    print(
        f"{'Group':>8s}  {'Count':>5s}  {'GT Pixel Range':>22s}  "
        f"{'Mean Dice':>9s}  {'Median Dice':>10s}  {'Mean Recall':>11s}"
    )
    print("-" * 82)
    for name in ["small", "medium", "large"]:
        s = group_stats[name]
        gtrange = f"[{s['min_gt_pixels']}, {s['max_gt_pixels']}]"
        print(
            f"{name:>8s}  {s['sample_count']:>5d}  {gtrange:>22s}  "
            f"{s['mean_dice']:>9.4f}  {s['median_dice']:>10.4f}  "
            f"{s['mean_recall']:>11.4f}"
        )
    print()

    # ---- Save group summary CSV ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group_name",
                "sample_count",
                "min_gt_pixels",
                "max_gt_pixels",
                "mean_gt_pixels",
                "mean_dice",
                "median_dice",
                "mean_iou",
                "mean_precision",
                "mean_recall",
            ]
        )
        for name in ["small", "medium", "large"]:
            s = group_stats[name]
            writer.writerow(
                [
                    name,
                    s["sample_count"],
                    s["min_gt_pixels"],
                    s["max_gt_pixels"],
                    f"{s['mean_gt_pixels']:.2f}",
                    f"{s['mean_dice']:.6f}",
                    f"{s['median_dice']:.6f}",
                    f"{s['mean_iou']:.6f}",
                    f"{s['mean_precision']:.6f}",
                    f"{s['mean_recall']:.6f}",
                ]
            )
    print(f"Group summary saved: {SUMMARY_CSV}")

    # ---- Scatter plots ----
    areas = np.array([r["gt_positive_pixels"] for r in records])
    dice_vals = np.array([r["dice"] for r in records])
    recall_vals = np.array([r["recall"] for r in records])

    def save_scatter(x, y, ylabel, filename):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x, y, alpha=0.7, edgecolor="white", s=40, color="steelblue")
        ax.set_xlabel("GT Positive Pixels", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Defect Area vs. {ylabel}", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Scatter plot saved: {filename}")

    save_scatter(areas, dice_vals, "Sample Dice", OUTPUT_DIR / "defect_area_vs_dice.png")
    save_scatter(areas, recall_vals, "Sample Recall", OUTPUT_DIR / "defect_area_vs_recall.png")

    print("Done.")


if __name__ == "__main__":
    main()

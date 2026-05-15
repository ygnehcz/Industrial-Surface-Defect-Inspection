"""Analyse boundary-touching vs. non-boundary defect segmentation performance.

Loads per-sample metrics and GT masks, classifies each defective sample
as boundary-touching or non-boundary, prints a comparison table, saves a
group CSV, and generates a bar chart comparing the two groups.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from datasets.ksdd2_dataset import KSDD2Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_CSV = "outputs/analysis/validation_sample_analysis/val_defective_sample_metrics.csv"
VAL_DIR = "data/processed/val"
OUTPUT_DIR = Path("outputs/analysis/boundary_vs_performance")
SUMMARY_CSV = OUTPUT_DIR / "boundary_group_summary.csv"
BAR_CHART = OUTPUT_DIR / "boundary_vs_non_boundary_metrics.png"
IMAGE_SIZE = (640, 256)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load metrics CSV ----
    csv_path = Path(METRICS_CSV)
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    metrics_map: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics_map[row["filename"]] = {
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "gt_positive_pixels": int(row["gt_positive_pixels"]),
                "pred_positive_pixels": int(row["pred_positive_pixels"]),
            }

    print(f"Input CSV: {csv_path}")
    print(f"Defective samples in CSV: {len(metrics_map)}")
    print()

    # ---- Build dataset ----
    dataset = KSDD2Dataset(VAL_DIR, image_size=IMAGE_SIZE, return_filename=True)

    # ---- Classify each defective sample ----
    boundary_samples: list[dict] = []
    non_boundary_samples: list[dict] = []

    touch_top = 0
    touch_bottom = 0
    touch_left = 0
    touch_right = 0

    for idx in range(len(dataset)):
        _, mask_tensor, fname = dataset[idx]

        if fname not in metrics_map:
            continue  # skip normal samples

        mask = mask_tensor.squeeze()  # [H, W], binary {0,1}
        top = bool(mask[0, :].any().item())
        bottom = bool(mask[-1, :].any().item())
        left = bool(mask[:, 0].any().item())
        right = bool(mask[:, -1].any().item())

        entry = {**metrics_map[fname], "filename": fname}
        if top or bottom or left or right:
            boundary_samples.append(entry)
            if top:
                touch_top += 1
            if bottom:
                touch_bottom += 1
            if left:
                touch_left += 1
            if right:
                touch_right += 1
        else:
            non_boundary_samples.append(entry)

    # ---- Group statistics ----
    def group_stats(samples: list[dict]) -> dict:
        if not samples:
            return {
                "sample_count": 0,
                "mean_dice": 0.0,
                "median_dice": 0.0,
                "mean_iou": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
            }
        d = [s["dice"] for s in samples]
        i = [s["iou"] for s in samples]
        p = [s["precision"] for s in samples]
        r = [s["recall"] for s in samples]
        return {
            "sample_count": len(samples),
            "mean_dice": float(np.mean(d)),
            "median_dice": float(np.median(d)),
            "mean_iou": float(np.mean(i)),
            "mean_precision": float(np.mean(p)),
            "mean_recall": float(np.mean(r)),
        }

    bound_stats = group_stats(boundary_samples)
    nonb_stats = group_stats(non_boundary_samples)

    # ---- Print summary table ----
    print(
        f"{'Group':>18s}  {'Count':>5s}  "
        f"{'Mean Dice':>9s}  {'Median Dice':>10s}  "
        f"{'Mean Precision':>14s}  {'Mean Recall':>11s}"
    )
    print("-" * 85)
    for label, stats in [("boundary_touching", bound_stats), ("non_boundary", nonb_stats)]:
        print(
            f"{label:>18s}  {stats['sample_count']:>5d}  "
            f"{stats['mean_dice']:>9.4f}  {stats['median_dice']:>10.4f}  "
            f"{stats['mean_precision']:>14.4f}  {stats['mean_recall']:>11.4f}"
        )
    print()

    # ---- Boundary detail counts ----
    print("Boundary-touching breakdown:")
    print(f"  Total boundary-touching: {bound_stats['sample_count']}")
    print(f"  Non-boundary:            {nonb_stats['sample_count']}")
    print(f"  Touch top:    {touch_top}")
    print(f"  Touch bottom: {touch_bottom}")
    print(f"  Touch left:   {touch_left}")
    print(f"  Touch right:  {touch_right}")
    print()

    # ---- Save summary CSV ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group_name",
                "sample_count",
                "mean_dice",
                "median_dice",
                "mean_iou",
                "mean_precision",
                "mean_recall",
            ]
        )
        for label, stats in [("boundary_touching", bound_stats), ("non_boundary", nonb_stats)]:
            writer.writerow(
                [
                    label,
                    stats["sample_count"],
                    f"{stats['mean_dice']:.6f}",
                    f"{stats['median_dice']:.6f}",
                    f"{stats['mean_iou']:.6f}",
                    f"{stats['mean_precision']:.6f}",
                    f"{stats['mean_recall']:.6f}",
                ]
            )
    print(f"Summary CSV saved: {SUMMARY_CSV}")

    # ---- Bar chart ----
    metric_names = ["Mean Dice", "Median Dice", "Mean Recall"]
    bound_vals = [bound_stats["mean_dice"], bound_stats["median_dice"], bound_stats["mean_recall"]]
    nonb_vals = [nonb_stats["mean_dice"], nonb_stats["median_dice"], nonb_stats["mean_recall"]]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, bound_vals, width, label="Boundary-touching", color="indianred", alpha=0.85)
    bars2 = ax.bar(x + width / 2, nonb_vals, width, label="Non-boundary", color="steelblue", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Boundary-touching vs. Non-boundary Defect Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Annotate bar values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(BAR_CHART, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved: {BAR_CHART}")
    print("Done.")


if __name__ == "__main__":
    main()

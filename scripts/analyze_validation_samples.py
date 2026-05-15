"""Per-sample error analysis on validation defective samples.

Loads the best checkpoint, computes per-sample segmentation metrics for
every defective validation sample, sorts by Dice, and generates best /
median / worst visualisation panels plus a CSV metrics table.
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
import torch

from datasets.ksdd2_dataset import KSDD2Dataset
from models.unet import UNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "outputs/checkpoints/best_unet.pth"
VAL_DIR = "data/processed/val"
OUTPUT_ROOT = "outputs/analysis/validation_sample_analysis"
CSV_PATH = Path(OUTPUT_ROOT) / "val_defective_sample_metrics.csv"
NUM_CASES_PER_GROUP = 6
DEFAULT_THRESHOLD = 0.5
EPS = 1e-6
OVERLAY_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_sample_metrics(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6
) -> dict[str, float]:
    """Compute per-sample Dice, IoU, Precision, Recall from binary masks."""
    pred = pred_mask.float()
    gt = gt_mask.float()

    TP = (pred * gt).sum().item()
    FP = (pred * (1.0 - gt)).sum().item()
    FN = ((1.0 - pred) * gt).sum().item()

    dice = (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": TP,
        "fp": FP,
        "fn": FN,
    }


def save_panel(
    img_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    fname: str,
    metrics: dict[str, float],
    out_dir: Path,
) -> None:
    """Save a 4-column panel: Original | GT | Prediction | Overlay."""
    overlay = img_np.copy()
    defect = pred_np > 0
    overlay[defect] = (
        OVERLAY_ALPHA * np.array([1.0, 0.0, 0.0])
        + (1.0 - OVERLAY_ALPHA) * overlay[defect]
    ).clip(0, 1)

    stem = Path(fname).stem
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["Original", "GT Mask", "Prediction", "Overlay"]
    images = [img_np, gt_np, pred_np, overlay]
    cmaps = [None, "gray", "gray", None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    suptitle = (
        f"{fname}  |  Dice: {metrics['dice']:.4f}  "
        f"Precision: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}"
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()

    save_path = out_dir / f"{stem}_dice_{metrics['dice']:.4f}_panel.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -- {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Load checkpoint ----
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_file():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", (640, 256))
    threshold = config.get("threshold", DEFAULT_THRESHOLD)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Threshold:  {threshold}")
    print(f"Image size: {image_size}")
    print()

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Data ----
    dataset = KSDD2Dataset(VAL_DIR, image_size=image_size, return_filename=True)
    print(f"Val samples: {len(dataset)}")

    # ---- Per-sample inference ----
    records: list[dict] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, mask_tensor, fname = dataset[idx]

            # Only defective samples
            if mask_tensor.sum() == 0:
                continue

            image_batch = image_tensor.unsqueeze(0).to(device)

            logits = model(image_batch)
            probs = torch.sigmoid(logits)
            pred_mask = (probs >= threshold).float().cpu().squeeze()
            gt_mask = mask_tensor.squeeze()

            metrics = compute_sample_metrics(pred_mask, gt_mask, eps=EPS)
            gt_pos = int(gt_mask.sum().item())
            pred_pos = int(pred_mask.sum().item())

            records.append(
                {
                    "filename": fname,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "gt_positive_pixels": gt_pos,
                    "pred_positive_pixels": pred_pos,
                    # Also keep tensors for visualisation later
                    "image_tensor": image_tensor,
                    "mask_tensor": mask_tensor,
                    "pred_mask": pred_mask,
                }
            )

    if not records:
        print("ERROR: No defective samples found in validation set.", file=sys.stderr)
        sys.exit(1)

    # ---- Sort by Dice (descending) ----
    records.sort(key=lambda r: r["dice"], reverse=True)
    n = len(records)

    # ---- Print summary ----
    dice_values = [r["dice"] for r in records]
    print(f"Defective val samples: {n}")
    print(f"Best Dice:   {dice_values[0]:.4f}  ({records[0]['filename']})")
    print(f"Worst Dice:  {dice_values[-1]:.4f}  ({records[-1]['filename']})")
    print(f"Mean Dice:   {np.mean(dice_values):.4f}")
    print(f"Median Dice: {np.median(dice_values):.4f}")
    print()

    # ---- Save CSV ----
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "dice",
                "iou",
                "precision",
                "recall",
                "gt_positive_pixels",
                "pred_positive_pixels",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "filename": r["filename"],
                    "dice": f"{r['dice']:.6f}",
                    "iou": f"{r['iou']:.6f}",
                    "precision": f"{r['precision']:.6f}",
                    "recall": f"{r['recall']:.6f}",
                    "gt_positive_pixels": r["gt_positive_pixels"],
                    "pred_positive_pixels": r["pred_positive_pixels"],
                }
            )
    print(f"CSV saved: {CSV_PATH}")
    print()

    # ---- Group selection ----
    best_cases = records[:NUM_CASES_PER_GROUP]
    worst_cases = records[-NUM_CASES_PER_GROUP:]

    mid = n // 2
    half = NUM_CASES_PER_GROUP // 2
    start = max(0, mid - half)
    end = min(n, start + NUM_CASES_PER_GROUP)
    median_cases = records[start:end]

    # ---- Generate panels ----
    for group_name, group_records in [
        ("best_cases", best_cases),
        ("median_cases", median_cases),
        ("worst_cases", worst_cases),
    ]:
        group_dir = out_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{group_name}]")
        for r in group_records:
            img_np = r["image_tensor"].permute(1, 2, 0).numpy()
            gt_np = r["mask_tensor"].squeeze().numpy()
            pred_np = r["pred_mask"].numpy()

            save_panel(
                img_np,
                gt_np,
                pred_np,
                r["filename"],
                {"dice": r["dice"], "precision": r["precision"], "recall": r["recall"]},
                group_dir,
            )
        print()

    print("Done.")


if __name__ == "__main__":
    main()

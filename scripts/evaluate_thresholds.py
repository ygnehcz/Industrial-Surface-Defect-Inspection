"""Threshold sweep on the validation set using the best U-Net checkpoint.

Runs forward pass once, then evaluates multiple binarisation thresholds
to compare Precision-Recall-Dice trade-offs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.ksdd2_dataset import KSDD2Dataset
from models.unet import UNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "outputs/checkpoints/best_unet.pth"
VAL_DIR = "data/processed/val"
BATCH_SIZE = 2
NUM_WORKERS = 0
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EPS = 1e-6


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
    print(f"Checkpoint: {ckpt_path}")
    print(f"Image size: {image_size}")
    print()

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Data ----
    dataset = KSDD2Dataset(VAL_DIR, image_size=image_size, return_filename=True)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"Val samples: {len(dataset)}")
    print(f"Thresholds:  {THRESHOLDS}")
    print()

    # ---- Accumulate TP/FP/FN per threshold ----
    # Use lists aligned with THRESHOLDS order
    accum_tp = [0.0] * len(THRESHOLDS)
    accum_fp = [0.0] * len(THRESHOLDS)
    accum_fn = [0.0] * len(THRESHOLDS)

    with torch.no_grad():
        for images, masks, *_ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            targets = masks.float()

            for i, t in enumerate(THRESHOLDS):
                preds = (probs >= t).float()
                tp = (preds * targets).sum().item()
                fp = (preds * (1.0 - targets)).sum().item()
                fn = ((1.0 - preds) * targets).sum().item()
                accum_tp[i] += tp
                accum_fp[i] += fp
                accum_fn[i] += fn

    # ---- Compute metrics ----
    results: list[dict] = []
    best_dice = -1.0
    best_threshold = None

    for i, t in enumerate(THRESHOLDS):
        TP = accum_tp[i]
        FP = accum_fp[i]
        FN = accum_fn[i]

        dice = (2.0 * TP + EPS) / (2.0 * TP + FP + FN + EPS)
        iou = (TP + EPS) / (TP + FP + FN + EPS)
        precision = (TP + EPS) / (TP + FP + EPS)
        recall = (TP + EPS) / (TP + FN + EPS)

        results.append(
            {
                "threshold": t,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            }
        )
        if dice > best_dice:
            best_dice = dice
            best_threshold = t

    # ---- Print table ----
    print(f"{'Threshold':>10s}  {'Dice':>8s}  {'IoU':>8s}  {'Precision':>10s}  {'Recall':>8s}")
    print("-" * 58)
    for r in results:
        print(
            f"{r['threshold']:10.2f}  "
            f"{r['dice']:8.4f}  "
            f"{r['iou']:8.4f}  "
            f"{r['precision']:10.4f}  "
            f"{r['recall']:8.4f}"
        )

    print()
    print(f"Best threshold by Dice: {best_threshold}")
    print(f"Best Dice:              {best_dice:.4f}")


if __name__ == "__main__":
    main()

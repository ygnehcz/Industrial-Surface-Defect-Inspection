"""First-version U-Net training script for binary defect segmentation.

Runs a train/val loop, prints per-epoch metrics, and saves the best
checkpoint by validation Dice score.
"""

from __future__ import annotations


from pathlib import Path

import torch
import torch.optim
from torch.utils.data import DataLoader

from datasets.ksdd2_dataset import KSDD2Dataset
from losses.segmentation_loss import BCEDiceLoss
from models.unet import UNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
OUTPUT_DIR = "outputs/checkpoints"

IMAGE_SIZE = (640, 256)  # (H, W)
EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
THRESHOLD = 0.5

SEED = 42

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_dataset = KSDD2Dataset(TRAIN_DIR, image_size=IMAGE_SIZE, return_filename=True)
val_dataset = KSDD2Dataset(VAL_DIR, image_size=IMAGE_SIZE, return_filename=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Epochs:        {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print()

# ---------------------------------------------------------------------------
# Model, loss, optimiser
# ---------------------------------------------------------------------------
model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, masks, *_ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate_one_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, dict[str, float]]:
    model.eval()

    running_loss = 0.0
    n_batches = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, masks, *_ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item()
        n_batches += 1

        # Pixel-level accumulation per batch
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        targets = masks.float()

        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute metrics once from globally accumulated counts
    TP = total_tp
    FP = total_fp
    FN = total_fn

    dice = (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    avg_loss = running_loss / max(n_batches, 1)
    avg_metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    best_val_dice = -1.0
    best_ckpt_path = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, threshold=THRESHOLD
        )

        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"  Train Loss     : {train_loss:.6f}")
        print(f"  Val Loss       : {val_loss:.6f}")
        print(f"  Val Dice       : {val_metrics['dice']:.4f}")
        print(f"  Val IoU        : {val_metrics['iou']:.4f}")
        print(f"  Val Precision  : {val_metrics['precision']:.4f}")
        print(f"  Val Recall     : {val_metrics['recall']:.4f}")

        # Checkpoint on best Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            best_ckpt_path = output_dir / "best_unet.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "epochs": EPOCHS,
                        "threshold": THRESHOLD,
                    },
                },
                best_ckpt_path,
            )
            print("  >> Best model saved.")
        print()

    print("Training finished.")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    if best_ckpt_path is not None:
        print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()

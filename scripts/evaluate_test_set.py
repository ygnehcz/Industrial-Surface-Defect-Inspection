"""Final test-set evaluation with the best checkpoint and fixed threshold.

Loads best_unet.pth, evaluates on data/processed/test with threshold=0.7,
and reports global pixel-level Dice / IoU / Precision / Recall.
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
TEST_DIR = "data/processed/test"
OUTPUT_DIR = Path("outputs/evaluation/test_set")
RESULT_TXT = OUTPUT_DIR / "test_set_metrics.txt"

BATCH_SIZE = 2
NUM_WORKERS = 0
THRESHOLD = 0.7
DEFAULT_IMAGE_SIZE = (640, 256)
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
    image_size = config.get("image_size", DEFAULT_IMAGE_SIZE)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Image size: {image_size}")
    print(f"Threshold:  {THRESHOLD}")
    print()

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Data ----
    dataset = KSDD2Dataset(TEST_DIR, image_size=image_size, return_filename=True)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ---- Inference ----
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    total_samples = 0
    defective_count = 0
    normal_count = 0

    with torch.no_grad():
        for images, masks, *_ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= THRESHOLD).float()
            targets = masks.float()

            tp = (preds * targets).sum().item()
            fp = (preds * (1.0 - targets)).sum().item()
            fn = ((1.0 - preds) * targets).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_samples += images.size(0)

            # Count defective / normal per sample in the batch
            for i in range(masks.size(0)):
                if masks[i].sum().item() > 0:
                    defective_count += 1
                else:
                    normal_count += 1

    # ---- Global metrics ----
    TP = total_tp
    FP = total_fp
    FN = total_fn

    dice = (2.0 * TP + EPS) / (2.0 * TP + FP + FN + EPS)
    iou = (TP + EPS) / (TP + FP + FN + EPS)
    precision = (TP + EPS) / (TP + FP + EPS)
    recall = (TP + EPS) / (TP + FN + EPS)

    # ---- Print & save results ----
    lines = [
        "Test Set Evaluation",
        "=" * 40,
        f"Checkpoint        : {ckpt_path}",
        f"Image size        : {image_size}",
        f"Threshold         : {THRESHOLD}",
        f"Total samples     : {total_samples}",
        f"Defective samples : {defective_count}",
        f"Normal samples    : {normal_count}",
        "-" * 40,
        f"Test Dice         : {dice:.4f}",
        f"Test IoU          : {iou:.4f}",
        f"Test Precision    : {precision:.4f}",
        f"Test Recall       : {recall:.4f}",
    ]

    for line in lines:
        print(line)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print(f"Results saved to: {RESULT_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()

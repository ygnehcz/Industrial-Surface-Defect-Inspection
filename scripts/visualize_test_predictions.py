"""Visualise U-Net predictions on defective test samples.

Loads the best checkpoint, runs inference on the first few defective
samples from the official test split, and saves 4-column prediction panels.
"""

from __future__ import annotations

import argparse
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
TEST_DIR = "data/processed/test"
OUTPUT_DIR = "outputs/figures/prediction_samples/test"
DEFAULT_NUM_SAMPLES = 6
DEFAULT_THRESHOLD = 0.7
OVERLAY_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise U-Net predictions on defective test samples."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of defective samples to visualise (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Binarisation threshold (default: {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load checkpoint ----
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_file():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", (640, 256))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Threshold:  {args.threshold}")
    print(f"Image size: {image_size}")

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Data ----
    dataset = KSDD2Dataset(TEST_DIR, image_size=image_size, return_filename=True)
    print(f"Test samples: {len(dataset)}")

    # ---- Collect defective samples ----
    defective_indices: list[int] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            if len(defective_indices) >= args.num_samples:
                break
            _, mask_tensor, _ = dataset[idx]
            if mask_tensor.sum().item() > 0:
                defective_indices.append(idx)

    if not defective_indices:
        print("ERROR: No defective samples found in test set.", file=sys.stderr)
        sys.exit(1)

    print(f"Defective samples to visualise: {len(defective_indices)}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference & visualisation ----
    with torch.no_grad():
        for idx in defective_indices:
            image_tensor, mask_tensor, fname = dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device)

            logits = model(image_batch)
            probs = torch.sigmoid(logits)
            pred_mask = (probs >= args.threshold).float().cpu().squeeze()

            img_np = image_tensor.permute(1, 2, 0).numpy()
            gt_np = mask_tensor.squeeze().numpy()
            pred_np = pred_mask.numpy()

            overlay = img_np.copy()
            defect = pred_np > 0
            overlay[defect] = (
                OVERLAY_ALPHA * np.array([1.0, 0.0, 0.0])
                + (1.0 - OVERLAY_ALPHA) * overlay[defect]
            ).clip(0, 1)

            stem = Path(fname).stem
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            titles = ["Original", "GT Mask", "Prediction", "Overlay (Red = Pred)"]
            images = [img_np, gt_np, pred_np, overlay]
            cmaps = [None, "gray", "gray", None]

            for ax, title, img, cmap in zip(axes, titles, images, cmaps):
                ax.imshow(img, cmap=cmap)
                ax.set_title(f"{title}\n{fname}  (t={args.threshold})", fontsize=10)
                ax.axis("off")

            fig.tight_layout()
            save_path = out_dir / f"{stem}_test_prediction_panel.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved -- {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()

"""Visualise U-Net predictions on defective validation samples.

Loads the best checkpoint, runs inference on the first few defective
samples from the validation set, and saves 4-column prediction panels.
"""

from __future__ import annotations

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
OUTPUT_DIR = "outputs/figures/prediction_samples/val"
NUM_SAMPLES = 6
DEFAULT_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Data ----
    dataset = KSDD2Dataset(VAL_DIR, image_size=image_size, return_filename=True)
    print(f"Val samples: {len(dataset)}")

    # ---- Collect defective samples ----
    defective_indices: list[int] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            if len(defective_indices) >= NUM_SAMPLES:
                break
            _, mask_tensor, _ = dataset[idx]
            if mask_tensor.sum() > 0:
                defective_indices.append(idx)

    if not defective_indices:
        print("ERROR: No defective samples found in validation set.", file=sys.stderr)
        sys.exit(1)

    print(f"Defective samples to visualise: {len(defective_indices)}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference & visualisation ----
    with torch.no_grad():
        for idx in defective_indices:
            image_tensor, mask_tensor, fname = dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

            logits = model(image_batch)
            probs = torch.sigmoid(logits)
            pred_mask = (probs >= threshold).float().cpu().squeeze()  # [H, W]

            # Convert tensors to numpy for plotting
            img_np = image_tensor.permute(1, 2, 0).numpy()             # (H, W, 3), RGB, [0,1]
            gt_np = mask_tensor.squeeze().numpy()                       # (H, W)
            pred_np = pred_mask.numpy()                                  # (H, W)

            # Build overlay: red on defect pixels
            overlay = img_np.copy()
            defect = pred_np > 0
            overlay[defect] = (
                OVERLAY_ALPHA * np.array([1.0, 0.0, 0.0])
                + (1.0 - OVERLAY_ALPHA) * overlay[defect]
            ).clip(0, 1)

            # Plot
            stem = Path(fname).stem
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            titles = ["Original", "GT Mask", "Prediction", "Overlay (Red = Pred)"]
            images = [img_np, gt_np, pred_np, overlay]
            cmaps = [None, "gray", "gray", None]

            for ax, title, img, cmap in zip(axes, titles, images, cmaps):
                ax.imshow(img, cmap=cmap)
                ax.set_title(f"{title}\n{fname}", fontsize=10)
                ax.axis("off")

            fig.tight_layout()
            save_path = out_dir / f"{stem}_prediction_panel.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved -- {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()

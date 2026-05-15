"""Single-image inference with the trained U-Net checkpoint.

Given an arbitrary image, loads the best checkpoint, runs inference,
and saves the predicted binary mask, red overlay, and a visualisation panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.unet import UNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "outputs/checkpoints/best_unet.pth"
OUTPUT_DIR = Path("outputs/inference/single_image")
DEFAULT_THRESHOLD = 0.7
DEFAULT_IMAGE_SIZE = (640, 256)  # (H, W)
OVERLAY_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run U-Net inference on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
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
    image_size = config.get("image_size", DEFAULT_IMAGE_SIZE)  # (H, W)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Image size: {image_size}")
    print(f"Threshold:  {args.threshold}")

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- Read & preprocess image ----
    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"ERROR: Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"ERROR: Failed to read image: {img_path}", file=sys.stderr)
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_size
    image_resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    image_float = image_resized.astype(np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0).to(device)
    )  # [1, 3, H, W]

    # ---- Inference ----
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs >= args.threshold).float().cpu().squeeze()  # [H, W]

    pred_positive_pixels = int(pred_mask.sum().item())
    detected = pred_positive_pixels > 0

    print(f"Input image:            {img_path}")
    print(f"Predicted positive pixels: {pred_positive_pixels}")
    print(f"Defect detected:        {'Yes' if detected else 'No'}")

    # ---- Save outputs ----
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    pred_np = pred_mask.numpy().astype(np.uint8) * 255  # [H, W], {0, 255}

    # Binary mask
    mask_path = out_dir / f"{stem}_pred_mask.png"
    cv2.imwrite(str(mask_path), pred_np)
    print(f"  Pred mask: {mask_path}")

    # Overlay
    overlay_rgb = image_resized.copy()
    defect = pred_mask.numpy() > 0
    overlay_rgb[defect] = (
        OVERLAY_ALPHA * np.array([255, 0, 0])
        + (1.0 - OVERLAY_ALPHA) * overlay_rgb[defect]
    ).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)
    print(f"  Overlay:   {overlay_path}")

    # Panel
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = [
        f"Original\n{img_path.name}",
        f"Predicted Mask\nThreshold={args.threshold}",
        f"Overlay\nDefect: {'Yes' if detected else 'No'}",
    ]
    images = [
        image_resized,
        pred_mask.numpy(),
        overlay_rgb,
    ]
    cmaps = [None, "gray", None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    panel_path = out_dir / f"{stem}_prediction_panel.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Panel:     {panel_path}")

    print("Done.")


if __name__ == "__main__":
    main()

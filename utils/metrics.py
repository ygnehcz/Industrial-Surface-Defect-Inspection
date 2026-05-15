"""Evaluation metrics for binary semantic segmentation.

Operates on raw logits from the U-Net and binary ground-truth masks.
Aggregates per-pixel TP / FP / FN across the entire batch.
"""

from __future__ import annotations

import torch


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute pixel-level Dice, IoU, precision, and recall for one batch.

    Args:
        logits: Raw model output, shape ``[B, 1, H, W]``.
        targets: Binary ground-truth mask, shape ``[B, 1, H, W]``.
        threshold: Binarisation threshold applied after sigmoid.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict with keys ``"dice"``, ``"iou"``, ``"precision"``, ``"recall"``.

    Raises:
        ValueError: If ``logits`` and ``targets`` have different shapes.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs "
            f"targets {tuple(targets.shape)}"
        )

    targets = targets.float()

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    # Accumulate across the entire batch
    TP = (preds * targets).sum()
    FP = (preds * (1.0 - targets)).sum()
    FN = ((1.0 - preds) * targets).sum()

    dice = (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Metrics self-test")
    print("-" * 40)

    logits = torch.randn(2, 1, 640, 256)
    targets = torch.randint(0, 2, (2, 1, 640, 256)).float()

    metrics = compute_segmentation_metrics(logits, targets)

    for key, value in metrics.items():
        print(f"  {key:<10s} = {value:.6f}")
    print("Done.")

"""Loss functions for binary semantic segmentation.

DiceLoss and BCEDiceLoss operate on raw logits from the U-Net and
binary ground-truth masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Applies sigmoid to logits, then computes the Sørensen–Dice
    coefficient per sample.  Returns ``1 - mean(dice)``.

    Args:
        smooth: Small constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs "
                f"targets {tuple(targets.shape)}"
            )
        targets = targets.float()

        probs = torch.sigmoid(logits)

        # Flatten each sample to [B, H*W]
        probs_flat = probs.flatten(start_dim=1)
        targets_flat = targets.flatten(start_dim=1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        probs_sum = probs_flat.sum(dim=1)
        targets_sum = targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (
            probs_sum + targets_sum + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE (with logits) and Dice loss.

    ``loss = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss``

    Args:
        bce_weight: Weight for the BCE term.
        dice_weight: Weight for the Dice term.
        smooth: Smoothing constant forwarded to ``DiceLoss``.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs "
                f"targets {tuple(targets.shape)}"
            )
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Segmentation loss self-test")
    print("-" * 40)

    logits = torch.randn(2, 1, 640, 256)
    targets = torch.randint(0, 2, (2, 1, 640, 256)).float()

    criterion_dice = DiceLoss(smooth=1e-6)
    criterion_bce_dice = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-6)

    loss_dice = criterion_dice(logits, targets)
    loss_bce_dice = criterion_bce_dice(logits, targets)

    print(f"DiceLoss      = {loss_dice.item():.6f}")
    print(f"BCEDiceLoss   = {loss_bce_dice.item():.6f}")
    print("Done.")

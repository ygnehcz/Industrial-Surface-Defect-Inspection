"""Standard U-Net for binary semantic segmentation.

Input:  RGB image tensor  [B, 3, H, W]
Output: single-channel defect logits  [B, 1, H, W]

Output values are raw logits (no sigmoid). Use BCEWithLogitsLoss or
apply torch.sigmoid to obtain probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU (spatial size preserved)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """MaxPool(2) → DoubleConv (halves spatial size, doubles channels)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d (scale=2) → concat skip → DoubleConv.

    If the upsampled feature map and the skip connection differ slightly
    in spatial size, padding or cropping is applied before concatenation.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad if the upsampled feature map is smaller than the skip
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        pad_h = max(dh, 0)
        pad_w = max(dw, 0)
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, [0, pad_w, 0, pad_h])

        # Crop if the upsampled feature map is larger than the skip
        if x.size(2) > skip.size(2) or x.size(3) > skip.size(3):
            x = x[:, :, : skip.size(2), : skip.size(3)]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """U-Net for binary defect segmentation.

    Encoder-decoder with skip connections.  The final 1×1 convolution
    produces raw logits — apply BCEWithLogitsLoss during training or
    sigmoid for probabilities at inference time.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        out_channels: Number of output channels (default 1 for binary).
        base_channels: Feature width of the first encoder block (default 32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        ch = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, ch)          #  -> ch
        self.down1 = Down(ch, ch * 2)                   # ch -> ch*2
        self.down2 = Down(ch * 2, ch * 4)               # ch*2 -> ch*4
        self.down3 = Down(ch * 4, ch * 8)               # ch*4 -> ch*8
        self.down4 = Down(ch * 8, ch * 16)              # ch*8 -> ch*16

        # Decoder
        self.up1 = Up(ch * 16, ch * 8)                  # ch*16 -> ch*8
        self.up2 = Up(ch * 8, ch * 4)                   # ch*8 -> ch*4
        self.up3 = Up(ch * 4, ch * 2)                   # ch*4 -> ch*2
        self.up4 = Up(ch * 2, ch)                        # ch*2 -> ch

        # Output layer — raw logits, no activation
        self.outc = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.inc(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)

        # Decoder with skip connections
        d3 = self.up1(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up3(d2, e1)
        d0 = self.up4(d1, e0)

        return self.outc(d0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("U-Net self-test")
    print("-" * 40)

    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    x = torch.randn(2, 3, 640, 256)
    with torch.no_grad():
        out = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    expected = (2, 1, 640, 256)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("Output shape matches expected [2, 1, 640, 256].")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

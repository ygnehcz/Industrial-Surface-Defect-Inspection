"""PyTorch Dataset for the KSDD2 processed split directories.

Each split (train/val/test) shares the same layout:

    {root_dir}/
    ├── images/   # 3-channel RGB originals
    └── masks/    # 1-channel ground-truth (non-zero = defect)

Images and masks are resized to a fixed spatial size on load because
original samples vary in dimensions.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KSDD2Dataset(Dataset):
    """Loads KSDD2 image-mask pairs from a single processed split.

    Args:
        root_dir: Path to a processed split directory
                  (e.g. ``data/processed/train``).
        image_size: Target (height, width) for resizing.  Default ``(640, 256)``.
        return_filename: If True, ``__getitem__`` also returns the image filename.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: tuple[int, int] = (640, 256),
        return_filename: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size  # (H, W)
        self.return_filename = return_filename

        images_dir = self.root_dir / "images"
        masks_dir = self.root_dir / "masks"

        # ---- existence checks ----
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

        # ---- collect filenames ----
        image_names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
        mask_names = sorted(p.name for p in masks_dir.iterdir() if p.is_file())

        # ---- count check ----
        if len(image_names) != len(mask_names):
            raise RuntimeError(
                f"File count mismatch in {self.root_dir}: "
                f"{len(image_names)} images vs {len(mask_names)} masks"
            )

        # ---- filename parity check ----
        if image_names != mask_names:
            only_images = set(image_names) - set(mask_names)
            only_masks = set(mask_names) - set(image_names)
            msg_parts = [f"Filename mismatch in {self.root_dir}:"]
            if only_images:
                msg_parts.append(
                    f"  Only in images/: {sorted(only_images)[:5]}..."
                )
            if only_masks:
                msg_parts.append(
                    f"  Only in masks/:  {sorted(only_masks)[:5]}..."
                )
            raise RuntimeError("\n".join(msg_parts))

        self._image_dir = images_dir
        self._mask_dir = masks_dir
        self._filenames = image_names

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str] | tuple[torch.Tensor, torch.Tensor]:
        fname = self._filenames[idx]
        img_path = self._image_dir / fname
        msk_path = self._mask_dir / fname

        # ---- read ----
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")

        # ---- colour conversion ----
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ---- resize ----
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # ---- normalise & binarise ----
        image = image.astype(np.float32) / 255.0               # [0, 1], (H, W, 3)
        mask = (mask > 0).astype(np.float32)                    # {0, 1}, (H, W)

        # ---- channel-last → channel-first ----
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)

        if self.return_filename:
            return image_tensor, mask_tensor, fname
        return image_tensor, mask_tensor

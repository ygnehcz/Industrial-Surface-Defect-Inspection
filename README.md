# Industrial Surface Defect Inspection

An industrial surface defect detection and segmentation project built on the
[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) dataset.  The
goal is to develop a portfolio-grade pipeline that covers the full defect
inspection workflow — from data understanding and classical baselines through
deep-learning segmentation to evaluation and interactive visualisation.

> **Current status:** baseline U-Net training and validation error analysis completed.
> Further optimisation, final test-set evaluation, and deployment work are still pending.

## Planned Pipeline

| Stage                         | Description                                              | Status     |
|-------------------------------|----------------------------------------------------------|------------|
| Dataset inspection            | Per-split sample counts; defective / normal statistics   | Done       |
| Defect sample visualisation   | Triplet figures (original, GT mask, overlay)             | Done       |
| Defect area analysis          | Pixel-level defect distribution & histograms             | Done       |
| Dataset preparation           | Stratified train/val split & processed directory layout  | Done       |
| PyTorch dataset loader        | Reusable image-mask dataset class and DataLoader check   | Done       |
| Classical CV baseline         | Traditional image-processing defect-detection pipeline   | Planned    |
| U-Net segmentation model      | Baseline encoder-decoder segmentation network with skip connections | Done       |
| Segmentation losses           | Dice loss and combined BCE-Dice loss for binary defect segmentation | Done       |
| Segmentation metrics          | Dice, IoU, pixel precision, and pixel recall for binary defect segmentation | Done       |
| Training pipeline             | Train/validation loop, checkpoint saving, and baseline U-Net optimisation | Done       |
| Prediction visualisation      | Qualitative panels for validation-set defect predictions  | Done       |
| Threshold sweep analysis      | Validation-set threshold comparison for Dice/IoU/precision/recall trade-offs | Done       |
| Per-sample validation analysis | Ranking defective validation samples by Dice and visualising best/median/worst cases | Done       |
| Area-performance analysis     | Relating GT defect area to Dice/Recall behaviour across validation samples | Done       |
| Boundary-performance analysis | Comparing boundary-touching and non-boundary defect segmentation quality | Done       |
| Evaluation & error analysis   | Metrics (IoU, Dice), confusion matrices, per-sample QA   | Planned    |
| Interactive demo              | Web-based or CLI demo for live inference on user images  | Planned    |

## Dataset Overview

Source: KolektorSDD2 — high-resolution industrial surface images with
pixel-level defect annotations.

| Split   | Total Samples | Defective | Normal |
|---------|---------------|-----------|--------|
| Train   | 2,331         | 246       | 2,085  |
| Test    | 1,004         | 110       | 894    |
| **All** | **3,335**     | **356**   | **2,979** |

## Processed Dataset Split

The official training split is further divided into train and validation
subsets using a fixed-seed stratified 80/20 split, while the official test
split is kept unchanged.

| Split  | Total Samples | Defective | Normal |
|--------|---------------|-----------|--------|
| Train  | 1,865         | 197       | 1,668  |
| Val    | 466           | 49        | 417    |
| Test   | 1,004         | 110       | 894    |

## Current Progress

Five data inspection, analysis, and preparation scripts have been implemented:

| Script                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `scripts/inspect_dataset.py`    | Count masks per split and classify defective vs. normal              |
| `scripts/visualize_samples.py`  | Generate triplet figures for selected defective samples              |
| `scripts/analyze_defect_area.py`| Compute per-defect pixel counts, ratios, and histograms              |
| `scripts/prepare_dataset.py`    | Build processed train/val/test directories with stratified split     |
| `scripts/inspect_processed_shapes.py` | Verify processed image-mask pairing, shape consistency, and channels |

The scripts are self-contained and use lightweight dependencies such as
`pathlib`, `cv2`, `numpy`, and `matplotlib` where needed. They include error
handling for missing directories, unreadable files, and insufficient samples.

### PyTorch Dataset Layer

- `datasets/ksdd2_dataset.py` has been implemented.
- It loads processed image-mask pairs, converts BGR images to RGB, resizes
  samples to 640 × 256, normalises images to [0, 1], binarises masks to
  {0, 1}, and returns PyTorch tensors.
- Dataset output shape has been verified as:
  - image: `[3, 640, 256]`
  - mask: `[1, 640, 256]`
- DataLoader batching has also been verified:
  - images: `[4, 3, 640, 256]`
  - masks: `[4, 1, 640, 256]`

### Baseline U-Net Model

- `models/unet.py` has been implemented as the first segmentation baseline.
- The model uses a standard encoder-decoder U-Net structure with skip
  connections.
- It accepts RGB image batches shaped `[B, 3, 640, 256]`.
- It outputs single-channel segmentation logits shaped `[B, 1, 640, 256]`.
- A forward-pass self-test has been completed successfully.
- The current baseline configuration contains 7,763,041 trainable parameters.
- The output is raw logits without sigmoid, so sigmoid or loss functions such
  as `BCEWithLogitsLoss` will be applied later in training/inference logic.

### Segmentation Losses

- `losses/segmentation_loss.py` has been implemented.
- It provides:
  - `DiceLoss`
  - `BCEDiceLoss`
- `DiceLoss` computes region-overlap supervision from sigmoid probabilities.
- `BCEDiceLoss` combines pixel-wise binary cross-entropy with region-level
  Dice supervision.
- The current baseline uses equal weights for BCE and Dice terms by default.
- A self-test has confirmed that both loss functions run successfully on
  tensors shaped `[B, 1, 640, 256]`.

### Segmentation Metrics

- `utils/metrics.py` has been implemented.
- It provides pixel-level binary segmentation metrics:
  - Dice score
  - IoU
  - Precision
  - Recall
- Raw logits are first converted to probabilities with sigmoid, then
  thresholded into binary predictions.
- The metrics are computed from aggregated TP, FP, and FN counts across the
  batch.
- A self-test has confirmed that the metrics run successfully on tensors
  shaped `[B, 1, 640, 256]`.

### Baseline Training Pipeline

- `train.py` has been implemented.
- It combines the processed dataset, U-Net, BCE-Dice loss, validation
  metrics, and best-checkpoint saving.
- The current follow-up baseline run used:
  - 20 epochs
  - batch size 2
  - learning rate 1e-3
  - Adam optimiser
  - validation checkpointing by Dice score
- The best checkpoint was saved at epoch 1 with:
  - Val Dice: `0.4540`
  - Val IoU: `0.2936`
  - Val Precision: `0.6755`
  - Val Recall: `0.3418`

### Prediction Visualisation

- `scripts/visualize_predictions.py` has been implemented.
- It loads the best checkpoint and produces four-panel validation examples:
  - original image
  - GT mask
  - predicted binary mask
  - red prediction overlay
- Initial visual inspection shows that the baseline U-Net can localise
  several defect regions, but predicted masks are often fragmented and
  under-cover the full ground-truth area.

### Threshold Sweep Analysis

- `scripts/evaluate_thresholds.py` has been implemented.
- It evaluates thresholds `0.1` to `0.7` on the full validation split using
  globally accumulated TP/FP/FN counts.
- The current best validation Dice is achieved at threshold `0.5`:

  | Threshold | Dice   | IoU    | Precision | Recall |
  |-----------|--------|--------|-----------|--------|
  | 0.40      | 0.4530 | 0.2928 | 0.6521    | 0.3470 |
  | 0.50      | 0.4540 | 0.2936 | 0.6755    | 0.3418 |
  | 0.60      | 0.4539 | 0.2936 | 0.6965    | 0.3367 |
  | 0.70      | 0.4536 | 0.2933 | 0.7182    | 0.3314 |

- The current checkpoint is relatively stable across thresholds 0.4–0.7,
  with the best Dice obtained at threshold 0.5.  Increasing the threshold
  improves precision while gradually reducing recall, indicating a normal
  precision-recall trade-off rather than a need for threshold relaxation.

### Per-sample Validation Analysis

- `scripts/analyze_validation_samples.py` has been implemented.
- It evaluates all 49 defective validation samples individually, saves a
  CSV metrics table, and produces best/median/worst prediction panels.
- Current defective-sample statistics:
  - Best sample Dice: `0.9320`
  - Worst sample Dice: `0.0000`
  - Mean sample Dice: `0.4878`
  - Median sample Dice: `0.6524`
- This shows that the model already segments many defect samples well, but
  a subset of difficult cases is still completely missed.

### Area vs. Performance Analysis

- `scripts/analyze_area_vs_performance.py` has been implemented.
- It groups defective validation samples by GT defect area and compares
  segmentation quality.
- Summary table:

  | Group  | GT Pixel Range | Mean Dice | Median Dice | Mean Recall |
  |--------|----------------|-----------|-------------|-------------|
  | Small  | 110–1249       | 0.4411    | 0.6524      | 0.4741      |
  | Medium | 1346–4179      | 0.6736    | 0.8038      | 0.6819      |
  | Large  | 4307–35464     | 0.3516    | 0.1139      | 0.3342      |

- The baseline performs most consistently on medium-sized defects.
  Large-area defects show the weakest median Dice and the strongest
  performance dispersion, indicating that defect area alone is not a
  monotonic predictor of segmentation quality.

### Boundary vs. Performance Analysis

- `scripts/analyze_boundary_vs_performance.py` has been implemented.
- It compares boundary-touching defect samples with non-boundary defects.
- Summary table:

  | Group              | Count | Mean Dice | Median Dice | Mean Precision | Mean Recall |
  |--------------------|-------|-----------|-------------|----------------|-------------|
  | Boundary-touching  | 24    | 0.4485    | 0.4836      | 0.6975         | 0.4328      |
  | Non-boundary       | 25    | 0.5256    | 0.6587      | 0.6789         | 0.5572      |

- Boundary-touching defects show lower median Dice and lower mean Recall,
  suggesting that border-adjacent defects are one important failure mode
  of the current baseline.

## Preliminary Data Findings

- **Class imbalance** — normal samples outnumber defective samples roughly
  8.4:1.  A classification-only metric would be misleading without accounting
  for this skew.
- **Small defect regions** — the median defect-to-image ratio is **1.56%**,
  indicating that most defects occupy a very small fraction of the image.
- **High variance** — defect ratios range from **0.017%** to **30.16%**,
  covering more than three orders of magnitude.
  This wide range suggests that segmentation models may need to handle
  defects across very different scales.
- **Evaluation strategy** — downstream evaluation must go beyond
  image-level accuracy and include region-level segmentation metrics (IoU,
  Dice, precision/recall at the pixel level).

## Project Layout

```
.
├── classical_cv/          # Future: classical CV baseline code
├── configs/               # Future: configuration files
├── train.py               # First U-Net baseline training script
├── data/                  # Raw and processed data
│   ├── raw/
│   │   ├── train/         # 2,331 images and 2,331 masks
│   │   └── test/          # 1,004 images and 1,004 masks
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
├── datasets/              # PyTorch dataset loaders
│   └── ksdd2_dataset.py
├── docs/                  # Documentation
├── losses/                # Segmentation loss functions
│   └── segmentation_loss.py
├── models/                # Segmentation model definitions
│   └── unet.py
├── outputs/               # Generated figures and analysis results
│   ├── figures/
│   │   ├── dataset_samples/
│   │   └── defect_area_analysis/
├── scripts/               # Data inspection, analysis, and preparation scripts
│   ├── inspect_dataset.py
│   ├── visualize_samples.py
│   ├── analyze_defect_area.py
│   ├── prepare_dataset.py
│   ├── inspect_processed_shapes.py
│   ├── visualize_predictions.py
│   ├── evaluate_thresholds.py
│   ├── analyze_validation_samples.py
│   ├── analyze_area_vs_performance.py
│   └── analyze_boundary_vs_performance.py
├── utils/                 # Evaluation utilities
│   └── metrics.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The project targets **Python 3.11**.  Core dependencies currently include `numpy`, `opencv-python`, `matplotlib`,
`torch`, and `torchvision`.

## Current Usage

```bash
# Per-split defective / normal counts
python scripts/inspect_dataset.py

# Triplet figures (original, GT mask, overlay) for 6 defective train samples
python scripts/visualize_samples.py --split train --num-samples 6

# Defect-area distribution statistics and histograms
python scripts/analyze_defect_area.py

# Prepare processed train/val/test directories
python scripts/prepare_dataset.py

# Verify processed image-mask pairing, shapes, and channels
python scripts/inspect_processed_shapes.py

# Run U-Net forward-pass self-test
python models/unet.py

# Run segmentation-loss self-test
python losses/segmentation_loss.py

# Run segmentation-metrics self-test
python utils/metrics.py

# Train the first U-Net baseline
python train.py

# Visualise validation predictions from the best checkpoint
python scripts/visualize_predictions.py

# Sweep validation thresholds
python scripts/evaluate_thresholds.py

# Analyse per-sample validation performance
python scripts/analyze_validation_samples.py

# Analyse defect area versus validation performance
python scripts/analyze_area_vs_performance.py

# Compare boundary-touching and non-boundary defects
python scripts/analyze_boundary_vs_performance.py
```

Output figures are written under `outputs/figures/`.

A basic DataLoader sanity check has confirmed that batches can be assembled as
`[B, 3, 640, 256]` images and `[B, 1, 640, 256]` masks.

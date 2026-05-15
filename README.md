# Industrial Surface Defect Inspection

An industrial surface defect detection and segmentation project built on the
[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) dataset.  The
goal is to develop a portfolio-grade pipeline that covers the full defect
inspection workflow — from data understanding and classical baselines through
deep-learning segmentation to evaluation and interactive visualisation.

> **Current status:** baseline U-Net training, validation error analysis, and fixed-threshold test-set evaluation completed.
> Further optimisation and deployment work are still pending.

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
| Test-set evaluation           | Fixed-threshold final evaluation on the official test split | Done       |
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
- The current optimised baseline run used:
  - 20 epochs
  - batch size 2
  - learning rate 3e-4
  - Adam optimiser
  - validation checkpointing by Dice score
- The best checkpoint was saved at epoch 14 with:
  - Val Dice: `0.4934`
  - Val IoU: `0.3275`
  - Val Precision: `0.4208`
  - Val Recall: `0.5964`
- After threshold sweeping, the best validation Dice improves further to
  `0.5016` at threshold `0.7`.

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
- The current best validation Dice is achieved at threshold `0.7`:

  | Threshold | Dice   | IoU    | Precision | Recall |
  |-----------|--------|--------|-----------|--------|
  | 0.40      | 0.4896 | 0.3241 | 0.4093    | 0.6089 |
  | 0.50      | 0.4934 | 0.3275 | 0.4208    | 0.5964 |
  | 0.60      | 0.4970 | 0.3307 | 0.4327    | 0.5837 |
  | 0.70      | 0.5016 | 0.3347 | 0.4486    | 0.5687 |

- The current optimised checkpoint improves steadily across thresholds 0.4–0.7,
  with the best Dice obtained at threshold 0.7.  Increasing the threshold
  raises precision while gradually reducing recall, and the `0.7` setting
  provides the strongest Dice/IoU balance for the current model.

### Per-sample Validation Analysis

- `scripts/analyze_validation_samples.py` has been implemented.
- It evaluates all 49 defective validation samples individually, saves a
  CSV metrics table, and produces best/median/worst prediction panels.
- Current defective-sample statistics:
  - Best sample Dice: `0.9335`
  - Worst sample Dice: `0.0000`
  - Mean sample Dice: `0.6237`
  - Median sample Dice: `0.7194`
- This shows that the optimised model segments a substantial share of defect
  samples well, although a small subset of difficult cases is still
  completely missed.

### Area vs. Performance Analysis

- `scripts/analyze_area_vs_performance.py` has been implemented.
- It groups defective validation samples by GT defect area and compares
  segmentation quality.
- Summary table:

  | Group  | GT Pixel Range | Mean Dice | Median Dice | Mean Recall |
  |--------|----------------|-----------|-------------|-------------|
  | Small  | 110–1249       | 0.5940    | 0.5993      | 0.7469      |
  | Medium | 1346–4179      | 0.6866    | 0.7662      | 0.8299      |
  | Large  | 4307–35464     | 0.5923    | 0.7078      | 0.5604      |

- The optimised baseline still performs best on medium-sized defects, but
  small- and large-area groups both improve substantially.  Large-area defects
  no longer show the severe collapse observed in the earlier baseline, although
  their recall remains lower than that of medium-sized defects.

### Boundary vs. Performance Analysis

- `scripts/analyze_boundary_vs_performance.py` has been implemented.
- It compares boundary-touching defect samples with non-boundary defects.
- Summary table:

  | Group              | Count | Mean Dice | Median Dice | Mean Precision | Mean Recall |
  |--------------------|-------|-----------|-------------|----------------|-------------|
  | Boundary-touching  | 24    | 0.5999    | 0.7183      | 0.7252         | 0.6413      |
  | Non-boundary       | 25    | 0.6465    | 0.7486      | 0.6905         | 0.7821      |

- Boundary-touching defects remain somewhat harder than non-boundary defects,
  especially in recall, but the performance gap narrows substantially after
  learning-rate optimisation.

### Final Test-set Evaluation

- `scripts/evaluate_test_set.py` has been implemented.
- The official test split is evaluated once using the selected best
  checkpoint and the validation-selected fixed threshold `0.7`.
- Test split composition:
  - Total samples: `1004`
  - Defective samples: `110`
  - Normal samples: `894`
- Final test-set metrics:

  | Metric    | Value  |
  |-----------|--------|
  | Dice      | 0.5486 |
  | IoU       | 0.3780 |
  | Precision | 0.5480 |
  | Recall    | 0.5492 |

- The test-set precision and recall are closely balanced, and the final Dice
  exceeds the validation Dice obtained during threshold selection.

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
│   ├── analyze_boundary_vs_performance.py
│   └── evaluate_test_set.py
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
python scripts/visualize_predictions.py --threshold 0.7

# Sweep validation thresholds
python scripts/evaluate_thresholds.py

# Analyse per-sample validation performance
python scripts/analyze_validation_samples.py --threshold 0.7

# Analyse defect area versus validation performance
python scripts/analyze_area_vs_performance.py

# Compare boundary-touching and non-boundary defects
python scripts/analyze_boundary_vs_performance.py

# Evaluate the selected checkpoint once on the official test split
python scripts/evaluate_test_set.py
```

Output figures are written under `outputs/figures/`.

A basic DataLoader sanity check has confirmed that batches can be assembled as
`[B, 3, 640, 256]` images and `[B, 1, 640, 256]` masks.

# Industrial Surface Defect Inspection

An industrial surface defect detection and segmentation project built on the
[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) dataset.  The
goal is to develop a portfolio-grade pipeline that covers the full defect
inspection workflow — from data understanding and classical baselines through
deep-learning segmentation to evaluation and interactive visualisation.

> **Current status:** data understanding & analysis phase. No models have been
> trained yet.

## Planned Pipeline

| Stage                         | Description                                              | Status     |
|-------------------------------|----------------------------------------------------------|------------|
| Dataset inspection            | Per-split sample counts; defective / normal statistics   | Done       |
| Defect sample visualisation   | Triplet figures (original, GT mask, overlay)             | Done       |
| Defect area analysis          | Pixel-level defect distribution & histograms             | Done       |
| Dataset preparation           | Stratified train/val split & processed directory layout  | Done       |
| Classical CV baseline         | Traditional image-processing defect-detection pipeline   | Planned    |
| U-Net segmentation model      | Deep-learning semantic segmentation of defects           | Planned    |
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

Four data preparation and analysis scripts have been implemented:

| Script                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `scripts/inspect_dataset.py`    | Count masks per split and classify defective vs. normal              |
| `scripts/visualize_samples.py`  | Generate triplet figures for selected defective samples              |
| `scripts/analyze_defect_area.py`| Compute per-defect pixel counts, ratios, and histograms              |
| `scripts/prepare_dataset.py`    | Build processed train/val/test directories with stratified split     |

The scripts are self-contained and use lightweight dependencies such as
`pathlib`, `cv2`, `numpy`, and `matplotlib` where needed. They include error
handling for missing directories, unreadable files, and insufficient samples.

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
├── datasets/              # Future: PyTorch Dataset classes
├── docs/                  # Documentation
├── losses/                # Future: custom loss functions
├── models/                # Future: model definitions
├── outputs/               # Generated figures and analysis results
│   ├── figures/
│   │   ├── dataset_samples/
│   │   └── defect_area_analysis/
├── scripts/               # Data inspection, analysis, and preparation scripts
│   ├── inspect_dataset.py
│   ├── visualize_samples.py
│   ├── analyze_defect_area.py
│   └── prepare_dataset.py
├── utils/                 # Future: utility helpers
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The project targets **Python 3.11**.  Core dependencies are `numpy`,
`opencv-python`, and `matplotlib`.

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
```

Output figures are written under `outputs/figures/`.

# Reproducibility Guide

This document describes how to reproduce the current paper-style experiment.

## Clone Repositories

Clone the ACDLR repository:

```bash
git clone <URL_DO_REPOSITORIO_ACDLR> ACDLR
cd ACDLR
```

Clone the CNN comparison repository into the expected local folder:

```bash
mkdir -p external
git clone https://github.com/sydney-machine-learning/crater-identification.git external/crater-identification
```

The ACDLR method does not use this repository internally. It is used only as
the YOLOv11/CNN comparison reference.

## Environment

Install with pip:

```bash
python -m pip install -r requirements.txt
```

Or create a Conda environment:

```bash
conda env create -f environment.yml
conda activate acdlr
```

## Dataset Placement

Expected dataset path:

```text
data/LU3M6TGT_yolo_format
```

Expected folders:

```text
train/images
train/labels
valid/images
valid/labels
```

The dataset is not versioned in Git. Copy or download it manually into this
folder before running benchmarks.

## Reproduce Current Smoke Test

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 3 ^
  --visual-count 3 ^
  --skip-cnn-train
```

If CNN weights are missing:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 3 ^
  --visual-count 3 ^
  --force-cnn-train
```

Expected output files:

```text
artifacts/acdlr_vs_crater_cnn/comparison_report.md
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_summary.json
```

## Current Recorded Smoke-Test Result

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| ACDLR | 0.6294 | 0.5114 | 0.5643 |
| CNN YOLOv11 | 0.3241 | 0.3324 | 0.3282 |

## Reproduce A Larger Subset

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 25 ^
  --visual-count 8 ^
  --skip-cnn-train
```

## Reproduce With Fresh CNN Training

CPU quick baseline:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 25 ^
  --force-cnn-train ^
  --cnn-train-epochs 1 ^
  --cnn-train-fraction 0.02
```

GPU/full-data baseline:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 100 ^
  --force-cnn-train ^
  --cnn-train-epochs 50 ^
  --cnn-train-fraction 1.0 ^
  --cnn-device 0
```

## Determinism Notes

- ACDLR is deterministic for a fixed image order and fixed parameters.
- CNN training can vary by hardware, seed, PyTorch/Ultralytics version and GPU
  kernels.
- Generated artifacts are ignored by Git; preserve final selected outputs in
  `reports/` or document their paths in the paper.

## Artifact Checklist

For each reported experiment, save:

- exact command;
- dataset split and number of images;
- ACDLR parameter values;
- CNN weights path and training settings;
- summary JSON;
- CSV with per-image metrics;
- visual examples;
- software environment.

# Reproducibility Guide

This document describes how to reproduce the current fair visual-dataset
experiment: ACDLR classical image processing versus pretrained Ellipse R-CNN.

## Clone Repositories

```bash
git clone <URL_DO_REPOSITORIO_ACDLR> ACDLR
cd ACDLR
```

Prepare Ellipse R-CNN:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Equivalent manual clone:

```bash
mkdir -p external
git clone https://github.com/wdoppenberg/ellipse-rcnn.git external/ellipse-rcnn
```

## Environment

Install base dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the visual CNN baseline:

```bash
python -m pip install -e "external/ellipse-rcnn[hf]"
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

## Reproduce Smoke Test

```powershell
python scripts\run_acdlr_vs_ellipse_rcnn_comparison.py ^
  --max-images 3 ^
  --visual-count 2
```

Expected output files:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
artifacts/acdlr_vs_ellipse_rcnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_ellipse_rcnn/ellipse_rcnn/ellipse_rcnn_yolo_summary.json
```

## Download Note

The Hugging Face model file is:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

If automatic download fails because the network cannot resolve
`cas-bridge.xethub.hf.co`, download `model.safetensors` manually from:

```text
https://huggingface.co/wdoppenberg/crater-rcnn/tree/main
```

and place it in the directory above.

## Determinism Notes

- ACDLR is deterministic for fixed image order and fixed parameters.
- Ellipse R-CNN inference is deterministic on CPU for this benchmark setup.
- Generated artifacts are ignored by Git; preserve final selected outputs in
  `reports/` or document their paths in the paper.

# Experiment Scripts

This folder contains reproducible command-line scripts for running ACDLR,
benchmarks and comparison reports.

## Main Article Scripts

| Script | Purpose |
|---|---|
| `setup_lu3m6tgt_dataset.py` | Download and prepare the Kaggle LU3M6TGT YOLO dataset |
| `benchmark_yolo_dataset.py` | Evaluate ACDLR on a YOLO-format annotated visual dataset |
| `setup_ellipse_rcnn_pretrained.py` | Clone Ellipse R-CNN and download crater-rcnn weights |
| `benchmark_ellipse_rcnn_yolo_dataset.py` | Evaluate Ellipse R-CNN on the same visual YOLO dataset |
| `run_acdlr_vs_ellipse_rcnn_comparison.py` | Generate the fair ACDLR x Ellipse R-CNN report, chart and visual comparison |

## Recommended Commands

Prepare Ellipse R-CNN:

```powershell
python scripts\setup_ellipse_rcnn_pretrained.py
```

Prepare the LU3M6TGT Kaggle dataset:

```powershell
python scripts\setup_lu3m6tgt_dataset.py
```

Skip the large weight download during quick setup:

```powershell
python scripts\setup_ellipse_rcnn_pretrained.py --skip-download
```

Run the fair ACDLR x Ellipse R-CNN comparison:

```powershell
python scripts\run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Run only ACDLR:

```powershell
python scripts\benchmark_yolo_dataset.py --split valid --max-images 25
```

Run only Ellipse R-CNN:

```powershell
python scripts\benchmark_ellipse_rcnn_yolo_dataset.py --split valid --max-images 25
```

## Outputs

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
```

## Method Boundary

The active article comparison uses only the scripts listed above. The ACDLR
script runs the classical method; the Ellipse script runs the external
pretrained neural baseline. Previous comparison scripts were removed from the
active repository structure to keep the experiment clean.

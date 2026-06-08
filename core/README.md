# Core Method Implementation

This folder contains the ACDLR method itself. The ACDLR detector is classical
computer vision only: no CNN, no learned model, no AI inference.

## Modules

| Module | Purpose |
|---|---|
| `preprocessing.py` | Grayscale conversion, CLAHE, smoothing and edge hints |
| `detection.py` | Multi-scale crater candidate generation and validation |
| `tiling.py` | Image tiling, coordinate merging and deduplication support |
| `measurement.py` | Radius, diameter, area and physical-size measurements |
| `risk.py` | Region-level landing-risk score and landing-point suggestion |
| `visualization.py` | Overlays for craters, risk grid and final result |
| `evaluation.py` | Precision, recall, F1 and normalized error metrics |

## Method Boundary

The external AI baseline lives outside `core/`. The current article-facing
baseline is Ellipse R-CNN, prepared by
`scripts/setup_ellipse_rcnn_pretrained.py`, evaluated by
`scripts/benchmark_ellipse_rcnn_yolo_dataset.py` and compared with ACDLR by
`scripts/run_acdlr_vs_ellipse_rcnn_comparison.py`.

This boundary matters for the paper claim:

```text
ACDLR = classical image processing
Ellipse R-CNN = external pretrained neural crater detector
```

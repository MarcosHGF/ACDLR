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

The CNN baseline lives in `scripts/benchmark_crater_cnn_yolo.py` and
`scripts/train_crater_cnn_yolo.py`. It is not part of the ACDLR method.

This boundary matters for the paper claim:

```text
ACDLR = classical image processing
CNN YOLOv11 = external comparison baseline
```

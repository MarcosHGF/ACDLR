# Experiment Protocol

## Research Question

Can a classical, explainable crater detector based on image processing compete
with a CNN baseline on a YOLO-annotated lunar crater dataset?

## Methods Compared

| Method | Uses AI? | Trainable? | Output |
|---|---|---|---|
| ACDLR | No | No | Circles `(x, y, r)` |
| CNN YOLOv11 | Yes | Yes | Boxes converted to circles |

## Primary Metric

F1 score.

Reason: crater detection needs a balance between not missing annotated craters
and not hallucinating too many false positives.

## Secondary Metrics

- Precision
- Recall
- True positives
- False positives
- False negatives
- Mean center error ratio
- Mean radius error ratio

## Matching Protocol

For a predicted crater and a ground-truth crater:

```text
center_error_ratio = center_distance / gt_radius
radius_error_ratio = abs(pred_radius - gt_radius) / gt_radius
```

Accepted match:

```text
center_error_ratio <= 1.34
radius_error_ratio <= 1.0
```

Matches are assigned greedily by lowest normalized error.

## Baseline Settings

ACDLR:

```text
min_radius = 4
max_radius = 70
canny_threshold = 45
strictness = 16
```

CNN:

```text
model = YOLOv11
conf = 0.001
iou = 0.15
max_det = 150
```

## Smoke-Test Experiment

Purpose: verify that the full benchmark pipeline works.

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py --max-images 3
```

This is not the final experiment.

## Recommended Paper Experiment

1. Train the CNN on `train`.
2. Evaluate ACDLR and CNN on the full `valid` split.
3. Report aggregate metrics and per-image CSV.
4. Include at least 6 visual examples:
   - ACDLR success case;
   - ACDLR failure case;
   - CNN success case;
   - CNN failure case;
   - dense-crater tile;
   - sparse-crater tile.

Recommended command:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 1545 ^
  --visual-count 12 ^
  --skip-cnn-train
```

If training a fresh CNN:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 1545 ^
  --visual-count 12 ^
  --force-cnn-train ^
  --cnn-train-epochs 50 ^
  --cnn-train-fraction 1.0
```

## Reporting Template

| Method | Detections | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | | | | | | | | |
| CNN YOLOv11 | | | | | | | | |

## Validity Threats

- YOLO boxes are approximated as circles.
- Craters can be partially visible or visually ambiguous.
- CNN smoke-test training is intentionally weak.
- ACDLR parameters were tuned on small observed subsets.
- Full validation metrics are needed before strong claims.

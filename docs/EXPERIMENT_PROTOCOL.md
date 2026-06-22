# Experiment Protocol

## Research Question

Can a classical, explainable crater detector based on image processing compete
with a pretrained visual CNN crater detector on the same annotated lunar image
dataset?

## Methods Compared

| Method | Uses AI? | Trainable in this repo? | Output | Evaluation source |
|---|---|---|---|---|
| ACDLR | No | No | Circles `(x, y, r)` | local visual YOLO dataset |
| Ellipse R-CNN | Yes | No | Ellipses converted to circles | same local visual YOLO dataset |

## Primary Metric

F1 score.

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

Ellipse R-CNN predictions are converted with:

```text
radius = (ellipse_a + ellipse_b) / 2
```

## Baseline Settings

ACDLR:

```text
min_radius = 4
max_radius = 70
canny_threshold = 45
strictness = 16
```

Ellipse R-CNN:

```text
model = wdoppenberg/crater-rcnn
score_threshold = 0.60
max_detections = 150
device = cpu
```

## Compact Paper Experiment

```powershell
python scripts\run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

## Full Validation Experiment

```powershell
python scripts\run_acdlr_vs_ellipse_rcnn_comparison.py ^
  --max-images 1545 ^
  --visual-count 12
```

## Reporting Template

| Method | Detections | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | | | | | | | | |
| Ellipse R-CNN | | | | | | | | |

## Validity Threats

- YOLO boxes are approximated as circles.
- Ellipse R-CNN was not trained on this exact dataset, so zero-shot domain shift
  is possible.
- ACDLR parameters were tuned on small observed subsets.
- Full validation metrics are needed before strong final claims.

## Baseline Scope

The active neural baseline is Ellipse R-CNN with pretrained crater weights,
executed without repository modifications or local fine-tuning. Any future
trained model should be reported as a separate experiment, not mixed with the
vanilla pretrained comparison.

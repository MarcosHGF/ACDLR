# Dataset Card

## Dataset Name

`LU3M6TGT_yolo_format`

## Local Path

```text
data/LU3M6TGT_yolo_format
```

## Task

Lunar crater detection in visual image tiles.

## Format

YOLO object-detection format:

```text
class x_center y_center width height
```

All values are normalized between 0 and 1.

## Splits

| Split | Images |
|---|---:|
| train | 8756 |
| valid | 1545 |

## Label Conversion For ACDLR Evaluation

ACDLR predicts circles. The YOLO labels are boxes. For fair comparison, each
YOLO annotation is converted to a circle:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

This lets both ACDLR and the CNN baseline be evaluated using the same matching
logic.

## Known Caveats

- The labels are box annotations, while craters are naturally circular/elliptic.
- Some crater rims may be degraded or partially shadowed.
- Illumination direction can change visual contrast.
- Small craters are harder for both classical and CNN detectors.
- The local dataset is ignored by Git because it is large.

## Recommended Use

- Train CNN baselines on `train`.
- Evaluate all methods on `valid`.
- Do not tune hyperparameters and report final metrics on the exact same small
  subset without disclosing it.
- For a paper-grade result, report the full validation split or a clearly
  defined subset with a fixed seed/order.

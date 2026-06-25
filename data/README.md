# Data Folder

Datasets are not versioned in this repository.

Official Kaggle dataset:

```text
https://www.kaggle.com/datasets/riccardolagrassa/lu3m6tgt
```

Recommended setup:

```bash
python scripts/setup_lu3m6tgt_dataset.py
```

This downloads the dataset with `kagglehub` and places the YOLO crater dataset here:

```text
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
```

Each label file must follow YOLO format:

```text
class x_center y_center width height
```

The benchmark scripts expect this path by default:

```text
data/LU3M6TGT_yolo_format
```

# Data Folder

Datasets are not versioned in this repository.

Place the YOLO crater dataset here:

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

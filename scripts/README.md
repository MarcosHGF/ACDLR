# Experiment Scripts

This folder contains reproducible command-line scripts for running the method,
benchmarks and comparisons.

## Main Scripts

| Script | Purpose |
|---|---|
| `benchmark_yolo_dataset.py` | Evaluate ACDLR on a YOLO-format annotated dataset |
| `train_crater_cnn_yolo.py` | Train the YOLOv11 CNN comparison baseline |
| `benchmark_crater_cnn_yolo.py` | Evaluate the CNN baseline on the same dataset |
| `run_acdlr_vs_crater_cnn_comparison.py` | Run the full ACDLR x CNN benchmark and visual report |

## Recommended Commands

Run ACDLR x CNN:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

Run only ACDLR:

```powershell
python scripts\benchmark_yolo_dataset.py --split valid --max-images 25
```

Train CNN baseline:

```powershell
python scripts\train_crater_cnn_yolo.py --epochs 1 --fraction 0.02
```

## Legacy/Reference Scripts

Some scripts related to DeepMoon and earlier LROC experiments remain in this
folder for traceability. The current primary article-style comparison is:

```text
ACDLR x CNN YOLOv11
```

not DeepMoon, because the local annotated dataset is visual imagery in YOLO
format.

from __future__ import annotations

"""
Train a small YOLOv11 crater detector for the CNN comparison baseline.

The ACDLR method remains classical image processing. This script is only for
the competing CNN baseline inspired by the open crater-identification repo.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ULTRALYTICS_CONFIG_DIR = REPO_ROOT / "artifacts" / "ultralytics_config"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--base-weights", default="external/crater-identification/YOLOv11model/yolo11n.pt")
    parser.add_argument("--out-dir", default="artifacts/crater_cnn_yolo_train")
    parser.add_argument("--name", default="moon_small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    base_weights = (REPO_ROOT / args.base_weights).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_dir = out_dir / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    if args.skip_if_exists and (best_path.exists() or last_path.exists()):
        weights = best_path if best_path.exists() else last_path
        _write_summary(run_dir, args, dataset_dir, base_weights, weights)
        print(f"Existing CNN weights found: {weights}")
        return

    if not base_weights.exists():
        raise SystemExit(f"Base YOLO weights not found: {base_weights}")
    _validate_dataset(dataset_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = out_dir / "acdlr_crater_yolo.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "nc: 1",
                "names: ['crater']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(out_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        fraction=args.fraction,
        val=args.val,
        cache=args.cache,
        plots=True,
        verbose=False,
    )

    weights = best_path if best_path.exists() else last_path
    if not weights.exists():
        raise SystemExit(f"Training finished but no weights were found in {run_dir / 'weights'}")

    _write_summary(run_dir, args, dataset_dir, base_weights, weights)
    print("")
    print("CNN YOLO training complete")
    print(f"Weights: {weights}")


def _validate_dataset(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "valid" / "images",
        dataset_dir / "valid" / "labels",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing YOLO dataset folders:\n" + "\n".join(str(path) for path in missing))


def _write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_dir: Path,
    base_weights: Path,
    weights: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "CNN YOLOv11 crater detector",
        "source_repo": "https://github.com/sydney-machine-learning/crater-identification",
        "source_article": "https://www.nature.com/articles/s44453-026-00036-x",
        "dataset_dir": str(dataset_dir),
        "base_weights": str(base_weights),
        "trained_weights": str(weights),
        "epochs": args.epochs,
        "fraction": args.fraction,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "val_during_training": args.val,
        "cache": args.cache,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

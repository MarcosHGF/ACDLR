from __future__ import annotations

"""
Benchmark a YOLO/Ultralytics crater detector on the same YOLO-format dataset
used by ACDLR.

This is the CNN comparison side only. It converts predicted bounding boxes to
circle detections so the metrics are directly comparable with ACDLR:
    cx = box_center_x
    cy = box_center_y
    radius = mean(box_width, box_height) / 2
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

ULTRALYTICS_CONFIG_DIR = REPO_ROOT / "artifacts" / "ultralytics_config"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

from benchmark_yolo_dataset import _draw_matches, _find_pairs, load_yolo_circles
from core import evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", default="artifacts/crater_cnn_yolo_benchmark")
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument("--visual-count", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--max-det", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-radius", type=float, default=2.0)
    parser.add_argument("--max-radius", type=float, default=0.0)
    parser.add_argument("--center-tolerance", type=float, default=1.34)
    parser.add_argument("--radius-tolerance", type=float, default=1.0)
    parser.add_argument("--allow-coco-weights", action="store_true")
    args = parser.parse_args()

    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    weights_path = (REPO_ROOT / args.weights).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise SystemExit(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    model_names = _normalise_names(model.names)
    if _looks_like_coco(model_names) and not args.allow_coco_weights:
        raise SystemExit(
            "The selected weights look like generic COCO YOLO weights, not crater "
            "weights. Train or provide a crater best.pt first, for example:\n"
            "python scripts\\train_crater_cnn_yolo.py --epochs 1 --fraction 0.02"
        )

    pairs = _find_pairs(dataset_dir, args.split)
    if not pairs:
        raise SystemExit(f"No YOLO image/label pairs found in {dataset_dir / args.split}")

    selected = pairs[: max(args.max_images, 1)]
    rows: list[dict[str, int | float | str]] = []
    results: list[evaluation.EvaluationResult] = []
    visual_jobs: list[
        tuple[Path, np.ndarray, np.ndarray, list[evaluation.GroundTruthCrater], evaluation.EvaluationResult, float]
    ] = []

    for index, pair in enumerate(selected, start=1):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        truth = load_yolo_circles(pair.label_path, image.shape)
        circles, confidences = _predict_circles(model, pair.image_path, args)
        result = evaluation.evaluate_circles(
            circles,
            truth,
            center_tolerance_ratio=args.center_tolerance,
            radius_tolerance_ratio=args.radius_tolerance,
        )
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        results.append(result)

        row = {
            "image": pair.image_path.name,
            "label": pair.label_path.name,
            "mean_confidence": mean_conf,
            **result.to_dict(),
        }
        rows.append(row)
        visual_jobs.append((pair.image_path, image, circles, truth, result, mean_conf))

        print(
            f"[{index:03d}/{len(selected):03d}] {pair.image_path.name}: "
            f"P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f} "
            f"TP={result.true_positive} FP={result.false_positive} FN={result.false_negative} "
            f"mean_conf={mean_conf:.3f}"
        )

    summary = evaluation.aggregate(results)
    rows.append({"image": "__TOTAL__", "label": "", "mean_confidence": "", **summary.to_dict()})
    _write_csv(out_dir / "cnn_yolo_benchmark.csv", rows)

    visual_paths = []
    for image_path, image, circles, truth, result, _ in visual_jobs[: max(args.visual_count, 0)]:
        visual = _draw_matches(image, circles, truth, result)
        visual_path = visuals_dir / f"{image_path.stem}_matches.png"
        cv2.imwrite(str(visual_path), visual)
        visual_paths.append(visual_path)

    summary_dict = {
        "method": "CNN YOLOv11 crater detector",
        "source_repo": "https://github.com/sydney-machine-learning/crater-identification",
        "source_article": "https://www.nature.com/articles/s44453-026-00036-x",
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "images_processed": len(results),
        "weights": str(weights_path),
        "model_names": model_names,
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "min_radius": args.min_radius,
            "max_radius": args.max_radius,
            "center_tolerance": args.center_tolerance,
            "radius_tolerance": args.radius_tolerance,
        },
        **summary.to_dict(),
        "visuals": [str(path) for path in visual_paths],
    }
    _write_json(out_dir / "cnn_yolo_summary.json", summary_dict)
    _write_report(out_dir / "cnn_yolo_report.md", summary_dict, visual_paths)

    print("")
    print("CNN YOLO benchmark complete")
    print(f"Images processed: {len(results)}")
    print(f"Precision: {summary.precision:.3f}")
    print(f"Recall: {summary.recall:.3f}")
    print(f"F1: {summary.f1:.3f}")
    print(f"Report: {out_dir / 'cnn_yolo_report.md'}")


def _predict_circles(model: YOLO, image_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, list[float]]:
    predictions = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )
    if not predictions:
        return np.empty((0, 3), dtype=np.float32), []

    boxes = predictions[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 3), dtype=np.float32), []

    xywh = boxes.xywh.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    circles: list[list[float]] = []
    kept_confidences: list[float] = []

    for (cx, cy, width, height), confidence in zip(xywh, confidences):
        radius = (float(width) + float(height)) / 4.0
        if radius < args.min_radius:
            continue
        if args.max_radius > 0 and radius > args.max_radius:
            continue
        circles.append([float(cx), float(cy), radius])
        kept_confidences.append(float(confidence))

    if not circles:
        return np.empty((0, 3), dtype=np.float32), []
    return np.asarray(circles, dtype=np.float32), kept_confidences


def _normalise_names(names: object) -> dict[str, str]:
    if isinstance(names, dict):
        return {str(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(index): str(value) for index, value in enumerate(names)}
    return {"0": str(names)}


def _looks_like_coco(names: dict[str, str]) -> bool:
    values = {value.lower() for value in names.values()}
    coco_markers = {"person", "bicycle", "car", "dog", "toothbrush"}
    return len(values) >= 70 and len(values & coco_markers) >= 3


def _write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(path: Path, summary: dict, visual_paths: list[Path]) -> None:
    lines = [
        "# CNN YOLOv11 Crater Benchmark",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Split: `{summary['split']}`",
        f"- Images processed: {summary['images_processed']}",
        f"- Weights: `{summary['weights']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Detections | {summary['detections']} |",
        f"| Ground truth | {summary['ground_truth']} |",
        f"| True positives | {summary['true_positive']} |",
        f"| False positives | {summary['false_positive']} |",
        f"| False negatives | {summary['false_negative']} |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        f"| F1 | {summary['f1']:.4f} |",
        f"| Mean center error ratio | {summary['mean_center_error_ratio']:.4f} |",
        f"| Mean radius error ratio | {summary['mean_radius_error_ratio']:.4f} |",
        "",
        "## Parameters",
        "",
    ]
    for key, value in summary["parameters"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Visuals", ""])
    lines.extend(f"- `{path}`" for path in visual_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


if __name__ == "__main__":
    main()

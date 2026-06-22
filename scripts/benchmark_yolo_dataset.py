from __future__ import annotations

"""
Benchmark ACDLR on a YOLO-format crater dataset.

YOLO annotations are converted to circular crater annotations:
    cx = x_center * image_width
    cy = y_center * image_height
    radius = mean(box_width_px, box_height_px) / 2

This lets the classical ACDLR detector be evaluated against datasets that were
prepared for CNN object detectors, without adding AI to the ACDLR method.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_classical import _detect_image
from core import evaluation


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--out-dir", default="artifacts/yolo_benchmark")
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument("--visual-count", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--clahe-clip", type=float, default=2.5)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--min-radius", type=int, default=4)
    parser.add_argument("--max-radius", type=int, default=70)
    parser.add_argument("--canny-threshold", type=int, default=45)
    parser.add_argument("--strictness", type=int, default=16)
    parser.add_argument("--center-tolerance", type=float, default=0.65)
    parser.add_argument("--radius-tolerance", type=float, default=0.65)
    args = parser.parse_args()

    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _clear_visuals(visuals_dir)

    pairs = _find_pairs(dataset_dir, args.split)
    if not pairs:
        raise SystemExit(f"No YOLO image/label pairs found in {dataset_dir / args.split}")

    selected = pairs[: max(args.max_images, 1)]
    rows: list[dict[str, int | float | str]] = []
    results: list[evaluation.EvaluationResult] = []
    visual_jobs: list[tuple[int, Path, np.ndarray, np.ndarray, list[evaluation.GroundTruthCrater], evaluation.EvaluationResult]] = []

    for index, pair in enumerate(selected, start=1):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        truth = load_yolo_circles(pair.label_path, image.shape)
        circles = _detect_image(image, args)
        result = evaluation.evaluate_circles(
            circles,
            truth,
            center_tolerance_ratio=args.center_tolerance,
            radius_tolerance_ratio=args.radius_tolerance,
        )
        results.append(result)

        row = {
            "image": pair.image_path.name,
            "label": pair.label_path.name,
            **result.to_dict(),
        }
        rows.append(row)
        visual_jobs.append((index, pair.image_path, image, circles, truth, result))

        print(
            f"[{index:03d}/{len(selected):03d}] {pair.image_path.name}: "
            f"P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f} "
            f"TP={result.true_positive} FP={result.false_positive} FN={result.false_negative}"
        )

    summary = evaluation.aggregate(results)
    rows.append({"image": "__TOTAL__", "label": "", **summary.to_dict()})
    _write_csv(out_dir / "acdlr_yolo_benchmark.csv", rows)

    visual_paths = []
    for _, image_path, image, circles, truth, result in visual_jobs[: max(args.visual_count, 0)]:
        visual = _draw_matches(image, circles, truth, result)
        visual_path = visuals_dir / f"{image_path.stem}_matches.png"
        cv2.imwrite(str(visual_path), visual)
        visual_paths.append(visual_path)

    summary_dict = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "images_processed": len(results),
        "parameters": {
            "clahe_clip": args.clahe_clip,
            "blur_ksize": args.blur_ksize,
            "min_radius": args.min_radius,
            "max_radius": args.max_radius,
            "canny_threshold": args.canny_threshold,
            "strictness": args.strictness,
            "center_tolerance": args.center_tolerance,
            "radius_tolerance": args.radius_tolerance,
        },
        **summary.to_dict(),
        "visuals": [str(path) for path in visual_paths],
    }
    _write_json(out_dir / "acdlr_yolo_summary.json", summary_dict)
    _write_report(out_dir / "acdlr_yolo_report.md", summary_dict, visual_paths)

    print("")
    print("YOLO benchmark complete")
    print(f"Images processed: {len(results)}")
    print(f"Precision: {summary.precision:.3f}")
    print(f"Recall: {summary.recall:.3f}")
    print(f"F1: {summary.f1:.3f}")
    print(f"Report: {out_dir / 'acdlr_yolo_report.md'}")


def load_yolo_circles(
    label_path: Path,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[evaluation.GroundTruthCrater]:
    h, w = image_shape[:2]
    craters: list[evaluation.GroundTruthCrater] = []
    if not label_path.exists():
        return craters

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx_px = float(cx) * w
            cy_px = float(cy) * h
            radius_px = ((float(bw) * w) + (float(bh) * h)) / 4.0
            if radius_px <= 0:
                continue
            craters.append(evaluation.GroundTruthCrater(cx_px, cy_px, radius_px))
    return craters


def _find_pairs(dataset_dir: Path, split: str) -> list[Pair]:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    pairs: list[Pair] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append(Pair(image_path, label_path))
    return pairs


def _draw_matches(
    image: np.ndarray,
    circles: np.ndarray,
    truth: list[evaluation.GroundTruthCrater],
    result: evaluation.EvaluationResult,
) -> np.ndarray:
    vis = image.copy()
    matched_detections = {match.detection_index for match in result.matches}
    matched_truth = {match.truth_index for match in result.matches}

    det_rows = np.asarray(circles, dtype=float) if circles.size else np.empty((0, 3), dtype=float)
    for idx, (x, y, radius) in enumerate(det_rows[:, :3]):
        color = (80, 255, 100) if idx in matched_detections else (70, 70, 255)
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(radius)), color, 1)

    for idx, crater in enumerate(truth):
        color = (80, 255, 100) if idx in matched_truth else (80, 220, 255)
        cv2.circle(
            vis,
            (int(round(crater.cx)), int(round(crater.cy))),
            int(round(crater.radius_px)),
            color,
            1,
        )

    header_h = 54
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
    _put_text(
        header,
        f"P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f}",
        (8, 20),
    )
    _put_text(header, "green=TP red=FP yellow=FN", (8, 40), scale=0.42)
    return np.vstack([header, vis])


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


def _clear_visuals(visuals_dir: Path) -> None:
    for path in visuals_dir.glob("*_matches.png"):
        if path.is_file():
            path.unlink()


def _write_report(path: Path, summary: dict, visual_paths: list[Path]) -> None:
    lines = [
        "# ACDLR YOLO Dataset Benchmark",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Split: `{summary['split']}`",
        f"- Images processed: {summary['images_processed']}",
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

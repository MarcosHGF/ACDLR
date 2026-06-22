from __future__ import annotations

"""
Benchmark Ellipse R-CNN on the same visual YOLO dataset used by ACDLR.

The model predicts ellipses [a, b, cx, cy, theta]. For a fair comparison with
ACDLR and YOLO labels, each ellipse is converted to a circle:
    radius = (a + b) / 2
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core import evaluation
from scripts.benchmark_yolo_dataset import SUPPORTED_IMAGE_EXTS, Pair, load_yolo_circles


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pretrained Ellipse R-CNN on visual YOLO crater dataset.")
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--out-dir", default="artifacts/ellipse_rcnn_yolo_benchmark")
    parser.add_argument("--model", default="artifacts/ellipse_rcnn_pretrained/crater-rcnn")
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument("--visual-count", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--max-detections", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--center-tolerance", type=float, default=1.34)
    parser.add_argument("--radius-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    dataset_dir = _resolve(args.dataset_dir)
    out_dir = _resolve(args.out_dir)
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _clear_visuals(visuals_dir)

    pairs = _find_pairs(dataset_dir, args.split)
    if not pairs:
        raise SystemExit(f"No YOLO image/label pairs found in {dataset_dir / args.split}")

    model_ref = _model_ref(args.model)
    _check_model_ready(model_ref)
    model = _load_model(model_ref, args.device)

    selected = pairs[: max(args.max_images, 1)]
    rows: list[dict[str, int | float | str]] = []
    results: list[evaluation.EvaluationResult] = []
    visual_jobs: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, list[evaluation.GroundTruthCrater], evaluation.EvaluationResult]] = []

    for index, pair in enumerate(selected, start=1):
        image_bgr = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        truth = load_yolo_circles(pair.label_path, image_bgr.shape)
        circles, ellipses = _predict_circles(
            model,
            pair.image_path,
            args.device,
            args.score_threshold,
            args.max_detections,
        )
        result = evaluation.evaluate_circles(
            circles,
            truth,
            center_tolerance_ratio=args.center_tolerance,
            radius_tolerance_ratio=args.radius_tolerance,
        )
        results.append(result)
        rows.append(
            {
                "image": pair.image_path.name,
                "label": pair.label_path.name,
                **result.to_dict(),
            }
        )
        visual_jobs.append((pair.image_path, image_bgr, circles, ellipses, truth, result))
        print(
            f"[{index:03d}/{len(selected):03d}] {pair.image_path.name}: "
            f"P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f} "
            f"TP={result.true_positive} FP={result.false_positive} FN={result.false_negative}"
        )

    summary = evaluation.aggregate(results)
    rows.append({"image": "__TOTAL__", "label": "", **summary.to_dict()})
    _write_csv(out_dir / "ellipse_rcnn_yolo_benchmark.csv", rows)

    visual_paths = []
    for image_path, image_bgr, circles, ellipses, truth, result in visual_jobs[: max(args.visual_count, 0)]:
        visual = _draw_matches(image_bgr, circles, ellipses, truth, result)
        visual_path = visuals_dir / f"{image_path.stem}_matches.png"
        cv2.imwrite(str(visual_path), visual)
        visual_paths.append(visual_path)

    summary_dict = {
        "method": "Ellipse R-CNN pretrained crater-rcnn",
        "model": args.model,
        "model_source": "https://huggingface.co/wdoppenberg/crater-rcnn",
        "repo_source": "https://github.com/wdoppenberg/ellipse-rcnn",
        "dataset_dir": _display_path(dataset_dir),
        "split": args.split,
        "images_processed": len(results),
        "parameters": {
            "score_threshold": args.score_threshold,
            "max_detections": args.max_detections,
            "device": args.device,
            "center_tolerance": args.center_tolerance,
            "radius_tolerance": args.radius_tolerance,
        },
        **summary.to_dict(),
        "visuals": [_display_path(path) for path in visual_paths],
    }
    _write_json(out_dir / "ellipse_rcnn_yolo_summary.json", summary_dict)
    _write_report(out_dir / "ellipse_rcnn_yolo_report.md", summary_dict, visual_paths)

    print("")
    print("Ellipse R-CNN YOLO benchmark complete")
    print(f"Images processed: {len(results)}")
    print(f"Precision: {summary.precision:.3f}")
    print(f"Recall: {summary.recall:.3f}")
    print(f"F1: {summary.f1:.3f}")
    print(f"Report: {out_dir / 'ellipse_rcnn_yolo_report.md'}")


def _load_model(model_ref: str, device: str):
    path = Path(model_ref)
    if path.exists() and path.is_dir() and (path / "config.json").exists() and (path / "model.safetensors").exists():
        try:
            from ellipse_rcnn import EllipseRCNN
            from safetensors.torch import load_file
        except Exception as exc:
            raise SystemExit(
                "Ellipse R-CNN or safetensors is not installed. Run: "
                "python scripts/setup_ellipse_rcnn_pretrained.py"
            ) from exc

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config["weights"] = None
        model = EllipseRCNN(**config)
        state_dict = load_file(str(path / "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        try:
            from ellipse_rcnn.hf import EllipseRCNN
        except Exception as exc:
            raise SystemExit(
                "Ellipse R-CNN is not installed. Run: "
                "python scripts/setup_ellipse_rcnn_pretrained.py"
            ) from exc

        model = EllipseRCNN.from_pretrained(model_ref, weights=None)
    model.eval()
    model.to(torch.device(device))
    return model


def _predict_circles(model, image_path: Path, device: str, score_threshold: float, max_detections: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("L")
    tensor = to_tensor(image).to(torch.device(device))
    with torch.no_grad():
        pred = model([tensor])[0]

    scores = pred["scores"].detach().cpu()
    ellipses = pred["ellipse_params"].detach().cpu()
    keep = scores >= score_threshold
    scores = scores[keep]
    ellipses = ellipses[keep]
    if scores.numel() > 0:
        order = torch.argsort(scores, descending=True)[:max_detections]
        ellipses = ellipses[order]

    if ellipses.numel() == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty((0, 5), dtype=np.float32)

    arr = ellipses.numpy().astype(np.float32)
    a = arr[:, 0]
    b = arr[:, 1]
    cx = arr[:, 2]
    cy = arr[:, 3]
    radius = (a + b) / 2.0
    circles = np.column_stack([cx, cy, radius]).astype(np.float32)
    return circles, arr


def _check_model_ready(model_ref: str) -> None:
    path = Path(model_ref)
    if path.exists() and path.is_dir() and not (path / "model.safetensors").exists():
        raise SystemExit(
            "Ellipse R-CNN weights are incomplete. Missing "
            f"`{_display_path(path / 'model.safetensors')}`.\n"
            "Run `python scripts/setup_ellipse_rcnn_pretrained.py`, or download "
            "the file manually from https://huggingface.co/wdoppenberg/crater-rcnn/tree/main."
        )


def _model_ref(model_text: str) -> str:
    path = Path(model_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path) if path.exists() else model_text


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
    image_bgr: np.ndarray,
    circles: np.ndarray,
    ellipses: np.ndarray,
    truth: list[evaluation.GroundTruthCrater],
    result: evaluation.EvaluationResult,
) -> np.ndarray:
    vis = image_bgr.copy()
    matched_detections = {match.detection_index for match in result.matches}
    matched_truth = {match.truth_index for match in result.matches}

    for idx, ellipse in enumerate(np.asarray(ellipses, dtype=float)):
        a, b, cx, cy, theta = ellipse[:5]
        color = (80, 255, 100) if idx in matched_detections else (70, 70, 255)
        center = (int(round(cx)), int(round(cy)))
        axes = (max(1, int(round(a))), max(1, int(round(b))))
        cv2.ellipse(vis, center, axes, float(np.degrees(theta)), 0, 360, color, 1)
        cv2.circle(vis, center, 2, color, -1)

    for idx, crater in enumerate(truth):
        color = (80, 255, 100) if idx in matched_truth else (80, 220, 255)
        cv2.circle(
            vis,
            (int(round(crater.cx)), int(round(crater.cy))),
            int(round(crater.radius_px)),
            color,
            1,
        )

    header_h = 58
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
    _put_text(header, f"Ellipse R-CNN  P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f}", (8, 22))
    _put_text(header, "green=TP red=FP yellow=FN", (8, 44), scale=0.42)
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
        "# Ellipse R-CNN YOLO Dataset Benchmark",
        "",
        f"- Model: `{summary['model']}`",
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
        "## Visuals",
        "",
    ]
    for visual_path in visual_paths:
        lines.append(f"- `{_display_path(visual_path)}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.5) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Run a classical ACDLR benchmark against manually annotated crater tiles.
The reported metrics include precision, recall, F1, and fractional center/radius
errors adapted to image pixels.

Example
-------
python benchmark_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations

Annotation formats
------------------
For an image named torricelli_y00000_x00000.png, create either:
- data/annotations/torricelli_y00000_x00000.csv
- data/annotations/torricelli_y00000_x00000.json

CSV columns may be cx,cy,radius_px or x,y,r.
JSON may be a list of objects or an object with a "craters" list.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from core import detection, evaluation, preprocessing, tiling


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directory with benchmark image tiles")
    parser.add_argument("--annotations-dir", required=True, help="Directory with CSV/JSON crater annotations")
    parser.add_argument("--output", default="benchmark_results.csv", help="CSV report path")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--clahe-clip", type=float, default=2.5)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=40)
    parser.add_argument("--canny-threshold", type=int, default=60)
    parser.add_argument("--strictness", type=int, default=34)
    parser.add_argument("--center-tolerance", type=float, default=0.50)
    parser.add_argument("--radius-tolerance", type=float, default=0.50)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    image_paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )

    rows: list[dict[str, float | int | str]] = []
    results: list[evaluation.EvaluationResult] = []

    for image_path in image_paths:
        annotation_path = _find_annotation(annotations_dir, image_path.stem)
        if annotation_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")

        circles = _detect_image(image, args)
        truth = evaluation.load_annotations(annotation_path)
        result = evaluation.evaluate_circles(
            circles,
            truth,
            center_tolerance_ratio=args.center_tolerance,
            radius_tolerance_ratio=args.radius_tolerance,
        )
        results.append(result)

        row = {
            "image": image_path.name,
            "annotation": annotation_path.name,
            **result.to_dict(),
        }
        rows.append(row)

    summary = evaluation.aggregate(results)
    rows.append({"image": "__TOTAL__", "annotation": "", **summary.to_dict()})
    _write_csv(Path(args.output), rows)

    print(f"Images with annotations: {len(results)}")
    print(f"Detections: {summary.detections}")
    print(f"Ground truth: {summary.ground_truth}")
    print(f"Precision: {summary.precision:.3f}")
    print(f"Recall: {summary.recall:.3f}")
    print(f"F1: {summary.f1:.3f}")
    print(f"Mean center error ratio: {summary.mean_center_error_ratio:.3f}")
    print(f"Mean radius error ratio: {summary.mean_radius_error_ratio:.3f}")
    print(f"Report: {Path(args.output).resolve()}")


def _detect_image(image_bgr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    all_circles: list[np.ndarray] = []
    tiles = tiling.split(image_bgr, tile_size=args.tile_size, overlap=args.overlap)

    for tile in tiles:
        prep = preprocessing.run(
            tile.image,
            clahe_clip=args.clahe_clip,
            blur_ksize=args.blur_ksize,
        )
        local = detection.detect(
            prep,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            param1=args.canny_threshold,
            param2=args.strictness,
        )
        if local.size > 0:
            all_circles.append(tiling.to_global(local, tile))

    if not all_circles:
        return np.empty((0, 3), dtype=int)

    return tiling.deduplicate(np.vstack(all_circles))


def _find_annotation(directory: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".json"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

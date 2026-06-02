from __future__ import annotations

"""
Tune the classical ACDLR detector using annotated tiles.

This is the practical "step 3" after creating a benchmark: run a compact
grid search and choose parameters by F1, mirroring DeepMoon's validation
strategy without introducing deep learning into the project.

Example
-------
python tune_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations
"""

import argparse
import csv
import itertools
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core import evaluation
from benchmark_classical import SUPPORTED_IMAGE_EXTS, _detect_image, _find_annotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directory with benchmark image tiles")
    parser.add_argument("--annotations-dir", required=True, help="Directory with CSV/JSON crater annotations")
    parser.add_argument("--output", default="tuning_results.csv", help="CSV report path")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--center-tolerance", type=float, default=0.50)
    parser.add_argument("--radius-tolerance", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=10, help="Number of best rows to print")

    parser.add_argument("--clahe-clip-values", default="2.0,2.5,3.0")
    parser.add_argument("--blur-ksize-values", default="3,5,7")
    parser.add_argument("--min-radius-values", default="8,10,12")
    parser.add_argument("--max-radius-values", default="35,40,50")
    parser.add_argument("--canny-threshold-values", default="50,60,70")
    parser.add_argument("--strictness-values", default="28,34,40")
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.images_dir), Path(args.annotations_dir))
    if not dataset:
        raise SystemExit("No image/annotation pairs found. Add CSV/JSON files in the annotations directory.")

    rows: list[dict[str, float | int]] = []
    combinations = list(_parameter_grid(args))

    for index, params in enumerate(combinations, start=1):
        results: list[evaluation.EvaluationResult] = []

        for image, truth in dataset:
            circles = _detect_image(image, params)
            results.append(
                evaluation.evaluate_circles(
                    circles,
                    truth,
                    center_tolerance_ratio=args.center_tolerance,
                    radius_tolerance_ratio=args.radius_tolerance,
                )
            )

        summary = evaluation.aggregate(results)
        rows.append(
            {
                "rank": 0,
                "run": index,
                "clahe_clip": params.clahe_clip,
                "blur_ksize": params.blur_ksize,
                "min_radius": params.min_radius,
                "max_radius": params.max_radius,
                "canny_threshold": params.canny_threshold,
                "strictness": params.strictness,
                **summary.to_dict(),
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            -float(row["mean_center_error_ratio"]),
            -float(row["mean_radius_error_ratio"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    _write_csv(Path(args.output), rows)
    _print_top(rows, min(args.top_k, len(rows)), Path(args.output))


def _load_dataset(
    images_dir: Path,
    annotations_dir: Path,
) -> list[tuple[np.ndarray, list[evaluation.GroundTruthCrater]]]:
    dataset: list[tuple[np.ndarray, list[evaluation.GroundTruthCrater]]] = []
    image_paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )

    for image_path in image_paths:
        annotation_path = _find_annotation(annotations_dir, image_path.stem)
        if annotation_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")

        truth = evaluation.load_annotations(annotation_path)
        dataset.append((image, truth))

    return dataset


def _parameter_grid(args: argparse.Namespace):
    for (
        clahe_clip,
        blur_ksize,
        min_radius,
        max_radius,
        canny_threshold,
        strictness,
    ) in itertools.product(
        _parse_floats(args.clahe_clip_values),
        _parse_ints(args.blur_ksize_values),
        _parse_ints(args.min_radius_values),
        _parse_ints(args.max_radius_values),
        _parse_ints(args.canny_threshold_values),
        _parse_ints(args.strictness_values),
    ):
        if max_radius <= min_radius:
            continue

        yield SimpleNamespace(
            tile_size=args.tile_size,
            overlap=args.overlap,
            clahe_clip=clahe_clip,
            blur_ksize=blur_ksize,
            min_radius=min_radius,
            max_radius=max_radius,
            canny_threshold=canny_threshold,
            strictness=strictness,
        )


def _parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one float value is required")
    return values


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_top(rows: list[dict[str, float | int]], top_k: int, output: Path) -> None:
    print(f"Parameter runs: {len(rows)}")
    print(f"Report: {output.resolve()}")
    print("")
    print("Top parameter sets:")

    for row in rows[:top_k]:
        print(
            "rank={rank} f1={f1:.3f} precision={precision:.3f} recall={recall:.3f} "
            "center_err={mean_center_error_ratio:.3f} radius_err={mean_radius_error_ratio:.3f} "
            "clahe={clahe_clip} blur={blur_ksize} min_r={min_radius} max_r={max_radius} "
            "canny={canny_threshold} strictness={strictness}".format(**row)
        )


if __name__ == "__main__":
    main()

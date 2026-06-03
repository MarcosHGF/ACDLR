from __future__ import annotations

"""
Run a small, reproducible ACDLR x DeepMoon comparison.

ACDLR is evaluated on local project tiles. If matching manual annotations are
available, precision/recall/F1 are reported. Without annotations, the script
still writes detector outputs and marks the quantitative ACDLR metrics as
unavailable instead of inventing ground truth.

DeepMoon is executed on the official bundled template-matching sample from the
cloned silburt/DeepMoon repository. The full CNN stack is legacy
TensorFlow/Keras and is documented in the report as not runnable in this modern
Python environment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_classical import SUPPORTED_IMAGE_EXTS, _detect_image, _find_annotation
from core import evaluation, measurement, risk, visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/lroc_nac_roi_toriceliloa_tiles")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--deepmoon-dir", default="external/DeepMoon")
    parser.add_argument("--out-dir", default="artifacts/method_comparison")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--scale-m-per-px", type=float, default=5.0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--clahe-clip", type=float, default=2.5)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=40)
    parser.add_argument("--canny-threshold", type=int, default=60)
    parser.add_argument("--strictness", type=int, default=34)
    parser.add_argument("--center-tolerance", type=float, default=0.50)
    parser.add_argument("--radius-tolerance", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    acdlr = _run_acdlr_section(args, out_dir)
    deepmoon = _run_deepmoon_section(args, out_dir)
    side_by_side = _write_side_by_side(out_dir, acdlr.get("first_visual"), deepmoon.get("visual_path"))
    report_path = _write_report(out_dir, acdlr, deepmoon, side_by_side, args)

    print("ACDLR x DeepMoon comparison complete")
    print(f"report: {report_path}")
    print(f"side_by_side: {side_by_side}")
    print(f"acdlr_annotations_found: {acdlr['annotations_found']}")
    if acdlr.get("summary"):
        summary = acdlr["summary"]
        print(
            "ACDLR metrics: "
            f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} f1={summary['f1']:.3f}"
        )
    else:
        print("ACDLR metrics: unavailable without local annotations")
    if deepmoon.get("metrics"):
        metrics = deepmoon["metrics"]
        print(
            "DeepMoon sample metrics: "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        )
    else:
        print(f"DeepMoon sample metrics: unavailable ({deepmoon.get('status', 'unknown')})")


def _run_acdlr_section(args: argparse.Namespace, out_dir: Path) -> dict:
    images_dir = (REPO_ROOT / args.images_dir).resolve()
    annotations_dir = (REPO_ROOT / args.annotations_dir).resolve()
    visuals_dir = out_dir / "acdlr_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not image_paths:
        raise SystemExit(f"No local image tiles found in {images_dir}")

    annotated_pairs: list[tuple[Path, Path]] = []
    if annotations_dir.exists():
        for image_path in image_paths:
            annotation_path = _find_annotation(annotations_dir, image_path.stem)
            if annotation_path is not None:
                annotated_pairs.append((image_path, annotation_path))

    selected: list[tuple[Path, Path | None]]
    if annotated_pairs:
        selected = [(img, ann) for img, ann in annotated_pairs[: max(args.max_images, 1)]]
    else:
        selected = [(img, None) for img in image_paths[: max(args.max_images, 1)]]

    rows: list[dict[str, str | int | float]] = []
    eval_results: list[evaluation.EvaluationResult] = []
    visual_paths: list[Path] = []

    for image_path, annotation_path in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        circles = _detect_image(image, args)
        craters = measurement.measure(circles, scale_m_per_px=args.scale_m_per_px)
        stats = measurement.summary_stats(craters)
        score_matrix, stats_grid = risk.analyse(
            craters,
            image_shape=image.shape,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            scale_m_per_px=args.scale_m_per_px,
        )
        best_r, best_c = risk.best_landing_cell(score_matrix)
        landing_point = risk.suggest_landing_point(
            craters,
            image_shape=image.shape,
            best_row=best_r,
            best_col=best_c,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            scale_m_per_px=args.scale_m_per_px,
        )

        result = None
        truth: list[evaluation.GroundTruthCrater] = []
        if annotation_path is not None:
            truth = evaluation.load_annotations(annotation_path)
            result = evaluation.evaluate_circles(
                circles,
                truth,
                center_tolerance_ratio=args.center_tolerance,
                radius_tolerance_ratio=args.radius_tolerance,
            )
            eval_results.append(result)

        row = {
            "image": image_path.name,
            "annotation": annotation_path.name if annotation_path is not None else "",
            "detections": int(len(circles)),
            "ground_truth": int(len(truth)),
            "crater_count": int(stats["count"]),
            "mean_diameter_m": float(stats["mean_diameter_m"]),
            "max_diameter_m": float(stats["max_diameter_m"]),
            "mean_risk": float(np.mean(score_matrix)),
            "best_cell": f"R{best_r + 1}C{best_c + 1}",
            "landing_clearance_m": float(landing_point.clearance_m),
        }
        if result is not None:
            row.update(result.to_dict())
        rows.append(row)

        _write_detection_csv(visuals_dir / f"{image_path.stem}_detections.csv", circles)
        visual = _draw_acdlr_visual(image, circles, truth, result, score_matrix, stats_grid, args, landing_point)
        visual_path = visuals_dir / f"{image_path.stem}_acdlr.png"
        cv2.imwrite(str(visual_path), visual)
        visual_paths.append(visual_path)

    summary = evaluation.aggregate(eval_results).to_dict() if eval_results else None
    _write_rows(out_dir / "acdlr_results.csv", rows)
    _write_json(
        out_dir / "acdlr_summary.json",
        {
            "images_processed": len(rows),
            "annotations_found": bool(eval_results),
            "summary": summary,
            "visuals": [str(path) for path in visual_paths],
        },
    )

    return {
        "rows": rows,
        "summary": summary,
        "annotations_found": bool(eval_results),
        "visuals": [str(path) for path in visual_paths],
        "first_visual": str(visual_paths[0]) if visual_paths else "",
    }


def _run_deepmoon_section(args: argparse.Namespace, out_dir: Path) -> dict:
    deepmoon_dir = (REPO_ROOT / args.deepmoon_dir).resolve()
    if not deepmoon_dir.exists():
        return {"status": f"DeepMoon repository not found: {deepmoon_dir}"}

    sample_path = deepmoon_dir / "tests" / "sample_template_match.hdf5"
    if not sample_path.exists():
        return {"status": f"DeepMoon sample not found: {sample_path}"}

    try:
        sys.path.insert(0, str(deepmoon_dir))
        import utils.template_match_target as tmt
    except Exception as exc:  # pragma: no cover - environment guard
        return {"status": f"Could not import DeepMoon template matcher: {exc}"}

    with h5py.File(sample_path, "r") as sample:
        csv_data = sample["csv"][...].T
        pred = sample["pred"][...]

    truth = np.array((csv_data[3], csv_data[4], csv_data[5] / 2.0)).T
    detections = tmt.template_match_t(pred.copy(), minrad=8, maxrad=11)
    metrics_raw = tmt.template_match_t2c(pred.copy(), truth.copy(), minrad=8, maxrad=11)
    metric_names = [
        "matches",
        "ground_truth",
        "detections",
        "max_radius_px",
        "mean_x_error_ratio",
        "mean_y_error_ratio",
        "mean_radius_error_ratio",
        "csv_duplicate_fraction",
    ]
    metrics = dict(zip(metric_names, [float(value) for value in metrics_raw]))
    metrics["precision"] = metrics["matches"] / metrics["detections"] if metrics["detections"] else 0.0
    metrics["recall"] = metrics["matches"] / metrics["ground_truth"] if metrics["ground_truth"] else 0.0
    metrics["f1"] = (
        2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"])
        else 0.0
    )

    _write_json(out_dir / "deepmoon_sample_metrics.json", metrics)
    _write_detection_csv(out_dir / "deepmoon_sample_detections.csv", detections)
    visual_path = out_dir / "deepmoon_sample_visual.png"
    _draw_deepmoon_visual(visual_path, pred, truth, detections)

    return {
        "status": "ok",
        "sample_path": str(sample_path),
        "metrics": metrics,
        "visual_path": str(visual_path),
    }


def _draw_acdlr_visual(
    image: np.ndarray,
    circles: np.ndarray,
    truth: list[evaluation.GroundTruthCrater],
    result: evaluation.EvaluationResult | None,
    score_matrix: np.ndarray,
    stats_grid,
    args: argparse.Namespace,
    landing_point: risk.LandingPoint,
) -> np.ndarray:
    craters = measurement.measure(circles, scale_m_per_px=args.scale_m_per_px)
    vis = visualization.draw_final(
        image,
        craters,
        score_matrix,
        stats_grid,
        args.grid_rows,
        args.grid_cols,
        landing_point=landing_point,
    )

    if result is None:
        _put_text(vis, "ACDLR: detections only | no local annotation metrics", (14, vis.shape[0] - 42))
        return vis

    matched_detections = {match.detection_index for match in result.matches}
    matched_truth = {match.truth_index for match in result.matches}
    det_rows = np.asarray(circles, dtype=float) if circles.size else np.empty((0, 3), dtype=float)

    for idx, (x, y, radius) in enumerate(det_rows[:, :3]):
        color = (80, 255, 100) if idx in matched_detections else (70, 70, 255)
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(radius)), color, 2)

    for idx, crater in enumerate(truth):
        if idx in matched_truth:
            continue
        cv2.circle(
            vis,
            (int(round(crater.cx)), int(round(crater.cy))),
            int(round(crater.radius_px)),
            (80, 220, 255),
            2,
        )

    _put_text(
        vis,
        (
            f"ACDLR: P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f} | "
            "green=TP red=FP yellow=FN"
        ),
        (14, vis.shape[0] - 42),
    )
    return vis


def _draw_deepmoon_visual(path: Path, pred: np.ndarray, truth: np.ndarray, detections: np.ndarray) -> None:
    pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR)

    for x, y, radius in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(radius)), (80, 220, 255), 1)
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (80, 220, 255), -1)

    for x, y, radius in np.asarray(detections):
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(radius)), (80, 255, 100), 1)
        cv2.drawMarker(
            vis,
            (int(round(x)), int(round(y))),
            (80, 255, 100),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )

    _put_text(vis, "DeepMoon sample: yellow=truth green=template match", (8, 18), scale=0.42)
    cv2.imwrite(str(path), vis)


def _write_side_by_side(out_dir: Path, acdlr_visual: str | None, deepmoon_visual: str | None) -> str:
    images: list[tuple[str, np.ndarray]] = []
    for label, raw_path in (("ACDLR local dataset", acdlr_visual), ("DeepMoon official sample", deepmoon_visual)):
        if not raw_path:
            continue
        img = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((label, img))

    if not images:
        return ""

    target_h = 620
    panels = []
    for label, img in images:
        h, w = img.shape[:2]
        scale = target_h / max(h, 1)
        resized = cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)
        band = np.zeros((38, resized.shape[1], 3), dtype=np.uint8)
        _put_text(band, label, (10, 25), scale=0.65, bold=True)
        panels.append(np.vstack([band, resized]))

    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)

    side_by_side = np.hstack(padded)
    out_path = out_dir / "method_side_by_side.png"
    cv2.imwrite(str(out_path), side_by_side)
    return str(out_path)


def _write_report(
    out_dir: Path,
    acdlr: dict,
    deepmoon: dict,
    side_by_side: str,
    args: argparse.Namespace,
) -> Path:
    report_path = out_dir / "comparison_report.md"
    acdlr_summary = acdlr.get("summary")
    deepmoon_metrics = deepmoon.get("metrics")

    lines = [
        "# ACDLR x DeepMoon small comparison",
        "",
        "## Scope",
        "",
        "ACDLR was run on local LROC visual tiles using only classical image processing.",
        "DeepMoon was run locally on its bundled post-CNN template-matching sample.",
        "The full DeepMoon CNN stack depends on legacy TensorFlow/Keras/Cartopy versions",
        "and is not runnable in this Python 3.13 environment without a separate legacy runtime.",
        "",
        "## Visual output",
        "",
        f"- Side-by-side PNG: `{side_by_side}`" if side_by_side else "- Side-by-side PNG: unavailable",
        f"- ACDLR visuals: `{out_dir / 'acdlr_visuals'}`",
        "",
        "## ACDLR local dataset",
        "",
        f"- Images processed: {len(acdlr.get('rows', []))}",
        f"- Annotations found: {bool(acdlr.get('annotations_found'))}",
        f"- Detector params: min_radius={args.min_radius}, max_radius={args.max_radius}, "
        f"canny={args.canny_threshold}, strictness={args.strictness}",
        "",
    ]

    if acdlr_summary:
        lines.extend(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Precision | {acdlr_summary['precision']:.3f} |",
                f"| Recall | {acdlr_summary['recall']:.3f} |",
                f"| F1 | {acdlr_summary['f1']:.3f} |",
                f"| Mean center error ratio | {acdlr_summary['mean_center_error_ratio']:.3f} |",
                f"| Mean radius error ratio | {acdlr_summary['mean_radius_error_ratio']:.3f} |",
                "",
            ]
        )
    else:
        lines.extend(_acdlr_descriptive_table(acdlr.get("rows", [])))
        lines.extend(
            [
                "Precision, recall and F1 are not reported for ACDLR yet because no",
                "`data/annotations/*.csv` or `*.json` files were found for the selected local tiles.",
                "The generated ACDLR CSV/PNG files are detector outputs, not ground-truth metrics.",
                "",
            ]
        )

    lines.extend(["## DeepMoon local sample", ""])
    if deepmoon_metrics:
        lines.extend(
            [
                f"- Sample: `{deepmoon.get('sample_path', '')}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Matches | {deepmoon_metrics['matches']:.0f} |",
                f"| Ground truth | {deepmoon_metrics['ground_truth']:.0f} |",
                f"| Detections | {deepmoon_metrics['detections']:.0f} |",
                f"| Precision | {deepmoon_metrics['precision']:.3f} |",
                f"| Recall | {deepmoon_metrics['recall']:.3f} |",
                f"| F1 | {deepmoon_metrics['f1']:.3f} |",
                f"| Mean x error ratio | {deepmoon_metrics['mean_x_error_ratio']:.3f} |",
                f"| Mean y error ratio | {deepmoon_metrics['mean_y_error_ratio']:.3f} |",
                f"| Mean radius error ratio | {deepmoon_metrics['mean_radius_error_ratio']:.3f} |",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: {deepmoon.get('status', 'unavailable')}", ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "These numbers are not an apples-to-apples scientific comparison yet:",
            "ACDLR is processing visual LROC tiles, while the DeepMoon sample is a saved CNN",
            "prediction mask from a DEM-based workflow. The useful comparison here is the",
            "evaluation protocol: precision, recall, F1 and normalized center/radius errors.",
            "",
            "To obtain a fair ACDLR precision/recall benchmark, annotate a small set of local",
            "tiles in `data/annotations` with `cx,cy,radius_px`, then rerun this script.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _acdlr_descriptive_table(rows: list[dict]) -> list[str]:
    if not rows:
        return []

    lines = [
        "| Tile | Detections | Mean diameter m | Mean risk | Best cell | Landing clearance m |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {image} | {detections} | {mean_diameter_m:.1f} | {mean_risk:.2f} | "
            "{best_cell} | {landing_clearance_m:.1f} |".format(**row)
        )
    lines.append("")
    return lines


def _write_detection_csv(path: Path, circles: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "radius_px"])
        for row in np.asarray(circles):
            writer.writerow([float(row[0]), float(row[1]), float(row[2])])


def _write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
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


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (255, 255, 255),
    bold: bool = False,
) -> None:
    thickness = 2 if bold else 1
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Run ACDLR on the official LROC NAC_ROI_TORICELILOA tile dataset.

This is a visual/descriptive validation pass. It does not report precision or
recall unless manual annotations are supplied separately through
benchmark_classical.py, because the LROC mosaic itself is not a crater
ground-truth file.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core import detection, measurement, preprocessing, risk, tiling


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        default="data/lroc_nac_roi_toriceliloa_tiles",
        help="Directory containing ROI_TORICELILOA image tiles",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/toricelli_validation",
        help="Directory where validation artifacts are written",
    )
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
    parser.add_argument(
        "--visual-count",
        type=int,
        default=12,
        help="Number of representative final-result PNGs to write",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not image_paths:
        raise SystemExit(f"No image tiles found in {images_dir}")

    rows: list[dict[str, str | int | float]] = []
    visual_candidates: list[tuple[float, Path, np.ndarray]] = []

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        result = _run_acdlr(image_bgr, args)
        row = _row_from_result(image_path, result, args.scale_m_per_px)
        rows.append(row)

        final_img = _draw_final(
            image_bgr,
            result["craters"],
            result["score_matrix"],
            result["stats_grid"],
            args.grid_rows,
            args.grid_cols,
            landing_point=result["landing_point"],
        )
        visual_priority = float(row["crater_count"]) + float(row["mean_risk"])
        visual_candidates.append((visual_priority, image_path, final_img))

        print(
            f"[{index:03d}/{len(image_paths):03d}] {image_path.name}: "
            f"{row['crater_count']} crater(s), mean risk {row['mean_risk']:.1f}, "
            f"best R{row['best_row']}C{row['best_col']}"
        )

    rows.sort(key=lambda row: str(row["image"]))
    _write_csv(out_dir / "acdlr_toricelli_5m_summary.csv", rows)

    selected_visuals = sorted(visual_candidates, key=lambda item: item[0], reverse=True)
    selected_visuals = selected_visuals[: max(args.visual_count, 0)]
    visual_paths = []
    for _, image_path, final_img in selected_visuals:
        out_path = visuals_dir / f"{image_path.stem}_final.png"
        cv2.imwrite(str(out_path), final_img)
        visual_paths.append(out_path)

    if visual_paths:
        montage = _make_montage([cv2.imread(str(path), cv2.IMREAD_COLOR) for path in visual_paths])
        cv2.imwrite(str(out_dir / "toricelli_validation_montage.png"), montage)

    summary = _summary(rows)
    _write_json(out_dir / "acdlr_toricelli_5m_summary.json", summary)
    _write_report(out_dir / "acdlr_toricelli_5m_report.md", summary, rows, visual_paths)

    print("")
    print("LROC ROI_TORICELILOA validation complete")
    print(f"tiles processed: {summary['tiles_processed']}")
    print(f"total detections: {summary['total_craters']}")
    print(f"mean detections/tile: {summary['mean_craters_per_tile']:.2f}")
    print(f"summary csv: {out_dir / 'acdlr_toricelli_5m_summary.csv'}")
    print(f"report: {out_dir / 'acdlr_toricelli_5m_report.md'}")
    if visual_paths:
        print(f"montage: {out_dir / 'toricelli_validation_montage.png'}")


def _run_acdlr(image_bgr: np.ndarray, args: argparse.Namespace) -> dict:
    tiles = tiling.split(image_bgr, tile_size=args.tile_size, overlap=args.overlap)
    all_circles: list[np.ndarray] = []

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

    circles = tiling.deduplicate(np.vstack(all_circles)) if all_circles else np.empty((0, 3), dtype=int)
    craters = measurement.measure(circles, scale_m_per_px=args.scale_m_per_px)
    score_matrix, stats_grid = risk.analyse(
        craters,
        image_shape=image_bgr.shape,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        scale_m_per_px=args.scale_m_per_px,
    )
    best_r, best_c = risk.best_landing_cell(score_matrix)
    landing_point = risk.suggest_landing_point(
        craters,
        image_shape=image_bgr.shape,
        best_row=best_r,
        best_col=best_c,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        scale_m_per_px=args.scale_m_per_px,
    )

    return {
        "circles": circles,
        "craters": craters,
        "score_matrix": score_matrix,
        "stats_grid": stats_grid,
        "best_r": best_r,
        "best_c": best_c,
        "landing_point": landing_point,
    }


def _row_from_result(image_path: Path, result: dict, scale_m_per_px: float) -> dict[str, str | int | float]:
    craters = result["craters"]
    stats = measurement.summary_stats(craters)
    score_matrix = result["score_matrix"]
    landing_point = result["landing_point"]

    return {
        "image": image_path.name,
        "scale_m_per_px": scale_m_per_px,
        "crater_count": int(stats["count"]),
        "mean_diameter_m": float(stats["mean_diameter_m"]),
        "max_diameter_m": float(stats["max_diameter_m"]),
        "min_diameter_m": float(stats["min_diameter_m"]),
        "mean_radius_px": float(stats["mean_radius_px"]),
        "mean_risk": float(np.mean(score_matrix)),
        "max_risk": float(np.max(score_matrix)),
        "min_risk": float(np.min(score_matrix)),
        "best_row": int(result["best_r"] + 1),
        "best_col": int(result["best_c"] + 1),
        "landing_x_px": int(landing_point.x),
        "landing_y_px": int(landing_point.y),
        "landing_clearance_m": float(landing_point.clearance_m),
    }


def _summary(rows: list[dict[str, str | int | float]]) -> dict[str, float | int | str]:
    if not rows:
        return {
            "tiles_processed": 0,
            "total_craters": 0,
            "mean_craters_per_tile": 0.0,
            "mean_risk": 0.0,
            "highest_risk_tile": "",
            "highest_risk": 0.0,
            "most_cratered_tile": "",
            "most_craters": 0,
        }

    total_craters = int(sum(int(row["crater_count"]) for row in rows))
    mean_risk = float(np.mean([float(row["mean_risk"]) for row in rows]))
    highest_risk_row = max(rows, key=lambda row: float(row["mean_risk"]))
    most_cratered_row = max(rows, key=lambda row: int(row["crater_count"]))

    return {
        "tiles_processed": len(rows),
        "total_craters": total_craters,
        "mean_craters_per_tile": float(total_craters / len(rows)),
        "mean_risk": mean_risk,
        "highest_risk_tile": str(highest_risk_row["image"]),
        "highest_risk": float(highest_risk_row["mean_risk"]),
        "most_cratered_tile": str(most_cratered_row["image"]),
        "most_craters": int(most_cratered_row["crater_count"]),
    }


def _make_montage(images: list[np.ndarray | None], thumb_w: int = 360) -> np.ndarray:
    valid = [img for img in images if img is not None]
    if not valid:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    thumbs = []
    for img in valid:
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb = cv2.resize(img, (thumb_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    thumb_h = max(img.shape[0] for img in thumbs)
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y = r * thumb_h
        x = c * thumb_w
        canvas[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    return canvas


def _draw_final(
    image_bgr: np.ndarray,
    craters: list[measurement.Crater],
    score_matrix: np.ndarray,
    stats_grid,
    grid_rows: int,
    grid_cols: int,
    landing_point: risk.LandingPoint,
) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    for crater in craters:
        cv2.circle(vis, (crater.cx, crater.cy), crater.radius_px, (90, 235, 110), 2)
        cv2.circle(vis, (crater.cx, crater.cy), 2, (255, 255, 0), -1)

    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    best_r, best_c = divmod(int(np.argmin(score_matrix)), score_matrix.shape[1])

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, x2 = x_edges[c], x_edges[c + 1]
            y1, y2 = y_edges[r], y_edges[r + 1]
            score = float(score_matrix[r, c])
            stats = stats_grid[r][c]
            fill = _risk_color(score)

            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

            is_best = r == best_r and c == best_c
            if is_best:
                border = (0, 255, 0)
                thickness = 3
            elif stats.risk_label == "HIGH":
                border = (0, 0, 220)
                thickness = 2
            elif stats.risk_label == "MEDIUM":
                border = (0, 165, 255)
                thickness = 2
            else:
                border = (210, 210, 210)
                thickness = 1

            cv2.rectangle(vis, (x1, y1), (x2, y2), border, thickness)
            _put_text(vis, f"Risk {score:.0f}", (x1 + 8, y1 + 24), 0.58)
            _put_text(vis, f"{stats.crater_count} craters", (x1 + 8, y1 + 46), 0.48)
            if is_best:
                _put_text(vis, "BEST ZONE", (x1 + 8, y2 - 10), 0.58, (0, 255, 0), True)

    cv2.drawMarker(
        vis,
        (landing_point.x, landing_point.y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
    )
    cv2.circle(vis, (landing_point.x, landing_point.y), 10, (255, 255, 255), 2)
    _put_text(
        vis,
        f"Landing clearance ~ {landing_point.clearance_m:.1f} m",
        (18, max(24, h - 18)),
        0.55,
        (255, 255, 255),
        True,
    )
    return vis


def _risk_color(score: float) -> tuple[int, int, int]:
    t = float(np.clip(score / 100.0, 0.0, 1.0))
    return (0, int(255 * (1.0 - t)), int(255 * t))


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    bold: bool = False,
) -> None:
    thickness = 2 if bold else 1
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, float | int | str]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(
    path: Path,
    summary: dict[str, float | int | str],
    rows: list[dict[str, str | int | float]],
    visual_paths: list[Path],
) -> None:
    top_rows = sorted(rows, key=lambda row: int(row["crater_count"]), reverse=True)[:8]

    lines = [
        "# ACDLR validation on LROC NAC_ROI_TORICELILOA",
        "",
        "Dataset: NAC_ROI_TORICELILOA_E047S0284_5M.TIF, official LROC browse TIF.",
        "Scale used in this run: 5.00 m/px.",
        "",
        "This run is a visual/descriptive validation. Precision, recall and F1 require",
        "manual crater annotations for these same tiles.",
        "",
        "## Summary",
        "",
        f"- Tiles processed: {summary['tiles_processed']}",
        f"- Total crater detections: {summary['total_craters']}",
        f"- Mean detections per tile: {summary['mean_craters_per_tile']:.2f}",
        f"- Mean risk score: {summary['mean_risk']:.2f}",
        f"- Highest-risk tile: {summary['highest_risk_tile']} ({summary['highest_risk']:.2f})",
        f"- Most-cratered tile: {summary['most_cratered_tile']} ({summary['most_craters']} crater(s))",
        "",
        "## Most-cratered tiles",
        "",
        "| Tile | Craters | Mean risk | Best cell | Landing clearance m |",
        "|---|---:|---:|---|---:|",
    ]

    for row in top_rows:
        lines.append(
            "| {image} | {crater_count} | {mean_risk:.2f} | R{best_row}C{best_col} | {landing_clearance_m:.1f} |".format(
                **row
            )
        )

    lines.extend(["", "## Visual artifacts", ""])
    for visual_path in visual_paths:
        lines.append(f"- {visual_path}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

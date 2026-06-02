from __future__ import annotations

"""
Run one complete ACDLR visual pipeline test on a LROC ROI_TORICELILOA tile.

The script writes one PNG per pipeline stage plus a contact sheet and a small
JSON/Markdown validation summary.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core import detection, measurement, preprocessing, risk, tiling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="data/lroc_nac_roi_toriceliloa_tiles/torricelli_5m_y05760_x03840.png",
        help="LROC tile to test",
    )
    parser.add_argument("--out-dir", default="artifacts/toricelli_step_test")
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
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    full_prep = preprocessing.run(
        image_bgr,
        clahe_clip=args.clahe_clip,
        blur_ksize=args.blur_ksize,
    )
    result = _run_pipeline(image_bgr, args)

    stage_images = [
        ("01_original", image_bgr),
        ("02_grayscale", _gray_to_bgr(full_prep.gray)),
        ("03_clahe", _gray_to_bgr(full_prep.enhanced)),
        ("04_denoised", _gray_to_bgr(full_prep.denoised)),
        ("05_sharpened", _gray_to_bgr(full_prep.sharpened)),
        ("06_edge_hint", _gray_to_bgr(full_prep.edge_hint)),
        ("07_detected_craters", _draw_craters(image_bgr, result["craters"])),
        (
            "08_risk_grid",
            _draw_risk_grid(
                image_bgr,
                result["score_matrix"],
                result["stats_grid"],
                args.grid_rows,
                args.grid_cols,
            ),
        ),
        (
            "09_final_result",
            _draw_final(
                image_bgr,
                result["craters"],
                result["score_matrix"],
                result["stats_grid"],
                args.grid_rows,
                args.grid_cols,
                result["landing_point"],
            ),
        ),
    ]

    written = []
    for name, stage_image in stage_images:
        path = out_dir / f"{name}.png"
        _write_png(path, stage_image)
        written.append(path)

    contact_sheet = _make_contact_sheet(stage_images)
    contact_path = out_dir / "00_pipeline_contact_sheet.png"
    _write_png(contact_path, contact_sheet)
    written.insert(0, contact_path)

    stats = measurement.summary_stats(result["craters"])
    summary = {
        "source_image": str(image_path),
        "scale_m_per_px": args.scale_m_per_px,
        "image_width_px": int(image_bgr.shape[1]),
        "image_height_px": int(image_bgr.shape[0]),
        "crater_count": int(stats["count"]),
        "mean_diameter_m": float(stats["mean_diameter_m"]),
        "max_diameter_m": float(stats["max_diameter_m"]),
        "mean_risk": float(np.mean(result["score_matrix"])),
        "max_risk": float(np.max(result["score_matrix"])),
        "min_risk": float(np.min(result["score_matrix"])),
        "best_row": int(result["best_r"] + 1),
        "best_col": int(result["best_c"] + 1),
        "landing_x_px": int(result["landing_point"].x),
        "landing_y_px": int(result["landing_point"].y),
        "landing_clearance_m": float(result["landing_point"].clearance_m),
        "stage_pngs": len(written),
        "all_stage_pngs_nonempty": all(path.exists() and path.stat().st_size > 0 for path in written),
    }

    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(out_dir / "validation_report.md", summary, written)

    print("ACDLR Toricelli step test complete")
    print(f"image: {image_path}")
    print(f"craters: {summary['crater_count']}")
    print(f"mean risk: {summary['mean_risk']:.2f}")
    print(f"best cell: R{summary['best_row']}C{summary['best_col']}")
    print(f"landing clearance: {summary['landing_clearance_m']:.1f} m")
    print(f"stage pngs: {summary['stage_pngs']}")
    print(f"all pngs nonempty: {summary['all_stage_pngs_nonempty']}")
    print(f"contact sheet: {contact_path}")
    print(f"summary: {summary_path}")


def _run_pipeline(image_bgr: np.ndarray, args: argparse.Namespace) -> dict:
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
        "craters": craters,
        "score_matrix": score_matrix,
        "stats_grid": stats_grid,
        "best_r": best_r,
        "best_c": best_c,
        "landing_point": landing_point,
    }


def _draw_craters(image_bgr: np.ndarray, craters: list[measurement.Crater]) -> np.ndarray:
    vis = image_bgr.copy()
    for crater in craters:
        cv2.circle(vis, (crater.cx, crater.cy), crater.radius_px, (90, 235, 110), 2)
        cv2.circle(vis, (crater.cx, crater.cy), 2, (255, 255, 0), -1)
    _put_text(vis, f"{len(craters)} detected craters", (16, 28), 0.7, bold=True)
    return vis


def _draw_risk_grid(image_bgr: np.ndarray, score_matrix: np.ndarray, stats_grid, grid_rows: int, grid_cols: int) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    best_r, best_c = divmod(int(np.argmin(score_matrix)), score_matrix.shape[1])

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, x2 = x_edges[c], x_edges[c + 1]
            y1, y2 = y_edges[r], y_edges[r + 1]
            score = float(score_matrix[r, c])
            stats = stats_grid[r][c]
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _risk_color(score), -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

            is_best = r == best_r and c == best_c
            border = (0, 255, 0) if is_best else (0, 0, 220) if stats.risk_label == "HIGH" else (0, 165, 255)
            thickness = 3 if is_best else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), border, thickness)
            _put_text(vis, f"Risk {score:.0f}", (x1 + 8, y1 + 24), 0.58)
            _put_text(vis, f"{stats.crater_count} craters", (x1 + 8, y1 + 46), 0.48)
            if is_best:
                _put_text(vis, "BEST ZONE", (x1 + 8, y2 - 10), 0.58, (0, 255, 0), True)
    return vis


def _draw_final(
    image_bgr: np.ndarray,
    craters: list[measurement.Crater],
    score_matrix: np.ndarray,
    stats_grid,
    grid_rows: int,
    grid_cols: int,
    landing_point: risk.LandingPoint,
) -> np.ndarray:
    vis = _draw_craters(image_bgr, craters)
    vis = _draw_risk_grid(vis, score_matrix, stats_grid, grid_rows, grid_cols)
    cv2.drawMarker(
        vis,
        (landing_point.x, landing_point.y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2,
    )
    cv2.circle(vis, (landing_point.x, landing_point.y), 11, (255, 255, 255), 2)
    _put_text(vis, f"Landing clearance ~ {landing_point.clearance_m:.1f} m", (16, vis.shape[0] - 16), 0.58, bold=True)
    return vis


def _make_contact_sheet(stage_images: list[tuple[str, np.ndarray]], thumb_w: int = 420) -> np.ndarray:
    thumbs = []
    for name, img in stage_images:
        h, w = img.shape[:2]
        thumb_h = max(1, int(h * thumb_w / max(w, 1)))
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        label_band = np.zeros((34, thumb_w, 3), dtype=np.uint8)
        _put_text(label_band, name, (8, 24), 0.58, bold=True)
        thumbs.append(np.vstack([label_band, thumb]))

    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(thumb.shape[0] for thumb in thumbs)
    canvas = np.zeros((rows * cell_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y = r * cell_h
        x = c * thumb_w
        canvas[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return canvas


def _gray_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _risk_color(score: float) -> tuple[int, int, int]:
    t = float(np.clip(score / 100.0, 0.0, 1.0))
    return (0, int(255 * (1.0 - t)), int(255 * t))


def _write_png(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise SystemExit(f"Failed to write PNG: {path}")


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


def _write_report(path: Path, summary: dict, written: list[Path]) -> None:
    lines = [
        "# ACDLR Toricelli step test",
        "",
        f"- Source image: {summary['source_image']}",
        f"- Scale: {summary['scale_m_per_px']:.2f} m/px",
        f"- Image size: {summary['image_width_px']} x {summary['image_height_px']} px",
        f"- Detected craters: {summary['crater_count']}",
        f"- Mean diameter: {summary['mean_diameter_m']:.1f} m",
        f"- Largest diameter: {summary['max_diameter_m']:.1f} m",
        f"- Mean risk: {summary['mean_risk']:.2f}",
        f"- Risk range: {summary['min_risk']:.2f} to {summary['max_risk']:.2f}",
        f"- Best cell: R{summary['best_row']}C{summary['best_col']}",
        f"- Landing point: ({summary['landing_x_px']}, {summary['landing_y_px']})",
        f"- Landing clearance: {summary['landing_clearance_m']:.1f} m",
        f"- PNG validation: {summary['stage_pngs']} file(s), non-empty={summary['all_stage_pngs_nonempty']}",
        "",
        "## Stage prints",
        "",
    ]
    lines.extend(f"- {path}" for path in written)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

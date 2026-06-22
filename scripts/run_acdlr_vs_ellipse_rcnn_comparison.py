from __future__ import annotations

"""
Run the fair visual-dataset comparison: ACDLR vs pretrained Ellipse R-CNN.

Both methods are evaluated on the same YOLO visual dataset split and with the
same circle-matching logic.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair ACDLR x Ellipse R-CNN comparison.")
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--visual-count", type=int, default=3)
    parser.add_argument("--out-dir", default="artifacts/acdlr_vs_ellipse_rcnn")
    parser.add_argument("--ellipse-model", default="artifacts/ellipse_rcnn_pretrained/crater-rcnn")
    parser.add_argument("--ellipse-score-threshold", type=float, default=0.60)
    parser.add_argument("--ellipse-max-detections", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-radius", type=int, default=4)
    parser.add_argument("--max-radius", type=int, default=70)
    parser.add_argument("--canny-threshold", type=int, default=45)
    parser.add_argument("--strictness", type=int, default=16)
    parser.add_argument("--center-tolerance", type=float, default=1.34)
    parser.add_argument("--radius-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    acdlr_dir = out_dir / "acdlr"
    ellipse_dir = out_dir / "ellipse_rcnn"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_yolo_dataset.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--split",
            args.split,
            "--max-images",
            str(args.max_images),
            "--visual-count",
            str(args.visual_count),
            "--out-dir",
            str(acdlr_dir),
            "--min-radius",
            str(args.min_radius),
            "--max-radius",
            str(args.max_radius),
            "--canny-threshold",
            str(args.canny_threshold),
            "--strictness",
            str(args.strictness),
            "--center-tolerance",
            str(args.center_tolerance),
            "--radius-tolerance",
            str(args.radius_tolerance),
        ]
    )

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_ellipse_rcnn_yolo_dataset.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--split",
            args.split,
            "--max-images",
            str(args.max_images),
            "--visual-count",
            str(args.visual_count),
            "--out-dir",
            str(ellipse_dir),
            "--model",
            args.ellipse_model,
            "--score-threshold",
            str(args.ellipse_score_threshold),
            "--max-detections",
            str(args.ellipse_max_detections),
            "--device",
            args.device,
            "--center-tolerance",
            str(args.center_tolerance),
            "--radius-tolerance",
            str(args.radius_tolerance),
        ]
    )

    acdlr_summary_path = acdlr_dir / "acdlr_yolo_summary.json"
    ellipse_summary_path = ellipse_dir / "ellipse_rcnn_yolo_summary.json"
    acdlr = json.loads(acdlr_summary_path.read_text(encoding="utf-8"))
    ellipse = json.loads(ellipse_summary_path.read_text(encoding="utf-8"))

    chart_path = _write_metrics_chart(charts_dir / "acdlr_vs_ellipse_rcnn_metrics.png", acdlr, ellipse)
    visual_path = _write_visual_comparison(out_dir, acdlr_dir, ellipse_dir, acdlr, ellipse)
    report_path = _write_report(out_dir, args, acdlr, ellipse, acdlr_summary_path, ellipse_summary_path, visual_path, chart_path)

    summary = {
        "comparison": "ACDLR classical detector vs Ellipse R-CNN pretrained visual crater detector",
        "selected_ai_baseline": "wdoppenberg/crater-rcnn",
        "baseline_repo": "https://github.com/wdoppenberg/ellipse-rcnn",
        "baseline_model": "https://huggingface.co/wdoppenberg/crater-rcnn",
        "acdlr_summary": _display_path(acdlr_summary_path),
        "ellipse_summary": _display_path(ellipse_summary_path),
        "visual_comparison": _display_path(visual_path),
        "metrics_chart": _display_path(chart_path),
        "report": _display_path(report_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("ACDLR x Ellipse R-CNN comparison generated")
    print(f"report: {report_path}")
    print(f"visual: {visual_path}")
    print(f"chart: {chart_path}")


def _run(command: list[str]) -> None:
    print("")
    print("Running:", " ".join(command))
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def _write_metrics_chart(path: Path, acdlr: dict, ellipse: dict) -> Path:
    labels = ["Precision", "Recall", "F1"]
    acdlr_values = [acdlr["precision"], acdlr["recall"], acdlr["f1"]]
    ellipse_values = [ellipse["precision"], ellipse["recall"], ellipse["f1"]]
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar([i - width / 2 for i in x], acdlr_values, width, label="ACDLR", color="#2bb673")
    ax.bar([i + width / 2 for i in x], ellipse_values, width, label="Ellipse R-CNN", color="#4c78a8")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Same visual dataset: ACDLR vs Ellipse R-CNN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_visual_comparison(out_dir: Path, acdlr_dir: Path, ellipse_dir: Path, acdlr: dict, ellipse: dict) -> Path:
    acdlr_visuals = sorted((acdlr_dir / "visuals").glob("*_matches.png"))
    ellipse_visuals = sorted((ellipse_dir / "visuals").glob("*_matches.png"))
    if not acdlr_visuals or not ellipse_visuals:
        raise SystemExit("Missing visual overlays for comparison")
    acdlr_img = cv2.imread(str(acdlr_visuals[0]), cv2.IMREAD_COLOR)
    ellipse_img = cv2.imread(str(ellipse_visuals[0]), cv2.IMREAD_COLOR)
    if acdlr_img is None or ellipse_img is None:
        raise SystemExit("Could not load comparison images")

    panels = [
        ("ACDLR classical CV", acdlr_img),
        ("Ellipse R-CNN pretrained visual CNN", ellipse_img),
    ]
    rendered = []
    max_panel_w = 680
    max_panel_h = 520
    for title, image in panels:
        resized = _fit_panel(image, max_panel_w, max_panel_h)
        band = cv2.copyMakeBorder(resized, 38, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        _put_text(band, title, (10, 25), scale=0.58, bold=True)
        rendered.append(band)

    max_h = max(panel.shape[0] for panel in rendered)
    padded = []
    for panel in rendered:
        if panel.shape[0] < max_h:
            panel = cv2.copyMakeBorder(panel, 0, max_h - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(panel)

    canvas = cv2.hconcat(padded)
    footer = cv2.copyMakeBorder(canvas, 0, 48, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    _put_text(
        footer,
        "Same input split, same YOLO labels converted to circles, same matching tolerances.",
        (10, canvas.shape[0] + 22),
        scale=0.48,
    )
    _put_text(
        footer,
        f"ACDLR F1={acdlr['f1']:.3f} | Ellipse R-CNN F1={ellipse['f1']:.3f}",
        (10, canvas.shape[0] + 42),
        scale=0.48,
    )
    out_path = out_dir / "visual_comparison.png"
    cv2.imwrite(str(out_path), footer)
    return out_path


def _fit_panel(image: cv2.typing.MatLike, max_width: int, max_height: int) -> cv2.typing.MatLike:
    h, w = image.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    acdlr: dict,
    ellipse: dict,
    acdlr_summary_path: Path,
    ellipse_summary_path: Path,
    visual_path: Path,
    chart_path: Path,
) -> Path:
    report_path = out_dir / "comparison_report.md"
    lines = [
        "# ACDLR x Ellipse R-CNN comparison",
        "",
        "## Why this comparison is fairer",
        "",
        "Both methods are evaluated on the same visual YOLO dataset split. ACDLR predicts circles; Ellipse R-CNN predicts ellipses, converted to circles with `radius=(a+b)/2`. Both are matched against the same labels with the same tolerances.",
        "",
        "- Ellipse R-CNN repo: https://github.com/wdoppenberg/ellipse-rcnn",
        "- Pretrained crater model: https://huggingface.co/wdoppenberg/crater-rcnn",
        "- Ellipse R-CNN paper: https://arxiv.org/abs/2001.11584",
        "",
        "## Metrics",
        "",
        "| Method | Dataset/source | Detections | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| ACDLR | `{args.dataset_dir}` `{args.split}` | {acdlr['detections']} | {acdlr['ground_truth']} | "
            f"{acdlr['true_positive']} | {acdlr['false_positive']} | {acdlr['false_negative']} | "
            f"{acdlr['precision']:.4f} | {acdlr['recall']:.4f} | {acdlr['f1']:.4f} |"
        ),
        (
            f"| Ellipse R-CNN | same split | {ellipse['detections']} | {ellipse['ground_truth']} | "
            f"{ellipse['true_positive']} | {ellipse['false_positive']} | {ellipse['false_negative']} | "
            f"{ellipse['precision']:.4f} | {ellipse['recall']:.4f} | {ellipse['f1']:.4f} |"
        ),
        "",
        "## Outputs",
        "",
        f"- ACDLR summary: `{_display_path(acdlr_summary_path)}`",
        f"- Ellipse R-CNN summary: `{_display_path(ellipse_summary_path)}`",
        f"- Visual comparison: `{_display_path(visual_path)}`",
        f"- Metrics chart: `{_display_path(chart_path)}`",
        "",
        f"![Metrics chart]({chart_path.relative_to(out_dir).as_posix()})",
        "",
        f"![Visual comparison]({visual_path.relative_to(out_dir).as_posix()})",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _put_text(image, text: str, origin: tuple[int, int], scale: float = 0.5, bold: bool = False) -> None:
    thickness = 2 if bold else 1
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


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

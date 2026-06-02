from __future__ import annotations

"""
Run the actual project comparison:
    MEU METODO: ACDLR, image processing only
    CNN: YOLOv11 crater detector inspired by the open 2026 crater repo

Both methods are evaluated on the same YOLO-format annotations.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/LU3M6TGT_yolo_format")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--out-dir", default="artifacts/acdlr_vs_crater_cnn")
    parser.add_argument("--visual-count", type=int, default=8)

    parser.add_argument("--min-radius", type=int, default=4)
    parser.add_argument("--max-radius", type=int, default=70)
    parser.add_argument("--canny-threshold", type=int, default=45)
    parser.add_argument("--strictness", type=int, default=16)
    parser.add_argument("--center-tolerance", type=float, default=1.34)
    parser.add_argument("--radius-tolerance", type=float, default=1.0)

    parser.add_argument("--cnn-weights", default="artifacts/crater_cnn_yolo_train/moon_small/weights/best.pt")
    parser.add_argument("--cnn-base-weights", default="external/crater-identification/YOLOv11model/yolo11n.pt")
    parser.add_argument("--cnn-train-epochs", type=int, default=1)
    parser.add_argument("--cnn-train-fraction", type=float, default=0.02)
    parser.add_argument("--cnn-imgsz", type=int, default=416)
    parser.add_argument("--cnn-batch", type=int, default=8)
    parser.add_argument("--cnn-device", default="cpu")
    parser.add_argument("--cnn-conf", type=float, default=0.001)
    parser.add_argument("--cnn-iou", type=float, default=0.15)
    parser.add_argument("--cnn-max-det", type=int, default=150)
    parser.add_argument("--skip-cnn-train", action="store_true")
    parser.add_argument("--force-cnn-train", action="store_true")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    acdlr_dir = out_dir / "acdlr"
    cnn_dir = out_dir / "crater_cnn_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    cnn_weights = _resolve_cnn_weights(args)
    if args.force_cnn_train or not cnn_weights.exists():
        if args.skip_cnn_train:
            raise SystemExit(
                f"CNN weights not found: {cnn_weights}\n"
                "Run without --skip-cnn-train or train first with scripts\\train_crater_cnn_yolo.py."
            )
        _train_cnn(args)
        cnn_weights = _resolve_cnn_weights(args)
        if not cnn_weights.exists():
            alt = cnn_weights.with_name("last.pt")
            cnn_weights = alt if alt.exists() else cnn_weights
        if not cnn_weights.exists():
            raise SystemExit("CNN training did not produce best.pt or last.pt.")

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
            str(REPO_ROOT / "scripts" / "benchmark_crater_cnn_yolo.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--split",
            args.split,
            "--weights",
            str(cnn_weights),
            "--max-images",
            str(args.max_images),
            "--visual-count",
            str(args.visual_count),
            "--out-dir",
            str(cnn_dir),
            "--imgsz",
            str(args.cnn_imgsz),
            "--conf",
            str(args.cnn_conf),
            "--iou",
            str(args.cnn_iou),
            "--max-det",
            str(args.cnn_max_det),
            "--device",
            args.cnn_device,
            "--center-tolerance",
            str(args.center_tolerance),
            "--radius-tolerance",
            str(args.radius_tolerance),
        ]
    )

    acdlr_summary_path = acdlr_dir / "acdlr_yolo_summary.json"
    cnn_summary_path = cnn_dir / "cnn_yolo_summary.json"
    acdlr_summary = json.loads(acdlr_summary_path.read_text(encoding="utf-8"))
    cnn_summary = json.loads(cnn_summary_path.read_text(encoding="utf-8"))

    visual_path = _write_visual_comparison(out_dir, acdlr_dir, cnn_dir, acdlr_summary, cnn_summary)
    report_path = _write_report(
        out_dir,
        acdlr_summary,
        cnn_summary,
        acdlr_summary_path,
        cnn_summary_path,
        visual_path,
        args,
    )

    run_summary = {
        "comparison": "ACDLR classical image processing vs CNN YOLOv11 crater detector",
        "acdlr_summary": str(acdlr_summary_path),
        "cnn_summary": str(cnn_summary_path),
        "visual_comparison": str(visual_path),
        "report": str(report_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("")
    print("ACDLR x CNN comparison generated")
    print(f"report: {report_path}")
    print(f"visual: {visual_path}")
    print(f"ACDLR F1: {acdlr_summary['f1']:.3f}")
    print(f"CNN F1: {cnn_summary['f1']:.3f}")


def _resolve_cnn_weights(args: argparse.Namespace) -> Path:
    path = Path(args.cnn_weights)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return path.resolve()
    alt = path.with_name("last.pt")
    return alt.resolve() if alt.exists() else path.resolve()


def _train_cnn(args: argparse.Namespace) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_crater_cnn_yolo.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--base-weights",
            args.cnn_base_weights,
            "--epochs",
            str(args.cnn_train_epochs),
            "--fraction",
            str(args.cnn_train_fraction),
            "--imgsz",
            str(args.cnn_imgsz),
            "--batch",
            str(args.cnn_batch),
            "--device",
            args.cnn_device,
            "--workers",
            "0",
            "--out-dir",
            "artifacts/crater_cnn_yolo_train",
            "--name",
            "moon_small",
        ]
    )


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    config_dir = REPO_ROOT / "artifacts" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    print("")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def _write_visual_comparison(
    out_dir: Path,
    acdlr_dir: Path,
    cnn_dir: Path,
    acdlr: dict,
    cnn: dict,
) -> Path:
    acdlr_visuals = sorted((acdlr_dir / "visuals").glob("*_matches.png"))
    cnn_visuals = sorted((cnn_dir / "visuals").glob("*_matches.png"))
    if not acdlr_visuals:
        raise SystemExit(f"No ACDLR visual overlays found in {acdlr_dir / 'visuals'}")
    if not cnn_visuals:
        raise SystemExit(f"No CNN visual overlays found in {cnn_dir / 'visuals'}")

    acdlr_img = cv2.imread(str(acdlr_visuals[0]), cv2.IMREAD_COLOR)
    cnn_img = cv2.imread(str(cnn_visuals[0]), cv2.IMREAD_COLOR)
    if acdlr_img is None or cnn_img is None:
        raise SystemExit("Could not load comparison images")

    panels = [
        (
            "MEU METODO: ACDLR",
            f"P={acdlr['precision']:.3f} R={acdlr['recall']:.3f} F1={acdlr['f1']:.3f}",
            acdlr_img,
        ),
        (
            "CNN: YOLOv11 CRATER",
            f"P={cnn['precision']:.3f} R={cnn['recall']:.3f} F1={cnn['f1']:.3f}",
            cnn_img,
        ),
    ]

    target_h = 520
    rendered = []
    for title, subtitle, image in panels:
        h, w = image.shape[:2]
        scale = target_h / max(h, 1)
        resized = cv2.resize(image, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)
        band = np.zeros((64, resized.shape[1], 3), dtype=np.uint8)
        _put_text(band, title, (10, 24), scale=0.62, bold=True)
        _put_text(band, subtitle, (10, 50), scale=0.54)
        rendered.append(np.vstack([band, resized]))

    max_h = max(panel.shape[0] for panel in rendered)
    padded = []
    for panel in rendered:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)

    canvas = np.hstack(padded)
    note = np.zeros((48, canvas.shape[1], 3), dtype=np.uint8)
    _put_text(
        note,
        "Same dataset, same YOLO labels, same matching tolerance. Green=TP, red=FP, yellow=FN.",
        (10, 30),
        scale=0.50,
    )
    canvas = np.vstack([canvas, note])

    visual_path = out_dir / "visual_comparison.png"
    cv2.imwrite(str(visual_path), canvas)
    return visual_path


def _write_report(
    out_dir: Path,
    acdlr: dict,
    cnn: dict,
    acdlr_summary_path: Path,
    cnn_summary_path: Path,
    visual_path: Path,
    args: argparse.Namespace,
) -> Path:
    report_path = out_dir / "comparison_report.md"
    winner = "ACDLR" if acdlr["f1"] >= cnn["f1"] else "CNN YOLOv11"
    lines = [
        "# ACDLR x CNN YOLOv11 comparison",
        "",
        "## Reference CNN",
        "",
        "- Article: Deep learning framework for crater detection and identification on the Moon and Mars.",
        "- Repo: https://github.com/sydney-machine-learning/crater-identification",
        "- Role here: CNN competitor only. ACDLR remains classical image processing.",
        "",
        "## Run setup",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Split: `{args.split}`",
        f"- Images compared: {acdlr['images_processed']}",
        f"- Matching tolerance: center <= {args.center_tolerance} * radius, radius error <= {args.radius_tolerance} * radius",
        f"- CNN inference: conf={args.cnn_conf}, iou={args.cnn_iou}, max_det={args.cnn_max_det}",
        f"- Visual comparison: `{visual_path}`",
        "",
        "## Metrics",
        "",
        "| Method | Detections | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("ACDLR", acdlr),
        _metric_row("CNN YOLOv11", cnn),
        "",
        f"Best F1 in this run: **{winner}**.",
        "",
        "## Output files",
        "",
        f"- ACDLR summary: `{acdlr_summary_path}`",
        f"- CNN summary: `{cnn_summary_path}`",
        f"- Side-by-side visual: `{visual_path}`",
        "",
        "## Notes",
        "",
        "- The CNN is allowed to learn from training labels because it is the comparison baseline.",
        "- The ACDLR side does not train, does not use neural networks and only uses classical image processing.",
        "- For a publication-quality result, increase `--max-images`, train for more epochs and evaluate on the full `valid` split.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _metric_row(name: str, data: dict) -> str:
    return (
        f"| {name} | {data['detections']} | {data['ground_truth']} | "
        f"{data['true_positive']} | {data['false_positive']} | {data['false_negative']} | "
        f"{data['precision']:.4f} | {data['recall']:.4f} | {data['f1']:.4f} |"
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    bold: bool = False,
) -> None:
    x, y = origin
    thickness = 2 if bold else 1
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


if __name__ == "__main__":
    main()

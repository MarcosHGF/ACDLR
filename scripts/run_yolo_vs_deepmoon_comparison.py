from __future__ import annotations

"""
Run and visualize an ACDLR x DeepMoon comparison.

This script is intentionally a convenience wrapper:
1. benchmarks the classical ACDLR detector on the YOLO annotated dataset;
2. runs the locally executable DeepMoon sample validation;
3. writes a compact Markdown report;
4. creates a side-by-side PNG with one ACDLR TP/FP/FN overlay and the DeepMoon
   sample prediction/matching visualization.

DeepMoon is not run on the YOLO visual dataset itself because the original
project expects DEM-based inputs and a legacy TensorFlow/Keras stack.
"""

import argparse
import json
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
    parser.add_argument("--out-dir", default="artifacts/yolo_vs_deepmoon")
    parser.add_argument("--min-radius", type=int, default=4)
    parser.add_argument("--max-radius", type=int, default=70)
    parser.add_argument("--canny-threshold", type=int, default=45)
    parser.add_argument("--strictness", type=int, default=16)
    parser.add_argument("--center-tolerance", type=float, default=1.34)
    parser.add_argument("--radius-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    acdlr_dir = out_dir / "acdlr"
    deepmoon_dir = out_dir / "deepmoon"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            str(REPO_ROOT / "scripts" / "validate_deepmoon_sample.py"),
            "--out-dir",
            str(deepmoon_dir),
        ]
    )

    acdlr_summary_path = acdlr_dir / "acdlr_yolo_summary.json"
    deepmoon_summary_path = deepmoon_dir / "deepmoon_sample_metrics.json"
    acdlr_summary = json.loads(acdlr_summary_path.read_text(encoding="utf-8"))
    deepmoon_summary = json.loads(deepmoon_summary_path.read_text(encoding="utf-8"))

    visual_path = _write_visual_comparison(out_dir, acdlr_dir, deepmoon_dir, acdlr_summary, deepmoon_summary)
    report_path = _write_report(
        out_dir,
        acdlr_summary,
        deepmoon_summary,
        acdlr_summary_path,
        deepmoon_summary_path,
        visual_path,
        args,
    )

    run_summary = {
        "acdlr_summary": str(acdlr_summary_path),
        "deepmoon_summary": str(deepmoon_summary_path),
        "visual_comparison": str(visual_path),
        "report": str(report_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("")
    print("ACDLR x DeepMoon comparison generated")
    print(f"report: {report_path}")
    print(f"visual: {visual_path}")
    print(f"ACDLR F1: {acdlr_summary['f1']:.3f}")
    print(f"DeepMoon sample F1: {deepmoon_summary['f1']:.3f}")


def _run(command: list[str]) -> None:
    print("")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _write_visual_comparison(
    out_dir: Path,
    acdlr_dir: Path,
    deepmoon_dir: Path,
    acdlr: dict,
    deepmoon: dict,
) -> Path:
    acdlr_visuals = sorted((acdlr_dir / "visuals").glob("*_matches.png"))
    if not acdlr_visuals:
        raise SystemExit(f"No ACDLR visual overlays found in {acdlr_dir / 'visuals'}")

    acdlr_img = cv2.imread(str(acdlr_visuals[0]), cv2.IMREAD_COLOR)
    deepmoon_img = cv2.imread(str(deepmoon_dir / "deepmoon_sample_prediction.png"), cv2.IMREAD_COLOR)
    if acdlr_img is None or deepmoon_img is None:
        raise SystemExit("Could not load comparison images")

    panels = [
        (
            "MEU METODO: ACDLR",
            f"P={acdlr['precision']:.3f} R={acdlr['recall']:.3f} F1={acdlr['f1']:.3f}",
            acdlr_img,
        ),
        (
            "METODO CNN: DeepMoon",
            f"P={deepmoon['precision']:.3f} R={deepmoon['recall']:.3f} F1={deepmoon['f1']:.3f}",
            deepmoon_img,
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
    note_h = 48
    note = np.zeros((note_h, canvas.shape[1], 3), dtype=np.uint8)
    _put_text(
        note,
        "Note: DeepMoon sample is not the same YOLO visual dataset; it is the local runnable CNN reference artifact.",
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
    deepmoon: dict,
    acdlr_summary_path: Path,
    deepmoon_summary_path: Path,
    visual_path: Path,
    args: argparse.Namespace,
) -> Path:
    report_path = out_dir / "comparison_report.md"
    lines = [
        "# ACDLR x DeepMoon visual comparison",
        "",
        "## How this was run",
        "",
        f"- YOLO dataset: `{args.dataset_dir}`",
        f"- Split: `{args.split}`",
        f"- Meu metodo ACDLR: processou {acdlr['images_processed']} imagens anotadas do dataset.",
        "- Meu metodo ACDLR: processamento classico de imagem, sem IA e sem treinamento.",
        "- DeepMoon: metodo CNN do repositorio oficial, executado no sample local disponivel.",
        "",
        "## Open this visual",
        "",
        f"- `{visual_path}`",
        "",
        "## Meu metodo: ACDLR",
        "",
        f"- Summary JSON: `{acdlr_summary_path}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Detections | {acdlr['detections']} |",
        f"| Ground truth | {acdlr['ground_truth']} |",
        f"| True positives | {acdlr['true_positive']} |",
        f"| False positives | {acdlr['false_positive']} |",
        f"| False negatives | {acdlr['false_negative']} |",
        f"| Precision | {acdlr['precision']:.4f} |",
        f"| Recall | {acdlr['recall']:.4f} |",
        f"| F1 | {acdlr['f1']:.4f} |",
        "",
        "## Metodo CNN: DeepMoon",
        "",
        f"- Summary JSON: `{deepmoon_summary_path}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Detections | {deepmoon['detections']:.0f} |",
        f"| Ground truth | {deepmoon['ground_truth']:.0f} |",
        f"| Matches | {deepmoon['matches']:.0f} |",
        f"| Precision | {deepmoon['precision']:.4f} |",
        f"| Recall | {deepmoon['recall']:.4f} |",
        f"| F1 | {deepmoon['f1']:.4f} |",
        "",
        "## Important limitation",
        "",
        "DeepMoon cannot be honestly run on this visual YOLO dataset as-is: the original project expects lunar DEM inputs and legacy TensorFlow/Keras dependencies. This script still runs the local DeepMoon sample so you can compare your classical method against the CNN reference artifact that is executable in this repository.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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

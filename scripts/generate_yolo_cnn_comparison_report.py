from __future__ import annotations

"""
Generate a compact report comparing the YOLO-dataset ACDLR benchmark with the
available DeepMoon CNN reference metrics.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acdlr-summary",
        default="artifacts/yolo_benchmark_deepmoon_tolerance25/acdlr_yolo_summary.json",
    )
    parser.add_argument(
        "--strict-summary",
        default="artifacts/yolo_benchmark_final25/acdlr_yolo_summary.json",
    )
    parser.add_argument(
        "--deepmoon-summary",
        default="artifacts/deepmoon_validation/deepmoon_sample_metrics.json",
    )
    parser.add_argument("--out-dir", default="artifacts/yolo_cnn_comparison")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    acdlr_path = root / args.acdlr_summary
    strict_path = root / args.strict_summary
    deepmoon_path = root / args.deepmoon_summary
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    acdlr = json.loads(acdlr_path.read_text(encoding="utf-8"))
    strict = (
        json.loads(strict_path.read_text(encoding="utf-8"))
        if strict_path.exists()
        else None
    )
    deepmoon = (
        json.loads(deepmoon_path.read_text(encoding="utf-8"))
        if deepmoon_path.exists()
        else None
    )

    report_path = out_dir / "yolo_cnn_comparison_report.md"
    report_path.write_text(
        _render(acdlr, strict, deepmoon, acdlr_path, strict_path, deepmoon_path),
        encoding="utf-8",
    )
    print(report_path)


def _render(
    acdlr: dict,
    strict: dict | None,
    deepmoon: dict | None,
    acdlr_path: Path,
    strict_path: Path,
    deepmoon_path: Path,
) -> str:
    lines = [
        "# ACDLR x CNN comparison on the new annotated dataset",
        "",
        "## What was run",
        "",
        "- ACDLR was evaluated on the new `LU3M6TGT_yolo_format` validation split.",
        "- YOLO boxes were converted to crater circles using the box center and mean box radius.",
        "- The ACDLR method remains classical image processing: no training and no neural inference.",
        "- DeepMoon is included as the CNN reference. Its local runnable sample is a saved CNN prediction mask, not this visual YOLO dataset.",
        "",
        "## ACDLR on annotated YOLO validation subset",
        "",
        f"- DeepMoon-like matching summary: `{acdlr_path}`",
        f"- Images processed: {acdlr['images_processed']}",
        "- Matching tolerance: center <= 1.34 x radius, radius error <= 1.0 x radius.",
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
        f"| Mean center error ratio | {acdlr['mean_center_error_ratio']:.4f} |",
        f"| Mean radius error ratio | {acdlr['mean_radius_error_ratio']:.4f} |",
        "",
    ]

    if strict is not None:
        lines.extend(
            [
                "## ACDLR stricter YOLO-box matching",
                "",
                f"- Strict summary: `{strict_path}`",
                "- Matching tolerance: center <= 0.65 x radius, radius error <= 0.65 x radius.",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Precision | {strict['precision']:.4f} |",
                f"| Recall | {strict['recall']:.4f} |",
                f"| F1 | {strict['f1']:.4f} |",
                f"| True positives | {strict['true_positive']} |",
                f"| False positives | {strict['false_positive']} |",
                f"| False negatives | {strict['false_negative']} |",
                "",
            ]
        )

    lines.extend(
        [
        "## CNN reference",
        "",
        ]
    )

    if deepmoon is None:
        lines.extend(
            [
                f"- DeepMoon local sample metrics were not found at `{deepmoon_path}`.",
                "- Run `python scripts/validate_deepmoon_sample.py` to regenerate them.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Source summary: `{deepmoon_path}`",
                "- Local executable DeepMoon step: template matching over the bundled saved CNN prediction.",
                "",
                "| Metric | Local DeepMoon sample |",
                "|---|---:|",
                f"| Detections | {deepmoon['detections']:.0f} |",
                f"| Ground truth | {deepmoon['ground_truth']:.0f} |",
                f"| Matches | {deepmoon['matches']:.0f} |",
                f"| Precision | {deepmoon['precision']:.4f} |",
                f"| Recall | {deepmoon['recall']:.4f} |",
                f"| F1 | {deepmoon['f1']:.4f} |",
                f"| Mean radius error ratio | {deepmoon['mean_radius_error_ratio']:.4f} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "This is the fairest comparison currently possible in this repository without training or importing a neural model for the new visual dataset.",
            "The ACDLR numbers are real metrics on the new annotated dataset. The DeepMoon numbers are a CNN reference sample because DeepMoon expects lunar DEM inputs and a legacy TensorFlow/Keras runtime, while this dataset is visual imagery with YOLO boxes.",
            "",
            "For an exact same-dataset CNN comparison, the missing artifact is a trained CNN model for `LU3M6TGT_yolo_format` or a DeepMoon-compatible DEM version of the same samples.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

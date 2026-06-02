from __future__ import annotations

"""
Validate the runnable DeepMoon post-CNN extraction step on its bundled sample.

The original DeepMoon training/inference stack depends on legacy TensorFlow and
Cartopy versions. This script validates the part that can run in a modern
environment: template matching over a saved CNN prediction target from
tests/sample_template_match.hdf5.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deepmoon-dir",
        default="external/DeepMoon",
        help="Path to the cloned silburt/DeepMoon repository",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/deepmoon_validation",
        help="Directory where metrics and screenshots will be written",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    deepmoon_dir = (repo_root / args.deepmoon_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(deepmoon_dir))
    import utils.template_match_target as tmt

    sample_path = deepmoon_dir / "tests" / "sample_template_match.hdf5"
    with h5py.File(sample_path, "r") as sample:
        csv_data = sample["csv"][...].T
        pred = sample["pred"][...]

    truth = np.array((csv_data[3], csv_data[4], csv_data[5] / 2.0)).T
    detections = tmt.template_match_t(pred.copy(), minrad=8, maxrad=11)
    metrics = tmt.template_match_t2c(pred.copy(), truth.copy(), minrad=8, maxrad=11)
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
    metric_dict = dict(zip(metric_names, [float(x) for x in metrics]))
    metric_dict["precision"] = (
        metric_dict["matches"] / metric_dict["detections"]
        if metric_dict["detections"]
        else 0.0
    )
    metric_dict["recall"] = (
        metric_dict["matches"] / metric_dict["ground_truth"]
        if metric_dict["ground_truth"]
        else 0.0
    )
    metric_dict["f1"] = (
        2.0 * metric_dict["precision"] * metric_dict["recall"]
        / (metric_dict["precision"] + metric_dict["recall"])
        if (metric_dict["precision"] + metric_dict["recall"])
        else 0.0
    )

    _write_json(out_dir / "deepmoon_sample_metrics.json", metric_dict)
    _write_csv(out_dir / "deepmoon_sample_detections.csv", detections)
    _write_visualisation(out_dir / "deepmoon_sample_prediction.png", pred, truth, detections)

    print("DeepMoon sample validation")
    print(f"sample: {sample_path}")
    for name in [
        "matches",
        "ground_truth",
        "detections",
        "precision",
        "recall",
        "f1",
        "mean_x_error_ratio",
        "mean_y_error_ratio",
        "mean_radius_error_ratio",
    ]:
        print(f"{name}: {metric_dict[name]:.6f}")
    print(f"screenshot: {out_dir / 'deepmoon_sample_prediction.png'}")


def _write_json(path: Path, data: dict[str, float]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, detections: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "radius_px"])
        for row in np.asarray(detections):
            writer.writerow([float(row[0]), float(row[1]), float(row[2])])


def _write_visualisation(
    path: Path,
    pred: np.ndarray,
    truth: np.ndarray,
    detections: np.ndarray,
) -> None:
    pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR)

    for x, y, r in truth:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(r)), (80, 220, 255), 1)
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (80, 220, 255), -1)

    for x, y, r in detections:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(r)), (80, 255, 100), 1)
        cv2.drawMarker(
            vis,
            (int(round(x)), int(round(y))),
            (80, 255, 100),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )

    cv2.putText(
        vis,
        "DeepMoon sample: yellow=ground truth, green=template match",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), vis)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
evaluation.py
-------------
Metricas para comparar as crateras detectadas pelo ACDLR com anotacoes
manuais, mantendo o projeto no campo de visao computacional classica.
As metricas reportam precision, recall, F1 e erros fracionarios adaptados para
coordenadas em pixels.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GroundTruthCrater:
    cx: float
    cy: float
    radius_px: float


@dataclass(frozen=True)
class DetectionMatch:
    detection_index: int
    truth_index: int
    center_error_px: float
    center_error_ratio: float
    radius_error_px: float
    radius_error_ratio: float


@dataclass(frozen=True)
class EvaluationResult:
    detections: int
    ground_truth: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_center_error_px: float
    mean_center_error_ratio: float
    mean_radius_error_px: float
    mean_radius_error_ratio: float
    matches: tuple[DetectionMatch, ...]

    def to_dict(self) -> dict[str, float | int]:
        return {
            "detections": self.detections,
            "ground_truth": self.ground_truth,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_center_error_px": self.mean_center_error_px,
            "mean_center_error_ratio": self.mean_center_error_ratio,
            "mean_radius_error_px": self.mean_radius_error_px,
            "mean_radius_error_ratio": self.mean_radius_error_ratio,
        }


def load_annotations(path: str | Path) -> list[GroundTruthCrater]:
    """Load crater annotations from CSV or JSON.

    Supported CSV columns:
    - cx, cy, radius_px
    - x, y, r

    Supported JSON:
    - a list of objects
    - an object with a "craters" list
    """
    annotation_path = Path(path)
    suffix = annotation_path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(annotation_path)
    if suffix == ".json":
        return _load_json(annotation_path)

    raise ValueError(f"Unsupported annotation file: {annotation_path}")


def evaluate_circles(
    circles: np.ndarray,
    ground_truth: list[GroundTruthCrater],
    center_tolerance_ratio: float = 0.50,
    radius_tolerance_ratio: float = 0.50,
) -> EvaluationResult:
    """Match detected circles against manual crater annotations.

    A match is accepted when center and radius errors are below tolerances
    relative to the annotated radius. Matching is greedy by lowest normalized
    error, which is simple, deterministic and enough for a compact benchmark.
    """
    detections = _normalise_circles(circles)
    candidates: list[tuple[float, int, int, float, float, float, float]] = []

    for det_idx, (dx, dy, dr) in enumerate(detections):
        for truth_idx, truth in enumerate(ground_truth):
            truth_radius = max(float(truth.radius_px), 1.0)
            center_error = float(math.hypot(dx - truth.cx, dy - truth.cy))
            radius_error = float(abs(dr - truth.radius_px))
            center_ratio = center_error / truth_radius
            radius_ratio = radius_error / truth_radius

            if (
                center_ratio <= center_tolerance_ratio
                and radius_ratio <= radius_tolerance_ratio
            ):
                combined_error = center_ratio + radius_ratio
                candidates.append(
                    (
                        combined_error,
                        det_idx,
                        truth_idx,
                        center_error,
                        center_ratio,
                        radius_error,
                        radius_ratio,
                    )
                )

    used_detections: set[int] = set()
    used_truth: set[int] = set()
    matches: list[DetectionMatch] = []

    for (
        _,
        det_idx,
        truth_idx,
        center_error,
        center_ratio,
        radius_error,
        radius_ratio,
    ) in sorted(candidates):
        if det_idx in used_detections or truth_idx in used_truth:
            continue
        used_detections.add(det_idx)
        used_truth.add(truth_idx)
        matches.append(
            DetectionMatch(
                detection_index=det_idx,
                truth_index=truth_idx,
                center_error_px=center_error,
                center_error_ratio=center_ratio,
                radius_error_px=radius_error,
                radius_error_ratio=radius_ratio,
            )
        )

    tp = len(matches)
    fp = max(len(detections) - tp, 0)
    fn = max(len(ground_truth) - tp, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    center_errors = [match.center_error_px for match in matches]
    center_ratios = [match.center_error_ratio for match in matches]
    radius_errors = [match.radius_error_px for match in matches]
    radius_ratios = [match.radius_error_ratio for match in matches]

    return EvaluationResult(
        detections=len(detections),
        ground_truth=len(ground_truth),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_center_error_px=float(np.mean(center_errors)) if center_errors else 0.0,
        mean_center_error_ratio=float(np.mean(center_ratios)) if center_ratios else 0.0,
        mean_radius_error_px=float(np.mean(radius_errors)) if radius_errors else 0.0,
        mean_radius_error_ratio=float(np.mean(radius_ratios)) if radius_ratios else 0.0,
        matches=tuple(matches),
    )


def aggregate(results: list[EvaluationResult]) -> EvaluationResult:
    if not results:
        return EvaluationResult(
            detections=0,
            ground_truth=0,
            true_positive=0,
            false_positive=0,
            false_negative=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mean_center_error_px=0.0,
            mean_center_error_ratio=0.0,
            mean_radius_error_px=0.0,
            mean_radius_error_ratio=0.0,
            matches=(),
        )

    detections = sum(result.detections for result in results)
    ground_truth = sum(result.ground_truth for result in results)
    tp = sum(result.true_positive for result in results)
    fp = sum(result.false_positive for result in results)
    fn = sum(result.false_negative for result in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    all_matches = tuple(match for result in results for match in result.matches)
    center_errors = [match.center_error_px for match in all_matches]
    center_ratios = [match.center_error_ratio for match in all_matches]
    radius_errors = [match.radius_error_px for match in all_matches]
    radius_ratios = [match.radius_error_ratio for match in all_matches]

    return EvaluationResult(
        detections=detections,
        ground_truth=ground_truth,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_center_error_px=float(np.mean(center_errors)) if center_errors else 0.0,
        mean_center_error_ratio=float(np.mean(center_ratios)) if center_ratios else 0.0,
        mean_radius_error_px=float(np.mean(radius_errors)) if radius_errors else 0.0,
        mean_radius_error_ratio=float(np.mean(radius_ratios)) if radius_ratios else 0.0,
        matches=all_matches,
    )


def _load_csv(path: Path) -> list[GroundTruthCrater]:
    annotations: list[GroundTruthCrater] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotations.append(_row_to_crater(row))
    return annotations


def _load_json(path: Path) -> list[GroundTruthCrater]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("craters", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"JSON annotation must be a list or contain 'craters': {path}")
    return [_row_to_crater(row) for row in rows]


def _row_to_crater(row: dict) -> GroundTruthCrater:
    def pick(*names: str) -> float:
        for name in names:
            if name in row and row[name] not in (None, ""):
                return float(row[name])
        raise ValueError(f"Missing any of {names} in annotation row: {row}")

    return GroundTruthCrater(
        cx=pick("cx", "x", "center_x"),
        cy=pick("cy", "y", "center_y"),
        radius_px=pick("radius_px", "r", "radius"),
    )


def _normalise_circles(circles: np.ndarray) -> list[tuple[float, float, float]]:
    if circles.size == 0:
        return []

    arr = np.asarray(circles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("circles must be an Nx3 array with x, y, radius")
    return [(float(x), float(y), float(r)) for x, y, r in arr[:, :3]]

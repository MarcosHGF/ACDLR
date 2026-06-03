from __future__ import annotations

import numpy as np

from core.evaluation import GroundTruthCrater, aggregate, evaluate_circles


def test_evaluate_circles_counts_true_positive_false_positive_and_false_negative():
    truth = [
        GroundTruthCrater(cx=50.0, cy=50.0, radius_px=10.0),
        GroundTruthCrater(cx=100.0, cy=100.0, radius_px=20.0),
    ]
    detections = np.asarray(
        [
            [52.0, 49.0, 11.0],
            [160.0, 160.0, 12.0],
        ],
        dtype=float,
    )

    result = evaluate_circles(
        detections,
        truth,
        center_tolerance_ratio=0.5,
        radius_tolerance_ratio=0.5,
    )

    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5


def test_aggregate_uses_global_counts_not_mean_of_image_scores():
    first = evaluate_circles(
        np.asarray([[10.0, 10.0, 5.0]], dtype=float),
        [GroundTruthCrater(cx=10.0, cy=10.0, radius_px=5.0)],
    )
    second = evaluate_circles(
        np.empty((0, 3), dtype=float),
        [GroundTruthCrater(cx=30.0, cy=30.0, radius_px=5.0)],
    )

    summary = aggregate([first, second])

    assert summary.true_positive == 1
    assert summary.false_positive == 0
    assert summary.false_negative == 1
    assert summary.precision == 1.0
    assert summary.recall == 0.5
    assert round(summary.f1, 6) == round(2.0 / 3.0, 6)

from __future__ import annotations

"""
Generate a reproducible ACDLR x DeepMoon comparison report from local evidence.
"""

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    deepmoon_metrics_path = root / "artifacts" / "deepmoon_validation" / "deepmoon_sample_metrics.json"
    report_path = root / "artifacts" / "method_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    deepmoon_metrics = json.loads(deepmoon_metrics_path.read_text(encoding="utf-8"))
    screenshot_path = root / "artifacts" / "deepmoon_validation" / "deepmoon_sample_prediction.png"

    report = f"""# ACDLR x DeepMoon Validation Report

## Scope

This report compares the local ACDLR implementation with the runnable parts of
the official `silburt/DeepMoon` repository cloned under `external/DeepMoon`.

DeepMoon's full CNN training/inference stack could not be executed in the
available runtime because the project depends on legacy packages such as
`Cartopy==0.14.2`, `Keras==1.2.2` and `TensorFlow==0.10.0rc0`, while the
available Python runtime is Python 3.12. The runnable post-CNN crater extraction
step was validated using DeepMoon's bundled `sample_template_match.hdf5`.

## DeepMoon Executed Validation

- Source sample: `external/DeepMoon/tests/sample_template_match.hdf5`
- Method executed: `utils.template_match_target.template_match_t` and
  `template_match_t2c`
- Screenshot: `{screenshot_path}`

| Metric | Value |
|---|---:|
| Matches | {deepmoon_metrics["matches"]:.0f} |
| Ground truth craters | {deepmoon_metrics["ground_truth"]:.0f} |
| Detections | {deepmoon_metrics["detections"]:.0f} |
| Precision | {deepmoon_metrics["precision"]:.3f} |
| Recall | {deepmoon_metrics["recall"]:.3f} |
| F1 | {deepmoon_metrics["f1"]:.3f} |
| Mean x error ratio | {deepmoon_metrics["mean_x_error_ratio"]:.3f} |
| Mean y error ratio | {deepmoon_metrics["mean_y_error_ratio"]:.3f} |
| Mean radius error ratio | {deepmoon_metrics["mean_radius_error_ratio"]:.3f} |

## DeepMoon Official Test Suite Status

The official DeepMoon pytest suite was attempted.

| Test command | Result |
|---|---|
| `pytest tests -q` | Collection failed: missing `cartopy` and `keras` |
| `pytest tests/test_utils.py -q` | 6 passed, 5 failed due legacy `setup()` not initializing attributes under modern pytest |
| `pytest tests/test_get_unique_craters.py -q` | 1 failed due legacy `setup()` not initializing attributes under modern pytest |

## ACDLR Validation Status

| Validation | Result |
|---|---|
| Python compilation of all ACDLR modules and scripts | Passed |
| Synthetic evaluator sanity check | Passed: TP=1, FP=1, FN=1, precision=0.5, recall=0.5, F1=0.5 |
| Quantitative crater benchmark on LROC tiles | Blocked: no local `data/annotations` ground truth set is present |
| Parameter tuning via `tune_classical.py` | Ready, blocked until annotated tiles exist |

## Method Comparison

| Dimension | ACDLR | DeepMoon |
|---|---|---|
| Core approach | Classical computer vision | CNN based on U-Net plus template matching |
| Required training | None | Required for full method |
| Runnable in current environment | Yes for ACDLR code; benchmark awaits annotations | Only post-CNN template matching sample is runnable |
| Inputs | Visual lunar images or tiles | DEM crops and crater catalogues |
| Outputs | Detected craters, risk grid, landing point | Crater detections/catalogue |
| Interpretability | High: explicit filters, thresholds and risk components | Lower: CNN mask prediction plus post-processing |
| Validation metric alignment | Precision, recall, F1, center/radius error ratios | Precision, recall, F1, longitude/latitude/radius error ratios |

## Conclusion

The DeepMoon post-CNN extraction method ran successfully on the bundled sample
prediction and produced measurable precision/recall/F1. The full DeepMoon method
could not be executed without a legacy TensorFlow/Keras/Cartopy environment and
Zenodo data/model artifacts.

For a fair final comparison, the next required project artifact is a local
annotated ACDLR tile set. Once present, `benchmark_classical.py` and
`tune_classical.py` can produce ACDLR metrics in the same evaluation family as
DeepMoon.
"""

    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

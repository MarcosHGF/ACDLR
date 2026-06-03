# Experiment Configs

This folder stores the command-level configurations used in the paper-style
experiments. The current scripts receive arguments directly from the command
line; these files document the exact parameter sets used for reproducibility.

## Files

| File | Purpose |
|---|---|
| `acdlr_default.yaml` | Default ACDLR detector settings used by the app and benchmarks |
| `benchmark_smoke.yaml` | Small ACDLR x CNN smoke-test protocol |
| `benchmark_valid25.yaml` | Larger validation subset protocol |
| `cnn_baseline_smoke.yaml` | CNN baseline training/inference settings for a quick run |

These YAML files are documentation configs. To run an experiment, copy the
values into the matching script command or use the ready commands in
`docs/COMO_RODAR.md`.

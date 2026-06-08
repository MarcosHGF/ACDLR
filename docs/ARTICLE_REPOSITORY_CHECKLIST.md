# Article Repository Checklist

Use this checklist before publishing the repository with a paper.

## Repository Metadata

- [x] README with abstract-like summary
- [x] License
- [x] Citation file
- [x] Requirements file
- [x] Conda environment file
- [x] Documentation folder
- [x] Config folder

## Method Documentation

- [x] ACDLR pipeline explained
- [x] Risk score explained
- [x] Evaluation metrics explained
- [x] Ellipse R-CNN AI baseline separated from ACDLR
- [x] Limitations documented

## Reproducibility

- [x] Quickstart commands
- [x] Benchmark commands
- [x] Smoke-test result recorded
- [x] Output artifact paths documented
- [ ] Full validation split result
- [x] Ellipse R-CNN pretrained-weight download instructions
- [ ] Exact hardware/runtime table

## Paper-Ready Results

- [x] Small smoke-test table
- [x] Side-by-side visualization path
- [ ] Full validation table
- [ ] Ablation study
- [ ] Parameter sensitivity study
- [ ] Runtime comparison
- [ ] Failure-case analysis

## Recommended Next Experiments

1. Run ACDLR on the full `valid` split.
2. Download/verify `wdoppenberg/crater-rcnn` weights.
3. Compare full validation metrics for ACDLR and Ellipse R-CNN.
4. Add runtime per image for both methods.
5. Add an ablation removing individual ACDLR validators.
6. Select visual examples for the paper.

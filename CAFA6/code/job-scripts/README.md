<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [CAFA6](../../README.md) › [Code](../README.md) › **job-scripts**</sub>

# `job-scripts/` — SLURM launchers

Original cluster scripts. The only edit for this portfolio: the absolute project path is replaced by `/path/to/cafa6` (in `#SBATCH --output/--error`, `cd` and `REPO_ROOT`). Partitions, memory and time limits are unchanged; create the `logs/` sub-directories before submitting.

| Script | Runs | Pipeline |
|---|---|---|
| `train.sh` | ProtT5 classifier for MF, then BP, then CC | [P1](../../pipelines/01_prott5_classifier/README.md) |
| `test.sh` | prediction with the three ProtT5 models + propagation | P1 |
| `train-test.sh` | `train.sh` followed by `test.sh` | P1 |
| `jepa-pretrain.sh` | span-masking JEPA pre-training (ProtT5 by default, 2,000 steps) | [P4](../../pipelines/04_jepa_encoder/README.md) |
| `jepa-go-train-predict.sh` · `jepa-go-predict.sh` | frozen-encoder GO head: train + predict / predict only | P4 |
| `label-jepa-full.sh` · `label-jepa-train.sh` · `label-jepa-predict.sh` | label-space JEPA: train + predict (ESM2-650M default) / train only (ProtT5 default) / predict only (default model dir `models/label-jepa-20260202`) | [P5](../../pipelines/05_label_space_jepa/README.md) |

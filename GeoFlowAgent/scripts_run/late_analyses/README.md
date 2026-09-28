<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [scripts_run](../README.md) › **late_analyses**</sub>

# 🧪 Late analyses (stage ⑤)

Stage-⑤ analysis scripts with their run logs. Each loads the saved feature store and checkpoints once and writes one findings JSON; the docstrings (Korean) state the question and the result that would overturn the current reading.

| Script | Question | Output (in [`results/geoacmg_working_findings/`](../../results/geoacmg_working_findings/)) |
|---|---|---|
| `frozen_space_ladder.py` | Is information in the frozen MedCPT space, or did the raw metrics simply fail to read an anisotropic space? Standardize/whiten/PCA/random projections fitted on train | `R8_frozen_space_ladder.json` |
| `ablation_arms.py` | Is the signal in MedCPT or in the training-free structured-state hash? | `R9_information_source_ablation.json` |
| `holdout_standardiser.py` | Does the standardized-cosine result survive when the standardizer is fitted on train or leave-one-gene-out? | `R2d_holdout_standardiser.json` |
| `newseed_readouts.py`, `newseed_readouts2.py` | Extend the per-pair readout margins to seeds 59/71 and 83/97 | `R2c_newseed_readout_stats.json` and the extended margin file |

The `.log` files are the run logs (machine paths shown as `$GEOWORK`).

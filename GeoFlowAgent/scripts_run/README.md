<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **scripts_run**</sub>

# 🏃 `scripts_run/` — GeoACMG runners and analyses (stages ④–⑤)

Scripts that produced the GeoACMG findings on the GPU container (comments are mostly Korean). Analyses read saved artifacts (feature store, checkpoints, per-run metrics) and write JSON into `artifacts/acmg/findings/`, collected here in `results/geoacmg_final/findings/` and `results/geoacmg_working_findings/`. The folder keeps its original name because the scripts call each other as `scripts_run/…`.

| Script | Produces / does | Finding |
|---|---|---|
| `geometry_batch.sh`, `geometry_worker.sh`, `geometry_aggregate.sh`, `snapshot_comparison.sh` | 5 energies × 3 seeds geometry sweep, batched per seed to share one ~29 GiB feature store; aggregation reuses finished runs only after hash checks | 15-run table |
| `r1_paired_seeds.sh`, `r1_paired_seeds2.sh` | Extra cosine/Euclidean seed pairs (59/71, then 83/97) | R1, R1b, R2f, R3c |
| `ablate_medcpt.sh`, `ablate_medcpt2.sh` | Retraining with the MedCPT view zeroed (seeds 17/29, then 43/59) | R11 |
| `train_flow.sh`, `train_flow_rootonly.sh`, `train_flow_rootonly_v2.sh` | State Flow training and the root-only dev evaluation protocol | R7 |
| `build_sealed_corpus.py` | Copy of the corpus without test rows, so later analyses cannot touch the test split | sealing |
| `ordering_mechanism.py`, `mechanism_decomposition.py`, `metric_profile.py` | Does a learned energy fix the raw-distance ordering failure; trunk vs energy-head contribution; which outputs a geometry changes | G01–G02 |
| `capacity_matched.py`, `capacity_contrast.py`, `capacity_ordering.py` (+ `run_capacity_*.sh`) | Parameter-matched capacity contrasts at planning and representation level | capacity control |
| `generalization_and_strata.py` (+ `run_generalization.sh`) | Train–dev generalisation gap and stratified checks in one store load | G02 |
| `c4_delta_moderator.py`, `c5_within_model.py`, `c5_crossmodel_control.py` | Pre-registered C4 moderator regression and the C5 within/cross-model control that led to C5's retraction | C4, C5 |
| `geometry_report.py`, `geometry_table.py` | Read-only summaries of the 15-run sweep | — |
| [`export_stage/`](export_stage/README.md) | Finishing scripts of the main GeoACMG analysis | R1–R4, R7 |
| [`late_analyses/`](late_analyses/README.md) | Readout ladder, information-source arms, holdout standardizer, margin extension | R2c–R2f, R8, R9 |

**Paths.** `$GEOACMG_WORK` (project checkout on the GPU container), `$GEOWORK` (scratch/log directory) and `$HF_CACHE_DIR` stand for the machine paths; Python scripts expand `$GEOWORK` at runtime. Their inputs (feature store, checkpoints, margin arrays) are kept with the large artifacts of the local research archive.

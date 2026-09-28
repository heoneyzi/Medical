<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › **scripts**</sub>

# 🧰 scripts

Entry points around the [`vcc_baselines`](../src/vcc_baselines/README.md) package. Run everything from the project root (`01_Medical/VCC_2026`). Long runners are resumable and print nested progress.

| Group | Script | What it does | Used in |
|---|---|---|---|
| Smoke test | [`quickstart.sh`](quickstart.sh) | Synthetic data → ladder → coverage → `vcc prep` dry-run and real `.vcc` → `vcc sample` | [E0](../experiments/E0_synthetic_smoke/README.md) |
| Data | [`download_gwps.sh`](download_gwps.sh) | Notes and steps for fetching the Replogle GWPS files | challenge runs |
| | [`download_shadow_data.sh`](download_shadow_data.sh) | Downloads and SHA-256-checks Tian–Kampmann (scPerturb), Jiang24 (PerturBench) and GSE270828 (GEO) | E1–E3 |
| | [`export_gse270828_rds.R`](export_gse270828_rds.R), [`build_gse270828_h5ad.py`](build_gse270828_h5ad.py) | Converts the double-gzipped GSE270828 Seurat object into a sparse raw-count H5AD without dense expansion | E2, E3 |
| | [`build_gene_map.py`](build_gene_map.py) | Explicit symbol → Ensembl map from HGNC; ambiguous aliases fail closed | E3 |
| Shadow benchmarks | [`run_jiang24_loco.sh`](run_jiang24_loco.sh) | Compacts Jiang24 to IFNG raw counts, builds the BxPC3 leave-one-out split, runs the local benchmark | [E1](../experiments/E1_jiang24_bxpc3_proxy/README.md) |
| | [`run_frozen_contexts.sh`](run_frozen_contexts.sh), [`state_safetensors_embed.py`](state_safetensors_embed.py) | Embeds every context's controls with STATE-SE and STACK, pools them and scores nearest / weighted transfer; the Python wrapper strictly loads the official SE-100M `model.safetensors` | E1 |
| | [`run_shadow_real.sh`](run_shadow_real.sh) | Tian–Kampmann iPSC → day-7-neuron state-shift sanity check | — |
| | [`run_two_dataset_full_matrix.sh`](run_two_dataset_full_matrix.sh), [`compare_six_metric_matrix.py`](compare_six_metric_matrix.py) | Non-strict 11-row method × representation matrix on Jiang24 and GSE270828, scored with exact `cell-eval2`; merges results without cross-dataset averaging | [E2](../experiments/E2_six_metric_robustness/README.md) |
| | [`run_two_dataset_zero_shot.sh`](run_two_dataset_zero_shot.sh), [`audit_zero_shot_models.py`](audit_zero_shot_models.py), [`compare_zero_shot.py`](compare_zero_shot.py) | Response-sealed zero-shot bundles, frozen-checkpoint eligibility audit, merged strict table | strict protocol |
| Strict Replogle-only track | [`build_replogle_only_experiment.py`](build_replogle_only_experiment.py) | Builds the Replogle-only configs and effect libraries for both benchmarks | [E3](../experiments/E3_replogle_only_strict/README.md) |
| | [`setup_external_frozen_models.sh`](setup_external_frozen_models.sh) | Installs one model environment or checkpoint at a time, with integrity checks and resumable downloads | E3 |
| | [`run_two_dataset_external_frozen_matrix.sh`](run_two_dataset_external_frozen_matrix.sh), [`run_external_frozen_embeddings.py`](run_external_frozen_embeddings.py) | Preflight → embeddings → prediction → exact six-metric scoring for UCE, TranscriptFormer, scGPT, scFoundation, scPRINT-2; `--protocol=shadow-existing --acknowledge-non-strict` gives the non-strict variant | E2 (non-strict), E3 |
| | [`audit_external_experiment.py`](audit_external_experiment.py) | Fail-closed audit of response sources and model roles | E3 |
| | [`score_external_coverage_strata.py`](score_external_coverage_strata.py), [`compare_external_coverage_matrix.py`](compare_external_coverage_matrix.py), [`compare_external_frozen_matrix.py`](compare_external_frozen_matrix.py) | Scores `ALL_TARGETS` / `DUAL_SOURCE` / `K562_ONLY` / `MISSING` strata separately and merges without averaging across strata or datasets | E3 |
| | [`adapters/scgpt_embed.py`](adapters/scgpt_embed.py) | Thin wrapper around scGPT's official cell-embedding API | E3 |

Model checkpoints, environments and data directories (`models/`, `envs/`, `external/`, `data*/`) are created by these scripts and are not part of the repository. Weight licences are listed in [`../docs/licenses.md`](../docs/licenses.md).

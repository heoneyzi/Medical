<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › **docs**</sub>

# 📚 docs

## Write-ups (English condensations of the Korean originals)

| Document | What it covers | Korean original |
|---|---|---|
| [`challenge_overview.md`](challenge_overview.md) | The 2026 task, 2025 → 2026 changes, evaluation and leaderboard logic, data strategy, 2025 winners' lessons, baseline ladder, recommended architecture (22 Aug 2026) | [`ko/01_…`](ko/01_challenge_explainer_2026-08-22_ko.md) |
| [`starting_plan.md`](starting_plan.md) | Metric-first plan: the six metrics in plain words, sanity experiments, shadow benchmark, statistical ladder, statistical vs learned comparison, generator study, decision gates | [`ko/02_…`](ko/02_start_plan_2026-08-22_ko.md) |
| [`experiments_and_metrics.md`](experiments_and_metrics.md) | Design, conditions and results of the two core experiments and the τ / seed / cell-count robustness extension, with what each metric really measures | [`ko/03_…`](ko/03_experiments_and_metrics_ko.md) |

## Technical protocols (from the code base)

| Document | What it covers |
|---|---|
| [`USAGE.md`](USAGE.md) | Original code README: install, offline smoke test, real challenge run, methods and knobs, troubleshooting |
| [`SHADOW_VCC.md`](SHADOW_VCC.md) | Jiang24 IFNG → BxPC3 leave-one-cell-line-out benchmark: split, hashes, leakage barriers, frozen STATE / STACK results ([E1](../experiments/E1_jiang24_bxpc3_proxy/README.md)) |
| [`STRICT_ZERO_SHOT.md`](STRICT_ZERO_SHOT.md) | Response-sealed two-dataset protocol (`blind/` vs `sealed/`), frozen-checkpoint eligibility rules |
| [`TWO_DATASET_SIX_METRIC.md`](TWO_DATASET_SIX_METRIC.md) | Exact `cell-eval2` `vcc2026` members, local anchors, the 11-row non-strict matrix ([E2](../experiments/E2_six_metric_robustness/README.md)) |
| [`STATE_STACK.md`](STATE_STACK.md) | Using STATE and STACK as frozen predictors or context encoders; the one-hot perturbation limitation and ESM-2 fixes |
| [`EXTERNAL_FROZEN_MODELS.md`](EXTERNAL_FROZEN_MODELS.md), [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) | External encoder experiment (UCE, TranscriptFormer, scGPT, scFoundation, scPRINT-2): protocols, setup, outputs, data audit, internal CV ([E3](../experiments/E3_replogle_only_strict/README.md)) |
| [`leakage_audit.md`](leakage_audit.md), [`licenses.md`](licenses.md), [`model_provenance.csv`](model_provenance.csv) | Leakage review checklist, code vs weight licences, checkpoint provenance (hashes, pretraining overlap: unknown) |
| [`model_smoke_tests/`](model_smoke_tests/README.md) | Real-input smoke records for UCE-4L and TranscriptFormer |

<sub>Edits for publication: absolute server paths in commands were replaced with the project root or `<repo>/`; the original README became `USAGE.md` with links re-rooted. Korean originals: see [`ko/README.md`](ko/README.md).</sub>

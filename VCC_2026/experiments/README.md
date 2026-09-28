<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › **experiments**</sub>

# 🔬 experiments

| | Experiment | Question | Headline |
|---|---|---|---|
| E0 | [Synthetic smoke test](E0_synthetic_smoke/README.md) | Does the pipeline run end-to-end and pack a valid `.vcc`? | Discrimination rank 0.4964 → 0.1141 (no-effect → direct transfer, synthetic) |
| E1 | [Jiang24 BxPC3 public proxy](E1_jiang24_bxpc3_proxy/README.md) | Does context-matched transfer beat doing nothing in a hidden cell line; do STATE / STACK help? | **0.5794** public-proxy PDS (raw nearest) vs 0.5102 no-effect |
| E2 | [Six-metric scoring & robustness](E2_six_metric_robustness/README.md) | How do 7 representations rank under the exact 2026 scorer, and is it stable? | Jiang24 Overall all < 0; winners change with τ and seed; Jaccard decides |
| E3 | [Strict Replogle-only track](E3_replogle_only_strict/README.md) | With only Replogle responses allowed, do frozen encoders pick better sources? | Scale 0.25 selected; raw −0.145 beats all frozen encoders on dual-source targets |

Each folder has a README (question, setup, results, takeaway, files) and a `results/` folder with small tables. Where the original run directories were too large to include, the tables were transcribed from the documented results and say so.

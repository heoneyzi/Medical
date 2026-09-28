<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [results](../README.md) › **geoacmg_latest_checkpoints**</sub>

# 📦 Per-run GeoACMG development metrics

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

GeoACMG 각 학습 run의 개발(dev) 분할 지표입니다(과제별 값 포함). 과제 ID가 `유전자:uuid` 형식이라 유전자 단위 신뢰구간을 이 파일들만으로 다시 계산할 수 있습니다.

</details>

One folder per run: `<group>/<energy>/seed-<n>/value_metrics.json` (metrics on the 584-task development split, including per-task `per_task_policy_accuracy`, `per_task_joint_accuracy` and `per_task_regret_at_1`) and a small `training_progress.json`. The model weights (`value_geometry.pt`, `training_resume.pt`) are listed with hashes in [`provenance/checkpoint_catalog.csv`](../../provenance/checkpoint_catalog.csv).

| Group | Runs | Used for |
|---|---|---|
| `geometry_comparison/` | cosine, Euclidean, Poincaré, directed quasimetric, pair MLP × seeds 17/29/43 (+ `comparison.json` with paired task-macro contrasts) | 15-run table, R2–R4 |
| `geometry_r1_pairs/` | cosine/Euclidean × seeds 59/71 | R1, R1b |
| `geometry_r1_pairs2/` | cosine/Euclidean × seeds 83/97 | 7-seed R2f/R3c |
| `capacity_matched/` | Euclidean at hidden width 147 and pair MLP at width 111 × seeds 17/29/43 | parameter-matched capacity contrasts |
| `geometry_ablate_medcpt/` | MedCPT-zeroed cosine/Euclidean × seeds 17/29 (the cosine/seed-29 metrics are stored in `comparison.json`) | R11 |
| `geometry_ablate_medcpt2/` | MedCPT-zeroed cosine/Euclidean × seeds 43/59 | 8-run ablation aggregate |
| `state_flow/` | Training progress of the State Flow checkpoint evaluated in R7 | R7 |

Task IDs have the form `GENE:uuid`, so gene-clustered intervals can be recomputed from these files alone — see [`verification/verify_portfolio.py`](../../verification/verify_portfolio.py).

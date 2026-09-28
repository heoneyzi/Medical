<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **configs**</sub>

# ⚙️ Experiment configs

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

실험별 설정 파일 15개와 각 파일이 쓰인 단계입니다. GPU 서버의 절대 경로는 `$GEOFLOW_*` 자리표시자로 바꿨습니다.

</details>


Each YAML fixes the data paths, encoder views (with pinned Hugging Face revisions), seeds and training settings of one experiment family. `project_root: ..` resolves relative paths from this folder.

| Config | Stage | Purpose |
|---|---|---|
| `smoke.yaml` | 01 | CPU smoke run of the stage-01 pipeline on the invented fixture in `data/raw/smoke/` |
| `search_smoke.yaml` | 02 | CPU smoke run of the search-distilled pipeline on `data/raw/search_smoke/` |
| `search_pilot_gpu_v1.yaml` | 02 | 210-task GPU procedural pilot (value geometry, DAgger, State Flow) |
| `search_pilot_frozen_qwen15.yaml`, `search_pilot_frozen_qwen7b.yaml` | 02 | Frozen-backbone baselines on the pilot (1.5B / 7B size control) |
| `search_hard_v2_hash.yaml` | 03 | hard-v2 development benchmark with the hash-embedding control |
| `search_hard_v2_frozen.yaml` | 03 | Real frozen encoders on the hard-v2 search graph |
| `search_hard_v2_hash_parameter_matched.yaml` | 03 | Hash control with parameters matched to the frozen model |
| `search_hard_v2_frozen_cosine_attribution.yaml` | 03 | Attribution/capacity control for the dev-selected geometry |
| `search_hard_v2_medcpt_only.yaml` | 03 | MedCPT-only follow-up selected after the input ablation |
| `search_hard_v2_final_frozen.yaml` | 03 | Pre-registered one-time final evaluation |
| `geoacmg.yaml` | 04–05 | GeoACMG run record (frozen MedCPT view; other views commented out) |
| `geoacmg_ablate_medcpt.yaml` | 05 | R11 ablation: `zero_views: [medcpt]` |
| `geomarrvel_mvp.yaml`, `search_geomarrvel.yaml` | next | MARRVEL-based protocol for live genomics tools: MARRVEL questions form the test split, and tasks enter after expert review at the quality gate |

## Machine paths → placeholders

The GPU configs point at run directories through these placeholders:

| Placeholder | Meaning |
|---|---|
| `$GEOFLOW_RUNS` | run root holding raw/processed data, caches, checkpoints and reports |
| `$GEOFLOW_RESULTS` | persistent mirror for results, progress files and the test-access ledger |
| `$GEOFLOW_PROJECT_ROOT` | repository checkout on the GPU machine |

Materialise a local copy before use, e.g. `GEOFLOW_RUNS=/data/geoflow envsubst < search_hard_v2_frozen.yaml > local.yaml`. The caches, checkpoints and processed search graphs these paths refer to are kept with the large artifacts of the local research archive.

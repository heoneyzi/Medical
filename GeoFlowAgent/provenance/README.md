<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **provenance**</sub>

# 🧾 Provenance summaries

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

연구 질문 단위의 실험 목록과, 학습된 체크포인트 목록(크기·SHA-256 포함)입니다.

</details>


| File | Content |
|---|---|
| [`experiment_catalog.csv`](experiment_catalog.csv) | One row per research question / control design (S01–R11): stage, sample, design, status, interpretation and evidence path (Korean text; paths updated to this repository's layout) |
| [`checkpoint_catalog.csv`](checkpoint_catalog.csv) | 723 checkpoint entries (`value_geometry.pt`, `search_state_flow.pt`, `training_resume.pt`, …) with size, SHA-256, role and status — file entries including backup copies and resume states, so fewer distinct models |


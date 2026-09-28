<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [Experiments](../README.md) › **① Synthetic pilot**</sub>

# 🧪 ① Synthetic pilot — does the whole agent run end to end?

> **Question —** Can frozen encoders + a small learned metric + a whole-plan flow planner drive a contract-checked tool agent on genomics-style tasks, and does the learned model's contribution show up in task success?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

첫 합성 실험(12개 과제)에서 동결 인코더 → 임베딩 캐시 → 학습된 거리 → flow 계획기 → 도구 실행 에이전트까지 전체 파이프라인이 GPU에서 끝까지 동작함을 확인했습니다. 학습된 bilinear 거리의 test 행동 정확도는 0.8551(3 시드 평균)이었습니다. 그런데 정확한 계약(contract)과 올바른 STOP을 주면 무작위 정책도 모든 test 과제를 풀었습니다. 시험지에 정답 후보가 거의 적혀 있으면 누가 풀어도 만점이 나오는 것처럼, 이 환경은 모델의 기여를 가려낼 수 없었습니다. 이 감사 결과가 다음 단계(정확 탐색 기반 벤치마크)의 출발점이 되었습니다.

</details>

| | |
|---|---|
| **Why this stage** | Before measuring anything, confirm that the full chain — frozen encoder → cache → learned geometry → flow planner → contract-checked tools — runs on GPUs with traceable inputs |
| **Data** | 12 synthetic `variant_to_report` tasks (7 train / 2 dev / 3 test), 8 genomics-style tools |
| **Models** | Frozen Qwen2.5-1.5B (7B as a size comparison), MedCPT, SapBERT, DNABERT-2; only the metric head and the flow model are trained |
| **Compute** | Remote GPU container (RTX 3090 in the run log) |
| **Headline** | Learned bilinear metric: test valid-set accuracy 0.8551 (mean of seeds 17/41/73) — and a **random policy with exact contracts and a correct STOP succeeded on 1.0000 of test tasks** |

## Setup

- **Pipeline.** Prepare tasks → embed each input field with the frozen encoders (immutable, checksummed caches) → sweep six distance families on dev (Euclidean, cosine, diagonal and low-rank Mahalanobis, bilinear, Poincaré) → train a rectified-flow planner over whole tool plans → run a contract-checked closed loop.
- **Selection.** Distances were compared on seed 17, then on seeds 17/41/73; the family moved from low-rank Mahalanobis to **bilinear** after the multi-seed check, and only then was the test split used.
- **Controls.** Random policy with exact contracts, metric without contracts, flow open-loop (execute the whole plan), flow commit-1/commit-2 (execute 1–2 steps, then replan), and a compute-matched blind replanner.

## Results

| Test, seeds 17/41/73 (3 tasks) | Qwen2.5-1.5B views | Qwen2.5-7B views |
|---|---:|---:|
| Learned bilinear metric, valid-set accuracy | **0.8551** | 0.7681 |
| Flow first-action accuracy | 0.4359 | 0.4744 |
| Closed loop: flow commit-1 (execute 1 step, replan) | 0.7778 | 0.4444 |
| Closed loop: random + exact contract + correct STOP | 1.0000 | 1.0000 |
| Closed loop: flow open-loop / blind replan / commit-1 without contracts | 0.0000 | 0.0000 |

Sources: [`results/synthetic_v1/GeoFlowAgent_RUN_STATE.md`](../../results/synthetic_v1/GeoFlowAgent_RUN_STATE.md) (metric and first-action means), per-seed closed-loop summaries in [`results/synthetic_v1/*/seeds/seed*/agent_*test/agent_summary.json`](../../results/synthetic_v1/), audit table in [`FIRST_EXPERIMENT_AUDIT_AND_MARRVEL_TRANSITION_PLAN.md`](../../docs/original/geoacmg/FIRST_EXPERIMENT_AUDIT_AND_MARRVEL_TRANSITION_PLAN.md).

## What it showed

- **The agent runs end to end.** Frozen encoder → cache → learned geometry → flow → agent ran on GPUs with checksummed inputs — the engineering base for every later stage.
- **The benchmark gave the answer away.** With exact contracts the executable tools were almost exactly the optimal tools, so even a random policy finished every test task; task success could not tell a good model from a random one.
- **Ranking and planning are different skills.** The metric ranked actions well while first-action accuracy of whole plans stayed at 0.4359–0.4744, and the 7B views scored below the 1.5B views on the metric (0.7681 vs 0.8551).
- Three test tasks make this an engineering pilot; the ± values in the run log are seed spread.

## What it led to

A benchmark in which the model's contribution can be measured: stage ② labels every state by exact search and adds executable-but-costly tools, so contracts alone no longer reveal the optimal action.

## Files

| File | What it is |
|---|---|
| [`results/synthetic_v1/GeoFlowAgent_RUN_STATE.md`](../../results/synthetic_v1/GeoFlowAgent_RUN_STATE.md) | Run log of both studies: selections, per-seed metrics, checkpoint hashes |
| [`results/synthetic_v1/frozen_smoke_qwen15_multiview/`](../../results/synthetic_v1/frozen_smoke_qwen15_multiview/) | 1.5B study: embedding audit report, per-seed dev/test agent summaries |
| [`results/synthetic_v1/frozen_qwen7b_multiview/`](../../results/synthetic_v1/frozen_qwen7b_multiview/) | 7B study: embedding audit, bilinear and Poincaré agent summaries |
| [`docs/original/geoacmg/FIRST_EXPERIMENT_AUDIT_AND_MARRVEL_TRANSITION_PLAN.md`](../../docs/original/geoacmg/FIRST_EXPERIMENT_AUDIT_AND_MARRVEL_TRANSITION_PLAN.md) | The audit that found the contract/STOP ceiling (Korean) |
| [`docs/original/INITIAL_RESEARCH_PLAN.md`](../../docs/original/INITIAL_RESEARCH_PLAN.md) | Starting plan: FlowAgent and the geometry-module study (Korean) |
| [`src/geoflowagent/models/distances.py`](../../src/geoflowagent/models/distances.py), [`models/flow.py`](../../src/geoflowagent/models/flow.py), [`training/metric.py`](../../src/geoflowagent/training/metric.py), [`training/flow.py`](../../src/geoflowagent/training/flow.py), [`evaluation/agent.py`](../../src/geoflowagent/evaluation/agent.py) | Distance families, whole-plan flow model, training loops and the closed-loop agent |
| [`configs/smoke.yaml`](../../configs/smoke.yaml), [`data/raw/smoke/`](../../data/raw/smoke/) | CPU smoke configuration and invented fixture that exercise the same code path |

---
<sub>[← Experiments](../README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [② Search pilot →](../02_search_pilot/README.md)</sub>

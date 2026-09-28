<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [Experiments](../README.md) › **② Search pilot**</sub>

# 🔎 ② Search pilot — exact-search labels instead of one "gold" trajectory

> **Question —** If every state is labelled by exact search (minimum remaining cost V\*, action values Q\*, regret, all optimal actions) rather than by one reference trajectory, can the contributions of frozen features, learned geometry, DAgger and replanning be separated?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

정답 경로 하나를 따라 하게 하는 대신, 환경을 정확히 탐색해 모든 상태의 남은 최소 비용(V*)과 최적 행동을 계산하도록 바꾼 210개 과제 파일럿입니다. 관측 후 재계획(0.8630)이 한 번에 계획하기(0.5731)와 같은 계산량의 blind 재계획(0.0859)보다 훨씬 좋다는 신호가 여기서 처음 나타났습니다. 동시에 무작위 정책도 41/41을 성공할 만큼 여전히 쉬운 문제였습니다. 너무 쉬운 모의고사에서는 실력 차이가 보이지 않는 것과 같아서, 다음 단계에서 더 어려운 hard-v2를 만들었습니다.

</details>

| | |
|---|---|
| **Why this stage** | In stage ① the contracts almost gave the answer away. Exact search labels every state against all optimal actions, and costly, low-quality or dead-end tools make the choice real |
| **Data** | 210 procedurally generated tasks (128 train / 41 dev / 41 test); exact search over 4,856 train+dev states |
| **Models** | Frozen Qwen2.5-1.5B + MedCPT views vs a hash-embedding control; value geometry, Search-DAgger, State Flow |
| **Compute** | GPU run (`search_pilot_gpu_v1.yaml`); only small heads trained |
| **Headline** | Dev goal completion: observe→replan 0.8630 vs one-shot 0.5731 vs compute-matched blind replan 0.0859 — with random baselines still at 41/41 |

## Setup

- **Search distillation.** A procedural generator builds tool environments with expensive or low-quality but executable tools, alternative paths and recovery situations. Exact search labels every reachable state with V\*, Q\* and the full optimal-action set ([design note](../../docs/original/geoacmg/REVISED_RESEARCH_DIRECTION_SEARCH_DISTILLED_STATE_FLOW.md), Korean).
- **Models.** A goal-conditioned value geometry over frozen (or hashed) features; Search-DAgger rounds that re-label the states the agent actually visits; a State Flow model that samples future state paths and is used one-shot or with receding-horizon replanning.
- **Controls.** Hash embeddings (same pipeline, no language model), input ablations (full / structured / frozen / contract / single views), a compute-matched blind replanner.

## Results (dev split, 41 independent tasks; accuracies are 3-seed means)

| Measure | Value | Reading |
|---|---:|---|
| Branch-state / nontrivial-regret / multi-path fraction | 0.7525 / 0.7398 / 0.5320 | the environment contains real choices |
| Joint STOP/action accuracy, frozen Euclidean vs hash Euclidean | 0.9617 ± 0.0042 vs 0.9578 ± 0.0078 | a small gap on an easy benchmark |
| Input ablation (joint): full · MedCPT-only · structured-only · Qwen-only · contract-only | 0.9617 · 0.9548 · 0.9434 · 0.8476 · 0.4192 | MedCPT increment +0.0146 [+0.0078, +0.0219]; the Qwen increment CI includes 0 |
| State Flow goal completion: one-shot → observe→replan | 0.5731 → **0.8630** | task-macro gain +0.2102 [+0.1604, +0.2550] |
| Compute-matched blind replan | 0.0859 | same planner calls, no observations |
| DAgger-guarded learned policy, clean and perturbed | 41/41 | the **perturbed random** policy also reached 41/41 |

Source: section 4 of the interim development report [`results/hard_v2/GeoFlowAgent_A_EXPERIMENT_REPORT_CURRENT.md`](../../results/hard_v2/GeoFlowAgent_A_EXPERIMENT_REPORT_CURRENT.md).

## What it showed

- **The replanning effect appears.** Observing results and replanning beat one-shot planning by a wide margin, and a blind replanner with the same compute failed — the first appearance of the result that held through every later stage.
- **The benchmark was still too easy.** Frozen and hash embeddings were within each other's seed spread (0.9617 ± 0.0042 vs 0.9578 ± 0.0078 joint accuracy), and a random policy with contracts recovered from perturbations as well as the learned one.
- The pilot validated the mechanics — exact search, test sealing, resumable training — on which hard-v2 was built.

## What it led to

**hard-v2** (stage ③): more tools, held-out compositions, irreversible degraded choices and decoys, a pre-registered final evaluation and a one-time test.

## Files

| File | What it is |
|---|---|
| [`configs/search_pilot_gpu_v1.yaml`](../../configs/search_pilot_gpu_v1.yaml) | GPU procedural pilot: data, value geometry, DAgger, State Flow settings |
| [`configs/search_pilot_frozen_qwen15.yaml`](../../configs/search_pilot_frozen_qwen15.yaml), [`search_pilot_frozen_qwen7b.yaml`](../../configs/search_pilot_frozen_qwen7b.yaml) | Frozen-backbone baselines on the pilot (1.5B and 7B size comparison) |
| [`configs/search_smoke.yaml`](../../configs/search_smoke.yaml), [`data/raw/search_smoke/`](../../data/raw/search_smoke/) | CPU end-to-end smoke version of the same path with synthetic fixtures |
| [`src/geoflowagent/data/procedural_search.py`](../../src/geoflowagent/data/procedural_search.py) | Procedural task/tool/snapshot generator |
| [`src/geoflowagent/search/oracle.py`](../../src/geoflowagent/search/oracle.py), [`data/search_supervision.py`](../../src/geoflowagent/data/search_supervision.py) | Exact search oracle and V\*/Q\*/regret supervision |
| [`src/geoflowagent/training/value.py`](../../src/geoflowagent/training/value.py), [`training/dagger.py`](../../src/geoflowagent/training/dagger.py), [`training/state_flow.py`](../../src/geoflowagent/training/state_flow.py) | Value-geometry training, Search-DAgger, State Flow training and evaluation |
| [`src/geoflowagent/evaluation/search_agent.py`](../../src/geoflowagent/evaluation/search_agent.py) | Closed-loop search agent with clean/perturbed scenarios |
| [`scripts/run_search_smoke.sh`](../../scripts/run_search_smoke.sh) | Smoke runner |

---
<sub>[← ① Synthetic pilot](../01_synthetic_v1/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [③ hard-v2 →](../03_hard_v2/README.md)</sub>

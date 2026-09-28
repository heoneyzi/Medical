# GeoFlowAgent phase 2: execution geometry of frozen biomedical representations

사전등록 2026-09-19T04:17:57Z · fingerprint `143e496ce323074c`

판정은 세 가지 증거만으로 내려집니다 — 짝지은 신뢰구간이 0을 배제하는가, 순열 귀무분포 대비 위치, TOST 동등성. 임계값은 쓰이지 않았습니다.

## 판정

| 주장 | 결론 | 근거 |
|---|---|---|
| **B0** The benchmark contains a planning problem. | 지지 | interval [0.6323, 0.7391] excludes zero in the pre-registered direction |
| **C1** Frozen encoder embeddings carry remaining-cost and next-action information. | 미측정 | no primary finding was reported |
| **C2** That information is biological rather than surface-procedural. | 미측정 | no primary finding was reported |
| **C3** Generating a whole plan beats step-wise selection at matched tool cost. | 미측정 | no primary finding was reported |
| **C4** Non-Euclidean geometry helps in proportion to measured task structure. | 미측정 | no primary finding was reported |
| **C5** Planner validity and representation validity are the same claim. | 미측정 | no primary finding was reported |

## 주검정

| finding | 추정 | 구간 | 단위 | n |
|---|---:|---|---|---:|
| `benchmark_needs_more_than_one_tool_family` | 0.6898 | [0.6323, 0.7391] | gene | 155 |

## 탐색적 결과

사전등록되지 않았으므로 주장을 지지하지 않습니다. p값은 Holm 보정 후 함께 적습니다.

| finding | 추정 | 구간 또는 p | Holm |
|---|---:|---|---|
| `E1_remaining_cost_r2_medcpt_minus_random_projection` | 0.3528 | [0.3528, 0.3528] | — |
| `E1_optimal_action_accuracy_medcpt` | 0.5523 | p=0.004975 | 0.004975 |
| `E2_depth_matched_ordering_medcpt` | -0.05218 | [-0.0934, -0.02256] | — |

## 벤치마크 진단

장비에 대한 서술이며 주장이 아닙니다.

- task 2921 · 유전자 155 · 패널 45
- 유효 클러스터 57.4 (유전자 수가 아니라 이 값으로 구간을 읽을 것)
- 최빈 클래스 0.387
- 단일 도구군 천장 대비 격차 0.690

## 아직 측정되지 않은 주장

주검정 finding이 보고되지 않았습니다: C1, C2, C3, C4, C5.

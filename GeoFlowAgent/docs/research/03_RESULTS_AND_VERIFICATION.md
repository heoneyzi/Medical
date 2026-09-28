# 최종 수치와 검증

[← 실험 목록](02_EXPERIMENT_MAP.md) · [주장과 근거 →](04_CLAIMS_AND_EVIDENCE.md)

이 문서의 모든 수치는 [`results/`](../../results/README.md)에 저장된 결과 파일에서 읽었고, 아래 네 스크립트로 다시 계산합니다.

| 검증 스크립트 | 다시 계산하는 것 | 필요한 자료 | 결과 |
|---|---|---|---|
| [`verify_hard_v2.py`](../../verification/verify_hard_v2.py) | hard-v2 test의 3 시드 평균, 과제별 짝 차이, flow 평균 | 이 저장소 | PASS |
| [`verify_portfolio.py`](../../verification/verify_portfolio.py) | GeoACMG 7 시드 행동 대비와 MedCPT 제거 실험의 추정치 · BCa 구간(run별 파일에서), R7 · B0 산술, 그리고 README · 단계 페이지 · 이 안내 문서에 적힌 모든 소수 수치 대조 | 이 저장소 | PASS |
| [`verify_geoacmg_metrics.py`](../../verification/verify_geoacmg_metrics.py) | R1–R4를 readout margin 배열에서 다시 계산 | margin 배열(51 MB, 로컬 연구 아카이브) | 276/276 PASS |
| [`verify_geoacmg_latest.py`](../../verification/verify_geoacmg_latest.py) | 7 시드 readout, R10, R11을 margin 배열에서 다시 계산 | margin 배열(74 MB, 로컬 연구 아카이브) | 233/233 PASS |

## 1. hard-v2 — 사전등록한 1회 test

선택된 모델(MedCPT-only + cosine + 2배 용량 head)은 시드 17/29/43의 평균이고, 기준 모델은 사전등록한 full + DAgger round 2의 checkpoint 하나입니다.

| 지표 | MedCPT 모델 (상태 수준 시드 평균) | 기준 모델 (상태 수준) | 과제별 짝 차이 Δ [95% 과제 CI] |
|---|---:|---:|---|
| 최적 행동 정책 정확도 ↑ | 0.8010 | 0.6875 | +0.1198 [+0.0708, +0.1682] |
| STOP · 행동 결합 정확도 ↑ | 0.7364 | 0.6702 | +0.0691 [+0.0153, +0.1231] |
| Regret@1 ↓ | 0.1403 | 0.3848 | −0.2489 [−0.3263, −0.1754] |

두 열의 단순 차이와 Δ가 다른 것은 집계 단위가 다르기 때문입니다(과제마다 상태 수가 다릅니다). Δ 점추정은 `verify_hard_v2.py`가 과제별 값에서 다시 계산하고, 구간은 [최종 결론 보고서](../../results/hard_v2/GeoFlowAgent_A_FINAL_EXPERIMENT_CONCLUSION.md)의 과제 bootstrap 구간입니다.

| State Flow (test 과제 루트, 3 시드 평균) | 목표 완수율 |
|---|---:|
| 한 번에 계획 | 0.1233 |
| **관측 후 재계획** | **0.4028** |
| 같은 계산량의 blind 재계획 | 0.0000 |
| 외부 verifier STOP 보조 (상한 대조군) | 0.4375 |

| 폐루프 에피소드 성공률 (시드 17, 과제 48개) | clean | 교란 |
|---|---:|---:|
| MedCPT 학습 정책 | 0.6667 | 0.6667 |
| 사전등록 기준 모델 | 0.6458 | 0.6875 |
| 계약을 준 무작위 정책 | 0.5000 | 0.6667 |

근거: [value 요약](../../results/hard_v2/final_reports/value_test/summary.json), [flow · 에이전트 결과](../../results/hard_v2/final_reports), [재계산 결과 JSON](../../verification/hard_v2_verified_metrics.json)

## 2. GeoACMG 주요 분석 (R0–R7)

분할은 train 1,753 / dev 584 / test 584 과제이고, 모델 · 정렬 평가는 584개 dev 과제에서 했습니다. B0만 코퍼스 전체(과제 2,921개, 유전자 155개)에 대한 진단입니다.

| 분석 | 수치 | 근거 |
|---|---|---|
| B0 단일 도구군 격차 | **0.6898** [0.6323, 0.7391] — 전체 도구 1.0에서 가장 좋은 단일 도구군 0.3102를 뺀 값 | [b0.jsonl](../../results/geoacmg_final/findings/b0.jsonl) |
| 동결 공간 probe | 잔여 비용 R² 0.3355 (무작위 사영 −0.0173), 다음 행동 정확도 0.5523 (우연 0.2896) | [rq1.jsonl](../../results/geoacmg_final/findings/rq1.jsonl) |
| 원시 거리 정렬 | cosine 0.5055 · L2 0.5078 · 내적 0.4966 | [R2](../../results/geoacmg_final/findings/R2_readout_decomposition.json) |
| 학습 좌표의 readout (15 run 평균) | 자기 에너지 0.7405 · 표준화 cosine 0.7199 · 일반 cosine 0.5761 · L2 0.5697 · 음의 내적 0.5579 · whitening cosine 0.4389 | [R2](../../results/geoacmg_final/findings/R2_readout_decomposition.json) |
| R1 cosine − Euclidean 정렬 (파라미터 · 초기값 정합, 5 시드) | +0.1432 [0.0950, 0.1802] | [R1](../../results/geoacmg_final/findings/R1_paired_seeds.json) |
| R1b 새 시드 59/71의 정책 차이 | −0.0099 [−0.0167, −0.0028] (2 시드 평균에 대한 과제 bootstrap) | [R1b](../../results/geoacmg_final/findings/R1b_new_seed_planning_axis.json) |
| R3 과제 길이에 따른 cosine − Euclidean 기울기 (자기 에너지) | −0.0196 [−0.0391, −0.0013] (유전자 47개 클러스터) | [R3](../../results/geoacmg_final/findings/R3_horizon_strata.json) |
| R4 정렬 × 정책: 같은 모델 / 다른 모델 / 차이 | 0.6354 / 0.6350 / +0.000424 [0.000169, 0.000766] | [R4](../../results/geoacmg_final/findings/R4_crossmodel_control.json) |
| R7 목표 완수율: 재계획 / 한 번에 / blind | 0.5993 / 0.1169 / 0.0274 · 재계획 − 한 번에 +0.4824 [0.4439, 0.5223] | [R7](../../results/geoacmg_final/findings/R7_flow_dev_root_only.json) |

R1–R4는 원시 readout margin 배열에서 다시 계산해 276/276 검사를 통과했습니다([검증 결과](../../verification/geoacmg_verified_metrics.json)). R7은 저장된 실패 수에서 완수율을 다시 계산하고, 구간은 보존된 과제 bootstrap 결과입니다.

## 3. 후속 분석 (7 시드, R8–R11)

| cosine − Euclidean, 7 시드 | 차이 | 시드 CI (7) | 유전자 CI (47) |
|---|---:|---|---|
| 정렬 · 자기 에너지 | +0.1523 | [0.1152, 0.1783] | [0.1157, 0.1868] |
| 정렬 · 표준화 cosine | +0.0197 | [−0.0180, 0.0485] | [−0.0007, 0.0291] |
| 정렬 · L2 | +0.0019 | [−0.0248, 0.0298] | [−0.0081, 0.0087] |
| 정렬 · whitening cosine | −0.0368 | [−0.0870, −0.0146] | [−0.0457, −0.0286] |
| 정책 정확도 | +0.0068 | [−0.0033, 0.0153] | [0.0016, 0.0148] |
| STOP · 행동 결합 | +0.0019 | [−0.0097, 0.0129] | [−0.0037, 0.0083] |
| Regret@1 | −0.0208 | [−0.0462, 0.0083] | [−0.0288, −0.0137] |

| 분석 | 수치 | 근거 |
|---|---|---|
| R2d train에서 적합한 표준화 (5 시드) | cosine − Euclidean +0.0170 [0.0118, 0.0222] | [R2d](../../results/geoacmg_working_findings/R2d_holdout_standardiser.json) |
| R8 MedCPT만, train 적합 readout | 표준화 L2 0.5663 · whitening cosine 0.5508 · 무작위 64차원 사영 0.5981 | [R8](../../results/geoacmg_working_findings/R8_frozen_space_ladder.json) |
| R9 PCA-64 + cosine 정렬 | 구조화 해시만 0.7054 · MedCPT + 해시 0.7040 | [R9](../../results/geoacmg_working_findings/R9_information_source_ablation.json) |
| R10 기하 5종 × readout 6종 | 서로 다른 순위 5개, 1위를 차지한 기하 4종 | [R10](../../results/geoacmg_working_findings/R10_readout_dependent_ranking.json) |

| MedCPT 제거 − 포함 | 4 run 추정 | run CI | 유전자 CI | 8 run 집계 |
|---|---:|---|---|---|
| 정책 정확도 | −0.0147 | [−0.0287, −0.0056] | [−0.0276, 0.0159] | −0.0196, 유전자 CI [−0.0323, 0.0071] |
| STOP · 행동 결합 | −0.0162 | [−0.0356, −0.0091] | [−0.0273, 0.0027] | −0.0227, 유전자 CI [−0.0335, −0.0033] |
| Regret@1 | **+0.0441** | [0.0260, 0.0631] | [0.0011, 0.0683] | +0.0550, run CI [0.0405, 0.0654], 유전자 CI [0.0310, 0.0810] |

7 시드 행동 대비(R3c)와 MedCPT 제거(R11, 8 run 집계 포함)의 추정치와 BCa 구간은 `verify_portfolio.py`가 run별 파일에서 다시 계산하며, 저장된 값과 1e-7 이내로 일치합니다. 7 시드 readout(R2f)은 margin 배열 검증(233/233)과 대조합니다.

## 4. 다시 계산하기

```bash
python verification/verify_hard_v2.py      # hard-v2 test 수치
python verification/verify_portfolio.py    # GeoACMG 행동 대비 · 제거 실험 · 문서 수치 대조
python assets/make_figures.py              # 모든 그림을 저장 결과에서 다시 그리기
```

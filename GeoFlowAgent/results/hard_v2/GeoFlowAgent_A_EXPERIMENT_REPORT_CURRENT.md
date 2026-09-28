> 📜 **Original report** — development report of stages ②–③, written before the one-time test; the test result is in [GeoFlowAgent_A_FINAL_EXPERIMENT_CONCLUSION.md](GeoFlowAgent_A_FINAL_EXPERIMENT_CONCLUSION.md). Machine paths are `$GEOFLOW_*` placeholders.

# GeoFlowAgent_a 실험 통합 중간 보고서

- 기준 시각: 2026-09-15 19:53 UTC
- 연구 상태: 진행 중(dev 후속 선택 단계, test 물리적 봉인 유지)
- 새 코드 작업공간: `$GEOFLOW_PROJECT_ROOT` → `$GEOFLOW_PROJECT_ROOT`
- 대형 실행 산출물: `$GEOFLOW_RUNS`
- 영구 핵심 산출물: `$GEOFLOW_RESULTS`
- 최신 소스 복구본: `$GEOFLOW_STATE_DIR/GeoFlowAgent_A_source_20260915T1514Z.tar.gz`
- 소스 복구본 SHA-256:
  `8973d91b69bde1a098d061043bf040825ba096d201eba43a3a45c381c7195130`

이 문서는 현재까지 수행한 새 GeoFlowAgent_a 연구 실험을 한곳에 모은다.
아직 진행 중인 결과는 명시적으로 구분하며, 완료되지 않은 값을 최종 결론처럼
사용하지 않는다.

## 1. 검증하려는 연구 가설

핵심 가설은 다음 세 부분이다.

1. frozen general/biomedical/genomics encoder의 임베딩에는 실행 가능한 workflow의
   진행도와 도구 선택에 유의미한 정보가 있다.
2. 그 표현 위에서 학습한 작은 functional geometry가 단순 hash 표현보다 더 좋은
   행동 선택과 낮은 regret을 만든다.
3. FlowAgent의 핵심인 `계획 → 한 단계 실행 → 결과 관측 → 재계획`은 같은 문제를
   한 번에 계획하는 one-shot 방식이나, 관측 없이 계산만 반복하는 방식보다
   유의미하게 낫다.

거리 함수의 복잡성이 실제로 필요한지도 별도 가설로 두었다. 즉, 방향성·비대칭·
Mahalanobis·hyperbolic geometry가 단순 cosine/Euclidean보다 좋은지 dev에서 비교하고,
좋지 않다면 그 사실도 연구 결과로 보고한다.

## 2. 평가 원칙과 데이터 봉인

- 모든 모델·거리·threshold·용량 선택은 train/dev만 사용한다.
- hard-v2 test 48개 task는 현재 processed data와 embedding cache에 물리적으로 없다.
- test를 읽는 명령은 명시적 `--include-test` 또는 `--split test`와 접근 사유를
  요구하며 hash-chain ledger를 남긴다.
- correlated search-state prefix를 독립 표본으로 간주하지 않는다. 주요 효과의
  신뢰구간은 seed 차이를 먼저 대응 평균한 뒤 48개 독립 dev task를 bootstrap한다.
- 최종 설계가 고정된 후 test를 단 한 번 materialize/encode/evaluate할 예정이다.

## 3. 구현과 실행 강건성

현재 통합 코드는 184개 테스트와 Ruff 검사를 모두 통과한다.

주요 강건성 변경은 다음과 같다.

- exact search를 task shard 단위로 원자 저장하고 재개한다.
- frozen embedding의 field array를 중복 제거·원자 저장·재개한다.
- value geometry와 State Flow는 매 epoch optimizer, scheduler, best model,
  replay pool, sampler state를 `/tmp`에 원자 저장한다.
- immutable frozen feature를 GPU에 한 번 올린 뒤 재사용한다. 실제 batch 256 기준
  첫 전송 0.635초, 재사용 0.0047초로 약 136배 빨라졌고 약 1.48 GiB를 사용한다.
- `/root` progress mirror가 ENOSPC/EDQUOT를 만나도 권위 있는 `/tmp` resume은
  계속 유지한다.
- Python global RNG를 사용하는 view dropout까지 정확히 재개하도록 Python,
  NumPy, Torch, CUDA RNG를 모두 저장한다.
- 새 resume contract는 source-tree hash도 고정한다. 과거 partial checkpoint를
  수정된 코드와 조용히 섞지 않고 새 output directory에서 다시 시작한다.

### 발견하여 수정한 결함

1. hard-v2 generator가 `sequence`를 썼지만 DNABERT serializer는
   `sequence_context`를 읽어 DNA 입력이 모두 비어 있었다. 240개 task 고유 96-nt
   sequence를 올바른 field로 재생성하고 quality gate에 다양성 검사를 추가했다.
2. wide successor frozen feature를 train 전체에 보존해 24 GB GPU OOM이 났다.
   평가에 필요한 label/logit/latent만 남기도록 compact cache로 바꿨다.
3. directed quasimetric의 zero initialization에서 direction gradient가 0이었다.
   bounded map을 zero에서도 학습 가능한 형태로 고치고 기존 partial 결과를 무효화했다.
4. 동일 시드의 uninterrupted run과 과거 resume run이 epoch 6까지 같고 hard replay가
   시작되는 epoch 7부터 달라지는 현상을 찾았다. 원인은 별도 sampler RNG만 저장하고
   view dropout이 쓰는 Python global RNG를 저장하지 않은 데 있었다. 새 resume-v2와
   value-result-v3 계약에서 수정했으며, 최종 선택 후보와 hash control을 새 디렉터리에서
   다시 실행하도록 큐에 넣었다.

## 4. Pilot 실험: 파이프라인 검증과 한계 발견

### 데이터와 exact search

- task 210개: train 128 / dev 41 / sealed test 41
- exact search state 4,856개: train 3,748 / dev 1,108
- branch-state fraction 0.7525
- nontrivial-regret fraction 0.7398
- multi-path fraction 0.5320

### 실제 frozen embedding

Qwen2.5-1.5B와 MedCPT를 실제 frozen backbone으로 사용했다.

- frozen Euclidean 3-seed dev joint: `0.9617 ± 0.0042`
- hash Euclidean 3-seed dev joint: `0.9578 ± 0.0078`
- frozen regret: `0.0040`; hash regret: `0.0138`

작은 이득은 있었지만 task가 너무 쉬워 결론을 내리기 어려웠다.

### Pilot input ablation

| 입력 | 3-seed mean dev joint |
|---|---:|
| full | 0.9617 |
| structured only | 0.9434 |
| frozen only | 0.9365 |
| contract only | 0.4192 |
| MedCPT only | 0.9548 |
| Qwen only | 0.8476 |

MedCPT의 full-model incremental policy 효과는 +0.0146
(95% CI [+0.0078,+0.0219])였고, Qwen incremental 효과는 +0.0021
(CI [-0.0030,+0.0069])로 불확실했다.

### Pilot DAgger와 State Flow

- DAgger guarded learned policy는 clean/perturbed 41/41에 성공했다.
- 하지만 perturbed random도 41/41에 성공해 task ceiling이 분명했다.
- Pilot State Flow one-shot completion: 0.5731
- observed replanning completion: 0.8630
- task-macro gain: +0.2102, 95% CI [+0.1604,+0.2550]
- compute-matched blind replanning: 0.0859

결론: CPU/pilot 결과는 파이프라인과 메커니즘 확인에는 유의미했지만 최종 연구
근거로는 너무 쉬웠다. 이 진단에 따라 hard-v2로 자연스럽게 넘어갔다.

## 5. Hard-v2 benchmark

### 데이터 구성

- task 240개: train 144 / dev 48 / sealed test 48
- tool 50개, snapshot 3,042개
- workflow family 31개, topology 56개
- 240개 고유 96-nt sequence context
- train은 primitive/pair/full composition을 보고, dev/test는 서로 분리된 held-out
  medium-order composition과 finishing tool을 사용한다.
- irreversible degraded-primary choice, alternate source, decoy, release validation,
  report quality control을 포함한다.

최종 fixture SHA-256:
`72e8cf56368320dbeab30e5f187b8e05c285be12bd3ff3804af63110de618bf2`

### Exact search와 난이도

- train+dev state 25,018개: train 16,858 / dev 8,160 / test 0
- applicable action label 92,622개, unresolved action 0
- branch-state fraction 0.8921
- nontrivial-regret fraction 0.9857
- multi-path fraction 0.4517
- oracle/contract collapse fraction 0.0971

Hash Euclidean 3-seed baseline은 dev joint `0.5423 ± 0.0292`, regret `0.4572`다.
따라서 hard-v2는 pilot의 ceiling을 제거했고 모델·표현 차이가 드러나는 난이도다.

## 6. Hard-v2 임베딩 공간 자체의 의미

사용한 frozen view는 Qwen2.5-1.5B, MedCPT, SapBERT, DNABERT2다.
backbone weight는 학습하지 않았다.

| 진단 | Qwen | MedCPT | 해석 |
|---|---:|---:|---|
| dev within-task distance/value Spearman | 0.8184 | 0.4186 | 특히 Qwen에 workflow 진행도 축이 강함 |
| dev pairwise progress-order accuracy | 0.8594 | 0.6684 | task 내부 앞/뒤 순서를 구분함 |
| exact applicability mask hit@1 | 0.4289 | 0.3017 | 후보가 실행 가능할 때 일부 도구 정보가 있음 |
| no-mask hit@1 | 0.0184 | 0.0237 | 임베딩만으로 실행 가능성을 결정할 수 없음 |
| state anisotropy | 0.9981 | 0.9852 | 매우 좁고 편향된 raw 공간 |
| state effective rank | 9.09 | 20.73 | nominal dimension보다 실질 차원이 작음 |
| ridge valid-set action accuracy | 0.1303 | 0.3473 | MedCPT가 선형 action readout에는 더 강함 |

추가 관찰:

- Qwen–MedCPT linear CKA는 0.5284로 완전히 같은 표현은 아니다.
- DNABERT2 effective rank는 수정 전 1.0에서 70.78로 회복됐다.
- DNABERT2–Qwen CKA 0.0386, DNABERT2–MedCPT CKA 0.0736으로 매우 보완적이다.
- train-only whitening은 direct hit@1을 개선하지 않고 일부 조건에서 hubness를
  악화시켰다.
- Qwen no-mask top-1 hub share는 0.7755다.
- 명시적 minimal-pair corpus가 없어 해당 intrinsic 항목은 아직 0개다.

해석: embedding space는 task 내부 진행도를 나타내는 데 분명히 유의미하지만,
전역 도구 검색 공간으로 단독 사용하기에는 anisotropy와 hubness가 심하다.
따라서 dense similarity는 장점인 의미/진행도 신호를 제공하고, symbolic contract
mask는 assembly/version/schema 같은 실행 가능성을 보장해야 한다. 이것이 현재
공간의 핵심 장점이자 한계다.

## 7. Frozen representation이 실제로 유의미한가

새 result-v3/source-bound 계약에서 12개 capacity-matched 입력 조건을 seed
17/29/43으로 모두 완료했다. 모든 조건의 trainable parameter는 791,750개로 같다.

| 입력 | dev policy | dev joint | regret@1 |
|---|---:|---:|---:|
| full multiview + structured | 0.6786 | 0.6223 | 0.3480 |
| frozen-only | 0.8243 | 0.7425 | 0.1824 |
| structured-only | 0.5554 | 0.5148 | 0.4235 |
| contract-only | 0.3583 | 0.3365 | 1.2574 |
| **MedCPT-only** | **0.8234** | **0.7715** | **0.1301** |
| Qwen-only | 0.7080 | 0.6340 | 0.3636 |
| DNABERT2-only | 0.3583 | 0.3365 | 1.2574 |
| SapBERT-only | 0.3583 | 0.2855 | 1.2574 |

대응 task-macro 결과는 frozen-only가 structured-only보다 policy +0.2657
(95% CI [+0.2346,+0.2962])이고, MedCPT-only는 full보다 +0.1356
(CI [+0.0747,+0.1927])이다. MedCPT를 full에서 제거하면 policy가 0.0950 감소
(CI [+0.0718,+0.1192])한다. 즉 frozen 신호, 특히 MedCPT는 강하지만 현재의 단순
concatenation/fusion은 noisy view와 structured feature의 negative transfer를 일으킨다.

표현과 거리의 혼동도 분리했다. full frozen-cosine과 parameter-matched hash-cosine은
policy 차이 -0.0009(CI [-0.0332,+0.0315])로 구분되지 않는다. 따라서 과거의
`full frozen-cosine − hash-Euclidean = +0.0478`만으로 frozen 표현 우위를 주장하면
안 된다. 반면 동일 cosine에서 MedCPT-only는 hash보다 policy +0.1347
(CI [+0.0784,+0.1876]), joint +0.1372(CI [+0.0777,+0.1973]), regret -0.2568
(CI [-0.3119,-0.2017])로 세 지표 모두 명확히 우세하다.

현재 타당한 결론은 “모든 frozen view를 합치면 좋다”가 아니라 “MedCPT frozen
coordinate가 핵심 행동 정보를 담지만, view 선택 또는 gated fusion이 필요하다”이다.
이에 따라 최종 dev 후보를 full+DAGGER에서 MedCPT-only로 수정하고, 그 조건에서
9개 거리·용량·closed-loop를 새로 확인 중이다. 권위 비교는
`$GEOFLOW_RESULTS/search_hard_v2/development/reports/decisive_dev_comparisons`
에 보존했다.

## 8. Hard-v2 거리 함수 비교

아래 값은 모두 dev이며, Poincaré를 제외하면 3개 시드 평균이다.

| 거리/energy | params | seeds | policy | joint | regret@1 | 현재 해석 |
|---|---:|---:|---:|---:|---:|---|
| cosine | 791,750 | 3 | 0.6786 | **0.6223** | 0.3480 | 가장 단순하며 현재 joint 1위 |
| pair MLP | 825,287 | 3 | 0.6584 | 0.6157 | **0.3118** | regret은 낮지만 4.2% 더 큼 |
| Euclidean | 791,750 | 3 | 0.6493 | 0.6098 | 0.3774 | cosine과 통계적으로 비슷함 |
| diagonal Mahalanobis | 791,814 | 3 | 0.6335 | 0.5975 | 0.3813 | 추가 축 가중치 이득 없음 |
| asymmetric bilinear | 795,846 | 3 | 0.6194 | 0.5878 | 0.3603 | 높은 비대칭이나 성능 이득 없음 |
| low-rank Mahalanobis | 793,798 | 3 | 0.5863 | 0.5613 | 0.3999 | Euclidean보다 joint 유의 열세 |
| directed quasimetric | 793,830 | 3 | 0.5725 | 0.5497 | 0.4348 | 방향은 학습되지만 이득 없음 |
| order violation | 793,798 | 3 | 0.5032 | 0.4758 | 0.4751 | Euclidean보다 명확히 열세 |
| Poincaré | 795,846 | 3 | 0.6066 | 0.5726 | 0.3865 | Euclidean보다 joint 유의 열세 |

대응 task 통계:

- cosine − Euclidean joint CI: [-0.0201,+0.0355] → 차이 불확실
- low-rank Mahalanobis − Euclidean joint CI: [-0.0827,-0.0146] → 유의 열세
- order violation − Euclidean joint CI: [-0.1656,-0.1097] → 강한 열세
- Poincaré − Euclidean joint: -0.0428, CI [-0.0777,-0.0082] → 유의 열세
- pair MLP − cosine joint: -0.0049, CI [-0.0319,+0.0226] → 차이 불확실
- pair MLP − cosine regret: -0.0440, CI [-0.0894,+0.0039] → 개선 경향이나 불확실

Directed quasimetric의 수정 후 learned direction norm은 0.040–0.119로 0이 아니므로
방향 parameter는 실제 학습됐다. 그러나 joint/regret이 단순 symmetric geometry보다
낮다. 9-family × 3-seed의 dev primary metric은 cosine을 선택했다. 현재 결과는
“workflow가 방향적이므로 반드시 비대칭 거리가 낫다”는 가설을
지지하지 않는다. 방향성은 transition과 replanning에서 다루고 latent distance는
단순한 symmetric geometry로 두는 편이 더 강한 설계일 수 있다.

입력 ablation 이후 시작한 MedCPT-only 후속 실험에서도 cosine 3시드가 정확히
재현됐다(policy 0.8234, joint 0.7715±0.0231, regret 0.1301). 먼저 완료된
Poincaré 3시드는 0.7569/0.6891±0.0304/0.1982였다. 같은 seed와 48개 task의
대응 비교에서 cosine − Poincaré는 policy +0.0639
(95% CI [+0.0491,+0.0784]), joint +0.0806(CI [+0.0682,+0.0921]),
regret -0.0633(CI [-0.0841,-0.0436])이다. 세 지표 모두 단순 cosine을 지지하며,
더 많은 parameter를 가진 hyperbolic geometry의 이점은 없다. 나머지 7개 거리도
같은 MedCPT-only 조건에서 계속 실행 중이다.

### Hard-v2 Search-DAgger

Cosine seed 17 checkpoint에서 세 라운드를 완료했고 composite dev selection score는
초기 0.7801, round 1 0.7700, round 2 0.7921, round 3 0.7746이었다. 사전 정의된
selection rule은 round 2를 선택했다. 그러나 선택 round와 초기 모델을 48개 독립
dev task에서 직접 비교하면 다음과 같다.

| 선택 round 2 − 초기 cosine | 평균 효과 | 95% task bootstrap CI |
|---|---:|---:|
| policy accuracy | -0.0316 | [-0.0843,+0.0218] |
| joint STOP/action | -0.0018 | [-0.0542,+0.0508] |
| regret@1 | +0.0423 | [-0.0256,+0.1110] |

모든 구간이 0을 포함하므로 DAgger가 분류 지표를 유의하게 개선했다는 주장은 하지
않는다. 현재 선택 checkpoint와 초기 cosine checkpoint를 동일한 clean/perturbed
closed-loop dev 환경에서 별도로 실행 중이며, 실행 성공률을 통해 DAgger의 실제
운영 이득을 최종 판단한다. 선택 checkpoint SHA-256은
`6bee43fb9bb20a8c15d3c20aac5c39340a5aac6cc79b9fcb1fccc885268d41b4`이다.

## 9. Hard-v2 State Flow: one-shot 대 관측 재계획

State Flow는 117,249개 trainable parameter를 가지며 best epoch는 51이다.
dev 4,996개 eligible state, 48개 독립 task, state당 one-shot sample 4개를 평가했다.

| 조건 | goal completion |
|---|---:|
| one-shot whole-plan | 0.3494 |
| execute-observe-replan | **0.7000** |
| same-noise compute-matched blind replanning | 0.0496 |
| external verifier STOP guard | 0.7292 |

통계와 계산량 제어:

- observed replanning − one-shot task-macro:
  `+0.3554`, 95% CI `[+0.2979,+0.4130]`
- root-only 48 task 효과:
  `+0.3802`, 95% CI `[+0.2500,+0.5312]`
- observed replanning − blind replanning:
  `+0.6477`, 95% CI `[+0.5856,+0.7091]`
- observed와 blind의 mean planner calls: 둘 다 `5.9654`
- per-state planner-call match rate: `1.0`
- 둘의 mean total NFE: 둘 다 `95.446`

blind 조건은 observed 조건과 같은 initial noise sequence와 정확히 같은 planner call/NFE
budget을 쓰지만 successor observation을 planner에 보여주지 않는다. 따라서 성능 향상은
단순히 모델을 여러 번 호출해서 생긴 것이 아니라, 중간 결과를 관측하고 다음 조사를
바꾼 feedback에서 비롯됐다는 강한 증거다.

외부 verifier STOP guard 0.7292는 upper-control이며 deployable Flow 성능으로 주장하지
않는다. 현재 intrinsic replanning 0.7000과의 작은 차이는 transition generation보다
STOP calibration이 주된 잔여 오류 중 하나임을 보여준다.

추가 training seed 29와 43의 독립 task-root 평가도 완료됐다. 각각 one-shot은
0.0677/0.0573, observed replan은 0.3542/0.4583, blind replan은 둘 다 0이었다.
`replan − one-shot`은 seed 17/29/43에서 각각 +0.3802, +0.2865, +0.4010으로 모두
같은 방향이며 각 seed의 task-bootstrap CI도 모두 0을 벗어난다. 세 seed root 평균은
one-shot 0.0608, observed 0.4167, blind 0.0000이다. 다만 최종 통합 CI는 같은 task를
seed 사이에서 먼저 대응 평균한 raw root outcome을 보존하도록 evaluator를 보강한 뒤
다시 산출한다.

권위 산출물:

- `$GEOFLOW_RESULTS/search_hard_v2/development/state_flow_frozen_qwen15_small/state_flow_metrics.json`
- metrics SHA-256:
  `d8677ffb12ce875ab22314c0cd9894625ffe8cc1c4386499fa3d11e132e18702`
- checkpoint SHA-256:
  `db95d512e2811eddee5300bdad21102b0232316642a05569725f3145ed1ad034`

## 10. 모델/데이터가 너무 작은가

현재 객관적 판단은 다음과 같다.

- 데이터가 너무 쉬운가: pilot은 그렇지만 hard-v2는 아니다. Hash joint 0.542,
  one-shot completion 0.349, frozen best joint 약 0.62로 ceiling과 멀다.
- 모델이 학습되지 않는가: 아니다. frozen/hash 차이와 State Flow feedback 효과가
  크고 신뢰구간이 0을 벗어난다. train/dev loss와 best epoch도 정상적으로 선택됐다.
- 모델이 너무 작은가: 아니다. cosine full-input의 half/configured/two-x는 dev joint
  0.6692/0.6223/0.6558이다. half는 configured보다 task-macro policy +0.0423
  (95% CI [+0.0163,+0.0679])로 오히려 좋다. Euclidean에서도 0.6244/0.6098/0.6041로
  큰 head의 단조 이득이 없다. 문제는 head under-capacity가 아니라 독립 task 수와
  fusion/generalization이다. MedCPT-only에서도 같은 capacity curve를 확인 중이다.
- 데이터가 너무 작은가: 144개의 독립 train task에 16,858개 state가 있으나 state는
  task 내부 상관이 있다. 유효 독립 단위는 task 144개에 가깝다. 따라서 더 많은
  독립 workflow family와 실제 MARRVEL-derived task가 필요한 한계는 남는다.

## 11. 현재 실행 중인 실험

2026-09-15 19:53 UTC 기준:

- 기존 dev autopilot은 완료된 36개 input-ablation 결과의 source/hash 호환성을 다시
  검증한 뒤 초기 cosine 및 DAgger-selected checkpoint의 clean/perturbed closed-loop와
  전체 test/Ruff를 수행한다. 이는 test를 열지 않는다.
- 별도 MedCPT follow-up은 GPU 0에서 MedCPT-only 9-distance × 3-seed sweep을 수행한다.
  이어 half/configured/two-x capacity와 geometry/capacity dev winner의 closed-loop를
  평가하고 각 단계 완료 시 `/root`에 checksum과 함께 복사한다.
- 이미 완료: 12 input variant × 3 seed, full cosine 및 hash Euclidean/cosine clean-v3,
  full cosine capacity curve, State Flow seed 29/43 root evaluation.

실행 컨트롤러와 로그:

- dev autopilot PID: `20542`
- dev guardian PID: `22131`
- MedCPT front/tail supervisors: `41683` / `41486`
- `$GEOFLOW_RESULTS/search_hard_v2/development/autopilot/autopilot.log`
- `$GEOFLOW_RESULTS/search_hard_v2/development/autopilot/medcpt_followup.log`

프로세스가 내려가더라도 다음 로컬 checkpoint에서 재개한다.

- `$GEOFLOW_RUNS/search_hard_v2/development/checkpoints`
- `$GEOFLOW_RESULTS/search_hard_v2/development/value_training_progress`
- `$GEOFLOW_STATE_DIR/GeoFlowAgent_A_run_state.json`
- `$GEOFLOW_STATE_DIR/GeoFlowAgent_A_RUN_STATE.md`

## 12. 남은 완료 조건

1. ~~Poincaré 3개 시드와 9-family dev aggregation 완료~~
2. ~~dev-selected geometry를 cosine으로 동결~~
3. ~~hard-v2 Search-DAgger 3 rounds 및 dev checkpoint 선택 완료~~;
   초기/선택 checkpoint의 clean/perturbed closed-loop 비교 완료
4. ~~hard-v2 12-condition input ablation 완료~~; MedCPT-only follow-up 완료
5. ~~full cosine half/configured/two-x clean capacity curve 완료~~;
   MedCPT-only capacity curve 완료
6. ~~corrected resume-v2 및 동일-cosine hash 대조 재현 완료~~
7. ~~State Flow seed 17/29/43 root 효과 방향 재현 완료~~;
   raw task×seed 통합 bootstrap 보고
8. 모든 선택과 source/config/checkpoint hash를 freeze manifest로 기록
9. test 48개를 단 한 번 materialize/encode하고 선택된 value/DAgger/State Flow 평가
10. 최종 통계·거리 해석·실패 slice·synthetic/offline 한계를 통합 보고

## 13. 현재까지의 핵심 결론

현재 dev evidence만으로도 다음은 강하게 지지된다.

1. 임베딩은 무의미하지 않다. 특히 Qwen 공간의 task 내부 진행도 순서가 강하고,
   MedCPT-only는 동일 cosine/거의 동일 parameter의 hash보다 policy +13.5%p로 유의하게
   높다. 다만 full frozen과 hash의 동일-cosine 차이는 없어서 view 선택이 핵심이다.
2. 임베딩만으로는 충분하지 않다. no-mask tool retrieval은 거의 실패하고 hubness와
   anisotropy가 크므로 symbolic applicability contract가 필수다.
3. 복잡한 거리가 자동으로 낫지 않다. 현재는 cosine/Euclidean 같은 단순 symmetric
   geometry가 가장 안정적이고, 방향성·order·low-rank 제약은 오히려 손해다.
4. FlowAgent의 여러 단계 조사와 관측 재계획은 one-shot보다 큰 폭으로 낫다.
   같은 계산량의 blind 반복이 실패하므로 핵심 원인은 중간 관측 feedback이다.
5. 현재 head는 너무 작지 않다. half-width가 configured보다 낫고 two-x도 단조 개선하지
   않아 더 큰 모델보다 gated/selected fusion과 독립 task 확대가 우선이다.
6. hard-v2는 너무 쉬운 task가 아니다. 다만 아직 synthetic exact-snapshot 환경이며,
   실제 API 오류·argument generation·MARRVEL external test로 일반화하는 근거는 남아 있다.

최종 결론은 진행 중인 다중 시드 robustness 결과와 봉인된 test 1회 평가가 끝난 뒤
이 문서를 갱신해 확정한다.

## 14. 향후 실험 진행 계획

향후 실험은 결과를 본 뒤 임의로 유리한 설정을 추가하는 방식이 아니라, 아래 순서와
판정 규칙을 먼저 고정하고 진행하는 것이 좋다.

### 단계 A. 현재 synthetic hard-v2 dev 실험 마무리

1. Poincaré를 포함한 9개 거리 × 3개 시드를 완료한다.
2. primary metric은 task-macro joint STOP/action accuracy로 둔다.
3. secondary metric은 task-macro policy accuracy, regret@1, seed 표준편차다.
4. 가장 높은 평균만 고르지 않는다. 단순 geometry와 paired CI가 겹치면 더 단순하고
   parameter가 적은 geometry를 선택한다.
5. 복잡한 geometry는 단순 기준보다 accuracy CI 전체가 0보다 높거나 regret CI 전체가
   0보다 낮을 때만 채택한다.
6. 선택한 geometry를 source-bound result-v3에서 3개 시드로 처음부터 재현한다.

현재 값이 유지된다면 cosine이 기본 선택이고 pair MLP는 regret-oriented 보조 실험으로
남기는 것이 타당하다. Poincaré 3개 시드 평균이 나오기 전에는 확정하지 않는다.

### 단계 B. 표현·용량 원인 분해

동일 split과 seed를 유지한 채 다음을 완료한다.

- frozen full 대 parameter-matched hash
- full 대 frozen-only 대 structured-only 대 contract-only
- 각 view-only: Qwen, MedCPT, SapBERT, DNABERT2
- 각 leave-one-view-out
- half/configured/two-x capacity curve
- encoder size control은 pilot 1.5B/7B 결과를 유지하되, hard-v2에서 필요하면
  dev-selected 1.5B와 한 개 큰 encoder만 비교한다.

판정은 다음처럼 한다.

- two-x가 train deficit과 dev 성능을 함께 개선하면 head under-capacity로 판단한다.
- train만 개선하고 dev가 그대로이거나 하락하면 독립 task 수 또는 regularization/
  representation generalization 문제로 판단한다.
- half와 configured가 같으면 현재 head가 이미 plateau에 있고 더 키울 근거가 약하다.
- view 제거 효과는 seed를 먼저 평균하고 task bootstrap CI를 계산한다.
- 여러 view 효과를 동시에 검정할 때는 raw CI와 함께 Holm 보정 또는 FDR 결과를
  보조 표로 보고한다.

### 단계 C. FlowAgent 핵심 가설의 다중 시드 재현

State Flow seed 17/29/43에서 각 독립 task root에 대해 다음 네 조건을 같은 noise와
예산으로 평가한다.

1. one-shot whole plan
2. execute-observe-replan
3. observation을 숨긴 compute-matched blind replan
4. external verifier STOP guard upper-control

Primary 효과는 `observed replan − one-shot`, 핵심 기전 제어는
`observed replan − blind`다. 세 training seed 안에서 같은 task의 차이를 먼저 평균한 뒤
task bootstrap을 해야 한다. seed별 효과 방향도 모두 함께 보고한다.

추가로 horizon 1–3, 4–6, 7+; branch 수; degraded-primary 여부; held-out finishing tool;
workflow family별 slice를 계산한다. 평균 효과가 좋아도 긴 horizon이나 특정 source에서
무너지면 그 조건을 명시한다.

### 단계 D. closed-loop stress와 baseline

Search-DAgger 이후 다음 perturbation을 사전 등록한다.

- primary source empty result
- stale source revision
- schema mismatch/누락 field
- timeout과 한 번의 retry
- entity alias ambiguity
- wrong assembly/version을 거부한 뒤 liftover로 복구
- decoy tool이 contract-valid하지만 높은 regret을 갖는 경우

각 조건에서 success뿐 아니라 excess tool calls, regret, recovery rate, premature STOP,
invalid action, compute/NFE를 보고한다. 비교 대상은 learned policy, random-contract,
rule/FSM, static one-shot, observed replan, blind replan, oracle upper bound다.

Random/FSM과 learned가 비슷하면 모델을 먼저 키우지 않는다. 독립 branch, decoy,
workflow composition과 실제 source failure를 늘려 task를 어렵게 만든 뒤 다시 평가한다.

### 단계 E. 실제 데이터 20–30 task pilot

전체 실제 dataset을 만들기 전에 다단계 dependency가 분명한 MARRVEL/비-MARRVEL
20–30개를 선정한다.

- 실제 wrapper 실행과 timestamped snapshot을 보존한다.
- QA 정답에서 trajectory를 역으로 만들어 내지 않는다.
- private verifier와 reference answer는 model input 밖에 둔다.
- 성공 branch뿐 아니라 실패/alternate/decoy branch도 캡처한다.
- 두 명의 reviewer가 task contract, verifier, snapshot replay를 독립 검수한다.
- one-step QA는 multi-step primary endpoint에 섞지 않고 routing subset으로 분리한다.

다음 gate를 모두 통과할 때만 full dataset으로 확장한다.

- snapshot replay와 private verifier 통과율 100%
- protected group leakage 0
- expected answer 노출 0
- unknown counterfactual이 regret/selection loss에 들어간 사례 0
- contract-only/random baseline이 포화되지 않음
- learned model이 최소 하나의 단순 baseline을 task level에서 일관되게 이김
- reviewer 합의율과 제외율이 사전 기준을 충족

### 단계 F. 논문용 최종 분석

- task가 독립 통계 단위다. state prefix 수를 표본 수로 쓰지 않는다.
- 세 seed를 단순히 합쳐 수천 개 표본처럼 취급하지 않는다.
- accuracy/success에는 task bootstrap 또는 hierarchical bootstrap, 이항 결과에는
  Wilson interval도 함께 제시한다.
- distance family는 dev에서 선택하므로 9개 family의 dev 최대값을 test 결과처럼
  제시하지 않는다. test에는 선택된 family 하나만 들어간다.
- 평균, seed 표준편차, task CI, workflow/entity/source slice와 failure count를 함께 낸다.
- synthetic와 real 결과를 합쳐 하나의 성공률로 만들지 않고 두 benchmark를 별도 표로
  보고한다.

## 15. 최종 데이터셋 분할 권고안

### 15.1 기본 원칙: task를 만들기 전에 group을 나눈다

절대로 trajectory나 search prefix를 만든 뒤 행 단위로 무작위 분할하면 안 된다.
먼저 원래 case/task의 protected group을 정하고 split을 배정한 뒤, 그 split 안에서만
exact search state와 prefix를 생성한다.

보호해야 할 group key는 최소 다음 다섯 개다.

- `case`: 같은 환자/질문/원본 사례 및 파생 task
- `entity`: 같은 variant, gene, disease family와 alias/paraphrase
- `template`: 같은 query/task 문형과 구조적 변형
- `source`: 같은 원천 dataset/provider 계열
- `release`: 같은 source snapshot/revision

두 task가 위 key 중 하나라도 공유하면 같은 connected component로 묶어 한 split에만
배치한다. 예를 들어 같은 variant를 다른 문장으로 표현하거나 동일 gene에서 파생한
여러 질문은 서로 다른 task ID여도 같은 entity/case group이다.

### 15.2 권장 논리적 partition

최종 연구는 가능하면 네 개의 논리적 partition을 둔다.

| partition | 권장 최소 독립 task | 용도 | 설정 변경 허용 여부 |
|---|---:|---|---|
| train | 500 | fitting, DAgger train rollout | 허용 |
| dev-selection | 60 | geometry/capacity/threshold 선택 | 허용 |
| dev-confirmation | 40 | 선택된 단일 protocol의 1회 내부 확인 | 확인 후 금지 |
| external test | 100 | 고정 MARRVEL-derived 최종 평가 | 절대 금지 |

현재 코드의 기본 quality floor인 `train 500 / dev 100 / test 100`과 호환하려면
dev 100개에 사전 고정된 `dev_role: selection|confirmation` 또는 group-fold ID를 추가한다.
trainer는 selection 60개로만 model family와 threshold를 고르고, confirmation 40개는
그 선택이 끝난 뒤 한 번만 평가하도록 다음 구현 단계에서 분리하는 것이 좋다.

데이터가 더 충분하면 external test 100은 최소치일 뿐이다. 성공률 50% 부근에서 100개
독립 task의 95% 이항 구간 반폭은 대략 10%p이므로, 약 ±5%p 정밀도가 필요하면
약 400개 독립 test task가 바람직하다. 현실적 중간 목표는 200개이며, 데이터가 부족할
때 prefix를 늘려 이 독립 표본 수를 대신해서는 안 된다.

### 15.3 source별 역할

- MARRVEL-derived task는 전부 external test 전용으로 둔다.
- non-MARRVEL의 curated executable task만 train/dev-selection/dev-confirmation에 둔다.
- MARRVEL item을 연구자가 executable task와 verifier로 큐레이션해야 하므로 최종 set을
  `pristine blind test`라고 부르지 않는다. 정확한 표현은
  `item-level curation 이후 고정한 held-out external benchmark`다.
- MARRVEL에서 exact executable task로 만들 수 없는 항목의 제외 기준은 결과를 보기
  전에 등록하고, 제외 분모와 이유를 함께 공개한다.

### 15.4 난이도와 분포 균형

보호 group을 지키는 것이 비율 균형보다 우선이다. 그 제약 안에서 다음 축을
stratification target으로 사용한다.

- workflow family와 composition order
- assembly와 coordinate convention
- variant/entity category
- source revision과 snapshot age
- shortest-path horizon
- branch/decoy/alternate-source 수
- degraded-primary 및 recovery 필요 여부
- finishing tool과 report requirement

workflow family는 compositional generalization을 보기 위해 여러 split에 나타날 수 있지만,
정확히 같은 template, entity, case, source release는 공유하면 안 된다. dev/test에는 train에
없는 composition과 finishing tool을 일부 의도적으로 배치하되, 어느 항목을 hold out할지
split 전에 고정한다.

### 15.5 split 생성 절차

1. 모든 raw task에 case/entity/template/source/release group ID를 완전하게 부여한다.
2. group ID가 하나라도 연결된 task들의 connected component를 계산한다.
3. component 단위로 MARRVEL 여부를 먼저 분리해 MARRVEL은 test에 고정한다.
4. non-MARRVEL component를 train/dev-selection/dev-confirmation에 constrained assignment한다.
5. 목표 stratification과 task 수를 검사하되 protected group을 쪼개지 않는다.
6. split manifest와 raw task hash를 동결한다.
7. 그 후에만 각 split 내부에서 exact search, prefix, counterfactual, embedding을 생성한다.
8. quality gate로 group 충돌, private label 노출, snapshot ownership, source revision,
   branch/regret coverage를 검사한다.
9. train/dev-only cache와 final test-inclusive cache를 서로 다른 디렉터리에 만든다.
10. 모든 선택이 끝난 뒤 freeze manifest의 hash와 일치할 때만 test 접근 ledger를 append한다.

### 15.6 최종 test 실행 순서

1. encoder revision, serializer, geometry, model width, seed protocol, threshold,
   DAgger round, perturbation, primary metric을 freeze manifest에 기록한다.
2. dev checkpoint와 모든 config/source/cache hash를 `/root`에 백업한다.
3. 별도 final config와 별도 processed/cache directory를 만든다.
4. `prepare-search --include-test`로 test를 처음 materialize한다.
5. `audit-search-data --include-test`로 group/private-label/snapshot gate를 통과시킨다.
6. frozen encoder로 test embedding만 추가 생성하고 training을 다시 하지 않는다.
7. 이미 선택된 checkpoint를 명시적으로 넘겨 value/DAgger/State Flow test를 평가한다.
8. test 결과를 본 뒤 prompt, threshold, serializer, task 제외, model을 바꾸지 않는다.

만약 최종 test를 본 뒤 수정이 필요하면 해당 test는 더 이상 final external test가 아니다.
접근 횟수와 변경 이유를 기록하고, 새 external set을 확보한 뒤 다시 고정해야 한다.

### 15.7 최종 보고 표 구조

최종 논문/보고서는 최소 다음 표를 분리해서 제시한다.

1. synthetic hard-v2 dev/test: hash, structured, frozen, capacity, geometry
2. one-shot/observed replan/blind/verifier: success, cost, calls, NFE
3. clean/perturbed closed loop: recovery와 excess calls
4. real dev-confirmation과 MARRVEL external test
5. workflow/entity/source/assembly/horizon별 slice
6. embedding intrinsic audit: progress order, rank, anisotropy, hubness, mask/no-mask
7. failure taxonomy와 실제 예시

이 분할과 실행 순서를 따르면 “임베딩이 유의미하다”, “특정 거리가 유리하다”,
“관측 재계획이 one-shot보다 낫다”는 세 주장을 서로 섞지 않고 각각 검증할 수 있다.

> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# GeoFlowAgent 첫 실험 감사와 MARRVEL 전환 계획

작성일: 2026-09-15  
대상 결과: `pp-result1/`  
현재 판정: `synthetic_engineering_studies_complete`

## 0. 결론

이번 실험은 **frozen encoder → embedding cache → learned geometry → flow planner → contract-constrained closed loop**가 실제 GPU 환경에서 끝까지 동작한다는 것을 보여 준다. 이 점에서는 분명히 의미 있는 성공이다.

그러나 현재 수치가 보여 주는 가장 강한 과학적 결론은 “FlowAgent가 잘한다”가 아니다. 실제 결론은 다음에 가깝다.

> Frozen embedding에는 합성 workflow 단계를 구분할 수 있는 신호가 일부 존재하지만, raw 거리 공간은 행동 선택에 적합하지 않다. 작은 학습 모듈로 action ranking은 크게 개선할 수 있으나, 좋은 metric이 좋은 전체 trajectory로 자동 전이되지는 않는다. 현재 closed-loop 성공은 learned planner보다 exact contract와 매 단계 피드백, 특히 STOP 동작에 크게 좌우된다.

따라서 다음 판단이 적절하다.

| 질문 | 판단 |
|---|---|
| 파이프라인이 구현·실행 가능한가? | **예** |
| frozen representation 위에 geometry를 학습할 가치가 있는가? | **예, 유망한 가설** |
| bilinear가 genomics에서 보편적으로 최고인가? | **아직 모름** |
| Flow가 단순 planner보다 우수한가? | **입증되지 않음** |
| 7B가 1.5B보다 좋은가? | **일관된 이득 없음** |
| 현재 성공률이 실제 genomics 능력인가? | **아님** |
| MARRVEL 100행을 지금 그대로 학습할 수 있는가? | **아님** |
| 현재 코드 구조를 MARRVEL 연구 기반으로 재사용할 수 있는가? | **예, 단 데이터·loss·평가를 먼저 보강해야 함** |

가장 합리적인 다음 단계는 **Qwen2.5-1.5B + bilinear + hard contract + commit-1**을 하나의 사전 고정 baseline으로 보존하고, 별도로 20~30개의 실행 검증 MARRVEL pilot trajectory를 구축하는 것이다. 이 선택은 1.5B가 통계적으로 우월해서가 아니라, 7B의 일관된 이득이 없고 비용이 더 작기 때문이다.

---

## 1. 무엇을 감사했는가

주요 근거 파일은 다음과 같다.

- 실행 상태: `pp-result1/GeoFlowAgent_RUN_STATE.md`
- 기계 판독 상태: `pp-result1/GeoFlowAgent_run_state.json`
- 1.5B 결과: `pp-result1/frozen_smoke_qwen15_multiview/`
- 7B 결과: `pp-result1/frozen_qwen7b_multiview/`
- 소스·설정 복구본: `pp-result1/GeoFlowAgent_resume_bundle_20260914.tar.gz`
- 첨부된 비판적 해석문도 독립적으로 수치와 코드를 대조했다.

확인한 범위는 다음과 같다.

1. 데이터 규모와 split 구조
2. raw embedding, ridge probe, minimal pair 결과
3. 여섯 거리 함수의 multi-seed 결과
4. flow first-action, 전체 suffix, edit distance
5. agent 조건별 closed-loop trace
6. contract-valid set과 zero-regret set의 관계
7. STOP 실패 위치
8. MARRVEL curation queue의 실제 완성도
9. `observed_only` 전처리와 selection loss 구현
10. 현재 `geomarrvel_mvp.yaml`의 모델 크기와 hyperparameter

아티팩트의 checksum 연결은 전반적으로 양호하다. processed/cache manifest, checkpoint–metrics, agent summary–details 연결은 일치한다. 다만 논문 수준 재현성을 위해서는 뒤에서 설명할 source version 및 bookkeeping 정리가 더 필요하다.

---

## 2. 실제 실험 규모

### 2.1 독립 데이터 단위

| 구분 | Train | Dev | Test | 전체 |
|---|---:|---:|---:|---:|
| 독립 synthetic task | 7 | 2 | 3 | 12 |
| terminal 포함 prefix row | 60 | 17 | 26 | 103 |
| nonterminal action row | 53 | 15 | 23 | 91 |

중요한 점은 103개 row가 103개의 독립 생물학 사례가 아니라는 것이다. 한 task의 trajectory를 앞에서부터 잘라 여러 prefix로 만든 것이므로, 같은 task에서 나온 row들은 강하게 상관되어 있다. 통계적 표본 수는 103보다 12에 가깝고, 최종 test의 독립 task는 3개뿐이다.

### 2.2 실험 환경의 성격

모든 task는 같은 `variant_to_report` 유형이며 같은 목표인 `report_ready=true`를 사용한다. 사용할 수 있는 도구도 다음 8개로 동일하다.

1. `parse_variant`
2. `liftover_grch37_to_grch38`
3. `verify_reference_allele`
4. `normalize_variant`
5. `annotate_variant`
6. `match_phenotype`
7. `lookup_gene_disease`
8. `integrate_evidence`

각 경로의 길이는 7~8개 action이지만, 본질적으로 같은 partial-order state machine에서 phenotype matching의 순서만 조금 달라진다.

- 실제 database/API snapshot: 0개
- private biological verifier: 0개
- 실제로 배정·실행된 perturbation: 0개
- minimal pair: 5개

따라서 task success는 실제 유전자·질환 답이 맞았다는 뜻이 아니다. 합성 transition을 따라 `report_ready=true`에 도달하고 STOP했음을 뜻한다.

### 2.3 경로 중복

정확한 tool suffix를 문자열 단위로 비교하면 다음과 같다.

- Dev nonterminal suffix 15개 중 12개가 train에 이미 존재
- Test nonterminal suffix 23개 중 19개가 train에 이미 존재
- 전체 trajectory도 dev 2개 중 1개, test 3개 중 2개가 train trajectory와 동일

즉 현재 test는 새로운 workflow 구조를 푸는 extrapolation보다, 같은 workflow에 새로운 synthetic variant 문자열을 넣은 interpolation에 가깝다.

---

## 3. 핵심 결과 표

아래의 `±`는 동일한 test 사례를 seed 17/41/73으로 다시 학습한 결과의 표본 표준편차다. 새로운 환자 또는 새로운 task 모집단에 대한 신뢰구간이 아니다.

Metric 수치는 23개 nonterminal test prefix를, flow 수치는 terminal 3개를 포함한 26개 test prefix를 평가한다. 두 지표는 분모와 문제가 다르므로 서로 직접적인 우열 수치로 비교하지 않는다.

| Test 지표 | Qwen2.5-1.5B + bilinear | Qwen2.5-7B + bilinear | 해석 |
|---|---:|---:|---|
| Raw Qwen cosine valid-hit@1 | 17.4% | 8.7% | raw 최근접 도구 선택은 chance 수준 |
| Ridge valid-set accuracy | 65.2% | 65.2% | 선형적으로 꺼낼 수 있는 stage 신호는 존재 |
| Learned metric valid-set accuracy | 85.5% ± 10.9% | 76.8% ± 6.6% | 학습 모듈은 synthetic action ranking을 개선 |
| Flow valid-first-action | 43.6% ± 5.9% | 47.4% ± 8.0% | 전체 row에서 첫 action은 중간 수준 |
| Full suffix exact match | 9.0% | 9.0% | 전체 계획 생성은 아직 약함 |
| Normalized edit distance ↓ | 0.579 ± 0.045 | 0.570 ± 0.097 | 예측 경로가 gold suffix와 상당히 다름 |
| Flow commit-1 task success | 77.8% ± 38.5% | 44.4% ± 50.9% | STOP seed 변동이 큼 |
| Random + contract + oracle STOP | 100% | 100% | contract가 사실상 정답 집합을 제공 |
| Metric + contract + oracle STOP | 100% | 100% | random baseline보다 개선 없음 |
| Flow open-loop | 0% | 0% | 한 번에 계획한 suffix는 실패 |
| Flow commit-1, no contract | 0% | 0% | contract 의존도가 매우 큼 |
| Compute-matched blind replan | 0% | 0% | 새 관측 없는 반복 계산은 도움 없음 |

---

## 4. Embedding 결과는 무엇을 뜻하는가

### 4.1 Raw 공간은 action geometry가 아니다

도구가 8개이고 test 상태당 평균 valid tool 수가 약 1.30개이므로, 모든 도구 중 무작위로 하나를 고를 때 valid set에 들어갈 확률은 대략 16.3%다.

- 1.5B raw cosine test hit@1: 17.4%
- 7B raw cosine test hit@1: 8.7%

즉 raw cosine 공간만으로는 다음 action을 거의 고르지 못한다.

Qwen state embedding은 매우 anisotropic하다.

| 지표 | 1.5B | 7B |
|---|---:|---:|
| State anisotropy | 약 0.992 | 약 0.976 |
| Effective rank | 약 3.6 | 약 4.6 |

벡터가 좁은 방향에 몰려 있어, 텍스트 의미가 있더라도 단순 cosine/Euclidean ranking이 행동 관계를 잘 드러내지 못한다.

### 4.2 그래도 학습 가능한 정보는 있다

Ridge probe가 test에서 두 backbone 모두 65.2%를 얻었다. 이는 frozen embedding과 입력 표현 안에 workflow stage를 구분하는 신호가 있다는 뜻이다.

그러나 이것을 곧바로 “genomics 지식이 들어 있다”고 해석하면 안 된다. 모델은 다음과 같은 더 쉬운 단서를 이용했을 수 있다.

- history에 어떤 tool 이름이 이미 등장했는가
- state의 boolean field가 켜졌는가
- 합성 query template가 어느 단계와 연관되는가
- structured-state feature가 무엇인가

또한 raw Qwen과 learned metric은 입력이 완전히 같지 않다. learned head는 Qwen, MedCPT, auxiliary view, structured state를 함께 사용하므로 `17.4% → 85.5%` 전체를 bilinear 거리 하나의 효과로 돌릴 수 없다.

필요한 ablation은 다음과 같다.

- Qwen-only + cosine
- Qwen-only + learned bilinear
- structured-state-only
- multiview without structured state
- MedCPT-only
- auxiliary view 제거
- embedding row shuffle
- tool description shuffle

이 비교를 해야 “backbone 정보”, “구조화 상태”, “멀티뷰 결합”, “거리 함수”의 기여가 분리된다.

### 4.3 Genomics exactness는 raw embedding에서 나타나지 않았다

5개 minimal pair에서 기능적으로 민감한 변경은 의미 보존 paraphrase보다 오히려 더 가까웠다.

| Backbone | Functional-sensitive 거리 | Invariant 거리 | Separation AUC |
|---|---:|---:|---:|
| Qwen-1.5B | 0.0021 | 0.0326 | 0.5 |
| Qwen-7B | 0.00745 | 0.0800 | 0.5 |

이는 `GRCh37/GRCh38`, accession version, coordinate convention 같은 exact field를 dense text embedding에 맡기면 안 된다는 근거다. 다만 pair가 5개뿐이고 paraphrase의 표면적 문장 변화량이 더 크므로, 이것 역시 확정적 생물학 결론이 아니라 경고 신호다.

정확한 결론은 다음이다.

> Dense embedding은 semantic context를 보조할 수 있지만, assembly·좌표계·transcript version·reference allele·database release는 typed state와 exact validation으로 보존해야 한다.

---

## 5. 거리 함수 실험의 의미

### 5.1 Multi-seed dev 결과

| Backbone | Geometry | Valid-set accuracy |
|---|---|---:|
| 1.5B | Bilinear | 0.911 ± 0.038 |
| 1.5B | Low-rank Mahalanobis | 0.867 ± 0.133 |
| 1.5B | Euclidean | 0.800 ± 0.115 |
| 1.5B | Cosine | 0.756 ± 0.102 |
| 7B | Poincaré | 0.956 ± 0.038 |
| 7B | Bilinear | 0.911 ± 0.077 |
| 7B | Euclidean | 0.844 ± 0.077 |
| 7B | Cosine | 0.822 ± 0.102 |

이 결과는 사용자의 원래 관점인 “경로와 목적에 맞는 geometry가 중요하다”는 가설을 지지하는 초기 신호다. 단순 Euclidean이 항상 최악은 아니지만, 1.5B에서는 bilinear가 seed 평균과 안정성에서 더 나았다.

### 5.2 가장 중요한 negative result: metric→flow 전이는 자동이 아니다

7B에서는 Poincaré가 metric 기준으로 bilinear보다 좋았다.

- Metric valid-set accuracy: Poincaré 0.956, bilinear 0.911
- Flow first-action: Poincaré 0.490, bilinear 0.549
- Flow edit distance: Poincaré 0.644, bilinear 0.509

즉 국소적인 next-action ranking에 좋은 geometry가 전체 vector field와 suffix 생성에도 좋은 것은 아니다.

이것은 실패라기보다 다음 연구 질문을 만든다.

> 좋은 action metric을 학습하는 목적과 좋은 trajectory vector field를 학습하는 목적은 어떻게 정렬해야 하는가?

후속 연구에서는 geometry를 `valid-set accuracy` 하나로 고르지 말고 다음을 함께 봐야 한다.

- end-to-end task success
- path regret 또는 utility
- first-action validity
- STOP calibration
- suffix edit distance
- perturbation recovery
- contract-only baseline 대비 이득

---

## 6. Closed-loop 성공률을 그대로 믿으면 안 되는 이유

### 6.1 Contract mask가 정답 집합과 정확히 같았다

전체 91개 nonterminal prefix에서 다음이 성립한다.

```text
set(contract_valid_tools) == set(zero_regret valid_next_tools)
```

후보 수는 다음과 같다.

| 구간 | 후보 1개 | 후보 2개 | 후보 3개 이상 |
|---|---:|---:|---:|
| 전체 91 prefix | 58 | 33 | 0 |
| Test 23 prefix | 16 | 7 | 0 |

두 후보가 있는 경우에도 둘 다 zero-regret였다. 즉 contract는 “실행 불가능한 action만 제거하는 안전장치”가 아니라, 이 fixture에서는 사실상 “정답 행동 집합”이었다.

그래서 random policy도 exact contract와 oracle STOP을 붙이면 모든 test task를 성공한다. 현재의 `metric_closed_loop=100%`는 learned geometry가 random보다 낫다는 증거가 아니다.

MARRVEL에서는 반드시 다음 상태가 있어야 한다.

- contract-valid이지만 정보 가치가 낮은 action
- contract-valid이지만 불필요하게 비싼 action
- contract-valid이지만 장기적으로 dead end를 만드는 action
- 서로 다른 DB release 또는 evidence source를 선택하는 action
- 실패·empty result 후 재시도 또는 대체 도구를 고르는 action

그래야 geometry가 “실행 가능성”이 아니라 “경로의 질”을 학습할 수 있다.

### 6.2 현재 closed-loop 차이는 주로 STOP 차이다

Commit-1 trace를 모두 합치면 다음과 같다.

| Backbone | 실행된 non-STOP action | Zero-regret non-STOP action | 조기 STOP | 성공 task-run |
|---|---:|---:|---:|---:|
| 1.5B | 65 | 65 | 2/9 | 7/9 |
| 7B | 44 | 44 | 5/9 | 4/9 |

Contract 뒤에서 실제 실행된 tool choice는 모두 zero-regret였고, 실패는 조기 STOP에서 발생했다. 따라서 `77.8% 대 44.4%`를 backbone 또는 geometry의 차이로만 읽으면 안 된다. 현재는 STOP head calibration 차이에 더 가깝다.

또한 비교 조건이 완전히 대칭이 아니다.

- Random/metric baseline: public-goal oracle STOP
- Flow: learned STOP

향후에는 planner와 STOP을 분리한 factorial evaluation이 필요하다.

| Action planner | STOP controller | Contract |
|---|---|---|
| Random | 동일 learned STOP | on/off |
| Metric | 동일 learned STOP | on/off |
| Flow first action | 동일 learned STOP | on/off |
| Random/Metric/Flow | public-goal oracle | on/off |

주 비교는 같은 STOP controller끼리 해야 하며, oracle STOP은 upper-bound 분석으로만 사용해야 한다.

### 6.3 피드백의 가치는 방향성 있게 보인다

- Open-loop: 0%
- 새 관측이 없는 compute-matched blind replan: 0%
- 새 state를 반영하는 commit-1: 1.5B 77.8%, 7B 44.4%
- Commit-2는 commit-1보다 낮음

이는 “계산을 반복해서”가 아니라 “실행 결과를 보고 다시 계획해서” 좋아졌을 가능성을 보여 준다. 다만 random-contract도 100%이고 test workflow가 3개뿐이므로, 아직 learned feedback policy의 인과적 우월성으로 부를 수는 없다.

---

## 7. 통계적으로 얼마나 유의미한가

### 7.1 효과 크기와 증거 강도를 분리해야 한다

Raw cosine에서 learned metric으로의 상승 폭은 synthetic prefix 기준으로 크다. 그러나 prefix들이 같은 task에서 반복 생성됐고 head의 parameter 수가 데이터보다 매우 크다.

| 모듈 | 1.5B 실험 | 7B 실험 |
|---|---:|---:|
| Bilinear metric head | 약 4.02M parameters | 약 5.08M parameters |
| Flow model | 약 1.26M parameters | 약 1.26M parameters |
| 독립 train task | 7 | 7 |
| Nonterminal train prefix | 53 | 53 |

직접적인 ID leakage를 막았더라도 template와 state-machine 암기 위험은 매우 크다.

### 7.2 Seed 3개는 새 환자 3명이 아니다

세 seed는 동일한 세 test task를 반복한다. 최종 성공을 9개의 독립 사례처럼 합치면 안 된다. 예를 들어 한 seed에서 3/3 성공의 Wilson 95% 구간도 대략 44%~100%다.

1.5B 대비 7B의 flow first-action 차이는 약 +3.85%p지만 seed 변동 안에 있다. 세 seed만으로는 통상적 5% 유의수준의 backbone 우월성을 주장할 수 없다.

### 7.3 현시점 증거 등급

| 주장 | 현재 증거 |
|---|---|
| 파이프라인 구현 및 재현 가능성 | 강함 |
| Raw embedding이 바로 action geometry는 아님 | 중간 이상 |
| 학습 모듈이 synthetic ranking을 개선 | 중간, 단 template confound 큼 |
| Geometry 선택이 결과에 영향을 줌 | 탐색적으로 유의미 |
| 좋은 metric이 좋은 flow로 자동 전이되지 않음 | 흥미로운 탐색적 negative result |
| Feedback가 open-loop보다 유리 | 방향성 신호 |
| Flow가 random/rule/BC보다 우월 | 증거 없음 |
| 실제 genomics 지식·일반화 | 증거 없음 |
| 임상적 유효성 | 평가하지 않음 |

---

## 8. 첨부 의견과의 통합 판단

첨부 의견의 핵심 결론에는 동의한다.

1. 현재 결과는 공학적 pipeline proof다.
2. `1.5B + bilinear + contract + commit-1`은 비용 대비 합리적인 baseline이다.
3. 7B, bilinear, STOP threshold 0.3을 보편적 최적값으로 고정하면 안 된다.
4. 새 외부 데이터에서 locked transfer와 새 train/dev 최적화를 분리해야 한다.
5. action과 STOP을 분리해 평가해야 한다.

단, “현재 설정을 다음 데이터셋에 그대로 적용해 보는 것이 타당하다”는 문장은 다음처럼 제한해야 정확하다.

> **실행 검증된 새 trajectory dataset에 사전 고정 baseline을 단 한 번 적용해 transfer를 측정하는 것은 타당하다. 비어 있는 MARRVEL QA queue에 현재 설정을 그대로 학습시키는 것은 타당하지 않다.**

또한 suffix 중복 수치는 정확히 확인하면 dev 12/15, test 19/23이며, 이는 현재 test가 workflow 일반화보다 interpolation에 가깝다는 해석을 강화한다.

---

## 9. 현재 MARRVEL 데이터의 실제 상태

복구 bundle의 MARRVEL revision은 다음과 같다.

```text
dataset_id: hjeong84/marrvel-mcp-benchmark-data
revision: 8e895924b1b19dc5ede19669bbf66c2d8d2eb5f0
rows: 100
```

카테고리 분포는 다음과 같다.

| Category | Rows |
|---|---:|
| Variant Prioritization | 34 |
| Gene and Variant Utilities | 20 |
| Gene-Phenotype Databases | 20 |
| Literature | 9 |
| Protein Interactions | 9 |
| Expression Databases | 5 |
| Ortholog Data | 3 |

100행 모두 다음 필드가 비어 있다.

- `initial_state`
- `goal`
- `private_verifier`
- `available_tools`
- `gold_tool_ids`
- `background_ids`
- `split_group`

따라서 이것은 현재 **학습 가능한 trajectory dataset이 아니라, QA에서 만든 human-curation queue**다.

100개는 synthetic 12개보다 크지만 ML 관점에서는 여전히 작다. 문자열 기준으로 NUTM2G가 26행, TTN이 8행, MECP2가 6행에 등장하고 질문 template도 반복된다. task ID만 무작위 분할하면 같은 entity와 template가 train/test에 함께 들어갈 가능성이 매우 높다.

---

## 10. MARRVEL을 어떤 방식으로 써야 하는가

### 10.1 세 가지 사용 방식

#### 가장 강한 방식: MARRVEL을 최종 외부 test로 보존

- 별도의 사례로 train/dev executable trajectory corpus를 만든다.
- MARRVEL 100문항은 모델·threshold를 고정한 후 한 번만 평가한다.
- 이 경우에만 비교적 깨끗한 external transfer 주장과 MARRVEL benchmark 성격을 유지할 수 있다.

#### 현실적인 인턴 프로젝트 방식: MARRVEL-derived corpus로 재구성

- 100행 중 20~30개를 먼저 pilot으로 큐레이션한다.
- 동일 entity/variant/template를 `split_group`으로 묶는다.
- grouped train/dev/test 또는 grouped cross-validation을 수행한다.
- 결과는 “공식 MARRVEL benchmark 성능”이 아니라 “MARRVEL-derived trajectory corpus 내부 일반화”로 표현한다.

#### 피해야 할 방식

- Expected answer에서 tool path를 자동 추측
- 100행 전체로 학습한 뒤 같은 100행 정확도를 benchmark 점수로 보고
- 관련 QA를 서로 다른 split에 무작위 배치
- live DB를 release 고정 없이 평가 중 호출

### 10.2 One-step QA와 multi-step episode를 분리한다

MARRVEL 문항 중 일부는 한 번의 lookup으로 끝날 수 있다. 이런 데이터는 다음 용도에는 좋다.

- tool routing
- action prototype 학습
- entity/background embedding 평가
- simple metric baseline

그러나 이것만으로는 flow trajectory 연구가 되지 않는다. Flow 연구용 주 데이터는 관련 QA와 실제 도구 의존관계를 case-level mission으로 재구성해야 한다.

예를 들어 하나의 variant case를 다음처럼 구성할 수 있다.

```text
raw HGVS
  → transcript/version 확인
  → assembly 및 reference 확인
  → normalization
  → consequence/gene annotation
  → gene–disease/phenotype evidence 조회
  → evidence integration
  → verified answer + STOP
```

이때 순서는 연구자가 expected answer를 보고 꾸며 내는 것이 아니라, pinned wrapper와 snapshot을 실제로 replay하고 reviewer가 승인해야 한다.

### 10.3 목표 데이터 단위 예시

아래는 현재 schema의 최종 형태가 아니라, 추가해야 할 의미를 보여 주는 예시다.

```json
{
  "task_id": "marrvel-case-000",
  "query": "What is the protein consequence of NM_001045477.4:c.187C>T?",
  "category": "Gene and Variant Utilities",
  "split_group": "variant:NM_001045477.4:c.187C>T|template:protein-consequence",
  "initial_state": {
    "raw_variant": "NM_001045477.4:c.187C>T",
    "transcript_accession": "NM_001045477",
    "transcript_version": 4,
    "assembly": null,
    "coordinate_system": "hgvs-c"
  },
  "goal": {
    "conditions": [
      {"field": "report.protein_change_verified", "op": "eq", "value": true}
    ]
  },
  "private_verifier": {
    "conditions": [
      {"field": "report.protein_change", "op": "eq", "value": "p.P63S"}
    ]
  },
  "available_tools": [
    "parse_hgvs",
    "resolve_transcript_version",
    "map_protein_consequence",
    "verify_consequence"
  ],
  "background_ids": ["hgvs-policy", "reference-release-policy"],
  "curation_status": "validated"
}
```

Expected answer는 model input에서 제외하고 private verifier에만 사용한다.

---

## 11. MARRVEL 전환 전에 필요한 코드 변경

### P0. `observed_only` selection loss의 unknown-negative 문제

현재 `src/geoflowagent/data/preprocess.py`는 미실행 applicable action에 대해 다음을 올바르게 설정한다.

```text
action_regret = null
action_regret_mask = false
action_outcome_status = unobserved
```

그러나 `src/geoflowagent/training/metric.py`의 selection loss는 여전히 모든 candidate를 denominator에 넣는다. `valid_next_tools`는 demonstrated gold action 하나뿐이므로, 실행하지 않았을 뿐 실제로는 유효할 수 있는 action도 암묵적 negative로 밀린다.

현재 개념은 다음과 같다.

```python
selection_loss = set_valued_nll(
    energy,
    valid_mask,
    candidate_mask,
)
```

필요한 방향은 별도의 supervision mask를 두는 것이다.

```python
selection_known_mask = candidate_mask & (
    regret_mask | valid_mask
)

selection_loss = set_valued_nll(
    energy,
    valid_mask & selection_known_mask,
    selection_known_mask,
)
```

구현 시 주의할 점은 다음과 같다.

- `contract_invalid`: known negative로 포함 가능
- `observed valid`: known positive로 포함
- `observed suboptimal`: 알려진 regret에 따라 ranking 가능
- `unobserved applicable`: selection denominator와 regret loss 모두에서 제외
- mask 후 positive/negative가 하나도 없는 row는 안전하게 loss 0 처리

이 수정만으로 applicable action 사이의 학습 문제가 모두 해결되지는 않는다. gold 하나밖에 관측하지 않았다면 다른 applicable action과의 상대 utility를 배울 수 없기 때문이다. 따라서 가능한 branch를 실제 실행하거나 reviewer가 복수 valid action을 표시해야 한다.

필요한 테스트:

- unknown applicable action의 energy를 바꿔도 selection loss가 변하지 않는지
- observed negative는 loss에 반영되는지
- 모든 known positive가 set-valued numerator에 들어가는지
- contract mask on/off에서 empty denominator가 발생하지 않는지

### P0. 단일 gold suffix를 multi-reference/partial-order로 확장

현재 flow는 한 row당 `gold_suffix_tool_ids` 하나를 target으로 사용한다. 병렬 실행 가능한 tool 순서가 여러 개면, 동등하게 좋은 경로 하나를 다른 정답의 negative처럼 학습하게 된다.

권장안은 다음 중 하나다.

1. `gold_suffixes: list[list[str]]`를 지원하고 epoch마다 승인된 reference를 sampling
2. Dependency DAG를 저장하고 topological order를 augmentation
3. 첫 action은 set-valued loss, 이후 suffix는 multi-reference minimum loss 사용
4. 충분한 데이터가 생기면 path distribution 자체를 mixture target으로 학습

Pilot에서는 1과 2가 가장 단순하고 해석 가능하다.

### P0. STOP을 action geometry와 분리

현재 가장 큰 end-to-end 실패 원인은 조기 STOP이다. 다음을 추가한다.

- terminal/nonterminal balanced STOP batch
- off-trajectory state의 STOP negative
- public goal 만족 여부를 직접 사용하지 않는 learned STOP baseline
- threshold를 grouped dev에서만 calibration
- STOP AUROC/AUPRC, terminal recall, premature-stop rate 별도 보고
- 모든 planner에 같은 STOP controller를 붙인 matched comparison

가능하면 STOP head는 flow slot decoding과 별도의 작은 binary classifier로도 비교한다.

### P1. Task-balanced sampling

긴 trajectory는 prefix가 많아 loss에서 더 큰 비중을 가진다. 각 row의 기본 가중치를 다음처럼 둔다.

```text
row_weight = 1 / number_of_prefixes_in_task
```

Category macro weighting도 별도로 비교한다. 주 통계 단위는 prefix가 아니라 task/case cluster로 유지한다.

### P1. View availability mask

MARRVEL 모든 문항에 DNA sequence가 있는 것은 아니다. DNABERT에 빈 문자열이나 gene symbol을 넣어서는 안 된다.

- 실제 assembly-specific sequence가 있을 때만 DNABERT view 활성화
- sequence source, assembly, strand, window를 typed metadata로 저장
- 없는 view는 zero vector가 아니라 explicit availability mask로 gate에서 제외
- SapBERT/MedCPT도 entity 또는 biomedical text가 없을 때 동일 처리

### P1. Runtime persistent embedding memoization

Gold prefix만 pre-cache하면 policy가 off-trajectory로 나간 순간 큰 encoder를 다시 부른다. 실제 MARRVEL 평가에서는 다음 key로 persistent cache를 둔다.

```text
hash(
  encoder_revision,
  serialization_version,
  query,
  typed_state,
  history,
  background_revision
)
```

Cache hit rate, miss context 수, cold/warm latency를 따로 보고한다.

### P1. Verifier와 snapshot provenance 강화

각 snapshot에는 최소한 다음을 저장한다.

- database/source name
- release 또는 retrieval timestamp
- wrapper git commit
- normalized input arguments
- assembly/transcript/accession version
- normalized output
- response checksum
- license/redistribution 여부
- reviewer와 validation status

Live endpoint는 데이터 생성 시에만 호출하고, 평가에는 가급적 versioned offline snapshot을 사용한다.

---

## 12. 파일별 변경 계획

| 우선순위 | 파일 | 변경 내용 |
|---|---|---|
| P0 | `data/curation/marrvel/marrvel_curation_queue.jsonl` | 20~30 pilot 행에 typed state, goal, verifier, tools, split group, reviewer 상태 입력 |
| P0 | `data/raw/geomarrvel/tasks.jsonl` | answer-free task와 entity/template/source grouping 저장 |
| P0 | `data/raw/geomarrvel/trajectories.jsonl` | single `tool_ids`에서 multi-reference 또는 dependency 정보로 확장 |
| P0 | `data/raw/geomarrvel/tools.jsonl` | 실제 wrapper schema, precondition, argument/output binding, source release 기록 |
| P0 | `data/raw/geomarrvel/snapshots.jsonl` | 실제 실행한 정규화 결과와 provenance 저장 |
| P0 | `data/raw/geomarrvel/verifiers.private.jsonl` | expected answer를 model input과 분리한 typed verifier 저장 |
| P0 | `src/geoflowagent/data/preprocess.py` | unknown action용 selection supervision mask, multi-reference expansion 구현 |
| P0 | `src/geoflowagent/models/functional.py` | known-label subset만 사용하는 set-valued NLL 지원 |
| P0 | `src/geoflowagent/training/features.py` | supervision mask, task weight, view availability 읽기 |
| P0 | `src/geoflowagent/training/metric.py` | observed-only loss 수정, task-balanced loss, macro metric 추가 |
| P0 | `src/geoflowagent/training/flow.py` | multi-reference target, partial-order augmentation, STOP 분리 평가 |
| P0 | `src/geoflowagent/evaluation/agent.py` | planner×STOP×contract matched conditions, rule/FSM baseline 추가 |
| P1 | `src/geoflowagent/data/marrvel.py` | validated row에 실행 trace/reviewer/provenance를 필수화하고 자동 path 추측 금지 유지 |
| P1 | `src/geoflowagent/embeddings/runtime.py` | persistent off-trajectory memoization과 cold/warm latency 분리 |
| P1 | `data/schemas/*.schema.json` | multi-reference, label source, unknown mask, reviewer, provenance schema 추가 |
| P1 | `tests/test_preprocess.py` | unknown-negative regression test, split group leakage test 추가 |
| P1 | `tests/test_flow.py` | equivalent suffix permutation 및 multi-reference target test 추가 |
| P1 | `tests/test_agent_episode.py` | matched STOP, contract-only saturation, failure recovery test 추가 |
| P1 | `configs/marrvel_transfer_locked.yaml` | synthetic 선택을 그대로 적용하는 one-shot transfer baseline |
| P1 | `configs/marrvel_pilot_small.yaml` | real dev에서 선택할 작은 모델용 config |
| P2 | `docs/DATA.md` | label source와 multi-reference curation 규칙 문서화 |
| P2 | `docs/EXPERIMENTS.md` | task-level 통계, locked transfer, test sealing 프로토콜 추가 |

---

## 13. Config는 어떻게 바꿔야 하는가

현재 `configs/geomarrvel_mvp.yaml`은 바로 실행하면 안 된다.

현재 설정의 대략적인 학습 parameter 수는 다음과 같다.

- Metric module: 19,907,333
- Flow module: 24,826,369

100 task에서 생기는 수백 개의 상관된 prefix로 학습하기에는 지나치게 크다. 또한 flow 설정은 synthetic에서 best epoch 1로 실패했던 초기 baseline 계열과 가깝다.

```yaml
# current geomarrvel_mvp.yaml의 주요 값
metric_training:
  shared_dim: 768
  hidden_dim: 1024
  distance: lowrank_mahalanobis
  distance_rank: 256

flow_training:
  hidden_dim: 1536
  layers: 4
  heads: 12
  learning_rate: 0.0001
  tool_weight: 0.5
  stop_threshold: 0.5
```

### 13.1 Locked transfer config

목적은 “synthetic에서 정한 선택이 real trajectory로 전이되는가?”를 보는 것이다. 이 config는 MARRVEL dev를 보고 조정하지 않는다.

아래 YAML은 **변경 후 목표를 나타내는 설계 초안**이다. `experiment_role`, `commit_horizon`, `contract_mask` 같은 일부 key는 현재 config loader가 바로 소비하지 않으므로, 실제 파일을 만들 때 기존 schema와 CLI 옵션에 맞춰 구현·검증해야 한다.

```yaml
experiment_role: locked_external_transfer

metric_training:
  distance: bilinear
  shared_dim: 256
  hidden_dim: 256
  learning_rate: 0.001
  batch_size: 16
  epochs: 300
  patience: 60

flow_training:
  hidden_dim: 256
  layers: 2
  heads: 8
  learning_rate: 0.001
  batch_size: 16
  epochs: 300
  patience: 60
  noise_scale: 0.25
  tool_weight: 2.0
  stop_threshold: 0.3

agent_evaluation:
  commit_horizon: 1
  contract_mask: true
```

이 값은 보편적 최적값이 아니라 transfer 측정을 위한 사전 고정 baseline이다.

### 13.2 Small pilot search config

새 MARRVEL train/dev에서 다시 선택할 실용적인 시작 범위다.

이 역시 P0/P1 코드 변경 후 사용할 설계 초안이다. `task_balanced`, `observed_only_selection_mask`, `multi_reference`는 현재 코드에 아직 구현되지 않은 목표 기능이다.

```yaml
experiment_role: marrvel_internal_model_selection

metric_training:
  structured_dim: 64        # 또는 128
  shared_dim: 128           # 128, 256 비교
  hidden_dim: 256
  distance_rank: 32         # 16, 32, 64 비교
  distance: bilinear        # cosine/euclidean/lowrank도 dev 비교
  learning_rate: 0.0003     # 3e-4, 1e-3 비교
  task_balanced: true
  observed_only_selection_mask: true

flow_training:
  hidden_dim: 256           # 128, 256 비교
  layers: 2                 # 1, 2 비교
  heads: 4
  learning_rate: 0.0003     # 3e-4, 1e-3 비교
  noise_scale: 0.25
  tool_weight: 2.0          # 1, 2 비교
  multi_reference: true
  task_balanced: true
  stop_threshold: null      # grouped dev에서 별도 calibration
```

7B는 주 모델이 아니라 embedding/backbone ablation으로 남긴다.

---

## 14. 권장 실험 설계

### Track A. 사전 고정 transfer

연구 질문:

> Synthetic에서 선택한 1.5B-bilinear geometry와 commit-1 policy가 새 real trajectory에서 tuning 없이 어느 정도 유지되는가?

규칙:

- 실행 검증된 외부 pilot test 사용
- model family, dimension, STOP threshold를 미리 고정
- 한 번 평가 후 재튜닝 금지
- 결과가 낮아도 그대로 보고

이 track은 진짜 transfer를 측정한다.

### Track B. MARRVEL-derived model selection

연구 질문:

> Real genomics workflow에서는 어떤 representation과 geometry가 action ranking 및 trajectory 품질에 가장 적합한가?

규칙:

- entity/template/source grouped split
- train/dev에서만 geometry, dimension, STOP 선택
- 별도 untouched test 한 번 개봉
- task-level bootstrap과 category macro 평균 사용

이 track은 최적화 가능성을 측정하지만 official external benchmark라고 부르지 않는다.

### Track C. Geometry–trajectory alignment

이번 결과에서 가장 독창적으로 발전시킬 수 있는 연구 축이다.

연구 질문:

> Local next-action geometry의 품질과 global trajectory vector field의 품질이 왜 어긋나며, 어떤 joint objective가 둘을 정렬하는가?

비교할 objective 예시는 다음과 같다.

- Action-only metric loss
- Transition prediction loss
- Suffix endpoint loss
- Path length/curvature regularization
- Regret-weighted flow matching
- Multi-reference set loss
- Feedback rollout consistency loss

주 결과는 단순 accuracy보다 다음 관계로 제시한다.

```text
local geometry quality
  → first-action quality
  → closed-loop recovery
  → final verified task success
```

---

## 15. 반드시 포함할 baseline과 ablation

### 15.1 Planner baseline

- Random among all tools
- Random among contract-valid tools
- Deterministic contract/FSM 또는 BFS planner
- Most-frequent tool
- Query-only linear classifier
- Structured-state-only MLP
- Autoregressive small Transformer/behavior cloning
- Learned metric nearest-action
- Flow open-loop
- Flow commit-1
- Flow commit-2
- Compute-matched blind replan

### 15.2 Representation ablation

- Qwen-only
- MedCPT-only
- SapBERT-only where applicable
- DNABERT-only where actual sequence exists
- Structured-only
- Qwen + structured
- All views
- Background removed
- Exact genomics field removed one at a time
- Embedding shuffled control

### 15.3 Contract/STOP ablation

- Contract on/off
- Oracle STOP vs 동일 learned STOP
- Separate STOP classifier vs flow STOP head
- Threshold calibration curve
- Contract-valid candidate count별 성능
- Contract-valid이지만 regret가 다른 state만 따로 평가

### 15.4 Robustness

- GRCh37/GRCh38 swap
- Transcript version 변경
- Coordinate convention 변경
- Reference allele mismatch
- Alias ambiguity
- Empty database result
- Tool timeout/failure
- Schema field 누락
- Stale database release
- Noisy observation
- Off-trajectory state

---

## 16. 평가 지표

### 16.1 Embedding/geometry

- Raw and whitened valid-hit@1/@3
- MRR
- Task-grouped linear probe
- Anisotropy/effective rank
- Minimal-pair separation with confidence interval
- Entity/template-held-out retrieval
- Regret–distance rank correlation on known labels only

### 16.2 Action과 STOP

- Valid first-action accuracy
- Single-reference exact accuracy
- Set-valued accuracy
- Contract-valid action rate
- Zero-regret action rate
- STOP AUROC/AUPRC
- Terminal recall
- Premature STOP rate

### 16.3 Trajectory

- Multi-reference suffix exact match
- Normalized edit distance
- Final accumulated regret/cost
- Unnecessary tool calls
- Tool failure recovery
- Open-loop vs feedback gap
- Commit horizon sensitivity

### 16.4 최종 task

- Private verifier task success
- Answer exact/F1 or typed field accuracy
- Category macro-average
- Entity/template/source held-out result
- Task-level bootstrap 95% interval
- Latency, NFE, online embedding miss

주 결과 표에는 prefix micro-average보다 task/case macro-average를 먼저 둔다.

---

## 17. 20~30 task pilot의 데이터 요건

각 task는 다음 조건을 모두 만족해야 한다.

- [ ] Typed initial state가 완전하다.
- [ ] Public goal에 expected answer가 노출되지 않는다.
- [ ] Private verifier가 typed field로 정답을 검증한다.
- [ ] 사용 가능한 tool과 schema가 실제 wrapper와 일치한다.
- [ ] 모든 gold step이 offline snapshot으로 replay된다.
- [ ] DB/tool/model revision이 고정돼 있다.
- [ ] 같은 entity/variant/template가 하나의 split group이다.
- [ ] Reviewer가 trajectory와 verifier를 이중 검수했다.
- [ ] 가능한 경우 복수의 승인된 경로가 기록된다.
- [ ] 최소 일부 state는 여러 contract-valid 행동의 utility가 다르다.
- [ ] Empty/failure/ambiguity branch가 포함된다.
- [ ] 실제 sequence가 없으면 DNA view를 비활성화한다.

Pilot이 모든 MARRVEL category를 억지로 동일하게 포함할 필요는 없다. 먼저 다단계 dependency가 명확한 category를 선택하고, one-step QA는 별도 routing subset으로 관리한다.

---

## 18. Go/No-Go 기준

아래 수치는 현재 관측 결과가 아니라 다음 단계의 권장 판정 기준이다.

### 데이터 준비 Gate

- 모든 pilot task가 snapshot replay와 private verifier를 통과
- split-group 충돌 0건
- expected answer의 model input 노출 0건
- unknown counterfactual의 selection/regret loss 참여 0건
- contract-valid set과 zero-regret set이 전체 state에서 항상 같지 않음
- 여러 실행 가능 action의 utility가 다른 독립 decision state를 충분히 확보

### 모델 연구 Gate

- learned model이 random-contract 및 structured-only baseline보다 task-level로 개선
- rule/FSM baseline과의 차이를 보고
- first-action 개선이 STOP controller 차이만으로 설명되지 않음
- geometry 선택이 최소 두 개 grouped fold에서 같은 경향
- minimal-pair와 perturbation에서 exactness가 유지
- no-mask 결과가 chance 수준이라면 contract 의존 연구임을 명시

### Full 100 확장 Gate

20~30 pilot에서 다음이 충족될 때만 전체 큐레이션으로 확대한다.

- 데이터 제작자 간 합의율과 오류율이 관리 가능
- snapshot/replay pipeline이 안정적
- contract-only baseline이 포화되지 않음
- model이 최소 하나의 단순 learned baseline을 일관되게 이김
- task-level 불확실성을 보고할 수 있을 만큼 독립 case가 확보됨

Random-contract가 learned model과 같은 수준이면 모델을 키우지 말고 task 설계를 먼저 바꾼다.

---

## 19. 실행 순서

### 단계 0. 현재 synthetic 결과 동결

- 현재 test는 이미 unseal됐으므로 추가 tuning에 사용하지 않는다.
- 현재 결과를 `synthetic-v1` frozen baseline으로 기록한다.
- 1.5B/7B 결과와 checksum을 보존한다.

### 단계 1. 재현성 bookkeeping 정리

- 모든 run에 실제 git commit과 dirty flag 기록
- 동일 multi-seed sweep을 같은 source-tree hash에서 다시 실행
- run-state의 cache array 수 `28`과 실제 `.npy` 27개 불일치 정정
- dev-only summary와 final all-split embedding summary를 분리 표기
- `test_sealed`는 시간 상태가 아니라 `sealed_at_selection`, `unsealed_at_final`처럼 사건으로 기록

Seed 17의 초기 distance sweep source-tree hash가 seed 41/73과 다르다. 최종 bilinear seed checkpoint는 같은 source tree이지만, 논문용 distance-family 비교는 동일 commit에서 재실행하는 것이 안전하다.

### 단계 2. MARRVEL 20~30행 선정

- Multi-step 가능성
- Category 다양성
- Entity 중복
- Source 안정성
- Snapshot 라이선스

을 기준으로 선정한다.

### 단계 3. 실제 실행·snapshot·이중 검수

- QA expected answer에서 path를 역추론하지 않는다.
- 실제 tool wrapper를 실행한다.
- 정규화 snapshot과 실패 branch를 저장한다.
- 두 reviewer가 verifier와 trajectory를 승인한다.

### 단계 4. P0 코드 변경 및 테스트

- observed-only selection mask
- multi-reference/partial-order
- STOP matched evaluation
- task-balanced metrics

### 단계 5. Embedding cache 생성

우선 1.5B를 사용하고 7B는 일부 ablation에만 사용한다. Frozen weight보다 cache 생성 시간이 들지만 저장 용량은 큰 병목이 아니다.

대략적인 단순 비례 추정은 다음과 같다.

- 400~800 prefix, 1.5B cache: 약 12~25 MB
- 400~800 prefix, 7B cache: 약 20~41 MB
- 실제 비용은 history 길이와 off-trajectory online miss에 따라 증가

### 단계 6. Track A locked transfer

- 한 번 실행
- tuning 금지
- 실패 포함 그대로 보고

### 단계 7. Track B train/dev 선택

- small config
- grouped folds
- baseline/ablation
- threshold calibration은 dev만 사용

### 단계 8. Untouched test 평가

- config와 checkpoint hash를 먼저 고정
- test 한 번 실행
- task-level CI와 category macro 보고
- test 결과를 본 뒤 같은 test에 재튜닝하지 않음

### 단계 9. Full 100 여부 결정

Pilot에서 dataset이 Flow 연구 질문을 실제로 식별할 수 있을 때만 확장한다.

---

## 20. 가장 권장하는 연구 프레이밍

사용자의 trajectory 관점과 현재 결과를 가장 잘 연결하는 질문은 다음이다.

> **Frozen general/domain embeddings가 genomic tool workflow의 상태 전이를 어느 정도 표현하며, learned functional geometry가 exact genomic constraints와 observation feedback 아래에서 장기적으로 더 좋은 trajectory 선택을 만들 수 있는가?**

세부 가설은 다음과 같이 둔다.

- H1: Raw embedding보다 learned functional geometry가 entity/template-held-out action ranking을 개선한다.
- H2: Typed exact fields는 dense embedding이 놓치는 build/version 차이를 보존한다.
- H3: Contract-valid action 사이에 utility 차이가 있을 때 learned geometry가 random/rule baseline보다 regret를 줄인다.
- H4: Commit-1 feedback가 open-loop와 blind replan보다 failure recovery를 개선한다.
- H5: Local metric 성능만으로 global flow 성능을 예측할 수 없으며 joint trajectory objective가 필요하다.
- H6: 더 큰 frozen backbone보다 data/geometry/feedback 정렬이 더 중요할 수 있다.

이 프레이밍의 장점은 7B가 이기지 않거나 Flow가 단순 baseline보다 약해도 연구가 실패하지 않는다는 점이다. 어떤 representation과 제약이 경로 성능을 만드는지 분해하는 것 자체가 연구 결과가 된다.

---

## 21. 현재 말할 수 있는 것과 말하면 안 되는 것

### 말할 수 있는 것

- 실제 frozen multiview encoder와 immutable cache pipeline이 작동했다.
- Raw embedding은 synthetic action selection에 적합한 거리 공간이 아니었다.
- 학습 가능한 representation/geometry는 workflow-stage signal을 추출했다.
- Geometry의 metric 순위와 flow 순위가 다를 수 있었다.
- Closed-loop feedback는 open-loop보다 좋은 방향성 신호를 보였다.
- 7B는 1.5B보다 일관되게 우수하지 않았다.

### 아직 말하면 안 되는 것

- Bilinear가 genomics의 최적 geometry다.
- FlowAgent가 random, rule, BC planner보다 우월하다.
- 모델이 GRCh37/GRCh38 의미를 embedding 자체로 이해한다.
- 77.8%가 실제 genomics task 성공률이다.
- 현재 결과가 MARRVEL 또는 임상 데이터로 일반화된다.
- 7B보다 1.5B가 통계적으로 우수하다.

---

## 22. 최종 권고

현재 프로젝트에서 가장 먼저 할 일은 backbone을 더 키우는 것이 아니다.

1. **Contract가 정답을 대신 주지 않는 real decision state를 만든다.**
2. **MARRVEL QA를 실행 가능한 typed trajectory로 큐레이션한다.**
3. **Observed-only unknown-negative 문제와 단일 suffix 문제를 고친다.**
4. **STOP을 action planner와 분리해 공정하게 비교한다.**
5. **작은 1.5B baseline으로 20~30 task pilot을 먼저 수행한다.**
6. **Metric 성능보다 trajectory regret와 verified task success를 주 지표로 둔다.**
7. **모든 결론을 task/entity/template 단위로 통계화한다.**

한 문장으로 요약하면 다음과 같다.

> **“시스템은 작동한다”는 검증은 통과했지만, “학습된 geometry와 flow가 실제 genomics trajectory를 더 잘 선택한다”는 가설 검증은 이제부터 시작이다.**

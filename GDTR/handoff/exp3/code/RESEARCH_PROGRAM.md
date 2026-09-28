# EXP3 연구 프로그램

## 1. 이 논문의 한 문장

EXP2가 확정한 `b28 writer → b30 re-encoder → x31 readout` 구조 위에서, EXP3는 **late stack이 scale, carrier, content를 서로 다른 방식으로 운반하고 사용하는 규칙**을 규명하고, 그 규칙이 서열·motif·variant·benchmark·실용적 intervention을 예측한다는 것을 보인다.

즉 이 연구의 중심 산출물은 또 하나의 EXP2 확인표가 아니다. 고정된 EXP2 관점을 입력으로 받아 **어디를 측정하고, 무엇을 역추적하며, 어느 지점을 개입해야 하는지 결정하는 재사용 가능한 방법론**과 그 방법론의 prospective validation이다.

핵심은 “late layer가 정보를 비운다”가 아니다. 더 정확한 질문은 다음과 같다.

> 이미 b28에서 쓰인 방향성 content가 b29에서 어떻게 증폭되거나 수정되고, b30에서 어떻게 출력에 맞게 재부호화되며, x31 carrier와 결합해 최종 readout으로 드러나는가?

## 2. 연구 경계: 확정된 기반과 후속에서 규명·활용할 것

### 2.1 EXP2에서 확정하여 가져오는 것

- b28은 causal content writer/transform이다.
- b30은 그 content를 출력에 맞게 re-encode/consolidate한다.
- x31은 late-stack 결과가 최종 출력으로 드러나는 readout gate다.
- 방향 변화가 raw norm 변화보다 정보성이 높다는 것이 GDTR의 이유다.
- GDTR, TDIG, After_GDTR, Handoff는 같은 late-stack 관점으로 해석할 수 있다.

EXP3는 이 다섯 문장을 다시 검정하지 않는다. 따라서 b29 또는 scale 실험이 예상과 다르더라도 “EXP2가 틀렸다”로 돌아가지 않는다. 대신 **확정된 stage scaffold 위에 어떤 EXP3 세부 모델을 올려야 하는가**를 결정한다.

### 2.2 EXP3가 새로 규명하고 활용할 것

1. b29 subtype: amplifier, second writer, editor, mixed, bounded-null 중 무엇인가?
2. scale boundary: 여러 block/model의 전이가 `q = alpha ||update|| / ||host||`로 정렬되는가?
3. factorization: scale·content·carrier를 바꾸었을 때 출력 효과가 독립적인가, 상호작용하는가?
4. reverse trace: 출력 방향을 입력 서열까지 역추적할 수 있는가?
5. motif causality: 역추적 후보가 실제 serial path를 타고 효과를 만드는가?
6. learned transition: 고정된 causal delta를 held-out에서 예측하고 실제 rescue할 수 있는가?
7. use: 기전 지표가 benchmark 차이와 오류 위치를 설명하고, repair·compression·steering에 쓰이는가?
8. robustness: chromosome, task, context, checkpoint, 학습 시점이 달라도 무엇이 보존되는가?
9. optional interpretation: 원시 인과 결과 이후 SAE/transcoder가 설명 가능한 vocabulary를 제공하는가?

## 3. 공통 설계 원칙

### 3.1 하나의 표본 단위

토큰 위치를 독립 표본으로 세지 않는다. 동일 genomic window, variant family, donor family, motif family에 속한 관측은 같은 dependency cluster로 취급한다. 신뢰구간과 bootstrap은 cluster 단위로 계산한다.

### 3.2 데이터 역할 분리

기본 역할은 다음과 같다.

| 역할 | 기본 chromosome | 허용되는 일 |
|---|---|---|
| discovery | chr22 | U28 안정화, 후보 방향·motif 생성, 모델 형태 제안 |
| development | chr21 | rank, threshold, dose 해상도, 복잡도 선택 |
| locked | chr17 | 사전에 봉인한 판정만 1회 평가 |
| external | 독립 assay | label 공개 전 봉인한 예측 평가 |

chr17을 이미 보며 선택했다면 다른 untouched chromosome을 locked로 지정한다. few-shot 학습을 쓰는 경우에도 train/development 데이터와 locked family가 겹치면 안 된다. 비교 방법은 모두 같은 sampled families와 label budget을 사용한다.

### 3.3 측정과 인과를 구분

- cosine, norm, probe, attribution은 관찰 또는 후보 생성이다.
- block은 necessity를 묻는다.
- only 또는 held-out predicted insertion은 sufficiency를 묻는다.
- block 후 rescue는 같은 성분이 실제 원인인지 묻는다.
- wrong-layer, random-direction, dose-matched, GC/k-mer/token-matched control은 대체 설명을 제거한다.

인과 주장은 최소한 `효과 + block + held-out rescue + specificity control`의 결합으로 제한한다.

### 3.4 판정은 pass/fail이 아니다

각 새 EXP3 질문은 `supported`, `equivalent`, `mixed`, `unresolved`, `refuted` 중 하나로 기록한다. 다섯 결과 모두 유효한 결론이다.

- `supported`: 사전 정의한 효과·복구 기준의 신뢰구간을 통과
- `equivalent`: 전체 신뢰구간이 사전 동등성 범위 안에 존재
- `mixed`: family, task, site, checkpoint에 따라 일관되게 다른 regime
- `unresolved`: power/식별성이 부족해 방향을 결정할 수 없음
- `refuted`: 해당 EXP3 가설이 관측과 양립하지 않음

과학적 비지지는 뒤 실험을 전역 차단하지 않는다. 그 결과와 양립하는 target과 분석으로 바꾼다. 반면 hash 불일치, leakage, NaN, 잘못된 tap처럼 결론 자체를 믿을 수 없으면 그 데이터 계통만 hard-fail한다.

## 4. 사전 준비: EXP2→EXP3 handoff

실험 전에 `Exp2Handoff`를 봉인한다.

반드시 포함할 내용은 다음과 같다.

- checkpoint, architecture, EXP2 code SHA-256
- `writer=28`, `b29_candidate=29`, `reencoder=30`, `readout_gate=31`
- 다섯 EXP2 foundation을 모두 `true`로 한 선언
- discovery/development에서 만든 `u28`, `carrier_axis_x31`,
  `b30_subspace`, `reference_directions` 네 frozen tensor와 각 semantic role
- effect, equivalence, rescue, specificity margin
- 각 artifact의 source split, source role, SHA-256

이 단계는 EXP2를 재실행하는 단계가 아니다. 이미 확정한 결과 중 EXP3가 사용할 최소 객체를 추적 가능하게 인계하는 단계다.

## 5. Phase 1 — m28 preparation, b29 subtype, scale boundary

### 5.1 실험 1A: m28이 고정된 b28 writer를 어떻게 준비하는지 규명한다

#### 질문

EXP2에서 b28의 writer 역할은 이미 확정되었다. 여기서는 그 결론을 다시 묻지 않고, b28에 들어오는 `m28`이 bilinear factor와 `g28` causal write를 어떤 방식으로 준비하는지 묻는다.

#### 조건과 판정

1. native
2. `m28=0` 뒤 b28 factor·product·g28·output 재계산
3. `m28=0`이지만 natural g28을 정확히 clamp한 path-isolation 진단
4. `m28=0` 뒤 development에서 학습한 held-out predicted g28 delta rescue

`m28=0`이 bilinear product와 output을 손상하고, exact clamp가 경로를 격리하며, target delta를 보지 않은 predictor의 g28 rescue가 locked output을 복구해야 “m28 preconditions the fixed b28 write”라고 말한다. exact natural clamp만으로는 sufficiency를 주장하지 않는다.

효과가 family별로 다르면 conditional/redundant input map으로, bounded null이면 m28-specific contribution의 상한으로, 불확실하면 upstream perturbation과 표본 수 개선으로 이어간다. 어느 경우에도 b28 writer라는 EXP2 결론은 다시 판정하지 않는다.

### 5.2 실험 1B: b29가 amplifier인지 분해한다

#### 질문

b29 반응 `g29`가 b28의 고정 causal subspace `U28`를 따라 기존 content를 증폭하는가, 아니면 `U28` 밖에 독립 정보를 쓰는가?

#### 측정

각 paired locus에서 intervention-induced b29 delta를 다음처럼 나눈다.

```text
g29_parallel = P_U28 g29
g29_perp     = (I - P_U28) g29
```

`U28`은 discovery에서 고정하고 locked에서 다시 맞추지 않는다. 전체 residual state가 아니라 intervention-induced delta를 분해한다. 그렇지 않으면 native background와 causal response가 섞인다.

#### 조건

동일 locus와 동일 output endpoint에서 다음을 비교한다.

1. native/free
2. 전체 b29 response 제거
3. parallel만 제거
4. perpendicular만 제거
5. parallel만 남김
6. perpendicular만 남김
7. 전체 제거 후 held-out predicted parallel rescue
8. 전체 제거 후 held-out predicted perpendicular rescue
9. 전체 제거 후 held-out predicted full rescue
10. 독립 sequence event가 만든 perpendicular response 제거
11. parallel dose scan

실제 delta를 그대로 캐시했다가 넣는 조건은 manipulation 확인에는 쓸 수 있지만 primary sufficiency 증거로 쓰지 않는다. 주된 rescue는 다른 split에서 학습하거나 사전 정의한 predictor가 만든 delta여야 한다.

#### amplifier 판정

다음이 함께 성립해야 한다.

- parallel 제거가 native output effect를 유의하게 손상
- predicted parallel만으로 효과가 나타남
- 전체 block 뒤 predicted parallel이 효과를 복구
- perpendicular energy와 output effect가 동등성 범위 안에서 작음
- 독립 event도 perpendicular mode를 사용하지 않음
- parallel dose가 충분한 q 이후 plateau

#### 대안 결론

| 결과 | 결론 | 다음 분석 |
|---|---|---|
| parallel necessity·sufficiency·rescue, perp null | amplifier | 다른 chromosome/task에서 amplifier replication |
| perp가 크고 독립 event·output에 영향 | second writer | perp direction과 motif를 별도 mapping |
| parallel과 perp가 모두 필요 | mixed writer/amplifier | 두 target을 분리한 branch-specific 분석 |
| 기존 방향을 회전·억제 | editor | 입력 조건별 편집 규칙과 signed effect 분석 |
| 전체 효과가 equivalence bound 안 | bounded-null | b29는 필수 stage가 아닌 조건부 modulator로 보고 상한 보고 |
| CI가 너무 넓거나 U28 불안정 | unresolved | paired cluster 수, U28 rank 안정성, dose 해상도만 개선 |

어느 결과도 EXP2의 b28 writer와 b30 re-encoder를 취소하지 않는다.

### 5.3 실험 1C: scale plateau의 실제 경계를 측정한다

#### 질문

왜 GDTR처럼 방향 중심 지표가 잘 작동하는가? update가 host에 비해 충분히 커지면 크기의 추가 증가가 거의 영향을 주지 않고, 방향 변화가 계속 중요하기 때문인가?

#### 설계

- `alpha = 1e-5 ... 1`의 21개 log-spaced dose
- 7B의 `g28/r28`, `g29/r29`, `m30/x30`
- 비교 모델의 대응 site. 기본 비교는 `evo2_1b_base`
- 각 radial dose와 같은 norm을 가진 angular control
- 같은 locus, 같은 endpoint, 같은 intervention으로 paired comparison

각 site에서 다음을 계산한다.

```text
alpha* = ||host|| / ||update||
q      = alpha / alpha* = alpha ||update|| / ||host||
```

raw alpha가 아니라 q 축에서 path fraction, output geometry, angle, beta/shape의 curve를 비교한다.

#### 공통 plateau 판정

- alpha*가 모든 유효 locus에서 유한하고 양수
- q가 정의와 정확히 일치
- high-q 마지막 증가량의 CI가 동등성 범위 안
- 각 site의 radial curve가 plateau
- equal-norm angular control과 radial scaling이 구별됨
- q로 정규화한 site/model curve의 RMSE가 사전 margin 이하

#### 대안 결론

| 결과 | 해석 | 다음 분석 |
|---|---|---|
| q curve collapse | shared dominance transition | held-out site/model transfer |
| block/model마다 임계점이 다름 | site-specific transition | site별 threshold와 구조 변수를 모델링 |
| radial과 angular가 비슷함 | scale-direction 분리 부족 | interaction 또는 measurement definition 점검 |
| plateau가 없음 | nonlinear continuous transport | saturation/curvature/host-update interaction 분석 |
| 전이 구간 CI가 넓음 | unresolved | 전이점 주변 dose와 반복만 추가 |

이 실험은 EXP2의 “방향이 중요하다”를 재검정하지 않는다. 그 확정된 현상에 대해 **언제 scale이 포화되어 방향이 판별력을 갖는지** 경계를 설명한다.

## 6. Phase 2 — scale·content·carrier 인과 분해

Phase 1의 세 결과에 맞춰 target을 정한다. 각 결과는 먼저 결과별 refinement와
`m28_target_spec`·`b29_target_spec`·`scale_target_spec`으로 봉인되고, 세 target이
모두 준비된 뒤에만 `exp3_mechanism_synthesis`가 열린다. m28 preparation은
upstream trigger를, amplifier이면 U28-parallel content를, second writer/editor이면
별도 b29-perp 또는 회전 delta를 정의하며, mixed이면 target을 분리한다.

### 6.1 2×2×2 causal cube

세 요소를 독립적으로 바꾼다.

- scale: native vs dominance-matched/controlled scale
- content: reference vs motif/variant/direction-swapped content
- carrier: native vs x31 carrier swapped/clamped/removed

모든 cell에서 같은 output endpoint를 측정하고 main effect와 interaction을 cluster bootstrap으로 계산한다.

예상되는 역할 구분은 다음과 같다.

- scale은 “얼마나 지배적으로 전달되는가”를 바꾼다.
- content는 “무엇을 전달하는가”와 signed output을 바꾼다.
- carrier는 이미 정해진 content가 readout에 노출되는 calibration을 바꾼다.

단, 이 문장은 결과가 아니라 검정 대상이다. interaction이 크면 독립 채널이라고 단정하지 않고 `content × carrier` 또는 `scale × content` 조건부 모델로 바꾼다.

### 6.2 b30 mediation과 방향 transfer

- b28/b29 target을 바꿔 자연스럽게 생긴 `delta m30`을 측정
- b30에서 해당 projected delta를 block
- held-out predicted `delta m30`으로 rescue
- wrong direction, random direction, wrong layer, norm-matched control 비교
- reference→alternate와 alternate→reference 양방향 transfer

“방향이 닮았다”만으로는 충분하지 않다. transfer가 output effect를 만들고, b30 block으로 사라지고, predicted b30 delta rescue로 돌아와야 한다.

causal cube, b30 mediation, 양방향 transfer의 세 판정이 모두 끝나면
`causal_factorization_synthesis`에서 공통·조건부·bounded-null·대안 transport
architecture 중 하나를 봉인한다. 역추적과 모든 활용 실험은 이 합성 판정 이후에
시작한다. 이는 양성 결과만 통과시키는 gate가 아니라, 어떤 유효한 결과가 나와도
그 결과에 맞는 operational target을 먼저 고정하기 위한 순서 제약이다.

### 6.3 이 단계가 benchmark 차이를 설명하는 방식

각 benchmark/task family에 대해 같은 기전 fingerprint를 만든다.

- b28 content write 크기와 방향 안정성
- b29 parallel/perp 비율과 gain
- scale q와 plateau 위치
- b30 mediation/rescue fraction
- x31 carrier sensitivity
- off-path leakage

성능 차이를 단순히 “task 난이도”로 설명하지 않고, 어떤 task가 content discrimination, scale dominance, carrier calibration, long-context routing을 더 요구하는지 예측한다. family 하나를 통째로 뺀 leave-family-out 평가로 새 family의 성능 또는 실패 stage를 예측한다.

## 7. Phase 3 — output에서 sequence로 역추적

### 7.1 시작점

task label로 방향을 새로 찾지 않는다. Phase 2에서 output-causal하다고 확인된 delta를 시작점으로 사용한다.

1. carrier axis를 제거한 b30 intervention delta를 unit-normalize
2. discovery-only SVD direction bank 구성
3. development에서 rank 선택
4. locked output intervention으로 선택된 방향 검증
5. b30→b29→b28→sequence로 역전파

### 7.2 서로 독립적인 후보 생성법

한 방법의 artifact를 motif로 착각하지 않도록 최소 두 방법의 합의를 요구한다.

- exact b28 bilinear contribution: `(W1 z) * (W2 z)` 채널이 고정 output 방향에 기여하는 양
- path-restricted ISM: 고정 path score를 대상으로 한 single-base mutation
- hierarchical window scan: 큰 window에서 시작해 효과가 있는 구간만 세분화
- local adjoint/Jacobian: 사전에 고정한 국소 선형화의 역방향
- matched k-mer enrichment: 후보 생성 보조 수단일 뿐 causal 증거가 아님
- donor-free natural occurrence scan: 같은 방향을 자연적으로 강하게 만드는 서열 탐색

후보는 candidate-method consensus를 통과해야 Phase 4로 간다. 합의가 없으면 “motif 없음”이 아니라 “현재 target에서 sequence attribution이 식별되지 않음”으로 기록하고 rank/window/target 정의를 좁힌다.

### 7.3 완전한 reverse chain 점수

다음 네 링크를 각각 held-out `0..1` strength로 정규화한다.

- output→b30
- b30→b29
- b29→b28
- b28→sequence

전체 점수는 강한 링크의 평균이 아니라 가장 약한 링크를 사용하고 off-path leakage로 벌점한다. 한 링크가 약한데 나머지가 강하다는 이유로 완전한 역추적을 주장하지 않는다.

## 8. Phase 4 — motif와 variant의 causal replay

### 8.1 후보별 필수 검증

각 motif/variant family에서 동일한 endpoint와 direction을 고정한 뒤 다음을 평가한다.

1. 자연 서열에서 target direction activation
2. motif 삽입 또는 variant edit로 activation 변화
3. b28 content 변화
4. b29에서 subtype에 맞는 transport
5. b30 mediation
6. 최종 output effect
7. upstream/downstream block
8. held-out predicted rescue
9. wrong motif/layer/direction/dose-matched control
10. 최소 세 specificity family

GC, k-mer likelihood, target token, repeat class, 상대 위치, orientation, token phase를 맞춘 control을 먼저 봉인한다.

### 8.2 serial ordering

`g28 → b29 → m30` 순서를 asymmetric bypass로 검증한다.

- b28 block 뒤 downstream `delta m30`은 효과를 rescue할 수 있어야 함
- b30 block 뒤 upstream `g28`만 복원해서는 효과가 돌아오면 안 됨

이 비대칭은 각 block이 중요하다는 사실보다 훨씬 강한 순서 증거다.

### 8.3 bilinear motif grammar

두 sequence element A와 B에 대해 `WT, A, B, AB` 2×2 cell을 만든다. b28의 정확한 `(W1 z)*(W2 z)` interaction과 downstream output interaction을 함께 계산한다.

W2 channel만 임의로 permutation한 control과 W1-W2를 함께 일관되게 relabel한 control을 비교하여, 단순 채널 개수나 norm이 아니라 **학습된 channel pairing**이 원인인지 확인한다.

### 8.4 context 길이

5, 9, 17, 33, 65, 129 bp window에서 효과가 full context의 사전 비율에 도달하고 이후 안정되는 최소 길이를 찾는다. 이는 benchmark가 다른 이유가 local motif 부족인지 long-range carrier/routing 요구인지 구분해 준다.

## 9. Phase 5 — learned delta transport

해석 모델은 absolute hidden state를 복원하지 않는다. 입력도 target도 intervention delta다.

### 9.1 비교 모델

- zero/null predictor
- scale-only 또는 radial-only model
- linear reduced-rank regression
- radial+tangential/spherical transition model
- bilinear factor model
- causal lag kernel, 특히 HCL의 장거리 전달
- sparse delta transcoder

rank와 ridge는 development에서 선택하고 locked에서는 고정한다. random intervention design으로 입력 공간을 충분히 자극하여 자연 상관만 학습하는 것을 막는다.

### 9.2 최종 증거

held-out delta R²만으로 causal map을 주장하지 않는다.

1. locked input으로 `predicted delta` 생성
2. native path를 block
3. 예측 delta를 실제 suffix 실행에 삽입
4. endpoint recovery fraction 측정
5. random, shuffled, wrong-layer, rank/dose-matched control과 비교

실제 관찰 delta를 다시 넣는 cached rescue는 upper bound일 뿐 primary evidence가 아니다.

예측이 약하면 구조를 숨기지 않는다. radial 성분만 예측되는지, branch별 model이 필요한지, nonlinear interaction이 필요한지 결과에 맞춰 model class를 바꾼다. locked 결과로 같은 모델을 재튜닝하지 않는다.

## 10. Phase 6 — 활용

### 10.1 benchmark heterogeneity

frozen mechanism fingerprint로 task family별 결과를 예측한다. few-shot을 사용할 때도 동일 family split과 label budget을 모든 baseline에 적용한다. zero-shot, 1, 5, 10, 25, 50, 100, 250, full label budget을 함께 그리면 “이 기전 지표가 label-efficient한가”를 확인할 수 있다.

### 10.2 EXP1 보강

EXP1의 현상을 새 ontology로 다시 읽는다. 단, EXP1 결과를 EXP3 target 선택에 누수시키지 않는다.

- EXP1 성공/실패 case가 어느 late-stack stage deficit과 대응하는가?
- 같은 GDTR 크기라도 b29 subtype, q, b30 mediation, carrier sensitivity가 다른가?
- benchmark별 성능 차이가 frozen fingerprint로 예측되는가?

이 분석은 EXP1을 단순 반복하는 것이 아니라, EXP2에서 확정하고 EXP3에서 세분화한 구조가 기존 현상을 더 잘 설명하는지 보여준다.

### 10.3 stage debugger와 selective repair

오류 case를 content-write 부족, b29 transport 이상, b30 re-encoding 이상, carrier/readout 이상으로 진단한다. 진단한 stage에만 intervention했을 때 맞춤 repair가 wrong-stage repair보다 커야 한다.

### 10.4 compression

원인 subspace를 보존하는 low-rank compression, bilinear channel 축소, canonical norm, frozen HCL router를 조합한다. 무조건 압축하지 않고, 해당 causal evidence가 확보된 component만 압축 계획에 넣는다. FLOP-matched generic low-rank baseline과 같은 unit에서 비교한다.

### 10.5 calibration과 steering

- carrier dose로 content direction을 바꾸지 않고 calibration만 조절할 수 있는지 검증
- mechanism score를 이용해 서열 후보를 탐색
- 실제 Evo 2 intervention에서 원하는 output gain, target alignment, off-target penalty 평가

### 10.6 외부 assay

assay label을 열기 전에 variant별 signed prediction을 SHA-256으로 봉인한다. label 공개 뒤 동일 prediction hash, 같은 variant ID, dependency cluster를 확인하고 mechanism prediction과 비기전 baseline을 비교한다. 외부 상관만으로 모델 내부 causal path를 주장하지 않고, Phase 4의 internal block/rescue와 함께 triangulation한다.

## 11. Phase 7 — 강건성

### 11.1 chromosome과 family

- chr22 discovery 결과를 chr17 locked에서 그대로 평가
- 같은 motif/variant family가 split을 가로지르지 않음
- leave-family-out과 leave-task-out을 별도 보고
- 평균뿐 아니라 family별 분산과 실패 regime 보고

### 11.2 checkpoint와 model size

7B 방향을 1B/base에 index로 억지 대응하지 않는다. discovery anchor로 rectangular Procrustes alignment를 고정하고, alignment fidelity가 충분할 때만 방향을 transfer한다. transferred effect, wrong-layer, random direction, b30 block, predicted rescue를 함께 요구한다. predicted delta는 train/development에서 적합한 봉인 모델과 hash-bound locked input으로 코드 내부에서 다시 생성해야 한다. 호출자가 제출한 prediction hash와 “regenerated hash”가 서로 같다는 사실만으로는 prospective prediction으로 인정하지 않는다.

### 11.3 training dynamics

가능하면 여러 checkpoint와 독립 seed/unit에서 다음 순서를 본다.

1. b28 content write 출현
2. b29 subtype/scale structure 출현
3. b30 mediation 출현
4. downstream performance 출현

각 unit은 동일 checkpoint grid를 가져야 하며, 한 번 threshold를 스친 시점을 onset으로
세지 않는다. 사전 지정한 수의 연속 checkpoint에서 기준을 유지하는 최초 sustained
crossing을 unit별로 계산하고, acquisition 순서가 성립하는 unit 비율과 인접 onset
차이를 dependency-cluster bootstrap 신뢰구간으로 판정한다. 순서가 다르거나 CI가
넓으면 단일 보편 발달 순서를 주장하지 않고 model-specific 또는 unresolved
acquisition으로 보고한다.

## 12. Phase 8 — 선택적 SAE/transcoder

SAE는 처음부터 켜지 않는다. 다음 원시 결과가 준비된 target에만 사용한다.

- b29 target이 정의됨
- direction과 scale이 분리됨
- b30 mediation이 확인됨
- held-out predicted delta rescue가 있음
- untouched chromosome이 남아 있음

이는 연구 전체를 막는 gate가 아니라 **SAE 주장의 국소 적격성 조건**이다. 조건이 안 되면 SAE를 생략하고 raw causal geometry 결론으로 논문을 끝낼 수 있다.

우선순위는 `delta_transcoder > delta_autoencoder > state SAE`다. reconstruction R²는 기술 통계다. feature causal claim에는 held-out delta prediction, feature ablation, decoded-feature rescue, random feature와 wrong-layer equivalence, cross-chromosome stability가 필요하다.

## 13. 적응형 분기표

| 시작 결과 | 멈추지 않고 이어갈 분석 |
|---|---|
| b29 amplifier 지지 | `amplifier_replication` |
| b29 second writer/editor/mixed | `second_writer_mapping` |
| b29 bounded-null | `b29_null_or_equivalence` |
| b29 unresolved | `b29_identifiability` |
| shared q transition | `normalized_scale_transfer` |
| heterogeneous transition | `site_specific_scale` |
| shared plateau refuted | `nonlinear_scale_mechanism` |
| scale unresolved | `scale_identifiability` |
| 공통 EXP3 target | `common_reverse_trace` |
| branch-specific/null common target | `branch_specific_reverse_trace` |
| target 자체가 불명확 | `reverse_trace_identifiability` |

m28, b29, scale에서 어떤 **유효한** 결과가 나오든 결과별 refinement를 거쳐
세 `target_spec`으로 봉인한 뒤 `exp3_mechanism_synthesis`로 통합한다. 이어지는
causal cube·b30 mediation·양방향 transfer를
`causal_factorization_synthesis`로 합성한 뒤에만 motif replay, learned transport,
benchmark heterogeneity, EXP1 extension, applications, cross-chromosome
generalization을 결과에 맞는 target 정의로 계속한다.

## 14. 논문에서 허용되는 주장 수준

### 최소 성공

- b29 subtype에 대한 bounded conclusion
- scale 경계의 공통성 또는 이질성에 대한 bounded conclusion
- scale·content·carrier 중 적어도 둘의 인과적 역할 분리
- chr17 또는 untouched chromosome에서 재현

### 강한 기전 논문

최소 성공에 더해:

- output→sequence reverse chain
- motif/variant block과 held-out rescue
- ordered bypass로 serial path 확인
- benchmark 차이의 out-of-family 예측

### 최고 수준 활용 논문

강한 기전 논문에 더해:

- predicted-delta intervention rescue
- stage-specific repair 또는 causal-subspace compression
- blinded external assay 또는 cross-checkpoint transfer

SAE는 어느 수준에서도 필수 조건이 아니다. 잘 작동하면 이미 확립한 computation의 vocabulary를 제공하고, 작동하지 않으면 raw causal result의 가치는 유지된다.

## 15. 리뷰어 질문에 대한 짧은 답

**“EXP2를 또 한 것 아닌가?”**  
아니다. EXP2의 stage ontology는 hash-sealed premise이고, EXP3는 b29 subtype, scale boundary, factorization, reverse use를 새로 검증한다.

**“norm explosion을 다시 기술한 것 아닌가?”**  
아니다. radial dose와 equal-norm angular control을 분리하고, host/update 비율로 정규화한 transition 및 output block/rescue를 측정한다.

**“motif는 attribution artifact 아닌가?”**  
후보 생성법 두 개 이상의 합의, matched controls, edit, block, predicted rescue, ordered bypass를 요구한다.

**“benchmark가 달라지면 이야기가 무너지지 않나?”**  
차이를 평균내지 않고 frozen mechanism fingerprint로 예측한다. heterogeneity 자체가 scale·content·carrier 요구량의 차이라는 검정 대상이다.

**“예상 결과가 안 나오면 뒤 실험을 못 하지 않나?”**  
아니다. 예상과 다른 유효한 결과는 alternative target을 정의한다. 믿을 수 없는 측정·provenance·split만 해당 계통을 중단한다.

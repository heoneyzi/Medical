<!-- Original Korean write-up kept verbatim (file: VCC2026_실험_종합해설_및_메트릭이해_1.md); only the file name was changed, LaTeX escapes that had been corrupted into control characters were repaired, and \[ \] / \( \) math delimiters were converted to $$ / $ for GitHub rendering. -->

# VCC2026 두 핵심 실험 + 강건성 확장: 기획 의도 · 조건 · 결과 · 메트릭 완전 해설

> **이 문서 한 장 요약**
>
> - **무엇을 한 실험인가**: "Frozen 파운데이션 모델(STATE·STACK·UCE·TranscriptFormer·scPRINT2)을 *예측기*가 아니라 *컨텍스트 유사도 계산기*로만 쓰고, 실제 반응은 Replogle GWPS 통계 라이브러리에서 전이한다"는 하나의 파이프라인을, 두 개의 서로 다른 zero-shot 분할(Jiang24 = 새 세포주, GSE270828 = 새 배치)에서 돌리고, 6축 VCC2026 메트릭으로 채점했다. 이어서 **온도(tau)·난수(seed)·예측 세포 수** 세 축으로 결과가 얼마나 흔들리는지(강건성)를 144개 GPU 실행으로 확장 검증했다.
> - **가장 중요한 결론 3가지**
>   1. **점수를 만드는 것은 "임베딩(신경망)"이 아니라 "효과 라이브러리(통계)"다.** no_effect → GWPS 전이의 도약이 크고, 그 다음 raw/PCA vs frozen 임베딩의 차이는 작고 조건에 따라 뒤집힌다. 비싼 파운데이션 모델이 컨텍스트 선택기로서 자기 몫을 못 했다.
>   2. **"흔한 vs 새로운"의 진짜 의미는 "새 배치 vs 새 세포주"이며, 이 둘은 난이도가 근본적으로 다르다.** 새 배치(GSE270828)는 Overall이 **양수(~0.75)**, 진짜 새 세포주(Jiang24 BxPC3)는 **음수(-0.76 ~ -2.5)** — 즉 자명한 baseline보다도 낮다. VCC 챌린지가 요구하는 건 후자다.
>   3. **어느 메트릭에 집중하느냐가 승자를 바꾼다.** Jiang24 Overall은 사실상 **Jaccard 한 축**이, GSE270828 순위는 **PDS 한 축**이 지배한다. 한 축만 빼도 winner가 바뀐다. "Overall 1등"은 특정 축의 그림자일 뿐이다.
> - **주의**: 여기 모든 점수는 공개 `cell-eval2` 구현을 외부 데이터에 돌린 **dataset-local 점수**이지 챌린지 서버 공식 점수가 아니다. 확장 스윕은 truth를 본 뒤의 **post-hoc 탐색**이므로 "최고 tau/세포수"를 새 정답으로 선언하면 안 된다(선택 편향).

---

## 0. 이 문서를 읽는 법

이 문서는 세 가지 질문에 순서대로 답한다.

1. **왜 이 실험을 이렇게 설계했는가** (2장) — 기획 의도.
2. **정확히 어떤 데이터·모델·방법·조건이었는가** (3장) — 재현 가능한 수준의 조건 명세.
3. **그래서 무엇을 알게 되었는가** (4~6장) — VCC 베이스라인의 실제 "수준", 6개 핵심 질문의 답, 그리고 확장 실험에서 추가로 뽑아낸 결과.

그리고 마지막에 <b>전략적 시사점(7장)</b>, <b>정직한 한계(8장)</b>, <b>재현 정보(9장)</b>를 둔다. 숫자 표는 근거로 쓰고, 그 사이사이에 "이 숫자가 왜 그런가"를 문장으로 풀었다.

---

## 1. 등장하는 개념 빠른 정의 (용어 사전)

먼저 이 문서에서 반복되는 핵심 용어를 한 곳에 모아둔다. 뒤에서 다시 설명하므로 지금은 훑고 지나가도 된다.

- **Zero-shot 컨텍스트 전이**: 모델이 한 번도 본 적 없는 세포 맥락(세포주/배치)에서, 그 맥락의 *대조군(control) 세포만* 보고 각 유전자 knockdown의 반응을 예측하는 것. 정답(perturbed truth)은 채점 때만 쓴다.
- <b>효과 라이브러리(effect library)</b>: "유전자 X를 억제하면 발현 프로파일이 이렇게 바뀐다"를 이미 알고 있는 참조표. 여기서는 Replogle 2022 GWPS(K562·RPE1 genome-wide Perturb-seq)에서 control 대비 perturbed의 <b>log-fold-change(logFC)</b>로 만든다.
- **Frozen 컨텍스트 인코더**: 학습을 시키지 않고(=frozen) 그대로 쓰는 파운데이션 모델. 여기서는 오직 *"holdout 맥락이 어느 source 맥락과 닮았는가"*를 재는 임베딩 공간으로만 쓴다. 반응 자체를 생성하지 않는다.
- **source vs holdout**: source = 효과를 만들 수 있는(정답 있는) 참조 맥락들. holdout = 예측 대상인 미지의 맥락.
- **gwps_nearest vs gwps_weighted**: nearest = 가장 닮은 source **하나**의 효과를 그대로 전이. weighted = 여러 source의 효과를 cosine 유사도 → softmax **가중 혼합**.
- **tau(온도)**: 유사도를 가중치로 바꿀 때의 softmax 온도. 작으면 가장 닮은 하나에 몰아주고(≈nearest), 크면 여러 source를 평탄하게 섞는다.
- **generator**: logFC로 예측한 평균 프로파일을, 제출 규격인 **정수 카운트 세포들**로 바꾸는 표본 생성기. 여기서는 multinomial 고정.
- **6축 메트릭 (VCC2026 preset)**: PDS, MSE, NMAE, fidelity, reach, Jaccard. Overall = 이 여섯 정규화 축의 **비가중 평균**. (각 축의 뜻은 6장에서 자세히.)
- **local-anchor 스케일링**: 각 데이터셋에서 "generic 반응 baseline = 0", "replicate split-half anchor = 1"로 맞춘 정규화. 그래서 **음수도, 1 초과도 정상**이다(단지 baseline/anchor 밖이라는 뜻).

---

## 2. 이 실험을 기획한 의도

### 2.1 큰 그림 — 무엇이 궁금했나

VCC 2026은 "**한 번도 본 적 없는 세포 맥락**에서 CRISPRi knockdown의 단일세포 전사체 반응을 예측하라"는 zero-shot 문제다. 여기엔 학습을 전혀 하지 않고도 리더보드에 올릴 수 있는 두 가지 상반된 접근이 있다.

- **(가) 신경망 파운데이션 모델 접근**: STATE·STACK 같은 대형 세포 모델의 표현력이 미지의 맥락을 잘 "이해"해서 좋은 예측을 준다는 가설.
- **(나) 통계 데이터베이스 접근**: 이미 측정된 Replogle GWPS의 방대한 효과표를 미지 맥락으로 "옮기기만" 해도 강하다는 가설.

이 실험의 기획 의도는 **이 둘을 공정하게, 통제된 조건에서 맞붙이는 것**이다. 핵심 설계 결정이 하나 있다: **파운데이션 모델이 반응을 직접 생성하게 두지 않는다.** 대신 모든 방법이 *동일한* Replogle 효과 라이브러리, *동일한* 생성기, *동일한* 난수와 세포 수, *동일한* 채점기를 쓰게 하고, **오직 "어느 source 맥락이 holdout과 가까운가"를 재는 임베딩 공간만 바꾼다.**

> **왜 이렇게 했는가 (설계의 핵심 논리)**: 만약 각 모델이 자기 방식으로 반응까지 생성하게 하면, 성능 차이가 "표현력 때문인지, 생성 방식 때문인지, 라이브러리 때문인지" 뒤섞여 해석 불가능해진다. 다른 모든 것을 고정하고 **컨텍스트 유사도 표현 하나만** 변수로 두면, 행 사이의 차이는 *그 벤치마크가 허용하는 한* 온전히 "컨텍스트 표현의 차이"로 귀속된다. 이것이 실험을 "공정한 A/B 비교"로 만드는 장치다.

### 2.2 Core 실험 1과 2의 역할 분담

- <b>실험 1 (STATE/STACK 중심)</b>: 기존에 이미 검증된 STATE·STACK 컨텍스트 임베딩을, raw·PCA 통계 기준선과 함께, 그리고 <b>여러 전이 방법(no_effect, global_mean, gwps_direct, gwps_nearest, gwps_weighted)</b>과 함께 비교한다. "임베딩을 쓰면 통계 기준선(raw/PCA)보다 나은가?"와 "nearest와 weighted 중 무엇이 안전한가?"를 본다.
- **실험 2 (동일조건 전체 frozen 통합)**: 실험 1의 파이프라인을 그대로 두고, 검증된 추가 인코더 **UCE·TranscriptFormer·scPRINT2**를 같은 비교 행렬에 편입한다. 다섯 개 frozen 인코더 + raw + PCA를 **완전히 동일한 조건**에서 `gwps_weighted`로 세워, 표현 공간 차이를 가장 직접적으로 드러낸다.

즉 실험 1은 "방법(nearest/weighted/…)의 축", 실험 2는 "표현(raw/PCA/5개 frozen)의 축"에 초점을 둔 것이다.

### 2.3 New 확장 실험(강건성)의 의도

Core는 각 조건에서 "누가 몇 점"인지를 냈다. 하지만 남는 의심이 있다.

1. Jiang24에서 scPRINT2의 우위, STACK의 fidelity 우위 같은 결과가 **특정 tau(온도) 하나**에만 나타나는 착시 아닌가?
2. GSE270828의 PDS–fidelity 상충이 **하필 그 난수 seed**의 우연 아닌가?
3. 예측 세포 수를 바꾸면 분포·DE 기반 축의 영향과 모델 순위가 **뒤집히지 않는가**?
4. Overall에서 **한 축을 빼면** winner가 바뀌는가? (= Overall이 특정 축에 얼마나 의존하는가)

그래서 New 확장은 <b>새 모델을 더 넣지 않고</b>, 기존 결론이 세 가지 잡음(온도·난수·표본깊이)에 얼마나 견디는지만 검증한다. 이것은 새로운 리더보드 주장이 아니라 <b>"우리가 이미 낸 결론을 얼마나 믿어도 되는가"</b>를 재는 자기검증 실험이다.

---

## 3. 정확한 실험 조건 — 데이터 · 모델 · 방법

### 3.1 두 데이터셋의 정체와 분할

| 항목 | **Jiang24 (IFNG, BxPC3 holdout)** | **GSE270828 (rep3 holdout)** |
|---|---|---|
| 분할 방식 | **leave-one-cell-line-out** (세포주 빼기) | **leave-one-batch-out** (복제배치 빼기) |
| source 맥락 | A549, HAP1, HT29, K562, MCF7 (5개 **서로 다른 세포주**) | rep1, rep2 (같은 실험의 **2개 복제배치**) |
| holdout(예측 대상) | **BxPC3** (췌장암, source에 없던 새 세포주) | **rep3** (같은 실험의 또 다른 복제) |
| 표적(perturbation) | 유전자 56개 | **HAR 조절요소 184개** (유전자가 아님) |
| 인코더 입력 유전자 | 15,473 | 19,698 |
| VCC2026 평가 유전자 패널 | 4,017 | 4,095 (실제 mapped 고유 유전자 141) |
| source control / perturbed 세포 | 2,500 / 27,686 | 1,000 / 35,998 |
| holdout truth 세포 | 5,979 (control 500 + perturbed 5,479) | 17,974 (control 484 + perturbed 17,490) |
| 기본 예측 세포(표적당 50) | 2,800 | 9,200 |

> <b>여기서 이미 핵심 통찰 하나가 나온다.</b> 사용자가 말한 "흔한 cell line vs 새로운 cell line"의 데이터적 실체는 <b>"새 세포주 예측(진짜 어려움) vs 새 복제배치 예측(상대적 쉬움)"</b>이다.
> - Jiang24는 5개의 진짜 다른 암세포주에서 **처음 보는 BxPC3**로 건너뛴다 → VCC가 원하는 **진짜 zero-shot 세포 맥락 일반화**.
> - GSE270828은 같은 HAR 실험의 rep1/rep2에서 **또 다른 복제 rep3**로 간다 → 맥락 자체는 거의 같고 배치 잡음만 다르다. 게다가 표적이 유전자가 아니라 HAR 조절요소라서, 유전자 기반 Replogle 라이브러리와 궁합이 특수하다(3.5 참조).

### 3.2 등록된 표현(representation)과 방법(method)

**Frozen 컨텍스트 인코더 (실험 2 동일조건 행렬에 실제 등록)** — 모두 실제 checkpoint 산출물이며, 기대 맥락 수·차원·finite·맥락 간 비상수성·provenance 해시를 통과함:

| 표현 | pooled 차원 | 역할 |
|---|---:|---|
| STATE | 1,034 | frozen 컨텍스트 인코더 |
| STACK | 1,600 | frozen 컨텍스트 인코더 |
| UCE-4L | 1,280 | frozen 컨텍스트 인코더 |
| TranscriptFormer-Sapiens (cell) | 2,048 | frozen 컨텍스트 인코더 |
| scPRINT-2 small-v2 | 424 | frozen 컨텍스트 인코더 |
| raw (log1p pseudobulk) | — | 통계 기준선(코사인) |
| PCA | 5~50 | 통계 기준선(차원축소 코사인) |

- **scGPT·scFoundation은 제외**됨 — 수동 checkpoint가 없어 무작위/대체 없이 **fail-closed**로 뺐다(정직성 장치).
- **Geneformer-deletion, scPRINT2-GRN, TranscriptFormer-CGE**는 이 컨텍스트 행렬에서 **의도적으로 제외** — 이들은 *표적별 prior*(반응 자체를 아는 모델)라서 "컨텍스트 인코더" 실험의 통제를 깨기 때문. registry에는 제외 사유와 함께 남겨둠.

**전이 방법(method)**:

| 방법 | 성격 | 설명 |
|---|---|---|
| `no_effect` | 자명 기준선 | 아무 효과 없음(control 그대로). "아무것도 안 하기". |
| `global_mean` | 자명 기준선 | 유전자 무관 평균 효과. |
| `gwps_direct` | 통계 전이 | source 효과를 유사도 없이 직접 전이. |
| `gwps_nearest` | 통계+표현 | 가장 닮은 **단일** source 효과 전이. |
| `gwps_weighted` | 통계+표현 | cosine→softmax(tau)로 **여러 source 가중 혼합**. ← 표현 차이가 가장 잘 드러나는 방법 |

### 3.3 공통 파이프라인 (통제 설계)

모든 행이 아래 5단계를 **똑같이** 밟는다. 오직 3단계의 "유사도 공간"만 표현마다 다르다.

1. 각 source 맥락의 control·perturbed count에서 **logFC 효과 라이브러리** 생성.
2. holdout 맥락에서는 **control 세포만** 인코더에 넣어 컨텍스트 벡터 생성 (perturbation truth는 절대 안 봄).
3. raw/PCA/frozen 표현의 **cosine 유사도**로 source를 선택(nearest) 또는 가중(weighted).
4. 모든 표현에 **동일한** multinomial 생성기, **seed=2026**, 표적당 **50세포**, **scale=1.0**, 음수 clipping, control 미포함 적용.
5. 고정 유전자 순서의 count H5AD로 예측 → **동일 truth·동일 dataset-local 스케일**로 채점.

기본 하이퍼파라미터(Core): `method=gwps_weighted, effect=logfc, generator=multinomial, cells_per_pert=50, similarity_space={raw|pca|embed}, scale=1.0, shrink=none, tau=0.1, seed=2026`.

### 3.4 평가기와 6축 규칙

- 평가기: **`cell-eval2==0.16.0`, preset `vcc2026`, GPU `gpudge` 백엔드**.
- 6축: **PDS, MSE, NMAE, fidelity, reach, Jaccard**. Overall = 여섯 정규화 축의 **정확한 비가중 평균**.
- 스케일: 데이터셋별 **generic-response baseline=0**, **다섯 seeded split-half replicate anchor 평균=1**. → 음수/1초과는 오류가 아니라 baseline/anchor 밖.
- **데이터셋 간 점수 평균·직접 크기 비교 금지** (anchor가 다름). Jiang24와 GSE270828 점수는 각자의 자로 잰 값이다.

### 3.5 GSE270828의 "0축 네 개" — 반드시 이해할 함정

GSE270828은 표적이 **HAR 조절요소**라, truth-only 지원 감사에서 `de_lfc_nmae` 지원 표적이 **0/184**였다. 공개 evaluator 규칙상 **MSE·NMAE·reach·Jaccard 네 축이 모두 명시적 0**으로 fallback된다. 즉:

- **GSE270828의 Overall은 사실상 PDS와 fidelity 두 축만으로 만들어진다.** 나머지 넷은 0으로 채워져 평균에 들어간다.
- 이건 "모델들이 그 축에서 동점"이 아니라 <b>"그 축이 이 데이터에선 정의되지 않는다"</b>는 뜻이다.
- Jiang24에서도 normalized **MSE는 전 행 0**이지만, 나머지 다섯 축은 정의된다.

이 사실을 모르면 "GSE는 왜 이렇게 점수가 안정적이지?"를 오해하게 된다. 답은 **채점되는 축이 적어서**다.

### 3.6 New 확장의 스윕 설계와 계산량

Core 기준 조건(`tau=0.1, seed=2026, cells=50`)을 축으로, **한 번에 한 변수만** 바꾼 9개 신규 조건:

| 실험군 | 고정 | 신규 값 | 기준(재사용) |
|---|---|---|---|
| **A. 온도 tau** | seed 2026, 50세포 | 0.01, 0.03, 0.3, 1.0 | 0.1 |
| **B. 생성 seed** | tau 0.1, 50세포 | 2024, 2025, 2027 | 2026 |
| **C. 예측 세포 수** | tau 0.1, seed 2026 | 25, 100 | 50 |

각 조건 × 2 데이터셋 × 8행(raw/no_effect + 7표현 gwps_weighted) = 신규 **144 GPU 행** (전부 `status=ok`). 통합 분석표는 신규 144 + 참조 48 = **192행**. (참조 48 = 기준 16행을 세 실험군에 각각 연결해 3번 표시한 것이므로 48개의 독립 실험이 아님.) 신규 144 예측에 포함된 세포는 **총 912,000개**(Jiang24 212,800 + GSE270828 699,200).

무결성: Overall 6축 재계산 최대 절대오차 **2.22e-16**, 비유한 metric **0**, 분산 기여율 그룹합 **1.0**(부동소수 오차 내).

---

## 4. VCC 초기 베이스라인의 "수준" — 숫자로 본 현실

여기가 사용자가 가장 궁금해한 "VCC의 초기 베이스라인 수준"의 핵심이다. 기준 조건(`tau=0.1, seed=2026, cells=50`, `gwps_weighted`)의 **6축 전체**를 데이터셋별로 본다. **높을수록 좋고, 음수도 정상**(baseline보다 낮다는 뜻).

### 4.1 Jiang24 (새 세포주 BxPC3) — 6축 전체

| 표현 | PDS | MSE | NMAE | fidelity | reach | Jaccard | **Overall** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **scPRINT2** | 0.221 | 0 | -2.508 | 0.310 | -0.218 | **-2.363** | **-0.760** |
| PCA | 0.224 | 0 | -1.447 | 0.626 | 0.032 | -4.999 | -0.927 |
| raw | **0.390** | 0 | -1.881 | 0.567 | 0.029 | -4.981 | -0.979 |
| *(raw / no_effect)* | -0.098 | 0 | -1.287 | 0.590 | -0.336 | -6.929 | *-1.343* |
| UCE | 0.257 | 0 | -1.589 | 0.542 | 0.027 | -7.522 | -1.381 |
| TranscriptFormer | 0.188 | 0 | -0.751 | 0.674 | -0.175 | -8.498 | -1.427 |
| STATE | 0.112 | 0 | -1.128 | 0.599 | -0.027 | -8.404 | -1.475 |
| STACK | 0.149 | 0 | **-0.723** | **0.735** | -0.271 | -10.110 | -1.703 |

**읽는 법 (매우 중요)**:

- **모든 Overall이 음수다.** 즉 진짜 새 세포주에서는, GWPS 효과를 frozen 컨텍스트 가중으로 전이해도 **"generic 반응 baseline(=0)"보다 낮다.** 이것이 VCC zero-shot의 냉정한 현실이다 — 문제는 정말 어렵고, 순진한 전이는 자명한 기준선도 못 넘는다.
- **scPRINT2가 1등인 이유는 오직 Jaccard 때문이다.** PDS·NMAE·fidelity 어디서도 1등이 아니다. 단지 Jaccard가 -2.36으로 *덜 나빴을* 뿐(다른 표현은 -5 ~ -10). "signature 겹침 손실이 가장 적었다"가 정확한 표현이다.
- **STACK은 fidelity 1등(0.735)인데 Overall 꼴찌다.** Jaccard -10.11이 모든 걸 삼켰다. **"한 축을 잘해도 전체 복원이 좋아진 게 아니다."**
- raw는 <b>PDS 1등(0.390)</b>이고 Overall 3등 — 통계 기준선이 신경망 임베딩에 전혀 밀리지 않는다.

### 4.2 GSE270828 (새 배치 rep3) — 정의된 두 축

여기선 MSE·NMAE·reach·Jaccard가 구조적으로 0이므로(3.5), 실질은 PDS·fidelity·Overall이다.

| 표현 | PDS | fidelity | (MSE/NMAE/reach/Jac) | **Overall** |
|---|---:|---:|:--:|---:|
| **PCA** | **0.423** | 4.098 | 0 | **0.753** |
| UCE | 0.226 | 4.198 | 0 | 0.737 |
| TranscriptFormer | -0.117 | 4.143 | 0 | 0.671 |
| raw | -0.097 | 4.081 | 0 | 0.664 |
| STATE | 0.039 | 3.944 | 0 | 0.664 |
| scPRINT2 | -0.443 | **4.229** | 0 | 0.631 |
| *(raw / no_effect)* | -0.438 | 4.072 | 0 | *0.606* |
| STACK | -0.819 | 4.157 | 0 | 0.556 |

**읽는 법**:

- **모든 Overall이 양수(~0.55–0.75)다.** 새 배치는 "쉽다" — 맥락이 거의 같으니 전이가 baseline을 넘는다. 하지만 이건 **채점 축이 둘뿐**이라 점수가 안정적으로 보이는 착시도 섞여 있다.
- **fidelity는 모든 표현이 ~4로 거의 붙어 있다.** 즉 fidelity는 "점수의 절대 크기"는 만들지만 **모델을 거의 구분하지 못한다.** 실제로 순위를 가르는 것은 **PDS 한 축**이다(PCA 0.423이 최고, STACK -0.819가 최저 → 그대로 Overall 순위).
- 여기서도 **PCA(통계)가 1등**, 신경망 인코더들은 그 아래다.

### 4.3 두 표를 나란히 놓고 얻는 "수준" 감각

- <b>난이도의 스케일 차이</b>: 같은 파이프라인이 한쪽(새 배치)에선 +0.75, 다른 쪽(새 세포주)에선 -0.76~-1.7. 이 격차가 곧 <b>"본 적 있는 실험의 새 복제를 맞추기" vs "처음 보는 세포주를 맞추기"</b>의 난이도 차이다. <b>VCC 리더보드가 요구하는 건 후자</b>임을 명심해야 한다.
- **베이스라인의 현주소**: 학습 0의 GWPS-전이 베이스라인은 (진짜 zero-shot 기준) 아직 자명한 baseline 아래에 있다. 개선 여지가 크다는 뜻이자, 순진한 전이만으로는 부족하다는 경고다.

---

## 5. 여섯 개 핵심 질문에 대한 답

### Q1. Frozen 모델의 성능은 어느 정도인가?

**요약: "쓸 수는 있지만, 통계 기준선을 안정적으로 이기지 못한다."**

- Jiang24 기준 조건에서 frozen 중 최고는 scPRINT2(-0.760)로 PCA(-0.927)·raw(-0.979)를 **약간** 앞선다. 그러나 이 우위는 **Jaccard 한 축에서만** 나오고, 아래(Q2·6장)에서 보듯 **seed·tau·세포수를 바꾸면 사라진다.**
- GSE270828에선 **frozen이 전부 PCA에 진다**(PCA 0.753 > UCE 0.737 > … > frozen들). 통계 표현이 최고.
- **strict Replogle-only** 보조 실험(외부 반응만 엄격히 사용)에서는 **raw가 모든 frozen을 이긴다**(dual-source 8타깃 층화에서도 raw -0.145가 최고). 즉 가장 엄격한 세팅에서 frozen이 유리하다는 근거가 **전혀 나오지 않았다.**

> **정직한 결론**: 이 파이프라인에서 frozen 파운데이션 모델은 "컨텍스트 유사도 계산기"로서 **자기 비용을 정당화하지 못했다.** 한 조건에서의 scPRINT2 우위는 실재하지만 **깨지기 쉬운(fragile) 우위**다.

### Q2. 메트릭 ablation — 각 메트릭의 민감성·특징·변화

이것은 "각 축이 얼마나 예민하고, 무엇에 반응하는가"의 질문이다. New 확장이 정확히 이걸 측정했다.

**(a) 어느 축이 Overall을 "지배"하는가 — 두 종류의 지배를 구분하라.**

- **절대 수준 지배축**(점수의 크기를 만드는 축): dominant-axis 집계에서 Jiang24는 **Jaccard가 85/96 결과**, NMAE 11/96. GSE270828은 **fidelity가 96/96**.
- **변동 지배축**(모델·조건 간 순위를 만드는 축, = 분산 기여율):

| 범위 | 데이터셋 | tau / seed / cells 분산 기여축 | 기여율 |
|---|---|---|---|
| 전체 frozen | Jiang24 | **Jaccard** | 98.2% / 93.4% / **163.1%** |
| 전체 frozen | GSE270828 | **PDS** | 86.3% / 86.7% / 81.3% |

  → Jiang24는 "크기도 순위도" Jaccard가 지배. GSE는 **크기는 fidelity가, 순위는 PDS가** 만든다(둘이 다르다!). cells에서 163%가 넘는 건 오류가 아니라, Jaccard 변동이 Overall 변동보다 크고 다른 축의 음의 공분산이 상쇄했다는 뜻(그룹 내 6축 기여율 합=1).

**(b) 한 축을 빼면 winner가 바뀌는가 — leave-one-axis-out.**

| 데이터셋 | 12개 조건 중 winner를 바꾼 "제거 축" |
|---|---|
| Jiang24 | **Jaccard 제거 11/12**, NMAE 제거 1/12. (나머지 축은 0) |
| GSE270828 | **PDS 제거 7/12**, fidelity 제거 5/12. (나머지 축은 0) |

  기준 조건 구체 예: **Jiang24에서 Jaccard를 빼면 winner가 scPRINT2 → TranscriptFormer로** 바뀐다. **GSE에서 PDS를 빼면 PCA → scPRINT2로** 바뀐다. 즉 **"Overall 1등"이라는 말은 특정 한 축을 채점에 넣었기 때문에 성립**한다.

**(c) 각 축의 개별 민감성(무엇에 예민한가)**:

- **Jaccard**: 가장 예민하고 변동이 크다. 세포 수(표본 깊이)에 극도로 민감(아래 6.3). Jiang24 Overall의 사실상 주인.
- **PDS**: 부호가 자주 바뀌는 예민한 축. GSE 순위의 주인. seed·tau에 따라 양↔음을 오간다(예: GSE cells=100에서 PCA PDS가 0.423→-0.762로 급락).
- **fidelity**: 절대값(크기)은 크지만 **모델 구분력이 약하다**(GSE에서 전부 ~4). 안정적이지만 순위를 못 만든다.
- **NMAE**: 표본 깊이에 강하게 반응. 세포 25개면 바닥(≈-6.0)에 붙어 구분력을 잃고, 100개면 0 근처로 회복(6.3).
- **MSE**: 이 두 데이터셋에선 정규화 후 전 행 0 — **이 세팅에선 정보가 없다.**
- **reach**: 대체로 작고 부차적. Jiang24에서 소폭만 움직인다.

### Q3. 두 데이터셋(새 세포주 vs 새 배치)의 성능 차이와 이해

이미 4.3에서 큰 그림을 봤다. 여기에 **"왜" 그런가**를 메커니즘으로 덧붙인다.

- **source의 성격이 다르다**: Jiang24 source는 5개의 **진짜 다른 세포주**라, "어느 게 BxPC3와 닮았나"가 실질적 질문이다(그리고 아무도 잘 못 맞힌다 → 음수). GSE source는 같은 실험의 **rep1/rep2 복제 2개**뿐이라, 유사도 질문이 거의 자명하다(전부 rep1 선택 → 안정적 양수).
- **표적의 성격이 다르다**: Jiang24는 유전자 표적이라 Replogle 유전자 라이브러리와 직접 매칭된다. GSE는 **HAR 조절요소**라 exact-gene 라이브러리에 없어서, DE 계열 4축이 0으로 죽는다(3.5). 그래서 GSE는 "쉬워 보이지만 실은 채점 정보가 빈약한" 데이터다.
- **일반화 종류가 다르다**: GSE는 **배치 잡음 일반화**(같은 맥락, 다른 날), Jiang24는 **세포 정체성 일반화**(다른 세포). 후자가 생물학적으로 훨씬 어렵고, VCC의 본질이다.

> **이해의 요점**: 두 데이터셋을 **절대 한 자로 비교하지 말 것**(anchor가 다르고 축 개수도 다르다). 대신 이렇게 읽어야 한다 — GSE의 +0.75는 "성공"이 아니라 "쉬운 문제에서의 하한선 확인"이고, Jiang24의 음수는 "실패"가 아니라 "진짜 문제의 난이도 측정"이다.

### Q4. Frozen vs 통계적 방법 — 차이와 특징

**핵심: 가치의 대부분은 "통계(효과 라이브러리)"에서 나오고, "신경망(임베딩)"의 추가 기여는 작고 불안정하다.**

두 단계로 분해해 보면 명확하다.

1. **1단계 — 효과 라이브러리의 가치(큰 도약)**: `no_effect`/`global_mean` → 아무 GWPS 전이.
   - Jiang24: no_effect -1.343 → weighted 전이 -0.98~-0.76 (**뚜렷한 개선**).
   - GSE270828: no_effect 0.606 → nearest 0.753 (**뚜렷한 개선**).
   - 이 도약이 **가장 크고 확실한 신호**다. "이미 측정된 CRISPRi 효과를 옮기는 것" 자체의 가치.
2. **2단계 — 표현의 추가 가치(작고 조건 의존)**: raw/PCA(통계 유사도) vs frozen(신경망 유사도).
   - Jiang24 기준: scPRINT2가 raw/PCA를 근소하게 앞서지만 Jaccard 한 축·특정 조건에 한함.
   - GSE270828: **PCA(통계)가 최고**, frozen 전부 아래.
   - strict Replogle-only: **raw(통계)가 최고**, frozen 전부 아래.

**방법의 성격 차이도 하나 더**: `gwps_nearest`(단일 source) vs `gwps_weighted`(혼합). Jiang24에서 STATE/STACK은 **nearest가 weighted보다 안정적**이었다. 유사도 신호가 약하고 잡음이 많을 때는 **여러 source를 섞으면 잘못된 가중치가 효과를 오염**시키고, 가장 닮은 하나만 고르는 게 더 안전하다. (반대로 GSE는 source가 2개뿐이라 nearest들이 같은 source를 골라 동점이 자주 났다 — 표현이 같아서가 아니라 argmax가 같아서.)

### Q5. 각 모델 파이프라인의 차이가 왜 성능 차이를 만드는가 (메커니즘)

파이프라인은 **완전히 동일**하고 오직 "컨텍스트 유사도를 재는 임베딩"만 다르다. 그 임베딩이 하는 일은 단 하나 — **"holdout이 어느 source와 닮았는가"의 판단**. 이 판단이 다르면 전이하는 효과가 달라지고, 점수가 달라진다. New의 source-weight 원자료가 이 메커니즘을 눈으로 보여준다.

**Jiang24에서 각 표현이 고른 "가장 닮은 source"** (tau=0.1 기준):

| 표현 | top source | top weight (tau 0.01 → 1.0) | 유효 source 수 (tau 0.01 → 1.0) |
|---|---|---|---|
| raw | **A549** | 0.802 → 0.220 | 1.47 → 4.97 |
| PCA | **A549** | 1.000 → 0.328 | 1.00 → 4.43 |
| TranscriptFormer | **A549** | 0.980 → 0.218 | 1.04 → 4.98 |
| UCE | **A549** | 0.999 → 0.226 | 1.00 → 4.96 |
| scPRINT2 | **HT29** | 0.668 → 0.215 | 1.80 → 4.98 |
| STATE | **HT29** | 0.888 → 0.230 | 1.25 → 4.93 |
| STACK | **HT29** | 0.999 → 0.246 | 1.00 → 4.88 |

> **이것이 "파이프라인 차이 → 성능 차이"의 정체다.** 표현들이 **두 진영으로 갈린다** — raw·PCA·UCE·TranscriptFormer는 BxPC3가 **A549**와 닮았다고 보고, scPRINT2·STATE·STACK은 **HT29**와 닮았다고 본다. 완전히 같은 라이브러리·생성기·seed인데도 **"어느 세포주가 이 미지의 세포주와 가까운가"에 대한 의견이 다르기 때문에** 예측과 점수가 갈린다. scPRINT2가 Jaccard에서 덜 나빴던 건, 결국 이 세팅에서 HT29 효과가 BxPC3의 DE 집합과 조금 덜 어긋났다는 뜻이다.

또 하나: **tau는 이 진영 선택을 "얼마나 확신하는가"의 손잡이**다. 위 표에서 tau를 키우면 top weight가 떨어지고 유효 source 수가 1→~5로 늘며 여러 세포주를 평탄하게 섞는다. 즉 tau sweep은 이름만 다른 같은 예측이 아니라 **날카로운 단일 선택 ↔ 평탄한 혼합** 사이를 실제로 이동한 실험이다. (반면 GSE PCA는 tau=0.1까지 rep1 weight가 반올림상 1.000이라 tau 0.01/0.03/0.1 점수가 동일 — source가 2개뿐이고 PCA가 rep1에 완전히 확신하기 때문.)

### Q6. 각 메트릭의 "진정한 의미"와 무엇에 집중해야 하는가

6축을 "무엇을 재는가 / 무엇에 예민한가 / 어떻게 속을 수 있는가"로 정리한다.

- **PDS (Perturbation Discrimination Score)** — *"예측이 자기 정답에 다른 표적의 정답보다 더 가까운가"*, 즉 **표적을 구별해내는 능력**. 절대 오차가 아니라 **상대적 순위/판별**을 본다. GSE 순위의 주인이자, 새 세포주에서 "그래도 방향은 맞나"를 보는 데 가장 정직한 축. **예민하고 부호가 잘 바뀐다** → 반드시 여러 seed 평균으로 봐야 함.
- **MSE** — 프로파일 제곱오차. **이 두 데이터셋에선 정규화 후 전 행 0**이라 정보 없음. (일반적으론 큰 발현 유전자에 지배되는 경향.)
- **NMAE (normalized MAE)** — 정규화 절대오차. **"평균 크기를 얼마나 맞췄나"**. **표본 깊이에 민감**(세포↑ → 안정↑). 하지만 2025에서 드러난 **"MAE 함정"** — 아무것도 안 하는 예측이 절대오차 지표에선 강할 수 있음 → NMAE만 좇으면 판별을 희생한다.
- **fidelity** — 예측 프로파일이 truth의 전체 구조와 **얼마나 충실히** 일치하는가. **크기는 크지만 구분력이 약한** 축(GSE에서 전부 ~4). "점수를 커 보이게" 하지만 모델 선택엔 별 도움이 안 될 수 있음.
- **reach** — 진짜 DE 유전자를 **얼마나 도달/회수**했는가(recall 성격). 대체로 부차적이지만, 놓친 신호의 양을 본다.
- **Jaccard** — 예측 DE 유전자 집합과 truth DE 집합의 **교집합/합집합**. **Wilcoxon 기반이라 세포 수(검정력)에 극도로 민감.** Jiang24 Overall의 사실상 주인. **여기서 지면 Overall이 무너진다.**

> **그래서 무엇에 집중해야 하나 (실전 지침)**:
> 1. <b>데이터셋마다 "지배축"이 다르다는 걸 전제로 하라.</b> 새 세포주류(Jiang24 성격) 리더보드에서는 <b>Jaccard(DE 집합 겹침)</b>가 승부처다. 배치/맥락 유사류(GSE 성격)에서는 <b>PDS(판별)</b>가 승부처다.
> 2. **Overall 한 숫자에 속지 마라.** 항상 6축을 펼쳐 보고, "이 1등이 어느 축 덕분인지"를 확인하라(leave-one-axis-out 사고). 한 축 잘해서 얻은 1등은 조건이 바뀌면 사라진다.
> 3. **fidelity·MSE 같은 "커 보이지만 안 갈리는" 축에 자원을 쓰지 마라.** 판별을 실제로 만드는 축(Jaccard, PDS)을 개선하는 데 집중하라.
> 4. **NMAE↔Jaccard의 상충을 기억하라**(6.3). 한쪽을 밀면 다른 쪽이 상한다. 최적점은 **shadow-CV로 찾아야** 하고 가정하면 안 된다.

---

## 6. New 확장에서 추가로 뽑아낸 결과 (강건성·민감도)

여기부터는 Core를 넘어 **추가로 유도한** 결과다. "우리 결론을 얼마나 믿어도 되나"의 답이다.

### 6.1 온도(tau) 민감도 — winner는 온도에 따라 바뀐다

전체 frozen 범위의 조건별 winner:

| 데이터셋 | tau 0.01 | 0.03 | **0.1 기준** | 0.3 | 1.0 |
|---|---|---|---|---|---|
| Jiang24 | STACK -0.494 | STATE -0.792 | **scPRINT2 -0.760** | PCA -0.984 | UCE -0.648 |
| GSE270828 | TranscriptFormer 0.779 | UCE 0.770 | **PCA 0.753** | scPRINT2 0.775 | STATE 0.738 |

**두 데이터셋 모두 다섯 tau에서 winner가 전부 다르다.** 즉 "누가 최고 표현인가"는 온도에 민감하다. 다만 **평균 순위**로 보면 안정적 상위가 있다 — GSE는 **PCA(평균순위 2.0, Overall 0.719±0.046)**, Jiang24는 어느 tau에서도 단독 1등은 아니지만 **raw가 평균 순위로 가장 안정적 상위권**. → **단일 조건 1등 ≠ 전 조건 안정성.**

### 6.2 난수(seed) 반복성 — 단일 표본을 확정으로 읽지 마라

| 데이터셋 | 2024 | 2025 | **2026 기준** | 2027 |
|---|---|---|---|---|
| Jiang24 | TranscriptFormer -0.752 | STACK -0.515 | **scPRINT2 -0.760** | scPRINT2 -1.079 |
| GSE270828 | STATE 0.736 | raw 0.603 | **PCA 0.753** | raw 0.733 |

**어느 표현도 4개 seed 전체에서 winner를 지키지 못했다.** Jiang24 scPRINT2는 2/4 seed에서 1등(평균순위 2.0), GSE raw도 2/4(평균순위 2.0). → **단일 multinomial 표본의 점수차를 확정적 우열로 해석하면 안 된다.** 반드시 **여러 seed 평균 ± 불확실성**으로 보고해야 한다.

### 6.3 예측 세포 수 민감도 — 가장 중요한 추가 발견

세포 수는 "평가 반복 수"가 아니라 **제출 예측 분포의 Monte Carlo 표본 깊이**를 바꾼다. 그런데 이게 축마다 **정반대 방향**으로 작용한다. Jiang24 예:

| 표현 | cells | NMAE | fidelity | Jaccard | **Overall** |
|---|---:|---:|---:|---:|---:|
| scPRINT2 | 25 | **-6.000** | 0.076 | **+0.048** | -0.998 |
| scPRINT2 | 50 | -2.508 | 0.310 | -2.363 | -0.760 |
| scPRINT2 | 100 | **+0.029** | 1.561 | **-15.052** | -2.192 |
| PCA | 25 | -6.000 | 0.134 | -0.796 | -1.139 |
| PCA | 50 | -1.447 | 0.626 | -4.999 | -0.927 |
| PCA | 100 | -0.169 | 1.465 | **-16.861** | -2.521 |

> **메커니즘(꼭 이해할 것)**: 세포 수를 늘리면
> - **NMAE·fidelity는 좋아진다** (평균 추정이 안정되어 크기를 잘 맞춤). 25세포에선 NMAE가 **바닥(≈-6.0)에 붙어 모든 표현이 동점** → 구분력 상실.
> - **Jaccard는 나빠진다** (Wilcoxon 검정력이 커져서, 예측 DE 집합이 틀렸다는 걸 *더 잘 잡아낸다*). 100세포면 -15~-17로 붕괴.
>
> 즉 **"몇 개의 세포를 제출하느냐"는 중립적 선택이 아니라, 어느 축을 유리하게/불리하게 만들지 고르는 손잡이다.** cells=100이 나쁜 게 아니라, 분포·DE 축이 표본 깊이에 민감하다는 뜻. (GSE는 DE축이 0이라 이 효과가 없고, PDS·fidelity만 완만히 움직인다.)

이 발견은 실전에서 직접적이다 — **제출 세포 수도 튜닝 대상**이며, Jaccard가 승부처인 리더보드에서는 무작정 세포를 늘리는 게 독이 될 수 있다.

### 6.4 leave-one-axis-out & 분산분해 — "Overall의 주인" 재확인

(수치는 Q2에 정리.) 요점: **Jiang24 = Jaccard가 크기·순위 모두 지배**, **GSE = fidelity가 크기·PDS가 순위 지배**. 한 축 제거로 Jiang24 11/12·GSE 12/12 조건에서 winner가 흔들린다. → **"6축 평균"이라는 이름과 달리, 실제로는 사실상 1축이 결과를 만든다.**

### 6.5 rank stability — 누가 "조건 전반에서" 믿을 만한가

단일 1등이 아니라 **모든 조건에 걸친 평균 순위·표준편차**로 본 안정적 상위:

- **GSE270828**: **PCA**가 가장 안정적(예: tau군 평균순위 2.0, cell_depth군 winner 2회). UCE·STATE가 뒤따름. frozen들이 raw보다 나을 때도 있으나 PCA를 못 넘음.
- **Jiang24**: 진영이 갈려 절대강자 없음. **scPRINT2**가 전체 frozen 중 평균순위 최상위권(seed군 평균순위 2.0, cell_depth군 winner 2회)이지만 표준편차가 크다. raw는 "가장 덜 나쁜" 안정적 상위.

→ **"안정적으로 상위"와 "특정 조건 1등"은 다른 개념**이며, 전략은 전자(평균순위+분산)로 세워야 한다.

---

## 7. 전략적 시사점 — 그래서 무엇에 집중할까

1. **효과 라이브러리를 키우고 정교화하는 데 투자하라 (가장 확실한 레버).** 점수의 큰 도약은 여기서 나온다. Replogle 커버리지를 넓히고(더 많은 표적·세포주), 미커버 유전자에 실제 ESM-2 KNN을 붙이는 것이 frozen 임베딩을 바꾸는 것보다 기대이익이 크다.
2. **frozen 임베딩은 "무료면 앙상블 후보, 유료면 보류".** 현재 세팅에서 통계 기준선(raw/PCA)을 안정적으로 못 이긴다. 쓰더라도 **여러 표현의 앙상블/합의**로 쓰고, 단일 신경망 표현에 베팅하지 말 것.
3. **승부처 축을 정조준하라.** 새 세포주류에서는 **Jaccard**(=DE 집합을 맞히기: rank-preserving sparsification, top-|logFC|만 전이, per-gene 신뢰가중), 배치/맥락류에서는 **PDS**(=판별: source-agreement 기반 confidence)를 직접 개선.
4. **튜닝은 반드시 shadow-CV로, 여러 seed 평균으로.** scale·tau·cells·shrink는 모두 축들을 상충시키는 손잡이다. 최적점을 **가정하지 말고** 별도 holdout에서 미리 고정한 뒤 여러 seed 평균과 불확실성으로 보고하라(post-hoc 최고점 보고는 선택 편향).
5. **제출 세포 수를 실험 변수로 취급하라.** Jaccard가 승부처면 세포 수를 무작정 늘리지 말 것(6.3).
6. **항상 6축을 펼쳐서 보고하라.** Overall 한 숫자로 결론 내리지 말고, "이 결과가 어느 축 덕/탓인지"를 leave-one-axis-out으로 확인하는 습관.

---

## 8. 정직한 한계와 올바른 사용법 (반드시 함께 읽을 것)

- **공식 점수가 아님**: 모든 값은 공개 `cell-eval2==0.16.0:vcc2026`를 외부 데이터에 돌린 **dataset-local anchor 점수**다. 챌린지 서버/리더보드 공식 점수와 다르다.
- **데이터셋 간 비교 금지**: anchor가 다르고 정의된 축 개수도 다르다(Jiang24 5축 유효, GSE 2축 유효). Jiang24 -0.76과 GSE +0.75를 직접 비교하면 안 된다.
- **GSE의 0축 4개는 오류가 아니라 fallback**: HAR 표적이 exact-gene Replogle 라이브러리에 없어서 MSE/NMAE/reach/Jaccard가 정의되지 않아 0으로 채워진 것. GSE strict 결과는 사실상 **missing-target 음성대조**다.
- **확장 스윕은 post-hoc 탐색**: truth를 본 뒤 돌린 민감도 분석이다. "최고 tau/세포수 조건"을 새로운 unbiased winner로 선언하면 **선택 편향**이 생긴다. 후속 확증은 별도 holdout에서 tau·cells를 **미리 고정**하고 여러 seed 평균으로 해야 한다.
- **frozen "동점"의 함정**: nearest에서 여러 표현이 같은 점수를 내는 것은 표현이 동등해서가 아니라 **같은 source를 argmax로 골라서**다.
- **해석 경계(B_FM_AUGMENTED)**: checkpoint 사전학습이 이 데이터를 봤는지는 `unknown`이므로, 이 결과는 "사전학습-청정 zero-shot"이 아니라 **"파운데이션 모델 보강"** 등급으로 읽어야 한다.

---

## 9. 재현 · 감사 정보

- **Core 설계 이유·파이프라인**: `reports/TWO_EXPERIMENT_FINAL_REPORT_2026-08-25.md` (3장), `docs/EXTERNAL_FROZEN_MODELS.md`
- **Core 원자료**: 실험1 `data_shadow/two_dataset_vcc2026_metrics/comparison_wide.csv`; 실험2 `data_experiments/external_frozen_matrix/shadow-existing/comparison_wide.csv`; strict `.../replogle-only/comparison_wide.csv`
- **New 확장**: `new/metric_robustness_20260825/` — 설계 `EXPERIMENT_DESIGN_KO.md`, 수행 `WHAT_WAS_RUN_KO.md`, 결과 `RESULTS_SUMMARY_KO.md`
- **New 분석표**: `analysis/combined_gpu_and_reference_results.csv`(192행), `axis_contributions_long.csv`, `dominant_axis_per_result.csv`, `leave_one_axis_out_{ranks,winner_summary}.csv`, `overall_variance_decomposition.csv`, `rank_stability.csv`, `source_{similarities_and_weights,weight_concentration}.csv`, `analysis_summary.json`
- **무결성**: 신규 GPU 144/144 `status=ok`, 통합 192행 중복 0, 비유한 metric 0, Overall 재계산 오차 ≤ 2.22e-16, 분산 기여율 그룹합 = 1.0. 감사: `analysis/validation_report.json`
- **재실행**: `bash new/metric_robustness_20260825/run_extensions.sh` (행·조건 단위 checkpoint resume). 감사만: `.venv/bin/python new/metric_robustness_20260825/scripts/validate_results.py`

---

### 부록 A. 실험 1 전체 방법 비교 (Jiang24, Overall)

| representation / method | Overall |
|---|---:|
| raw / no_effect | -1.343 |
| raw / global_mean | -0.846 |
| raw / gwps_direct | -1.540 |
| raw / gwps_nearest | -1.062 |
| raw / gwps_weighted | -0.979 |
| PCA / gwps_weighted | -0.927 |
| STATE / gwps_nearest | -0.989 |
| STATE / gwps_weighted | -1.475 |
| STACK / gwps_nearest | -0.989 |
| STACK / gwps_weighted | -1.703 |

(GSE270828 실험1: raw no_effect 0.606 → nearest 0.753; STATE/STACK/PCA nearest 모두 0.753로 동점 = 같은 source argmax.)

### 부록 B. strict Replogle-only 보조 (Jiang24 dual-source 8타깃, weighted Overall)

| representation | Overall |
|---|---:|
| **raw** | **-0.145** |
| scPRINT2 | -0.182 |
| STACK | -0.210 |
| UCE | -0.220 |
| STATE | -0.246 |
| TranscriptFormer | -0.253 |
| PCA | -0.579 |

→ 가장 엄격한 외부-반응 세팅에서도 **raw(통계)가 최고**. frozen이 source weighting을 개선했다는 근거 없음. (K562↔RPE1 교차-source cosine ≈ 0.11, sign agreement ≈ 0.54, 선택된 전역 lnFC scale = 0.25 — 세포주 간 효과 전이가 근본적으로 약함을 시사.)

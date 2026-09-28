<!-- Original Korean write-up kept verbatim (file: VCC2026_공식_챌린지_완전해설_및_전략_2026-08-22.md); only the file name was changed, LaTeX escapes that had been corrupted into control characters were repaired, and \[ \] / \( \) math delimiters were converted to $$ / $ for GitHub rendering; four currency amounts were written as "USD …" instead of "$…" so they are not parsed as math. -->

# Virtual Cell Challenge 2026
## 공식 Task · Data · Evaluation · Leaderboard 완전 해설 + 2026 대응 전략

> **기준일:** 2026-08-22 (KST)  
> **대상 사이트:** https://virtualcellchallenge.org/  
> **핵심 페이지:**  
> - Main: https://virtualcellchallenge.org/  
> - Data: https://virtualcellchallenge.org/data  
> - Evaluation: https://virtualcellchallenge.org/evaluation  
> - Leaderboard: https://virtualcellchallenge.org/leaderboard  
> - Rules: https://virtualcellchallenge.org/rules  
>
> **이 문서의 목적:**  
> 1. 2026 VCC가 정확히 무엇을 요구하는지 이해한다.  
> 2. 2025와 무엇이 바뀌었는지 이해한다.  
> 3. Validation / Test / Leaderboard / Evaluation이 어떻게 연결되는지 이해한다.  
> 4. 지금까지 조사한 외부 데이터셋과 2025 상위권 전략을 2026 문제에 맞게 다시 설계한다.  
> 5. 실제 참가 팀이 바로 구현을 시작할 수 있는 데이터·모델·검증 파이프라인을 제안한다.

---

# 0. 먼저 가장 중요한 결론

## 2026 VCC를 한 문장으로 말하면

> **처음 보는 세포 맥락(cellular context)에 대해서, 그 세포의 non-targeting control transcriptome만 보고 특정 유전자를 CRISPRi로 knockdown했을 때의 single-cell transcriptomic response를 zero-shot으로 예측하는 대회**이다.

2025와 가장 큰 차이는 다음이다.

### 2025

```text
H1 hESC라는 하나의 새로운 context

H1에서:
150 perturbations → 정답이 있는 training data 제공
50 perturbations  → validation leaderboard
100 perturbations → final test
```

즉 H1이라는 새로운 세포이긴 했지만,

> **H1 자체의 perturbation example 150개를 보고 적응(few-shot adaptation)할 수 있었다.**

---

### 2026

```text
Validation:
3개의 서로 다른 cell line

각 cell line에 대해:
- non-targeting control profile만 제공
- 예측할 CRISPRi target gene ID 300개 제공
- 그 cell line의 perturbation response 정답은 제공하지 않음

Final test:
validation과 다른 3개 cell line
- 최종 단계인 10월까지 비공개
```

즉,

> **새로운 cell line에서 Challenge-specific perturbation label을 하나도 보지 않고 예측해야 한다.**

이것이 2026에서 말하는 **multi-context generalization + zero-shot prediction**의 핵심이다.

---

# 1. 2026 Challenge의 공식 핵심 정보

Arc의 2026 launch announcement에서 현재 확인되는 사항은 다음과 같다.

| 항목 | 2026 |
|---|---|
| Task | **Multi-context generalization + zero-shot perturbation prediction** |
| Perturbation | **CRISPRi gene knockdown** |
| Validation context | **3개 cell line** |
| Validation에서 제공되는 것 | 각 cell line의 **non-targeting control profiles** + 예측 대상 **300 perturbation gene IDs** |
| Validation prediction condition | 최대 개념상 **3 × 300 = 900 context–perturbation 조건** |
| Challenge-specific perturbation training label | **없음** |
| Final test | validation과 **다른 3개 cell line** |
| Final test 공개 | **10월 final phase** |
| Evaluation | **6개 complementary metrics** |
| Metric normalization | 각 cell context에서 **context mean baseline ↔ real replicate experiment** 사이로 scale |
| Final overall score | **metrics × cell contexts를 동일 가중치로 평균한 단일 score** |
| Leaderboard | 현재 **submission + leaderboard open** |
| Prize pool | **USD 175,000 total** |
| Grand prize | 사전/공식 홍보 기준 **USD 100,000** |
| Launch | **2026-08-20** |
| Launch 당시 등록자 | **1,800명 이상** |
| Sponsors | NVIDIA / NVIDIA Healthcare, 10x Genomics, Ultima Genomics |

---

# 2. Task를 수식으로 이해하기

## 2.1 우리가 받는 정보

새로운 cell context를 $c$, knockdown할 gene을 $g$라고 하자.

Validation에서 모델은 대략 다음 정보를 가진다.

$$
X_{\mathrm{control}}^{(c)}
$$

여기서

- $X_{\mathrm{control}}^{(c)}$ = 해당 cell line의 non-targeting control single-cell transcriptome population
- $g$ = knockdown해야 할 target gene ID

그리고 예측해야 하는 것은

$$
X_{\mathrm{perturbed}}^{(c,g)}
$$

이다.

즉 함수는

$$
f\left(
X_{\mathrm{control}}^{(c)},
g
\right)
\rightarrow
\widehat{X}_{\mathrm{perturbed}}^{(c,g)}
$$

이다.

---

## 2.2 2026의 진짜 어려운 점

training 중에는 특정 다른 context $c_1,c_2,\dots$에서 perturbation response를 보았다고 하자.

그런데 leaderboard context는

$$
c^*
$$

라는 처음 보는 cell line이다.

그리고 이 context에서는

$$
X_{\mathrm{control}}^{(c^*)}
$$

만 제공된다.

따라서 모델이 해야 할 일은

```text
"이 gene은 일반적으로 이런 일을 한다"
+
"지금 들어온 cell은 이런 상태다"
+
"이 cell 상태에서는 이 gene effect가 이렇게 달라질 것이다"
```

를 결합하는 것이다.

즉 단순 gene memorization 문제가 아니다.

---

# 3. 2026의 핵심 개념: Multi-context Generalization

## 왜 그냥 perturbation prediction이라고 하지 않는가?

같은 gene을 억제해도 모든 cell에서 똑같은 response가 나오지 않는다.

예를 들어 가상의 gene $G$를 억제했을 때:

```text
K562:
A ↓↓↓
B ↑
C 변화 없음

HepG2:
A ↓
B 변화 없음
C ↑↑

A549:
A ↓↓
B ↑
C ↑
```

일 수 있다.

여기에는 두 종류의 효과가 섞여 있다.

### ① Shared perturbation effect

cell과 무관하게 비교적 공통적으로 나타나는 효과.

$$
\Delta_{\mathrm{shared}}(g)
$$

### ② Context-specific deviation

현재 cell state 때문에 추가적으로 생기는 효과.

$$
\Delta_{\mathrm{context}}(c,g)
$$

따라서 우리가 이상적으로 학습하고 싶은 구조는

$$
\Delta(c,g)
=
\Delta_{\mathrm{shared}}(g)
+
\Delta_{\mathrm{context}}(c,g)
$$

이다.

그리고 최종 prediction은

$$
\widehat X_{\mathrm{pert}}
=
X_{\mathrm{control}}
+
\widehat{\Delta}(c,g)
$$

형태로 생각할 수 있다.

이 식은 2025 2위의 **residual / delta prediction**과도 정확하게 연결된다.

---

# 4. 2025와 2026의 차이 — 반드시 이해해야 한다

| 구분 | 2025 | 2026 |
|---|---|---|
| Generalization axis | 새로운 H1 context에서 **unseen perturbation** | 새로운 여러 context에서 **zero-shot perturbation** |
| Cell context 수 | H1 중심 | Validation 3 + Final Test 3 |
| Challenge-specific train labels | H1 150 perturbation 공개 | **target context perturbation labels 없음** |
| Validation | H1의 50 gene | 3개 unseen cell line × 300 perturbation |
| Final test | H1 100 gene | 또 다른 unseen 3 cell line |
| Adaptation | H1 labels로 fine-tuning 가능 | **label-based target-context fine-tuning 불가능** |
| 평가 | PDS + DES + MAE | **6 complementary metrics** |
| Metric weighting | 2025 metric 구조가 특정 metric gaming 가능 | metric × context **equal-weight holistic score** |
| 가장 중요한 능력 | perturbation transfer | **cell-context transfer + perturbation transfer** |
| 좋은 데이터 | public Perturb-seq + H1 adaptation | **multi-context CRISPRi + context representation** |

---

# 5. 왜 2026이 2025보다 훨씬 어렵나?

2025에서는 다음이 가능했다.

```text
H1 train response
↓
"H1이라는 세포는 perturbation에 이렇게 반응하는구나"
↓
H1 validation/test
```

2026에서는 target context에 대해:

```text
new cell line
↓
non-targeting control만 있음
↓
perturbation response는 0개
```

이다.

따라서 모델은

> **control transcriptome 자체로 cell context를 읽어야 한다.**

이것 때문에 2026에서는 `context encoder`의 중요도가 훨씬 높아졌다.

---

# 6. Data 페이지를 어떻게 이해해야 하나?

## 6.1 Validation dataset의 역할

현재 공개된 validation dataset은 leaderboard용이다.

공식 launch 설명 기준:

```text
Validation
├── Cell line A
│   ├── non-targeting control profiles
│   └── 300 CRISPRi target gene IDs
│
├── Cell line B
│   ├── non-targeting control profiles
│   └── 300 CRISPRi target gene IDs
│
└── Cell line C
    ├── non-targeting control profiles
    └── 300 CRISPRi target gene IDs
```

중요한 것은 각 context의 실제 perturbation response는 숨겨져 있다는 점이다.

---

## 6.2 우리가 실제로 해야 할 prediction unit

하나의 예측 condition을

```text
(cell context c, target gene g)
```

라고 생각하면 된다.

3개의 cell context 각각에서 300개 perturbation을 예측하므로 개념적으로

$$
3\times300=900
$$

개의 context–target response를 만들어야 한다.

### 단, 주의

공식 launch 문구는

> “300 perturbations per cell line”

이라고 한다.

현재 static 웹 검색으로는 **세 cell line의 300 target gene set이 완전히 동일한지**까지 확인되지 않았다.

따라서 실제 validation package를 받으면 반드시:

```python
targets_A = set(...)
targets_B = set(...)
targets_C = set(...)

len(targets_A & targets_B & targets_C)
```

를 계산해야 한다.

---

# 7. Non-targeting control이란?

CRISPRi 실험에서는 sgRNA를 세포에 넣지만 특정 gene을 실질적으로 target하지 않는 control을 사용할 수 있다.

이것이 <b>non-targeting control(NT control)</b>이다.

쉽게 말하면:

> “CRISPRi 실험 과정은 거쳤지만 특정 target gene을 억제하지 않은 baseline cells”

이다.

2026에서는 이것이 매우 중요하다.

왜냐하면 unseen context에 대해 우리가 알고 있는 biological state의 가장 직접적인 정보가 바로 이 control transcriptome이기 때문이다.

---

# 8. Control이 왜 단순 평균 이상의 의미를 가지나?

새로운 cell line의 control population에서 다음을 읽을 수 있다.

- 어떤 genes가 baseline에서 높게/낮게 발현되는가
- cell cycle state
- stress state
- lineage
- tissue-specific programs
- signaling activity
- metabolic state
- target gene 자체의 baseline expression
- pathway neighbors의 baseline activation
- population heterogeneity

즉

$$
X_{\mathrm{control}}
$$

은 단순한 baseline 숫자가 아니라

> **새로운 cell context를 설명하는 prompt**

처럼 볼 수 있다.

---

# 9. 2026을 “cellular prompt → perturbation response” 문제로 볼 수 있다

개념적으로:

```text
Prompt 1:
new cell의 control cells

Prompt 2:
target gene identity

Model:
"이 cell state에서 이 gene을 억제하면?"

Output:
perturbed cell population
```

이는 Arc의 최신 `State/Stack` 계열이 강조하는 **set-of-cells / in-context generalization** 방향과도 잘 맞는다.

---

# 10. Final Test 데이터

공식 launch 기준:

- Validation과 **다른 3개의 cell line**
- 10월 final phase까지 held out

즉 validation의 3개 cell line에 과적합하면 안 된다.

구조적으로 우리가 대비해야 할 것은:

```text
현재:
Validation cell A / B / C

10월:
완전히 새로운 Test cell D / E / F
```

이다.

---

# 11. 이것이 Leaderboard overfitting을 위험하게 만드는 이유

Validation leaderboard를 계속 보면서:

```text
A cell에서 score 조금 올라감
→ hyperparameter 변경
→ leaderboard 다시 제출
→ 또 변경
```

을 반복하면 모델이 암묵적으로

```text
A/B/C validation contexts
```

에 최적화된다.

하지만 final은

```text
D/E/F
```

이다.

그래서 2026에서는 **public leaderboard를 하나의 noisy validation signal로만 보고, local leave-one-context-out CV를 더 신뢰해야 한다.**

---

# 12. Evaluation — 2026에서 공식적으로 확인된 핵심

공식 launch 설명에서 확인된 것은 다음이다.

> **6개의 complementary metrics**

를 사용한다.

그리고 각 metric은 cell context마다

> **cell context mean과 real replicate experiment 사이로 scale**

된다.

최종 overall score는

> **6 metrics × cell contexts에 대한 단일 unweighted average**

이다.

---

# 13. 3 context × 6 metric 구조

Validation context가 3개이므로 개념적으로:

```text
Cell A
  Metric 1
  Metric 2
  Metric 3
  Metric 4
  Metric 5
  Metric 6

Cell B
  Metric 1
  ...
  Metric 6

Cell C
  Metric 1
  ...
  Metric 6
```

총

$$
3\times 6=18
$$

개의 metric–context component가 overall score에 동등하게 기여한다고 이해하면 된다.

---

# 14. “cell context mean” normalization은 무엇을 의미하는가?

가장 단순한 baseline은:

> perturbation별 차이를 거의 예측하지 않고 해당 cell context의 평균적인 상태를 반환하는 것

이다.

2025에서도 평균 prediction은 특히 MAE에서 강했다.

2026에서는 이를 명시적인 하한(anchor)으로 사용한다.

개념적으로:

```text
0에 가까운 기준:
cell-context mean 수준

1에 가까운 기준:
실제 독립 replicate experiment가 서로를 예측할 수 있는 수준
```

으로 해석할 수 있다.

---

# 15. 왜 replicate experiment가 upper anchor인가?

single-cell biological experiment에는 완벽한 deterministic truth가 없다.

같은 perturbation을 독립적으로 두 번 측정해도:

- sampling noise
- technical noise
- cell heterogeneity
- sequencing noise
- guide efficacy variation

때문에 완전히 동일하지 않다.

따라서

> “실험 A가 실험 B를 얼마나 잘 재현할 수 있는가?”

가 현실적인 ceiling을 제공한다.

이는 모델에게 **불가능한 완벽함을 요구하지 않는 평가**라는 장점이 있다.

---

# 16. Normalized scoring을 직관적으로 이해하기

정확한 공식 transformation은 Evaluation UI/organizer evaluator를 그대로 따라야 하지만,
개념적 해석은 다음과 같다.

```text
baseline -------- model -------- replicate
   0               ?               1
```

model이 mean baseline보다 좋을수록 0보다 위로 간다.

replicate experiment 수준에 접근할수록 1에 가까워진다.

### 중요

metric마다

- higher is better
- lower is better

방향이 다를 수 있으므로 **직접 임의로 `(score-baseline)/(ceiling-baseline)`을 구현하지 말고 공식 evaluator를 사용해야 한다.**

---

# 17. 2026의 정확한 6개 metric 이름에 대한 검증 경계

현재 공개 launch announcement에서는:

> **six complementary metrics**

라고 명확히 확인된다.

하지만 현재 `virtualcellchallenge.org/evaluation`은 JavaScript SPA로 동적으로 렌더링되어,
외부 static crawler에서는 상세 metric label들을 직접 회수하지 못했다.

따라서 이 문서에서는 **6개의 정확한 metric 이름을 추측해서 확정하지 않는다.**

이것은 매우 중요한 원칙이다.

### 우리가 확실히 아는 것

- 6개다.
- 여러 biological 측면을 보완적으로 평가한다.
- context별로 normalize한다.
- context mean과 real replicate를 anchor로 쓴다.
- 최종은 equal-weight average다.

### 반드시 실제 Evaluation 페이지/official evaluator에서 복사해야 하는 것

- metric 1~6의 정확한 이름
- metric별 방향
- top-N / DE threshold 같은 parameter
- aggregation level
- clipping 여부
- normalization 공식
- missing gene 처리
- cell count 처리
- replicate anchor 계산 방식

---

# 18. Arc `cell-eval`과의 관계

Arc는 공개 evaluation package인

```text
ArcInstitute/cell-eval
```

을 유지하고 있다.

현재 package:

```text
cell-eval 0.8.2
Python >= 3.11
```

이다.

주요 기능:

```text
cell-eval prep
cell-eval run
cell-eval baseline
cell-eval score
```

---

# 19. `cell-eval prep`

예측 AnnData를 VCC-compatible 형태로 정리하는 기능이 있다.

예:

```bash
cell-eval prep \
  -i prediction.h5ad \
  -g expected_genelist
```

기능:

- 필요한 정보만 유지
- compression
- naming convention 조정
- gene list compatibility 확인

---

# 20. `cell-eval run`

실제 prediction과 real data가 둘 다 있을 때:

```bash
cell-eval run \
  -ap prediction.h5ad \
  -ar ground_truth.h5ad \
  --num-threads 64 \
  --profile full
```

처럼 evaluation 가능하다.

---

# 21. 중요한 함정 — 현재 `cell-eval --profile vcc`를 맹신하지 말 것

현재 공개 `cell-eval` 코드의 `vcc` profile은 역사적으로 2025용 metric profile을 포함해 왔다.

따라서:

> **현재 package의 `--profile vcc`가 곧바로 2026 공식 six-metric evaluator와 동일하다고 가정하면 안 된다.**

2026 참가 시에는 반드시:

1. Challenge Evaluation page
2. 공식 submission guide
3. 공식 evaluator/config/version

을 기준으로 package/version/config를 pin해야 한다.

---

# 22. `cell-eval`이 지원하는 metric universe

공개 코드에서는 다음과 같은 다양한 metric family가 존재한다.

### Expression-space

- Pearson delta
- MSE
- MAE
- MSE delta
- MAE delta
- perturbation discrimination
- clustering agreement
- energy distance 계열

### Differential-expression

- overlap@N
- precision@N
- Spearman correlations
- direction match
- significant gene recall
- ROC AUC
- PR AUC

즉 2026의 6 metrics도 이런 서로 다른 생물학적 축을 함께 보려는 방향으로 이해할 수 있다.

단,

> **이 목록에서 어떤 정확한 6개가 2026 leaderboard에 쓰이는지는 공식 Evaluation page 값을 그대로 확인해야 한다.**

---

# 23. 2026 scoring redesign의 철학

2025에서 문제가 있었다.

## 2025

MAE는 mean baseline이 너무 강했다.

결과적으로 거의 모든 경쟁 모델이 MAE에서는 baseline을 이기지 못했고,
상위 팀은 PDS/DES에 집중하게 되었다.

즉:

```text
Leaderboard optimization
→ 특정 metric gaming
```

이 가능했다.

---

# 24. 2026은 이것을 어떻게 개선했나?

### ① metric 수 확대

3 → 6

### ② 각 context마다 평가

한 cell context에서만 잘하면 부족.

### ③ baseline ↔ experimental reproducibility로 normalization

metric scale 차이를 줄인다.

### ④ equal weighting

특정 metric에 지나치게 집중하기 어렵다.

즉 2026은

> **“한 숫자를 잘 맞추는 모델”이 아니라 여러 biological fidelity 축에서 균형 있게 잘 맞는 generalist model**

을 더 선호한다.

---

# 25. Leaderboard의 의미

현재 공식 launch announcement에서:

> **Submissions and the leaderboard are now open.**

이라고 확인된다.

현재 leaderboard는 **validation dataset에 대한 public feedback** 역할을 한다.

---

# 26. Leaderboard score는 무엇을 의미하나?

scoring 설명상 기본 구조는:

```text
모델 submission

↓ hidden validation ground truth와 비교

Cell A × 6 metrics
Cell B × 6 metrics
Cell C × 6 metrics

↓ 각 context / metric normalization

↓ equal-weight aggregation

Overall Score

↓
Leaderboard ranking
```

이다.

---

# 27. “Leaderboard의 정확한 UI 컬럼”에 대한 검증 경계

`/leaderboard` 페이지 역시 현재 JavaScript SPA이므로 static crawler에서는 실제 테이블 rows/columns가 렌더링되지 않는다.

따라서 이 문서에서는:

- 현재 순위
- 현재 1위 팀
- 정확한 표시 column 명
- 개별 metric breakdown UI

를 보지 않고 임의로 적지 않는다.

### 공식적으로 확인된 것

- leaderboard는 열려 있다.
- submission이 가능하다.
- validation score가 live leaderboard에 반영된다.
- scoring은 six metrics × contexts의 normalized equal-weight overall이다.

실제 화면에서 보이는 세부 column/필터는 로그인 후 live UI를 기준으로 확인해야 한다.

---

# 28. Leaderboard를 어떻게 사용해야 하나?

좋은 사용:

```text
local OOD CV에서 좋아짐
↓
leaderboard에서도 대체로 좋아지는지 확인
```

나쁜 사용:

```text
leaderboard 0.002 상승
↓
모든 architecture를 그 방향으로 변경
↓
다시 제출
↓
validation context에 과적합
```

---

# 29. 2026 leaderboard에서 특히 위험한 이유

Final test는 validation과 다른 3 contexts다.

따라서:

$$
\operatorname{argmax}
\text{Validation Leaderboard}
$$

가 반드시

$$
\operatorname{argmax}
\text{Final Test Generalization}
$$

가 아니다.

오히려 내부적으로 여러 held-out contexts에 안정적인 모델이 더 중요하다.

---

# 30. 우리가 지금까지 모은 데이터셋 전략이 2026에 얼마나 맞았나?

놀랍게도 상당히 정확했다.

우리가 미리 잡은 핵심은:

1. GSE281048 — multi-context CRISPRi
2. Replogle — genome-scale gene perturbation backbone
3. Marson primary T — primary-cell OOD
4. scBaseCount/CELLxGENE — context encoder
5. VCC2025 H1 — OOD benchmark
6. X-Atlas — gene coverage
7. GSE264667 — extra cell contexts
8. KOLF2.1J — pluripotent context

였다.

2026 task를 알고 나니 각각의 역할이 더 명확해졌다.

---

# 31. 1순위: GSE281048 / Jiang24 / Mixscale

## 왜 2026에 가장 직접적인가?

2026 문제는:

> 새로운 cell context에서 gene perturbation response를 예측

이다.

GSE281048은 같은 pathway perturbation을

```text
6 cell lines
×
5 signaling contexts
```

에서 반복한다.

즉 모델이 직접:

```text
shared perturbation effect
+
context-specific deviation
```

을 학습할 수 있다.

---

# 32. GSE281048로 “가짜 2026 대회”를 만들 수 있다

예:

```text
Train:
A549
MCF7
HT29
HAP1
K562

Hidden target:
BxPC3
```

BxPC3 perturbation response를 training에서 모두 제거.

모델에게는:

```text
BxPC3 NT control
+
target gene IDs
```

만 제공.

그다음 실제 BxPC3 perturbed data를 hidden GT로 사용.

이 구조가 2026 validation과 거의 동일하다.

---

# 33. Replogle 2022의 역할

Replogle은 context 수가 적지만 gene perturbation coverage가 매우 크다.

대표적으로:

```text
K562 genome-scale
~9,867 targets
>2.5M cells 전체 규모
```

즉 모델이

> “gene G를 억제하면 일반적으로 어떤 downstream response가 생기는가?”

를 배우는 **gene-effect backbone** 역할을 한다.

---

# 34. X-Atlas/Orion의 역할

X-Atlas는 약

```text
18,903 protein-coding targets
```

를 CRISPRi로 커버한다.

따라서 2026 target gene이 public perturbation corpus에서 드물 때:

> “이 gene을 실제로 CRISPRi한 response를 어디선가 본 적 있는가?”

의 coverage를 크게 높인다.

단점:

```text
HCT116
HEK293T
```

두 context 중심.

그래서 **gene prior용으로 강하고 context-generalization 단독 데이터로는 약하다.**

---

# 35. GSE264667의 역할

Jurkat + HepG2 CRISPRi.

적은 비용으로:

- immune/T-cell-like context
- liver-like context

를 추가한다.

Context encoder + perturbation transfer를 넓히는 데 매우 효율적이다.

---

# 36. Marson primary T-cell CRISPRi의 역할

cell line만 잘 맞는 모델을 벗어나야 한다.

Primary human CD4+ T cell에서 genome-scale CRISPRi가 가능하면:

```text
cell-line trained model
→ primary cell zero-shot
```

을 시험할 수 있다.

이는 final 3 contexts가 public cancer lines와 매우 다를 경우를 대비하는 강력한 훈련이 된다.

---

# 37. scBaseCount / CELLxGENE Census가 2026에서 더 중요해진 이유

2026 validation context에서는 perturbation label이 없다.

즉 새로운 cell을 이해하는 가장 중요한 source가:

$$
X_{\mathrm{control}}
$$

이다.

따라서 context encoder가 매우 강해야 한다.

대규모 observational atlas는 모델에게:

```text
이 expression pattern이면
어떤 lineage/state/tissue와 가까운가?
```

를 학습시킨다.

---

# 38. Context encoder를 이렇게 생각하면 쉽다

```text
새로운 validation cell의 NT cells
            ↓
        Set Encoder
            ↓
      context embedding z_c
```

이 embedding이:

- K562에 가까운지
- HepG2에 가까운지
- stem-like인지
- immune-like인지
- stressed인지
- cycling인지

를 함축해야 한다.

---

# 39. VCC 2025 H1의 역할도 바뀐다

2025 H1 data를 단순 train data로 쓰는 것보다:

```text
Train:
모든 external contexts

Test:
H1 perturbation completely held out
```

로 사용하는 것이 2026 준비에 더 유용하다.

즉 H1을 **shadow zero-shot test**로 삼는다.

---

# 40. KOLF2.1J iPSC의 역할

H1을 holdout으로 남겨두되,
pluripotent biology를 전혀 못 배우는 문제를 막는다.

```text
KOLF2.1J iPSC perturbations
↓
pluripotent perturbation rules 학습

H1
↓
unseen pluripotent validation
```

이 된다.

---

# 41. 2025 1위 전략에서 가져갈 것

## xTrimoSCPerturb의 핵심 교훈

- pure foundation model만 믿지 않았다.
- public perturbation data 활용
- protein embedding
- pseudo-bulk
- DEG frequency
- mean expression
- classical statistics + deep model hybrid

2026에도 매우 중요하다.

특히 context-generalization은 deep model 하나로 자동 해결될 것이라고 생각하면 위험하다.

---

# 42. 2025 2위 전략에서 가져갈 것

## XLearning Model X

핵심:

```text
control pseudo-bulk
+
ESM-2 gene embedding
+
simple FCN
→
Δ expression
```

이다.

2026에서는 이를 확장:

```text
target-context control embedding
+
ESM-2 target embedding
+
multi-context perturbation prior
→
Δ expression
```

로 만들 수 있다.

이건 가장 먼저 만들어야 할 neural baseline이다.

---

# 43. 2025 3위 TransPert가 오히려 2026에 더 중요해졌다

TransPert의 철학:

> 여러 reference cell line에서 측정된 perturbation effect를 target cell과의 similarity로 weighted transfer

이다.

이건 사실 2026 task에 거의 정확히 맞는다.

---

# 44. 가장 먼저 만들 classical baseline

Reference contexts를 $r=1,\dots,R$라고 하자.

target context control embedding:

$$
z_{c^*}
$$

reference context embedding:

$$
z_r
$$

similarity:

$$
w_r
=
\operatorname{softmax}
\left(
\operatorname{sim}(z_{c^*},z_r)/\tau
\right)
$$

각 gene $g$의 reference perturbation delta:

$$
\Delta_{r,g}
$$

그러면:

$$
\widehat{\Delta}_{c^*,g}
=
\sum_r
w_r\Delta_{r,g}
$$

최종:

$$
\widehat X_{\mathrm{pert}}
=
X_{\mathrm{ctrl}}^{c^*}
+
\widehat{\Delta}_{c^*,g}
$$

---

# 45. 이 baseline이 매우 중요한 이유

Deep model이 정말 generalize하는지 보려면 최소한 이것을 이겨야 한다.

이 baseline은:

- 설명 가능
- 빠름
- leaderboard feedback 해석 쉬움
- dataset leakage 발견 쉬움
- cell-context similarity 자체를 평가 가능

하다.

---

# 46. 추천 Baseline Ladder

## B0 — Control Mean

```text
prediction = target-context control
```

가장 기본.

---

## B1 — Global Gene Effect

public contexts에서 gene $g$의 평균 perturbation delta를 계산:

$$
\bar\Delta_g
=
\frac1R\sum_r\Delta_{r,g}
$$

prediction:

$$
X_{ctrl}^{c^*}+\bar\Delta_g
$$

---

## B2 — Nearest Context Transfer

target control과 가장 비슷한 reference context 하나 선택.

그 context의 gene delta를 복사.

---

## B3 — Similarity-weighted TransPert-like

여러 reference context delta를 가중 평균.

**매우 중요한 baseline.**

---

## B4 — Pseudo-bulk MLP + ESM-2

Input:

```text
context control embedding
+
ESM2(g)
+
baseline expression(g)
+
historical perturbation summary(g)
```

Output:

```text
Δ pseudo-bulk
```

---

## B5 — Hybrid Context × Gene Model

- set context encoder
- gene multi-prior encoder
- cross-attention / FiLM / gating
- shared effect + context residual

---

## B6 — Distributional Model

pseudo-bulk만 아니라

$$
p(X_{\mathrm{pert}}\mid X_{\mathrm{control}},g)
$$

를 flow/diffusion/State-style로 생성.

이 단계는 B0~B5가 잘 된 뒤 간다.

---

# 47. 내가 가장 추천하는 최종 구조

```text
                    ┌─────────────────────────┐
                    │ Target NT control cells │
                    └────────────┬────────────┘
                                 │
                           Set Encoder
                                 │
                          context z_c
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
          Context similarity           Context state features
          to reference atlas           pathway / baseline expr
                    │                          │
                    └────────────┬─────────────┘
                                 │
                           Context branch
                                 │
                                 ▼
Target gene → ESM2 ──┐       Cross interaction
          → STRING ──┼────→  Context × Gene
          → DepMap ──┤
   → PerturbAtlas ───┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
       Shared gene effect Δ_g       Context residual Δ_(c,g)
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                            Δ prediction
                                 │
                                 ▼
                 X_control + Δ = X_perturbed
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
        pseudo-bulk head                 cell-distribution head
```

---

# 48. 왜 `Shared + Residual` 구조를 추천하나?

2026 task의 본질을 architecture에 직접 반영한다.

$$
\Delta(c,g)
=
\Delta_g^{shared}
+
\Delta_{c,g}^{residual}
$$

### Shared branch가 배우는 것

- gene의 일반적인 downstream biology
- Replogle/X-Atlas가 강점

### Context residual branch가 배우는 것

- target cell state에 따른 effect modulation
- GSE281048/multi-context data가 강점

---

# 49. Gene representation은 one-hot이면 안 된다

특히 zero-shot gene까지 대비한다면:

```text
Gene ID = 1234
```

만으로는 의미를 전달할 수 없다.

추천:

$$
z_g
=
[
ESM2;
STRING;
DepMap;
PerturbAtlas;
baseline\ expression
]
$$

---

# 50. ESM-2의 역할

protein sequence semantics.

training에 특정 gene perturbation이 적어도:

> 비슷한 protein family/function의 정보를 제공

할 수 있다.

2025 2위에서 이미 강력한 힌트가 있었다.

---

# 51. STRING의 역할

target gene 주변:

- physical interaction
- functional association
- pathway neighbors

를 제공.

특히 unseen target transfer에 유용.

---

# 52. DepMap이 2026에서 특히 매력적인 이유

DepMap은:

```text
cell line × gene dependency
```

정보를 준다.

같은 gene이 cell마다 얼마나 중요한지가 다르다.

즉 바로:

$$
gene\times context
$$

prior다.

2026의 core interaction과 방향이 맞다.

---

# 53. PerturbAtlas / public DE prior

target gene마다:

- historical DEG frequency
- signed logFC
- pathway enrichment
- effect size
- cross-context variance

를 미리 만들어두면 좋다.

2025 1위가 사용한 classical statistical feature와 연결된다.

---

# 54. Control에서 꼭 뽑아야 할 features

단순 mean expression만 쓰지 말고:

### Global context features

- pseudo-bulk expression
- HVG embedding
- pathway activity
- cell-cycle score
- stress signatures
- lineage signatures

### Target-local features

- target gene baseline expression
- STRING-neighbor expression
- pathway neighbor expression
- target essentiality/dependency
- target module activity

### Population features

- cell-to-cell variance
- covariance/latent dispersion
- subclusters
- heterogeneity

---

# 55. Context encoder는 “set”으로 만드는 것이 좋다

NT cells는 순서가 없는 population이다.

따라서:

```text
cell 1
cell 2
cell 3
...
```

순서가 의미 없다.

Set Transformer / DeepSets / State embedding 계열처럼 permutation-invariant architecture가 자연스럽다.

---

# 56. State / Stack에서 배울 점

Arc의 State는 perturbation response model을:

- state embedding
- state transition

으로 분리하는 방향을 갖는다.

또 공개 State repo는 아예:

```toml
[zeroshot]
"dataset.cell_type" = "test"
```

형태로 entire context holdout을 지원한다.

즉 2026 zero-shot benchmark와 매우 잘 맞는다.

---

# 57. State를 그대로 쓰는 것보다 중요한 것

STATE baseline을 돌리는 건 좋다.

하지만 반드시 다음과 비교해야 한다.

```text
Control mean
Global Δ
Nearest-context
TransPert-like
Pseudo-bulk MLP
STATE
Our hybrid
```

Foundation model이 classical baseline을 못 이기면 구조를 다시 봐야 한다.

2025의 가장 큰 교훈이다.

---

# 58. X-Cell에서 배울 수 있는 Test-Time Adaptation 아이디어

X-Cell 연구에서는 unseen target context의 **control cells만** 사용해 self-supervised test-time adaptation(TTA)을 하는 아이디어가 제시된다.

개념:

```text
target-domain NT controls
↓
control → control self-supervision
↓
context encoder를 약하게 adaptation
↓
perturbation inference
```

### 왜 2026에 흥미로운가?

2026 validation이 정확히 target-context NT controls를 제공하기 때문이다.

### 단

**Challenge rules에서 test-time adaptation이 허용되는지 반드시 확인 후 사용해야 한다.**

label은 쓰지 않지만 challenge input으로 제공된 control을 optimization에 쓰는 것이므로 rules 해석이 중요하다.

---

# 59. TTA를 한다면 무엇을 업데이트할까?

보수적으로:

- context encoder 일부
- normalization / adapters
- small LoRA/adapters

만 update.

Gene perturbation cross-attention 전체를 target control로 update하면 biologically meaningless collapse가 생길 수 있다.

---

# 60. 2026 local validation은 random-cell split을 사용하면 안 된다

가장 나쁜 split:

```text
same K562 perturbation cells
→ random 80/20
```

이건 zero-shot generalization을 전혀 검증하지 않는다.

---

# 61. 반드시 해야 할 Split A — Leave-One-Cell-Line-Out

예:

```text
Train:
K562
RPE1
Jurkat
A549
MCF7

Test:
HepG2
```

HepG2 perturbation data는 **0개** training.

HepG2 NT만 model input.

---

# 62. Split B — Gene Zero-Shot

특정 genes를 모든 contexts에서 제거.

```text
Train:
genes 1~800

Test:
genes 801~1000
```

이렇게 해야 ESM2/STRING 등의 gene prior 효과를 볼 수 있다.

---

# 63. Split C — Double Zero-Shot

```text
unseen cell context
+
unseen target gene
```

가장 어렵지만 foundation-generalization의 진짜 test.

---

# 64. Split D — Dataset/Lab Holdout

예:

```text
Train:
Replogle
GSE264667
X-Atlas

Test:
GSE281048
```

이 split은:

- lab
- sequencing
- guide design
- batch
- preprocessing

차이까지 견디는지 본다.

---

# 65. 가장 추천하는 “Shadow VCC 2026”

공개 datasets에서 3개의 contexts를 고른다.

예:

```text
Shadow Validation:
HepG2
BxPC3
H1
```

각 context에서:

- perturbation labels 숨김
- NT controls만 모델에 전달
- 300 targets sample
- prediction 생성

그다음 hidden GT와 local evaluation.

---

# 66. 왜 3개 context를 동시에 holdout해야 하나?

실제 leaderboard가 3 contexts다.

한 context에서만 좋아지는 architecture는 위험하다.

우리 local validation도:

```text
CV Score
=
mean(
cell A,
cell B,
cell C
)
```

처럼 설계해야 한다.

---

# 67. Metric-aware training은 필요하지만 metric gaming은 피해야 한다

6 metrics가 서로 다른 biological axis를 본다면 loss도 한 방향만 쓰면 안 된다.

추천 multi-objective:

```text
L =
λ1 L_delta_expression
+ λ2 L_gene_ranking
+ λ3 L_DE_direction
+ λ4 L_effect_size
+ λ5 L_distribution
+ λ6 L_regularization
```

단 정확한 6 evaluation metrics가 확인되면 그 metric에 맞춰 다시 mapping해야 한다.

---

# 68. 가장 먼저 optimize할 것은 pseudo-bulk Δ

2025 결과에서 single-cell raw noise가 매우 컸다.

따라서 초기:

$$
\mu_{pert}
-
\mu_{control}
$$

을 안정적으로 예측하는 모델이 좋다.

---

# 69. 그다음 population distribution

평균만 맞히면:

- responder / non-responder
- cell-cycle heterogeneity
- subpopulation shift

를 못 잡는다.

따라서 후속 head에서 distributional prediction을 추가한다.

Flow Matching / State / diffusion 계열이 후보.

---

# 70. Ensemble을 강하게 추천하는 이유

2025 1위 교훈:

> Deep + classical hybrid.

2026에서는 예:

$$
\hat\Delta
=
\alpha(c,g)\hat\Delta_{\mathrm{TransPert}}
+
(1-\alpha(c,g))\hat\Delta_{\mathrm{Neural}}
$$

처럼 blend 가능.

---

# 71. α를 고정하지 말고 confidence-aware하게

예:

- reference contexts에 해당 gene 데이터가 많음
  → TransPert/statistical weight ↑

- gene이 public perturbation data에서 거의 없음
  → ESM2/STRING neural prior weight ↑

- target context가 reference manifold에서 멂
  → global shared effect shrinkage ↑

---

# 72. “reference context similarity”를 여러 방식으로 만들어야 한다

단순 cosine of all genes 하나만 쓰지 말고 ensemble.

### 후보

1. pseudo-bulk transcriptome cosine
2. PCA latent cosine
3. scBaseCount-pretrained encoder
4. pathway activity similarity
5. DepMap dependency similarity
6. gene-module-specific similarity

---

# 73. Gene-specific context similarity도 중요하다

전체 transcriptome이 비슷한 두 cell이라도
특정 pathway에 대해서는 다를 수 있다.

따라서 target gene $g$에 대해:

```text
global similarity
+
target pathway similarity
```

를 결합할 수 있다.

---

# 74. 추천 similarity formula

$$
S(c,r,g)
=
\alpha S_{\mathrm{global}}(c,r)
+
\beta S_{\mathrm{pathway}(g)}(c,r)
+
\gamma S_{\mathrm{DepMap}}(c,r,g)
$$

이후:

$$
w_r
=
softmax(S(c,r,g)/\tau)
$$

---

# 75. 데이터 수집 우선순위 — 2026 task를 알고 난 뒤

## Tier 1 — 반드시

### GSE281048
multi-context CRISPRi.

### Replogle
gene perturbation backbone.

### GSE264667
Jurkat/HepG2 context expansion.

### VCC2025 H1
OOD anchor.

### X-Atlas/Orion
target coverage.

---

# 76. Tier 2 — 매우 중요

### Marson primary T
primary-cell generalization.

### KOLF2.1J iPSC
pluripotent context.

### scBaseCount/CELLxGENE
context encoder.

---

# 77. Tier 3 — auxiliary

- Parse PBMC
- Tahoe-100M
- OP3
- PerturbFate
- sci-Plex
- other KO/CRISPRa sets

이들은 modality가 다르므로 **CRISPRi target label과 동일 취급하지 않는다.**

---

# 78. 2026 validation data를 받은 직후 제일 먼저 할 분석

## 78.1 Context identity

```text
cell line A
cell line B
cell line C
```

정확한 이름 확인.

---

## 78.2 Target overlap

```python
A ∩ B
A ∩ C
B ∩ C
A ∩ B ∩ C
```

---

## 78.3 Public coverage

각 target gene에 대해:

| Gene | Replogle | X-Atlas | GSE281048 | GSE264667 | Marson |
|---|---:|---:|---:|---:|---:|
| G1 | ✓ | ✓ | ✓ |  | ✓ |
| G2 |  | ✓ |  | ✓ | ✓ |

이 matrix를 만든다.

---

# 79. Context coverage matrix도 만든다

Validation control embedding과 external reference context들의 similarity 계산.

예:

| Validation | K562 | RPE1 | HepG2 | Jurkat | A549 | HCT116 |
|---|---:|---:|---:|---:|---:|---:|
| Cell A | .21 | .44 | **.83** | .12 | .71 | .64 |
| Cell B | .74 | .33 | .31 | **.88** | .20 | .29 |

이게 TransPert-like baseline의 출발점이다.

---

# 80. 반드시 확인할 “target baseline expression”

knockdown target gene이 해당 validation cell에서 거의 발현되지 않는다면:

> perturbation effect 자체가 작을 가능성

이 높을 수 있다.

따라서 feature:

$$
x_g^{control}
$$

를 반드시 넣는다.

---

# 81. Target knockdown efficacy를 명시적으로 모델링할 수도 있다

공개 dataset에서:

```text
baseline target expression
vs
post-CRISPRi target expression
```

으로 knockdown efficacy를 추정.

target/context에 따라 effect magnitude calibration에 활용.

---

# 82. Submission pipeline은 모델과 분리해야 한다

추천 구조:

```text
raw model output
↓
gene-order validation
↓
cell-count / population formatting
↓
AnnData construction
↓
metadata normalization
↓
cell-eval prep
↓
submission artifact
```

---

# 83. Submission 파일에서 흔한 실패

- gene order 불일치
- gene names mismatch
- duplicated genes
- wrong control label
- wrong target label
- NaN / inf
- negative count-like values
- wrong normalization space
- cell count 부족/과다
- metadata column mismatch
- compressed file corruption

모델 성능과 별개로 이런 이유로 submission fail이 날 수 있다.

---

# 84. Official sample submission을 반드시 truth로 사용

Data/Submission package가 제공하는:

```text
sample_submission
expected_gene_list
metadata schema
```

가 있다면 이것을 그대로 template으로 삼아야 한다.

우리 임의 schema를 만들면 안 된다.

---

# 85. Leaderboard experiment log를 남겨야 한다

예:

```text
submission_id
git_commit
model
external_data
CV_mean
CV_worst_context
LB_overall
notes
```

---

# 86. “CV worst-context”를 꼭 기록

평균만 보면 위험하다.

예:

```text
A = 0.8
B = 0.8
C = 0.2
mean = 0.6
```

보다

```text
A = 0.62
B = 0.61
C = 0.60
mean = 0.61
```

가 final unseen contexts에는 더 안전할 수 있다.

2026의 generalization 철학에서는 **worst-context robustness**가 중요하다.

---

# 87. 추천 model-selection criterion

local에서는 단순 mean 외에:

$$
Score_{\mathrm{select}}
=
MeanScore
-
\lambda\cdot StdAcrossContexts
$$

같이 context variance에 penalty를 줄 수 있다.

---

# 88. 2026에서 하지 말아야 할 것

## 1. Validation cell lines를 보고 그 cell line 전용 public dataset만 찾기

rules상 허용되더라도 leaderboard-specific overfitting 위험.

Final은 다른 cell lines.

---

## 2. random-cell split 성능 자랑

zero-shot과 무관.

---

## 3. X-Atlas 하나만 대량 학습

gene coverage는 좋지만 context 2개.

---

## 4. observational atlas만 크게 pretrain

cell state는 이해해도 causal perturbation response는 못 배움.

---

## 5. foundation model 성능을 baseline 없이 믿기

2025에서 이미 위험성이 증명됨.

---

## 6. CRISPRa/KO/drug를 CRISPRi와 동일 label로 합치기

modality mismatch.

---

## 7. leaderboard에 매일 맞춰 architecture 변경

final context에서 깨질 가능성.

---

# 89. 2026에서 우리가 가져가야 할 가장 현실적인 Method v1

## Stage 1 — Public CRISPRi pseudobulk library

각:

```text
(dataset, context, target)
```

에 대해:

- control mean
- perturbed mean
- Δ
- logFC
- DE genes
- n_cells
- target baseline
- knockdown strength

를 만든다.

---

# 90. Stage 2 — Context encoder

pretrain:

```text
scBaseCount
CELLxGENE
+
perturbation datasets의 controls
```

목적:

$$
X_{control}^{cells}
\rightarrow z_c
$$

---

# 91. Stage 3 — Gene encoder

```text
ESM2
STRING
DepMap
PerturbAtlas
baseline expression
```

→ $z_g$

---

# 92. Stage 4 — Statistical transfer baseline

$$
\hat\Delta_{stat}(c,g)
=
\sum_r w(c,r,g)\Delta(r,g)
$$

---

# 93. Stage 5 — Neural residual

Neural model은 statistical prediction 자체도 input으로 받게 할 수 있다.

$$
\hat\Delta
=
\hat\Delta_{stat}
+
f_\theta(z_c,z_g,\hat\Delta_{stat})
$$

이 구조는 굉장히 추천한다.

---

# 94. 왜 statistical prediction을 neural input으로 넣는가?

Deep model이 처음부터 모든 것을 다시 배우는 대신:

> “public data를 단순 transfer하면 이 정도이고, 어디를 수정해야 하는가?”

를 학습한다.

2025의 hybrid winner 철학과 잘 맞는다.

---

# 95. Stage 6 — Single-cell distribution head

pseudo-bulk model이 안정적으로 leaderboard를 만든 후:

$$
\mu,\Sigma \;\text{or latent distribution}
$$

을 conditioning하여 individual cells 생성.

Flow matching 후보가 적합.

---

# 96. 가장 추천하는 첫 neural model

너무 크게 시작하지 말고:

```text
Context:
NT pseudo-bulk → MLP/PCA encoder

Gene:
ESM-2 projection

Reference:
TransPert Δ

Concat:
[z_context, z_gene, Δ_stat, target_baseline]

MLP:
→ Δ_residual

Final:
Δ_stat + Δ_residual
```

부터 시작.

이것이 2025 2위 + 3위 + 1위 교훈을 모두 합친 가장 현실적인 baseline이다.

---

# 97. 그다음 확장

### v2

context를 pseudo-bulk가 아니라 Set Transformer.

### v3

STRING/DepMap/PerturbAtlas 추가.

### v4

Mixture-of-context experts.

### v5

Distributional Flow Matching.

순서가 좋다.

---

# 98. Mixture-of-Experts를 context 관점에서 사용할 수 있다

각 expert가:

- immune-like
- epithelial-like
- pluripotent-like
- cancer/proliferative-like

등을 전문으로 하고,

target control이 gating을 결정.

$$
\Delta
=
\sum_k
p(k\mid z_c)
f_k(z_c,z_g)
$$

---

# 99. 하지만 class label 기반 expert는 피하라

2026 test cell line이 우리가 정한 카테고리에 안 맞을 수 있다.

따라서 hard:

```text
if cell == immune:
```

보다는 continuous latent gating이 좋다.

---

# 100. 내가 권하는 실전 실험 순서

## Experiment 0

Control mean.

## Experiment 1

Global gene delta.

## Experiment 2

Transcriptome cosine nearest context.

## Experiment 3

Similarity-weighted TransPert.

## Experiment 4

TransPert + ESM2 residual MLP.

## Experiment 5

Pretrained context encoder.

## Experiment 6

STRING/DepMap multi-prior.

## Experiment 7

Set encoder.

## Experiment 8

Distribution model.

---

# 101. 첫 일주일 실행 계획

## Day 1 — Official package audit

- validation download
- exact 3 cell lines
- exact 300 target lists
- gene set
- control cell counts
- AnnData schema
- normalization/count layer
- submission sample
- rules

---

## Day 2 — External overlap matrix

- targets vs Replogle
- targets vs X-Atlas
- targets vs GSE281048
- targets vs GSE264667
- targets vs Marson
- contexts vs external context embeddings

---

## Day 3 — Classical baseline

- control mean
- global delta
- nearest context
- weighted TransPert

---

## Day 4 — Shadow VCC CV

3 held-out public contexts 구성.

---

## Day 5 — ESM2 residual MLP

TransPert prediction을 residual correction.

---

## Day 6 — Context encoder

scBaseCount/CELLxGENE pretrained representation 비교.

---

## Day 7 — 첫 leaderboard submission

CV와 leaderboard correlation 확인.

---

# 102. Official Rules에서 당장 확인해야 할 것

현재 static browser에서 Rules의 동적 본문이 노출되지 않아,
다음 항목은 live 사이트에서 **반드시 직접 체크**해야 한다.

- external public data 허용 범위
- proprietary data 허용 여부
- data publication cutoff
- validation controls로 test-time optimization 가능 여부
- submission 횟수/rate limit
- 팀 인원 제한
- 코드/모델 공개 의무
- final submission selection 방식
- deadline exact dates
- prize split exact amounts
- prohibited datasets
- collaboration / multiple-account rules

---

# 103. 특히 “External Data” 규칙이 중요

2026 zero-shot이 external perturbation biology를 얼마나 잘 transfer하는지 보는 대회라면 external data가 자연스럽게 중요하다.

하지만:

> 공개되어 있다는 것 ≠ competition에서 자동으로 사용 가능

이다.

특히 launch 직전/직후 공개된:

- X-Atlas/Pisces
- Marson primary T
- 최신 2026 perturbation atlas

는 publication cutoff 규칙을 확인해야 한다.

---

# 104. 2026 Data Leakage audit

정식 challenge manifest를 받으면:

```python
challenge_targets
challenge_cell_lines
challenge_contexts
```

와 external corpus를 join.

### 필요한 구분

```text
Legitimate pre-existing public data
vs
Challenge hidden ground truth leakage
```

---

# 105. Current leaderboard를 보는 방법론적 태도

Leaderboard는:

> “현재 validation 3 context에서 이 모델이 어떻게 작동하는가?”

를 알려준다.

그 이상도 이하도 아니다.

특히 2026은 final contexts가 교체되므로:

> leaderboard 1위 구조를 validation context에 맞춰 복제하는 것

보다

> local LOCO CV에서 context-independent한 구조를 만드는 것

이 더 중요하다.

---

# 106. 2026에서 성공할 가능성이 높은 전략을 한 줄로

$$
\boxed{
\text{Strong statistical transfer}
+
\text{control-context representation}
+
\text{gene biological prior}
+
\text{neural residual correction}
+
\text{multi-context OOD validation}
}
$$

---

# 107. 우리가 지금까지 한 데이터셋 조사의 가치

지금까지 조사한 데이터셋은 2026 task 공개 후에도 대부분 유효하다.

오히려 우선순위가 더 분명해졌다.

| 데이터 | 2026에서의 핵심 역할 |
|---|---|
| **GSE281048** | context dependence 직접 학습 |
| **Replogle** | CRISPRi gene-effect backbone |
| **X-Atlas** | target gene coverage |
| **GSE264667** | Jurkat/HepG2 context expansion |
| **Marson T** | primary-cell OOD |
| **KOLF2.1J** | pluripotent CRISPRi |
| **scBaseCount** | context encoder |
| **CELLxGENE** | broad cell-state manifold |
| **VCC2025 H1** | challenge-style zero-shot shadow test |
| **PerturbAtlas** | historical DE prior |
| **DepMap** | gene×cell dependency prior |
| **ESM-2** | gene sequence semantics |

---

# 108. 현재 가장 중요한 데이터는 결국 GSE281048인가?

**전략적으로는 여전히 그렇다.**

왜냐하면 2026이 직접 묻는 질문이:

> “cell context가 바뀌어도 perturbation effect를 transfer할 수 있는가?”

이고 GSE281048은 이 질문을 가장 직접적으로 반복 측정한 공개 CRISPRi 데이터 중 하나이기 때문이다.

하지만 GSE281048만으로는 target coverage가 부족하다.

따라서:

```text
GSE281048
+
Replogle
+
X-Atlas
```

가 핵심 삼각형이다.

---

# 109. 핵심 삼각형

```text
                  GSE281048
         context generalization
                /       \
               /         \
              /           \
     Replogle ----------- X-Atlas
  robust gene effect     gene coverage
```

그리고 그 아래에서:

```text
scBaseCount / CELLxGENE
```

가 context representation을 받쳐준다.

---

# 110. 최종 추천 모델 구성

## Backbone 1 — Context

```text
NT control cells
→ Set Encoder
→ z_context
```

## Backbone 2 — Gene

```text
ESM2
+ STRING
+ DepMap
+ PerturbAtlas
→ z_gene
```

## Backbone 3 — Statistical memory

```text
multi-context CRISPRi pseudobulk library
→ weighted transfer Δ_stat
```

## Fusion

```text
[z_context, z_gene, Δ_stat]
→ cross-attention / FiLM / gating
→ neural correction
```

## Output

```text
Δ_pseudobulk
+
single-cell distribution
```

---

# 111. 평가 관점에서 balanced model이 중요한 이유

6 metrics가 동등하게 들어가므로:

- expression magnitude만 맞음
- DEG만 맞음
- discrimination만 맞음

중 하나에 특화된 모델보다

> 모든 축에서 평균 이상인 모델

이 유리할 가능성이 높다.

이것은 2025 Generalist Prize의 철학을 2026 main scoring으로 끌어온 변화라고 해석할 수 있다.

---

# 112. 2025 Generalist Prize의 중요성

2025에는 메인 score 이외에 7-metric Generalist Prize를 따로 만들었다.

Altos가 여기서 강했다.

2026에서는 처음부터 broad evaluation을 main scoring에 넣었다.

즉 2025 Generalist experiment가 사실상 2026 scoring redesign의 실험장이었다고 볼 수 있다.

---

# 113. PRiMeFlow/flow matching은 언제 쓸까?

초기에 바로 쓰기보다는:

```text
B0~B4 baseline 안정화
↓
pseudo-bulk Δ 잘 맞음
↓
single-cell distribution mismatch가 leaderboard/metrics에 중요함 확인
↓
flow matching
```

순서를 권한다.

---

# 114. 왜 처음부터 대형 foundation model을 추천하지 않나?

2025에서:

> pure AI model이 statistical baseline을 명확히 압도하지 못했다.

그리고 2026은:

- target-context labels 없음
- context distribution shift 큼

이므로 더 큰 network가 자동으로 해결해주지 않는다.

먼저 **inductive bias가 task와 맞아야 한다.**

---

# 115. 2026의 inductive bias

좋은 모델은 구조적으로 다음을 표현해야 한다.

### 1.

Gene에는 context-independent biology가 있다.

### 2.

그 biology는 context에 따라 modulation된다.

### 3.

새로운 context는 control population에서 읽을 수 있다.

### 4.

public reference contexts로부터 transfer할 수 있다.

### 5.

technical noise보다 robust biological effect를 우선해야 한다.

---

# 116. 그래서 우리가 가장 먼저 구현할 수 있는 모델

```text
          Validation NT cells
                  │
             pseudo-bulk
                  │
                  ▼
           context vector
                  │
                  ├─────────────┐
                  │             │
target gene → ESM2         nearest reference
                  │          contexts
                  │             │
                  ▼             ▼
                  [ z_c , z_g , Δ_stat ]
                           │
                           ▼
                         MLP
                           │
                           ▼
                       Δ_residual
                           │
                           ▼
                 Δ_stat + Δ_residual
                           │
                           ▼
                    predicted profile
```

이 정도부터 시작해도 충분히 경쟁력 있는 research baseline이 된다.

---

# 117. 이후 연구적으로 더 흥미로운 확장

- gene-specific context attention
- pathway-aware attention
- graph neural network on STRING
- DepMap-conditioned MoE
- control-cell TTA
- optimal transport between control/perturbed distributions
- conditional flow matching
- latent diffusion
- uncertainty calibration
- ensemble across statistical/neural models

---

# 118. Uncertainty도 활용할 수 있다

특정 target gene에 대해 public data가 거의 없거나
target context가 reference manifold에서 매우 멀면:

```text
confidence ↓
```

로 표시.

ensemble weight를 더 conservative하게 조절.

---

# 119. Context distance를 꼭 monitor해야 한다

Validation cell이 public corpus와 얼마나 OOD인지 알아야 한다.

예:

$$
OOD(c)
=
\min_r d(z_c,z_r)
$$

이 값이 크면:

- nearest-context transfer 신뢰 낮춤
- global gene effect로 shrink
- neural prior 비중 조절
- uncertainty 증가

가능.

---

# 120. 최종 프로젝트 구성 추천

```text
vcc2026/
├── data/
│   ├── challenge/
│   ├── replogle/
│   ├── gse281048/
│   ├── gse264667/
│   ├── xatlas/
│   ├── vcc2025/
│   └── context_atlas/
│
├── manifests/
│   ├── datasets.parquet
│   ├── targets.parquet
│   └── contexts.parquet
│
├── preprocessing/
│   ├── controls.py
│   ├── pseudobulk.py
│   ├── gene_map.py
│   └── de.py
│
├── priors/
│   ├── esm2/
│   ├── string/
│   ├── depmap/
│   └── perturbatlas/
│
├── baselines/
│   ├── control_mean.py
│   ├── global_delta.py
│   ├── nearest_context.py
│   └── transpert.py
│
├── models/
│   ├── context_encoder.py
│   ├── gene_encoder.py
│   ├── residual_mlp.py
│   ├── moe.py
│   └── flow.py
│
├── validation/
│   ├── loco.py
│   ├── gene_holdout.py
│   ├── double_ood.py
│   └── dataset_holdout.py
│
├── submission/
│   ├── build_anndata.py
│   ├── validate.py
│   └── prep.sh
│
└── experiments/
    └── leaderboard_log.csv
```

---

# 121. 공식 확인 / 추론 / 아직 확인 필요 구분

| 내용 | 상태 |
|---|---|
| 2026 task = multi-context zero-shot | ✅ 공식 확인 |
| Challenge-specific perturbation train label 없음 | ✅ 공식 확인 |
| Validation = 3 cell lines | ✅ 공식 확인 |
| 각 cell = NT control + 300 CRISPRi targets | ✅ 공식 확인 |
| Final test = 다른 3 cell lines | ✅ 공식 확인 |
| Final test = October phase | ✅ 공식 확인 |
| 6 complementary metrics | ✅ 공식 확인 |
| mean ↔ replicate normalization | ✅ 공식 확인 |
| equal-weight overall score | ✅ 공식 확인 |
| leaderboard/submission open | ✅ 공식 확인 |
| USD 175k total prizes | ✅ 공식 확인 |
| 현재 validation cell line 정확한 이름 | ⚠️ live Data SPA/package에서 확인 필요 |
| 세 context의 300 target이 동일한지 | ⚠️ package에서 확인 필요 |
| exact 6 metric names | ⚠️ live Evaluation SPA/evaluator에서 확인 필요 |
| leaderboard UI의 exact columns/current ranking | ⚠️ live JS UI에서 확인 필요 |
| external-data cutoff | ⚠️ Rules 확인 필요 |
| submission rate limit | ⚠️ Rules 확인 필요 |
| TTA 허용 여부 | ⚠️ Rules 확인 필요 |
| exact final deadline | ⚠️ Rules 확인 필요 |

---

# 122. 결론

2026 Virtual Cell Challenge는 더 이상

> “한 새로운 cell line에서 일부 perturbation을 보고 나머지를 맞히는 문제”

가 아니다.

이제는:

$$
\boxed{
\text{Unseen Cell Context}
+
\text{Control Cells Only}
+
\text{Target Gene}
\rightarrow
\text{Perturbation Response}
}
$$

이다.

따라서 핵심 연구 질문은:

> **세포의 baseline state만으로 그 세포가 gene perturbation에 어떻게 반응할지를 다른 contexts의 경험으로부터 transfer할 수 있는가?**

가 된다.

---

# 123. 우리가 가져가야 할 최종 전략

## 데이터

```text
GSE281048
+
Replogle
+
X-Atlas
+
GSE264667
+
Marson/KOLF2.1J
+
scBaseCount/CELLxGENE
```

## 표현

```text
Context:
NT control set embedding

Gene:
ESM2 + STRING + DepMap + PerturbAtlas
```

## Prediction

```text
public multi-context statistical transfer
+
neural residual correction
```

## Validation

```text
leave-one-cell-line-out
+
gene holdout
+
double OOD
+
dataset/lab holdout
```

## Output

```text
strong pseudo-bulk Δ
→ 이후 single-cell distribution modeling
```

## Model philosophy

$$
\boxed{
\text{Hybrid statistical + deep model}
>
\text{blindly scaling a single foundation model}
}
$$

라는 2025 교훈을 2026 zero-shot 구조에 맞게 확장하는 것이 가장 합리적이다.

---

# 124. 가장 먼저 해야 할 실무 5개

1. **2026 validation package 다운로드 후 정확한 3 cell line / 300 target / file schema 확인**
2. **external corpus와 target overlap + context similarity matrix 생성**
3. **GSE281048 기반 leave-one-context-out shadow challenge 구축**
4. **TransPert-like weighted statistical baseline 완성**
5. **ESM2 + context embedding을 이용한 residual MLP를 baseline 위에 추가**

이 다섯 단계가 끝나기 전에는 대형 flow/diffusion 모델부터 시작하지 않는 것을 추천한다.

---

# Sources

## 2026 current challenge

1. Virtual Cell Challenge  
   https://virtualcellchallenge.org/

2. Data  
   https://virtualcellchallenge.org/data

3. Evaluation  
   https://virtualcellchallenge.org/evaluation

4. Leaderboard  
   https://virtualcellchallenge.org/leaderboard

5. Rules  
   https://virtualcellchallenge.org/rules

6. Arc 2026 launch announcement — current launch text syndicated/reposted by official sponsor 10x Genomics / Arc Institute social feed  
   Current launch confirms:
   - multi-context generalization
   - zero-shot
   - 3 validation cell lines
   - NT control + 300 CRISPRi IDs/context
   - 3 final test lines in October
   - six metrics
   - context-mean ↔ replicate normalization
   - equal-weight overall
   - USD 175k prizes

## Arc evaluation tooling

7. ArcInstitute/cell-eval  
   https://github.com/ArcInstitute/cell-eval

8. ArcInstitute/STATE  
   https://github.com/ArcInstitute/state

9. Arc Virtual Cell Atlas  
   https://github.com/ArcInstitute/arc-virtual-cell-atlas

## 2025 background

10. 2025 Wrap-up  
    https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up

11. 2025 benchmark paper  
    “Virtual Cell Challenge: Toward a Turing test for the virtual cell”  
    Cell, 2025.

## External methods/data referred to in strategy

12. GSE281048 — Jiang/Mixscale  
13. Replogle et al. 2022 genome-scale Perturb-seq  
14. X-Atlas/Orion  
15. GSE264667 / TRADE  
16. scBaseCount / CELLxGENE Census  
17. DepMap  
18. STRING  
19. ESM-2  
20. PerturbAtlas  
21. PRiMeFlow / PerturBench / X-Cell as methodological references

---

# Verification note

`virtualcellchallenge.org`의 현재 Data/Evaluation/Leaderboard/Rules 페이지는 JavaScript SPA로 동적으로 렌더링된다.  
따라서 본 문서는 **2026-08-20 launch 시점에 Arc가 공식적으로 공개한 최신 task/data/scoring 설명과 Arc의 공개 evaluation/model repositories를 교차 확인**하여 작성했다.

SPA의 로그인 후 실시간 UI에서만 보이는 다음 정보는 추측하지 않았다.

- 현재 leaderboard team rows
- exact metric six labels
- exact validation cell-line names
- exact submission limits/deadlines
- live UI column names

이 항목들은 실제 참가 계정에서 보이는 페이지/다운로드 package를 기준으로 최종 고정해야 한다.

---

**End — 2026-08-22**

<!-- Original Korean write-up kept verbatim (file: VCC2026_시작_계획서_메트릭_베이스라인_통계_vs_학습.md); only the file name was changed, LaTeX escapes that had been corrupted into control characters were repaired, and \[ \] / \( \) math delimiters were converted to $$ / $ for GitHub rendering. -->

# Virtual Cell Challenge 2026 — 시작 계획서
## Metric-first Baseline → Statistical Transfer → Learning Model 순서로 가는 실전 로드맵

> **기준일:** 2026-08-22  
> **목적:** 2026 VCC를 “일단 큰 모델부터 돌리는 방식”이 아니라, **평가 지표를 정확히 이해하고 → 완전한 baseline을 만들고 → 통계적 접근과 학습형 접근을 공정하게 비교한 뒤 → 필요한 부분만 복잡하게 확장**하는 방식으로 시작하기 위한 계획서  
> **근거:** 2026 공식 `cell-eval2` metric specification / `configs/vcc2026.yaml`, 2025 VCC 공식 평가·wrap-up, 지금까지 조사한 multi-context CRISPRi 데이터셋과 2025 상위권 전략

---

# 0. 결론부터 — 지금 생각한 방향은 맞는가?

## 결론: **맞다. 다만 한 가지 수정이 필요하다.**

현재 생각한 시작점:

1. **Metric 분석**
2. **완전한 baseline 구축**
3. 어떤 metric이 중요한지 확인
4. **통계 기반 방법 vs 학습 기반 방법 성능 비교**

이 순서는 매우 좋다.

하지만 3번은 이렇게 바꾸는 것이 더 정확하다.

> ❌ “6개 metric 중 하나가 제일 중요하니 그것부터 최적화한다.”
>
> ✅ **“6개 metric은 최종 점수에서 동일 가중이므로, 현재 baseline에서 어떤 metric이 병목인지, 어떤 metric은 이미 천장에 가까운지, 그리고 어떤 종류의 모델 변경이 어느 metric을 움직이는지 찾는다.”**

2026에서는 6개의 **scaled metric이 동일한 가중치**로 overall score에 들어간다.

따라서 명목상:

```text
PDS            1/6
Expression MSE 1/6
Direction      1/6
Reach          1/6
Jaccard        1/6
LFC NMAE       1/6
```

이다.

Cell context도 동일하게 평균되므로 특정 cell 하나만 잘 맞혀서도 안 된다.

즉 올해의 핵심은:

> **“어떤 metric을 게임할까?”가 아니라  
> “우리 방법의 가장 큰 biological failure mode가 무엇인가?”를 metric으로 진단하는 것**

이다.

---

# 1. 2025와 2026 — Metric이 어떻게 달라졌는가?

## 아주 쉽게 설명하면

### 2025

평가 질문이 크게 세 개였다.

```text
1. PDS
   "A perturbation을 A답게 구별했어?"

2. DES
   "어떤 유전자들이 변하는지는 맞혔어?"

3. MAE
   "전체 발현 숫자는 실제와 가까워?"
```

문제는 실제 대회에서:

- `MAE`는 단순 평균 baseline이 너무 강했고
- 대부분 모델이 baseline보다 MAE가 나빴으며
- 경쟁적으로는 `PDS`와 `DES`가 훨씬 중요해졌다.
- 특히 PDS는 scale에 민감해서 prediction magnitude 조정으로 점수를 움직일 수 있었다.

즉 2025는 의도는 3축 평가였지만 실제 경쟁에서는:

```text
PDS + DES 중심
```

으로 많이 수렴했다.

---

# 2. 2026은 무엇이 달라졌는가?

2026은 이 문제를 훨씬 세분화했다.

## 2025 → 2026 개념 변화

| 2025 | 2026 | 쉽게 말하면 |
|---|---|---|
| **PDS (L1)** | **`pds_cosine`** | perturbation을 서로 구별하되, 단순 크기 scaling 꼼수를 줄임 |
| **MAE** | **`expr_mse_unbiased_capped_norm`** | 전체 expression error를 보되 sampling noise를 보정하고 control 변화량으로 정규화 |
| **DES 하나** | **DE 관련 4개 metric** | “DE를 맞혔다”를 집합·방향·순위·크기로 쪼개서 평가 |
| 3개 metric | **6개 metric** | 하나를 최적화해서 버티기 어려움 |
| mean baseline 중심 정규화 | **baseline=0, replicate=1** | 무정보 예측부터 실험 재현 수준까지의 거리로 해석 |
| metric별 trade-off 존재 | **6 metric × context 동일 가중** | generalist가 유리 |

---

# 3. 2026의 6개 Metric — 연구자 관점의 역할

## Metric 1 — `pds_cosine`

### 질문

> **“이 perturbation prediction이 다른 perturbation과 구별되는가?”**

예:

```text
TP53 KD prediction
```

이 300개의 실제 perturbation 중에서 진짜 TP53 response와 가장 가까워야 한다.

### 2025와 차이

2025:

```text
L1 distance
```

2026:

```text
cosine distance
```

즉 prediction vector의 **방향(pattern)** 에 훨씬 집중한다.

단순히 모든 delta를 크게 키운다고 PDS가 계속 좋아지는 2025식 scaling 전략은 훨씬 덜 먹힌다.

### 우리가 확인할 것

- Global statistical effect만으로 얼마나 PDS가 나오는가?
- Context conditioning을 넣으면 PDS가 얼마나 개선되는가?
- Gene embedding이 PDS에 얼마나 도움되는가?

---

# 4. Metric 2 — `expr_mse_unbiased_capped_norm`

### 질문

> **“예측한 전체 transcriptome이 실제 값과 얼마나 가까운가?”**

2025 MAE와 비슷한 역할이지만 훨씬 정교하다.

### 2025 MAE의 문제

single-cell에는:

- dropout
- sampling noise
- biological heterogeneity

가 있어서 단순 expression error는 noisy하다.

그래서 mean-like prediction이 강했다.

### 2026

sampling noise를 jackknife로 추정해서 빼고,
실제 perturbation이 control에서 얼마나 멀어졌는지를 기준으로 오차를 정규화한다.

즉:

> **“실제로 움직인 만큼을 고려했을 때 네 prediction이 얼마나 틀렸나?”**

에 가까워졌다.

### 중요한 특징

이 metric은 scaled score가 **[0,1]로 clip**된다.

따라서 여기서 replicate를 넘어 엄청난 bonus를 얻는 전략은 없다.

### 연구 관점

이 metric은:

> **모델이 말도 안 되는 expression magnitude를 만들고 있는지 감시하는 안전장치**

로 생각하면 좋다.

---

# 5. Metric 3 — Direction Fidelity + Yield

`de_wilcoxon_direction_fidelity_yield_raw`

### 질문

> **“반응한다고 예측한 유전자들이 실제와 같은 방향으로 움직이며, 충분히 많은 유전자를 잡았는가?”**

예:

```text
Actual:
Gene A ↑
Gene B ↓
Gene C ↑

Prediction:
Gene A ↑   ✅
Gene B ↑   ❌
Gene C ↑   ✅
```

를 본다.

그런데 confident gene 3개만 예측해서 정확도를 높이는 꼼수를 막기 위해 **yield/coverage도 함께** 본다.

### 왜 중요?

6개 중 baseline ↔ replicate raw span이 비교적 좁다.

따라서 **raw metric의 작은 개선도 scaled score를 꽤 움직일 수 있다.**

하지만 이것을 “가중치가 더 높다”고 해석하면 안 된다.

> **scaled 이후 가중치는 동일하지만, raw sensitivity가 높은 metric**

이라고 이해해야 한다.

---

# 6. Metric 4 — Direction Reach

`de_wilcoxon_direction_reach_raw`

### 질문

> **“가장 자신 있는 DE gene부터 내려갔을 때, 얼마나 깊은 곳까지 방향이 믿을 만한가?”**

Fidelity와 다른 점:

```text
Fidelity:
선택한 DE들의 전체 정확도

Reach:
네 confidence ranking의 앞부분부터 얼마나 오래 90% 방향 정확도를 유지하나
```

### 의미

모델이:

> “어떤 gene을 가장 확실하게 믿어야 하는가?”

까지 제대로 ranking해야 한다.

### 특징

replicate가 이미 0.96~0.98 수준이라 천장이 매우 가깝다.

즉 어느 정도 잘 맞으면 여기서 추가로 벌 수 있는 headroom은 작다.

---

# 7. Metric 5 — Significant Gene Jaccard

`de_wilcoxon_sig_jaccard`

### 질문

> **“실제로 반응한 gene 집합과 내가 반응한다고 한 gene 집합이 얼마나 겹치는가?”**

$$
J=
\frac{|Real\cap Pred|}
{|Real\cup Pred|}
$$

이다.

### 매우 중요한 특징

실험 replicate끼리도 raw Jaccard가 약 0.38~0.42 정도밖에 안 된다.

즉 single-cell DE set 자체가 꽤 불안정하다.

그래서 scaled score에서 완벽한 reproduction은 **2.5 이상**까지 갈 수 있다.

### 전략적으로 왜 흥미로운가?

6개 중 **위쪽 headroom이 가장 크다.**

즉 좋은 DEG selection 모델이 replicate 수준을 넘어가면 overall score를 꽤 끌어올릴 수 있다.

### 하지만 위험

Jaccard만 직접 최적화하려고 DE gene 수를 인위적으로 맞추면:

- direction
- LFC magnitude
- expression MSE

가 무너질 수 있다.

따라서 독립적인 목표라기보다 **모델의 DE calibration quality를 확인하는 지표**로 써야 한다.

---

# 8. Metric 6 — LFC NMAE

`de_wilcoxon_lfc_nmae`

### 질문

> **“반응하는 gene을 맞힌 것을 넘어서, 그 gene이 얼마나 많이 올라가고 내려가는지도 맞혔는가?”**

예:

```text
Actual:
Gene A = +3.0 log2FC

Prediction 1:
+2.8      → 좋음

Prediction 2:
+0.2      → 방향은 맞지만 magnitude는 나쁨
```

Direction metric에서는 둘 다 맞을 수 있지만,
LFC metric은 차이를 잡아낸다.

### 즉

> **2026에서는 sign-only prediction으로는 부족하다.**

---

# 9. 2025 DES가 2026에서 사실상 네 조각으로 분해되었다고 생각하면 쉽다

2025 DES:

```text
"DE gene을 잘 찾았나?"
```

2026:

```text
1. 어떤 gene인가?        → Jaccard
2. 방향이 맞나?          → Fidelity
3. confidence ranking?   → Reach
4. magnitude까지 맞나?   → LFC NMAE
```

즉 훨씬 진단적이다.

---

# 10. Metric 계획에서 가장 중요한 수정

## “어떤 metric이 제일 중요하지?”를 이렇게 바꾸자

### 질문 1

**현재 baseline에서 scaled score가 가장 낮은 metric은 무엇인가?**

### 질문 2

그 metric은:

- 통계적 transfer 부족?
- context 인식 부족?
- target gene representation 부족?
- single-cell distribution 생성 문제?

중 무엇 때문에 낮은가?

### 질문 3

모델 변경 하나가 6 metric을 어떻게 움직이는가?

예:

```text
ESM-2 추가

PDS       +0.07
Expr      +0.00
Fidelity  +0.05
Reach     +0.02
Jaccard   +0.04
LFC       +0.01
```

이런 표를 만들어야 한다.

---

# 11. 우리가 만들 핵심 분석값 — `Metric Gap`

각 scaled metric을 $s_m$이라고 하자.

replicate level = 1.

그러면:

$$
Gap_m=\max(0,1-s_m)
$$

를 정의한다.

예:

| Metric | Score | Gap to replicate |
|---|---:|---:|
| PDS | 0.82 | 0.18 |
| Expr | 0.74 | 0.26 |
| Fidelity | 0.31 | **0.69** |
| Reach | 0.67 | 0.33 |
| Jaccard | 0.22 | **0.78** |
| LFC | 0.40 | 0.60 |

그러면 이 모델의 핵심 문제는:

```text
Jaccard + Fidelity + LFC
```

쪽이라는 것을 바로 알 수 있다.

---

# 12. 하지만 Gap만 보면 안 된다 — Headroom과 noise도 봐야 한다

Metric별로:

1. 현재 score
2. replicate까지 gap
3. theoretical headroom
4. split/repeat noise
5. engineering cost

를 함께 본다.

예:

```text
Metric Priority
=
Gap
× attainable gain
÷ engineering cost
```

식으로 생각할 수 있다.

정확한 수식이 필요한 것은 아니고 **의사결정 프레임**이다.

---

# 13. 전체 프로젝트를 어떤 순서로 시작해야 하나?

# Phase 0 — Evaluation을 먼저 완전히 재현한다

## 목표

> 모델을 만들기 전에 **공식 evaluator가 우리 로컬에서 동일하게 돌아가게 만든다.**

### 해야 할 일

- `cell-eval2` 공식 버전 pin
- `configs/vcc2026.yaml` pin
- `rule_version` 기록
- 6 metric 모두 local 실행
- raw score 저장
- baseline / replicate anchor 저장
- scaled score 저장
- overall score 계산
- context별 score 저장

### 산출물

```text
evaluation/
├── run_eval.py
├── metric_config.yaml
├── reference_bundles/
└── metric_report.parquet
```

---

# 14. Metric 결과 저장 schema

실험마다 최소한 아래를 저장한다.

```text
experiment_id
model_name
context
metric
raw_score
baseline_b
replicate_r
scaled_score
overall_score
seed
git_commit
```

---

# 15. Phase 1 — Metric Sanity Experiment

이 단계가 매우 중요하다.

모델 학습 전에 **일부러 이상한 prediction**을 만들어 metric의 행동을 직접 확인한다.

## 실험 A — No-effect baseline

```text
prediction = control
```

확인:

- PDS ≈ baseline
- expression error는 no-skill
- DE metrics 낮음

---

## 실험 B — All-perturbation mean

모든 perturbation에 같은 평균 effect.

확인:

- PDS ≈ 0.5 raw
- Jaccard/Fidelity에서 common response가 얼마나 통하는지

---

## 실험 C — Global gene effect

다른 contexts에서 gene G의 평균 delta:

$$
\bar\Delta_g
$$

를 target cell control에 더한다.

이것이 **cell-agnostic statistical baseline의 최소 형태**다.

---

## 실험 D — Magnitude scaling sweep

$$
\alpha\Delta,\qquad
\alpha\in
\{0.25,0.5,1,1.5,2,4\}
$$

### 왜?

2025에서는 PDS가 scaling에 민감했다.

2026 cosine PDS가 실제로 magnitude scaling에 훨씬 둔감한지 직접 확인.

동시에:

- Expression MSE
- LFC NMAE

가 어떻게 망가지는지도 본다.

---

# 16. Metric sanity — DE 특성 실험

## 실험 E — Correct sign / wrong magnitude

방향은 맞추고 magnitude를 일부러 줄인다.

예:

```text
true Δ × 0.1
```

예상:

- Fidelity/Reach 상대적으로 유지
- LFC NMAE 악화
- Expression metric 악화

---

## 실험 F — Correct DEG set / wrong sign

DE gene 선택은 맞지만 sign을 뒤집는다.

예상:

- Jaccard 유지
- Fidelity 붕괴
- Reach 붕괴
- LFC 붕괴

---

## 실험 G — Correct top genes only

정말 강한 DEG 일부만 예측.

예상:

- 방향 precision은 높을 수 있음
- Fidelity의 yield penalty
- Jaccard recall 부족

---

## 실험 H — Too many DEG

많은 genes를 변화시킨다.

예상:

- recall 증가 가능
- Jaccard false positive 증가
- Fidelity/Expression 악화

---

# 17. Phase 1의 최종 산출물

## `metric_behavior_matrix.csv`

| Prediction Variant | PDS | Expr | Fidelity | Reach | Jaccard | LFC | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | | | | | | | |
| Mean effect | | | | | | | |
| Global gene effect | | | | | | | |
| ×0.5 scale | | | | | | | |
| ×2 scale | | | | | | | |
| Sign only | | | | | | | |
| Top DEG only | | | | | | | |

이 표 하나만 만들어도 metric을 거의 몸으로 이해할 수 있다.

---

# 18. Phase 2 — Shadow VCC 2026 Benchmark를 만든다

공식 validation은 정답이 없다.

따라서 leaderboard만 보면서 개발하면 안 된다.

**정답이 있는 public perturbation data로 2026 구조를 복제**해야 한다.

---

# 19. 가장 중요한 Shadow Dataset

## GSE281048

왜?

- CRISPRi
- 여러 cell line
- 같은 perturbation biology가 반복됨
- leave-one-cell-line-out 가능

예:

```text
Train:
A549
MCF7
HT29
HAP1
K562

Test:
BxPC3
```

BxPC3에서:

```text
training에 사용:
NT control only

hidden:
BxPC3 perturbation responses
```

로 만든다.

---

# 20. Shadow benchmark를 하나가 아니라 여러 개 만든다

## Fold 1

```text
BxPC3 holdout
```

## Fold 2

```text
A549 holdout
```

## Fold 3

```text
MCF7 holdout
```

그리고 별도 strong OOD:

## Fold 4

```text
H1 VCC2025 holdout
```

## Fold 5

```text
HepG2 holdout
```

## Fold 6

```text
Jurkat holdout
```

---

# 21. Local model selection은 평균만 보지 않는다

각 모델마다:

```text
Mean Context Score
Worst Context Score
Std Across Contexts
```

를 모두 저장한다.

추천 model-selection score:

$$
S_{\mathrm{select}}
=
\operatorname{Mean}(S_c)
-
\lambda\operatorname{Std}(S_c)
$$

이유:

> final test는 새로운 cell context이므로 한 context에서 대박 나고 다른 context에서 망하는 모델은 위험하다.

---

# 22. Phase 3 — “완전한 Baseline Ladder” 구축

가장 중요한 단계.

모델 하나를 baseline이라고 하지 말고 **난이도가 한 단계씩 올라가는 baseline 계단**을 만든다.

---

# 23. Baseline B0 — Control / No-effect

$$
\widehat X_{pert}=X_{control}
$$

질문:

> “아무 effect도 예측하지 않는 것보다 나은가?”

모든 모델이 반드시 이겨야 한다.

---

# 24. Baseline B1 — Global Perturbation Mean

모든 perturbation의 평균적인 변화:

$$
\Delta_{\mathrm{global}}
$$

을 사용.

질문:

> “일반적인 CRISPRi stress response만 넣어도 얼마나 되나?”

---

# 25. Baseline B2 — Gene-specific Global Effect

gene $g$에 대해 여러 reference contexts의 perturbation effect를 평균:

$$
\bar\Delta_g
=
\frac1R
\sum_r
\Delta_{r,g}
$$

prediction:

$$
\hat X
=
X_{ctrl}^{target}
+
\bar\Delta_g
$$

### 매우 중요

이 baseline이 강하면:

> **gene identity 자체가 response 대부분을 결정한다.**

는 뜻.

---

# 26. Baseline B3 — Nearest Context Transfer

target NT control과 가장 비슷한 reference context:

$$
r^*
=
\arg\max_r
sim(z_{target},z_r)
$$

그 context의 gene effect를 그대로 transfer.

질문:

> **“cell similarity만 알아도 얼마나 좋아지나?”**

---

# 27. Baseline B4 — Similarity-weighted TransPert

$$
\widehat{\Delta}_{c,g}
=
\sum_r
w(c,r)
\Delta_{r,g}
$$

### 이것이 통계 baseline의 핵심

2025 3위 TransPert 철학과 유사하고,
2026 task 구조에 매우 잘 맞는다.

---

# 28. Baseline B5 — Gene-specific Similarity-weighted Transfer

전체 cell similarity가 아니라:

```text
global transcriptome similarity
+
target pathway similarity
+
target baseline expression
+
DepMap similarity
```

를 이용.

$$
w(c,r,g)
$$

로 확장.

이 단계까지는 **neural network 없이도 가능**하다.

---

# 29. 여기까지가 “통계 기반” 진영

통계/statistical baseline:

```text
B0 Control
B1 Global mean
B2 Gene mean
B3 Nearest context
B4 Weighted TransPert
B5 Gene-specific weighted transfer
```

를 먼저 완성한다.

그리고 그 이후부터 learning을 추가한다.

---

# 30. Phase 4 — Statistical vs Learning 비교

이 비교는 반드시 **같은 데이터, 같은 split, 같은 evaluator**에서 한다.

## Statistical

```text
B4/B5
```

## Simple Learning

```text
Linear regression / Ridge
MLP
```

## Representation Learning

```text
Context encoder
+ ESM-2
```

## Advanced

```text
Set Transformer
Flow
Foundation model
```

---

# 31. 첫 Learning Baseline L1 — Ridge / Linear

Input:

```text
target control pseudo-bulk
+
gene embedding
```

Output:

```text
Δ expression
```

### 왜 Linear를 넣나?

Deep model이 정말 nonlinear biology를 배우는지,
아니면 단순 linear transfer면 충분한지 확인할 수 있다.

---

# 32. Learning Baseline L2 — MLP

Input:

```text
context PCA
+
ESM2 gene embedding
+
target baseline expression
```

Output:

```text
Δ pseudo-bulk
```

2025 2위 접근과 가장 비슷한 baseline.

---

# 33. Learning Baseline L3 — Statistical + Neural Residual

가장 추천.

먼저 B5가:

$$
\Delta_{stat}
$$

를 예측.

MLP는 처음부터 response를 만들지 않고:

$$
\Delta_{residual}
=
f_\theta(
z_c,
z_g,
\Delta_{stat}
)
$$

만 학습.

최종:

$$
\boxed{
\Delta_{pred}
=
\Delta_{stat}
+
\Delta_{residual}
}
$$

---

# 34. 왜 이 비교가 가장 중요하나?

이 실험으로 바로 답할 수 있다.

### Case A

```text
B5 = 0.55
L2 = 0.48
L3 = 0.58
```

→ 통계적 transfer가 핵심이고 neural은 correction 역할.

### Case B

```text
B5 = 0.40
L2 = 0.58
L3 = 0.61
```

→ nonlinear context×gene interaction 학습이 중요.

### Case C

```text
B5 = 0.55
L2 = 0.55
L3 = 0.55
```

→ 데이터/representation이 병목이지 model complexity가 문제가 아님.

---

# 35. Statistical vs Learning 비교에서 꼭 해야 하는 Ablation

| Experiment | Statistical prior | Context encoder | ESM2 | 목적 |
|---|:---:|:---:|:---:|---|
| A | ✕ | ✕ | ✕ | naïve |
| B | ✅ | ✕ | ✕ | 통계 효과 |
| C | ✕ | ✅ | ✕ | context representation |
| D | ✕ | ✕ | ✅ | gene semantics |
| E | ✅ | ✅ | ✕ | stats + context |
| F | ✅ | ✕ | ✅ | stats + gene |
| G | ✅ | ✅ | ✅ | full hybrid |

---

# 36. Metric별로 Ablation 결과를 봐야 한다

Overall만 보면 안 된다.

예:

| Model | PDS | Expr | Fidelity | Reach | Jaccard | LFC | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| TransPert | .70 | .61 | .52 | .60 | .31 | .44 | .53 |
| + ESM2 | .76 | .62 | .58 | .65 | .38 | .48 | .58 |
| + Context Encoder | .78 | .66 | .65 | .68 | .46 | .55 | .63 |

그러면:

> context encoder가 특히 DEG selection/Jaccard와 magnitude에 도움

같은 해석이 가능하다.

---

# 37. Phase 5 — Pseudobulk와 Single-cell을 분리해서 분석

2025 상위권에서 pseudo-bulk가 강했다.

그러므로 처음부터 flow model로 가지 않는다.

## 먼저

```text
control pseudo-bulk
→ Δ pseudo-bulk prediction
```

을 완성.

---

# 38. 하지만 2026 제출은 single-cell raw counts다

최종 제출은 각 perturbation마다 정확히 **400 predicted cells**가 필요하다.

따라서 pseudo-bulk prediction을 single-cell population으로 변환하는 단계가 필요하다.

---

# 39. “같은 pseudo-bulk 모델 + 다른 cell generator” 비교도 해야 한다

이 실험이 중요하다.

## Generator G0 — Control-cell bootstrap

control cell 400개를 resample하고,
predicted gene-wise effect를 multiplicatively 적용.

---

## Generator G1 — Library-size-preserving multinomial

각 control cell의 library size $L_i$를 유지.

예측된 expression probability $p_g$에서:

$$
x_i
\sim
Multinomial(L_i,p)
$$

---

## Generator G2 — Negative Binomial / dispersion model

reference perturbation datasets에서 gene/context별 dispersion을 학습.

---

## Generator G3 — Conditional flow / generative model

마지막 단계.

---

# 40. 왜 generator가 metric에 영향을 주는가?

PDS/Expression은 group pseudobulk에 많이 의존하지만,
DE metrics는 **single-cell Wilcoxon test**를 사용한다.

즉 같은 평균을 가진 두 prediction도:

```text
분산
zero fraction
heterogeneity
```

가 다르면 DE significance가 달라진다.

따라서:

> **Pseudo-bulk 모델의 성능과 single-cell generator의 성능을 분리해 측정해야 한다.**

---

# 41. 매우 중요한 실험

## Fixed mean experiment

동일한 pseudo-bulk $\Delta$를 고정하고:

```text
Generator G0
Generator G1
Generator G2
```

만 바꾼다.

6 metric을 비교.

### 결과

어떤 metric이:

- mean prediction 문제인지
- distribution 문제인지

구분할 수 있다.

---

# 42. Metric별 “무엇을 먼저 개선해야 하는가” 진단표

| 낮은 Metric | 의심할 부분 | 먼저 해볼 것 |
|---|---|---|
| PDS | gene-specific effect 구별 실패 | ESM2 / gene prior / target-specific Δ |
| Expression MSE | magnitude calibration | shrinkage / calibration / residual |
| Fidelity | sign + DEG recall | pathway/context conditioning |
| Reach | confidence ranking | DE probability/ranking head |
| Jaccard | DEG set calibration | threshold/count calibration + distribution |
| LFC NMAE | effect magnitude | gene-wise Δ regression / context residual |

---

# 43. “어떤 metric이 중요한가?”에 대한 실제 답

## 명목상

**6개 모두 동일하게 중요하다.**

## Raw sensitivity

`Fidelity`처럼 baseline–replicate span이 좁은 metric은 작은 raw 개선이 scaled score를 크게 움직일 수 있다.

## Headroom

`Jaccard`는 replicate 이후의 위쪽 공간이 매우 크다.

## 안정성

`Reach`와 `PDS`는 replicate가 천장에 가까워 어느 정도 잘하면 추가 이득이 작다.

## 따라서

> **Metric 중요도 = 고정된 순위가 아니라 현재 모델 상태에 따라 바뀐다.**

---

# 44. 추천 Metric Optimization Rule

각 실험 후:

```text
1. scaled metric이 가장 낮은 2개 확인
2. 그 metric의 gap-to-replicate 계산
3. 그 metric에 영향을 줄 수 있는 가장 단순한 변경 수행
4. 다른 4 metric이 무너지는지 확인
5. overall + worst-context 둘 다 좋아질 때만 채택
```

---

# 45. 2026에서 특히 조심해야 할 Metric Gaming

## 2025식 PDS scaling

2026 cosine PDS에서는 훨씬 덜 유효.

## 모든 gene을 약간씩 변화

Expression은 버틸 수 있지만:
Jaccard/Fidelity가 망가질 수 있다.

## 매우 적은 DE만 예측

방향 정확도는 높아 보이지만:
Fidelity yield가 벌점.

## 많은 DE 예측

Recall은 늘지만:
Jaccard false positive와 expression error 증가.

## sign만 맞춤

Fidelity는 좋아질 수 있지만:
LFC NMAE가 잡는다.

---

# 46. 이게 바로 6 metric을 만든 이유

2025:

```text
한 metric의 취약점을 활용할 여지가 상대적으로 큼
```

2026:

```text
PDS
+ global expression
+ sign
+ confidence ranking
+ DEG set
+ magnitude
```

로 서로를 견제한다.

따라서 진짜 biological fidelity를 올리는 것이 결국 가장 안전하다.

---

# 47. 첫 번째 큰 연구 질문

> **Q1. Cell-agnostic prediction에서 통계적 perturbation transfer만으로 어디까지 갈 수 있는가?**

실험:

```text
B2
B3
B4
B5
```

비교.

이것만으로도 매우 중요한 insight가 나올 수 있다.

---

# 48. 두 번째 연구 질문

> **Q2. Neural model은 statistical baseline의 무엇을 개선하는가?**

Metric별 비교:

```text
PDS?
DE set?
LFC?
Context-specific effect?
```

---

# 49. 세 번째 연구 질문

> **Q3. Context encoder가 실제 zero-shot에서 필요한가?**

비교:

```text
raw pseudo-bulk cosine
vs
PCA
vs
scBaseCount/CELLxGENE pretrained encoder
```

---

# 50. 네 번째 연구 질문

> **Q4. Gene embedding은 unseen gene 또는 sparse gene에서 얼마나 도움이 되는가?**

split:

```text
seen genes
vs
completely held-out genes
```

비교:

```text
one-hot
ESM2
ESM2 + STRING
ESM2 + STRING + DepMap
```

---

# 51. 다섯 번째 연구 질문

> **Q5. Single-cell distribution modeling이 실제 leaderboard metric에 필요한가?**

동일 pseudo-bulk prediction을:

```text
simple bootstrap
NB
flow
```

로 생성해서 비교.

이 실험을 하기 전에는 flow model을 크게 개발하지 않는다.

---

# 52. 추천 프로젝트 단계

## Milestone 1 — Evaluator trustworthy

완료 조건:

- 공식 6 metric local reproduction
- baseline/replicate normalization 확인
- submission raw count validator 구현

## Milestone 2 — Shadow challenge trustworthy

완료 조건:

최소 3개 full-context holdout에서:

```text
train contexts
→ unseen context NT only
→ prediction
→ hidden GT evaluation
```

가능.

## Milestone 3 — Statistical baseline trustworthy

완료 조건:

B0~B5 모두 구현.

모든 baseline이:

- 동일 preprocessing
- 동일 generator
- 동일 evaluator

사용.

## Milestone 4 — Statistical vs Learning answered

완료 조건:

최소:

```text
B5 statistical
L1 Ridge
L2 MLP
L3 Statistical + residual MLP
```

을 비교.

그리고 metric별 개선 원인을 설명할 수 있음.

## Milestone 5 — Representation ablation

완료 조건:

- context encoder
- ESM2
- STRING
- DepMap

각각의 효과를 OOD에서 정량화.

## Milestone 6 — Distribution modeling decision

질문:

> simple pseudo-cell generator로도 충분한가?

YES:

```text
flow 개발 후순위
```

NO:

```text
flow / STATE / distribution head 개발
```

---

# 53. 첫 2주 추천 일정

## Day 1–2 — Metric / evaluator

- `cell-eval2` 실행
- 6 metric source/config 확인
- sanity predictions 제작
- raw ↔ scaled 관계 확인

## Day 3 — Metric behavior report

다음 그림/표 생성:

```text
Scaling sweep
DE count sweep
Sign flip
Magnitude shrink
Noise injection
```

→ 각 6 metric 변화.

## Day 4–5 — Shadow VCC dataset

- GSE281048
- GSE264667
- VCC2025

중 사용 가능한 contexts로 LOCO split 제작.

## Day 6 — B0/B1/B2

- control
- global effect
- gene average effect

## Day 7 — B3/B4

- nearest context
- weighted TransPert

## Day 8 — B5

gene-specific context weighting.

## Day 9 — L1 Ridge

## Day 10 — L2 MLP + ESM2

## Day 11 — L3 Statistical Residual MLP

## Day 12 — Ablation

```text
- stats
- context
- ESM2
```

## Day 13 — Single-cell generator comparison

## Day 14 — 첫 method decision

결정:

```text
A. Statistical 중심으로 갈지
B. Hybrid로 갈지
C. Full neural이 필요한지
```

---

# 54. 모델을 선택할 때 Overall만 보지 않는다

최종 experiment table:

| Model | Overall | Worst Context | PDS | Expr | Fidelity | Reach | Jaccard | LFC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | | | | | | | | |
| B2 | | | | | | | | |
| B4 | | | | | | | | |
| B5 | | | | | | | | |
| Ridge | | | | | | | | |
| MLP | | | | | | | | |
| Stat+MLP | | | | | | | | |

---

# 55. 추가로 반드시 기록할 것

### 데이터 측면

```text
n_train_contexts
n_targets
n_cells
target overlap
context similarity
```

### 모델 측면

```text
parameter count
training time
GPU memory
inference time
```

### OOD 측면

```text
distance_to_nearest_train_context
target_seen_count
```

---

# 56. 왜 OOD distance를 기록해야 하는가?

같은 모델이라도:

```text
reference와 가까운 cell
```

에서는 잘 되고,

```text
매우 새로운 cell
```

에서는 무너질 수 있다.

그러면 score만 보면:

> 모델이 랜덤하게 불안정해 보임.

하지만 context distance와 함께 보면:

> **OOD가 멀어질수록 특정 metric이 떨어지는구나**

라는 원인을 볼 수 있다.

---

# 57. Metric × OOD Distance 분석

예:

```text
x-axis:
distance to nearest training context

y-axis:
metric score
```

각 metric에 대해 plot.

이 분석은 2026에 특히 중요하다.

---

# 58. 예상되는 패턴

가설:

### PDS

gene prior가 충분하면 context distance에 비교적 덜 민감할 수 있음.

### Jaccard/LFC

context-specific pathway activity에 민감해 OOD distance가 커질수록 많이 떨어질 수 있음.

이걸 실제 데이터로 검증한다.

---

# 59. 통계 vs 학습 비교에서 꼭 볼 것 — Data Scaling

training contexts를:

```text
2
4
6
8
...
```

로 늘리면서 성능을 본다.

질문:

> **모델 성능이 “cell 수” 때문인가 “context 수” 때문인가?**

2026에서는 후자가 훨씬 중요할 가능성이 높다.

---

# 60. 추천 Data Scaling Experiment

동일한 총 cell 수를 맞추고:

### Setting A

```text
2 contexts × 많은 cells
```

### Setting B

```text
8 contexts × 적은 cells
```

비교.

이것은 매우 중요한 연구 결과가 될 수 있다.

---

# 61. 또 하나 — Target Coverage vs Context Coverage

### Experiment A

많은 target genes / 적은 contexts

예:

```text
X-Atlas
```

### Experiment B

적은 targets / 많은 contexts

예:

```text
GSE281048
```

### Experiment C

둘 결합.

우리가 알고 싶은 것:

$$
Performance
=
f(
GeneCoverage,
ContextCoverage
)
$$

에서 2026 zero-shot에서는 어느 쪽이 더 중요한가?

---

# 62. 이 결과가 데이터 수집 우선순위도 바꾼다

예를 들어:

### Context coverage가 압도적으로 중요

→ GSE281048 / GSE264667 / primary data 확대.

### Gene coverage가 병목

→ X-Atlas / Replogle 확대.

---

# 63. 최종적으로 추천하는 개발 철학

## 하지 말 것

```text
Data 전부 다운로드
↓
STATE/Transformer 크게 돌리기
↓
Leaderboard 확인
↓
튜닝
```

## 해야 할 것

```text
Metric 이해
↓
Evaluator 재현
↓
Metric sanity test
↓
Shadow challenge
↓
Statistical baseline ladder
↓
Metric별 실패 분석
↓
Simple learning
↓
Statistical vs learning
↓
Ablation
↓
필요할 때만 복잡한 모델
```

---

# 64. 가장 중요한 의사결정 Gate

## Gate 1

**B4/B5 statistical baseline이 얼마나 강한가?**

강하다:

```text
→ hybrid residual
```

약하다:

```text
→ context representation부터 개선
```

## Gate 2

**ESM2가 seen-gene뿐 아니라 held-out gene에서도 개선하는가?**

YES:

```text
→ multi-prior gene encoder
```

NO:

```text
→ protein prior보다 perturbation data coverage 우선
```

## Gate 3

**pseudo-bulk mean은 좋은데 DE metrics가 안 좋은가?**

YES:

```text
→ single-cell distribution generator 문제
```

NO:

```text
→ effect prediction 자체 문제
```

## Gate 4

**Context distance가 멀수록 급격히 무너지나?**

YES:

```text
→ context encoder / MoE / similarity model 개선
```

## Gate 5

**Neural model이 통계 baseline을 consistently 이기나?**

NO:

> 큰 neural model로 가지 않는다.

2025 결과의 가장 중요한 교훈이다.

---

# 65. 처음 구현할 최종 baseline 조합

## Statistical Baseline

```text
Reference pseudobulk CRISPRi library
+
target-control similarity
+
gene-specific weighted transfer
```

## Learning Baseline

```text
NT control pseudo-bulk/PCA
+
ESM2(gene)
→ MLP
→ Δ
```

## Hybrid Baseline — 가장 추천

```text
Statistical Δ
+
NT context embedding
+
ESM2
→ residual MLP
→ corrected Δ
```

---

# 66. 이후에 추가할 것

순서:

1. STRING
2. DepMap
3. pretrained context encoder
4. target-pathway attention
5. Set Transformer
6. Mixture-of-Experts
7. Flow / STATE-style distribution

---

# 67. 2025 → 2026 변화가 우리 계획에 주는 가장 큰 메시지

## 2025

Metric optimization 자체가 꽤 중요한 경쟁 요소였다.

특히:

- PDS scaling
- pseudo-bulk
- loss weighting

이 큰 영향을 줬다.

## 2026

Metric을 이해하는 것은 여전히 중요하지만,
**한 metric의 허점을 공략하는 전략은 훨씬 약해졌다.**

왜냐하면:

```text
PDS만 잘함      → 다른 5개가 잡음
Sign만 잘함     → Jaccard/LFC가 잡음
Mean만 잘함     → PDS/DE가 잡음
DE set만 잘함   → direction/magnitude가 잡음
```

이기 때문이다.

---

# 68. 따라서 Metric 분석의 목적도 바뀐다

2025식:

> “어떤 metric을 최적화하면 leaderboard가 오르나?”

가 아니라,

2026식:

> **“우리 모델이 biology의 어떤 부분을 못 맞히고 있고, 그걸 가장 싸게 고칠 방법이 무엇인가?”**

가 되어야 한다.

---

# 69. 가장 중요한 세 개의 초기 연구 목표

## Goal A — Metric map

**Prediction 변형 → 6 metric 반응**

을 완전히 이해.

## Goal B — Statistical ceiling

**학습 없이 public perturbation transfer만으로 어디까지 가능한가?**

를 확인.

## Goal C — Neural added value

Neural model이:

> “통계 baseline에서 무엇을 추가로 배우는가?”

를 증명.

---

# 70. 우리가 제일 먼저 만들어야 하는 결과 Figure

## Figure 1 — Baseline Ladder

```text
B0
B1
B2
B3
B4
B5
L1
L2
L3
```

x-axis = method complexity  
y-axis = overall scaled score

## Figure 2 — Metric Heatmap

Method × 6 metrics.

어떤 모델이 어느 metric에 강한지.

## Figure 3 — Context Robustness

Method × held-out cell context.

## Figure 4 — Statistical vs Neural

각 metric에서:

```text
Statistical
Neural
Hybrid
```

비교.

## Figure 5 — Context Distance

OOD distance vs score.

## Figure 6 — Data Scaling

```text
# contexts
vs
performance
```

---

# 71. 최종적으로 “성공”이라고 판단할 기준

새 모델을 채택하려면:

### 조건 1

Local shadow CV overall 증가.

### 조건 2

Worst-context 성능 증가/유지.

### 조건 3

6 metric 중 1개만 크게 올리고 3개를 희생하지 않음.

### 조건 4

최소 두 개 이상의 held-out contexts에서 효과 재현.

### 조건 5

개선량이 metric의 noise floor보다 큼.

---

# 72. Metric noise보다 작은 차이는 무시

공식 reference split 분석상 metric별 구별 불가능한 차이가 대략:

- PDS: 0.5–1.3%
- Expr: 0.8–2.2%
- Fidelity: 1.5–4.8%
- Reach: 0.2–0.9%
- Jaccard: 0.8–2.2%
- LFC: 0.7–2.7%

정도 존재한다.

따라서:

```text
+0.001
```

같은 미세한 차이를 architecture superiority로 해석하지 않는다.

---

# 73. 최종 프로젝트 흐름

```text
               ┌──────────────────────┐
               │  Official Evaluator  │
               └──────────┬───────────┘
                          │
                          ▼
                Metric Sanity Tests
                          │
                          ▼
               Shadow VCC Benchmark
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 Statistical Baseline                Learning Baseline
 B0→B5                              Ridge→MLP
          │                               │
          └───────────────┬───────────────┘
                          ▼
                  Fair Comparison
                          │
                          ▼
              Metric-specific Diagnosis
                          │
                          ▼
             Statistical + Neural Hybrid
                          │
                          ▼
                  Representation Ablation
                          │
                          ▼
            Single-cell Generator Analysis
                          │
                          ▼
               Advanced Model 필요 판단
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         충분함                     부족함
      Hybrid 유지             Set/Flow/STATE 확장
```

---

# 74. 가장 중요한 실무 To-do — 우선순위

## P0 — 오늘 바로

- [ ] 공식 `cell-eval2` 환경 구축
- [ ] 6 metric을 코드로 직접 실행
- [ ] metric raw/scaled report 자동 생성
- [ ] raw-count submission validator 생성

## P1 — 바로 다음

- [ ] GSE281048 LOCO split
- [ ] H1 zero-shot split
- [ ] HepG2/Jurkat holdout split
- [ ] Shadow benchmark manifest 고정

## P2

- [ ] B0 Control
- [ ] B1 Global
- [ ] B2 Gene mean
- [ ] B3 Nearest
- [ ] B4 Weighted TransPert
- [ ] B5 Gene-specific transfer

## P3

- [ ] Ridge
- [ ] MLP + ESM2
- [ ] Statistical + residual MLP

## P4

- [ ] Context encoder ablation
- [ ] STRING / DepMap
- [ ] Generator comparison

## P5

- [ ] Set encoder
- [ ] Flow / generative population model

---

# 75. 최종 결론

현재 생각한 방향은 정확하다.

다만 이렇게 표현하는 것이 가장 좋다.

> **“우리는 먼저 2026의 6개 metric을 완전히 재현하고, 각 metric이 어떤 prediction failure를 잡는지 synthetic/sanity experiment로 파악한다. 그다음 no-effect → global effect → gene-specific effect → nearest-context → similarity-weighted transfer까지의 완전한 statistical baseline ladder를 구축한다. 이후 동일한 데이터·split·single-cell generator·evaluator 조건에서 Ridge/MLP/context encoder 등의 학습 모델과 공정하게 비교한다. 이때 overall score뿐 아니라 6 metric과 held-out context별 성능을 분해하여, neural model이 statistical baseline의 어떤 한계를 실제로 보완하는지 확인한다. 그 결과가 확인된 뒤에만 Set Transformer, STATE, Flow 같은 복잡한 모델로 확장한다.”**

이 순서는:

- 2025의 <b>“통계 baseline이 생각보다 강했다”</b>는 교훈
- 2025의 **metric-specific optimization 문제**
- 2026의 **6-metric generalist 평가**
- 2026의 **zero-shot multi-context 구조**

를 모두 반영한다.

---

# 76. 핵심 한 줄

$$
\boxed{
\text{Metric 이해}
\rightarrow
\text{Statistical Ceiling 확인}
\rightarrow
\text{Neural Added Value 증명}
\rightarrow
\text{필요한 만큼만 복잡하게}
}
$$

이 순서로 시작하는 것이 현재 가장 합리적이다.

---

# 참고

## 2026 Metric specification
- Virtual Cell Challenge Evaluation
- `cell-eval2`
- `configs/vcc2026.yaml`
- `rule_version 3`

## 2025 comparison
- Arc Institute, **Behind the Data of the Virtual Cell Challenge**
- Arc Institute, **Virtual Cell Challenge 2025 Wrap-Up: Winners and Reflections**

2025의 핵심 공식 지표는 `PDS / DES / MAE`였으며,
2026에는 `pds_cosine / noise-corrected normalized expression MSE / 4개의 DE-specific metrics`로 확장되었다.

---

**End of plan**

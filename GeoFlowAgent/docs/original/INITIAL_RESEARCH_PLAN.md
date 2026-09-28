> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../README.md) · [주장과 근거](../research/04_CLAIMS_AND_EVIDENCE.md).

# FlowAgent의 Frozen Weight 현황과 거리·Geometry 모듈 비교 연구안

> 확인 기준: 2026-09-14  
> 대상 논문: *Tools as Continuous Flow for Evolving Agentic Reasoning* (FlowAgent), arXiv v2  
> 목표: 공개된 frozen backbone을 활용해, 작은 학습 모듈만으로 agent trajectory의 기능적 geometry를 학습하는 실험을 설계한다.

---

## 0. 먼저 결론

1. **FlowAgent가 frozen encoder로 사용한 기반 모델의 원본 가중치는 있다.**
   - 기본 encoder는 공개된 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)다.
   - 논문의 encoder sensitivity 실험에는 Llama-3.2-3B-Instruct와 BERT-base-uncased도 사용되었다.

2. **하지만 저자가 학습한 FlowAgent 자체의 가중치는 공식 저장소·Releases와 확인 가능한 공식 배포 경로에서 찾지 못했다.**
   - flow-matching velocity network
   - 3,584→768 projection 및 trajectory target용 projection들
   - stop module과 학습형 decoder variant
   - executor LoRA adapter
   - 부가 실험에 사용된 soft-prefix·ToolRL·selector checkpoint

3. 따라서 지금 가능한 것은 논문의 완성 모델을 그대로 불러오는 **checkpoint reproduction**이 아니라,

   > 공개 Qwen 가중치를 특정 revision으로 고정하고, 그 위에 작은 geometry 모듈을 새로 학습하는 controlled reimplementation

   이다.

4. 여러 거리를 비교하는 발상은 좋다. 다만 연구 질문을 단순히 “Euclidean보다 어떤 거리가 좋은가?”로 두기보다는 다음처럼 잡는 것이 강하다.

   > **Frozen semantic space를 유지한 채, 얼마나 작은 학습 모듈로 state-dependent functional trajectory geometry를 복원할 수 있는가?**

5. 가장 추천하는 최종 방향은 단순 Mahalanobis distance에서 끝나지 않고, **현재 state에서 각 tool이 만들 successor effect를 비교하는 모듈**까지 가는 것이다.

---

## 1. “Frozen model의 가중치가 있다”는 말의 정확한 의미

### 1.1 Frozen model은 별도의 특별한 checkpoint가 아니다

`frozen Qwen encoder`란 Qwen 가중치를 불러온 뒤 학습 중 gradient update를 막는다는 뜻이다.

```python
encoder = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    torch_dtype=torch.bfloat16,
)
encoder.eval()
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
```

즉 Qwen 원본 weight와 “FlowAgent frozen weight”가 따로 존재하는 것이 아니다. 원본 Qwen을 불러와 고정해서 사용한 것이다.

### 1.2 FlowAgent에서 실제로 frozen인 부분

[논문 §3.3](https://arxiv.org/html/2605.07339v2)는 기본 `Enc`를 Qwen2.5-7B-Instruct의 frozen transformer trunk로 정의한다. Appendix B.4에 따르면 텍스트를 최대 2,048 token으로 자르고, 마지막 layer hidden state를 attention-mask mean pooling한 뒤 L2 정규화하고 768차원 latent로 projection한다. 논문이 명시적으로 frozen이라고 한 것은 encoder trunk와 tokenizer이며, 마지막 projection은 Qwen 원본 weight에 포함되지 않는다. 다만 해당 projection의 학습·초기화 세부는 공개 설명만으로 완전히 확정하기 어렵다.

```text
canonicalized text
    ↓ frozen Qwen transformer trunk
last hidden states
    ↓ masked mean pooling
3,584-dimensional vector
    ↓ L2 normalization
FlowAgent-side projection
768-dimensional latent representation
```

핵심은 마지막 projection부터는 Qwen 원본에 포함된 weight가 아니라는 점이다.

### 1.3 무엇이 공개되어 있고 무엇이 없는가

| 구성 요소 | 논문에서의 역할 | Frozen? | 공개 weight 현황 |
|---|---|---:|---|
| Qwen2.5-7B-Instruct trunk | text feature encoder | 예 | **공개** |
| Llama-3.2-3B-Instruct trunk | encoder sensitivity | 예 | 공개되어 있으나 사용 승인 필요 |
| BERT-base-uncased | encoder sensitivity | 예 | 공개 |
| 3,584→768 projection | encoder 출력을 latent로 정렬 | Qwen에는 없음 | 저자 artifact 미제공; 학습 세부 일부 불명확 |
| `W_t, W_r, W_o, W_p` | tool/rationale/observation/phase 결합 | Qwen에는 없음 | 저자 artifact 미제공 |
| conditional flow velocity field | latent future trajectory 생성 | 아니요 | 미공개 |
| raw L2 tool decoder | anchor를 discrete tool로 변환 | 해당 없음 | 거리 규칙 자체는 별도 weight 없음 |
| learned pointer/decoder variant | anchor와 tool compatibility 학습 | 아니요 | 저자 checkpoint 미제공 |
| stop mechanism | 계획 종료 위치 예측 | 아니요 | 미공개 |
| executor LoRA | 실제 JSON action·argument 생성 | 아니요 | 미공개 |

공식 저장소는 curated snapshot에서 large checkpoint를 제외했다고 [README](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/README.md#L25)에 명시한다. [재현성 문서](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/docs/BASELINE_REPRODUCIBILITY.md#L5)도 model checkpoint, LoRA adapter, ToolRL checkpoint 등을 포함하지 않았다고 적고 있으며, 공식 GitHub에는 현재 [release도 없다](https://github.com/ssy166/FlowPlan/releases). 따라서 “세상 어디에도 없다”가 아니라, **확인한 공식 배포 범위에서는 제공되지 않는다**고 표현하는 것이 정확하다.

### 1.4 확인된 버전과 현재 로컬 상태

- 논문: arXiv v2, 2026-08-12
- 확인한 공식 FlowPlan repository commit: `c65155b85931e5d38c38573c8e4f84d31b056c45` (2026-07-25)
- 확인한 Qwen2.5-7B-Instruct revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Qwen 라이선스: Apache-2.0
- 현재 Qwen BF16 safetensor shard 합계: 약 15.2 GB
- 이 Mac의 일반 Hugging Face cache에는 위 Qwen/Llama/BERT snapshot이 현재 없음
- 이 workspace에도 FlowAgent `.pt`, `.ckpt`, `.safetensors` checkpoint가 없음

따라서 실험을 시작할 때 Qwen weight를 처음 한 번 내려받아야 한다. 다만 embedding을 cache하고 나면 metric 실험 중에는 7B encoder를 계속 GPU에 올릴 필요가 없다.

> 주의: 논문 v2가 저장소 마지막 commit보다 늦다. 공개 코드 snapshot과 논문 v2의 최종 실험 코드가 완전히 동일하다고 가정해서는 안 된다. 또한 저장소에는 현재 명시적인 LICENSE 파일이 보이지 않으므로, 코드를 그대로 재배포하려면 저자에게 별도 확인하는 편이 안전하다.

---

## 2. 정확히 무엇을 재현할 수 있고, 무엇을 재현할 수 없는가

### 가능한 것

- 동일한 공개 Qwen backbone을 frozen encoder로 사용
- 논문에 기술된 pooling과 projection 구조를 재구현
- 공개 benchmark와 script를 바탕으로 새로운 planner·decoder를 학습
- 같은 frozen representation 위에서 여러 geometry head를 공정하게 비교
- 01_Medical/Genomics용 state, tool contract, transition data로 새 모듈을 학습

### 현재 불가능한 것

- 저자의 trained flow latent를 그대로 추출
- 저자의 generated anchor `z`에 새 metric head만 꽂아 완전히 동일한 조건으로 비교
- 논문 수치를 checkpoint inference만으로 재현

이 차이 때문에 프로젝트 이름도 “FlowAgent checkpoint fine-tuning”보다는 다음이 정확하다.

> **FlowAgent-inspired frozen-backbone trajectory geometry study**

---

## 3. Euclidean distance가 왜 부족할 수 있는가

FlowAgent의 discrete decoder는 다음 확률을 사용한다.

\[
p(t\mid z,T)\propto
\exp\left(-\frac{\lVert z-e_t\rVert_2^2}{\epsilon}\right)
\]

여기서 `z`는 flow planner가 생성한 latent anchor이고, `e_t`는 tool의 semantic embedding이다. 이 식은 자연스럽고 강한 baseline이지만 다음 가정을 암묵적으로 둔다.

1. latent의 모든 축이 같은 중요도를 가진다.
2. anchor와 tool embedding이 같은 좌표계와 통계를 가진다.
3. tool 사이의 관계가 현재 state와 무관하게 고정되어 있다.
4. tool description이 비슷하면 실제 effect도 비슷하다.
5. tool의 유용성은 대칭적인 거리로 표현할 수 있다.

Agent의 실제 action 선택에서는 이 가정들이 자주 깨진다.

- 같은 `search_variant`라도 variant ID가 이미 있는 상태와 없는 상태에서 가치는 다르다.
- 이름과 설명이 비슷한 두 tool이 서로 다른 database version이나 evidence level을 반환할 수 있다.
- `HGVS → genomic coordinate`와 그 역변환은 텍스트상 가깝지만 방향과 precondition이 다르다.
- 현재 action이 좋은지는 그 action 자체보다 이후 goal까지의 경로 비용과 복구 가능성에 의해 결정된다.
- frozen LLM embedding은 특정 방향에 몰리는 anisotropy와 norm 차이를 가질 수 있다.

따라서 “semantic similarity”와 “현재 trajectory에서의 functional usefulness”를 구분해야 한다.

하지만 Euclidean이 안 좋을 것이라고 미리 결론 내리면 안 된다. 반드시 baseline으로 두고, **어떤 가정이 깨져서 성능이 달라지는지**를 단계적으로 검증해야 한다.

---

## 4. 매우 중요한 함정: Cosine과 normalized Euclidean은 사실상 같다

두 벡터를 거리 계산 직전에 unit normalization하면,

\[
\lVert \hat z-\hat e\rVert_2^2
=2\left(1-\cos(\hat z,\hat e)\right)
\]

이므로 nearest-neighbor 순위는 완전히 같다. 따라서 다음 두 방법을 별개의 핵심 모델처럼 비교하는 것은 정보량이 적다.

- cosine similarity
- unit-normalized squared Euclidean distance

FlowAgent 논문은 pooling 결과를 정규화한 뒤 768차원 projection을 적용한다. projection 이후 다시 정규화하지 않는다면 최종 decoder 공간에서는 둘이 달라질 수 있다. 그래서 실험에서는 다음을 분리해야 한다.

1. projection 전 normalization
2. projection 후 normalization
3. raw L2
4. post-normalized L2/cosine

이 작은 통제가 없으면 “metric이 좋아졌다”는 결과가 단순 norm calibration 때문일 수 있다.

순위는 같아도 같은 temperature를 넣은 확률분포까지 같지는 않다. 위 식에서 squared L2가 cosine distance의 두 배이므로 동일한 softmax 분포를 맞추려면 `ε_cos = ε_L2 / 2`로 scale을 대응시켜야 한다. 실제 구현에서는 영벡터 정규화를 피하도록 norm 분모에 `clamp_min(1e-8)` 같은 안정화도 필요하다.

### 더 중요한 함정: learned projection 뒤의 Euclidean은 이미 “학습된 거리”일 수 있다

원래 frozen embedding을 `x`라고 하고 trainable linear projection `P`를 거친 뒤 L2를 계산하면,

\[
\lVert P(x_1-x_2)\rVert_2^2
=(x_1-x_2)^\top P^\top P(x_1-x_2)
\]

이다. 즉 원래 embedding 공간에서 보면 이는 이미 `M=PᵀP`인 Mahalanobis distance다. 따라서 다음 비교는 연구적으로 모호하다.

```text
Model A: trainable projection A + Euclidean
Model B: trainable projection B + Mahalanobis
```

Model B가 좋아져도 projection 차이인지 metric 차이인지 알 수 없다. 특히 projection을 충분히 자유롭게 학습하면 low-rank Mahalanobis head와 표현력이 중복될 수 있다.

깨끗한 비교를 위해서는 두 track을 분리한다.

- **Metric-only track:** 같은 frozen embedding, 같은 고정 projection, 같은 입력만 사용하고 거리 함수만 교체
- **Representation+metric track:** projection과 metric을 함께 학습하되 parameter budget을 맞추고, 개선을 “거리만의 효과”라고 주장하지 않음

이 때문에 단순 Mahalanobis가 최종 novelty가 되기 어렵고, 고정 선형 변환으로 표현할 수 없는 **state-conditioned geometry**와 **successor effect**가 더 중요한 연구 대상이 된다.

---

## 5. 또 하나의 구분: Flow loss와 tool-decoding distance

FlowAgent에는 서로 다른 두 종류의 Euclidean 가정이 들어 있다.

### A. Flow를 학습하는 velocity loss

\[
\mathcal L_{FM}
=\mathbb E\left[\lVert v_\phi-u^*\rVert_2^2\right]
\]

이는 trajectory vector field를 맞추는 MSE다.

### B. Anchor에서 tool을 선택하는 decoder distance

\[
D(z,e_t)=\lVert z-e_t\rVert_2^2
\]

이는 discrete action 선택용 geometry다.

첫 프로젝트에서는 **A는 고정하고 B만 바꾸는 것**이 좋다. 둘을 동시에 바꾸면 성능 변화가 flow 자체 때문인지 decoder 때문인지 알 수 없다. decoder에서 가장 좋은 두 방법이 확인된 뒤, 2차 실험에서 flow loss까지 geometry-aware하게 바꾸면 된다.

---

## 6. 추천하는 가설 계층

여러 모델을 단순 나열하기보다, 각 모델이 하나의 명확한 실패 원인을 검사하게 구성한다.

| ID | 방법 | 검사하는 가설 | 우선순위 |
|---|---|---|---:|
| B0 | Raw Euclidean | 원래 FlowAgent 가정으로 충분한가 | 필수 |
| B1 | Cosine / post-normalized L2 | norm이 문제인가 | 필수 |
| B2 | Whitened cosine | frozen space의 anisotropy가 문제인가 | 필수 |
| M1 | Diagonal Mahalanobis | 축별 재가중이면 충분한가 | 높음 |
| M2 | Low-rank Mahalanobis | 축 간 선형 상호작용이 필요한가 | 높음 |
| M3 | Asymmetric bilinear | anchor와 tool이 서로 다른 modality인가 | 높음 |
| M4 | State-conditioned mixture | 올바른 geometry가 state마다 달라지는가 | 매우 높음 |
| M5 | Contract-aware energy | precondition/effect가 text similarity보다 중요한가 | 매우 높음 |
| M6 | Successor-effect matching | action이 아니라 결과 변화의 geometry가 핵심인가 | 핵심 제안 |

이 순서대로 성능이 올라간다면 단순 leaderboard가 아니라 원인을 해석할 수 있다.

```text
cosine 향상
  → norm 문제
whitening 추가 향상
  → embedding anisotropy 문제
Mahalanobis 추가 향상
  → global linear distortion
bilinear 추가 향상
  → anchor/tool modality mismatch
state conditioning 추가 향상
  → 정적 geometry의 한계
successor effect가 long-horizon 성능 향상
  → 결과 상태를 보는 trajectory geometry가 필요
```

---

## 7. 각 학습 모듈의 구체적인 형태

### 7.1 B0 — Raw Euclidean

\[
D_{L2}(z,e_t)=\lVert z-e_t\rVert_2^2
\]

- 논문과 직접 대응되는 기준점
- 학습 파라미터 0개
- 반드시 포함해야 한다.

### 7.2 B1 — Cosine / post-normalized L2

\[
D_{cos}(z,e_t)=1-
\frac{z^\top e_t}{\lVert z\rVert\lVert e_t\rVert}
\]

- vector norm의 영향을 제거한다.
- post-normalized L2와 ranking이 같으므로 하나의 모델군으로 보고한다.
- 이것만 좋아지면 복잡한 metric learning이 아니라 normalization 문제였던 것이다.

### 7.3 B2 — Whitened cosine

훈련 split에서만 평균과 covariance를 추정해 frozen embedding을 whitening한 뒤 cosine을 계산한다.

\[
\tilde x=\Sigma^{-1/2}(x-\mu)
\]

- trainable neural module 없이 frozen space의 공통 방향을 제거하는 baseline
- validation/test 정보로 whitening statistics를 계산하면 leakage이므로 주의
- full covariance가 불안정하면 PCA 차원 축소 또는 shrinkage covariance 사용

### 7.4 M1/M2 — Diagonal 또는 low-rank Mahalanobis

\[
D_M(z,e_t)
=(z-e_t)^\top M(z-e_t),
\qquad M=L^\top L+\delta I
\]

- diagonal: 768개 scale parameter 정도
- low-rank `r=32`: 약 24.6K parameter
- low-rank `r=64`: 약 49.2K parameter
- `M=L^T L`은 positive semi-definite이고, `δ>0`인 `M=L^T L+δI`는 positive definite가 되어 collapse를 줄일 수 있다.

권장 비교:

```text
identity → diagonal → rank 8 → rank 32 → rank 64 → rank 128
```

full 768×768 matrix는 데이터가 작을 때 과적합하기 쉬우므로 처음부터 사용하지 않는다. 고전적인 supervised metric learning의 대표적 출발점으로 [LMNN](https://jmlr.org/papers/v10/weinberger09a.html)을 참고할 수 있다.

### 7.5 M3 — Asymmetric bilinear compatibility

\[
S_\theta(z,t)=
\frac{(Az)^\top(Be_t)}{\sqrt r}
\]

`z`에는 query, state, observation, plan position 정보가 섞이고 `e_t`에는 name, description, input/output schema가 들어간다. 두 벡터가 같은 공간에서 대칭적으로 비교되어야 할 이유가 없으므로, 별도의 projection `A`, `B`를 두는 것이 합리적이다.

다만 공개 FlowPlan의 [pointer decoder](https://github.com/ssy166/FlowPlan/blob/main/scripts/train_fm_pointer_decoder.py)에도 anchor와 tool feature를 따로 projection해 dot product로 비교하는 아이디어가 이미 들어 있다. 따라서 **bilinear 자체를 연구의 novelty로 주장하기보다는**, 이후의 state·contract·regret supervision과 결합하는 것이 중요하다.

### 7.6 M4 — State-conditioned mixture of local energies

\[
D_\theta(z,e_t\mid s)=
\sum_{k=1}^{K}\alpha_k(s)
(z-e_t)^\top M_k(z-e_t)
\]

\[
\alpha(s)=\operatorname{softmax}(g_\theta(s))
\]

- `K=4` 정도의 diagonal 또는 rank-32 metric 사용
- 현재 state가 routing weight를 결정
- identifier normalization, evidence retrieval, evidence aggregation, report/write 단계가 서로 다른 geometry를 사용하게 한다.
- router weight를 시각화하면 어떤 단계에서 어떤 metric이 선택되는지 해석할 수 있다.

각 state 안에서는 PSD quadratic form을 유지하지만, state에 따라 함수가 달라지므로 전체 입력 공간에서 하나의 고정 metric 공리를 만족한다고 보장할 수는 없다. 따라서 엄밀하게는 state-conditioned distance보다 **local dissimilarity 또는 energy**라고 부르는 편이 정확하다. 이 모델은 다음 가설을 직접 검증한다.

> 기능적 거리는 하나의 고정 공간이 아니라 agent의 현재 state에 따라 변한다.

### 7.7 M5 — Contract-aware energy

거리만으로 tool을 고르는 대신 작은 compatibility energy를 더한다.

\[
E_\theta(s,z,t)
=\alpha D_M(z,e_t)
+w_p^\top\phi_{penalty}(s,t)
-w_u^\top\phi_{utility}(s,t)
\]

`φ_penalty(s,t)` 후보:

- tool precondition 위반 여부
- 빠진 필수 argument 수
- input/output type incompatibility
- 이미 얻은 정보를 반복하는 정도
- 비용·latency
- 비가역성
- safety flag
- evidence freshness와 database version

`φ_utility(s,t)`에는 현재 goal predicate coverage나 새로운 evidence gain처럼 클수록 좋은 값을 넣는다. penalty와 utility의 부호를 분리해야 어떤 feature가 energy를 높이거나 낮추는지 해석할 수 있다.

\[
p_\theta(t\mid s,z,T)=
\frac{\exp[-E_\theta(s,z,t)/\tau]}
{\sum_{u\in T}\exp[-E_\theta(s,z,u)/\tau]}
\]

처음에는 선형 additive energy로 시작하고 이후 작은 2-layer MLP와 비교한다. 명시적으로 실행 불가능하거나 위험한 action은 learned score에만 맡기지 말고 hard applicability mask를 적용한 버전도 별도로 평가한다.

### 7.8 M6 — Successor-effect matching: 가장 추천하는 모델

단순 decoder는 다음을 묻는다.

> 현재 anchor와 어느 tool description이 가까운가?

trajectory 관점에서는 질문을 이렇게 바꾸는 편이 더 자연스럽다.

> 현재 상태에서 필요한 변화와, 이 tool을 실행했을 때 일어날 변화가 가까운가?

실제 tool effect는 tool 이름만으로 결정되지 않는다. 같은 tool도 argument와 stochastic observation에 따라 결과가 달라진다. 따라서 action을 `a=(t,args)`로 두고, 현재 상태 표현을 `ψ(s)`, 실행 후 상태를 `s'`라고 하면 작은 transition distribution을 학습한다.

\[
p_\theta\!\left(\Delta\psi\mid
s,a,\operatorname{contract}(t)\right)
\]

MVP에서는 이 분포의 평균만 예측해도 된다.

\[
\widehat{\psi(s')}=
\psi(s)+\mathbb E_\theta[\Delta\psi\mid s,a]
\]

anchor와 goal까지 남은 상태 차이로부터 원하는 progress vector를 만든다.

\[
u_h=h_\omega\!\left(
z_h,\psi(G)-\psi(s_h)
\right)
\]

그리고 실제 action 선택은 기대 effect space에서 한다.

\[
E_{effect}(s,z,a)=
\mathbb E_{\Delta\psi\sim p_\theta}
\left[D\left(u_h,\Delta\psi\right)\right]
+\lambda_c\operatorname{Cost}(a)
+\lambda_v\operatorname{Violation}(s,a)
\]

tool-selection 단계에서 아직 arguments가 없다면 두 가지 구현이 가능하다. 첫째, 고정 argument generator로 후보 argument를 먼저 만든 뒤 action energy를 계산한다. 둘째, 가능한 argument에 대한 expected effect를 tool-level score로 marginalize한다. 모든 비교 모델에 동일한 argument generator를 사용해야 한다.

이 결과가 좋으면 결론도 더 강해진다.

> Agent의 올바른 action geometry는 tool-description space가 아니라 successor-state/effect space에서 형성된다.

이것이 사용자의 “trajectory를 이해하는 정도가 모델의 성능을 좌우한다”는 관점과 가장 직접적으로 연결된다.

---

## 8. 거리뿐 아니라 supervision도 바꿔야 한다

하나의 expert tool만 정답인 one-hot cross entropy는 여러 유효한 경로가 존재하는 agent 문제와 잘 맞지 않는다. 같은 goal에 도달하는 두 tool 중 하나만 정답으로 주면 나머지는 잘못된 negative가 된다.

### Functional regret label

작은 symbolic simulator에서 goal까지의 최소 기대 잔여 비용을 `V*(s)`라 하자. 실제 임상 환경에서는 정확한 `V*`를 알 수 없으므로 이 정의를 직접 계산할 수 없으며, offline return이나 heuristic cost-to-go로 근사해야 한다.

\[
r(s,a)=
c(s,a)+
\mathbb E_{s'\sim P(\cdot\mid s,a)}V^*(s')
-V^*(s)
\]

- 최적 action: `r=0`
- 유효하지만 우회하는 action: 작은 양수
- 중복·비효율 action: 더 큰 양수
- invalid·unsafe action: 매우 큰 penalty 또는 hard mask

이를 action-level soft label로 만든다.

\[
q_\beta(a\mid s)\propto
\mathbf1[a\text{ is safe and applicable}]
\exp\left(-\frac{r(s,a)}{\beta}\right)
\]

tool-only ranking에서는 고정 argument generator 아래의 expected regret를 사용해 `a`를 `t`로 축약할 수 있다. safe/applicable action이 하나도 없는 state에서도 분모가 0이 되지 않도록 `STOP`, `ASK`, `ABSTAIN` 중 task에 맞는 fallback action을 candidate set에 반드시 포함한다.

모든 decoder에 동일한 listwise objective를 적용한다.

\[
\mathcal L_{list}
=\operatorname{KL}
\left(q_\beta(\cdot\mid s)\Vert
p_\theta(\cdot\mid s,z)\right)
\]

보조적으로 pairwise ranking loss나 supervised contrastive loss를 사용할 수 있다. 단, contrastive learning은 “거리의 종류”가 아니라 “표현을 학습하는 목적 함수”라는 점을 구분해야 한다. [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)은 참고점이 될 수 있지만, 이번 문제에는 연속적인 regret 정보를 보존하는 soft listwise target이 더 자연스럽다.

필수 비교:

1. 동일 decoder + one-hot CE
2. 동일 decoder + functional-regret soft target
3. 동일 decoder + regret target + semantic hard negatives

이렇게 해야 향상이 metric 때문인지 label 설계 때문인지 구분할 수 있다.

---

## 9. 추천 실험 설계

### 9.1 실험 A — Encoder-only frozen geometry study

현재 가장 현실적인 MVP다.

```text
Frozen Qwen encoder
    ↓ 한 번만 실행해 projection 전 3,584-D embedding cache
normalized context/state, tool, goal embedding
    ↓
작은 metric/energy module만 학습
    ↓
우선 next-tool ranking 평가
```

장점:

- 저자의 FlowAgent checkpoint가 없어도 가능
- 계산량이 작음
- module 차이를 깨끗하게 비교 가능
- 01_Medical/Genomics simulator를 붙이면 closed-loop로 확장하기 쉬움

한계:

- 저자의 flow-generated `z`가 아니라 우리가 정의한 anchor/state representation을 사용
- 따라서 “FlowAgent를 개선했다”보다 “FlowAgent-inspired geometry study”라고 표현해야 정확함

MVP 자체는 tool identity ranking만 평가한다. closed-loop success까지 주장하려면 모든 모델에 동일한 고정 또는 oracle argument grounder와 동일한 STOP 정책을 제공하고, tool execution으로 실제 state를 갱신해야 한다. 그렇지 않으면 metric head 성능과 argument 생성·종료 오류가 뒤섞인다.

metric-only 비교에서는 projection 전의 normalized 3,584-D frozen trunk 출력을 cache하고, 모든 모델에 같은 고정 projection을 제공한다. projection까지 모델별로 학습하는 결과는 별도의 joint-training 표로 분리한다.

### 9.2 실험 B — 공통 flow planner를 학습한 뒤 decoder만 비교

1. Qwen embedding을 cache한다.
2. 공개 구조를 바탕으로 하나의 conditional flow planner를 학습한다.
3. planner를 frozen한다.
4. 동일 anchor에 B0–M6 decoder를 붙여 비교한다.

이는 “이미 형성된 trajectory latent를 작은 geometry head가 얼마나 교정하는가?”를 본다.

주의점: planner가 원래 L2 decoder와 함께 학습되었다면 L2에 유리한 co-adaptation bias가 생긴다. 가능하면 decoder-independent trajectory loss로 planner를 먼저 학습하거나, 이 한계를 명시한다.

### 9.3 실험 C — Geometry별 planner 공동학습

각 geometry에 대해 동일 구조의 planner를 처음부터 다시 학습한다.

이는 가장 엄밀하게 inductive bias를 비교하지만 비용이 크고, planner 차이와 decoder 차이가 함께 들어간다. MVP 이후 논문 수준의 확장 실험으로 둔다.

---

## 10. 최소 실험 세트

처음부터 모든 방법을 구현할 필요는 없다. 다음 여섯 개면 충분하다.

1. Raw L2
2. Cosine/post-normalized L2
3. Whitened cosine
4. Low-rank Mahalanobis (`rank=32` 또는 `64`)
5. State-conditioned metric 또는 bilinear compatibility
6. Contract-aware successor-effect energy

추천 순서:

```text
Stage 1 — data pipeline 검증
Raw L2 / cosine / whitened cosine

Stage 2 — global geometry 학습
diagonal / low-rank Mahalanobis / bilinear

Stage 3 — trajectory-aware geometry
state-conditioned mixture / contract-aware energy

Stage 4 — 핵심 연구 모델
successor-effect matching
```

Stage 1에서 차이가 없다면 data와 evaluation을 먼저 점검한다. Stage 2가 좋아지지만 Stage 3가 개선되지 않으면 현재 task가 state dependence를 충분히 포함하지 않았을 가능성이 있다.

---

## 11. 공정한 비교를 위한 통제

### 모든 모델에 고정할 것

- 동일한 frozen encoder revision
- 동일한 embedding cache
- 동일한 train/dev/test split
- 동일한 candidate tool set
- 동일한 negative sample
- 동일한 update 수, batch 구성, hyperparameter tuning budget
- 동일한 random seed 목록
- 가능한 한 비슷한 trainable parameter budget
- 동일한 state와 tool text serialization

### 정보량과 metric 효과를 분리할 것

`state-conditioned` 또는 `contract-aware` 모델이 raw L2를 이겼다고 해서 곧바로 거리 함수가 우수하다고 말할 수는 없다. 해당 모델은 더 많은 입력 정보를 받기 때문이다. 결과 표를 두 층으로 나눈다.

1. **동일 정보 비교:** `z`와 `e_t`만 사용하는 L2, cosine, whitening, Mahalanobis, bilinear
2. **추가 정보의 가치:** 모든 비교군에 동일한 state/contract feature를 제공한 뒤 static energy와 dynamic energy 비교

또한 state나 contract를 하나씩 제거하는 ablation으로 정보의 기여와 geometry의 기여를 분리한다.

### 반드시 별도로 calibration할 것

거리마다 scale이 다르기 때문에 temperature `τ` 또는 `ε`를 공통 고정하면 불공정할 수 있다. 각 모델은 dev set에서 temperature를 calibration하고, ranking metric과 calibrated probability metric을 구분해 보고한다.

### split 원칙

무작위 row split만 사용하면 거의 같은 workflow template가 train과 test에 동시에 들어갈 수 있다. 최소한 다음 split을 둔다.

- held-out task template
- held-out tool family
- tool description paraphrase
- counterfactual state
- unseen database/source combination

unseen-tool 일반화를 주장하려면 trainable tool-ID embedding을 사용하지 말고, tool description·schema·contract로만 표현한다.

---

## 12. 평가 지표

### Step-level

- valid-action Recall@1, Recall@3
- NDCG
- mean action regret
- invalid action rate
- unsafe action rate
- calibration error

### Geometry-level

- functional-neighbor Recall@k
- distance와 action regret 사이의 Spearman correlation
- optimal action과 hard negative 사이의 margin
- state 변화 전후 neighbor ranking 변화

### Trajectory-level

- final task success
- cumulative regret
- optimality gap
- 평균 tool-call 수
- redundant/cyclic call 수
- recovery latency: 오류 후 정상 경로로 돌아오는 데 필요한 step 수

### Generalization

- unseen tool family
- unseen description 표현
- 입력 누락·오류·충돌 state
- database 결과가 empty/ambiguous한 경우
- 도구가 추가·삭제된 dynamic toolset

### Efficiency

- trainable parameter 수
- embedding cache 크기
- step당 decoder latency
- peak memory

모든 주요 수치는 여러 seed와 confidence interval로 보고한다. 같은 workflow에서 나온 여러 prefix row는 독립 표본이 아니므로 row 단위 bootstrap을 피하고, workflow/template 단위 paired 또는 hierarchical bootstrap을 사용한다.

---

## 13. 01_Medical/Genomics에서 특히 좋은 counterfactual 예시

같은 tool 설명이라도 state에 따라 정답이 바뀌도록 만드는 것이 핵심이다.

### Variant interpretation 예시

Goal: 환자의 variant에 대한 임상적 근거를 수집한다.

```text
State A: HGVS만 있고 genome build가 없음
  → normalize / resolve-build가 먼저

State B: GRCh38 coordinate와 allele가 검증됨
  → population frequency 또는 clinical evidence 조회

State C: ClinVar와 population evidence는 있지만 gene-disease 관계가 없음
  → gene-disease evidence tool

State D: 충분한 evidence가 있고 conflict가 존재함
  → conflict aggregation / expert-review routing
```

tool description 간 semantic similarity는 거의 그대로지만 최적 action은 state에 따라 달라진다. 이는 static Euclidean과 state-conditioned model의 차이를 검사하기 좋은 설정이다.

### Hard negative 구성

- 같은 entity를 입력받지만 다른 genome build를 요구하는 tool
- 같은 database의 search와 fetch-detail endpoint
- gene-level evidence와 variant-level evidence
- 읽기 전용 조회와 record mutation tool
- 최신 release와 오래된 release
- 유사한 description이지만 반환 type이 다른 tool

이러한 negative가 없으면 어떤 metric도 쉬운 keyword matching으로 높은 점수를 얻을 수 있다.

---

## 14. Genomics-safe frozen cache: 의미와 정확성을 분리하기

### 14.1 문제 제기는 정확하다

긴 state를 하나의 Qwen vector로 mean pooling하면 `GRCh37`과 `GRCh38`처럼 한두 token만 다른 핵심 정보가 희석될 수 있다. 만약 encoder가 두 입력을 사실상 같은 vector로 압축했다면, 뒤의 작은 모듈은 사라진 정보를 복원할 수 없다.

다만 두 assembly를 전역 embedding에서 무조건 멀리 보내는 것도 올바르지 않다.

```text
Semantic relation:
GRCh37 ≈ GRCh38
둘 다 인간 reference genome assembly이므로 가까운 것이 자연스럽다.

Operational identity:
GRCh37 ≠ GRCh38
특정 genomic coordinate와 tool input에서는 서로 교환할 수 없다.

Valid trajectory:
Variant<GRCh37> --remap/liftover--> Variant<GRCh38>
```

[NCBI Remap](https://ncbiinsights.ncbi.nlm.nih.gov/2014/01/16/ncbis-genome-remapping-service-assists-in-the-transition-to-the-new-human-genome-reference-assembly-grch38/)이 GRCh37과 GRCh38 사이를 별도의 coordinate remapping 문제로 다루는 이유도 여기에 있다. [HGVS nomenclature](https://varnomen.hgvs.org/bg-material/simple/) 역시 reference sequence의 accession과 version이 필요하며, reference sequence가 없으면 genome build를 알아야 한다고 설명한다.

따라서 문제는 “Qwen이 두 단어를 가깝게 놓았다”가 아니라,

> semantic proximity를 operational substitutability로 잘못 사용한 decoder 설계

에 있다.

### 14.2 하나의 embedding에 모든 책임을 맡기지 않는다

권장 표현은 최소 세 채널로 나눈다.

\[
h_s=\operatorname{Fuse}\left(
h_{text},h_{structured},h_{state}
\right)
\]

- `h_text`: frozen Qwen이 담당하는 자연어 query, goal, tool description의 의미
- `h_structured`: assembly, sequence accession, coordinate convention, REF/ALT, transcript/database version
- `h_state`: 현재 보유한 artifact, 완료된 작업, 실패·관측 결과

tool 쪽에는 별도의 contract를 둔다.

```yaml
operation: coordinate_remap
input_entity: genomic_variant
input_assembly: GRCh37
output_entity: genomic_variant
output_assembly: GRCh38
requires:
  - normalized_sequence_reference
  - position
  - ref
  - alt
possible_outcomes:
  - mapped_once
  - unmapped
  - multi_mapped
```

반대 방향의 tool은 input/output assembly를 정확히 뒤집는다. 두 방향이 의미적으로 비슷하더라도 현재 state에서의 applicability는 다르다. 또한 remapping은 실패하거나 다중 mapping을 만들 수 있으므로 두 방향을 완전한 역함수로 가정해서는 안 된다.

### 14.3 Variant identity를 text string으로만 저장하지 않는다

`chr1:100:A:G` 같은 문자열은 충분한 variant key가 아니다. 최소한 다음 provenance를 보존한다.

```text
variant_key = (
    species,
    assembly_accession_and_patch,
    sequence_accession.version,
    coordinate_convention,
    position_or_interval,
    ref,
    alt,
    strand_if_relevant,
    transcript_accession.version,
    normalization_method_and_version,
)
```

[NCBI SPDI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7523648/)는 variant를 reference sequence accession/version, position, deletion, insertion으로 명시한다. [GA4GH VRS SequenceLocation](https://vrs.ga4gh.org/en/latest/concepts/LocationAndReference/SequenceLocation.html)도 coordinate가 특정 `sequenceReference` 위에 정의됨을 데이터 모델에 포함한다. 이처럼 identity와 coordinate provenance는 dense embedding이 아니라 구조화된 필드로 보존해야 한다.

### 14.4 권장 cache 단위

전체 JSON이나 state를 한 문장으로 합쳐 하나의 pooled vector만 저장하지 않는다. 다음을 함께 보존한다.

```text
cache_record
├── raw canonical text
├── query_embedding
├── goal_embedding
├── tool_name_embedding
├── tool_description_embedding
├── input_schema_embedding
├── output_schema_embedding
├── precondition_embedding
├── effect_embedding
├── structured_state
├── tool_contract
├── canonical entity IDs
└── provenance
    ├── encoder revision
    ├── tokenizer revision
    ├── prompt/serialization version
    ├── pooling rule
    ├── schema version
    └── database/annotation release
```

이렇게 field별로 cache하면 `GRCh37↔GRCh38` counterfactual에서 structured field만 교체할 수 있고, 어떤 field가 판단에 사용되었는지도 분석할 수 있다.

### 14.5 Cache 이후에도 할 수 있는 것과 재-cache가 필요한 것

| 변경 | 기존 pooled cache만으로 가능? | 설명 |
|---|---:|---|
| cosine, whitening, PCA | 가능 | vector 후처리 |
| residual adapter, projection head | 가능 | cache 위에서 학습 |
| Mahalanobis, bilinear, prototype | 가능 | cache 위에서 학습 |
| ranking·contrastive loss | 조건부 가능 | label과 candidate metadata가 남아 있어야 함 |
| structured contract reranking | 조건부 가능 | raw structured field가 남아 있어야 함 |
| field별 Qwen embedding | 불가능 | field를 나누어 재-encoding 필요 |
| domain definition을 text에 삽입 | 불가능 | 입력 text가 바뀌므로 재-cache 필요 |
| pooling 또는 hidden layer 변경 | 불가능 | token/layer output을 저장하지 않았다면 재-cache 필요 |
| Qwen 내부 LoRA | 불가능 | encoder output 자체가 바뀌므로 전체 재-cache 필요 |
| biomedical encoder 추가 | 별도 cache 필요 | Qwen cache는 그대로 보조 view로 재사용 가능 |

즉 첫 cache를 만들 때 raw text와 structured metadata를 버리지 않는 것이 가장 중요하다.

### 14.6 추천 scorer

\[
\operatorname{score}(s,t)=
q_s^\top W e_t
+h_{struct}(s)^\top Uc_t
+\operatorname{MLP}\!\left(\phi(s,t)\right)
+\operatorname{mask}_{contract}(s,t)
\]

- `q_s`, `e_t`: cached Qwen semantic vectors
- `h_struct(s)`: exact typed state representation
- `c_t`: tool input/output/precondition/effect contract
- `φ(s,t)`: assembly match, missing field, goal-output overlap, remap 필요 여부
- `mask_contract`: 결정론적으로 잘못된 action에 `-∞`

배포 환경에서는 build mismatch처럼 명백한 오류를 학습된 score에만 맡기지 않는다. 연구에서는 hard mask on/off를 ablation하여 “metric이 스스로 배운 부분”과 “symbolic guardrail의 효과”를 분리한다.

### 14.7 GRCh37과 GRCh38을 어떻게 학습시킬 것인가

두 단어 자체를 static contrastive negative로 두어 전역적으로 멀리 밀지 않는다. 대신 `(state, goal, action)` minimal pair를 만든다.

```text
State: artifact=Variant<GRCh37>
Goal:  Annotation<GRCh38>

Positive: remap_37_to_38
Negative: remap_38_to_37
Invalid:  annotate_GRCh38_without_remap
```

remap을 실행한 다음 state에서는 label이 달라진다.

```text
State: artifact=Variant<GRCh38>
Goal:  Annotation<GRCh38>

Positive: annotate_GRCh38
Invalid:  remap_37_to_38
```

이렇게 해야 모델은 “37과 38이 다른 단어”가 아니라 “state와 goal에 따라 올바른 방향의 transition이 달라진다”는 것을 배운다.

### 14.8 데이터셋에 넣어야 할 counterfactual

- 동일 task에서 assembly만 GRCh37↔GRCh38로 바꾼 pair
- 숫자 coordinate가 같지만 assembly가 다른 near-miss
- 실제로 검증된 올바른 remap pair
- transcript accession은 같고 version만 다른 pair
- 0-based half-open과 1-based coordinate confusion
- chromosome accession과 `chr1`/`1` alias normalization
- REF/ALT reversal 또는 strand-related 오류
- annotation/database release가 다른 case
- remap 실패와 one-to-many mapping
- assembly가 누락된 경우의 `ASK` 또는 `ABSTAIN`

실제 biological coordinate pair는 임의로 만들지 않고 NCBI Remap 같은 검증된 mapping으로 생성한다. 동일 숫자를 의도적으로 쓰는 case는 “같은 숫자라도 reference namespace가 다르면 동일 locus로 간주할 수 없다”는 오류 검사용 negative로 표시한다.

### 14.9 핵심 지표

- assembly-confusion rate
- remap-omission rate
- wrong-direction remap rate
- invalid-tool rate
- build-swap action-flip accuracy
- structured provenance retention
- remap 후 successor-state correctness
- final task success와 cumulative regret

split은 동일 variant가 train과 test에 흩어지지 않도록 variant/gene/tool-family 단위로 묶는다. 특히 assembly만 바꾼 counterfactual pair는 같은 split에 유지하여 정보 leakage를 막는다.

### 14.10 Qwen에 genomics background를 더 넣어야 하는가

tool card에 짧은 정의와 typed contract를 제공하는 것은 유용하다.

```text
GRCh37 and GRCh38 are distinct human reference assemblies.
Coordinates are valid only within their declared sequence reference.
Cross-assembly use requires an explicit remapping operation.
```

이 text를 넣으면 Qwen-only enhanced baseline을 만들 수 있다. 하지만 이는 guardrail이 아니며, exact compatibility는 여전히 structured contract가 책임져야 한다.

biomedical encoder는 첫 해결책이 아니다. 예를 들어 SapBERT는 biomedical synonym/entity alignment에, MedCPT는 biomedical retrieval에 유용할 수 있지만 genome-build 방향성과 tool effect를 자동으로 보장하지 않는다. 필요하다면 Qwen을 교체하기보다 auxiliary view로 late fusion한다.

\[
k_t=P_Qe_t^{Qwen}
+\lambda P_Be_t^{bio}
+P_K\kappa_t
\]

오류 분석에서 실제 문제가 biomedical synonym이나 entity linking으로 확인될 때만 이 실험을 추가하는 것이 좋다.

### 14.11 Cache의 또 다른 한계: 새로운 입력의 inference

훈련 데이터의 embedding을 미리 cache하면 작은 head를 싸게 반복 학습할 수 있다. 그러나 새로운 환자 query나 새로운 자유문장 observation이 들어오면 그것을 Qwen embedding으로 바꾸기 위해 inference 시점에도 Qwen이 필요하다.

Qwen 없이 작은 모델만 배포하려면 별도의 단계가 필요하다.

1. 작은 student text encoder를 Qwen embedding과 task label로 distillation
2. state를 제한된 typed slot으로 구성하여 cached atomic embedding을 조합
3. 더 작은 sentence/biomedical encoder를 online encoder로 사용

따라서 **offline cache는 연구 비용을 줄이는 방법이지, 그 자체가 작은 배포 모델을 완성하는 방법은 아니다.** 첫 프로젝트에서는 Qwen을 offline teacher로 두고 geometry head를 연구한 뒤, 결과가 확인되면 student distillation을 별도 실험으로 추가하는 것이 가장 깔끔하다.

### 14.12 이 문제를 반영한 최소 비교군

1. 전체 state를 text로 직렬화한 Qwen-only
2. Qwen + assembly categorical embedding
3. Qwen + full typed state 및 soft contract
4. 3 + hard validity mask
5. 3/4 + post-cache low-rank state-conditioned adapter
6. 3/4 + successor-effect head
7. 선택 사항: biomedical encoder late fusion

예상 해석:

- 2가 개선: mean pooling이 exact build identity를 약화함
- 3이 개선: semantic geometry보다 type/contract가 중요함
- 4만 크게 개선: 안전성은 metric learning보다 deterministic validation 문제임
- 5가 개선: state-dependent geometry가 필요함
- 6이 long-horizon 성공률을 개선: successor-state trajectory 가설을 지지함

### 14.13 Genomics background를 가진 다른 모델을 함께 쓸 수 있는가

가능하다. 다만 “genomics model”은 입력 종류에 따라 역할이 완전히 다르다.

| 임베딩 대상 | 적합한 모델·방법 | Agent 연구에서의 역할 |
|---|---|---|
| 자연어 biomedical query·논문·공식 문서 | [MedCPT](https://github.com/ncbi/MedCPT) | query–background/tool document retrieval |
| 질병·약물·표현형 등 biomedical entity mention | [SapBERT](https://aclanthology.org/2021.naacl-main.334/) | UMLS 기반 synonym 정렬과 entity linking |
| 일반 biomedical text feature | [PubMedBERT](https://arxiv.org/abs/2007.15779) | 도메인 NLP encoder baseline |
| 실제 DNA sequence window | [Nucleotide Transformer](https://www.nature.com/articles/s41592-024-02523-z), [DNABERT-2](https://arxiv.org/abs/2306.15006) | sequence motif·variant-effect signal |
| 매우 긴 genomic sequence 생성·scoring | [Evo 2](https://www.nature.com/articles/s41586-026-10176-5) | 대규모 sequence task; 이 프로젝트의 초기 단계에는 과함 |
| protein sequence | [ESM](https://github.com/facebookresearch/esm) | protein sequence·structure/function view |
| assembly·coordinate·transcript version | typed field, VRS/SPDI, exact contract | dense model에 맡기지 않는 exact identity |

특히 DNABERT-2, Nucleotide Transformer, Evo 2는 `GRCh37`이라는 자연어 label이나 tool description을 이해하기 위해 학습된 모델이 아니다. 이들은 `ACGT...` 형태의 실제 염기서열을 입력받아 sequence pattern을 표현한다. 따라서 tool routing 문제에 바로 Qwen 대신 넣는 것은 입력–모델 역할이 맞지 않는다.

DNA model을 추가할 타이밍은 다음과 같다.

- action 선택이 실제 reference/alternate sequence context에 의존할 때
- variant-effect prediction 결과가 어느 downstream tool을 부를지 바꿀 때
- promoter, splice site, regulatory sequence 같은 local sequence feature가 필요한 경우

예를 들어 variant 주변 reference와 alternate sequence window를 각각 encode해 차이를 사용할 수 있다.

\[
h_{seq}=P_D\left(
E_{DNA}(x_{alt})-E_{DNA}(x_{ref})
\right)
\]

하지만 coordinate remapping과 API compatibility만 연구하는 첫 단계에는 실제 sequence model보다 structured contract가 더 직접적이고 해석 가능하다.

### 14.14 가장 추천하는 조합

첫 프로젝트에는 다음 네 층이면 충분하다.

```text
1. Frozen Qwen
   └── 일반 query·goal·tool 의미

2. Frozen MedCPT
   ├── Query Encoder: 현재 질문·state narrative
   └── Article Encoder: 공식 background card·tool documentation

3. SapBERT 또는 exact dictionary linker
   └── disease/phenotype/drug entity를 canonical ID 후보로 연결

4. Structured contract graph
   └── assembly, version, input/output type, direction, precondition/effect
```

MedCPT는 PubMed 검색 로그에서 얻은 2억 5,500만 query–article pair로 학습된 biomedical retrieval model이고 query encoder와 article encoder가 같은 768차원 retrieval space를 제공한다. 따라서 질문과 공식 문서 card를 연결하는 auxiliary encoder로 쓰기 좋다. SapBERT는 UMLS의 400만 개 이상 concept와 synonym을 정렬하도록 학습되어 biomedical entity linking에 적합하다. 그러나 어느 모델도 genome-build 호환성을 자동으로 보장하지 않으므로 exact structured channel은 남겨야 한다.

### 14.15 Background를 넣는 세 가지 방법

#### 방법 A — Early concatenation

```text
[TASK]
Annotate this variant ...

[ENTITY BACKGROUND]
GRCh37 and GRCh38 are distinct coordinate namespaces ...

[TOOL CONTRACT]
This endpoint accepts GRCh38 only ...
```

전체를 Qwen에 한 번 넣어 cache한다.

장점:

- 구현이 가장 쉬움
- frozen Qwen 내부 attention이 문맥을 통합
- background가 없는 Qwen-only baseline과 바로 비교 가능

단점:

- 문서 한 줄이 바뀌어도 전체 embedding 재계산
- 긴 background가 assembly 방향 같은 짧은 정보를 다시 희석할 수 있음
- 출처별 기여도를 추적하기 어려움
- 문서 순서와 token budget에 민감

따라서 중요한 baseline이지만 주 architecture로는 권하지 않는다.

#### 방법 B — Background card separate encoding + late fusion

긴 문서를 atomic claim card로 나누어 각각 encode한다.

```yaml
card_id: assembly_grch37_to_grch38
subject_id: GRCh37
relation: remap_target
object_id: GRCh38
scope: human_genome_coordinate
claim: Coordinates on GRCh37 require explicit remapping before use in a GRCh38-only resource.
source_url: https://...
source_release: ...
retrieved_at: ...
content_hash: ...
parser_version: ...
```

현재 state가 필요한 card만 작은 attention으로 결합한다.

\[
\alpha_i=
\operatorname{softmax}_i\left(
(W_q q_h)^\top W_b b_i
\right)
\]

\[
h_{background}=\sum_i\alpha_i b_i
\]

\[
k_t=P_Qe_t^{Qwen}
+P_Me_t^{MedCPT}
+P_Bh_{background}
+P_K\kappa_t
\]

각 encoder 차원과 norm이 다르므로 바로 concatenate하지 말고, 각각 LayerNorm/L2 normalization과 별도 low-rank projection을 거쳐 동일 차원으로 맞춘다. state-conditioned gate 또는 attention이 각 view의 비중을 정하도록 한다.

이 방식의 장점:

- tool, contract, background를 독립적으로 갱신 가능
- 같은 tool도 state에 따라 다른 card를 읽음
- attention weight와 card ID로 provenance 확인 가능
- 문서가 추가되어도 Qwen tool embedding을 다시 계산할 필요가 없음
- cached vector 위의 작은 head만 학습 가능

#### 방법 C — Domain-adapted Qwen을 만든 뒤 다시 freeze

Qwen에 genomics corpus로 continued pretraining 또는 LoRA를 먼저 수행하고, 그 weight를 고정한 뒤 전체 dataset을 다시 cache할 수도 있다.

```text
Qwen base
  → genomics adaptation
  → adapted checkpoint freeze
  → dataset embedding cache
  → small trajectory head training
```

이는 “frozen model” 원칙과 모순되지 않는다. trajectory head를 학습할 때 backbone이 frozen이면 되기 때문이다. 그러나 다음 이유로 첫 실험보다는 upper-bound 실험에 적합하다.

- adaptation이 바뀔 때 전체 cache를 다시 계산해야 함
- 학습 corpus와 test task 사이 answer leakage 점검 필요
- exact coordinate 논리는 여전히 보장되지 않음
- 성능 향상이 background 지식인지 representation 변화인지 분리하기 어려움
- 계산량과 연구 범위가 크게 증가

### 14.16 Authoritative background의 범위와 leakage

background knowledge를 두 종류로 구분해야 한다.

#### Planner에 미리 제공해도 되는 operational knowledge

- tool이 받는 identifier 형식
- 지원 reference assembly
- input/output schema
- coordinate convention
- precondition과 가능한 effect
- database scope와 release
- 알려진 failure mode

#### 미리 넣으면 answer leakage가 되는 task/world knowledge

- 특정 variant의 정답 ClinVar classification
- 특정 환자의 검사 결과
- benchmark가 질문하는 gene–disease association의 정답
- 아직 호출하지 않은 tool이 반환해야 할 evidence
- expert trajectory의 다음 action 또는 미래 observation

후자는 tool card에 넣지 않고 실제 tool 실행이나 query-time evidence retrieval을 통해서만 얻도록 한다. 그렇지 않으면 agent planning이 아니라 정답이 들어 있는 문장을 찾는 실험이 된다.

모든 card에는 최소한 다음 provenance를 기록한다.

```text
source URL
source/database release
retrieved_at
content hash
parser version
validity interval if available
```

temporal generalization을 평가할 때는 test 시점보다 미래의 card가 training cache에 들어가지 않도록 corpus time cutoff를 둔다.

### 14.17 Static background와 trajectory-time dynamic retrieval

모든 배경지식을 첫 query에 한 번 붙이고 끝내는 것보다, execution state가 바뀔 때 관련 card를 다시 선택하는 것이 trajectory 관점에 더 맞다.

| 조건 | 지식 사용 방식 |
|---|---|
| No-background | tool description만 사용 |
| Initial-background | 첫 state에서 선택한 card를 끝까지 고정 |
| Dynamic-background | observation 이후 매 phase card를 다시 선택 |
| Oracle-background | 현재 state에 필요한 gold-relevant card 제공 |

예를 들어 GRCh37→GRCh38 remap이 완료되기 전에는 remapping contract가 중요하다. 완료된 후에는 해당 card의 중요도가 낮아지고 GRCh38 annotation endpoint의 card가 활성화되어야 한다.

\[
K_h=\operatorname{Retrieve}(s_h,G),
\qquad
M_h=M(s_h,G,K_h)
\]

즉 background는 tool embedding에 한 번 붙이는 정적 설명이 아니라, 각 trajectory state에서 가능한 functional direction을 갱신하는 local field로 볼 수 있다.

### 14.18 추천 비교 실험

| ID | 구성 | 확인하는 것 |
|---|---|---|
| D0 | Qwen text only | 일반 encoder의 기준 성능 |
| D1 | Qwen + early-concat background | 배경지식 자체가 부족했는가 |
| D2 | Qwen + separate Qwen card attention | 정보 희석·구성 방식의 효과 |
| D3 | Qwen + MedCPT background view | biomedical retrieval geometry의 기여 |
| D4 | D3 + SapBERT/entity linking | synonym·entity identity의 기여 |
| D5 | D4 + structured contract/KG | exact compatibility와 reachability의 기여 |
| D6 | D5 + dynamic state-wise retrieval | trajectory에 따라 지식을 갱신하는 효과 |
| D7 | 선택: D6 + DNA encoder | 실제 sequence signal의 추가 가치 |
| D8 | 선택: domain-adapted frozen Qwen | adaptation upper bound |

공정한 비교를 위해 동일 card corpus, 동일 retrieval candidate, 동일 token budget과 비슷한 trainable fusion-head parameter budget을 사용한다.

추가 평가:

- relevant-card Recall@k
- entity-link accuracy와 unresolved rate
- background distractor robustness
- stale-release card robustness
- background 출처를 바꿨을 때 action stability
- Initial-background 대비 Dynamic-background의 final success와 recovery latency
- assembly/direction field 제거 시 성능 하락
- oracle retrieval과 learned retrieval 사이의 격차

가장 현실적인 첫 결론 후보는 다음이다.

> Qwen을 genomics model로 완전히 교체할 필요는 없다. 일반 의미, biomedical retrieval, exact structured identity, sequence biology를 서로 다른 frozen view로 분리하고 작은 state-conditioned head가 현재 trajectory에 필요한 view를 선택하게 하는 편이 더 안전하고 연구적으로도 해석 가능하다.

---

## 15. 추천 ablation

최소 ablation:

1. state 제거
2. tool contract 제거
3. hard applicability mask 제거
4. goal embedding 제거
5. one-hot CE ↔ functional-regret target
6. random negative ↔ semantic hard negative
7. rank `8/32/64/128`
8. projection 후 normalization 유무
9. tool name만 ↔ description ↔ schema ↔ structured contract
10. seen tool ↔ unseen tool family

Successor-effect 모델의 추가 ablation:

- 실제 successor state supervision 제거
- predicted effect 대신 tool text embedding 직접 사용
- state delta가 아니라 absolute next state 예측
- 비용·위험 penalty 제거
- one-step 성능과 long-horizon 성공률을 별도로 비교

---

## 16. 이 프로젝트에서 가장 좋은 논문형 주장

약한 질문:

> Euclidean, cosine, Mahalanobis 중 무엇이 더 좋은가?

더 좋은 질문:

> Frozen LLM semantic space의 어떤 왜곡을 작은 metric head가 교정할 수 있는가?

가장 강한 질문:

> Agent의 functional geometry는 정적인 tool-description distance로 충분한가, 아니면 현재 state와 예상 successor effect에 의해 동적으로 정의되어야 하는가?

이를 다음처럼 하나의 이야기로 묶을 수 있다.

```text
Frozen semantic representation
    ↓
단순 norm/anisotropy 보정으로 충분한가?
    ↓ 아니면
global learned metric이 필요한가?
    ↓ 그래도 부족하면
state-conditioned functional metric이 필요한가?
    ↓ 최종적으로
tool identity가 아닌 successor effect를 비교해야 하는가?
```

이 흐름은 사용자가 기존 연구에서 중요하게 본 “점 자체보다 경로와 변화의 구조를 이해해야 한다”는 관점을 Agent 연구로 자연스럽게 확장한다.

---

## 17. 최종 추천

### 바로 시작할 MVP

- backbone: Qwen2.5-7B-Instruct, revision 고정, 완전 frozen
- embedding: 한 번 계산해 disk cache
- task: 01_Medical/Genomics symbolic simulator의 next-tool selection
- baseline: raw L2, cosine, whitened cosine
- learned head: rank-32 Mahalanobis, state-conditioned bilinear
- main model: contract-aware successor-effect energy
- label: exact tool one-hot과 functional-regret soft target 모두 비교
- evaluation: step accuracy보다 final success, cumulative regret, recovery를 우선

### 예상되는 연구 기여

1. frozen LLM embedding의 semantic geometry와 agent functional geometry 사이의 차이를 측정한다.
2. 그 차이가 norm, anisotropy, modality mismatch, state dependence 중 어디서 오는지 분해한다.
3. 작은 모듈만으로 unseen tool과 counterfactual state에 일반화 가능한지 검증한다.
4. action을 tool description이 아니라 예상 state transition으로 비교하는 trajectory-centric decoder를 제안한다.

가장 중요한 것은 “여러 거리를 돌려봤다”가 아니라, **각 모듈이 어떤 geometry 가정을 완화하며 그것이 long-horizon trajectory에 어떤 영향을 주는지** 보여주는 것이다.

---

## 참고 자료

- [FlowAgent 논문, arXiv v2](https://arxiv.org/html/2605.07339v2)
- [FlowPlan 공식 repository](https://github.com/ssy166/FlowPlan)
- [공개 snapshot 및 checkpoint 제외 설명](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/README.md#L25)
- [Baseline reproducibility와 제외된 artifact](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/docs/BASELINE_REPRODUCIBILITY.md#L5)
- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [LMNN: Distance Metric Learning for Large Margin Nearest Neighbor Classification](https://jmlr.org/papers/v10/weinberger09a.html)
- [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

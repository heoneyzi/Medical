> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# GeoFlowAgent 재설계안

## Search-Distilled Goal-Conditioned State Flow

> 핵심 질문: **프로즌 모델 위의 작은 모듈만 학습하여, 유전체 도구 실행 공간에서 “목표까지 남은 실행 비용”을 나타내는 임베딩 기하를 만들 수 있는가? 그리고 그 기하 위에서 하나의 정답 순서가 아니라 여러 유효 경로의 분포를 생성할 수 있는가?**

작성 기준일: 2026-09-15

---

## 1. 결론부터

현재의 7개 train trajectory를 그대로 늘려 학습하는 방향은 중단하는 것이 맞다. 그것들은 **파이프라인 smoke test**로는 쓸 수 있지만, 연구 모델을 학습시키는 데이터는 아니다.

가장 타당한 새 방향은 다음 세 단계다.

1. **Search-distilled goal-conditioned geometry**
   - MARRVEL 밖의 실행 가능한 snapshot 환경에서 상태 그래프를 만든다.
   - 탐색 알고리즘이 각 상태의 `V*`, `Q*`, action regret, 모든 최적 action을 자동 계산한다.
   - 프로즌 encoder 위의 작은 모듈은 단순한 의미적 유사도가 아니라 **목표까지 남은 최소 실행 비용과 도달 가능성**을 학습한다.
2. **Search-DAgger / Expert Iteration**
   - 학습된 agent가 실제로 방문하는 잘못된 상태를 다시 search oracle로 라벨링한다.
   - 정답 trajectory 바깥에서 회복하는 법을 반복적으로 학습한다.
3. **State Flow 확장**
   - 기존 FlowAgent처럼 tool prototype의 나열을 직접 생성하지 않는다.
   - search가 만든 여러 좋은 경로를 이용해 **미래 상태 또는 상태 변화량의 연속 경로**를 생성한다.
   - 매 단계에는 첫 action만 실행하고, 실제 관측을 받은 뒤 다시 계획한다.

그리고 **MARRVEL 100문항은 전부 최종 external test-only로 잠그는 것이 맞다.** 다만 MARRVEL만으로는 장기 경로 연구를 충분히 평가할 수 없으므로, 별도의 unseen multi-step internal test도 반드시 필요하다.

이 연구를 한 문장으로 부르면 다음이 가장 정확하다.

> **Counterfactual Search-Distilled State Flow for Frozen Genomic Agents**

---

## 2. 현재 실험이 알려 준 것

현재 결과는 실패라기보다 문제 설정을 교정해 준 진단 실험이다.

### 2.1 확인된 사실

- 실제 독립 task는 synthetic 12개이며 train/dev/test는 7/2/3개였다.
- 103개의 prefix는 새로운 독립 사례 103개가 아니라 동일한 소수 trajectory를 잘라 만든 상관된 표본이다.
- raw embedding 자체의 tool 선택력은 약했다.
- metric module은 학습 표본에서는 좋아졌지만, 작은 데이터에 비해 매우 컸고 일반화 증거는 없었다.
- contract-valid action과 zero-regret action이 사실상 같아서, contract mask가 문제를 거의 풀어 주는 oracle처럼 작동했다.
- Flow suffix exact match는 낮았고, 특히 STOP 예측이 주요 실패 요인이었다.
- geometry가 좋아 보여도 그 위의 Flow 성능이 함께 좋아지지 않았다. 즉 현재의 geometry objective와 Flow target이 정렬되어 있지 않았다.

### 2.2 가장 중요한 해석

문제는 단순히 표본 수가 작다는 것만이 아니다.

현재 학습 목표는 사실상 다음을 배우고 있었다.

> “이 state에서 expert가 기록한 다음 tool은 무엇인가?”

하지만 우리가 정말 배우고 싶은 것은 다음이다.

> “이 state와 goal에서 어떤 action이 실제로 목표에 가까워지게 하며, 얼마나 가까워지게 하고, 다른 유효 경로는 무엇인가?”

따라서 정답 sequence를 더 많이 손으로 만드는 것보다, **실행 환경과 verifier를 이용해 경로의 구조를 자동 라벨링하는 방식**으로 문제를 다시 정의해야 한다.

---

## 3. 연구의 중심 관점: 의미적 거리에서 실행적 거리로

사용자가 중요하게 본 `trajectory` 관점은 이 프로젝트의 가장 좋은 연구 축이 될 수 있다. 다만 여기서 trajectory는 단순히 latent point들을 부드럽게 잇는 곡선이 아니다.

이 프로젝트에서 좋은 공간은 다음 성질을 가져야 한다.

1. 목표에 도달하기 위해 필요한 단계 수 또는 비용을 표현한다.
2. 어떤 action을 실행하면 그 비용이 얼마나 줄어드는지 표현한다.
3. 겉으로 비슷한 문자열이더라도 실행 전제가 다르면 구분한다.
4. 서로 다른 순서지만 같은 결과를 내는 복수 경로를 가까운 효용으로 표현한다.
5. 실패, empty result, assembly mismatch와 같은 상태에서 회복 방향을 표현한다.

예를 들어 `GRCh37`과 `GRCh38`이 언어 임베딩에서 가깝더라도 문제는 없다. 정확한 assembly는 typed state에 별도로 남아 있고, 실행 그래프에서는 다음과 같이 다른 위치를 갖는다.

```text
state: variant@GRCh37
goal:  gnomAD evidence@GRCh38

variant@GRCh37
    └── liftover
          └── normalize/verify reference
                └── gnomAD query
                      └── goal satisfied
```

`variant@GRCh37`과 `variant@GRCh38`의 텍스트 임베딩이 가까워도, search가 계산한 cost-to-go와 exact precondition은 서로 다르다. 학습 모델은 문자열 차이를 억지로 벌리는 것이 아니라 **liftover를 거치면 goal potential이 감소한다**는 사실을 배운다.

### 3.1 왜 유클리디안 거리만으로는 부족한가

유전체 tool workflow는 대체로 방향성이 있다.

- raw identifier에서 normalized identifier로 이동할 수 있어도 그 역방향은 자연스럽지 않을 수 있다.
- evidence를 획득한 상태는 이전의 무지한 상태보다 정보가 많다.
- tool failure나 rate limit 이후에는 같은 구조화 state라도 retry budget과 최근 observation에 따라 미래가 달라진다.

유클리디안 거리는 대칭이다.

\[
d(s,g)=d(g,s)
\]

하지만 실제 실행 비용은 일반적으로 대칭이 아니다.

\[
d_{exec}(s\rightarrow g)\neq d_{exec}(g\rightarrow s)
\]

따라서 주 모델은 굳이 수학적 metric일 필요가 없다. **방향성 있는 goal-conditioned energy 또는 quasimetric**이 더 자연스럽다.

권장 비교군은 다음과 같다.

| 표현 | 역할 | 기대 | 한계 |
|---|---|---|---|
| Cosine/Euclidean | 가장 단순한 baseline | 구현과 해석이 쉬움 | 방향성과 계층적 graph를 표현하기 어려움 |
| Low-rank Mahalanobis | 현재 코드의 강화 baseline | task-specific 축을 학습 | 여전히 기본적으로 대칭적 |
| Hyperbolic/Poincaré | 계층적 분기를 표현하는 ablation | tree-like hierarchy에 유리할 수 있음 | 실제 workflow는 merge와 cycle이 있는 DAG라서 필연적 우위는 없음 |
| Asymmetric bilinear energy | 권장 주 모델 | 방향성과 state-action-goal 관계 표현 | 엄밀한 metric은 아니므로 calibration 평가 필요 |
| Goal-conditioned Q/Energy | 최종 권장 | 거리를 실제 remaining cost에 직접 연결 | 정확한 search/verifier supervision 필요 |

핵심은 “어느 거리 함수가 예쁜가?”가 아니라 <b>학습된 energy가 실제 도달 비용과 일치하는가?</b>이다.

---

## 4. 사람 trajectory 없이 학습 데이터를 만드는 방법

사람이 작성한 expert trajectory를 대규모로 모으지 않아도 된다. 그러나 학습 신호 자체가 없어도 된다는 뜻은 아니다. 이 프로젝트에서는 expert를 다음 세 가지로 대체한다.

- exact snapshot executor
- goal verifier
- graph search oracle

### 4.1 MARRVEL 밖에서 task를 역생성한다

가장 좋은 방법은 public genomic record에서 **정답을 숨겨 query를 만드는 inverse task generation**이다.

예시:

1. test entity와 겹치지 않는 ClinVar/Ensembl/HGNC 등의 record를 pinned snapshot에서 고른다.
2. record의 일부 필드를 초기 관측으로 공개한다.
3. 다른 필드를 private target으로 숨긴다.
4. tool schema와 dependency graph를 이용해 goal을 회수할 수 있는지 검사한다.
5. 실제 tool replay로 성공한 task만 저장한다.

```text
원본 record
  variant + transcript + assembly + gene + annotation + evidence

공개 initial state
  variant + transcript + assembly

비공개 verifier target
  gene + selected evidence fields

생성된 task
  "이 transcript variant와 관련된 gene 및 지정 evidence를 찾아라"
```

LLM은 query 문장의 표현을 다양화하는 용도로만 사용할 수 있다. entity, target, tool argument, 정답은 LLM이 생성하지 않고 실제 snapshot과 schema가 결정해야 한다.

### 4.2 권장 workflow family

서로 다른 entity만 바꾼 동일 template를 수천 개 만드는 것은 충분하지 않다. 최소한 다음과 같은 서로 다른 workflow family가 필요하다.

1. gene/identifier resolution
2. transcript variant normalization
3. GRCh37/GRCh38 liftover와 reference verification
4. ClinVar/gnomAD/dbNSFP evidence retrieval
5. phenotype-to-gene 또는 disease-to-gene linking
6. ortholog/expression/protein-function evidence
7. literature identifier conversion과 evidence retrieval
8. 여러 source를 합치는 evidence dossier
9. no-result/ambiguous alias/build mismatch가 포함된 recovery task
10. 일부 tool 또는 source가 unavailable한 replanning task

### 4.3 future-goal relabeling

한 번의 유효 rollout이 다음 상태열을 만들었다고 하자.

\[
s_0,a_0,s_1,a_1,\ldots,s_T
\]

각 미래 상태 `s_j`의 의미 있는 공개 가능한 일부를 goal `g_j`로 투영하면, 모든 `i < j`에 대해 다음 학습 쌍을 만들 수 있다.

\[
(s_i,g_j,d=j-i)
\]

즉 하나의 길이 `T` trajectory에서 최대 `O(T^2)`개의 reachability/value pair를 얻는다. 실패 rollout도 실제로 도달한 의미 있는 상태를 goal로 다시 붙여 representation warm-start에 사용할 수 있다.

단, 이것은 독립 임상 사례 수를 늘리는 기법이 아니다. 따라서 entity family와 workflow template 자체의 다양성은 별도로 확보해야 한다.

### 4.4 counterfactual branch를 실제로 실행한다

현재처럼 관측된 gold action만 positive이고 나머지를 모두 negative로 두면 안 된다.

각 state에서 action을 다음 세 종류로 구분한다.

- `known executable`: snapshot이 있고 실제 successor를 확인한 action
- `known invalid/failure`: contract 또는 실행 결과로 실패를 확인한 action
- `unknown`: 실행 snapshot이 없어 결과를 모르는 action

`unknown`은 negative가 아니다. **selection softmax denominator와 regret loss 모두에서 mask**해야 한다.

학습 가치가 높은 상태는 다음과 같다.

- contract-valid action이 둘 이상이다.
- 그 action들의 downstream cost/regret가 서로 다르다.
- 하나는 당장 그럴듯하지만 장기적으로 dead end에 이른다.
- 대체 경로, source failure, retry 여부가 결과를 바꾼다.

이 조건이 없으면 contract가 다시 정답 oracle이 된다.

---

## 5. Search oracle이 만드는 새로운 supervision

현재 `ContractEngine`에는 이미 exact snapshot edge만 이용하는 `shortest_distance`와 shortest-path action/regret 계산이 있다. 따라서 처음부터 RL로 갈 이유가 없다.

### 5.1 unweighted graph에서 weighted graph로

현재 BFS의 step count를 다음과 같은 edge cost로 확장한다.

\[
c(s,a,s') = c_{call} + \lambda_{latency}c_{latency}
            + \lambda_{failure}c_{failure}
            + \lambda_{redundancy}c_{redundancy}
\]

그러면 BFS 대신 Dijkstra 또는 작은 graph에서는 exhaustive dynamic programming을 쓸 수 있다.

가능하다면 목적을 임의의 단일 가중합보다 다음 lexicographic 순서로 정의하는 것이 낫다.

1. verifier를 만족할 수 있는가
2. 요구된 evidence/source quality를 만족하는가
3. failure/redundancy가 적은가
4. call 수, latency, 금전 비용이 작은가

모든 cost 규칙은 MARRVEL을 보기 전에 non-MARRVEL dev에서 고정해야 한다.

### 5.2 각 상태에 붙는 라벨

goal `g`가 있을 때 search oracle은 다음을 계산한다.

\[
V^*(s,g)=\min_{\tau:s\rightarrow g} C(\tau)
\]

\[
Q^*(s,a,g)=c(s,a,s')+V^*(s',g)
\]

\[
A^*(s,a,g)=Q^*(s,a,g)-V^*(s,g)
\]

이로부터 한 개의 gold action이 아니라 다음을 저장한다.

- 모든 optimal 또는 epsilon-optimal next action
- 모든 known action의 cost-to-go
- action regret
- successor state
- top-K shortest 또는 Pareto-valid suffix
- goal reachability

여러 좋은 action에 대한 soft target도 만들 수 있다.

\[
p^*(a\mid s,g) \propto \exp(-Q^*(s,a,g)/\tau)
\]

이것이 단일 expert suffix보다 훨씬 풍부하고 안정적인 supervision이다.

### 5.3 Markov state를 제대로 정의한다

현재 canonical state만 graph key로 사용하지만, 다음 항목이 미래 실행 가능성을 바꾼다면 state에 포함해야 한다.

- 마지막 observation의 status
- retry 횟수와 남은 budget
- 현재 사용 가능한 source/tool
- assembly, transcript, reference version
- 이미 획득한 evidence와 provenance
- relevant failure history

즉 graph node는 단순 record가 아니라 다음이어야 한다.

```text
(typed knowledge state,
 last relevant observation,
 source availability,
 retry/call budget,
 compact history summary)
```

그렇지 않으면 겉으로 동일한 state에 서로 모순된 Q label이 붙는다.

---

## 6. 권장 모델: Goal-Conditioned Directed Geometry

### 6.1 입력 표현

프로즌 upstream model은 유지한다.

- Qwen: query, history, tool description의 일반 의미
- MedCPT: biomedical query/evidence 의미
- SapBERT: gene, disease, phenotype entity 표현
- DNABERT2: 실제 sequence가 존재할 때만 보조 view

그러나 다음 exact field는 텍스트 임베딩에 맡기지 않는다.

- genome assembly
- chromosome/position/ref/alt
- transcript ID와 version
- HGVS namespace
- source/database version
- resolved/unresolved flag
- available tool과 known snapshot mask

이들은 typed categorical/numeric feature로 별도 입력한다.

### 6.2 모델 구조

```text
frozen views(state, goal, tool)
          │
          ├── small view projectors
typed exact fields ── structured encoder
          │
          └── goal-conditioned fusion z(s, g)
                         │
             ┌───────────┴───────────┐
             │                       │
       Q/Energy head          transition head
       Q(s,a,g)               T(z(s,g), a) -> z(s',g)
             │                       │
             └───────────┬───────────┘
                         │
                  separate STOP/verifier head
```

현재처럼 수천만 parameter의 head부터 시작하지 않는다. 권장 MVP는 shared dimension 64–256, hidden dimension 128–256, 1–2 layer다. 모델 크기는 task-group learning curve를 보고 늘린다.

### 6.3 학습 loss

권장 objective는 다음 조합이다.

\[
\mathcal{L}=
\lambda_{policy}\mathcal{L}_{soft\ action}
+\lambda_Q\mathcal{L}_{Q^*}
+\lambda_{rank}\mathcal{L}_{regret}
+\lambda_{bellman}\mathcal{L}_{Bellman}
+\lambda_{trans}\mathcal{L}_{transition}
+\lambda_{contrast}\mathcal{L}_{reachability}
\]

구체적으로는 다음과 같다.

1. **soft/set-valued action loss**
   - 하나의 gold action이 아니라 optimal set 또는 `p*`를 학습한다.
2. **Q regression**
   - `Q(s,a,g)`가 search의 `Q*`를 예측하도록 Huber loss를 쓴다.
3. **regret ranking**
   - lower-regret action의 energy가 higher-regret action보다 낮아지게 한다.
4. **Bellman consistency**
   - `Q(s,a,g) ≈ c + V(s',g)`를 만족시킨다.
5. **transition prediction**
   - 선택한 action 뒤의 실제 successor latent를 예측한다.
6. **contrastive reachability**
   - 같은 trajectory의 reachable future state는 가깝게, 비슷한 entity이지만 build/goal이 틀리거나 unreachable한 state는 멀게 둔다.

Goal-conditioned contrastive learning이 action-labeled trajectory의 representation과 goal-conditioned value를 연결할 수 있다는 선행 근거가 있다. 이 프로젝트에서는 그것을 단독 RL 방법으로 쓰기보다 search-supervised value geometry의 보조 loss로 쓰는 편이 안정적이다.

### 6.4 첫 planner는 Flow가 아니라 greedy Q여야 한다

첫 번째 완성 agent는 다음처럼 단순해야 한다.

\[
a_t=\arg\min_{a\in A_{known/allowed}(s_t)} Q_\theta(s_t,a,g)
\]

이 baseline이 중요한 이유는 다음과 같다.

- embedding geometry 자체가 좋은지 직접 확인할 수 있다.
- Flow가 이득을 주는지 분리해 평가할 수 있다.
- geometry가 실패했는데 Flow가 그 위를 가리는 일을 막는다.

이 greedy value planner가 grouped unseen test에서 rule/structured baseline보다 좋아진 뒤에만 Flow를 붙인다.

### 6.5 거리 함수 비교를 별도의 연구 질문으로 둔다

거리 함수 sweep은 유지할 가치가 충분하다. 다만 모든 후보를 한 표에 넣고 accuracy가 가장 높은 것을 고르는 식으로 끝내면 연구적 해석이 약하다. 각 후보는 **서로 다른 workflow geometry 가설**을 나타내도록 설계해야 한다.

용어도 엄밀히 나눈다.

- **metric**: 비음수, 자기거리 0, 대칭성, triangle inequality를 만족한다.
- **quasimetric**: 대칭성을 요구하지 않는 directed distance다.
- **energy/compatibility**: bilinear/MLP처럼 거리 공리를 보장하지 않는 선택 점수다.

또한 state와 goal에 서로 다른 projector를 적용한 뒤 Euclidean을 계산한다면, 그것은 원래 input space의 엄밀한 Euclidean metric이라기보다 learned compatibility에 가깝다. 논문과 표에서 `metric`, `quasimetric`, `energy`를 한꺼번에 “distance”라고 부르지 않는다.

#### A. 기준이 되는 oracle graph distance

먼저 임베딩과 무관한 정답 기준을 둔다.

\[
D_G(s,g)=\text{weighted shortest-path cost from }s\text{ to }g
\]

또한 action을 포함한 기준은 다음이다.

\[
D_G((s,a),g)=c(s,a,s')+D_G(s',g)=Q^*(s,a,g)
\]

이것은 학습할 거리 함수가 아니라 다른 모든 거리/energy가 얼마나 실행 구조를 보존하는지 평가하는 기준이다.

#### B. 실제로 비교할 후보

| Family | 식 또는 개념 | 구조적 가정 | 이 프로젝트에서 묻는 질문 |
|---|---|---|---|
| Raw cosine | `1-cos(x,y)` | vector 방향이 의미를 결정 | frozen model의 의미 공간만으로 다음 행동을 찾을 수 있는가? |
| Squared Euclidean | `||x-y||²` | 평평하고 대칭적인 공간 | vector norm과 절대 위치까지 실행 정보를 갖는가? |
| Whitened Euclidean | train covariance로 whitening 후 Euclidean | 공통 고분산 방향은 nuisance | Qwen/MedCPT의 anisotropy를 제거하면 retrieval이 좋아지는가? |
| Diagonal Mahalanobis | `Σ w_i(x_i-y_i)²` | 축별 중요도가 다름 | 소수 의미/구조 축의 reweighting만으로 충분한가? |
| Low-rank Mahalanobis | `||L(x-y)||²` | 낮은 차원의 실행 subspace가 존재 | frozen 고차원 공간 안에 compact planning manifold가 있는가? |
| Bilinear energy | `-xᵀWy` | 두 항의 역할과 관계가 다름 | state→tool/goal의 방향성 compatibility가 중요한가? |
| Poincaré distance | hyperbolic geodesic | 지수적으로 분기하는 hierarchy | long-horizon/branching workflow가 계층 공간에서 더 잘 펴지는가? |
| Order-violation energy | `||ReLU(g-s)||²` 형태 | 정보 획득이 partial order를 이룸 | evidence가 누적되는 방향을 좌표별 포함 관계로 표현할 수 있는가? |
| Learned quasimetric | 비대칭, 비음수, triangle inequality | directed reachability가 핵심 | 실제 `s→g` 비용의 비대칭성을 geometry 자체가 보존하는가? |
| Unconstrained pair MLP | `MLP([x,g,x-g,x⊙g])` | geometry 가정 없음 | 구조적 distance inductive bias가 정말 필요한가? |

마지막 MLP는 강한 통제군이다. MLP가 모든 metric을 이기면 “특정 geometry가 맞았다”기보다 “충분히 유연한 compatibility function이 필요했다”는 결론에 가깝다.

#### C. 후보별 상세 해석

##### 1. Cosine

\[
D_{cos}(x,y)=1-\frac{x^Ty}{\|x\|\|y\|}
\]

장점:

- frozen language/biomedical encoder의 기본 의미 유사도를 가장 직접적으로 본다.
- scale에 둔감해서 안정적이다.
- 학습 parameter가 없어 raw baseline으로 명확하다.

예상 한계:

- `liftover_GRCh37_to_GRCh38`과 `query_gnomAD_GRCh38`의 의미적 유사성이 실행 순서를 보장하지 않는다.
- 동일 gene 이름 때문에 goal과 무관한 tool이 가깝게 나올 수 있다.
- 방향성과 remaining step 수를 표현하지 못한다.

해석:

- raw cosine만 좋다면 frozen encoder가 이미 tool semantics를 상당히 담은 것이다.
- contract mask를 켰을 때만 좋아지면 geometry가 아니라 contract가 문제를 푼 것이다.
- unit normalization된 vector에서는 `||x-y||² = 2(1-cos(x,y))`이므로 cosine과 Euclidean의 ranking이 같다. 이 조건의 두 결과를 서로 독립적인 geometry 발견처럼 해석하면 안 된다.

##### 2. Euclidean

\[
D_{EUC}(x,y)=\|x-y\|_2^2
\]

장점:

- FlowAgent의 anchor decoding과 가장 가까운 기준이다.
- vector norm이 progress/confidence 정보를 담는 경우 cosine보다 유리할 수 있다.

예상 한계:

- 고차원 frozen embedding의 anisotropy와 norm drift에 민감하다.
- 대칭적이어서 directed workflow를 직접 표현하지 못한다.

해석:

- Euclidean이 cosine보다 좋다면 norm이 실제 정보를 담는지, 아니면 text length/source에 의한 artifact인지 별도 회귀로 확인한다.
- embedding norm과 query 길이, history 길이, completion ratio의 상관을 함께 보고한다.

##### 3. Whitened Euclidean

train split의 평균 `μ`와 covariance에서만 whitening transform `W_train`을 추정한다.

\[
D_{white}(x,y)=\|W_{train}(x-\mu)-W_{train}(y-\mu)\|^2
\]

장점:

- 모든 문장에 공통으로 나타나는 지배적 방향을 줄인다.
- 학습된 neural head 없이 frozen space의 구조를 진단한다.

주의:

- dev/test를 포함해 covariance를 추정하면 leakage다.
- sample 수가 차원보다 작으면 shrinkage 또는 PCA rank 제한이 필요하다.

##### 4. Diagonal Mahalanobis

\[
D_{diag}(x,y)=\sum_i w_i(x_i-y_i)^2,\quad w_i\ge0
\]

장점:

- parameter가 적다.
- 어떤 projected dimension이 중요한지 상대적으로 해석하기 쉽다.

해석:

- diagonal만으로 충분하면 복잡한 curvature보다 축별 reweighting이 핵심이다.
- 학습된 weight가 seed마다 크게 바뀌면 feature 해석을 주장하지 않는다.

##### 5. Low-rank Mahalanobis

\[
D_{LRM}(x,y)=\|Lx-Ly\|^2,\quad L\in\mathbb{R}^{r\times d}
\]

장점:

- planning에 필요한 낮은 차원의 subspace를 찾는 가설과 잘 맞는다.
- rank `r` 자체가 intrinsic dimension에 대한 실험 축이 된다.

필수 실험:

- `r ∈ {8,16,32,64}`의 scaling curve
- effective singular value spectrum
- rank 증가가 train만 올리고 grouped dev를 떨어뜨리는지 확인

해석:

- 작은 rank에서 안정적으로 좋으면 frozen representation 안에 compact execution subspace가 있다는 근거다.
- 큰 rank에서만 train 성능이 오르면 과적합 가능성이 높다.

##### 6. Bilinear energy

\[
E_{bil}(s,a,g)=-q(s,g)^T W k(a)
\]

이것은 엄밀한 거리가 아니라 낮을수록 좋은 compatibility energy다. `W`가 비대칭이고 state/goal과 action의 projector가 다르므로 역할 방향성을 표현할 수 있다.

장점:

- `state→action` 관계가 `action→state`와 같을 필요가 없다.
- exact tool choice에는 효율적이다.

한계:

- `E(x,x)=0`, 비음수, triangle inequality 같은 metric 성질을 보장하지 않는다.
- 값 자체를 step 수로 해석하려면 Q regression/calibration이 필요하다.

해석:

- bilinear가 symmetric family보다 좋고 reverse-role 성능 차이도 크다면 directionality가 중요하다는 근거다.
- 다만 parameter 수가 더 많아서 parameter-matched control 없이는 거리 함수의 효과라고 할 수 없다.

##### 7. Poincaré/hyperbolic distance

Poincaré ball은 중심에서 멀어질수록 같은 반지름 안에 더 많은 분기를 배치할 수 있어 hierarchy를 compact하게 표현할 수 있다.

장점:

- identifier→annotation→evidence처럼 분기가 늘어나는 구조에 적합할 가능성이 있다.
- 낮은 차원에서 tree-like graph를 표현하는 가설을 시험할 수 있다.

한계:

- Poincaré distance 자체는 여전히 대칭이다.
- 실제 genomics workflow는 tree가 아니라 branch, merge, alternative source, cycle이 있는 directed graph다.
- ball boundary 근처의 수치 불안정과 curvature tuning이 있다.

필수 진단:

- executor state graph의 tree-likeness 또는 graph hyperbolicity를 먼저 측정
- graph의 tree-likeness/hyperbolicity와 branching factor별 성능
- latent radius와 search depth/remaining cost의 상관
- long-horizon에서만 이득이 생기는지
- metric은 좋아도 cosine tool decoder에서 이득이 사라지는지

해석:

- long-horizon·high-branching에서만 좋아지면 hierarchy 가설을 지지한다.
- 모든 subset에서 조금 좋아지는 정도라면 parameterization/regularization 효과일 수 있다.

##### 8. Order-violation energy

정보를 얻을수록 state가 partial order 위에서 증가한다고 가정한다. 예를 들어 coordinate-wise order를 사용할 때 한 방향의 violation만 penalize한다.

\[
E_{ord}(s,g)=\|\max(0,z(s)-z(g))\|^2
\]

부호 방향은 “더 많은 정보를 가진 state가 위인가 아래인가”라는 convention에 맞춰 고정한다.

장점:

- evidence accumulation과 prerequisite 관계를 직접 모델링한다.
- 비대칭이다.

한계:

- 잘못 얻은 정보를 삭제하거나 build를 교체하는 transition은 단순 monotone order가 아니다.
- 모든 workflow를 하나의 coordinate order로 표현하기 어렵다.

해석:

- monotone evidence task에서는 좋고 repair/revision task에서는 나쁘다면 예상과 일치하는 유익한 결과다.

##### 9. Learned quasimetric

Quasimetric은 symmetry를 요구하지 않지만 비음수, 자기거리 0, triangle inequality를 유지하는 directed distance다. goal-conditioned optimal value와 temporal distance를 연결한다는 선행 연구가 있어 이 프로젝트의 가장 직접적인 고급 후보다.

\[
D_Q(s,g)\ge0,\quad D_Q(s,s)=0
\]

\[
D_Q(s,g)\le D_Q(s,u)+D_Q(u,g)
\]

하지만 일반적으로 다음은 허용한다.

\[
D_Q(s,g)\ne D_Q(g,s)
\]

권장 구현 순서:

1. 현재 bilinear/Q head로 directionality의 필요성을 먼저 확인한다.
2. triangle consistency penalty를 가진 simple directed energy를 실험한다.
3. 신호가 있으면 Interval Quasimetric Embedding 같은 구조적으로 보장된 head를 추가한다.

해석:

- unseen long-horizon goal에서 bilinear보다 quasimetric이 좋다면 triangle/path-stitching inductive bias의 효과를 주장할 수 있다.
- short-horizon에서만 같고 long-horizon에서 차이가 나야 가설과 특히 잘 맞는다.

##### 10. Unconstrained pair MLP

\[
E_{MLP}(s,a,g)=MLP([z_s,z_g,z_a,z_s-z_g,z_s\odot z_g])
\]

장점:

- 최대한 유연한 compatibility baseline이다.

한계:

- geometry 성질과 외삽 bias가 없다.
- 작은 데이터에서 과적합하기 쉽다.

해석:

- MLP가 in-domain에서는 최고지만 unseen template/horizon에서 quasimetric보다 떨어지면 structured geometry의 일반화 이득이다.
- MLP가 모든 split에서 최고면 특정 metric의 필요성을 주장하기 어렵고, representation 또는 데이터가 더 중요한 결론이 된다.

### 6.6 공정한 distance study를 위한 3단계 protocol

현재 구조에서는 distance family를 바꾸면 projector도 함께 다른 gradient로 학습된다. 따라서 결과를 곧바로 “순수 거리 함수 효과”라고 해석할 수 없다. 이를 세 층으로 분리한다.

#### Tier 1 — Raw-space diagnostic

- frozen cache 완전히 동일
- trainable projector 없음
- cosine, Euclidean, train-only whitened Euclidean 비교
- 목적: upstream embedding 자체의 geometry 진단

#### Tier 2 — Frozen-common-projector diagnostic

- transition/future-state objective로 공통 projector 하나를 먼저 학습
- 그 projector를 고정
- 그 위에서 가능한 한 작은 distance head만 학습
- 목적: 표현 변화의 confound를 줄여 head inductive bias 비교

#### Tier 3 — End-to-end downstream comparison

- 각 distance family가 projector와 함께 end-to-end 학습
- shared dimension, hidden dimension, optimizer, training step, data order를 통제
- parameter 수가 다른 경우 parameter-matched variant 또는 parameter-count covariate를 함께 보고
- 동일한 greedy Q/STOP/Flow budget에서 최종 agent 성능 비교
- 목적: 실제 시스템으로서 어느 조합이 가장 유용한지 비교

보고서에서는 다음 표현을 구분한다.

- Tier 1/2: **distance/geometry diagnostic**
- Tier 3: **joint representation-and-distance system comparison**

### 6.7 거리 함수별 공통 정량 지표

#### Oracle alignment

- `MAE(Dθ, D_G)` 또는 `MAE(Qθ,Q*)`
- Spearman/Kendall correlation with cost-to-go
- optimal-set top-1/top-k accuracy
- regret@1
- unreachable state AUROC/AUPRC

#### 방향성과 경로 구조

- **asymmetry alignment**

  \[
  \Delta_{asym}=|(D_\theta(s,g)-D_\theta(g,s))-(D_G(s,g)-D_G(g,s))|
  \]

- triangle violation rate와 magnitude
- Bellman residual

  \[
  |D_\theta(s,g)-\min_a(c+D_\theta(s',g))|
  \]

- optimal path에서 potential descent 비율
- step별 평균 potential 감소량
- multi-step path cost additivity

#### 임베딩 품질과 failure mode

- hubness: 특정 tool이 과도하게 nearest neighbor가 되는 비율
- anisotropy/effective rank
- entity identity만으로 예측할 수 있는 정도
- hard negative margin:
  - GRCh37 vs GRCh38
  - 같은 gene의 다른 transcript version
  - 같은 position의 다른 ref/alt
  - 같은 tool description이지만 다른 precondition
- perturbation 뒤 recovery action ranking

#### downstream

- verified task success
- excess cost vs oracle
- invalid/redundant action rate
- learned STOP 포함 success
- horizon/branching/source-failure별 success
- State Flow의 next-anchor decoding error와 whole-plan utility

### 6.8 사전에 정할 해석 규칙

결과를 본 뒤 이야기를 맞추지 않기 위해 다음을 preregister한다.

| 관측 결과 | 허용되는 해석 |
|---|---|
| Cosine ≈ learned methods | frozen semantic space로 충분하며 복잡한 metric의 이득이 없음 |
| Whitened > raw Euclidean | 주 문제 중 하나가 anisotropy/common direction이었음 |
| Low-rank Mahalanobis > diagonal | 축 reweighting보다 feature interaction/rotation이 중요함 |
| Bilinear > symmetric families | state-action role 및 방향성 compatibility가 중요함 |
| Quasimetric > bilinear, 특히 long horizon | triangle/path-stitching inductive bias가 일반화를 도움 |
| Poincaré가 high-branching에서만 우세 | hierarchy/branch expansion 가설과 정합적 |
| Order energy가 accumulation에서만 우세 | monotone information partial order 가설과 정합적 |
| Structured-only ≈ multimodal | frozen text/domain view의 추가 가치가 입증되지 않음 |
| Exact contract mask에서만 모든 방법이 높음 | embedding이 아니라 symbolic contract가 문제를 해결함 |
| Geometry metric은 좋지만 Flow는 나쁨 | geometry와 Flow target/decoder의 불일치 또는 Flow 불필요 |
| Flexible MLP가 모든 unseen split에서 우세 | 특정 거리 공리보다 데이터와 function capacity가 중요함 |

### 6.9 distance family 선택 규칙

- primary selection은 non-MARRVEL grouped dev의 lexicographic criterion으로 한다.
- 권장 순서:
  1. verified reachability/optimal-action accuracy
  2. lower regret@1
  3. lower Bellman residual와 better calibration
  4. lower parameter/latency
- 단일 seed 최고값이 아니라 task-level bootstrap CI와 최소 3개 seed의 안정성을 본다.
- MARRVEL에서는 dev에서 고른 primary model 하나를 confirmatory model로 평가한다.
- 나머지 distance family의 MARRVEL 결과는 사전 등록된 secondary ablation으로만 보고, 그 결과로 primary를 다시 고르지 않는다.
- 여러 family 간 유의성 검정을 많이 하면 Holm correction 또는 명확한 exploratory 표기를 사용한다.
- cosine처럼 값의 범위가 제한된 family와 step cost scale을 직접 맞출 때는 train/dev만으로 fitting한 동일 형태의 작은 monotone calibrator를 각 후보에 붙인다. primary ranking metric과 regret는 scale-invariant하게 유지한다.

### 6.10 최종 후보: typed product geometry

단일 전역 거리 하나가 biomedical semantics, ontology hierarchy, procedural progress, exact genomic identity를 모두 맡는다고 가정할 필요는 없다. 단일 family ablation을 끝낸 뒤 다음 product geometry를 최종 가설로 시험한다.

\[
D_{total}(s,g)=
w_{proc}(g)D_{quasi}
+w_{onto}(g)D_{hyp}
+w_{sem}(g)D_{cos}
+D_{exact}
\]

- `D_quasi`: tool 실행과 prerequisite의 방향성 있는 진행
- `D_hyp`: HPO/disease ontology와 같은 계층적 관계
- `D_cos`: query, evidence, tool description의 일반 의미
- `D_exact`: assembly, coordinate, ref/alt, transcript version의 symbolic mismatch/cost
- `w(g)`: goal 유형에 따라 작은 gate가 정하는 가중치

예를 들어 Qwen 공간에서 GRCh37과 GRCh38이 가까워도 `D_exact`의 build mismatch는 0이 되지 않는다. liftover를 수행해 exact field가 바뀌면 `D_quasi`와 predicted cost-to-go가 함께 감소해야 한다.

다만 처음부터 mixture만 학습하면 무엇이 기여했는지 알 수 없다. 반드시 각 component 단독 실험을 먼저 하고, product geometry는 마지막 조합 실험으로 둔다.

### 6.11 State Flow에서도 거리를 다시 비교한다

State Flow에는 서로 다른 두 거리 역할이 있다.

1. **planning potential**: `D(s,g)` 또는 `Q(s,a,g)`
2. **generated next-anchor matching**: `D(T(z_s,a), z_hat_next)`

둘을 항상 같은 family로 묶지 않는다. 예를 들어 다음 조합이 가능하다.

| Potential | Anchor decoder | 의미 |
|---|---|---|
| Quasimetric | Euclidean | goal progress는 방향성 있게, local successor matching은 안정적인 평면 거리로 |
| Bilinear Q | Cosine | action ranking은 role-aware, codebook decode는 정규화 similarity로 |
| Poincaré value | Poincaré | hierarchy 가설을 end-to-end로 일관되게 시험 |
| Search Q | learned transition residual | geometry upper bound와 Flow generation을 분리해 시험 |

따라서 최종 ablation은 “metric 하나를 전체 시스템에 일괄 적용”하는 방식보다 다음과 같이 factorize한다.

```text
potential family × local transition distance × planner family
```

다만 전체 Cartesian product는 돌리지 않는다. non-MARRVEL dev에서 다음 최소 조합만 비교한다.

- symmetric baseline: cosine/cosine
- learned symmetric: low-rank Mahalanobis/Euclidean
- directed baseline: bilinear/cosine
- proposed: quasimetric/Euclidean 또는 quasimetric/learned residual
- hierarchy ablation: Poincaré/Poincaré

각 geometry에는 그에 맞는 trajectory interpolation을 사용해야 한다.

- Euclidean: straight interpolation
- Mahalanobis: whitening/projection 좌표에서 straight interpolation
- cosine/angular: unit sphere 위 spherical interpolation
- Poincaré: log/exp map을 이용한 hyperbolic geodesic 또는 Riemannian flow
- order/quasimetric: monotone cone을 벗어나지 않는 nonnegative increment/projection
- bilinear/MLP: 고유한 geodesic이 없으므로 공통 Euclidean state latent에서 Flow를 만들고 energy는 reranking에만 사용

Poincaré candidate에 단순 Euclidean 직선 Flow를 적용한 뒤 Flow 성능을 비교하면 geometry가 아니라 잘못된 interpolation을 벌주는 실험이 된다. 따라서 다음 두 효과를 분리한다.

1. 같은 common Flow에서 distance head만 바꾸는 통제 실험
2. 각 geometry에 맞는 geodesic/flow를 쓰는 end-to-end 실험

---

## 7. Search-DAgger: 정답 경로 밖에서 회복하는 학습

정답 prefix만 학습하면 agent가 한 번 틀린 뒤 도달하는 상태를 본 적이 없다. 이것이 sequential imitation learning의 대표적인 distribution shift다.

Search-DAgger는 human expert 대신 graph search를 oracle로 쓴다.

```text
Round 0: search-generated dataset으로 Q/energy 학습

Round 1:
  learned agent rollout
  -> low-margin / wrong / failed state 수집
  -> search oracle이 V*, Q*, optimal set 재계산
  -> replay buffer에 추가
  -> 재학습

Round 2..N: 반복
```

권장 초기값은 3–5 round지만, round 수는 MARRVEL이 아닌 internal dev learning curve로 결정한다.

모든 graph를 전수 확장할 수 있을 만큼 작다면 먼저 exhaustive search distillation을 한다. DAgger는 graph가 커지거나 perturbation으로 learner-specific state가 많이 생길 때 가장 유용하다.

---

## 8. FlowAgent를 살리는 방법: Tool Flow가 아니라 State Flow

여기서 Flow를 버리는 것이 아니다. 먼저 search/value로 학습 신호를 만든 뒤, **Flow가 생성해야 할 대상 자체를 바꾼다.**

### 8.1 기존 방식의 불일치

원래 FlowAgent는 각 future plan position을 tool semantic anchor로 나타내고, Euclidean nearest neighbor로 tool을 decode한다. 또한 expert trace의 각 prefix와 남은 suffix를 짝지어 continuous flow target을 만든다. 원 논문은 6,865 task와 3,930 tool schema를 사용했기 때문에, 7개의 training task로 같은 supervision 구조를 재현하기는 어렵다.

현재 GeoFlowAgent에서도 metric module이 배우는 state-tool compatibility와 Flow가 회귀하는 tool-anchor polyline이 정확히 같은 기하를 보장하지 않는다. 이것이 geometry 개선이 downstream Flow 개선으로 이어지지 않은 이유 중 하나다.

### 8.2 새 방식: 미래 상태 anchor를 생성한다

search path가 다음과 같다고 하자.

\[
s_0\xrightarrow{a_0}s_1\xrightarrow{a_1}s_2\cdots\xrightarrow{a_{T-1}}s_T
\]

Flow target은 tool prototype 열이 아니라 다음 중 하나로 만든다.

- future state anchors: `[z(s_1,g), z(s_2,g), ..., z(s_T,g)]`
- transition residuals: `[Δz_0, Δz_1, ..., Δz_{T-1}]`

\[
\Delta z_i=z(s_{i+1},g)-z(s_i,g)
\]

이렇게 하면 Flow가 직접 “어떤 tool 이름이 다음인가”를 그리는 것이 아니라 **목표 상태로 향하는 실행적 변화 방향**을 그린다.

### 8.3 action decoding

생성된 다음 state anchor를 `\hat z_{t+1}`라 하면 action은 다음 점수로 고른다.

\[
a_t=\arg\min_{a\in A(s_t)}
\left[
\|T_\theta(z_t,a)-\hat z_{t+1}\|^2
+\alpha Q_\theta(s_t,a,g)
\right]
\]

- `Tθ`는 action-conditioned successor predictor다.
- `Qθ`는 search-distilled remaining cost다.
- contract는 불가능한 action만 제거한다.
- 첫 action만 실제 실행한 뒤 새로운 state를 다시 embed하고 replan한다.

### 8.4 potential-descent regularization

`Φ(s,g)`를 목표까지 남은 비용으로 두고, 생성 경로가 가능하면 단조 감소하도록 학습한다.

\[
\mathcal{L}_{descent}=
\sum_t \max(0,m+\Phi(\hat s_{t+1},g)-\Phi(\hat s_t,g))
\]

이 loss는 단순히 latent path가 매끄럽다는 것보다 더 의미가 있다. “생성된 경로가 실제 목표에 가까워지고 있는가?”를 직접 묻기 때문이다.

### 8.5 한 개 path 대신 path distribution을 학습한다

search에서 얻은 top-K shortest/Pareto-valid path에 utility를 붙인다.

\[
p^*(\tau\mid s,g)\propto\exp(-C(\tau)/\beta)
\]

Flow는 이 multi-reference distribution을 distill한다. 독립적인 tool끼리 순서를 바꾼 topological variant도 모두 valid reference가 될 수 있다.

이 설계의 연구 가치는 다음과 같다.

- human expert trajectory 의존성을 줄인다.
- 한 개 임의의 정답 순서를 강요하지 않는다.
- embedding geometry와 trajectory generator가 모두 state transition/cost-to-go를 공유한다.
- `GRCh37`/`GRCh38`처럼 semantic similarity와 operational equivalence가 다른 경우를 typed state와 transition으로 처리한다.

### 8.6 Flow가 꼭 이겨야 하는 지점

Flow가 greedy Q보다 이득을 보일 가능성이 있는 조건은 다음이다.

- horizon이 길다.
- 여러 유효 경로가 있다.
- 초반 action의 장기 결과가 다르다.
- source failure 이후 대체 계획이 필요하다.
- unseen tool이 기존 transition role과 유사하다.

반대로 single-hop MARRVEL 문항에서는 Flow의 장점이 거의 나타나지 않을 수 있다. 이것은 실패가 아니라 benchmark와 연구 질문의 불일치다.

---

## 9. STOP은 별도 문제로 분리한다

현재 실험에서는 valid action 선택보다 premature STOP이 더 큰 실패였다. STOP을 일반 tool prototype처럼 같은 공간에서만 예측하면 class imbalance와 길이 예측 문제가 섞인다.

권장 구조는 다음과 같다.

1. **연구용 oracle ceiling**
   - exact verifier가 만족되면 hard STOP한다.
   - geometry/action selection의 최대 성능을 측정한다.
2. **실제 agent용 learned STOP**
   - `goal_satisfied`, answer completeness, evidence coverage, uncertainty를 입력하는 작은 별도 head를 둔다.
3. **평가 분리**
   - goal reached but failed to stop
   - premature stop
   - correct continuation
   - correct stop

모든 baseline에는 같은 STOP controller를 사용해야 한다. `random action + oracle STOP`과 `learned planner + learned STOP`을 비교하면 planner보다 STOP oracle의 차이를 측정하게 된다.

---

## 10. MARRVEL의 정확한 역할

### 10.1 MARRVEL은 train이 아니라 external test다

공개 Hugging Face 데이터는 실제로 `test` split 100행으로 제공되고 있다. 질문, 이름, category, expected answer는 있지만, 현재 GeoFlowAgent가 요구하는 typed state, trajectory, private executable verifier가 바로 포함되어 있지는 않다.

따라서 권장 split은 다음이다.

```text
TRAIN
  non-MARRVEL entities + non-MARRVEL generated workflows

DEV
  entity-group와 template-group가 TRAIN과 분리된 non-MARRVEL workflows

INTERNAL TEST
  완전히 unseen entity + unseen template + multi-step/recovery workflows

EXTERNAL TEST
  locked MARRVEL 100 questions
```

### 10.2 사용할 수 있는 것과 없는 것

학습에서 사용 가능:

- 동일한 tool schema와 contract
- 공식적인 tool dependency 지식
- MARRVEL 문항과 겹치지 않는 public database record
- pinned non-test snapshots
- 일반적인 genomics workflow 규칙

학습/선택에 사용 금지:

- MARRVEL 질문 문장
- expected answer
- MARRVEL에서 역으로 만든 trajectory
- MARRVEL entity-predicate-answer tuple
- MARRVEL 성능을 보고 고른 distance, threshold, NFE, prompt, model size

엄격한 generalization claim을 위해서는 gene alias, transcript/HGVS, rsID/SPDI, GRCh37/38 coordinate, disease/phenotype synonym까지 정규화한 test entity closure를 만들고 train generator에서 제외하는 것이 좋다.

### 10.3 이미 test 예시를 본 사실의 처리

이 프로젝트에서는 MARRVEL 예시와 구조를 이미 설계 단계에서 확인했다. 따라서 논문에서는 이를 **pristine blind test**라고 부르면 안 된다.

정직한 표현은 다음이다.

> “MARRVEL was locked as an external test after initial benchmark-format inspection; no MARRVEL item was used for training or model selection.”

강한 confirmatory claim이 필요하면 별도의 숨겨진 전문가 평가셋을 추가해야 한다.

### 10.4 MARRVEL만으로 Flow를 증명할 수 없는 이유

MARRVEL 100문항 중 상당수는 한 번의 lookup 또는 짧은 chain으로 답할 수 있다. 따라서 이것은 다음을 평가하기에는 좋다.

- 실제 biomedical/genomic tool grounding
- argument 정확성
- answer correctness
- source/evidence 사용

하지만 다음을 단독으로 검증하기에는 부족하다.

- long-horizon planning
- multi-path generation
- recovery after early mistakes
- trajectory geometry
- Flow의 global planning 이점

그래서 두 종류의 test가 필요하다.

1. **Internal Path Test**: 경로/회복/비용을 정밀 측정하는 unseen multi-step graph
2. **MARRVEL External Test**: 실제 도메인 질문으로 외부 일반화를 확인

필요하면 MARRVEL 문항에 action dependency DAG와 복수 valid path를 평가 전 미리 annotation한 `MARRVEL-Path`를 만들 수 있다. 단 이것도 전부 evaluation-only여야 한다.

---

## 11. 실험 설계

### 11.1 핵심 가설

**H1 — Geometry**

> Goal-conditioned energy는 unseen task에서 search의 실제 cost-to-go와 상관되고, 단순 의미 임베딩보다 optimal action을 잘 선택한다.

**H2 — Recovery**

> Search-DAgger는 expert prefix만 학습한 모델보다 off-trajectory와 source failure 상태에서 회복률을 높인다.

**H3 — State Flow**

> State Flow는 single-reference tool-anchor Flow보다 multi-path/long-horizon task에서 낮은 regret과 높은 success를 보인다.

**H4 — Domain views**

> biomedical/entity encoder는 의미적 retrieval을 돕지만 assembly, coordinate, version의 정확성은 typed structured channel이 담당한다.

**H5 — Flow의 필요 조건**

> Flow의 이득은 single-hop가 아니라 branch가 있고 여러 유효 경로가 있는 task에서 주로 나타난다.

### 11.2 반드시 포함할 baseline

1. contract-valid random + 동일 learned STOP
2. deterministic rule/FSM
3. exact Dijkstra search oracle ceiling
4. frozen embedding nearest neighbor
5. structured-only small MLP
6. 현재 7-task behavior cloning
7. autoregressive next-action planner
8. search-distilled Q/energy
9. `Q/energy + future-goal contrastive pretraining`
10. `Q/energy + Search-DAgger`
11. 기존 tool-anchor Flow
12. 새 state-anchor Flow

GFlowNet은 “좋은 경로의 다양성” 자체가 중심 연구 질문이 될 때 추가한다. GFlowNet의 state-flow conservation과 continuous flow matching은 이름만 비슷할 뿐 다른 방법이라는 점을 문서에서 명확히 구분해야 한다.

### 11.3 geometry 평가

- predicted `V/Q`와 exact cost-to-go의 MAE/Huber
- Spearman rank correlation
- optimal-set top-1 accuracy
- regret@1
- `P(success | predicted value)` calibration
- trajectory를 따라 potential이 감소하는 비율
- entity-held-out 성능
- template-held-out 성능
- tool-held-out 또는 source-held-out 성능
- hard negative별 성능: build, transcript version, alias, wrong allele

UMAP/t-SNE 그림은 보조 시각화일 뿐 핵심 증거로 쓰지 않는다. 핵심은 실행 비용과 action 결과의 정량적 대응이다.

### 11.4 planner 평가

- private verifier task success
- cost gap/regret vs search oracle
- invalid action rate
- redundant call rate
- snapshot miss rate
- perturbation recovery rate
- success by horizon
- success by branching factor
- correct STOP / premature STOP / missed STOP
- alternative valid path coverage
- generated path utility와 diversity
- wall-clock, API call 수, NFE, peak memory

### 11.5 MARRVEL primary metric

- `Verified Task Success@1` including learned STOP
- 7개 category macro average
- strongest preregistered non-flow baseline과 paired bootstrap 95% CI
- paired binary success에 대한 McNemar test

100문항은 작기 때문에 seed별 시행을 독립 표본처럼 세면 안 된다. 통계 단위는 task다. official free-text LLM judge와 exact trajectory EM은 보조 지표로 둔다.

---

## 12. 권장 데이터 규모와 split

다음 수치는 시작점을 위한 권장값이지 성공을 보장하는 magic number가 아니다.

### Pilot

- 8–10 workflow families
- 약 500 independent workflows
- 각 state에서 2개 이상의 known executable branch가 있는 사례를 의도적으로 포함
- entity-group와 template-group를 동시에 분리

### Main training candidate

- 2,000–5,000 independent workflows
- 대략 8,000–30,000 prefix/state examples
- common predicted tool마다 서로 다른 entity/template context 50–100개 이상을 초기 목표로 설정
- 실제 필요량은 data scaling curve로 결정

중요한 것은 prefix 수가 아니라 **독립 goal, entity family, workflow topology, failure mode의 수**다.

---

## 13. 단계별 실행 순서와 go/no-go

### Phase 0 — benchmark sealing

- MARRVEL revision/hash를 고정한다.
- questions/answers/verifier를 read-only evaluation 영역으로 옮긴다.
- train contamination 검사 규칙을 만든다.
- MARRVEL을 보기 전 선택할 metric과 baseline을 기록한다.

### Phase 1 — graph supervision

- weighted graph search를 구현한다.
- `V*`, `Q*`, regret, optimal set, top-K path를 cache한다.
- unknown action mask를 loss denominator까지 적용한다.
- Markov state에 failure/retry/source 상태를 추가한다.

**Gate 1:** non-MARRVEL grouped dev에서 contract만으로 optimal action이 결정되지 않고, 최소한 일부 상태에 서로 다른 nonzero regret branch가 있어야 한다.

### Phase 2 — goal-conditioned geometry

- 작은 directed Q/energy head를 학습한다.
- future-goal contrastive/transition objective를 보조로 붙인다.
- Euclidean, Mahalanobis, hyperbolic, asymmetric energy를 동일 parameter budget에서 비교한다.

**Gate 2:** structured-only, frozen nearest-neighbor, simple MLP보다 unseen entity+template split에서 regret/value calibration이 개선되어야 한다.

### Phase 3 — Search-DAgger

- learned agent가 방문한 low-margin/failure state를 수집한다.
- search oracle이 다시 라벨링한다.
- task-balanced replay buffer에 합친다.

**Gate 3:** held-out perturbation recovery와 long-horizon success가 개선되어야 한다.

### Phase 4 — Flow comparison

- 기존 tool-anchor Flow를 search의 multiple suffix로 재학습한다.
- state-anchor/residual Flow를 구현한다.
- greedy Q와 autoregressive planner를 compute-matched하게 비교한다.

**Gate 4:** multi-path/long-horizon subset에서 State Flow가 greedy Q 또는 compact AR보다 개선되지 않으면 Flow를 억지로 주 기여로 주장하지 않는다. 그 경우 연구의 주 결과는 goal-conditioned execution geometry와 search distillation이다.

### Phase 5 — locked evaluation

- 모든 model/hyperparameter/STOP/NFE를 고정한다.
- internal path test를 한 번 실행한다.
- MARRVEL 100문항을 한 번 실행한다.
- MARRVEL 결과를 본 뒤 수정한 결과는 exploratory v2로 분리한다.

---

## 14. 현재 코드에서 재사용할 것과 바꿀 것

### 그대로 재사용할 기반

- frozen embedding cache
- multi-view encoder 구조
- typed schema와 contract engine
- exact snapshot replay
- private verifier separation
- closed-loop execute-one-action-and-replan
- current `shortest_distance` / `action_analysis`의 기본 논리
- embedding/agent report infrastructure

### 즉시 바꿀 부분

1. `contracts.py`
   - BFS를 weighted Dijkstra/A*로 일반화
   - full state graph와 predecessor 저장
   - top-K/Pareto path와 `V*/Q*` 생성
2. processed schema
   - goal embedding/typed goal
   - full known-action mask
   - per-action successor, cost, Q, regret
   - multiple suffix references
3. `training/metric.py`
   - unknown action을 denominator에서도 제거
   - goal-conditioned Q/V/Bellman objective 추가
4. `models/functional.py`
   - state-tool similarity 중심에서 state-action-goal energy로 변경
   - asymmetric energy와 scalar value head 추가
5. STOP
   - flow slot과 분리된 completion head
6. 새 `dagger.py`
   - rollout, hard-state selection, oracle relabel, replay aggregation
7. Flow
   - tool-prototype target 외에 state-anchor/residual target 추가

### 현재 config에서 축소할 것

현재 metric/flow head는 데이터 크기에 비해 크다. 우선 다음 범위에서 시작한다.

```yaml
geometry:
  shared_dim: [64, 128, 256]
  hidden_dim: [128, 256]
  layers: [1, 2]
  energy:
    - cosine
    - euclidean
    - diagonal_mahalanobis
    - lowrank_mahalanobis
    - asymmetric_bilinear
    - poincare
    - order_violation
    - quasimetric
    - pair_mlp_control

search:
  objective: lexicographic
  top_k_paths: [4, 8]

dagger:
  rounds: [3, 5]
  query_policy: [error, low_margin, uncertainty]

state_flow:
  target: [future_state, state_delta]
  max_plan_length: [4, 8, 12]
```

모든 조합을 무작정 grid search하지 않는다. Phase별 질문에 답하는 최소 ablation만 수행한다.

---

## 15. 다른 학습 방법과의 비교

### Vanilla offline RL / IQL

mixed-quality reward-labeled transition이 충분하면 baseline으로 의미가 있다. 그러나 현재처럼 exact executor와 search oracle을 만들 수 있는 작은 discrete 환경에서는 supervised dynamic programming/Q distillation보다 variance와 튜닝 부담이 크다. 주력으로 삼을 이유가 없다.

### Decision Transformer

랜덤/혼합 rollout과 return label이 충분할 때 sequence baseline으로 좋다. 하지만 작은 trajectory 집합의 coverage를 스스로 늘려 주지는 않는다.

### GFlowNet

여러 terminal plan을 reward에 비례해 다양하게 sampling하는 것이 연구 목표라면 매우 흥미롭다.

```text
terminal object = STOP까지의 완성된 tool plan
reward = verifier_success × exp(-cost/redundancy)
```

그러나 sparse terminal reward, retry/cycle, incomplete snapshot 문제가 있다. `(state, step)` time expansion과 predecessor graph도 필요하다. 따라서 graph와 multi-path corpus가 충분해진 뒤 State Flow와 비교하는 후속 실험으로 적합하다.

### Reward-only continuous Flow

권장하지 않는다. Continuous flow matching은 결국 target sample distribution이 필요하다. verifier만 주고 좋은 plan을 스스로 발견해 주는 방법이 아니다. 먼저 search가 좋은 suffix/path distribution을 만든 뒤 Flow가 그것을 amortize해야 한다.

---

## 16. 연구의 독창성과 주장 가능한 범위

### 독창적인 지점

단순히 “genomics에 FlowAgent를 적용했다”는 것보다 다음 조합이 더 분명한 연구 기여가 된다.

1. frozen multi-view encoder 위에서 **semantic similarity가 아니라 directed executable cost-to-go**를 학습한다.
2. expert trajectory 대신 exact tool execution과 counterfactual search로 supervision을 만든다.
3. 하나의 gold sequence가 아니라 multiple valid path distribution을 학습한다.
4. tool semantic anchor가 아니라 future state transition을 continuous flow의 대상으로 삼는다.
5. typed genomic exactness와 learned semantic geometry를 분리해 결합한다.

### 과장하면 안 되는 지점

- 자동 생성한 prefix가 많아도 독립 환자/질병 사례가 많아진 것은 아니다.
- MARRVEL 100개만으로 임상적 일반화를 주장할 수 없다.
- UMAP이 예쁘다고 좋은 planner는 아니다.
- frozen biomedical encoder를 썼다는 사실만으로 genomic reasoning이 생기지는 않는다.
- search oracle과 동일 snapshot을 평가하면 단순 distillation 성공이지 real-world generalization은 아니다.
- Flow가 Q/AR baseline을 이기지 못하면 Flow 자체의 우수성을 주장하면 안 된다.

### 성공했을 때 가능한 정확한 주장

> “A small trainable module over frozen general and biomedical encoders learned a directed latent geometry aligned with executable cost-to-go in a genomics tool environment. Search-generated counterfactual supervision improved goal-conditioned planning and recovery on entity- and template-held-out workflows. A state-trajectory flow extension was evaluated separately for multi-path, long-horizon planning, and the frozen system was finally tested on locked MARRVEL questions.”

---

## 17. 최종 추천

이 프로젝트의 중심을 **Flow Matching 기법 자체**로 두지 않는 것이 좋다. 중심은 다음이어야 한다.

> **목표까지의 실행 가능성과 남은 비용을 표현하는 trajectory-aware geometry**

그리고 Flow는 그 geometry가 정말 유용하다는 것이 확인된 뒤, 여러 미래 상태 경로를 생성하는 두 번째 가설로 둔다.

가장 좋은 최소 연구 단위는 다음이다.

1. non-MARRVEL executable task generator
2. weighted search oracle
3. directed goal-conditioned Q/energy geometry
4. future-goal relabeling
5. Search-DAgger
6. greedy value planner
7. state-anchor Flow ablation
8. locked MARRVEL external evaluation

이 방향은 현재의 실패 원인인 소수 trajectory, contract oracle, single suffix, off-policy state 부재, STOP 혼입, geometry–Flow 불일치를 한 번에 정면으로 다룬다. 동시에 사용자가 계속 중요하게 본 “좋은 경로와 좋은 공간은 분리된 것이 아니라 서로를 규정한다”는 연구 관점도 가장 선명하게 보존한다.

---

## 18. 주요 근거 자료

- FlowAgent: [Tools as Continuous Flow for Evolving Agentic Reasoning](https://arxiv.org/html/2605.07339v2)
  - expert prefix와 complete remaining suffix로 plan supervision을 만들며, 6,865 tasks와 3,930 tool schemas를 사용한다.
- MARRVEL benchmark: [Hugging Face dataset](https://huggingface.co/datasets/hjeong84/marrvel-mcp-benchmark-data)
  - 공개 구성은 `test` 100 rows다.
- Search/learner 반복: [Expert Iteration, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html)
- Learner 방문 상태의 재라벨링: [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)
- Goal-conditioned representation/value: [Contrastive Learning as Goal-Conditioned Reinforcement Learning, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html)
- Directed temporal distance와 optimal value: [Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning, ICML 2023](https://proceedings.mlr.press/v202/wang23al.html)
- Hyperbolic hierarchy baseline: [Poincaré Embeddings for Learning Hierarchical Representations, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html)
- Partial-order representation baseline: [Order-Embeddings of Images and Language, ICLR 2016](https://arxiv.org/abs/1511.06361)
- 다양한 reward-proportional trajectory 생성의 대안: [GFlowNet Foundations, JMLR 2023](https://www.jmlr.org/papers/v24/22-0364.html)
- Continuous Flow Matching의 원형: [Flow Matching for Generative Modeling, ICLR 2023](https://openreview.net/forum?id=PqvMRDCJT9t)

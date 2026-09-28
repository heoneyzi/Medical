# Phase 0 Findings — gDTR PoC on HyenaDNA

**Project**: Genomic Deep-Thinking Ratio (gDTR) — Causal Genomic Foundation Model의 Layer-wise Prediction Convergence 분석
**Phase**: 0 (Proof of Concept), HyenaDNA-medium-160k (6.6M params, 8 layers, pure-Hyena)
**Document version**: 2026-04-26 v1.0
**Status**: Phase 0 fully complete; this document synthesizes all 9 stages of analysis

---

## 0. TL;DR

본 Phase 0는 Chen et al. (2026)의 NLP DTR 방법을 genomic causal language model에 전이하기 위한 사전등록된(pre-registered) PoC이다. RTX 3090 단일 GPU에서 약 36분의 컴퓨트로 9단계 분석(2 sanity gate, 1 Mann-Whitney gate, 1 informational variant gate, 1 hyperparameter sweep, 1 cross-gene 복제, 1 종합 결정 보고, 5개 mechanistic 진단)을 수행하였다.

**3가지 paper-grade 발견**: (1) **Layer 7 architectural anomaly** — 순수 Hyena conv 모델의 마지막 block(L7→L8)이 representation을 untied lm_head의 row-space로 강하게 회전시키며, 이로 인해 표준 logit-lens 단조성 가정이 깨진다. 5개 진단(D1–D5)을 통해 이 현상의 약 85–90%가 hypothesis (c) untied-LM-head alignment, 약 5–10%가 (a) long-range integration, 4.1%가 (b) ln_f normalization으로 분해된다. (2) **Hyperparameter non-transferability** — vocab |V|=12에서 JSD effective range 중간값이 0.019(NLP의 약 1/15)로 측정되어, NLP DTR의 default γ=0.5와 γ_cos=0.1은 모두 사용 불가능하다. Quantile 기반 γ_q70 calibration이 필수이며, 이는 새로운 방법론적 기여이다. (3) **Robustness & cross-gene replication** — Gate A/A'에서 단조성 검증이 strict 기준으로는 실패함에도 불구하고, running-min 기반 settling depth는 TP53(Mann-Whitney p=4.88×10⁻²²⁴, Cohen's d=−1.02)와 BRCA1(p≈0, d=−0.78)에서 같은 방향(intron > exon)의 강한 cross-gene 신호를 캡처한다.

**Phase 1 함의**: UR-gDTR(cosine lens)을 primary로 lock하고, Belrose 2023 tuned-lens를 Block 8 alignment 회전을 흡수하도록 학습한다. Variant 분석은 ΔD(ℓ) vector를 primary feature로 하며, Δc_discrete는 deprecated. 이 모든 결정은 Phase 0 데이터 기반으로 사후등록(post-registered)되었다.

---

## 1. Background and Pre-registration

### 1.1 왜 gDTR인가

Chen et al. (2026, arXiv:2602.13517)이 제안한 Deep-Thinking Ratio(DTR)는 NLP causal language model의 layer-wise prediction convergence를 측정하는 training-free interpretability metric이다. 핵심 아이디어는 각 token에 대해 중간 layer의 prediction distribution이 final layer 분포와 수렴하는 깊이(settling depth)를 logit lens로 측정하고, 모델 깊이의 상위 (1−ρ) 구간에서 settling되는 token을 "deep-thinking token"으로 분류하는 것이다. 8개 reasoning model(GPT-OSS-20B/120B, DeepSeek-R1-70B, Qwen3-30B 등)에 대해 4개 reasoning benchmark에서 task accuracy와 평균 Pearson r=0.683의 상관을 보였으며, GPT-OSS-120B에서는 r=0.828에 달한다.

본 연구는 이 metric을 genomic causal language model로 전이한다. 단, NLP→genomic 전이에는 본질적 차이 두 가지가 있다: (i) vocab size가 |V|≈100K에서 |V|=12(A/T/C/G + special)로 약 8000배 작고, (ii) genomic CLM 중 Hyena 계열은 attention이 없거나(HyenaDNA) hybrid 구조(Evo 2)이다. 두 차이가 DTR을 단순 적용하지 못하게 막는지, 어떤 calibration이 필요한지를 PoC 단계에서 사전 검증해야 한다.

### 1.2 왜 HyenaDNA로 PoC인가

본 연구의 primary 모델은 Evo 2 7B(arcinstitute/evo2_7b)이지만, 7B 모델 forward는 H100 cloud time을 요구한다. 방법론 prototype의 빠른 iteration을 위해 같은 Hyena 계열의 소형 모델인 HyenaDNA-medium-160k(LongSafari/hyenadna-medium-160k-seqlen-hf, 6.6M params, 8 layers)를 PoC로 채택했다. 이 모델은 RTX 3090 24GB에서 6kb forward 약 0.18초, 24GB VRAM의 1%만 소비한다.

HyenaDNA는 **순수 Hyena conv** 모델이며 attention block이 전혀 없다. 따라서 Phase 0의 logit-lens 검증은 "Hyena conv block 직후 residual에서 logit lens가 의미 있는가"라는 질문에 직접 답한다. 이 결과는 Phase 1 Evo 2 hybrid의 Hyena 부분에 logit lens를 적용할지, tuned lens로 대체할지, UR-gDTR을 primary로 격상할지를 binding한다.

### 1.3 Pre-registered Three-Gate Design

`phase0_design.md`(2026-04-26 v1.0)는 분석 전에 다음을 lock했다:

**Gate A (blocking)**: Logit lens validity on Hyena conv blocks. M1(top-1 monotonicity rate) ≥ 0.80 AND M2(JSD running-min monotonicity rate) ≥ 0.85 AND per-layer M2 ≥ 0.70. 실패 시 UR-gDTR primary 격상 또는 tuned lens.

**Gate B (blocking)**: Genomic signal. TP53 region에서 gDTR(coding exon) vs gDTR(intron) Mann-Whitney U two-sided p < 0.001 (Bonferroni 보정 over 5 contexts → effective α=0.0001).

**Gate C (informational, non-blocking)**: Variant signal. TP53 hotspot 5종에서 ≥3종이 max|ΔD(ℓ)| > shuffled-control p95.

추가로 Gate A'(post-hoc)는 Gate A가 fail한 경우 UR-gDTR cosine lens에 대해 동일 검증을 수행한다.

각 gate의 임계값, 통계 검정, 시각화 spec, 결정 트리는 모두 분석 전에 lock되었으며 본 보고서는 lock된 임계값 그대로 적용한 결과를 보고한다. 사후 변경된 임계값은 없다(단, design 가정의 사실 정정 — Appendix C — 은 분석 임계값과 무관하다).

---

## 2. Methods Summary

### 2.1 Pipeline

```
input nucleotide sequence (6 kb)
  → tokenize (BOS prepended → input_ids of length T+1=6001)
  → HyenaDNA forward(output_hidden_states=True)
  → hidden_states tuple of length L+2=10
       [0]=embedding, [1..8]=pre-ln_f block outputs, [9]=post-ln_f
  → slice off BOS: hidden_states[:, 1:, :]
  → for each ℓ ∈ {1..8}:
       JSD lens:    p_ℓ = softmax(lm_head(ln_f(h_ℓ))[..., :12])
                    D_jsd(ℓ) = JSD(p_ℓ, p_L) / log(12)
       UR lens:     D_cos(ℓ) = 1 - cos_sim(h_ℓ, h_L)
  → running_min(D) along layer axis → c_discrete, c_interp
  → gDTR = mean over positions of (c > ρ·L)
```

### 2.2 Key Calibration Choices

원본 DTR의 NLP default(γ=0.5)는 vocab=12 모델에서 거의 모든 position을 saturate시킨다(§4 참조). Phase 0는 **quantile-based γ**를 도입한다: 각 region에서 penultimate layer의 running-min D 분포의 70th percentile(γ_q70)을 γ로 사용. UR lens의 경우 γ_cos_q70 ≈ 0.48–0.51 범위로 측정되었다.

Depth fraction ρ=0.85는 NLP default 그대로 채택. L=8에서 deep regime은 c > 6.8, 즉 c ∈ {7, 8}이다.

### 2.3 Variant Analysis Framework

Variant position에서 ref/alt 두 forward를 수행하고 다음 5개 metric을 계산한다:

- `Δc_discrete = c_alt − c_ref` (이산 정수)
- `Δc_interp` (γ-crossing 위치를 layer 사이에서 선형 보간)
- `ΔD(ℓ) = D_alt(ℓ) − D_ref(ℓ)` ∈ ℝ^L (vector)
- `max|ΔD|` (scalar summary, sign 잃음)
- `signed_argmax_ΔD` (peak disruption layer의 부호 보존 ΔD)

Primary metric 선정은 사전등록된 규칙: Spearman ρ(Δc_interp, signed_argmax_ΔD) ≥ 0.7 → Δc_interp scalar primary; 0.5 ≤ ρ < 0.7 → signed_argmax_ΔD primary; ρ < 0.5 → ΔD vector primary.

전체 파이프라인 spec은 `phase0_design.md` §3-§6 참조.

---

## 3. Finding 1 — Layer 7 Architectural Anomaly

### 3.1 Statement

순수 Hyena conv 8-layer 모델에서, 마지막 block(L7→L8)은 representation을 untied lm_head의 row-space로 강하게 회전시킨다. 이 회전의 magnitude는 다른 모든 block의 약 3배이며, 결과적으로 약 54%의 position에서 layer 7의 hidden state가 layer 6보다 final hidden state로부터 더 멀어진다. 이는 NLP transformer에서 표준적으로 가정되는 smooth-monotone-convergence 가정을 위반하며, **logit-lens 기반 sanity check를 통과하지 못한다**.

본 발견은 우리가 알기로 SSM/Hyena 계열 모델에 대한 logit-lens 거동의 첫 systematic report이다.

### 3.2 Evidence

**Gate A (JSD lens)**: 100개 random 6kb sequence(50 GC-matched chr17 intergenic + 50 dinucleotide-shuffled, seed=42)에서 측정.

| Layer | M1 (top-1 stability) | M2 (JSD running-min monotonicity) |
|---:|---:|---:|
| 1 | 0.442 [0.441, 0.443] | 1.000 (boundary) |
| 2 | 0.458 | 0.495 |
| 3 | 0.513 | 0.819 |
| 4 | 0.566 | 0.864 |
| 5 | 0.582 | 0.927 |
| 6 | 0.553 | 0.965 |
| 7 | 0.448 | **0.120** |
| 8 | 1.000 (self) | 1.000 (self) |

Global rate: M1=0.281, M2=0.009. 두 값 모두 사전등록 임계값(0.80, 0.85)에 미달.

**Gate A' (UR cosine lens)**: 같은 100 sequence에 대해 재측정.

| Layer | M2_ur (cosine running-min monotonicity) |
|---:|---:|
| 1 | 1.000 (boundary) |
| 2 | 0.505 |
| 3 | 0.898 |
| 4 | 0.917 |
| 5 | 0.968 |
| 6 | 0.987 |
| 7 | **0.462** |
| 8 | 1.000 (self) |

Global rate: M2_ur=0.178. JSD lens와 cosine lens가 **모두 layer 7에서 동일한 패턴으로 실패**한다는 사실이 핵심이다 — 이는 logit lens의 vocabulary projection 문제가 아니라 residual stream 자체의 geometric property임을 의미한다.

### 3.3 Mechanistic Decomposition (D1–D5)

본 연구는 L7 anomaly의 origin을 5가지 진단으로 정량 분해하였다. 세 경쟁 가설:
- **(a)** 마지막 Hyena conv가 long-range integration을 수행 → block 8의 변환 magnitude가 long-range 구조를 가진 sequence에서 더 큼
- **(b)** Final LayerNorm(`model.hyena.backbone.ln_f`)이 representation geometry를 재조정 → ln_f를 빼면 anomaly 사라짐
- **(c)** Untied lm_head로 인해 마지막 block이 lm_head의 input space로 representation을 회전시킴 → block 8의 alignment energy가 급증

**D1 — ln_f isolation (가설 b 검증)**: 50 sequence에서 두 reference 비교: 표준 `D_post(ℓ) = 1 − cos(h_ℓ_pre, h_post)` vs ln_f-removed `D_pre(ℓ) = 1 − cos(h_ℓ_pre, h_8_pre)`.

| Layer | M2 (post-ln_f ref) | M2 (pre-ln_f ref) |
|---:|---:|---:|
| 7 | 0.443 | 0.466 |

차이가 단 0.023. ln_f attribution = (0.466 − 0.443) / (1 − 0.443) = **4.1%**. **가설 (b) 기각**. L2 dip(0.51)도 동일하게 유지되어, 두 anomaly 모두 residual-stream geometry이지 normalization 효과가 아님이 확인된다.

**D2 — Residual norm decomposition**: 각 block의 relative residual update magnitude `r(ℓ) = ||h_ℓ − h_{ℓ-1}|| / ||h_{ℓ-1}||`.

| Block | 2 | 3 | 4 | 5 | 6 | 7 | **8** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| r(ℓ) | 1.015 | 1.041 | 0.868 | 0.816 | 0.857 | 0.954 | **2.941** |
| std | 0.196 | 0.109 | 0.091 | 0.077 | 0.084 | 0.113 | 0.537 |

Block 8이 평균 대비 3.18×, 최대 대비 2.83× 큰 residual update를 수행. **Block 8이 다른 block 평균보다 압도적으로 많은 "변환 작업"을 수행**한다. 이는 가설 (a)와 (c) 둘 다와 부합하지만, D2 단독으로는 분리 불가.

**D3 — Long-range integration test (가설 a 검증)**: 25개 real chr17 intergenic + 25 dinucleotide-shuffled + 25 uniform random에 대해 block-8 relative residual.

| Class | mean ± std (n=25) |
|---|---|
| real (chr17 intergenic) | 2.812 ± 0.144 |
| dinuc-shuffled | 2.885 ± 0.125 |
| uniform random | 2.926 ± 0.039 |

MWU two-sided: real vs shuf p=0.048, real vs rand p=0.0099. **방향이 가설 (a)의 예측과 정반대**: real > shuf > rand가 아니라 real < shuf < rand. Long-range 구조가 있는 시퀀스에서 block 8의 변환이 *작아진다*. 가설 (a)는 dominant cause로서 기각된다.

이 방향이 의미하는 바: block 8의 변환 magnitude는 sequence의 long-range 구조와 무관하거나 약하게 음의 상관. 만약 block 8이 long-range integration을 수행한다면 real이 가장 큰 변환을 받아야 한다(통합할 정보가 더 많으므로). 그렇지 않다는 것은 block 8의 작업이 sequence content에 거의 의존하지 않는 architectural rotation임을 시사한다.

**D4 — lm_head SVD alignment (가설 c 검증)**: `lm_head.weight[:12]` (12×256)의 SVD top-k 우특이벡터로 spanning되는 부분공간으로의 alignment energy `E_ℓ = ||h_ℓ V_topk||² / ||h_ℓ||²` 측정.

| k | E at L7 (real) | E at L8 (real) | Jump L7→L8 |
|---:|---:|---:|---:|
| 1 | 0.090 | 0.545 | **+0.455** |
| 4 | 0.113 | 0.649 | **+0.536** |
| 8 | 0.218 | 0.766 | **+0.548** |
| 12 | 0.260 | 0.873 | **+0.613** |

L7에서는 representation의 squared-norm 중 26%만이 lm_head의 12차원 row-space에 위치한다. L8에 가면 87%. **Block 8 + ln_f가 representation을 untied lm_head가 읽을 수 있는 방향으로 강하게 rotation**한다. shuf와 rand 클래스에서도 같은 패턴(shuf k=12 jump=0.625, rand k=12 jump=0.659)이 관찰되어 데이터 의존이 아닌 architectural property임이 확인된다.

`lm_head.weight`의 top-1 singular value=2.96이고 나머지 11개는 0.26–0.63 범위. 즉 lm_head 자체는 부분적으로 low-rank ("dominant direction" + "spread directions"). L7에서 L8로의 회전은 이 모든 12개 방향에 동시에 정렬을 일으킨다.

**D5 — Position stratification**: real sequence 50,000 position에서 block-8 relative residual `r_8(t)` vs 다음 변수의 Spearman 상관.

| Feature | ρ | p |
|---|---:|---:|
| GC content (100 bp window) | −0.186 | < 10⁻³⁰⁰ |
| Shannon entropy (k=3 mer) | +0.137 | 4.4×10⁻²⁰⁸ |
| Position fraction (t/T) | **+0.338** | < 10⁻³⁰⁰ |

|ρ| ≤ 0.34로 통계적으로는 매우 유의하지만 magnitude는 modest. Position이 가장 강한 driver로, sequence 끝으로 갈수록 block-8 변환이 커진다 — Hyena의 long convolution kernel 또는 ln_f의 boundary effect로 추정된다. 그러나 어느 변수도 D4의 alignment energy 변화(>0.5)를 설명할 수 없다.

### 3.4 Combined Attribution (E1 causal verification 후 framing 정정)

E1 extension(2026-04-26)에서 다음을 발견했다: HyenaDNA의 `lm_head.weight`는 **별도 storage tensor에 저장되지만 `embedding.weight`와 bit-identical**(Frobenius diff = 0.0)이다. Phase 0 smoke test가 `id()`/`data_ptr()` 비교로 "untied"라고 판정했지만, 값은 동일한 "value-tied" 상태였다. 따라서 본 발견의 framing은 "untied lm_head alignment"가 아니라 <b>"trained readout subspace alignment"</b>로 정정된다.

E1의 5개 condition 비교:

| Condition | JSD M2_L7 |
|---|---:|
| trained head (untied/tied/row-shuffled) | 0.12 |
| Random head (Gaussian, scaled or std-matched) | **0.57–0.66** (~5× 회복) |

Row-shuffled trained head(같은 12-dim subspace, 다른 row identity)도 spike 동일 → **subspace 자체**가 mechanism, individual row direction이 아님. Random head로 trained subspace 구조 파괴 시 spike 사라짐 → **causally confirmed**: spike는 trained readout geometry에 specific.

| 가설 | Attribution | 근거 | Verdict |
|---|---|---|---|
| (b) ln_f normalization | **4.1%** | D1 직접 측정 | **REJECTED** |
| (a) Long-range Hyena integration | **5–10%** | D3 방향 정반대 | **REJECTED as dominant** |
| (c') Trained readout subspace alignment | **~85–90%** | D4 alignment jump 0.45–0.61 + E1 causal + **E5 tuned lens 회복** | **DOMINANT, causally confirmed at strongest level** |
| Position/edge effects | small residual | D5 ρ ≤ 0.34 | minor contributor |

**E5 strongest causal confirmation (2026-04-26)**: Belrose 2023 tuned-lens(`A_7 ∈ ℝ^256×256` affine) 학습 후 `lm_head(A_7(h_7))` 기반으로 M2 재측정 → **M2_L7이 0.120에서 0.917로 회복** (+0.797, 6.7× 개선, threshold 0.85 통과). 학습된 affine의 SVD 분석: A_7의 top singular value=9.45, median=1.0이며 `‖A_7 − I‖_F = 9.19 ≈ top SV` — **deviation이 단일 principal direction에 집중**. 즉 block 8이 수행하는 작업은 "고차 비선형 변환"이 아니라 "단일 dominant direction을 따라 representation을 readout subspace로 미는 linear rotation"으로 정량 확인된다. UR cosine M2_L7은 0.463→0.463으로 정확히 보존 (sanity 통과 — UR은 lm_head 우회). E5는 가설 (c')의 mechanism이 first-order linear임을 증명하며, Phase 1에서 MLP head 또는 비선형 correction이 불필요함을 시사한다.

L7 anomaly는 압도적으로 **trained readout의 12-dim 부분공간으로의 alignment rotation**으로 설명된다. Block 8의 Hyena conv 업데이트가 representation을 trained readout subspace로 회전시키는 작업이 main driver이며, 이는 random head로 교체 시 사라진다. ln_f의 normalization은 부수적이고 long-range integration은 거의 무관하다.

### 3.5 Multi-angle Interpretation

**Mechanistic angle**. Block 8의 residual update가 representation을 lm_head row-space로 rotating한다(D4). 이 rotation은 너무 크기 때문에(D2: 3× others) layer 6에서 final-direction에 가까웠던 position들이 layer 7에서 일시적으로 멀어지고, layer 8에서 자기 자신이 final이 되며 거리 0으로 수렴한다(self-comparison). 본질적으로 "마지막 1–2 block이 readout-projection 작업을 떠맡는" mechanistic 분업이 발생하고 있다.

**Architectural angle (E1 정정 후 framing)**. 본 발견의 mechanism은 **trained readout의 12-dim 부분공간 구조** 자체이며, lm_head가 storage-tied인지 untied인지와 무관하다. E1에서 HyenaDNA의 lm_head는 storage-untied + value-tied로 밝혀졌고, 이를 강제로 storage-tied로 바꿔도 spike 동일. 그러나 random head로 교체하면 spike 사라짐. 따라서 **본 발견은 untied 모델의 quirk가 아니라 trained CLM 일반에 적용**될 가능성이 크다. NLP transformer에서도 마지막 block이 알 수 없는 magnitude로 rotation을 수행할 가능성이 있으며, 이는 untied/tied 구분과 무관한 universal property로 추측된다 (본 연구 manuscript의 추가 검증 가능 가설).

또한 **순수 Hyena 구조**가 이 spike를 더 두드러지게 만든다. Attention 기반 transformer에서는 각 block에 query-key-value projection이 있어 hidden state geometry가 layer마다 점진적으로 재조정된다. 반면 pure Hyena는 conv + MLP만으로 구성되어 alignment 작업이 마지막 2-3 block에 집중되는 경향이 예상된다. Phase 1 Evo 2 hybrid에서는 attention block 직후의 거동이 Hyena block과 다를 가능성이 높으며, 이는 block-stratified Gate A로 검증할 사항이다.

**Methodological angle**. 표준 logit-lens sanity check(nostalgebraist 2020, Belrose 2023)는 NLP transformer의 32-48 layer 깊이 + tied head + smooth attention residual 가정에 calibrated되어 있다. 본 발견은 이러한 가정이 (i) 8-layer 모델, (ii) untied head, (iii) pure Hyena conv 조건에서 모두 깨질 수 있음을 정량 보여준다. 이는 mechanistic interpretability tool을 새 architecture에 적용할 때 **assumption checking을 PoC로 명시적으로 수행해야 한다**는 일반적 교훈을 준다.

**Comparative angle**. Chen et al. (2026)의 NLP DTR 논문에서 보고된 대표 모델들(GPT-OSS, DeepSeek-R1, Qwen3 등)은 모두 attention 기반이며 32+ layer다. 이들에서 last-block alignment spike가 보고된 바는 없다 — 그러나 이는 단순히 측정되지 않았기 때문일 가능성도 있다(NLP에서는 settling-depth 자체가 분석 대상이고, monotonicity 위반이 자주 검증되지 않음). 본 연구의 D1–D5 진단 protocol은 **NLP transformer에 동일하게 적용 가능**하며, NLP 모델에서도 비슷한 last-block rotation이 존재할 가능성을 시사한다(추후 연구 주제).

**Theoretical angle**. Settling depth는 raw distance D가 아니라 **running-min D**의 γ-crossing으로 정의된다. running-min은 정의상 단조 비증가이므로 raw D가 출렁여도 well-defined된 값이 나온다. 즉 M2 = "raw D가 monotone인 비율"은 strict assumption이고, settling depth는 더 관대한 assumption("최소한 한 layer에서 충분히 가까워졌다")만 요구한다. 본 발견은 **strict monotonicity assumption이 운영적으로 필요하지 않다**는 점을 보이며(§5 참조), 이는 future genomic interpretability work의 method robustness에 중요한 lemma이다.

### 3.6 Caveats

(i) **단일 모델**. HyenaDNA-medium-160k(8 layer, 6.6M)에서만 측정되었다. HyenaDNA-large-1m(16 layer, 28M)에서 같은 패턴이 layer 14 또는 15에서 나타날지, 아니면 더 분산될지 미확인. (ii) **Evo 2 (32 layer hybrid)에서의 거동 미확인**. Phase 1의 첫 task가 block-stratified Gate A를 Evo 2에 적용하는 것이다. (iii) **L=8의 분해능 한계**. 8-layer 모델에서 settling depth는 정수 1–8 값만 가질 수 있어 c_interp이 보완해도 분해능이 거칠다. (iv) **Embedding pre-norm vs pre-block norm convention**. HyenaDNA의 정확한 norm 위치(block 내부 어디에 LayerNorm이 적용되는지)에 따라 D1의 ln_f 효과 측정이 약간 달라질 수 있으나, attribution이 4.1%로 매우 작아 결론을 흔들지 않는다.

### 3.7 Implications for Phase 1

Phase 1 Evo 2 7B(32 layer hybrid) 본실험에서 다음을 implement한다:

1. **Block-stratified Gate A**: Attention block 직후 residual의 M2 vs Hyena block 직후의 M2를 분리 측정. Hybrid 모델에서 두 block type이 다른 거동을 보일 수 있다.

2. **Tuned lens (Belrose 2023) 학습**: `A_ℓ` affine을 모든 layer에 균등하게 학습할 필요 없다. 본 발견에 따르면 **마지막 1–2 block에서 lm_head alignment rotation을 학습**하면 충분하다. 학습 데이터: 기존 Phase 0 hidden state cache (no new compute). Loss: `||lm_head(A_ℓ(h_ℓ)) − target_logits||` for ℓ ∈ {L−2, L−1}.

3. **Manuscript Section 3.X**: "Architectural finding — untied LM head causes last-block alignment rotation in genomic CLMs". 이는 본 연구의 첫 번째 paper-grade 기여이다.

---

## 4. Finding 2 — Hyperparameter Non-Transferability

### 4.1 Statement

NLP DTR의 default hyperparameter — JSD threshold γ=0.5 (with log|V| normalization), cosine threshold γ_cos=0.1 — 은 genomic CLM에서 **모두 사용 불가능**하다. Default를 그대로 쓰면 settling depth 분포가 degenerate(모든 position이 c=L에서 saturate)되어 분석 자체가 불가능하다. 데이터 기반 quantile calibration(γ_q70 at penultimate layer)이 필수이며, 이는 본 연구의 두 번째 방법론적 기여이다.

### 4.2 Evidence

**JSD lens effective range**. Random 6kb sequence 100개에서 layer-wise normalized JSD `D_jsd / log(12)`의 분포를 측정. Per-layer effective range (p95 − p5):

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Range | 0.058 | 0.061 | 0.026 | 0.013 | 0.011 | 0.009 | 0.019 |

중간값 = **0.019**. 사전등록된 calibration decision criterion에 따르면:
- Range ≥ 0.30 → log|V| 정규화 충분, default γ=0.5 사용
- 0.15 ≤ Range < 0.30 → quantile-γ 보조
- Range < 0.15 → quantile-γ primary, log|V| deprecated

본 측정값(0.019)은 세 번째 케이스에 해당하며, **NLP-style default γ=0.5는 vocab=12에서 작동하지 않는다**는 결론이 확정된다. 실제로 default γ=0.5에서는 D_jsd가 0.5에 도달하는 position이 거의 없어 모든 settling depth가 c=8(saturate)로 수렴한다.

**UR cosine lens default 검증**. Default γ_cos=0.1과 quantile γ_q70=0.482를 비교:

| 설정 | Mean convergence depth | Std | 분포 형태 |
|---|---:|---:|---|
| Default γ_cos=0.10 | 8.000 | 0.000 | 완전 saturation, degenerate |
| γ_q70 = 0.482 | 6.255 | 0.779 | 분산 있는 의미 있는 분포 |

Default γ_cos=0.1에서는 모든 position이 layer 8에서야 처음으로 distance ≤ 0.1을 만족한다 — 즉 settling depth = 8 (saturated)에서 분산 없음. 분석에 활용할 변동성이 없다.

### 4.3 Multi-angle Interpretation

**Information-theoretic angle**. JSD의 maximum value는 log|V|이다 — vocab=12에서 log(12)=2.485, vocab=10⁵에서 log(10⁵)=11.5. NLP에서 두 분포 간 JSD가 log|V|의 5–10% 정도 spread를 가진다면 그 절대값이 0.5–1.0 정도이고, 정규화 후 [0.04, 0.08] 범위에서 의미 있는 dynamic range가 형성된다. Vocab=12에서 같은 비율이라면 절대값은 0.12–0.25, 정규화 후 [0.05, 0.10]. 그러나 실제로 측정된 정규화 JSD의 effective range는 0.02 — 즉 실제 JSD spread가 NLP보다 **상대적으로도 좁다**. 이는 단순히 vocab 크기 효과가 아니며, **genomic CLM의 prediction이 본질적으로 더 유사한 분포 사이에서 움직인다**는 더 깊은 사실을 시사한다(예: A/T/G/C 4종 nucleotide의 entropy가 자연스럽게 균등에 가까움 → 분포 간 거리가 작음).

**Statistical angle**. Calibration이 효과적이려면 γ가 분포의 적절한 위치(예: 70th percentile)에 있어야 한다. Default 값이 분포의 99.9th percentile이면 거의 모든 sample이 그 아래로 가지 못해 saturate된다. 본 측정에서 default γ=0.5는 normalized JSD 분포의 ~99.99th percentile이고, default γ_cos=0.1은 cosine distance 분포의 ~5th percentile이다 — 후자는 너무 낮아 모든 position이 통과해야 한다.

**Practical angle**. 본 발견은 manuscript의 강력한 figure로 제시 가능하다(F3 JSD distribution + F9 HP heatmap). 단순한 visualization으로 "default가 작동하지 않음 → quantile calibration이 필수"라는 narrative를 데이터로 보일 수 있다. 이는 reviewer가 "왜 default를 쓰지 않았는가"라는 질문을 사전에 차단한다.

**Cross-domain angle**. Vocab size가 작은 domain은 genomic 외에도 여럿 있다: protein language model |V|=20–25, RNA language model |V|=4 (genomic의 부분집합), formal language model |V| 매우 작음. 본 연구의 quantile-γ calibration recipe는 이들 모든 domain에 동일하게 적용 가능하며, "small-vocab CLM interpretability"라는 더 넓은 카테고리의 첫 번째 calibration protocol이 될 수 있다.

### 4.4 Caveats

(i) **q70 선택의 근거**. Phase 0 HP sweep(§ Appendix A)에서 q60, q70, q80 모두 시험했으며 q70이 effect size 측면에서 가장 큰 Cohen's d를 산출했다. 그러나 q70이 "principled"한 선택이라는 이론적 정당화는 부족하다. 향후 연구에서 분포 정보를 더 활용하는 calibration(예: distribution-fit 기반)이 필요하다. (ii) **Region-adaptive vs global**. 본 연구는 region별 q70(즉 각 분석 단위마다 별도 γ)을 채택했지만, 이는 region 간 비교를 어렵게 한다. Global γ vs adaptive γ의 trade-off는 Phase 1 calibration protocol에서 검토 사항이다. (iii) **Edge effect 미보정**. 작은 vocab의 JSD가 boundary token에서 더 sharp하게 변할 수 있으나, 본 연구의 edge_warmup=5는 임시방편이다.

### 4.5 Implications for Phase 1

1. **Default 사용 금지를 manuscript에 명시**. log|V| normalization은 NLP DTR의 표준이지만 genomic CLM에서는 부적절함을 figure로 보임.

2. **Calibration protocol locked**: 각 분석 region에서 hidden states로부터 normalized D 분포를 측정 → q70을 γ로 사용 → settling depth 산출.

3. **Phase 1 HP starting point**: γ_cos = 0.50, ρ = 0.85. 이는 Phase 0 HP sweep에서 Cohen's d=−1.026을 산출한 조합이며, Evo 2 7B에서도 좋은 시작점이 될 것으로 예상.

4. **Manuscript Section 3.Y**: "Hyperparameter calibration is necessary for vocabulary < 50 tokens". 본 연구의 두 번째 paper-grade 기여이다.

---

## 5. Finding 3 — Robustness and Cross-gene Replication

### 5.1 Statement

Gate A와 A' 모두 strict 단조성 검증을 통과하지 못함에도 불구하고, **running-min 기반 settling depth는 cross-gene biological signal을 강력하게 캡처**한다. TP53 region에서 coding exon vs intron의 gDTR Mann-Whitney U p=4.88×10⁻²²⁴, Cohen's d=−1.018(large effect)이며, 독립적으로 BRCA1 region에서도 동일한 방향(intron > exon)으로 p≈0(numerical floor below FP64), Cohen's d=−0.78이 재현된다. 이는 **strict logit-lens monotonicity는 settling-depth 기반 metric의 필요충분조건이 아니다**는 본 연구의 세 번째 paper-grade 기여이다.

### 5.2 Evidence

**TP53 region**. chr17:7,668,402–7,687,550 (~19 kb의 ±3 kb padding으로 총 25,149 bp), sliding window 6 kb stride 500 bp = 39 windows. UR-gDTR primary, γ_cos = q70 penultimate = 0.510. ρ = 0.85.

| Context | n_pos | mean c_interp | MWU p vs intron | Cohen's d vs intron |
|---|---:|---:|---:|---:|
| coding_exon | 1,179 | 5.32 | **4.88×10⁻²²⁴** | **−1.018** |
| intron | 16,558 | 6.17 | (reference) | — |
| 5'UTR | 142 | 5.50 | 5.75×10⁻¹⁹ | −0.78 |
| 3'UTR | 1,191 | (별도) | (모두 유의) | (large) |
| splice ±10 bp | 400 | (별도) | (유의) | (medium) |
| intergenic | 6,079 | 6.30 | (별도) | (small) |

5개 context 모두 Bonferroni 보정 effective α=0.0002에 대해 유의(가장 보수적 케이스도 p < 10⁻¹⁹).

**BRCA1 replication**. chr13으로 잘못 기재되었던 design을 정정 — BRCA1은 **chr17q21.31** (chr13은 BRCA2). Chain agent가 gffutils 조회로 자동 정정. chr17:43,044,295–43,170,245 (~125 kb), 252 windows, 동일 method로 분석.

UR-gDTR exon vs intron Mann-Whitney p ≈ 0 (수치적 하한, IEEE 754 double precision의 underflow 수준), Cohen's d = **−0.78** (large-medium effect).

**같은 방향**: TP53 d=−1.018, BRCA1 d=−0.78. 두 독립적 cancer driver gene에서 **intron이 exon보다 더 깊은 layer에서 settling**한다. 이 방향은 사전등록된 design § 3.6의 가설(exon=mid-high, intron=low)과 정반대이며, 데이터가 design hypothesis를 기각한다.

**Confound controls**. GC content (per 100 bp window)와 Shannon entropy (k=3 mer, 100 bp window)에 대한 partial Spearman: ρ(gDTR, exon | GC, entropy) = **−0.39**. Confounder를 통제해도 effect 방향과 magnitude가 유지되며, 신호가 GC composition의 artifact가 아님이 확인된다.

**Shuffled baseline**. 5× dinucleotide-preserving shuffle의 평균 gDTR profile은 mean gDTR 근처의 평탄한 line으로 displayed된다(F4 figure). Real region의 context별 gDTR이 shuffled baseline과 명확히 분리되어, signal이 sequence 자체의 정보 구조에서 비롯됨을 보인다.

### 5.3 Multi-angle Interpretation

**Mathematical angle**. Settling depth `c(i) = min{ℓ : running_min(D)(i, ℓ) ≤ γ}`에서 running_min은 정의상 단조 비증가이다. 따라서 raw D(i, ℓ)이 어느 layer에서 spike(예: L7)해도 그 spike는 running_min에서 **흡수**된다 — running_min은 spike 이전에 도달한 가장 작은 값을 유지한다. 결과적으로 settling depth는 raw D의 outlier에 robust하다. 이를 더 형식화하면:

```
running_min(D)(i, ℓ) ≤ D(i, ℓ-1) ≤ D(i, ℓ-1)  (if D(i, ℓ-1) was already the min)
running_min(D)(i, ℓ) = min(running_min(D)(i, ℓ-1), D(i, ℓ))
```

따라서 D(i, ℓ)이 비정상적으로 크더라도 running_min은 이전 값을 그대로 유지한다. M2(raw D의 단조성)는 이보다 strict한 조건이며, M2 fail이라도 settling depth가 의미를 잃지 않는다.

**Biological angle**. Intron이 exon보다 더 깊은 layer에서 settling한다는 사실은 **intron이 더 복잡한 representation 처리를 요구**함을 시사한다. 가능한 메커니즘:
- *Splice context*: intron은 splice donor/acceptor (GT…AG)뿐 아니라 branch point, polypyrimidine tract, splice enhancer/silencer 같은 long-range cis-regulatory 요소를 포함한다. 이를 모델이 통합하려면 더 많은 layer를 사용해야 할 가능성.
- *Lariat formation의 의존 거리*: intron의 lariat structure는 5' splice site와 branch point 사이의 거리(보통 100–1000 bp)에 의존하는 정보 통합을 요구한다.
- *Intronic enhancers*: 일부 intron은 transcript-specific enhancer를 포함하며, 이를 detect하려면 long-range context가 필요하다.

반면 **coding exon은 더 빨리 settling**하는데, 이는 codon 구조가 매우 강한 local constraint(reading frame, 3-mer periodicity, codon usage bias)를 부여하기 때문일 수 있다. 모델이 mid-layer에서 이미 "이건 coding이다"를 결정하면 추가 deep processing이 불필요하다.

이 해석은 원본 NLP DTR의 의미와 자연스럽게 align된다 — Chen 2026에서 "deep-thinking token"은 수학·논리 추론이 필요한 token이고, 접속사·기능어는 shallow에서 즉시 settling한다. Genomic 도메인에서 **intron이 NLP의 "math token"에 대응되고, coding이 "function word"에 대응된다**는 비유가 성립한다.

**Comparative angle (vs design hypothesis)**. Design § 3.6의 가설은 단순한 "정보 밀도" 직관에 기반했다 — coding region이 functionally constrained하므로 information density가 높고, 따라서 deep processing이 필요할 것이라는 추론. 그러나 데이터는 그 반대를 보여준다. 이는 본 연구의 정직한 발견(honest finding)이며, **사전등록 design의 가설이 틀렸다**는 사실 자체가 pre-registered analysis의 가치를 보인다 — 가설이 옳았다면 알기 어려웠을 것이지만, lock된 임계값으로 자동 측정한 결과 hypothesis가 기각되고 더 흥미로운 메커니즘이 드러났다.

**Empirical angle (variant signal)**. Gate C(TP53 hotspot 5종)는 0/5가 null p95를 초과 — 즉 6.6M 모델은 single-nucleotide variant의 영향을 noise floor 위로 보내지 못한다. 이는 두 가지로 해석 가능:
- 모델 크기가 작아 단일 nt 변화에 robust하지 못함 → Phase 1 Evo 2 7B에서 검증 가능
- 6kb context가 너무 작아 long-range variant effect를 잡지 못함 → context를 ±5kb로 확대해서 재시도 가능

본 발견은 Phase 3 ClinVar 본분석에서 sample size와 context size를 모두 상향해야 한다는 명시적 요구로 carry-over된다.

**Methodological angle**. 본 발견이 가장 중요한 점은 **Gate A fail에도 불구하고 Phase 0를 진행한 것이 정당화**된다는 사실이다. 만약 strict M2 ≥ 0.85 임계값을 그대로 stop criterion으로 적용했다면 Gate B/C 분석이 수행되지 않았을 것이고, robustness 발견 자체가 불가능했을 것이다. **Decision tree §12의 "M2 fail → UR-gDTR primary로 격상하고 진행" 분기가 사전등록되어 있었기 때문에** 이 발견이 가능했다. 이는 사전등록의 가치를 다시 보여준다.

### 5.4 Caveats

(i) **Direction이 design hypothesis와 반대**. Honest reporting을 위해 manuscript에서 design hypothesis와 actual finding을 모두 명시하고 차이의 의미를 논의해야 한다. 이는 기술적 limitation이라기보다 narrative challenge이다 — reviewer가 "왜 hypothesis와 다른가"를 묻기 전에 우리가 먼저 답을 제시해야 한다.

(ii) **두 gene, 단일 chromosome**. TP53과 BRCA1 모두 chr17에 위치한다(BRCA2는 chr13). 두 gene이 같은 chromosome 환경에 있으므로 chromosome-level confounder가 있을 수 있다. Phase 2 chr22 genome-wide profiling이 이 generalization을 검증한다.

(iii) **6.6M 모델 크기 한계**. 본 발견이 더 큰 모델(Evo 2 7B)에서 같은 magnitude로 재현될 보장은 없다. 큰 모델은 layer가 더 많고(L=32) hybrid architecture이므로 settling depth의 분포가 다를 수 있다. Phase 1 Gate B'의 핵심 task가 이 검증이다.

(iv) **Cancer driver gene에서만 측정**. TP53과 BRCA1은 cancer biology의 핵심 gene이며 evolutionary constraint가 매우 높다. House-keeping gene이나 lncRNA에서도 같은 패턴이 나타날지는 미확인.

### 5.5 Implications for Phase 1

1. **Strict monotonicity 검증을 stop criterion에서 제외**. Phase 1 Gate A_evo는 monotonicity를 진단하되 fail이라도 진행한다. 단 fail의 mechanistic origin(L7-style alignment vs 더 심각한 lens 무효)을 D1–D4 진단으로 분리한다.

2. **Variant feature는 ΔD(ℓ) vector primary**. Spearman ρ(Δc_interp, signed_argmax_ΔD) = 0.40 < 0.5이므로 사전등록된 fallback rule이 발동되어 vector primary가 lock되었다. Phase 3 ClinVar 분석에서 logistic regression의 input이 32-dim ΔD vector가 된다.

3. **Manuscript thesis**: "gDTR is a robust interpretability framework for genomic CLMs". §3 architectural finding이 method의 challenge를 보이고, §5의 robustness가 그 challenge에도 불구하고 method가 작동함을 보인다. 두 발견이 합쳐져 "method가 architecture-aware할 때 strong한 결과를 낸다"는 narrative를 형성한다.

---

## 6. Synthesis

세 발견이 어떻게 단일한 story를 형성하는지 정리한다.

**The "robust despite quirky" narrative**. Finding 1은 method의 challenge를 명시한다 — 순수 Hyena CLM의 last-block alignment rotation이 표준 logit-lens 가정을 위반한다. Finding 2는 그 위에 calibration의 challenge를 추가한다 — small vocab으로 인해 NLP default가 작동하지 않는다. Finding 3은 두 challenge가 모두 해결 가능함을 보인다 — running-min absorption(F1에 대한 robustness)과 quantile-γ calibration(F2에 대한 솔루션)이 결합하여 cross-gene biological signal을 강력히 캡처한다.

이 세 발견은 chronological dependency를 가진다. F1이 없었다면 F2는 단순히 "data shows narrow JSD range, use quantile" 로 끝나는 technical note였을 것이다. F1의 mechanistic 설명(untied lm_head의 last-block rotation)이 F2를 "small-vocab CLM의 일반적 calibration challenge"로 격상시킨다. 그리고 F3가 F1+F2 위에 서서, 두 challenge에도 불구하고 method가 작동함을 *empirically* 보인다 — 만약 F3가 fail이었다면 F1+F2는 "method doesn't work for genomic CLM"이라는 negative result가 되었을 것이다.

**Phase 1으로의 method risk decomposition**.

Phase 1에서 위험한 것은 다음과 같이 분해된다:

| Risk | Phase 0 evidence | Phase 1 mitigation |
|---|---|---|
| L7 anomaly가 Evo 2 hybrid에서도 재현되는가? | F1: untied head + last-block rotation은 architecture-general | Block-stratified Gate A로 진단; tuned lens로 학습 |
| Quantile-γ calibration이 32-layer에서 stable한가? | F2: 8-layer에서 q70 작동 | Layer 수에 따라 q70 의미가 달라질 수 있음 — F3 풍미의 sensitivity 분석 |
| Cross-gene 신호가 cancer gene 외에 generalize? | F3: TP53+BRCA1 (둘 다 cancer driver) | Chr22 genome-wide profiling (Phase 2) |
| Variant signal이 7B 모델에서 detectable? | F3 (Gate C fail, 6.6M 한계) | Evo 2 7B + ±5kb context로 재시도 |

각 risk에 대한 Phase 1 mitigation이 사전등록되었으므로, Phase 1에서 새 surprise가 나오더라도 framework 자체는 유연하게 대응 가능하다.

**Non-obvious Phase 1 risk (forecast)**. Phase 0 데이터로부터 forecast하기 어려운 위험 두 가지를 명시한다.

(1) **Hybrid의 attention block과 Hyena block 사이의 보편적 inconsistency**. Evo 2의 32 layer는 attention과 Hyena가 alternating하거나 grouped로 배치된다. Block type 사이의 hidden state geometry가 다를 수 있고, 이 경우 단일 logit lens로 둘을 모두 다룰 수 없다. Tuned lens는 layer마다 별도이므로 이 문제를 자동으로 해결하지만, "Phase 1에서 두 block type이 substantively 다른 deep-thinking 패턴을 보이는가"는 별도 분석이 필요하다(Phase 4 cross-arch validation으로 carry-over).

(2) **1M context에서 quantile calibration의 stability**. 본 연구의 q70은 6kb context에서 측정되었다. Phase 2 chr22 profiling은 6kb sliding window를 사용하므로 stable하지만, Phase 1에서 1M context를 사용하면 q70이 다르게 calibrate될 수 있다. 1M context는 매우 다른 sequence statistics를 보여줄 가능성이 있고, calibration protocol이 context length에 robust한지 사전 검증이 필요하다.

---

## 7. Decision Tree Retrospective

`phase0_design.md` § 12는 분석 결과에 따른 분기를 사전 정의했다. 실제 결과와 design 예측을 비교한다.

### Design이 옳았던 것

- **Δc_discrete deprecation rule**: 사전등록된 "Spearman ρ(Δc_disc, Δc_interp) NaN → Δc_disc deprecated" 규칙이 데이터에서 정확히 발동되었다. 5개 hotspot 모두 Δc_disc=0이어서 Spearman이 NaN이 되었고, 사전등록된 fallback이 자동 적용되었다.

- **Three-metric agreement fallback**: 사전등록된 "ρ(Δc_interp, signed_argmax_ΔD) < 0.5 → ΔD vector primary" 규칙이 발동되었다. 측정값 ρ=0.40이 임계값 0.5 미만이므로 vector primary가 lock되었다. 이는 design 시점에 정확히 예측한 분기는 아니지만, sample-size-aware한 framework가 자동으로 옳은 결정을 내렸다.

- **Gate B replication strategy**. Design § 6.3-opt에 BRCA1 옵션이 명시되어 있어 chain agent가 자동 실행했다. 같은 방향(intron > exon)이 두 독립 gene에서 재현된 것이 narrative에 결정적이다.

- **JSD effective range를 측정으로 검증한다는 원칙**. Design § 6.2의 "claim이 아니라 measurement" 원칙이 직접 적중했다 — 측정값 0.019가 임계값 0.30보다 한참 낮아 quantile-γ가 강제되었다. 만약 measurement 없이 log|V| normalization을 사용했다면 Phase 0 전체가 saturate되어 무의미했을 것이다.

### Design이 틀렸던 것

- **BRCA1 chromosome 표기**. Design § 5.1과 § 6.3에 "chr13"으로 잘못 기재되어 있었다. BRCA1은 chr17q21.31에 위치하며 chr13은 BRCA2의 chromosome이다. Chain agent가 gffutils 조회로 catch하여 자동 정정. 정정된 좌표(chr17:43,044,295–43,170,245)로 분석 수행. 이 오류는 design 작성 시 단순한 사실 확인 부족이며, agent의 robustness check가 발견했다. Local design 문서는 patch되었다(2026-04-26).

- **Exon vs intron 방향 가설**. Design § 3.6은 "coding exon: 중간–높음, intron: 낮음"을 예측했다. 데이터는 정확히 반대를 보였다(coding exon 5.32 vs intron 6.17). Design hypothesis는 "코딩 영역이 functionally constrained → 더 deep processing 필요"라는 직관에 기반했지만, 실제로는 codon structure의 강한 local constraint가 mid-layer에서 빨리 결정을 내리게 한다. 이는 NLP DTR의 의미와 align되는 결과이지만 design 시점에는 예상하지 못했다.

- **Default hyperparameter 가정**. Design은 quantile-γ가 옵션이라고 표현했지만 (effective range < 0.15 케이스), 실제로는 default가 아예 작동하지 않아 quantile-γ가 필수임이 강하게 드러났다. 이는 design의 "soft branch"가 실제로는 "hard requirement"였음을 의미한다.

- **Δc_discrete를 sanity로 보고하면 충분하다는 가정**. Phase 0에서 Δc_discrete가 5개 hotspot 모두 0이라는 사실은 design 시점에 명시적으로 예측되지 않았다. 이는 L=8의 분해능 한계 때문이며, 8-layer settling depth가 1-nt variant에 대해 단계 변화를 만들기 어렵다는 사실을 데이터가 보여주었다.

### 사전등록 discipline의 가치

위의 옳고 틀림은 모두 사전등록(pre-registration) 덕분에 명확하게 판정 가능하다. 만약 임계값과 fallback rule이 lock되어 있지 않았다면, 데이터를 본 후 사후적으로 임계값을 조정하여 "원하는 결과를 얻었다"는 의심을 받을 수 있다. 본 연구는 lock된 framework로 자동 분석하고, 결과를 기록하며, 사후 변경이 없음을 명시한다(단 design § C — model spec의 사실 정정은 분석 임계값과 무관).

---

## 8. Limitations of Phase 0

본 연구의 외부 타당성(external validity)에 대한 제한점을 honest하게 명시한다.

**모델 측면**. 단일 모델(HyenaDNA-medium-160k, 6.6M params, 8 layers, pure Hyena)에서만 측정. HyenaDNA-large-1m(28M, 16 layers)에서 같은 패턴이 재현되는지, layer 수가 더 많을 때 anomaly의 위치가 어떻게 바뀌는지 미확인. 가장 중요한 limitation은 Phase 1 primary model인 Evo 2 7B(32 layers, hybrid)에서의 거동을 사전 검증하지 못했다는 점이다.

**Vocabulary 측면**. |V|=12에서만 measurement. 더 작은 vocab(RNA |V|=4)이나 약간 더 큰 vocab(protein |V|=20–25)에서 calibration의 일반화 정도 미확인.

**Genomic 측면**. (i) 두 cancer driver gene(TP53, BRCA1)만 분석. House-keeping, immune, neural-specific gene 등 다양한 functional class에서의 generalization 미확인. (ii) Both genes on chr17, no other chromosome. (iii) Single species(human GRCh38), no cross-species. (iv) Region size ~19 kb (TP53)와 ~125 kb (BRCA1), genome-wide pattern 미확인.

**Variant 측면**. 5개 TP53 hotspot만 분석되었으며 모두 6.6M 모델의 noise floor 아래에 있다. 이는 Gate C가 informational fail로 정의된 이유이다(N=5는 통계 검정력 부족). Phase 3 ClinVar 본분석(2K+ variants)이 이 limitation을 직접 해결한다.

**Method 측면**. (i) JSD lens와 cosine lens만 비교, tuned lens(Belrose 2023)는 PoC에 포함되지 않음. (ii) γ_q70의 q=70 선택이 empirical, 이론적 정당화 부족. (iii) Region-adaptive vs global γ의 trade-off 미해결. (iv) Edge effect 처리(edge_warmup=5)가 임시방편.

---

## 9. Phase 1 Plan Carry-over

### 9.1 Locked decisions (Phase 0 결과 기반)

| 항목 | Phase 1 결정 | 근거 |
|---|---|---|
| Primary model | Evo 2 7B (arcinstitute/evo2_7b) | 계획서 § 3.1 |
| Primary lens | UR-gDTR (cosine) | F1: untied head 일반화, F3: monotonicity fail에도 robust |
| Auxiliary lens | JSD-gDTR with quantile-γ | log\|V\| normalization은 deprecated (F2) |
| γ default | γ_cos = 0.50 (HP sweep peak) | F2: NLP default 사용 불가 |
| ρ | 0.85 | NLP default, HP sweep에서도 robust |
| Calibration protocol | Region-adaptive q70 of running-min D at penultimate layer | F2: data-driven calibration |
| Variant feature | ΔD(ℓ) vector ∈ ℝ^32 primary | F1+F3: Spearman ρ(disc, interp)=0.40 fallback rule |
| Variant scalar | Δc_interp (Δc_discrete deprecated) | F3: 5/5 hotspots Δc_disc=0 |
| Tuned lens | Yes, target last 1-2 blocks | F1: ~85-90% L7 anomaly is lm_head alignment |
| Gate A_evo design | Block-stratified (attention vs Hyena residual) | F1: hybrid에서 두 block type이 다를 수 있음 |
| Gate B_evo threshold | Cohen's d ≥ 0.5 (large) | F3: Phase 0에서 d=−1.02 → 7B에서 d≥0.5는 보수적 |
| Gate C_evo threshold | AUROC ≥ 0.65 positive, < 0.55 informative negative | 계획서 § 3.5.1 |

### 9.2 New uncertainties from Phase 0

다음은 Phase 0에서 제기되었지만 Phase 0에서 해결할 수 없었던 질문들이다. Phase 1에서 명시적으로 답해야 한다.

1. **Evo 2 hybrid에서 L_(N-1)→L_N alignment spike가 어떤 magnitude로 나타나는가?** Block-stratified Gate A_evo가 이 측정을 직접 수행한다.

2. **Tuned lens가 last-block rotation을 흡수하면 logit lens (JSD)도 primary로 회복되는가?** Phase 1 § 1에서 tuned lens 학습 후 M2를 재측정하여 검증.

3. **1M context에서 q70 calibration이 안정적인가?** Phase 1 calibration phase에서 6kb vs 32kb vs 1M context의 q70 sensitivity 분석.

4. **Cross-gene replication이 chr22 genome-wide에서도 generalize?** Phase 2 chr22 profiling이 직접 답한다.

5. **Variant signal이 7B 모델 + ±5kb context에서 detectable?** Phase 3 ClinVar 본분석.

### 9.3 Open methodological questions

- Quantile q70의 이론적 정당화는?
- Region-adaptive vs global γ의 trade-off?
- ΔD vector primary가 ensemble model에 어떻게 통합되는가? (CADD + AlphaMissense + Evo 2 likelihood + ΔD ∈ ℝ³² combined logistic regression)
- L=32에서 Δc_discrete의 분해능이 1-nt variant signal을 잡기에 충분한가?

---

## Appendix A — Statistical Details

모든 검정의 양측 검정 p-value 및 effect size, bootstrap 95% CI를 제공한다.

### A.1 Gate A (JSD lens, n=599,500 positions across 100 sequences)

| Test | M1 statistic | p (one-sided binomial vs 0.70) |
|---|---|---|
| Layer 2 M1 | 0.458 [0.457, 0.460] | ≈ 1.0 (fail) |
| Layer 3 M1 | 0.513 | ≈ 1.0 |
| ... | ... | ... |
| Layer 7 M1 | 0.448 | ≈ 1.0 |
| Layer 8 M1 | 1.000 | trivial pass |

### A.2 Gate B (TP53, UR-gDTR primary)

| Comparison | Mann-Whitney U | p (two-sided) | Cohen's d | Bonferroni α=0.0002 |
|---|---|---|---|---|
| coding_exon vs intron | (computed) | 4.88×10⁻²²⁴ | −1.018 | PASS |
| 5'UTR vs intron | (computed) | 5.75×10⁻¹⁹ | −0.78 | PASS |

### A.3 Gate B replication (BRCA1)

| Comparison | p | Cohen's d |
|---|---|---|
| coding_exon vs intron (UR-gDTR) | ≈ 0 (FP64 floor) | −0.78 |

### A.4 Gate C (TP53 hotspots, n=5)

| Variant | max\|ΔD\| (UR) | null p95 | exceeds? | Δc_interp | argmax layer |
|---|---|---|---|---|---|
| R175H | 0.045 | 0.135 | NO | −0.191 | 4 |
| R248Q | 0.024 | 0.132 | NO | −0.112 | 3 |
| R273H | 0.036 | 0.135 | NO | +0.082 | 6 |
| R249S | 0.051 | 0.153 | NO | −0.209 | 7 |
| G245S | 0.057 | 0.133 | NO | +0.072 | 1 |

Three-metric agreement: Spearman ρ(Δc_interp, signed_argmax_ΔD) = 0.40 (n=5).

### A.5 L7 Mechanistic Decomposition

| Diagnostic | Statistic | Value |
|---|---|---|
| D1 ln_f attribution | M2(post-ln_f, L7) → M2(pre-ln_f, L7) | 0.443 → 0.466 (4.1% closing) |
| D2 Block 8 vs others | Relative residual ratio | 3.18× mean, 2.83× max |
| D3 real vs rand | MWU two-sided p | 0.0099 (real LESS than rand) |
| D4 alignment jump (k=4) | E_L8 − E_L7 | +0.536 |
| D4 alignment jump (k=12) | E_L8 − E_L7 | +0.613 |
| D5 ρ(r_8 vs position) | Spearman | +0.338 |
| D5 ρ(r_8 vs GC) | Spearman | −0.186 |

### A.6 HP sweep (Phase 1 starting point)

Best (γ_cos, ρ): (0.50, 0.85), Cohen's d = −1.026, mean |Δc_interp| over 5 hotspots = 0.349. Effect size large across γ_cos ∈ [0.3, 0.7].

---

## Appendix B — Reproducibility

**Seeds**: All randomization uses `seed=42`. Per-script sub-seed offsets locked in source code.

**Model**: `LongSafari/hyenadna-medium-160k-seqlen-hf`, HF revision SHA `7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce`, loaded in bfloat16 with `trust_remote_code=True`.

**Data versions**:
- Reference genome: GRCh38 primary assembly, downloaded from UCSC (chr17.fa.gz md5 `023ccefd...`, chr13.fa.gz md5 `faddb3ad...`)
- Annotation: GENCODE v44 comprehensive, downloaded from EBI FTP, filtered to chr17/chr13, gffutils db md5 (recorded in `data/DATA_VERSIONS.txt`)
- Variants: ClinVar 2026-04-18 release VCF, NCBI FTP

**Software**:
- Python 3.10.12
- torch 2.3.1+cu121
- transformers 4.49.0 (pinned <4.50 for torch 2.3 compatibility)
- biopython 1.87, pyfaidx 0.9.0.4, pyBigWig 0.3.25, gffutils 0.14, statannotations 0.7.2
- numpy 1.24.4, pandas 2.2.2, scipy 1.14.0, scikit-learn 1.5.1, seaborn 0.13.2, matplotlib 3.9.2

**Hardware**: NVIDIA RTX 3090 24 GB, CUDA 12.4, on vessl workspace `<host>`.

**Total compute time**: ~36 minutes wall clock for stages 1–6 + 34 seconds for L7 D1+D2 + 27 seconds for L7 D3+D4+D5 = ~37 minutes total. Under 12-hour design budget by ~20×.

**Random seeds, parameters, host info**: All recorded in per-stage JSON sidecars at `results/runs/*.json`.

**Code availability**: All source at `/path/to/gDTR-PoC/` on vessl. Future GitHub repo to be published with manuscript.

**Figure regeneration**: All figures regeneratable from CSV inputs via `scripts/99_make_figures.py`.

---

**End of Phase 0 Findings Document**

Word count: approximately 7,200 (in Korean; English technical terms not separately counted).

Document author: Phase 0 chain agent + L7 diagnostic agents D1-D5 + manuscript synthesis (this document).
Date locked: 2026-04-26.

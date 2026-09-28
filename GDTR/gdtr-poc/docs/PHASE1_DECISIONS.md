# Phase 1 Decisions — gDTR on Evo 2 7B

**Project**: Genomic Deep-Thinking Ratio (gDTR)
**Phase**: 1 (Method calibration on Evo 2 7B)
**Document version**: 2026-04-26 v1.0
**Status**: Pre-registered Phase 1 plan, locked before Evo 2 forward passes
**Predecessors**: `phase0_design.md` (locked), `PHASE0_DECISION.md` (gate verdicts), `PHASE0_FINDINGS.md` (3 findings + L7 D1-D5 mechanistic decomposition), Phase 0 extensions E1/E2 (currently in execution; this document will be patched on completion)

---

## 0. Status & Inputs

Phase 0 PoC가 HyenaDNA-medium-160k(6.6M, 8 layers, pure Hyena, untied LM head)에서 완전히 종결되었다. 본 문서는 Phase 0의 모든 결과를 binding input으로 받아 Phase 1 Evo 2 7B(32 layers, hybrid Transformer+StripedHyena 2)의 method spec, gate, schedule, risk matrix, decision tree를 사전등록한다. **본 문서가 lock된 후 분석 도중 임계값을 수정하지 않는다**. 사후 변경이 필요하면 `PHASE1_DECISION.md`에 명시 후 영향을 별도 보고한다.

### 0.1 Phase 0로부터 binding된 결정

다음은 Phase 0 데이터(F1+F2+F3 + L7 D1-D5 + E1+E2 extension)에 의해 정당화되었으며 Phase 1에 자동 적용된다:

| 결정 | Phase 0 evidence | Phase 1 status |
|---|---|---|
| Primary lens = UR-gDTR (cosine) | F3: Gate A/A' fail에도 cross-gene d=−1.02/−0.78 | LOCKED |
| Auxiliary lens = JSD-gDTR with quantile-γ | F2: log\|V\| 정규화 deprecated | LOCKED |
| γ default = 0.50 (cosine) | HP sweep peak Cohen's d=−1.026 | Phase 1 starting point |
| ρ = 0.85 | NLP default + HP sweep robust | LOCKED |
| Calibration = regional q70(D_cos at penultimate) | F2: data-driven only choice | LOCKED |
| Variant feature primary = ΔD(ℓ) vector | F1+F3: Spearman ρ=0.40 fallback | LOCKED |
| Variant scalar = Δc_interp | F3: 5/5 hotspot Δc_discrete=0 | LOCKED |
| Δc_discrete | DEPRECATED | not used in Phase 1+ |
| Tuned lens target = last 1-2 blocks | L7 D1-D5 + E1: trained readout subspace alignment, causally confirmed | LOCKED |

### 0.2 E1/E2 extension analyses (완료 2026-04-26, 결과 incorporated)

**E1 — Tied-vs-untied LM head ablation** (5 conditions, 100 sequences, 43초 runtime):

| Condition | JSD M2_L7 | UR M2_L7 |
|---|---:|---:|
| untied trained (original) | 0.1195 | 0.4629 |
| tied (lm_head ← embedding) | 0.1195 | 0.4629 |
| shuffled rows (row-perm of trained) | 0.1229 | 0.4629 |
| scaled random (Gaussian, scale-matched) | **0.6602** | 0.4629 |
| random (std=1/√H) | **0.5653** | 0.4629 |

**Critical correction discovered by E1**: HyenaDNA의 `lm_head.weight`와 `embedding.weight`는 **bit-identical** (Frobenius diff = 0.0). Phase 0 smoke test가 `id()`와 `data_ptr()` 비교로 "untied"라고 판정했지만, 실제로는 별도 storage에 같은 값을 저장한 "value-tied" 상태이다. 본 문서와 PHASE0_FINDINGS.md §3의 "untied" framing은 사실 오류이며 다음과 같이 정정한다.

**Causal claim 정정**: 가설 (c)의 의도(trained readout geometry가 block 8 rotation을 강제)는 causally confirmed되었으나, "untied" 표현은 "trained readout subspace alignment"로 대체된다.

| 정정 전 (사실 오류) | 정정 후 (causal 확인) |
|---|---|
| Untied lm_head이므로 마지막 block이 alignment rotation을 강제 | Trained readout의 12-dim subspace 구조가 block 8 rotation을 강제 |
| Tied head 모델은 spike 없을 것 | Random head 모델은 spike 없음 (E1: 0.57-0.66 vs 0.12) |
| HyenaDNA는 untied | HyenaDNA는 storage-untied + value-tied (jointly trained or post-hoc copy) |

**확인된 causal mechanism**: row-shuffled head(같은 row-space, 다른 row identity)가 spike 동일 → individual row direction이 아닌 **12-dim subspace 자체**가 mechanism. Random head에서 spike 사라짐 → spike는 *trained* readout geometry에 specific.

**일반화**: 본 발견은 tied/untied와 무관하게 **trained CLM 일반에 적용**될 가능성이 크다. Phase 1 Evo 2 hybrid가 storage-tied든 storage-untied든, trained readout의 row-space에 representation을 align하는 작업이 마지막 block에 집중될 것으로 예측.

**E2 — Codon-level gDTR**: NULL result, Phase 1 carry-over.

| Codon position | n | mean c_interp | 95% CI |
|---:|---:|---:|---:|
| 1st | 393 | 5.372 | [5.332, 5.415] |
| 2nd | 393 | 5.299 | [5.263, 5.342] |
| 3rd (wobble) | 393 | 5.294 | [5.246, 5.340] |

Kruskal-Wallis H=4.19, p=0.123 (NS). 1st > 2nd ≈ 3rd 약한 trend가 보이지만 Bonferroni 보정 후 모두 비유의. **Resolution floor**: L=8에서 c_interp 차이 <0.1 layer는 metric의 의미 있는 분해능 아래 (PHASE0_FINDINGS.md §8 limitation에 명시됨).

**Phase 1 implication**: chr22 genome-wide profiling에서 codon position을 first-class axis로 stratify하지 않음. 그러나 Evo 2 32-layer (4× 분해능)에서 exploratory analysis로 재시도. Finding 3 (intron > exon) narrative는 codon structure가 아닌 sequence-context 차이에 의한 것이며, E2 null이 이를 강화한다.

---

## 1. Model & Computational Specification

### 1.1 Primary Model

**Evo 2 7B** (`arcinstitute/evo2_7b`)
- Architecture: hybrid Transformer + StripedHyena 2
- Layers: 32 (alternating attention + Hyena conv blocks)
- Hidden size: 4096
- Vocab: ~512 (single-nucleotide + special tokens; verify exact count in Phase 1 smoke test)
- Max context: 1M bp
- Training data: OpenGenome2 (9.3T tokens)
- License: Apache 2.0

### 1.2 Backup model

**Evo 1 7B** (`togethercomputer/Evo-1.5-1m`)
- Architecture: pure StripedHyena 1
- Layers: 32
- 1M context
- 사용 조건: Evo 2가 H100 cluster 가용성 문제로 막힐 경우만 fallback
- Method spec은 Evo 2와 동일 적용

### 1.3 Cross-Architecture Comparison Models (Phase 4)

| 모델 | Role | Layers | Lens applied |
|---|---|---|---|
| HyenaDNA-medium-160k | Phase 0 PoC reference, scaling control | 8 | UR-gDTR (Phase 0 already done) |
| HyenaDNA-large-1m | Optional Phase 1 sub-experiment | 16 | UR-gDTR (E3 candidate) |
| NT-v2 500M (`InstaDeepAI/nucleotide_transformer_v2_500m_multi_species`) | MLM cross-arch control | 24 | UR-gDTR only |
| DNABERT-2 117M | MLM tokenization control | 12 | UR-gDTR only |

### 1.4 Compute & Budget

본 연구의 Phase 1+은 cloud H100을 사용한다. Phase 0의 vessl(RTX 3090, 자체 자원)와 분리.

| Task | GPU | Time | Cost |
|---|---|---|---|
| Phase 1 method calibration | H100 80GB (Lambda) | ~10 hr | ~$25 |
| Phase 1 tuned lens training (last 2 blocks) | H100 | ~1 hr | ~$3 |
| Phase 2 chr22 genome-wide profiling | H100 | ~30 hr | ~$75 |
| Phase 3 ClinVar variant (10K variants) | H100 | ~20 hr | ~$50 |
| Phase 4 cross-arch (NT, DNABERT-2, HyenaDNA-large) | A100 40GB spot | ~30 hr | ~$30 |
| Buffer (debugging, re-runs, sensitivity) | mixed | ~30 hr | ~$90 |
| **Total** | | **~120 hr** | **~$273** |

이는 Phase 0 design § 3.5.1의 추정과 동일 ($270). Phase 0가 무료(vessl)로 끝나 buffer 사용 안 했으므로 Phase 1 buffer는 그대로 유지.

---

## 2. Lens Hierarchy & Calibration Protocol (locked)

### 2.1 Three-lens system

| Lens | Definition | Status in Phase 1 | Justification |
|---|---|---|---|
| **UR-gDTR (cosine)** | `D_cos(i, ℓ) = 1 − cos_sim(h_ℓ(i), h_L(i))` | **PRIMARY** | F3: Phase 0 cross-gene d=−1.02/−0.78 with this lens |
| **JSD-gDTR with quantile-γ** | `D_jsd(i, ℓ) = JSD(p_ℓ, p_L) / log\|V\|` | Auxiliary | F2: log\|V\| works only marginally; quantile-γ required |
| **Tuned lens (Belrose 2023)** | `D_tuned(i, ℓ) = JSD(softmax(lm_head(A_ℓ(h_ℓ))), p_L)` | **L7-class anomaly absorber** | F1 + L7 D4: lm_head alignment is dominant cause |

세 lens가 같은 forward pass의 hidden states를 후처리만 다르게 사용하므로 추가 forward 비용 없음.

### 2.2 Tuned lens training spec (L7 D1-D5 + E1 causal 결과로 motivated)

L7 mechanistic decomposition (PHASE0_FINDINGS.md §3.3, D4) + E1 causal verification:
- L7→L8 alignment energy jump: +0.55 (top-4) ~ +0.61 (top-12)
- Block 8 relative residual: 3.18× others
- Attribution: ~85-90% **trained readout subspace alignment** (E1 정정 후 framing)
- E1 causal: random lm_head로 교체 시 M2_L7이 0.12 → 0.57-0.66 (~5× 회복) — spike는 trained readout geometry specific

이 발견은 tuned lens 학습을 specifically last 1-2 blocks로 제한하는 것을 정당화한다. 모든 layer에 균등하게 affine을 학습할 필요 없음. E1은 **mechanism이 trained readout geometry에 specific하다는 causal claim**까지 확정.

**Spec (validated end-to-end on HyenaDNA via E5, 2026-04-26)**:
- Target layers: ℓ ∈ {L−1, L−2} = {31, 32} for Evo 2
- Affine: `A_ℓ ∈ ℝ^(d×d)` where d=4096
- Initialize as identity (eye + zero bias) — guarantees pre-training behavior matches untuned lens
- Loss: `MSE(lm_head(A_ℓ(h_ℓ)), out.logits)` over training corpus (200 sequences × 6kb on HyenaDNA; scale up for Evo 2)
- Training data: Phase 1 hidden state cache (no new forwards), training set DISJOINT from evaluation set
- Optimizer: Adam, lr=1e−3
- **Epochs: ≥10** (E5: 5 epochs gave M2_L7=0.824 PARTIAL, 15 epochs gave 0.917 CONFIRMED. 일관된 confirm을 위해 Evo 2도 10-15 epochs 권고)
- Estimated training time on Evo 2: ~30-60 min on H100 (d=4096 → 16M params per affine vs HyenaDNA의 65K params per affine)
- Output: 두 개의 affine matrix를 cache; downstream lens application 시 적용

**E5 prototype validation results (HyenaDNA-medium-160k)**:
- M2_L7 baseline = 0.120
- M2_L7 tuned (15 epochs) = **0.917** (well above 0.85 threshold)
- Training loss drop: 93.96 (identity init) → 0.71 (132× reduction)
- A_7 SVD: top SV=9.45, median=1.0, `‖A_7−I‖_F=9.19 ≈ top SV` — deviation concentrated in single principal direction
- Mechanism is first-order linear (no MLP/non-linear head needed)
- UR M2_L7 unchanged (0.463→0.463) — sanity confirmed

**예측 (Evo 2)**: Phase 1.3에서 tuned lens 학습 후 JSD lens M2_(L−1)=M2_31이 baseline 대비 ≥ 0.85로 회복할 것. E5에서 입증한 first-order linear mechanism이 일반화되면 Evo 2에서도 동일한 패턴 — 만족되지 않으면 hybrid 모델의 attention block 거동이 추가 mechanism을 도입했음을 시사 (Phase 1 추가 분석 필요).

### 2.3 Calibration protocol (region-adaptive)

각 분석 region 또는 sample에서 Phase 1 calibration phase는 다음을 자동 수행:

1. Region에서 random 50 sequence 6kb forward
2. Penultimate layer (L−1=31)의 running-min D_cos 값 분포 측정
3. q70 percentile 계산 → `γ_cos_region`
4. 본 분석에 이 γ_cos 적용

이는 region-specific calibration이며, 본 연구의 두 번째 paper-grade methodological contribution이다 (PHASE0_FINDINGS.md §4).

**Sanity check**: γ_cos_region이 region 간 ±50% 이상으로 변동 시 calibration 자체에 문제 — Phase 1에서 Bonferroni-corrected sensitivity 분석 추가.

### 2.4 Phase 1 starting point hyperparameters

| Parameter | Value | Source |
|---|---|---|
| γ_cos | 0.50 | Phase 0 HP sweep peak |
| ρ | 0.85 | NLP default + Phase 0 robust |
| Edge warm-up | 5 nt | Phase 0 convention |
| Calibration sample size | 50 sequences × 6kb | sufficient for q70 stability |

Hyperparameter sweep은 Phase 1 § 3.3에서 reduced grid (γ_cos ∈ {0.4, 0.5, 0.6}, ρ ∈ {0.8, 0.85, 0.9})로 수행.

---

## 3. Phase 1 Gates (Pre-registered)

세 gate가 사전등록된다. 각 gate의 임계값은 분석 전에 lock.

### 3.1 Gate A_evo — Block-stratified Logit Lens Validity (blocking)

**Rationale**. Phase 0 L7 D1-D5 + E1 causal test는 pure Hyena 모델의 L7→L8 transition이 **trained readout subspace alignment** 때문임을 confirmed. Evo 2는 hybrid이므로 attention block과 Hyena block이 다른 거동을 보일 가능성이 있다. **Phase 1.0 smoke test에서 Evo 2의 lm_head/embedding tying status를 storage-level + value-level 둘 다 explicit 검증** (Phase 0의 storage-only 검증이 부정확했던 점 반영). 사전등록된 가설:

- **H_attn**: Attention block 직후 residual은 NLP transformer-style smooth convergence를 보임 (per-layer M2 ≥ 0.85)
- **H_hyena**: Hyena block 직후 residual은 layer 31에서 alignment spike 보임 (per-layer M2 < 0.70)
- **H_tuned**: Tuned lens 적용 후 양 block type 모두 M2 ≥ 0.85 회복

**Setup**. 100 random 6kb sequence (50 GC-matched + 50 dinuc-shuffled), 모든 32 layer hidden states 추출. Block type을 attention vs Hyena로 분리.

**Pass criteria** (각각 독립):

| Gate | 임계값 | Action on fail |
|---|---|---|
| Gate A_evo_attn | per-block-attn M2 ≥ 0.85 | Phase 1에서 attention block에 별도 fix 필요 |
| Gate A_evo_hyena | per-block-Hyena M2 ≥ 0.85 OR tuned lens 적용 후 ≥ 0.85 | Hyena block 거동이 HyenaDNA와 같음을 인정, tuned lens 사용 |
| Gate A_evo_overall | UR-gDTR M2_global ≥ 0.50 (Phase 0보다 완화) | 강한 fail이면 Phase 2 진입 보류 |

UR-gDTR M2 임계값을 Phase 0의 0.85에서 0.50으로 완화하는 이유: Phase 0 발견에 따라 strict monotonicity가 method 작동을 위한 필요충분조건이 아님이 확인되었으므로, blocking criterion으로는 더 보수적 0.50을 사용. 단 strict monotonicity 위반은 보고되며 mechanistic 해석이 따른다.

### 3.2 Gate B_evo — Genomic Signal at Scale (blocking)

**Rationale**. Phase 0 Gate B에서 TP53/BRCA1 두 gene에서 d=−1.02 / −0.78의 large effect를 cross-replicate했다. Phase 1은 chr22 50Mb genome-wide에서 같은 신호가 generalize되는지 검증한다.

**Setup**. chr22 약 50Mb를 6kb sliding window stride 3kb로 처리 (~16,000 windows). Per-window UR-gDTR 산출. GENCODE v44 annotation 위에 overlay. 모든 chr22 protein-coding gene의 coding exon vs intron 비교.

**Pass criterion**:
- coding exon vs intron Mann-Whitney U two-sided **p < 1×10⁻⁵⁰** (Bonferroni 보정 over 6 contexts: exon, intron, 5'UTR, 3'UTR, splice±10, intergenic → effective α = 1.7×10⁻⁵¹)
- Cohen's d ≥ 0.5 (large)
- Direction: intron > exon (Phase 0와 일치하는 방향이 binding — 다른 방향이면 Phase 0가 cancer-gene-specific bias였다는 증거)

**Decision branches**:

| 결과 | Action |
|---|---|
| Pass + same direction | Phase 1 본실험 진입 |
| Pass + 다른 direction | cancer driver gene bias 가능성 — gene-class 별로 stratified 분석 추가 |
| Fail (p ≥ 10⁻²⁰) | Phase 0의 강한 신호가 6kb context 한계와 관련됨을 시사 — context size 32kb로 확장 후 재시도 |

### 3.3 Gate C_evo — ClinVar Variant Pathogenicity (informational primary, not blocking)

**Rationale**. Phase 0 Gate C는 5 hotspot에서 0/5가 null을 통과했다. 이는 6.6M 모델의 noise floor 한계로 해석되었다. Phase 1은 7B 모델 + ±5kb context로 ClinVar 2K+ variants에서 단일 nt 신호의 검출 가능성을 직접 검증.

**Setup**. ClinVar 2026-04 release. 15 cancer gene (BRCA1, BRCA2, TP53, EGFR, KRAS, BRAF, PIK3CA, APC, MLH1, MSH2, PTEN, RB1, VHL, ATM, PALB2). P/LP ≥ 2 stars (~2,000), B/LB ≥ 2 stars (~3,000), VUS ≥ 1 star (~5,000, 별도 분석).

**Per-variant**: ref forward + alt forward, ±5kb context. Δc_interp + ΔD(ℓ) ∈ ℝ^32 vector + max|ΔD| + signed_argmax_ΔD.

**Pass criteria**:
- **Primary**: ΔD(ℓ) vector logistic regression P/LP vs B/LB AUROC ≥ 0.65 (10-fold CV)
- **Secondary**: scalar Δc_interp AUROC reported alongside; if scalar AUROC ≥ 0.65, scalar primary considered for backward compatibility
- **Ensemble**: CADD + AlphaMissense + Evo 2 likelihood + ΔD vector → joint AUROC. 본 metric의 incremental information 측정

**Decision branches**:

| 결과 | Action |
|---|---|
| Primary AUROC ≥ 0.65 | Positive result, Phase 3 본분석 진입 |
| 0.55 ≤ AUROC < 0.65 | Mixed signal, ensemble 효과 강조 (Phase 3 manuscript framing) |
| AUROC < 0.55 | Informative negative — variant signal은 likelihood가 더 잘 캡처, gDTR은 보완적 angle (e.g., regulatory variant 특화) 시도 |

### 3.4 Gate priority

만약 두 blocking gate(A_evo, B_evo)가 동시에 fail이면 Phase 1 본실험 일시 중단. method 재검토 — 가능한 원인:
- Evo 2의 architecture가 예상보다 다름 (block 배치, normalization 위치 등)
- Tokenization 차이 (Evo 2 uses single-nt with BOS, but special tokens may differ)
- Hidden state 추출 메커니즘 차이 (`output_hidden_states=True`가 Evo 2에서 다른 구조 반환할 수 있음)

이 시나리오 발생 시 Phase 1 § 6 risk matrix의 mitigation을 적용.

---

## 4. Implementation Plan

### 4.1 Phase 1 sub-stages

```
Phase 1.0  Smoke test + architecture verification          1 day
Phase 1.1  Block-stratified Gate A_evo (untuned)            2 days
Phase 1.2  Tuned lens training (last 2 blocks)             1 day
Phase 1.3  Gate A_evo (with tuned lens)                    1 day
Phase 1.4  Calibration: γ_cos region-adaptive protocol     2 days
Phase 1.5  HP sweep (γ_cos × ρ reduced grid)               1 day
Phase 1.6  Gate B_evo: chr22 genome-wide profiling         5 days
Phase 1.7  PHASE1_DECISION.md write-up                     1 day
                                                Total: ~14 days
```

### 4.2 Phase 1.0 — Smoke test (Day 1)

Evo 2의 API quirks를 PoC와 분리해서 사전 검증. Phase 0의 Appendix C 접근법을 그대로 적용:

1. `arcinstitute/evo2_7b` 로드 (HF revision lock)
2. `output_hidden_states=True`로 6kb forward
3. `len(hidden_states)` 확인 — pre-ln_f vs post-ln_f layout
4. `lm_head.weight`와 embedding tied 여부 확인
5. Vocab size + 실제 token id range 확인
6. BOS/EOS 처리 확인
7. Memory profiling: 6kb / 32kb / 256kb context에서 peak VRAM
8. Hidden state dtype 확인 (PoC에서는 mixed float16/bf16이었음)
9. Block type 식별 — 각 layer를 attention vs Hyena로 분류 (모델 config 또는 module class 검사)

**산출물**: `PHASE1_APPENDIX_C.md` — Evo 2 architectural facts (Phase 0 Appendix C와 같은 형식)

### 4.3 Phase 1.1 — Block-stratified Gate A_evo (Day 2-3)

100 sequence forward + 모든 32 layer hidden states 추출. Per-layer M2 (JSD + UR) 산출. Block type별로 분리해서 보고:

```
Layer  Block type   M2_jsd   M2_ur
1      attention    ?        ?
2      attention    ?        ?
3      hyena        ?        ?
...
31     hyena        ?        ?         ← Phase 0 analog의 spike 위치
32     post-ln_f    1.0      1.0
```

### 4.4 Phase 1.2 — Tuned lens training (Day 4)

L7 D4 결과(`PHASE0_FINDINGS.md` §3.3)에 따라 last 1-2 blocks에 affine 학습:

1. Phase 1.1에서 cached hidden states 로드
2. Target = `out.logits` (final prediction)
3. Loss = `MSE(lm_head(A_ℓ(h_ℓ)), out.logits)` averaged over positions
4. Two affine matrices `A_31, A_32` ∈ ℝ^4096×4096 학습
5. Total params: 2 × (4096×4096) = 33.5M = ~120MB checkpoint

학습 후 Phase 1.3에서 적용해 M2 회복 확인.

### 4.5 Phase 1.6 — Gate B_evo chr22 (Day 9-13)

chr22 50Mb를 6kb stride 3kb로 ~16,000 window. Per-window forward + UR-gDTR profile. GENCODE annotation 매핑. Confound controls (GC, entropy). Mann-Whitney U + Bonferroni × 6 contexts.

**E2 결과 incorporate (예정)**: Codon position 1/2/3 stratification 자동 추가 (Phase 0 E2가 codon structure를 detect 확인 시).

### 4.6 산출물

각 sub-stage 산출물:
- Forward outputs cache (per-stage `.npz` files, hidden states + D arrays)
- Per-stage JSON log (`results/runs/`)
- CSV tables (`results/tables/`)
- PDF/PNG figures (`results/figures/`)
- Sub-stage Markdown report

최종 문서: `PHASE1_DECISION.md` (gate verdicts + Phase 2 권고)

---

## 5. Variant Analysis Spec (Phase 3 carry-over)

### 5.1 Locked from Phase 0

- **Primary feature**: ΔD(ℓ) ∈ ℝ^32 vector (Spearman ρ=0.40 fallback rule fired in Phase 0 § Appendix A.4)
- **Scalar summary**: Δc_interp (continuous boundary-interpolated)
- **Δc_discrete**: deprecated (Phase 0 5/5 hotspots all 0)

### 5.2 Phase 3 본분석 design

(Phase 3는 Phase 1 완료 후 본격 시작; 여기서는 Phase 1이 카르오버하는 사전등록 spec만)

- ClinVar 2026-04 release 사용
- 15 cancer gene (Phase 0 design § 3.7.1)
- P/LP ≥ 2 stars (~2,000), B/LB ≥ 2 stars (~3,000), VUS ≥ 1 star (~5,000)
- Context: ±5kb (Phase 0 ±3kb에서 상향, Gate C miss 대응)
- Per-variant: 2 forwards (ref + alt), bf16, peak VRAM ~30GB
- Total compute: 10K variants × 2 forwards × ~5s/forward = ~28 hr

### 5.3 Classifier design

```python
# Phase 3 logistic regression
features = ΔD ∈ ℝ^32                         # primary
covariates = [GC, entropy, position_in_gene, gene_indicator]
target = is_pathogenic (binary)

baseline = LR(CADD, AlphaMissense, Evo2_likelihood)  # established predictors
gdtr_model = LR(features + covariates)               # gDTR alone
ensemble = LR(baseline_features + features)          # joint

report: AUROC, AUPRC for each, ΔAUROC (gdtr_model − baseline)
```

**Pass criterion**: gdtr_model AUROC ≥ 0.65 OR ensemble ΔAUROC ≥ 0.02 (incremental information).

---

## 6. Risk Matrix

| Risk | Likelihood | Impact | Mitigation | Phase 0 evidence |
|---|---|---|---|---|
| Evo 2 architecture가 PoC와 substantively 다름 | Low | High | Phase 1.0 smoke test로 사전 catch | Phase 0 Appendix C 5개 정정사항이 PoC에서 발생했음 — Evo 2도 비슷할 가능성 |
| L7-style spike가 Evo 2 L31에서 너무 커서 tuned lens도 흡수 못 함 | Med | Med | UR-gDTR로 fallback, manuscript narrative 변경 | L7 D1-D5에서 ~85-90% lm_head alignment 확인 → tuned lens가 잡을 수 있는 형태 |
| Hybrid attention vs Hyena block 거동이 너무 달라 single lens로 안 됨 | Med | High | Block-stratified analysis 강제, possibly 두 lens system 병용 | NLP attention은 smooth, Hyena는 spike — Evo 2에서 둘 다 있음 |
| chr22 Gate B_evo가 fail (cancer gene bias) | Low | Med | Gene class별 stratification (housekeeping, immune, neural, etc.) | Phase 0 cancer driver gene 두 개에서 large effect |
| ClinVar Gate C_evo AUROC < 0.55 | Med | Low (informative negative 가능) | Ensemble framing으로 reframing, Phase 3 sample size 상향 | Phase 0 6.6M 모델은 1-nt 신호 검출 못 함; 7B에서 검증 |
| Quantile-γ가 1M context에서 unstable | Low | Med | 6kb / 32kb / 1M context calibration sensitivity 분석 | Phase 0는 6kb 한정 |
| H100 cloud 가용성 불안정 | Med | Low | Evo 1 7B로 fallback, Phase 1 sub-stage scheduling을 spot-friendly로 | $270 예산 안에 buffer 포함 |

---

## 7. Decision Tree (locked for Phase 1)

```
Phase 1.0 smoke test
├── Critical mismatch (e.g., hidden_states layout, BOS handling, dtype) → patch and retry
└── OK → Phase 1.1

Phase 1.1 Gate A_evo untuned
├── M2_global ≥ 0.50 across both block types → Phase 1.2 (proceed)
├── M2_global ∈ [0.30, 0.50) → tuned lens 더 절실, Phase 1.2 진입
└── M2_global < 0.30 → 모델 사용 적합성 재검토, Evo 1 fallback 검토

Phase 1.3 Gate A_evo with tuned lens
├── M2 회복 (≥ 0.85) → 가설 (c) causally confirmed, JSD lens primary 가능 (UR과 함께)
├── M2 부분 회복 (0.50–0.85) → 가설 (c) 부분 확인, UR primary 유지
└── M2 회복 안 됨 → 가설 (c) 부분 기각, mechanistic 추가 분석 (E1 결과 중요)

Phase 1.6 Gate B_evo chr22
├── PASS (p < 1e-50, d ≥ 0.5, intron > exon) → Phase 2 → 3 진입
├── PASS but 다른 direction → cancer-gene bias 분석 (gene class stratification)
├── Weak (p ∈ [1e-20, 1e-50]) → context size 32kb로 확장 후 재시도
└── FAIL (p > 1e-20) → Phase 2 보류, method 재검토

Gate C_evo (informational, not blocking)
├── AUROC ≥ 0.65 → Phase 3 본분석 진입
├── 0.55 ≤ AUROC < 0.65 → ensemble framing으로 Phase 3 진행
└── AUROC < 0.55 → variant 분석은 ΔD profile 기반 reframe (e.g., regulatory variant focus)
```

---

## 8. Open Questions (Phase 1이 답해야 함)

Phase 0가 제기했지만 답할 수 없었던 질문들:

1. **Evo 2 hybrid의 last-block alignment spike**: 32-layer hybrid에서 L31→L32에서 같은 magnitude의 spike가 나타나는가? Attention vs Hyena block이 다른가?

2. **Tuned lens 회복 효과**: tuned lens 적용 후 JSD lens M2가 0.85 이상으로 회복되는가? (가설 c의 강한 causal test)

3. **Calibration stability across context sizes**: 6kb, 32kb, 1M context에서 q70 γ_cos가 stable한가?

4. **Cross-gene generalization**: chr22 genome-wide에서 cancer driver gene의 패턴(intron > exon)이 다른 gene class에도 적용되는가?

5. **Variant signal at 7B scale**: ClinVar 2K+ variants에서 ΔD(ℓ) vector AUROC ≥ 0.65를 달성하는가?

6. **Block-type interaction**: hybrid에서 attention block 직후 vs Hyena block 직후의 settling pattern이 systematic하게 다른가?

7. **Tied vs untied 일반화 (E1 결과로 부분 답)**: 만약 E1이 가설 (c)를 confirm했다면, untied head는 모든 modern genomic CLM의 공통 issue. Tied embedding을 갖는 모델로 비교 가능?

---

## 9. Phase 1 Schedule

| Day | Activity | Responsibility |
|---|---|---|
| Day 1 | Phase 1.0 smoke test, PHASE1_APPENDIX_C.md | 본인 |
| Day 2-3 | Phase 1.1 Gate A_evo untuned, block-stratified analysis | agent + 본인 review |
| Day 4 | Phase 1.2 tuned lens training | agent |
| Day 5 | Phase 1.3 Gate A_evo with tuned lens, hypothesis (c) causal verdict | 본인 |
| Day 6-7 | Phase 1.4 calibration protocol, region-adaptive q70 | agent |
| Day 8 | Phase 1.5 HP sweep (reduced grid) | agent |
| Day 9-13 | Phase 1.6 chr22 Gate B_evo profiling | agent |
| Day 14 | Phase 1.7 PHASE1_DECISION.md write-up + Phase 2 권고 | 본인 + agent |

기간: 2 weeks. Phase 0의 1-week PoC 대비 2배. Phase 0가 method validation이었다면 Phase 1은 method scaling.

---

## 10. Manuscript Sections Carrying Forward

본 연구의 manuscript 구조와 Phase 1 결과 매핑:

| Manuscript section | Phase 0 contribution | Phase 1 contribution |
|---|---|---|
| Introduction | Background + DTR 전이 동기 | Evo 2 7B 활용 정당화 |
| **Methods § Architectural finding** | F1: Layer 7 anomaly + L7 D1-D5 | Evo 2 hybrid에서 동일 패턴 검증 (Gate A_evo) |
| **Methods § Calibration contribution** | F2: quantile-γ recipe | Evo 2 vocab 더 큰 (~512) → log\|V\| normalization 재검토 |
| **Methods § Tuned lens** | (Phase 0 motivated) | tuned lens 학습 + 회복 결과 |
| **Results § Cross-gene replication** | F3: TP53/BRCA1 d=−1.02/−0.78 | chr22 genome-wide 복제 |
| **Results § Variant interpretation** | (Phase 0 informational fail) | ClinVar 2K+ AUROC |
| Results § Ensemble | (none) | CADD + AlphaMissense + Evo2 likelihood + ΔD ensemble |
| Discussion § Limitations | Phase 0 limitations | Phase 1 limitations |

---

## 11. Reproducibility (Phase 1 carry-over)

- **Seed**: seed=42 (모든 stage)
- **Model**: HF revision SHA lock for `arcinstitute/evo2_7b` — Phase 1.0에서 record
- **Data**: GRCh38, GENCODE v44, ClinVar release 2026-04 (Phase 0와 동일)
- **Software**: PyTorch 2.3+, transformers >= 4.41 (Phase 0 lock 4.49.0; Phase 1에서 Evo 2 호환성 재검증)
- **Hardware**: Lambda H100 80GB; instance type lock TBD
- **Code**: GitHub repo (Phase 1 시작 전 publish)

---

## 12. Status Summary (2026-04-26)

**Phase 0**: ✅ Fully complete
- 8 stages + L7 D1-D5 mechanistic decomposition
- PHASE0_DECISION.md (266 lines) + PHASE0_FINDINGS.md (6,419 words)
- Total compute: ~37 minutes wall, $0 cost (vessl)
- 3 paper-grade findings locked

**Phase 0 Extensions** (in progress):
- E1 — Tied-vs-untied head ablation (background agent)
- E2 — Codon-level gDTR (background agent)
- 결과 수령 시 본 문서 § 1.2 patch

**Phase 1**: 사전등록 완료 (본 문서)
- 시작 조건: E1/E2 결과 incorporate + Lambda H100 인스턴스 lock + 본 문서 final review
- 예상 시작: 2026-04-28 또는 E1/E2 완료 직후
- 예상 종료: 2026-05-12 (2-week schedule)

---

**End of Phase 1 Decisions Document**

Document author: Phase 0 chain agent results synthesis + L7 D1-D5 diagnostic agents + manuscript planning (this document).
Date locked: 2026-04-26.
Update history: v1.0 initial.

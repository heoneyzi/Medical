# Phase 0 Design — gDTR PoC on HyenaDNA

**Project**: Genomic Deep-Thinking Ratio (gDTR) — Causal Genomic Foundation Model의 Layer-wise Prediction Convergence 분석
**Phase**: 0 (Proof of Concept, Week 1–3)
**Document version**: 2026-04-26 v1.0
**Status**: Pre-registered analysis plan (lock before data inspection)

---

## 1. Objective

원본 DTR(Chen et al. 2026)을 genomic CLM에 처음 적용하기 전, **HyenaDNA-medium-160k (6.6M, 8 layers, pure Hyena)** 를 사용하여 (i) logit-lens 기반 layer-wise prediction convergence 측정 파이프라인을 구축하고, (ii) Hyena conv block에 대한 logit lens의 거동을 검증하며, (iii) gDTR signal이 random baseline 대비 비자명한 genomic 구조를 잡는지를 사전등록된 gate 기준으로 fail-fast 판정한다. 본 단계의 산출은 Phase 1 (Evo 2) 본실험의 primary method 선택을 binding한다.

## 2. Pre-registration Statement

본 문서는 **데이터를 분석하기 전에 lock**한다. 다음 항목은 분석 시점에 변경하지 않는다:
- Gate A/B/C의 정량 임계값 (§3)
- Primary metric 선정 규칙 (§4.4)
- 통계 검정 방법 및 다중검정 보정 (§8)
- 시각화 figure spec (§7)
- 결정 트리 (§12)

분석 도중 임계값을 사후 조정하면 그 사실을 `PHASE0_DECISION.md`에 명시하고 그 영향을 별도로 보고한다.

---

## 3. Gate Structure (3 independent gates)

### 3.1 Gate A — Logit Lens Validity on Hyena conv blocks (blocking)

**Rationale**. HyenaDNA는 attention block이 없는 pure Hyena 모델이다. 본 gate는 "Hyena conv block 직후 residual에 logit lens를 적용했을 때 의미 있는 신호를 얻는가"를 직접적으로 검증한다. 이 결과는 Phase 1 Evo 2 hybrid의 Hyena 부분에 logit lens를 사용할지, tuned lens(Belrose 2023)로 대체할지, UR-gDTR을 primary로 격상할지를 결정한다.

**Setup**. GC-content & dinucleotide-frequency matched random 6 kb sequence × 100, 각 forward에서 모든 8 layer의 hidden state 획득.

**Two metrics**:
- **M1 (Top-1 monotonicity rate)**: 한 position에서 layer를 따라 top-1 token이 final layer의 top-1과 일치하기 시작한 후 다시 이탈하지 않는 position의 비율. NLP의 logit-lens 표준 sanity check (nostalgebraist 2020).
- **M2 (JSD running-min monotonicity rate)**: layer-wise JSD가 (running min 의미에서) 단조 비증가인 position의 비율. DTR의 settling-depth 정의가 이 가정 위에서 작동.

**Pass criteria** (모두 충족시 pass):
- M1 ≥ 0.80 (전체 평균)
- M2 ≥ 0.85 (전체 평균)
- Per-layer breakdown에서 모든 layer가 M2 ≥ 0.70

**Decision branches**:
| 결과 | Phase 1 영향 |
|---|---|
| **Pass** | Evo 2의 logit-lens는 attention/Hyena 모두 무난할 가능성 → 전 residual stream 사용 (default) |
| **Fail (M1 < 0.80 만)** | top-1은 변동이 있으나 분포 수렴은 안정 → JSD-DTR은 유효, top-1 기반 보조 분석 제외 |
| **Fail (M2 < 0.85)** | logit lens 자체가 noisy → Phase 1에서 (a) tuned lens 학습, (b) UR-gDTR primary, 둘 중 택일. 본 phase 0에서는 UR-gDTR로 모든 후속 분석 재산출하여 우회 가능성 사전 검증 |
| **부분 pass (특정 layer만 fail)** | 해당 layer를 logit lens 분석에서 제외하고 나머지로 진행 |

### 3.2 Gate B — Genomic Signal (blocking)

**Rationale**. gDTR이 genomic 구조와 연결됨을 보이는 가장 약한 형태의 검증. Pass하지 못하면 방법 자체가 random sequence와 구분되는 신호를 만들지 못한다는 의미.

**Setup**. TP53 region (chr17:7,668,402–7,687,550, GRCh38, ~19 kb) 전체에 sliding window (window=6 kb, stride=500 bp). 각 window를 HyenaDNA에 forward → 가운데 1 kb의 single-nt gDTR profile 산출. Stitching으로 region-wide profile 완성. GENCODE v44 annotation (canonical transcript ENST00000269305) 위에 overlay.

**Pass criterion**:
- **Primary**: gDTR(coding exon) vs gDTR(intron) Mann-Whitney U test, two-sided, *p* < 0.001 (Bonferroni 보정 over {exon, intron, 5'UTR, 3'UTR, intergenic}, 즉 effective α = 0.0001)
- **Secondary** (informational, gate에 포함되지 않음): visible transition at known splice site (donor/acceptor ±10 bp), shuffled-sequence baseline 대비 |Cohen's d| > 0.5

**Decision branches**:
| 결과 | Phase 1 영향 |
|---|---|
| **Pass** | gene structure를 잡는 signal 존재 → Evo 2로 scaling-up 정당화 |
| **Fail (p ≥ 0.001 but < 0.05)** | weak signal — UR-gDTR로 재산출하여 신호가 vocabulary projection의 손실인지 확인. Phase 1 진입 시 sample size 상향 |
| **Fail (p ≥ 0.05)** | 모델 크기(6.6M)가 부족하거나 logit lens가 부적절. HyenaDNA-large-1m (28M)으로 재시도 → 그래도 fail이면 informative negative로 보고 |

### 3.3 Gate C — Variant Signal (informational, non-blocking)

**Rationale**. TP53 hotspot 5종은 통계적 검정력이 부족하므로 차단 gate로는 부적절. 그러나 Phase 3 sample size 산정과 ΔD(l) heatmap의 시각적 해석에 필수.

**Setup**. TP53 hotspot 5: R175H, R248Q, R273H, R249S, G245S. ClinVar 2026-03 release에서 좌표·alleles 확정. 각 variant ±3 kb context로 ref/alt forward → Δ-metric 산출.

**Reported metrics** (모두 보고, gate criterion은 informational):
- |Δc_discrete| (이산), Δc_interp (연속), max|ΔD(l)| (vector summary)
- Layer-wise ΔD(l) profile (variant × layer matrix → primary visualization)

**Informational thresholds**:
- 5종 중 ≥3종에서 max|ΔD(l)| > 95th percentile of shuffled-control distribution
- Three-metric Spearman ρ ≥ 0.5 → scalar metric 채택 가능; ρ < 0.5 → ΔD(l) profile primary

---

## 4. Methodology

### 4.1 Model & Tokenization

**Primary**: `LongSafari/hyenadna-medium-160k-seqlen-hf` (HuggingFace).
- Params: 6.6M
- Layers: 8 (모두 Hyena conv block, attention 없음)
- Hidden: 256
- Vocab: character-level {A, C, G, T, N, +special}, |V| = 12 (실제 token id 확인 필요)
- Max context: 160 kb
- Tied LM head weight: 확인 필요 (smoke test에서 검증)

**Backup** (1M context 필요 시): `LongSafari/hyenadna-large-1m-seqlen-hf` (28M, 16 layers).

**Tokenization**. Single-nucleotide. position i의 input token = 해당 위치의 nucleotide. Context는 input_ids로 변환되고 첫 special token (BOS) offset 보정.

### 4.2 gDTR Computation (Primary signal — JSD lens)

원본 DTR(Chen 2026) 공식을 변형 없이 적용.

```
1. forward(x) → hidden_states: tuple of L+1 tensors of shape (1, T, H)
   (output_hidden_states=True, captured layer 0 is embedding, 1..L are layer outputs)
2. for each layer ℓ ∈ {1, ..., L}:
   3. h_ℓ = hidden_states[ℓ]                  # (1, T, H)
   4. h_ℓ_norm = final_layer_norm(h_ℓ)        # apply model's final norm
   5. logits_ℓ = h_ℓ_norm @ W_U.T             # tied weight
   6. p_ℓ = softmax(logits_ℓ, dim=-1)         # (1, T, V)
7. for position i:
   8. D(i, ℓ) = JSD(p(i, ℓ), p(i, L)) for ℓ ∈ 1..L
   9. D_normalized(i, ℓ) = D(i, ℓ) / log(|V|)  # to [0, 1]
   10. running_min(i, ℓ) = min_{k ≤ ℓ} D_normalized(i, k)
   11. settling_depth_discrete c(i) = min{ℓ : running_min(i, ℓ) ≤ γ}
                                    (or L if never crossed)
   12. settling_depth_interp c_interp(i):
       - 만약 running_min이 ℓ과 ℓ+1 사이에서 γ를 가로지름:
         c_interp(i) = ℓ + (D_normalized(i, ℓ) − γ) / (D_normalized(i, ℓ) − D_normalized(i, ℓ+1))
       - never crossed: c_interp(i) = L (saturated marker)
   13. deep_thinking(i) = 1 if c(i) > ρ · L else 0
14. gDTR(seq) = mean_i deep_thinking(i)
```

**Hyperparameters (default for HyenaDNA, L=8)**:
- γ = 0.5 (NLP DTR default; calibrated empirically in §6.2)
- ρ = 0.85 → deep regime is c > 6.8, i.e., c ∈ {7, 8}
- Edge handling: position의 first 5 token (HyenaDNA의 receptive-field warm-up)은 분석에서 제외

**Vocab size warning**. |V| ≈ 12 → log|V| ≈ 2.48. NLP는 log|V| ≈ 11.5 (vocab 100K). 정규화 후에도 effective dynamic range가 NLP와 다를 수 있음 → §6.2에서 직접 측정.

### 4.3 UR-gDTR (Auxiliary signal — cosine lens)

Vocabulary projection을 우회하는 보조 신호.

```
D_cos(i, ℓ) = 1 − cos_sim(h_ℓ(i), h_L(i))
```

이후 settling depth와 gDTR 정의는 §4.2와 동일하되, γ는 cosine distance 분포에서 별도로 calibrate (default 시도값: γ_cos = 0.1).

**용도**:
1. JSD-gDTR과의 cross-check (Spearman ρ)
2. Gate A fail 시 fallback primary signal
3. Phase 4에서 MLM 비교군의 일관 비교 신호

### 4.4 Δ-metrics for Variant Analysis

세 metric을 모두 산출하고 보고. **Primary 결정은 §6.4의 결과에 따라 사후 결정** (단, 결정 규칙은 본 문서에서 lock).

| Metric | 정의 | 성질 |
|---|---|---|
| `Δc_discrete` | c_alt − c_ref (이산 정수) | γ 근방에서 불연속, 해석 단순 |
| `Δc_interp` | c_interp_alt − c_interp_ref | γ 근방에서 연속, 보간 가정 의존 |
| `ΔD(ℓ)` | D_alt(ℓ) − D_ref(ℓ) for ℓ ∈ 1..L | vector-valued, 정보 손실 없음 |
| `max\|ΔD\|` | max_ℓ \|ΔD(ℓ)\| | scalar summary, sign 잃음 |
| `signed_argmax_ΔD` | ΔD(argmax_ℓ \|ΔD(ℓ)\|) | scalar summary, sign 보존 |

**Primary 선정 규칙**:
- Spearman ρ(Δc_interp, signed_argmax_ΔD) ≥ 0.7 → `Δc_interp`를 scalar primary
- 0.5 ≤ ρ < 0.7 → `signed_argmax_ΔD`를 scalar primary (ΔD profile에 더 가까움)
- ρ < 0.5 → ΔD(ℓ) profile 자체를 feature vector로 사용. Phase 3 classifier는 vector input 채택

**`Δc_discrete`는 모든 경우에 sanity 보조로만 보고.**

---

## 5. Data

### 5.1 Reference Genome
- **GRCh38** primary assembly, UCSC `hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/`
- chr17 (~83 Mb, ~80 MB gzipped) — TP53, BRCA1; chr13 (~114 Mb, ~110 MB gzipped) — BRCA2 (BRCA1은 chr17q21.31에 위치, design 초안의 chr13 표기는 오류)

### 5.2 Annotation
- **GENCODE v44** comprehensive gene annotation, EBI FTP
- chr17, chr13 subset만 추출 (gffutils로 indexed)
- Canonical transcript: TP53 = ENST00000269305, BRCA1 = ENST00000357654

### 5.3 Variant Set
- **ClinVar** 2026-03 release, NCBI FTP `vcf_GRCh38/clinvar.vcf.gz`
- TP53 hotspot 5종 (좌표 GRCh38, ClinVar ID 별도 기록):
  - R175H: c.524G>A, p.Arg175His
  - R248Q: c.743G>A, p.Arg248Gln
  - R273H: c.818G>A, p.Arg273His
  - R249S: c.747G>T, p.Arg249Ser
  - G245S: c.733G>A, p.Gly245Ser
- 좌표·allele은 `data/variants/tp53_hotspots.tsv`에 명시 후 lock

### 5.4 Random Sequence Baseline
- **GC-matched random sequence**: chr22 intergenic region(GRCh38, RepeatMasker로 repeat 제거)에서 6 kb window 100개 추출 + 각 window의 dinucleotide composition을 보존하는 shuffled control 생성 (uShuffle algorithm).
- **Pure-random sequence**: 길이별 (6 kb × 100) i.i.d. uniform draw from {A,C,G,T}, GC=0.5. (sanity 절대 baseline용)

---

## 6. Analysis Plan (Pre-registered)

### 6.1 Sanity Check (Gate A) — `01_sanity_check.py`

**Input**: random seq 100개 (GC-matched + dinucleotide-shuffled). HyenaDNA forward × hidden states.

**Outputs**:
1. `tables/sanity_M1_M2.csv` — per-layer M1, M2 with bootstrap 95% CI (1000 resamples)
2. `tables/sanity_per_position_breakdown.csv` — top-1 trajectory category counts (always-correct, late-converge, oscillate, never-converge)
3. `figures/F2_sanity.pdf` — see §7

**Statistical test**: 각 layer의 M2가 0.70 (per-layer threshold)을 넘는지 one-sided binomial test (n=100×6000≈600K positions), Bonferroni over 8 layers.

**Gate decision**: §3.1.

### 6.2 JSD Distribution Characterization — `01b_jsd_distribution.py`

**Rationale**. Vocab |V|=12에서 JSD의 effective dynamic range를 직접 측정. 결과는 Phase 1 calibration의 입력.

**Outputs**:
1. `tables/jsd_stats.csv` — per-layer (mean, median, p5, p25, p75, p95, p99) of D_normalized
2. `figures/F3_jsd_distribution.pdf` — see §7

**Decision criterion (Phase 1 calibration)**:
- p95 − p5 (effective range) ≥ 0.30 → log|V| 정규화로 충분, γ=0.5 그대로 사용
- 0.15 ≤ range < 0.30 → quantile-based γ를 보조로 추가 보고 (γ_q = 70th percentile of running-min D_normalized at layer L)
- range < 0.15 → log|V| 정규화 부적절. Phase 1에서 quantile-based γ를 primary로 사용 (방법론 contribution)

이 결정은 Phase 0에서는 적용하지 않음 (분포만 측정·보고). Phase 0의 gate B/C는 default γ=0.5로 산출.

### 6.3 Gene Structure Profiling (Gate B) — `02_gene_structure.py`

**Input**: TP53 region (~19 kb) sliding window (window=6 kb, stride=500 bp, ~26 windows).

**Per-window**: HyenaDNA forward → 가운데 1 kb의 per-position settling depth & deep-thinking flag. Stitching 시 overlap region은 평균.

**Annotation overlay**: GENCODE v44 canonical TP53 (ENST00000269305)의 coding exon, intron, 5'UTR, 3'UTR, splice-donor/acceptor (±10 bp).

**Confound controls**:
- GC content (per 100-bp sliding)
- Shannon entropy (k=3 mer, 100-bp window)
- Shuffled sequence baseline: 동일 region의 dinucleotide-preserving shuffle 5회 → mean ± SD

**Outputs**:
1. `tables/gdtr_by_context.csv` — per-context (mean, median, n) gDTR with Mann-Whitney U vs intron
2. `tables/gdtr_vs_confounders.csv` — partial Spearman ρ(gDTR, context | GC, entropy)
3. `figures/F4_tp53_profile.pdf`, `F6_context_boxplot.pdf` — see §7

**Statistical test**: Mann-Whitney U (Gate B primary). Bonferroni over 5 contexts. Permutation test (100 shuffles) for context label permutation.

**Optional 확장 (Gate B pass 시)**: BRCA1 region **chr17:43,044,295–43,170,245** (~125 kb, 정정: 초안에서 chr13으로 잘못 기재되었음 — BRCA1은 chr17q21.31). HyenaDNA-large-1m로 segmented forward + stitching. `figures/F5_brca1_profile.pdf`.

### 6.4 Variant Pilot (Gate C) — `03_variant_pilot.py`

**Input**: TP53 hotspot 5종, 각 ±3 kb context.

**Per-variant**:
1. ref forward → c_ref_disc, c_ref_interp, D_ref(ℓ)
2. alt forward → c_alt_disc, c_alt_interp, D_alt(ℓ)
3. Δc_discrete, Δc_interp, ΔD(ℓ), max|ΔD|, signed_argmax_ΔD
4. variant ±50 bp gDTR profile (영향 전파 측정)

**Null distribution**: 같은 context에서 variant position을 ±100 bp 안에서 random 위치로 옮기고 random allele 변경 → 100회 sampling. Δ-metric의 null dist 산출.

**Outputs**:
1. `tables/tp53_hotspot_metrics.csv` — variant × {Δc_discrete, Δc_interp, max|ΔD|, signed_argmax_ΔD, percentile_vs_null}
2. `tables/metric_agreement.csv` — three-metric Spearman ρ matrix (n=5, 단순 보고)
3. `figures/F7_delta_jsd_heatmap.pdf` — primary fig
4. `figures/F8_three_metric_agreement.pdf` — see §7

### 6.5 UR-gDTR Cross-check — integrated in 6.1, 6.3, 6.4

각 분석을 UR-gDTR로 재산출. JSD-gDTR과의 Spearman ρ를 보고. ρ ≥ 0.7이면 두 신호 일관, < 0.4이면 vocabulary projection이 정보를 잃거나 추가하는 것.

### 6.6 Hyperparameter Sensitivity — `04_hp_sweep.py`

γ ∈ {0.25, 0.4, 0.5, 0.6, 0.75}, ρ ∈ {0.7, 0.8, 0.85, 0.9}.

**Per (γ, ρ)**: gDTR(coding exon) vs gDTR(intron)의 Cohen's d 산출. TP53 hotspot 5종의 |Δc_interp| 평균.

**Output**: `figures/F9_hp_heatmap.pdf` — see §7. Phase 1으로 carry-over할 (γ, ρ) 권고.

---

## 7. Visualization Spec (Publication-ready)

모든 figure는 **vector PDF + PNG (300 DPI)** 동시 산출. 색맹친화 palette (Wong 2011 또는 viridis). 모든 axis label은 단위 포함. Statistical annotation (asterisks, p-values)은 `statannotations` 사용.

**Figure size convention**: single column = 89 mm, double column = 183 mm width. Font: Helvetica/Arial 8pt body, 10pt panel labels.

### F1 — Method Schematic (manuscript Figure 1)
- 3 panel: (a) HyenaDNA forward + hook diagram, (b) JSD trajectory plot (illustrative single position), (c) gDTR aggregation
- 단일 column, 90 × 70 mm
- **Verification**: cartoon이지만 layer 수·hidden state shape이 실제와 일치

### F2 — Sanity Check (Gate A primary fig)
- 3 panel:
  - (a) M1 (top-1 monotonicity rate) per-layer, error bars = bootstrap 95% CI. y축 [0,1]. 임계선 0.80 점선.
  - (b) M2 (JSD running-min monotonicity) per-layer, 동일 양식. 임계선 0.85.
  - (c) Per-position trajectory category stacked bar (always-correct / late-converge / oscillate / never-converge), per-layer.
- Double column, 180 × 60 mm
- **Annotation**: pass/fail label per layer (■ pass, □ fail)
- **Verification**: random vs GC-matched 두 baseline의 결과 값 차이가 0.05 미만이면 representative random으로 인정

### F3 — JSD Distribution
- 2 panel:
  - (a) Per-layer violin plot of D_normalized (8 violins for 8 layers).
  - (b) Empirical CDF overlay (8 lines, viridis colormap), x-axis = D_normalized, y-axis = cumulative density. NLP-DTR Chen 2026의 reported ranges (Table 2 of paper)을 reference shaded band으로 overlay.
- Double column, 180 × 70 mm
- **Annotation**: effective range (p95 − p5) text annotation per layer
- **Verification**: shuffled control도 같은 plot에 점선으로 그려서 "random sequence는 더 좁은 분포"인지 sanity check

### F4 — TP53 gDTR Profile (Gate B primary fig)
- 4 stacked tracks (genome browser style), x-axis = chr17 coord (~19 kb):
  - (1) GENCODE annotation lane (exons as boxes, introns as line, UTR shading)
  - (2) gDTR profile (single-nt, smoothed with rolling 50-bp window for visibility, raw underneath as light gray)
  - (3) Settling-depth (c_interp) heatmap (1 row, color = layer 1..8 viridis)
  - (4) GC content & shuffled-baseline gDTR (gray)
- Splice site positions 표시 (▼)
- Double column, 180 × 100 mm
- **Verification**: smoothing이 인공물이 아닌지 raw also visible. shuffled baseline이 mean과 떨어져 있어야 함.

### F5 — BRCA1 gDTR Profile (optional, ~85 kb)
- F4와 동일 구조. 큰 region이므로 zoom-in inset 추가.

### F6 — Context Boxplot
- Boxplot: x = {coding exon, intron, 5'UTR, 3'UTR, splice±10, intergenic, shuffled}, y = gDTR per window
- Mann-Whitney p-value annotation (vs intron) using `statannotations`
- 점 overlay (각 window n) + violin background
- Single column, 90 × 80 mm
- **Verification**: n별 표시, 각 context의 sample size를 caption에 명시

### F7 — Variant ΔD(ℓ) Heatmap (primary variant fig)
- 5 × 8 heatmap: rows = 5 hotspot, cols = layer 1..8, color = ΔD(ℓ) (diverging RdBu_r centered at 0)
- 옆에 bar plot: 각 variant의 max|ΔD| with null-percentile annotation (※ = > p95, ※※ = > p99)
- Single column, 90 × 80 mm
- **Verification**: shuffled-position null heatmap을 별도 panel로 비교

### F8 — Three-metric Agreement
- 3 × 3 scatter matrix: Δc_discrete vs Δc_interp vs signed_argmax_ΔD
- 대각선에 marginal histogram. off-diagonal에 Spearman ρ + 95% CI 텍스트
- Single column, 90 × 90 mm

### F9 — Hyperparameter Heatmap
- 2 panel:
  - (a) Cohen's d (exon vs intron) heatmap on (γ, ρ) grid
  - (b) mean |Δc_interp| (TP53 hotspots) heatmap on (γ, ρ) grid
- 점선으로 default (0.5, 0.85) 표시
- Double column, 180 × 60 mm

### Tables (publication supplement)

| ID | 내용 | 위치 |
|---|---|---|
| T1 | Sanity statistics (M1, M2, per-layer) | `tables/sanity_M1_M2.csv` |
| T2 | gDTR by genomic context (Mann-Whitney) | `tables/gdtr_by_context.csv` |
| T3 | TP53 hotspot Δ-metrics + null percentile | `tables/tp53_hotspot_metrics.csv` |
| T4 | Three-metric agreement matrix | `tables/metric_agreement.csv` |
| T5 | HP sweep (γ, ρ) grid | `tables/hp_sweep.csv` |
| T6 | Computational budget actual | `tables/runtime_log.csv` |

---

## 8. Statistical Tests & Multiple Testing

| Test | Where | α | Correction |
|---|---|---|---|
| One-sided binomial (M2 ≥ 0.70) | Gate A per-layer | 0.05 | Bonferroni × 8 |
| Two-sided Mann-Whitney U (gDTR by context) | Gate B | 0.001 | Bonferroni × 5 contexts |
| Permutation test (context label) | Gate B sensitivity | 0.001 | n_perm = 1000 |
| Spearman ρ test (UR-gDTR cross-check) | §6.5 | 0.05 | none (single test per analysis) |
| Empirical p-value vs null (variant Δ) | Gate C informational | n/a | per-variant percentile reported |

**Effect size 보고 (필수)**: Cohen's d for parametric pairs, rank-biserial r for Mann-Whitney, Spearman ρ.

**Bootstrap CI (필수)**: 모든 point estimate에 1000-resample 95% CI 보고.

---

## 9. Code Structure

```
/path/to/gDTR-PoC/
├── README.md
├── phase0_design.md           # 본 문서 사본
├── requirements.txt
├── env_setup.sh
├── data/
│   ├── reference/             # chr17.fa.gz, chr13.fa.gz (+ .fai)
│   ├── annotation/            # gencode.v44.chr17_chr13.gtf.db
│   ├── variants/              # tp53_hotspots.tsv, clinvar_tp53.vcf
│   └── baselines/             # random_seqs_100.fa, shuffled_seqs_100.fa
├── src/
│   ├── __init__.py
│   ├── model_loader.py        # HyenaDNA + hook
│   ├── logit_lens.py          # JSD lens (tied weight unembedding)
│   ├── ur_gdtr.py             # cosine lens
│   ├── gdtr.py                # JSD trajectory, settling depth (discrete/interp), gDTR
│   ├── variant_delta.py       # 5 Δ-metrics
│   ├── controls.py            # GC-match, dinuc shuffle (uShuffle)
│   ├── viz.py                 # figure helpers (palette, axis style)
│   └── stats.py               # bootstrap CI, MWU+effect size
├── tests/
│   ├── test_gdtr.py           # synthetic JSD trajectory → known c
│   ├── test_logit_lens.py     # forward consistency
│   ├── test_interp.py         # boundary cases for c_interp
│   └── test_variant_delta.py  # idempotence (ref vs ref → 0)
├── scripts/
│   ├── 00_smoke_test.py
│   ├── 01_sanity_check.py
│   ├── 01b_jsd_distribution.py
│   ├── 02_gene_structure.py
│   ├── 03_variant_pilot.py
│   ├── 04_hp_sweep.py
│   └── 99_make_figures.py     # 모든 fig 일괄 재생성
├── results/
│   ├── figures/               # F1..F9 PDF + PNG
│   ├── tables/                # T1..T6 CSV
│   └── runs/                  # raw per-script log + JSON results
└── PHASE0_DECISION.md         # 최종 gate 판정 보고서
```

**Coding standards**:
- Python 3.10, type hints 필수
- Logging: `logging` stdlib, INFO 레벨로 모든 script
- Reproducibility: `set_seed(seed=42)` at script entry, save to JSON
- 모든 계산 결과는 CSV로 dump (figure는 from CSV로 재생성)
- 모든 script는 `--dry-run` 플래그로 lazy import 후 시그니처 검증 가능

---

## 10. Computation Budget

| Task | GPU | Time | VRAM peak |
|---|---|---|---|
| 00 smoke test | 3090 | 5 min | < 4 GB |
| 01 sanity (100 seq × 6kb × 8 layer) | 3090 | 0.5 hr | ~6 GB |
| 01b JSD dist (overlap with 01) | 3090 | 0.1 hr | — |
| 02 TP53 sliding window (~26 windows × 6kb) | 3090 | 0.5 hr | ~6 GB |
| 02-opt BRCA1 (HyenaDNA-large 1m) | 3090 | 4 hr | ~18 GB |
| 03 variant pilot (5 × 2 forwards × 6kb + null 100×) | 3090 | 1 hr | ~6 GB |
| 04 HP sweep (post-hoc, no extra forward) | CPU | 0.1 hr | — |
| Buffer (debugging) | 3090 | 5 hr | — |
| **Total** | | **~12 hr** | **24 GB headroom OK** |

추가 클라우드 비용 **$0** (vessl 단일 RTX 3090 활용).

---

## 11. Risk Matrix

| Risk | Detection | Mitigation | Owner |
|---|---|---|---|
| HyenaDNA HF API breaking change | smoke test Day 1 | LongSafari 공식 repo direct fallback (custom inference) | 본인 |
| Tied weight 미지원 | logit lens implementation Day 3 | LM head를 별도 nn.Linear로 wrap, 그래도 안 되면 input embedding transpose 시도 | 본인 |
| `output_hidden_states` shape 비일관 | smoke test | hook 직접 등록으로 우회 | 본인 |
| 8 layer가 너무 적어 deep regime이 c=8만 의미 | sanity check 결과 | ρ를 0.5로 하향 검토, c_interp scalar로 우회 | Phase 0 → 1 결정 |
| JSD effective range가 매우 좁음 (< 0.15) | §6.2 measurement | quantile-based γ로 전환, manuscript 방법론 contribution으로 격상 | Phase 1 calibration |
| Hyena layer logit lens가 noisy (Gate A fail) | §6.1 | UR-gDTR primary 격상, Phase 1 tuned lens 학습 추가 | Phase 1 method 결정 |
| TP53 region이 GC-bias가 강해 confound | §6.3 partial correlation | shuffled baseline + GC-matched random 비교 | 분석 시 |
| 1M context BRCA1 OOM | §6.3-opt | medium-160k segmented + 50% overlap stitching | 분석 시 |
| Variant Δ-metric이 모두 zero (model이 1 nt 변화 못 감지) | §6.4 | layer-wise ΔD가 noise 수준이면 informative negative, |context| 늘리기 (±5 kb) | Phase 1 sample size 재산정 |

---

## 12. Decision Tree (locked)

```
Gate A (Logit Lens validity)
├── Pass → Gate B
├── Fail (M2 < 0.85)
│   ├── UR-gDTR로 §6.3, §6.4 재산출
│   ├── UR-gDTR도 Gate B fail → STOP, informative negative paper
│   └── UR-gDTR Gate B pass → UR-gDTR primary, JSD-gDTR 보조, Phase 1 tuned lens 추가 학습
└── Partial pass (특정 layer만 fail) → 해당 layer 제외, Gate B 진행

Gate B (Genomic signal)
├── Pass (p < 0.001) → Gate C 보고만 하고 Phase 1 진입
├── Weak (0.001 ≤ p < 0.05)
│   ├── HyenaDNA-large (28M)으로 재시도
│   └── 그래도 weak → Phase 1 Evo 2 직접 진입 (모델 크기 부족 가설)
└── Fail (p ≥ 0.05)
    ├── HyenaDNA-large 재시도
    └── fail → Phase 1 진입 보류, methodology 재검토 (logit lens 적용 위치, tuned lens 도입)

Gate C (Variant)
├── ≥ 3/5 hotspot signal → Phase 3 sample size = 2000 P/LP × 3000 B/LB (계획대로)
├── 1–2/5 hotspot signal → context를 ±5 kb로 확대, Phase 3 sample size +50% 상향
└── 0/5 hotspot signal → variant analysis는 ΔD profile 기반으로 reframing, Δc는 deprecate

Three-metric agreement (Phase 0 → Phase 1 primary metric 결정)
├── ρ(Δc_interp, signed_argmax_ΔD) ≥ 0.7 → Δc_interp primary
├── 0.5 ≤ ρ < 0.7 → signed_argmax_ΔD primary
└── ρ < 0.5 → ΔD vector를 feature로 사용

JSD distribution (Phase 0 → Phase 1 calibration)
├── effective range ≥ 0.30 → γ=0.5 default
├── 0.15 ≤ range < 0.30 → quantile-based γ 보조
└── range < 0.15 → quantile-based γ primary, log|V| 정규화 deprecate
```

---

## 13. Reproducibility

- **Seeds**: 모든 random 작업에 seed=42 (script별 sub-seed offset 별도 lock).
- **Model checksum**: HyenaDNA HF revision SHA를 `data/MODEL_REVISION.txt`에 lock.
- **Data versions**: GRCh38 release date, GENCODE v44 release date, ClinVar release date를 `data/DATA_VERSIONS.txt`에 lock.
- **Environment lockfile**: `requirements.txt` + `env_setup.sh`로 PyTorch CUDA build까지 명시.
- **Computation log**: 모든 script가 시작 시 `runs/<script>_<timestamp>.json`에 git SHA, hostname, GPU model, seeds, args를 기록.
- **Figure 재현**: `99_make_figures.py`로 CSV 입력만으로 모든 figure 재생성. CSV 자체가 raw forward output에서 재현 가능.

---

## 14. Acceptance Criteria for Phase 1 Entry

`PHASE0_DECISION.md`는 다음을 명시:
1. Gate A: M1, M2 per-layer values + bootstrap CI. Pass/Fail 판정 근거.
2. Gate B: Mann-Whitney p-value (5개 context vs intron) + Cohen's d. Pass/Fail.
3. Gate C: 5 hotspot 각각의 max|ΔD|, percentile vs null. Three-metric ρ.
4. JSD effective range, calibration 권고.
5. Phase 1 method 권고 (primary metric, primary lens, hyperparameter starting point).
6. 사후 변경된 임계값/방법이 있으면 명시 + 영향 분석.

**Phase 1 진입 조건**: Gate A pass (또는 UR-gDTR fallback이 Gate B를 pass) AND Gate B pass (p < 0.05까지 허용, primary 분석은 p < 0.001 기준 보고).

---

## 15. Limitations of Phase 0 (사전 명시)

- **N=5 hotspot은 변량 분석 검정력 부족**: Gate C는 informational. Phase 3 본분석에서 ClinVar 2K+ variant로 검정력 확보.
- **HyenaDNA는 attention 없음**: Phase 0이 Hyena conv block의 logit lens를 검증하지만, Evo 2의 attention block 거동은 Phase 1에서 별도 검증 필요.
- **8 layer 모델의 c 분해능 한계**: c ∈ {1..8}로 NLP의 32+ layer 모델 대비 분해능이 거칠다. c_interp가 부분적으로 보완하나, Phase 1 (Evo 2 32 layer)에서야 NLP 수준 분해능 확보.
- **TP53/BRCA1만 검증**: gene-specific bias 가능. Phase 2의 chr22 genome-wide profiling에서 일반화 검증.
- **Single species**: HyenaDNA의 training data는 human genome reference. Cross-species generalization은 Phase 4에서.

---

## Appendix A — HyenaDNA Architecture Notes

- 모든 block은 Hyena conv (no attention).
- Block 구조: `Hyena → MLP → residual` (pre-LN). Hidden state는 매 block 후 residual stream.
- Tokenizer: character-level, vocab 12 (실제 token id는 smoke test로 확인).
- Position embedding: rotary 또는 implicit Hyena filter (모델별 상이, 확인 필요).
- LM head: tied with input embedding (확인 필요).
- HF wrapper: `LongSafari/hyenadna-medium-160k-seqlen-hf`, `trust_remote_code=True` 필수.

## Appendix B — Forward Pass Reference Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LongSafari/hyenadna-medium-160k-seqlen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda").eval()

# BOS handling: 확인 필요. tokenizer.bos_token_id가 있으면 input에 prepend.
seq = "ACGT" * 1500  # 6 kb
input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

# out.hidden_states: tuple of (L+1) tensors, each (1, T, H)
# out.logits: (1, T, V) — final layer prediction
# 모든 hidden state에 final_layer_norm을 적용하고 lm_head로 project하여 logit 산출
```

`output_hidden_states`가 미작동하면 forward hook 등록으로 대체:
```python
captured = []
def hook(module, input, output):
    captured.append(output if not isinstance(output, tuple) else output[0])
for blk in model.backbone.layers:  # 정확한 attr는 smoke test로 확인
    blk.register_forward_hook(hook)
```

---

---

## Appendix C — Empirical Corrections from Smoke Test (2026-04-26)

본 Appendix는 §A의 추정값을 vessl smoke test로 검증한 결과 **5개 가정이 틀렸음을 확인**하고 정정한 내용이다. 본 문서가 lock된 후 발견된 사실이지만, **Implementation 시 §A 대신 본 Appendix를 정확한 spec으로 사용해야 한다**. (사전등록 정신은 분석 임계값·통계·gate에 적용되며, 모델 구현 사실 정정은 그에 해당하지 않음.)

### C.1 lm_head는 tied가 아니다
- §A 가정: tied with input embedding
- 실제: `lm_head.weight`와 input embedding은 별도 weight (id() 및 data_ptr() 모두 상이)
- **Implementation**: `model.lm_head`를 직접 사용. transposed embedding 사용 금지.

### C.2 lm_head.out_features = 16 ≠ vocab_size = 12
- §A 가정: vocab=12, lm_head out=12
- 실제: vocab=12, **lm_head.out_features=16** (compute efficiency padding)
- **Implementation**: softmax/JSD 계산 전 `logits_real = logits[..., :12]`로 마스킹 필수. 마스킹 안 하면 padding token에 확률 질량 누설 → JSD 왜곡.
- 정규화는 **log(12) = 2.484**로 수행 (`log(16)`이 아님)

### C.3 hidden_states 레이아웃
- §A 가정: tuple of L+1 (embedding + L blocks)
- 실제: `len(hidden_states) = 10` = embed + 8 block outputs + post-ln_f 적용본
  - `hidden_states[0]` = embeddings output
  - `hidden_states[1..8]` = pre-ln_f block outputs (residual stream raw)
  - `hidden_states[9]` (=`[-1]`) = **post-ln_f**, `lm_head` 입력으로 final logits 산출
- **검증**: `lm_head(hidden_states[-1]) - out.logits` 정확히 0
- **검증**: `lm_head(ln_f(hidden_states[-1])) - out.logits` ≈ 0.69 (ln_f 이중 적용)
- **Implementation** logit lens at intermediate layer ℓ ∈ {1..8}:
  ```python
  logits_l = model.lm_head(model.hyena.backbone.ln_f(hidden_states[l]))[..., :12]
  ```
  Layer L=8 만큼은 `model.lm_head(hidden_states[-1])[..., :12]`로 ln_f 생략 가능.

### C.4 Tokenizer가 BOS를 prepend함
- §A 가정: position i nucleotide → input_ids[0, i]
- 실제: `tokenizer(seq).input_ids.shape = [1, len(seq)+1]`. BOS(id=2)가 자동 prepend, EOS는 미추가.
- 토큰 ID: BOS=2, EOS=1, PAD=4
- **Implementation**: 시퀀스 인덱스 i (0-based) → `input_ids[0, i+1]`. 모든 per-position gDTR 분석에서 hidden states를 `[:, 1:, :]`로 slice.

### C.5 Backbone 경로
- §A Appendix B 가정: `model.backbone.blocks`
- 실제: `model.hyena.backbone.layers` (ModuleList of 8 `HyenaBlock`). Final norm: `model.hyena.backbone.ln_f`.

### C.6 환경 quirks
- `transformers>=5.0`은 `torch>=2.4` 요구 → torch 2.3.1 환경에서는 `transformers<4.50` 필요. **Lock: transformers==4.49.0**.
- `trust_remote_code=True`가 cv2를 import하므로 `libGL.so.1` 필요. `apt install libgl1-mesa-glx libglib2.0-0`.

### C.7 TP53 hotspot GRCh38 좌표 확정 (ClinVar 2026-04-18 lookup)
| variant | chrom | pos (GRCh38, fwd strand) | ref | alt | ClinVar ID |
|---|---|---|---|---|---|
| R175H | chr17 | 7,675,088 | C | T | 12374 |
| R248Q | chr17 | 7,674,220 | C | T | 12356 |
| R273H | chr17 | 7,673,802 | C | T | 12366 |
| R249S | chr17 | 7,674,230 | C | A | 12349 |
| G245S | chr17 | 7,674,241 | G | A | 12359 |

TP53는 minus strand → VCF forward-strand allele은 transcript HGVS의 complement.

### C.8 Reference forward pass code (정정판)

```python
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LongSafari/hyenadna-medium-160k-seqlen-hf"
HF_REVISION = "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce"
VOCAB_REAL = 12  # mask logits[..., :12] before softmax

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=HF_REVISION,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=HF_REVISION,
    torch_dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda").eval()

ln_f = model.hyena.backbone.ln_f
lm_head = model.lm_head

seq = "ACGT" * 1500  # 6 kb
input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cuda")
# input_ids.shape == [1, 6001]; [0, 0] is BOS

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

# hidden_states[1..8] are pre-ln_f block outputs; [9] is post-ln_f
def logit_lens(h, ell):
    """h: hidden_states tuple. ell: 1..L. Returns logits over real vocab."""
    if ell == len(h) - 1:  # final post-ln_f layer
        logits = lm_head(h[ell])
    else:
        logits = lm_head(ln_f(h[ell]))
    return logits[..., :VOCAB_REAL]  # mask padding

# Slice off BOS for per-nucleotide analysis
def per_nt_logits(logits):
    return logits[:, 1:, :]
```

### C.10 HyenaDNA intermediate hidden state dtype 불일치 (impl 시 발견)

`torch_dtype=torch.bfloat16`로 모델 로드 시 hidden_states tuple의 dtype이 **혼합됨**:
- `hidden_states[0]` (embedding output): bfloat16
- `hidden_states[1..8]` (intermediate Hyena block outputs): **float32** — 모델 내부에서 자동 upcast
- `hidden_states[9]` (post-ln_f): bfloat16
- `ln_f.weight.dtype = lm_head.weight.dtype = bfloat16`

**Implementation**: 중간 layer logit lens 적용 시 dtype cast 필수
```python
def _layer_logits(h, ln_f, lm_head, vocab_real=12, is_final=False):
    if is_final:
        h = h.to(lm_head.weight.dtype)
        logits = lm_head(h)
    else:
        h = h.to(ln_f.weight.dtype)  # float32 → bfloat16
        logits = lm_head(ln_f(h))
    return logits[..., :vocab_real]
```

`test_logit_lens.py`에서 final-layer p_lens가 `softmax(out.logits[:,:,:12])`와 < 1e-3 일치하는지 검증으로 catch.

### C.9 영향 정리

| § (이전) | 영향 받음 | 정정 |
|---|---|---|
| §4.2 step 4 | logit lens 구현 | ln_f 적용 + lm_head + vocab mask |
| §4.2 step 9 | 정규화 상수 | `log(12)` 사용 |
| §6.* per-nt analysis | position alignment | `[:, 1:, :]` slice |
| §A architecture | model attr 경로 | `model.hyena.backbone.{layers, ln_f}` |
| §B reference code | 변경 | C.8로 대체 |

본 정정은 **분석 임계값·gate·통계 검정에 영향이 없으며** 모델 spec의 사실 정정에 한한다. 사전등록(pre-registration) 정신은 그대로 유지된다.

---

**End of Phase 0 Design Document**

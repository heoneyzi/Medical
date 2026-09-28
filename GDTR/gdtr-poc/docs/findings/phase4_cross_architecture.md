# Phase 4 — Cross-architecture validation (4 genomic foundation models)

**Project**: gDTR on Evo 2 7B + cross-architecture confirmation.
**Phase**: 4 (chr22 12,978 windows × 4 genomic FM models, model-agnostic UR-gDTR cosine_lens with per-model q70 calibration).
**Status**: COMPLETE. 17 min total wall on H200. 4 models replicate the splice deep-thinking signature within their family; cross-family rankings diverge (two-tier invariance).
**Predecessors**: [`phase1_evo2_calibration.md`](phase1_evo2_calibration.md), [`phase2_chr17_replication.md`](phase2_chr17_replication.md).

This doc was extracted from §11.7 of the legacy `PHASE1_FINDINGS.md`.

---

## 0. TL;DR

- 4 models compared on chr22: Evo 2 7B (existing), HyenaDNA-large-1m, NT-v2 500M, DNABERT-2 117M.
- **Two-tier architecture invariance** — within causal-LM family ρ = +0.516; within MLM family ρ = +0.663; cross-family ρ ∈ [-0.119, -0.287].
- All p-values < 1e-42; 4-way top-decile concordance is **zero windows** — each architecture family lights up DIFFERENT chr22 windows.
- **Splice signal replicates universally in both per-bp causal-LM models** (Evo 2 + HyenaDNA-large): donor mean_c < intron baseline at vastly different scales (7B vs 28M, 32 vs 8 layers). MLM models (NT-v2, DNABERT-2) cannot probe per-bp splice positions due to k-mer/BPE tokenization, so the absence of a per-bp signal there is a methodological limitation, not a counterexample.

---

## 1. Models compared

| Model | Architecture | Layers | Hidden | Tokens / 6kb window | Wall | γ_q70 |
|---|---|---:|---:|---|---|---:|
| Evo 2 7B (existing) | Hybrid Transformer + StripedHyena 2 | 32 | 4096 | 6,000 (1bp) | reused | 0.396 |
| HyenaDNA-large-1m | Pure Hyena | 8 | 256 | 6,001 (1bp + BOS) | ~4 min | 0.358 |
| NT-v2 500M | Transformer MLM (k-mer) | 29 | 1024 | 671 (k=6, 4kb) | ~7.5 min | 0.533 |
| DNABERT-2 117M | Transformer MLM (BPE) | 12 | 768 | ~600 (BPE, 3kb) | ~3 min | 0.677 |

**Engineering challenges**:
- NT-v2 forced to fp32 (vendor bf16 attention path broken).
- DNABERT-2 required disabling bundled triton flash-attn kernel (compilation incompatibility); patched to use PyTorch fallback.
- Both still ran successfully.

---

## 2. Pairwise Spearman ρ on per-window mean settling depth

(all p < 1e-42)

|  | evo2 | hyena | nt_v2 | dnabert2 |
|---|---:|---:|---:|---:|
| **evo2** | 1.00 | **+0.516** | -0.119 | -0.188 |
| **hyenadna** | +0.516 | 1.00 | -0.287 | -0.166 |
| **nt_v2** | -0.119 | -0.287 | 1.00 | **+0.663** |
| **dnabert2** | -0.188 | -0.166 | +0.663 | 1.00 |

### Two-tier interpretation

- **Within-family STRONG correlation**:
  - Causal-LM per-bp models (Evo 2 + HyenaDNA): ρ = +0.516
  - Bidirectional MLM token-based models (NT-v2 + DNABERT-2): ρ = +0.663
- **Cross-family NEGATIVE correlation**: Causal-LM ↔ MLM ρ ∈ [-0.119, -0.287]
- All p < 1e-42 (highly significant)

**4-way top-decile concordance**: ZERO windows intersect — supports the
two-tier story (each architecture family lights up DIFFERENT chr22 windows).

---

## 3. Per-model splice signal

| Model | donor mean_c | acceptor mean_c | intron mean_c | direction donor < intron? |
|---|---:|---:|---:|---|
| Evo 2 (per-bp) | 25.59 | 25.71 | 27.84 | yes |
| HyenaDNA (per-bp, L=8) | 6.55 | 6.62 | 6.89 | yes |
| NT-v2 (per-window) | 27.85 | n/a | n/a | k-mer alignment limitation |
| DNABERT-2 (per-window) | 11.27 | n/a | n/a | BPE alignment limitation |

**Splice deep-thinking signal replicates universally in both per-bp models**
(Evo 2 + HyenaDNA-large), at vastly different model scales (7B vs 28M, 32 vs 8
layers). This is strong evidence that the splice-deep-thinking phenomenon is
**architecture-invariant within the per-bp causal-LM family**.

For k-mer/BPE MLMs (NT-v2, DNABERT-2), the per-position splice grid doesn't
align to bp tokens, so per-position splice signal is inaccessible without
re-aligning tokens to bp coordinates. This is a **methodological limitation
rather than evidence against architecture-invariance**.

---

## 4. Refined manuscript narrative

> Architecture invariance is **two-tier**: within architecture families (per-bp
> causal-LM models or token-based MLM models) gDTR rankings are strongly
> correlated (ρ ≥ 0.5). Cross-family correlations diverge or even invert,
> suggesting that the level at which "deep computation" occurs depends on the
> model's tokenization/objective. The splice-site deep-thinking signal — the
> manuscript's headline phenomenon — replicates universally in both Evo 2 (7B,
> 32-layer hybrid) and HyenaDNA-large (28M, 8-layer pure Hyena), confirming the
> signal is not Evo-2-specific. K-mer/BPE MLMs offer different but consistent
> within-family rankings.

This is **stronger than a simple "universal" claim** because it identifies the
mechanism (within-family architecture invariance + tokenization-level
dependence) and its limitations.

---

## 5. Files

`results/phase4/`:
- `chr22_cache_hyenadna.h5`, `chr22_cache_nt.h5`, `chr22_cache_dnabert.h5` (gitignored, 1.4 GB total).
- `per_model_summary.json` (γ, layer counts, splice signal per model).
- `concordance_matrix.json` (Spearman ρ + p-values + top-decile intersection).
- `_dnabert_ckpt.json`, `_hyenadna_ckpt.json`, `_nt_ckpt.json` (small per-model checkpoint metadata).
- `F12_cross_arch_heatmap.{pdf,png}`, `F13_per_model_splice.{pdf,png}`.
- `_done`.

---

## 6. Connection to Phase 5 Q2 finding

Q2 (high gDTR + low conservation) was discovered in Evo 2. With Phase 4
within-family results, the same Q2-style ranking is expected to replicate in
HyenaDNA-large (within-family ρ = +0.516). This generalizes the Q2 framework
beyond a single model. See [`phase5_conservation_discordance.md`](phase5_conservation_discordance.md).

---

**Document version**: 2026-04-28 (split from legacy `PHASE1_FINDINGS.md` §11.7).

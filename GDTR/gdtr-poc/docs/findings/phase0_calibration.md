# Phase 0 — HyenaDNA-medium-160k calibration

**Status**: COMPLETE (2026-04-26). 14/14 tasks, 4 paper-grade findings, ~38 minutes
on a single RTX 3090.

This document is an index pointer; the full Phase 0 synthesis lives in
[`../PHASE0_FINDINGS.md`](../PHASE0_FINDINGS.md) (~6,400 words) with the locked
verdicts in [`../PHASE0_DECISION.md`](../PHASE0_DECISION.md) and the
pre-registered design in [`../phase0_design.md`](../phase0_design.md).

## Headline numbers

- Primary lens lock: **UR-gDTR (cosine)** with γ_cos=0.50, ρ=0.85.
- L7 anomaly: penultimate-block (L7→L8) alignment SPIKE; ~85–90% of the L7
  effect is explained by representation rotation into the trained readout
  subspace (causal: tuned-lens lifts JSD M2_L7 from 0.120 → 0.917).
- Calibration finding: NLP-DTR defaults DO NOT transfer to genomic CLMs with
  small vocabularies (|V|=12). Effective JSD range ≈ 0.019; γ_q70 quantile
  calibration is mandatory.
- Robustness lemma: TP53 coding-exon vs intron MWU p = 4.88×10⁻²²⁴, Cohen's d
  = −1.018 (intron > exon). Replicated in BRCA1 (p ≈ 0, d = −0.78).

## What Phase 0 locked for downstream phases

| Knob | Phase 0 lock | Used unchanged in |
|---|---|---|
| Primary lens | UR-gDTR cosine | Phases 1, 2, 3, 4 |
| Auxiliary lens | JSD-gDTR + quantile-γ | Phases 1, 2, 3 |
| γ_cos | 0.50 | Phase 1 confirmed plateau (best 0.40, ±0.10) |
| ρ | 0.85 | Phase 1 confirmed plateau (best 0.80, ±0.05) |
| Variant feature primary | ΔD(ℓ) vector ∈ ℝ^L | Phase 3 (L=32) |
| Tuned-lens target | last 1–2 blocks (HyenaDNA L7/L8) | OVERRIDDEN in Phase 1 (Evo 2 L31 idle); Phase 1 carry-over uses L=29 |

See `PHASE0_FINDINGS.md` for: D1–D5 mechanistic decomposition, E1 tied-head
ablation, E2 codon-position stratification, E5 tuned-lens prototype, and the
full set of 21 pytest unit tests under `tests/`.

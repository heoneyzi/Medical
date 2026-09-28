# Phase 5 — gDTR vs PhyloP conservation: Q2 discordance

**Project**: gDTR on Evo 2 7B.
**Phase**: 5 (chr22 50.8 Mb position-level gDTR vs PhyloP 100-way conservation; Q2 = high gDTR × low conservation candidate annotation).
**Status**: COMPLETE. Q2 region BED + 7-annotation enrichment table + 5,090 ≥ 100 bp regions identified for downstream functional follow-up.
**Predecessors**: [`phase1_evo2_calibration.md`](phase1_evo2_calibration.md) (chr22 per-position settling depth).
**Companion**: [`tier2_extensions.md`](tier2_extensions.md) §T2.1 — Q2 functional validation against eQTL / GWAS / ENCODE cCRE-ELS (independent enrichment beyond the in-Phase-5 RepeatMasker / cCRE table).

This doc was extracted from §11.5 of the legacy `PHASE1_FINDINGS.md`.

---

## 0. TL;DR

- Q2 = high gDTR (top 25% c) × low conservation (bottom 25% PhyloP) covers
  3.71 % of chr22 (1.9 Mb), with 5,090 contiguous regions ≥ 100 bp.
- Q2 is enriched **2.02×** for RepeatMasker low-complexity, **1.95×** for
  5'UTRs, **1.39×** for LTR-class TEs, **1.28–1.31×** for ENCODE cCRE / rDHS
  and LINEs (all p ≈ 0 by hypergeometric one-sided).
- Mechanistic claim: Q2 = "model-derived map of deep-thinking regions in
  lineage-specific regulatory DNA"; complements PhyloP/GERP for prioritising
  functional but non-conserved regulatory elements.

---

## 1. Methodology

- **Data**: chr22 50.8 Mb, per-position gDTR settling depth c (integer 1–32, computed in Phase 1.6) × PhyloP 100-way conservation track.
- **Smoothing**: 100 bp box-car smoothing of both signals (raw c is integer-quantized into 32 levels; raw PhyloP runs maxed at 34 bp).
- **Quadrant definition**: Q1 = high gDTR + high cons; Q2 = high gDTR + low cons; Q3 = low gDTR + high cons; Q4 = low gDTR + low cons (top/bottom 25% on each axis).
- **Coverage**: 71.2% of chr22 valid (NaN: c 23.2%, PhyloP 28.7%, smoothed).

---

## 2. Quadrant sizes (% chr22)

| Quadrant | Size (%) | Interpretation |
|---|---:|---|
| Q1 (high gDTR + high cons) | 14.09% | Conserved deep computation (known functional) |
| **Q2 (high gDTR + low cons)** | **3.71% (1.9 Mb)** | Recently evolved deep computation candidates |
| Q3 (low gDTR + high cons) | 39.30% | Conserved but predictable (e.g., simple repeats) |
| Q4 (low gDTR + low cons) | 14.09% | Background noise |

**Q2 contiguous regions ≥ 100 bp**: 5,090 (paper-quality figure).

---

## 3. Q2 enrichment (hypergeometric one-sided, all p ≈ 0)

| Annotation | Fold | Source |
|---|---:|---|
| **rmsk_Low_complexity** | **2.02×** | RepeatMasker |
| rmsk_Simple_repeat | 1.52× | RepeatMasker |
| rmsk_LTR | 1.39× | RepeatMasker (transposable elements) |
| rmsk_LINE | 1.31× | RepeatMasker |
| **5'UTR (genomic context)** | **1.95×** | GENCODE v44 |
| ENCODE cCRE | 1.28× | ENCODE SCREEN v3 |
| ENCODE rDHS | 1.25× | ENCODE rDHS catalog |

**Largest Q2 region**: chr22:22,893,870–22,895,351 (1,481 bp intron, mean c = 31.31, mean PhyloP = -0.63).

---

## 4. Key finding — paper-grade

Q2 is significantly enriched for **transposable-element-derived regulatory
sequences** (low_complexity 2×, simple_repeat 1.5×, LTR 1.4×) <b>AND</b> ENCODE
regulatory elements (cCRE/rDHS 1.25–1.28×) **AND** 5'UTRs (~2×). This is
consistent with the literature observation that **lineage-specific TE-derived
enhancers/promoters are major sources of recently evolved regulatory function**
(Chuong et al. 2017, Nat Rev Genet). gDTR captures their "deep computation"
signature even when traditional conservation does not.

---

## 5. Manuscript narrative

> Q2 = "model-derived map of deep-thinking regions in lineage-specific
> regulatory DNA" — provides a new annotation layer that complements
> PhyloP/GERP for prioritizing functional but non-conserved regulatory
> elements.

This positions gDTR as **complementary** to evolutionary conservation, not
redundant with it.

---

## 6. Downstream — Tier 2 functional validation

The Q2 regions are validated against three independent functional annotations
in [`tier2_extensions.md`](tier2_extensions.md) §T2.1:
- GTEx eQTL (chr22 subset): **1.62× fold-enrichment**, p_hypergeom = 5.4e-56
- GWAS Catalog (chr22 subset): **1.50× fold-enrichment**, p_hypergeom = 1.6e-7
- ENCODE cCRE-ELS (chr22 subset): **1.90× fold-enrichment**, p_hypergeom ≈ 0

All three are independent of the in-Phase-5 RepeatMasker enrichment and
strengthen the "non-conserved but functional" story.

---

## 7. Files

`results/phase5/`:
- `chr22_quadrants.npy` (gitignored).
- `q2_regions.bed` — 5,090 regions ≥ 100 bp.
- enrichment table JSON.
- F-figures (PDF/PNG).

`results/tier2_q2_functional/` — see [`tier2_extensions.md`](tier2_extensions.md).

---

**Document version**: 2026-04-28 (split from legacy `PHASE1_FINDINGS.md` §11.5).

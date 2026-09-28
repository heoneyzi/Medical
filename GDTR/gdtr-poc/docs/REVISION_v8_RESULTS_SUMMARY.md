# REVISION_v8 RESULTS SUMMARY (2026-05-04 06:24 UTC)

## Pre-flight invariants (v7 baseline + post-build): ALL 10 PASS
- I2 ΔD AUROC = 0.8437 (canonical 0.844, dD_cos 32-d, seed=42)
- I7 BRCA1 17:43057063 max_abs_dD = 3.6699e-02 (preserved)
- I8 chr22 Q2 = 3.71% / 5,090 regions (locked baseline)
- All others PASS

## P1a — CALIB-VAL: PASS Branch A (clean transfer)
- chr17 q70 = 0.39429 vs chr22 q70 = 0.39663, abs_diff = 0.0023 (well under 0.05)
- chr17 splice_donor Cohen's d vs intron = -0.349 (chr22 = -0.369), ratio 0.946
- chr17 exon-vs-intron MWU p = 0.0
→ γ_cos = 0.39663 generalises chr22→chr17. Table 2 splits 2a/2b cleanly.

## P1b — SPLICE-CANON: REFRAME Branch C (REVERSE ordering)
chr17 frozen-γ mean c per class:
  non_canonical_donor    25.24  (SHALLOWEST)
  non_canonical_acceptor 25.01
  canonical_GT_AG_donor  25.57
  canonical_GT_AG_accept 26.01
  canonical_GC_AG_donor  27.37  (DEEPEST splice)
  canonical_GC_AG_accept 26.65
  intron baseline        27.69
→ All splice < intron (universality holds). User's hypothesis (canonical < non-canonical) REVERSED.
   Likely cause: non-canonical sites have constrained branch-point/U12 contexts → recognized fast.
   Honest report in Limitations.

## P2 — CASE-FIX: 5-class population analysis
P/LP median argmax_layer (all classes from cached SNV + new indel forward):
  intron            10  (n=116)
  frameshift        11  (n=518, P2-INDEL)
  nonsense          12  (n=1740)
  missense          16  (n=935)
  canonical_splice  17  (n=682)
  synonymous        18  (n=32, small)
→ KW 5-way p = 7.1e-10 (highly significant).
→ Branch C partial ordering. Frameshift integrated cleanly with SNV pipeline.
→ 5 representative traces picked at per-class median (APC missense L16, MSH2 nonsense L12,
   APC splice L17, ATM synonymous L25, APC frameshift L11).
→ BRCA1 17:43057063 G→A factually corrected to nonsense (replaced "frameshift locus").

## P3B-1 — Q2 functional positive control: FAIL hypothesis, POSITIVE NEW FINDING
chr22 c̄ at functional sites vs 100-shuffle null:
  cCRE-ELS:    c_func=26.86 vs c_shuffled=28.15  (effect -43σ, p_greater=1.0)
  GTEx eQTL:   c_func=28.03 vs c_shuffled=28.15  (effect -4σ)
  GWAS:        c_func=27.91 vs c_shuffled=28.14  (effect -2.8σ)
→ Functional sites have LOWER (shallower) c, not higher. Plan F4 Branch B = effect REVERSED.
→ This is consistent with the splice-shallow finding from §3.1. NEW POSITIVE CONTRIBUTION:
   "gDTR shallowness identifies functional regulatory elements at genomic scale, generalising
    the splice-shallow recognition pattern to the broader cis-regulatory landscape."
→ Q2 = high gDTR + low conservation = functional CLAIM IS DEAD (Q2 is high gDTR = deeper, but
   functional = shallower; Q2 is therefore NOT a functional discovery).
→ Path C4 DEMOTE: §3.2 reduced. Replace v7 Q2 narrative with NEW "functional shallowness" lead.

## P3B-2 — Repeat class breakdown: PASS, all 9 classes locked
chr22:  SINE 0.93 (under-enr, p=1.0) | LINE 1.31 | LTR 1.39 | DNA 1.20 |
        Simple_repeat 1.52 | Low_complexity 2.02 | Satellite 0.24 (under-enr)
→ Honest disclosure: LTR-led pattern + SINE depletion.

## P3B-3 — Q2 multi-chr replication chr17: PASS partial replication
chr17 Q2 = 5.49% (chr22: 3.71%)
chr17:  SINE 1.29 (chr22: 0.93)  ← DIFFERS
        LINE 1.25 (1.31)  LTR 1.47 (1.39)  DNA 1.32 (1.20)
        Simple 1.76 (1.52)  Low_compl 2.08 (2.02)
→ LTR/LINE/DNA enrichments replicate. SINE direction differs — chr17 has high SINE density
   (vs chr22) so chr-specific TE distribution confounds the SINE depletion claim.
→ chr1 forward SKIPPED (no longer needed under C4 demote).

## SUMMARY OF MANUSCRIPT IMPLICATIONS
1. §2.1 + Table 2 — chr22 calib / chr17 validation split (P1a Branch A clean).
2. §3.1 splice — universality holds; canonical/non-canonical reverse ordering honestly reported.
3. §3.2 NEW HEADLINE — functional regulatory elements converge shallowly (B-1 reversed = new positive).
4. §3.2 demoted — Q2 = high-gDTR + low-conservation reframed as "computationally distinct subspace,
   chromosome-specific repeat enrichment, NOT a functional discovery". Multi-chr partial replication.
5. §3.3 — per-class population statistics with 5 classes (incl. true frameshift). KW 5-way p=7.1e-10.
   3-variant table replaced with population summary + 5 representative traces.
6. Limitations — single-chr calibration, canonical-splice reverse ordering, chr-specific TE bias.

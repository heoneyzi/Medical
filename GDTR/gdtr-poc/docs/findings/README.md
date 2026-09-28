# Findings — per-phase index

This directory contains one synthesis document per phase of the gDTR (genomic
Deep-Thinking Ratio) project, plus two cross-cutting Tier 1 / Tier 2
extension docs. The legacy `PHASE1_FINDINGS.md` (873 lines) was split into
these per-phase files on 2026-04-28; the original is removed in commit 1 and
its split-plan is preserved at `_split_plan.md` for traceability.

## Phase status table (2026-04-28)

| Doc | Phase | Headline | Status |
|---|---|---|---|
| [`phase0_calibration.md`](phase0_calibration.md) | 0 — HyenaDNA-medium-160k PoC | UR-cos lens locked; L7 tuned-lens recovers M2 0.12→0.92 | DONE |
| [`phase1_evo2_calibration.md`](phase1_evo2_calibration.md) | 1 — Evo 2 7B method calibration | L31 idle quirk; chr22 splice deep-thinking; HP transfers cleanly | DONE |
| [`phase2_chr17_replication.md`](phase2_chr17_replication.md) | 2 — chr17 multi-chromosome replication | Splice signature replicates; cancer-driver bias confirmed (Cohen's d=0.87) | DONE |
| [`phase3_variant_pathogenicity.md`](phase3_variant_pathogenicity.md) | 3 — ClinVar variant classification | **AUROC 0.844** (15 genes, 10K variants); ensemble +0.017 over ΔD alone (DeLong p=3.6e-15) | DONE |
| [`phase4_cross_architecture.md`](phase4_cross_architecture.md) | 4 — Cross-architecture validation | Two-tier invariance: within causal-LM ρ=+0.52, within MLM ρ=+0.66, cross ρ negative | DONE |
| [`phase5_conservation_discordance.md`](phase5_conservation_discordance.md) | 5 — Q2 gDTR vs PhyloP | 5,090 chr22 regions; 2× TE-derived enrichment; complements PhyloP | DONE |
| [`tier1_extensions.md`](tier1_extensions.md) | T1 — Phase-3 diagnostics | Per-layer ablation L29 best (0.794); bootstrap CI [0.833, 0.853]; case studies P > B 11×/8×/1.5× | T1.1/T1.3/T1.4 DONE; T1.2 running |
| [`tier2_extensions.md`](tier2_extensions.md) | T2 — robustness + Q2 functional | Q2 1.62× eQTL / 1.50× GWAS / 1.90× cCRE-ELS; HP grid range 0.0017; CADD-disagree FN 0.46 | T2.1/T2.2/T2.3 DONE; T2.4 running |

## Pre-registered decisions and architectural facts

These live in [`../decisions/`](../decisions/):

- [`phase1_decisions.md`](../decisions/phase1_decisions.md) — pre-registered Phase 1 plan (locked thresholds, decision tree).
- [`phase1_appendix_c.md`](../decisions/phase1_appendix_c.md) — Evo 2 7B architectural facts from the smoke test.
- `phase1_execution_plan.md` — server-specific Phase 1 execution doc (not included in this portfolio copy).
- [`phase2_decisions.md`](../decisions/phase2_decisions.md) — auto-generated Phase 2 gate verdicts (chr17 prep / forward / Gate B / cross-chr / gene-class / splice fine).

## Cross-reference table (legacy → new)

| Legacy section | New location |
|---|---|
| `PHASE1_FINDINGS.md` §0 (TL;DR) | `phase1_evo2_calibration.md` §0 |
| `PHASE1_FINDINGS.md` §§1–10 + Appendices A, B | `phase1_evo2_calibration.md` §§1–10 + Appendices |
| `PHASE1_FINDINGS.md` §11.1 (ClinVar pilot) | `phase3_variant_pathogenicity.md` §1 |
| `PHASE1_FINDINGS.md` §11.2 (32-layer landscape) | `phase1_evo2_calibration.md` §11 |
| `PHASE1_FINDINGS.md` §11.3 (chr17 IN PROGRESS marker) | `phase2_chr17_replication.md` (full results from `PHASE2_DECISION.md`) |
| `PHASE1_FINDINGS.md` §11.4 (Phase 3 main 15 genes) | `phase3_variant_pathogenicity.md` §2 |
| `PHASE1_FINDINGS.md` §11.5 (Q2) | `phase5_conservation_discordance.md` |
| `PHASE1_FINDINGS.md` §11.6 (CADD+AM+DeLong) | `phase3_variant_pathogenicity.md` §3 |
| `PHASE1_FINDINGS.md` §11.7 (Phase 4 cross-arch) | `phase4_cross_architecture.md` |
| `PHASE1_FINDINGS.md` §12 (final 5-finding rollup) | summarized at top of each per-phase doc |
| (NEW) Tier 1 — per-layer ablation, bootstrap, case studies, baselines | `tier1_extensions.md` |
| (NEW) Tier 2 — Q2 functional, HP sensitivity, failure analysis, cost | `tier2_extensions.md` |

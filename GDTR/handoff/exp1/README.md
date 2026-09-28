<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GDTR](../../README.md) › [Line C · Handoff](../README.md) › **EXP1**</sub>

# 🔬 C1 · EXP1 — Numerator vs denominator, carrier vs content: what survives the handoff?

> **Question —** When the alignment between the state and the output frame collapses at block 28, is earlier computation *lost*, *diluted*, or *overwritten* — and does the late stack discard it or summarise it?

| | |
|---|---|
| **Status** | ✅ sessions 1–4b done (2026-09-18 → 09-19); 40 pre-registered predictions, 14 failed or refuted, 6 half-supported |
| **Model / data** | `evo2_7b` (1M) with an `evo2_7b_base` replication; chr22 panel of 400 non-overlapping 6 kb windows (central 3 kb scored, seed 42): 4.8 M position-rows over 4 input variants (real, random, poly-A, dinucleotide shuffle; 1.2 M each) + a weight-shuffled control; 2,600 motif-implant forward passes |
| **Compute** | single H100 (sessions 1–2) and A100 (session 4); resampling unit = window (χ²/df = 471.8 overdispersion) |
| **Headline** | the collapse is a denominator effect — \|p\| ×**78.5** [75.7, 81.5] while ‖h‖ ×**241.7** [236.2, 247.3]; the state after the onset is linearly recoverable (R² **0.808**), after block 30 it is not (0.270) |

## Setup

Every tap stores the aligned component `p = ⟨h, r̂⟩`, the norm `n = ‖h‖`, the orthogonal remainder and the cosine `a = p/n`; the reference is split exactly into a global **carrier** direction `u` and a per-position **content** direction `v(t)`. Gates run first (self-tests, operator map, onset stability, branch-reconstruction `x_in + mixer + mlp = x_out`, unembedding identification); the analysis plan is hashed and frozen before the extraction pass (`python -m exp1 step6-prereg`). See [`code/README.md`](code/README.md) for the step-by-step pipeline.

## Results

| Question | Result |
|---|---|
| Q0 reproduce the event | onset 28 / rotation 30 from an independent code path; reconstruction gate worst relative error 4.24 × 10⁻³; unembedding gate separates from a row-permutation null by 110× |
| Q1 numerator or denominator? | aligned component flips +1.529 → −50.86, exactly the block-28 MLP projection (−51.04; residual 8 × 10⁻⁴) |
| Q2 where does it come from? | b28/b29 MLP output maps: stable rank 9.34 / 3.23 vs mean 177 for blocks 0–27 (σ₁ 6.1× / 11.4× the median); the onset write has effective rank 8.9; b30/b31 MLPs were never trained; all replicated in `evo2_7b_base` |
| Q3 what is the carrier? | a single coordinate, **#3756** (90.7 % of ‖u‖²), which the final RMSNorm amplifies most (1st of 4,096, 22.8× the median); same on chr17 and in both checkpoints |
| Q4 can the state be recovered? | held-out R² (top-50 PCs): h26→h27 0.937, **h28→h27 0.808**, **h30→h27 0.270** (EXP2 re-analysis: ≈ 0.35) |
| Q5 what survives? | information after the onset relative to block 27: entropy 1.47, margin 1.44, next base 1.07 vs GC 0.73, intergenic 0.45, donor-vs-intron 0.20; after block 30 even the current base drops to 0.51; a kernel decoder adds ≤ 1.5 % |
| Q9 back to sequence (first intervention) | implanting motifs vs composition-matched scrambles moves the block-27 content channel in 5 of 6 families; a canonical splice donor changes the state (KL +0.331 nats) but not the output; response radius ~25 bp |
| Q10 the GDTR readout on this panel | splice donor vs intron: settling depth *d* = −0.023 vs a per-layer maximum \|*d*\| = 1.261 of the same metric family; a dinucleotide shuffle moves c(t) by 0.95 layers |

Survival also tracks how much the model's own output already encodes each task (Spearman 0.769 over 13 tasks), but only 0.515 (*p* = 0.128) once three tasks that are functions of the output are removed — reported as a pilot, not a law. All numbers: [`docs/EXP1_summary_ko.md`](docs/EXP1_summary_ko.md) (§ per question; raw result directories stayed on the GPU volume).

<p align="center"><img src="../figures/figB_compression.png" width="760" alt="Output-relevant information rises and annotation information falls across the onset; recoverability drops at block 30"></p>
<p align="center"><sub>(a) arch_compare probes; (b) EXP1 capacity-matched decoder ladder; (c) recoverability of the last pre-onset state — <code>../figures/figB_compression.png</code>.</sub></p>

## Takeaway

- Neither discard nor faithful preservation: the handoff is a **lossy compression toward what the output head needs**, and the irreversible bottleneck is block 30, not the onset.
- The event is written in the weights as a *rank* statement (low-rank, high-gain maps), which is why a row-norm statistic had missed it.
- Correlational routes back to sequence failed (k-mer "motifs" were mononucleotide bias); the first intervention shows the content channel responds to base order — about the model, not yet about biology.

## Files

| File | What it is |
|---|---|
| [`code/exp1/`](code/exp1/) | package: extraction, branch discovery (`blockmap.py`, `modelio.py`), metrics, nulls, statistics, motifs/perturbations, pre-registration, self-test |
| [`code/configs/`](code/configs/), [`code/run_all.sh`](code/run_all.sh), [`code/requirements.txt`](code/requirements.txt) | model/panel configs and the ordered pipeline |
| [`code/rehearsal_report.json`](code/rehearsal_report.json) | 20-step rehearsal on a synthetic reference with a mock model, 0 critical failures |
| [`docs/EXP1_summary_ko.md`](docs/EXP1_summary_ko.md) | full results summary (Korean): methods, numbers, 17 bugs found, audit corrections, scorecard |

Self-test re-run on CPU for this portfolio: `cd code && python -m exp1 selftest` → **35 passed, 0 failed**.

---
<sub>[← arch_compare](../arch_compare/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [EXP2 →](../exp2/README.md)</sub>

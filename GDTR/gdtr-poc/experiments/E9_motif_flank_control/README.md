<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../../README.md) › [GDTR](../../../README.md) › [Line A](../../README.md) › **E9 · Motif vs flank**</sub>

# 🔬 E9 · v11 control — Motif edit vs flank shuffle at splice donors

> **Question —** Is the early settling at splice donors driven by the 2-nt GT motif itself, or by the surrounding sequence context ("grammar")?

| | |
|---|---|
| **Status** | ✅ done (2026-05-04) — added during the v11 revision |
| **Model / data** | Evo 2 7B, γ_cos = 0.39663; 1,000 canonical GT-AG donors on chr22, ±3 kb context, 5 shuffles each; settling depth averaged over ±10 bp |
| **Compute** | one H200, ~32 min |
| **Headline** | flank shuffle (GT kept) lifts c̄ 26.77 → 23.59 (*d* = +0.515, *p* = 4.1 × 10⁻⁵⁹); GT→AA (flank kept) → 27.24 (*d* = −0.086, *p* = 2.3 × 10⁻³²) |

## Setup

Three paired conditions per donor ([`scripts/exp2_shuffled_motif_control.py`](../../scripts/exp2_shuffled_motif_control.py)): (1) the real sequence; (2) the ±100 bp flank dinucleotide-shuffled with the central GT preserved; (3) GT replaced by AA with the flank preserved. Paired Wilcoxon tests against the real donor.

## Results

| Condition | mean c̄ | median | Cohen's *d* vs real | paired Wilcoxon *p* |
|---|---|---|---|---|
| real GT-AG donor | 26.77 | 28.90 | — | — |
| flank shuffled, GT kept | **23.59** | 23.38 | **+0.515** | 4.1 × 10⁻⁵⁹ (two-sided) |
| GT → AA, flank kept | 27.24 | 29.86 | −0.086 | 2.3 × 10⁻³² (one-sided) |

Source: [`results/exp2_shuffled_meta.json`](../../results/exp2_shuffled_meta.json); the paper's appendix table of motif and flank controls reports the same values.

## Takeaway

- The two edits move depth in **opposite** directions: breaking the motif delays settling slightly, removing the real context makes the lone motif settle much earlier.
- Read as: real donors need deeper context integration than the motif alone — but depth alone cannot rule out that on shuffled flanks the model simply "gives up" and commits early (the paper states this caveat).
- The interventions cover one locus class (canonical donors), so they do not establish causal circuits genome-wide.

## Files

| File | What it is |
|---|---|
| [`scripts/exp2_shuffled_motif_control.py`](../../scripts/exp2_shuffled_motif_control.py) | the perturbation experiment |
| [`results/exp2_shuffled_meta.json`](../../results/exp2_shuffled_meta.json) | all numbers above |

---
<sub>[← E8](../E8_entropy_control/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [E10 →](../E10_v8_revision/README.md)</sub>

<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GDTR](../../README.md) › [Line C · Handoff](../README.md) › **arch_compare**</sub>

# 🔬 C0 · arch_compare — Locating the handoff across chromosomes, scales and models

> **Question —** Where exactly does the representation-to-output handoff happen, is its detection robust across chromosomes, model scales and architectures, and what does it imply for *which layer to read*?

| | |
|---|---|
| **Status** | 🔄 runs 2026-09-16 → 09-21 (archived 2026-09-21); base measurements of the ongoing paper |
| **Model / data** | Evo 2 7B (1M context), 40B (FP8), 1B; HyenaDNA-medium, NT-v2 (plus Caduceus-PS and GPN-MSA comparisons); 100-window panels on chr22, chr17, chr19; downstream: phyloP, ClinVar missense, BRCA1 saturation genome editing, BEND gene finding |
| **Compute** | Slurm GPU cluster; hidden-state dumps of ~355 GB stayed on the host |
| **Headline** | onset **b28** in 7B (norm ratio 214–252× vs ≤ 3.51 in every earlier block) and **b21** in 40B (256–272× vs ≤ 5.26), identical on chr22/chr17/chr19 |

## Setup

`extract_any.py` / `extract_hf.py` store per-block residual states for each panel; `analyze_layers.py` computes the block-to-block norm ratio, detects the onset as the first block whose ratio exceeds a threshold *T* (stable for any *T* in the empty interval between regimes; the detector abstains if no such gap exists), and evaluates region-probe AUROC per block, effective rank, and a label-free reading rule — *read at onset − 4*. Follow-up scripts add causal branch ablations on 40B (`atlas40*.py`, `radial40.py`), recoverability (`exp1_40b.py`) and downstream transfer (`phylop_layers.py`, `clinvar_eval.py`, `brca1_eval.py`, `bend_eval.py`).

## Results

| Measurement | Result | File |
|---|---|---|
| Onset detection | 7B: b28 on all three chromosomes; 40B: b21; empty interval 3.51–214 (7B) and 5.26–256 (40B) | `chr*_evo2_*/profile.json`, [`../figures/figure_data.json`](../figures/figure_data.json) |
| Other architectures | HyenaDNA-medium (max ratio 3.21) and NT-v2 (3.12): no gap → detector abstains | [`results/PIPELINE_STATUS.tsv`](results/PIPELINE_STATUS.tsv), [`chr22_hyenadna_med/analysis.json`](results/chr22_hyenadna_med/analysis.json) |
| Rank collapse | 7B chr22 effective rank 478.7 (b27) → 11.1 (b28) → 1.9 (b30); 40B chr19 303.7 (b20) → 6.5 (b21) | [`chr22_evo2_7b_utr/analysis.json`](results/chr22_evo2_7b_utr/analysis.json), [`chr19_evo2_40b_all/analysis.json`](results/chr19_evo2_40b_all/analysis.json) |
| Output gains, state loses | next-base probe accuracy 0.337 (b0) → 0.645 (b28) → 0.669 (post-norm); splice-donor probe AUROC 0.965 (b27) → 0.873 (b30) | [`surviving_dims.json`](results/surviving_dims.json), [`chr22_evo2_7b_utr/analysis.json`](results/chr22_evo2_7b_utr/analysis.json) |
| Where to read (onset − 4) | mean AUROC regret vs the best layer chosen in hindsight: 0.0050 / 0.0036 / 0.0095 (7B chr22/17/19), 0.0092 (40B chr19) vs 0.0294 (HyenaDNA) and 0.0427 (NT-v2) | [`results/PIPELINE_STATUS.tsv`](results/PIPELINE_STATUS.tsv) |
| 40B causal atlas (ΔNLL, 95 % CI) | b21 MLP 0.452 [0.414, 0.491], b21 mixer 0.481, b23 mixer 0.523, **b23 MLP 0.000**; same-class controls b18–b20 0.04–0.24 | [`atlas40_ci.json`](results/atlas40_ci.json) |
| 40B recoverability of b20 | 0.727 from b21 → 0.235 from b23 (chr22) | [`exp1_40b.json`](results/exp1_40b.json) |
| Conservation transfer | phyloP (non-coding) Spearman 0.258 at b24 vs 0.112 at the post-norm state (7B chr22) | [`phylop_layers.json`](results/phylop_layers.json) |

<p align="center"><img src="../figures/figA_existence.png" width="760" alt="Norm ratio, causal atlas and direction geometry in Evo 2 7B and 40B"></p>
<p align="center"><sub>Norm ratios (a, d), causal atlases (b, e) and state direction (c, f); panels (b) use the EXP2 panel — <code>../figures/figA_existence.png</code>.</sub></p>

## Takeaway

- The handoff is a sharp, reproducible event in both released Evo 2 scales, at 87.5 % of depth in 7B but 42 % in 40B — depth fraction is not the aligning coordinate, the onset is.
- Its absence in HyenaDNA and NT-v2 lines up with their larger penalty for reading at the end, which gives a practical, label-free reading rule for models that do have a handoff.
- The norm ratio only *locates* the event; EXP2 shows that the size of the jump is causally inert and the direction written is what matters.

## Files

| File | What it is |
|---|---|
| [`scripts/extract_any.py`](scripts/extract_any.py), [`extract_hf.py`](scripts/extract_hf.py), [`extract_40b.py`](scripts/extract_40b.py) | hidden-state extraction (7B/1B via Vortex, other models via Hugging Face) |
| [`scripts/analyze_layers.py`](scripts/analyze_layers.py), [`layer_select.py`](scripts/layer_select.py) | onset detection, per-block probes, reading rule |
| [`scripts/atlas40*.py`](scripts/), [`radial40.py`](scripts/radial40.py), [`fact40.py`](scripts/fact40.py), [`exp1_40b.py`](scripts/exp1_40b.py) | 40B causal atlas, dose, factorial, recoverability |
| [`scripts/phylop_layers.py`](scripts/phylop_layers.py), [`clinvar_eval.py`](scripts/clinvar_eval.py), [`brca1_eval.py`](scripts/brca1_eval.py), [`bend_eval.py`](scripts/bend_eval.py), [`gpnmsa_eval.py`](scripts/gpnmsa_eval.py), [`caduceus_probe.py`](scripts/caduceus_probe.py) | downstream transfer and other-model comparisons |
| [`results/`](results/) | every non-array output (JSON/TSV); [`PIPELINE_STATUS.tsv`](results/PIPELINE_STATUS.tsv) logs each run's headline |
| [`configs/evo2-7b-1m-noflash.yml`](configs/evo2-7b-1m-noflash.yml) | model config used without flash-attention |

Left out: Slurm job files and shell helpers (cluster paths and user names), job logs, one-off in-place patch scripts, 355 GB of hidden-state dumps. Hard-coded cluster paths in the scripts were replaced by `/path/to/TDiG`.

---
<sub>[← Line C](../README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [EXP1 →](../exp1/README.md)</sub>

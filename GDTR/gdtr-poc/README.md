<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GDTR](../README.md) › **Line A · gDTR-PoC**</sub>

# 🧪 Line A · gDTR-PoC — building and validating the settling-depth lens

The code, results and write-ups behind the [GDTR paper](https://github.com/heoneyzi/Paper/blob/main/GDTR/README.md): a pre-registered proof of concept on HyenaDNA, calibration on Evo 2 7B, replication on a second chromosome and three more architectures, variant analyses, and the controls added during revision.

![Line A: GDTR lens](https://img.shields.io/badge/Line%20A-GDTR%20lens-2563eb?style=flat-square) ![Status: Done · paper accepted](https://img.shields.io/badge/Status-Done%20%C2%B7%20paper%20accepted-16a34a?style=flat-square) ![Tests: 21 passed · 6 skipped](https://img.shields.io/badge/Tests-21%20passed%20%C2%B7%206%20skipped-334155?style=flat-square)

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

GDTR 논문의 실험 코드와 결과를 단계별로 정리한 폴더입니다.
작은 모델(HyenaDNA)로 방법이 통하는지 먼저 확인한 뒤(E0), Evo 2 7B에서 기준값을 정하고(E1), 다른 염색체·변이·모델에서 재현했으며(E2–E7), 논문 수정 단계에서 엔트로피·모티프 대조 실험과 염색체 간 전이 검증을 추가했습니다(E8–E10).
각 단계는 미리 정한 통과 기준(gate)과 자동 검증 스크립트로 확인했습니다.
비유하자면 새 온도계를 만든 뒤 여러 장소와 조건에서 같은 값을 주는지 하나씩 확인하는 과정입니다.

</details>

> [!NOTE]
> **Provenance.** Curated copy of the team's proof-of-concept repository `darejinn/gDTR-PoC` (MIT, © 2026 Yoonjin Cho — see [`LICENSE`](LICENSE)). Code and result files are unchanged except that hard-coded server roots were replaced with `/path/to/…` placeholders and host names were redacted. Manuscript sources, DOCX drafts, revision memos, PDF duplicates of figures, logs and data caches were left out.

## 🔬 Experiments

| # | Question | Setup | Key result | Folder |
|---|---|---|---|---|
| E0 | Does a DTR-style lens work on a DNA LM? | HyenaDNA-medium-160k (8 blocks), TP53 + BRCA1, RTX 3090 | TP53 exon vs intron *d* = −1.018; BRCA1 *d* = −0.78; tuned lens lifts L7 monotonicity 0.120 → 0.917 | [E0](experiments/E0_phase0_hyenadna/README.md) |
| E1 | Calibrate on Evo 2 7B | chr22, 12,978 × 6 kb windows, H200 | block 31 idle; γ_cos = 0.3966; splice donor c̄ 25.57 vs intron 27.82 | [E1](experiments/E1_phase1_evo2_calibration/README.md) |
| E2 | Replicate on chr17 | 27,586 windows, γ frozen | ranking replicates (intron ↔ 3′UTR swap, < 0.1 layer); donor minimum 24.06 at +20 bp | [E2](experiments/E2_phase2_chr17_replication/README.md) |
| E3 | Variant information | 15 cancer genes, 8,008 labelled ClinVar SNVs | ΔD_cos AUROC 0.844; + Evo 2 ΔLL 0.861 (DeLong *p* = 3.6 × 10⁻¹⁵) | [E3](experiments/E3_phase3_clinvar/README.md) |
| E4 | Cross-architecture | Evo 2, HyenaDNA-large, NT-v2, DNABERT-2 | two tiers: ρ +0.516 (causal LMs), +0.663 (MLMs), negative across | [E4](experiments/E4_phase4_cross_architecture/README.md) |
| E5 | Depth vs conservation | chr22 × phyloP 100-way | Q2 "deep, not conserved" = 3.71 % of chr22 | [E5](experiments/E5_phase5_conservation/README.md) |
| E6 | Variant diagnostics & baselines | per-layer ablation, bootstrap, 4 methods | ‖Δh‖₂ 0.926 > ΔD_cos 0.844 > rollout 0.672 > IG 0.527 | [E6](experiments/E6_tier1_variant_diagnostics/README.md) |
| E7 | Robustness | HP grid, failure modes, cost, Q2 overlaps | AUROC range 0.0017 across 9 classifier settings | [E7](experiments/E7_tier2_robustness/README.md) |
| E8 | Entropy control | 120 chr22 windows (720,000 positions) | donor *d* −0.452 → −0.583 after removing entropy | [E8](experiments/E8_entropy_control/README.md) |
| E9 | Motif vs flank | 1,000 GT-AG donors × 5 shuffles | flank shuffle *d* = +0.515; GT→AA *d* = −0.086 | [E9](experiments/E9_motif_flank_control/README.md) |
| E10 | Revision checks | held-out chr17, motif classes, consequences, cCREs | 94.6 % transfer; KW *p* = 3.0 × 10⁻¹⁰; cCRE-ELS *d* = −0.190 | [E10](experiments/E10_v8_revision/README.md) |

Timeline: Phase 0 finalized 2026-04-26 → Phase 1 2026-04-27 → Phases 2–5 and most of Tiers 1–2 by 2026-04-28 → revision v8 (E10) and v11 controls (E8–E9) 2026-05-04 → camera-ready figures June 2026.

## 🗂️ Code map

```text
gdtr-poc/
├── src/                    ← lens + settling-depth library (gdtr.py, ur_gdtr*.py cosine lens,
│                             logit_lens*.py, tuned_lens.py, calibration.py, variant_delta.py, stats.py)
├── scripts/                ← numbered pipeline: 0x Phase 0 · 1x Phase 1 · 2x Phase 2 · 3x Phase 3 ·
│                             4x Phase 4 / Tier 1 · 50 Phase 5 · p1a–p3b revision v8 · exp1/exp2 v11 controls ·
│                             verify_*.py stage verifiers · figures/ figure scripts
├── tests/                  ← pytest suite (21 pass, 6 skip without GPU weights)
├── paper_figures/          ← camera-ready figures + local (no-GPU) regeneration scripts
├── docs/                   ← per-phase findings, pre-registered decisions, Phase 0 design, v8 results summary
├── results/                ← JSON/CSV/PNG outputs, one folder per stage (as in the original repo)
├── experiments/E0…E10/     ← one README per experiment, pointing into scripts/ and results/
├── data/                   ← DATA_VERSIONS.txt, MODEL_REVISIONS.txt (pinned inputs)
└── requirements*.txt · env_setup.sh · LICENSE
```

Key definitions (from `src/gdtr.py`, `src/ur_gdtr_evo2.py`): `D_cos(ℓ,t) = 1 − cos(h_ℓ(t), h_norm(t))`; settling depth `c(t)` = first **1-based** layer whose running minimum ≤ γ (value `L` means "never crossed"); γ = q70 of the running minimum at the penultimate layer.

## ♻️ Reproduce

```bash
bash env_setup.sh                                   # Phase 0 environment (pinned: requirements.txt)
python -m pytest -q tests                           # 21 pass / 6 skip on CPU
python scripts/00_smoke_test.py && python scripts/01_sanity_check.py       # Phase 0 gates (HyenaDNA, ~38 min, RTX 3090)
bash scripts/run_phase1_all.sh                      # Phase 1 on Evo 2 7B (H200; set paths in the script)
python scripts/exp1_entropy_correlation.py          # E8
python scripts/exp2_shuffled_motif_control.py       # E9
python paper_figures/scripts/regen_fig_shallowness_local.py   # no-GPU figure rebuild
```

Inputs: GRCh38 primary assembly, GENCODE v44, ClinVar 2026-04-18, phyloP 100-way, ENCODE SCREEN v3, RepeatMasker, GTEx v8, GWAS Catalog (versions in [`data/DATA_VERSIONS.txt`](data/DATA_VERSIONS.txt)). Model revisions are pinned in [`data/MODEL_REVISIONS.txt`](data/MODEL_REVISIONS.txt). Multi-GB hidden-state caches are not included; scripts regenerate them.

> [!IMPORTANT]
> **Scope notes** — Every gate and threshold was fixed before the corresponding run (see `docs/`); several pre-registered hypotheses failed and are reported as such (e.g. Phase 0's intron > exon direction reversed on the whole of chr22; the "deep & not conserved = functional" idea was dropped after E10). Paths inside the scripts point to `/path/to/gDTR`; adjust before running.

---
<sub>[← GDTR program](../README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [Line B: TDiG →](../tdig/README.md)</sub>

# Design decisions log

Decisions across 6 planning iterations (2026-05-24). Each entry records what was decided, what was reversed, and why.

---

## Iteration 1 — initial CLM/MLM dual-track plan (superseded)

- Proposed metric family: M1 direction, M2 magnitude, M3 JSD, M4 tortuosity, M5 step-wise, M6 trajectory vector
- Workstreams for 4 team members
- Target: 5/27 YAICON final presentation

**Reversed in iter 2+**: JSD, M5 step-wise, MLM transferability, hard timeline.

---

## Iteration 2 — narrow to Evo 2 CLM, lock L\* = 29

- **Lock**: $L^\star = 29$ as canonical interpretively-distinct tap (carried from gDTR-PoC Phase 1's L31-idle finding)
- **Drop**: JSD — genomic vocab $|V| = 4$ makes distribution comparison weak; cosine/L2/Mahalanobis suffice
- **Drop**: M5 step-wise as primary metric — only relevant for MLM, which is deferred
- **Defer**: MLM (NT-v2, DNABERT-2) transferability to Phase 2
- **Reasoning**: Phase 1 should establish a coherent CLM-focused framework before generalizing across architectures

---

## Iteration 3 — separate M2 into direction and magnitude; 3 reference variants

- **Decision**: M2 unbundles into a clean **magnitude-only** metric. Direction stays as M1 (cosine); magnitude becomes $\lvert r - 1\rvert$ where $r = \lVert h_\ell\rVert / \lVert h_{\text{ref}}\rVert$.
- **Reasoning**: The previous unified L2 residual proposal mixed direction and magnitude — useful as a consistency check but obscures the 2D signature analysis.
- **Decision**: 3 reference variants (A no-norm / B both-norm / C current gDTR) applied uniformly across all reference-dependent metrics
- **Reasoning**: Isolates RMSNorm $\gamma$ asymmetry effect — A vs C measures $\gamma$ removal; B vs C measures DTR-style symmetrization; A vs B measures pure $\gamma$.
- **Caveat**: M2 × Ref B is mathematically degenerate (RMSNorm flattens magnitudes); reported diagnostically but not used as a settling cell

---

## Iteration 4 — adopt c_geo (velocity + curvature), drop tortuosity (later reversed)

- **Decision**: New M3 = $c_{\text{geo}}$, defined as a reference-free combination of velocity $v_\ell$ and curvature $\kappa_\ell$. Settling = "all subsequent $k$ stay small" (strict, no running-min).
- **Reasoning**: Reference-free + naturally robust to single-dip artifact + theoretically grounded in Neural ODE / Transformer Dynamics literature + Evo 2 L30 rotation auto-absorbed as velocity spike
- **Decision (this iteration)**: tortuosity $\tau$ subsumed into c_geo's curvature; removed from family
- **Reversed in iter 6**: tortuosity and c_geo measure different things — pointwise dynamics vs cumulative path efficiency
- **Also considered (then dropped in iter 5)**: c_fn functional/causal settling as M5, with H0/H1/H2 falsification framework

---

## Iteration 5 — drop c_fn; drop falsification framework; switch to descriptive science

- **Drop**: c_fn (functional / causal settling) — computationally expensive (per-token layer interventions) and shifts the project from geometric to causal-mechanistic, which is a different kind of paper
- **Drop**: H0 / H1 / H2 falsification framework that was built around c_geo vs c_fn comparison
- **Decision**: Project mode is now **descriptive science** — "spread out all metrics and analyze + visualize convergence patterns + connect to biology"
- **Reasoning**: User instruction prioritizes meaningful research over the narrower falsification frame. Systematic mapping is the value; binary verdicts are not required.
- **Implication**: Visualization (V1–V9) and context-stratification (S4 98-cell heatmap) become the headline deliverables rather than hypothesis-test verdicts

---

## Iteration 6 — add tortuosity back as M5

- **Reverse iter 4**: $\tau$ tortuosity restored as a distinct metric (M5)
- **Reasoning**: c_geo (pointwise velocity + curvature) and $\tau$ (cumulative path efficiency) measure different quantities:
  - A trajectory can have low curvature but high tortuosity (slow consistent drift away from endpoint)
  - Or high curvature but low tortuosity (zigzag that net-cancels)
- **Family now**: M1 direction, M2 magnitude, M3 c_geo (reference-free), M4 Mahalanobis, M5 tortuosity, M6 L2 (consistency)
- **Cell count**: 14 main cells (M1×3 + M2×2 + M3×1 + M4×3 + M5×3 + M6×2)

---

## Assets reused from `gDTR-PoC`

| Asset | Role in TDiG |
|---|---|
| `src/ur_gdtr.py`, `src/tuned_lens.py` | M1 baseline implementation |
| `phase1/scripts/` Evo 2 forward infrastructure | Hidden-state caching pipeline |
| `scripts/40_t11_per_layer_ablation.py` | Per-layer AUROC pipeline, reused for all 14 cells |
| `scripts/41_t14_bootstrap.py` | 1000× bootstrap CI |
| `scripts/45_t12_delta_h.py` | $\lVert h_\ell - h_{\text{ref}}\rVert$ vector arithmetic |
| `scripts/exp1_entropy_correlation.py` (E8) | Entropy gate — required sanity for every cell |
| `scripts/exp2_shuffled_motif_control.py` (E9) | Motif gate — required sanity for every cell |
| `results/phase1.6/chr22_cache.h5` (server snapshot) | Hidden-state input for all cells |
| `results/phase2.1/chr17_cache.h5` (server snapshot) | chr17 replication input |
| `results/phase3_main/variants_features.csv` (regenerable) | T-B ClinVar variant features |
| `results/phase4/per_model_summary.json` + 3 caches | Future cross-architecture validation (Phase 2) |

## What is NOT carried over

- Phase 0 HyenaDNA-medium codepath — superseded by HyenaDNA-large in Phase 4
- LaTeX / figure-build infrastructure — TDiG paper is separate
- v1 / v3 / v4 paper iterations and DOCX history
- Decision documents (`docs/findings/`, `docs/decisions/` from gDTR-PoC) — TDiG writes its own
- JSD-related code — dropped in iter 2

---

## Iteration 13 — STOP AND REDESIGN (2026-05-24)

Mid-pipeline, user delivered a thorough internal critique. Key points accepted:

1. **Reference selection circularity**: "Ref A is healthy → Ref A is production" is
   a tautology. We picked the reference under which our metric is well-behaved,
   then declared the metric well-behaved. The underlying issue is that
   $\cos(h_{28}, h_{\text{norm}}) \approx -0.013$ — the trajectory does *not*
   approach $h_{\text{norm}}$ in direction across layers 0–28; block 30 performs
   a rotation that lands on $h_{\text{norm}}$. So "settling toward $h_{\text{norm}}$"
   is not what any direction-cosine metric is measuring in our range.

2. **Running-min single-dip vulnerability**: M1/M2/M4/M5 used running-min envelope,
   meaning a single noisy dip below γ at layer 5 locks $c(t) = 5$. M3 alone used
   persistence ("all subsequent ≤ γ"). The 0517 meeting explicitly required
   persistence universally.

3. **M2 conceptual misnomer**: Evo 2 pre-RMSNorm magnitude grows monotonically;
   $|r - 1|$ measures *residual accumulation* (a transformer architectural fact),
   not semantic settling. Calling it "magnitude settling" overclaims.

4. **M6 mislabeled as metric**: it's a Pythagorean consistency identity for
   M1+M2; should be a unit test, not a settling cell.

5. **Diagonal Σ for M4 ignores superposition**: GLM features are correlated;
   diagonal Σ assumes independence. (Concurrent with iter 12's M4 revision.)

6. **Downstream validation absent**: 0517 meeting required external evidence
   (downstream task, perplexity, splicing dependence, etc.). We have none yet.

7. **q70 chosen arbitrarily**: inherited from gDTR without sensitivity ablation.

### Stop action

chr22 forward halted at 4,500/12,978 windows (Ctrl-C in tmux).
- `_OLD_tier1_settling_design_v1.parquet` archived (4,500 windows under v1 design)
- `tier2_scalars_subset.h5` and `tier3_raw.h5` preserved (70/100 subset windows; γ-independent raw scalars + raw h_ℓ) — used as validation testbed for v2 design without re-forward
- `_REDESIGN_STOP.json` records the stop reason + v2 changes

### Design v2 changes (committed)

Locked changes (see `metric_definitions.md` v2 for math):
- **Settling 3-definition split**: Def 1 (trajectory stops, M3), Def 2 (→h_29, M1/M5/M4_set Ref A), Def 3 (→h_norm — *no metric captures; reported as honest limitation*)
- **M2 renamed**: "Residual accumulation magnitude" — diagnostic only
- **M6 demoted**: explicit "consistency check / unit test"
- **Persistence check** $W = 3$ rolling window applied to **all settling metrics** except M3 (strict suffix) and M4_set (monotone by construction)
- **M4_set adoption**: Σ_ref-whitened settling distance (replaces v1 Mahalanobis diagonal), Ledoit-Wolf shrinkage, no running-min
- **M5 Option B locked**: RMSNormed trajectory under Ref B (numerator + denom consistent)
- **α/β ablation matrix** for M3: (1,0), (0,1), (1,1), (1,0.5), (0.5,1)
- **γ ablation**: percentiles {50, 70, 90} reported in supplementary; production γ chosen by downstream task signal

### Phase Pre-V validation (added)

Before any new full forward, a **downstream validation experiment** must pass:

> Splice canonical vs non-canonical settling distribution test
> - Use 70 subset windows' raw h_ℓ already on disk
> - Input: `chr22_splice_class_labels.npy` (codebook: 1=GT-AG donor canonical, 5=non-canonical donor)
> - For each metric × ref cell, settling depth at splice donor positions; split canonical vs non-canonical
> - Pass criterion: ≥1 metric × ref with |d| ≥ 0.2, p < 0.05

This addresses critique point (D) "검증의 부재" before committing more compute.

### What got preserved from v1

| Asset | Status |
|---|---|
| `population_stats/per_layer_*.npy` | Reusable (design-independent) |
| `population_stats/sigma_diagonal.npy` | Superseded by full Σ_ref (v2); keep for diff |
| `chr22/tier2_scalars_subset.h5` (70 windows) | Reusable for v2 metric recomputation |
| `chr22/tier3_raw.h5` (70 windows) | Reusable for v2 metric recomputation + M4_set |
| `subset_window_ids.json` | Reusable |
| `window_metadata.parquet` | Reusable |

### What got archived (v1, not deleted)

| Asset | Reason |
|---|---|
| `_OLD_tier1_settling_design_v1.parquet` | Settling depths under v1 (wrong γ, wrong persistence, wrong M4 placeholder). Kept for v1→v2 diff. |

---

## Open questions that remain open

See [`../PLAN.md`](../PLAN.md) §7 for the live list (12 open questions, mostly S1-gating decisions).

# Thesis — why TDiG is meaningful research

## The core question

> **When and how does a genomic foundation model converge?**
>
> Direction, magnitude, trajectory dynamics, distribution, and path efficiency are each separately measurable axes of "settling". By systematically exposing each one's biological mapping, we obtain a multi-axis picture of how the model computes biological grammar.

## Why "spread it out" instead of "pick the best metric"

The existing literature collapses representation convergence into a single number:
- gDTR (Paper 1): cosine settling depth $c(t)$ against $h_{\text{norm}}$
- Paper 2: $\lVert\Delta h_\ell\rVert_2$ as a single classifier feature
- DTR (Chen et al. 2026): JSD lens settling

Each captures one axis of convergence and is **blind by design** to the others. The result: every metric forces an implicit theoretical choice (direction matters more than magnitude, or vice versa) that is rarely defended on its merits.

This project takes the opposite position. Each of the five metrics measures a *different question*:

| Metric | Question it answers |
|---|---|
| M1 $c_{\text{dir}}$ | At what layer does the **direction** match the final frame? |
| M2 $c_{\text{mag}}$ | At what layer does the **magnitude** match the final frame? |
| M3 $c_{\text{geo}}$ | At what layer does **motion** (velocity and curvature) stop? |
| M4 $c_M$ | At what layer does the **distribution** anisotropy match? |
| M5 $c_\tau$ | At what layer does the **remaining path** become direct? |

None subsumes the others; together they map the geometry of representation convergence.

## What makes this research-meaningful (not just metric-zoo work)

**C1. Each metric resolves a known limitation of cosine settling.**

| Limitation of cosine | Resolved by |
|---|---|
| Single-dip artifact (running-min lets a fluke pass) | M3 with "all $k$ stay small" condition; M5 with cumulative path |
| Cross-layer basis mismatch (direction means different things at different layers) | M3 trajectory-intrinsic; M5 path-intrinsic |
| RMSNorm $\gamma$ asymmetry (h_ℓ raw vs h_norm post-γ) | 3 reference variants quantify the effect directly |
| Evo 2 L30 rotation artifact | M2 and M5 are rotation-invariant; M3 absorbs L30 as a velocity spike |

**C2. The dissociation patterns between metrics are themselves the finding.**

If $c_{\text{dir}}$ is small but $c_{\text{mag}}$ is large for a token, the model has committed to a direction without committing to a magnitude — *grammar commitment in progress*. If both are small, the token is in a *simple context* and the model committed quickly. This dissociation, measured across the 14 reference-cell variants, is mechanistically interpretable and obtained from a **single forward pass** — no perturbation experiments needed.

**C3. The biological mapping is systematic, not anecdotal.**

The S4 context-stratification heatmap (14 metric-cells × 7 biological contexts = 98 entries) gives a single dense view of "which axis of convergence is recruited where in the genome". This map is the artifact downstream tasks (T-A splice prediction, T-B variant classification) draw features from, and it is the artifact biologists can reason about directly.

## What this project deliberately does NOT do

| Not doing | Why |
|---|---|
| Falsification framework (H0/H1/H2) | Considered in iter 5, dropped in iter 6. The value here is the systematic mapping, not binary verdicts. |
| Causal/functional intervention metric ($c_{\text{fn}}$) | Considered in iter 4, dropped in iter 6. Pure geometry suffices for the descriptive goal. |
| MLM transferability in Phase 1 | Deferred to Phase 2. Evo 2 CLM focus first; M3 $c_{\text{geo}}$'s reference-free nature provides a natural opening for the MLM extension. |
| Performance-tuning the metrics for downstream AUROC | Downstream tasks (T-A, T-B) are validation — confirming the metrics carry biological signal — not the headline. |
| JSD settling | Considered and dropped; genomic vocabulary is too small for distributional comparison to add resolution. |
| Single-best metric framing | The point is the family, not a winner. |

## Honest assessment of limitations

- **Evo 2 only.** All claims about "genomic foundation model" generalization rely on Phase 4-style cross-architecture replication, which is Phase 2 work for TDiG.
- **chr22 + chr17 only.** Full-genome scaling is out of scope.
- **No causal validation.** A token whose settling pattern looks distinctive may still be functionally irrelevant. The variant-task results in S6 T-B provide partial functional grounding through pathogenicity but not full causal evidence.
- **No interpretability of the metrics themselves.** We measure $c_{\text{dir}}$, $c_{\text{mag}}$, etc., but the connection to specific features inside the residual stream (e.g. via SAEs) is future work.

These are deliberate scope decisions, not unknowns.

## What success looks like

After S6 lands, the project produces:
1. A **14-cell settling table** for every chr22 + chr17 position
2. The **98-cell context heatmap** as the central biological-mapping artifact
3. A **2D signature scatter** (V5) that visually separates grammar-commitment tokens from simple-context tokens
4. **Downstream task results** (T-A splice prediction, T-B ClinVar variant) demonstrating that the family carries biological signal competitive with or complementary to existing single-metric baselines

If the 14 cells turn out highly redundant (Spearman $\rho > 0.95$ across most pairs), the family approach is empirically refuted — and that is also a publishable, honest result.

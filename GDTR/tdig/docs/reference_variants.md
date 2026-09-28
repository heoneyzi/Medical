# Reference variants

Three variants of the reference frame, applied uniformly to all reference-dependent metrics. The variant matrix is the project's main tool for isolating RMSNorm $\gamma$ effects.

## Definitions

For Evo 2 7B with $L^\star = 29$ (canonical interpretively-distinct tap):

### Variant A — "no-norm"

- $h_\ell$: raw, no normalization
- reference: $h_{29}$ raw (the residual-stream state right out of block 29, before any RMSNorm)
- $\gamma$: not applied anywhere

Cosine: $\cos(h_\ell, h_{29})$
Magnitude ratio: $\lVert h_\ell\rVert / \lVert h_{29}\rVert$

**Interpretation.** Symmetric, $\gamma$-free comparison. Measures pure residual-stream geometry.

### Variant B — "both-norm" (DTR-style)

- $h_\ell$: $\mathrm{RMSNorm}(h_\ell)$ — applies the final RMSNorm $\gamma$ to *every* intermediate layer
- reference: $\mathrm{RMSNorm}(h_{29})$
- $\gamma$: applied to both sides (symmetric)

Cosine: $\cos\bigl(\mathrm{RMSNorm}(h_\ell), \mathrm{RMSNorm}(h_{29})\bigr)$

**Interpretation.** Both vectors live in the same post-normalization space. Closest in spirit to the original NLP DTR pipeline (which projects through the same lm_head for every layer). Within-pair $\gamma$ effect cancels in cosine; per-dimension $\gamma$ weights are reflected uniformly.

**Caveat for M2.** RMSNorm flattens all vector magnitudes toward a common value (determined by the $\gamma$ vector's norm). Under Variant B, $\lVert\mathrm{RMSNorm}(h_\ell)\rVert$ is essentially independent of $\ell$ — making $D_{\text{mag}}$ trivially small for all layers. M2 × Ref B is therefore degenerate; we report it for diagnostic completeness but do not use it as a settling cell.

### Variant C — "current gDTR"

- $h_\ell$: raw
- reference: $h_{\text{norm}}$ — Evo 2's post-final-norm output, equal to $\mathrm{RMSNorm}\bigl(h_{30}\bigr)$ (since blocks 30, 31 are saturated / idle)
- $\gamma$: applied to reference only — **asymmetric**

Cosine: $\cos(h_\ell, h_{\text{norm}})$

**Interpretation.** The existing gDTR baseline. The asymmetry is the source of the $\gamma$-distortion concern that motivates Variants A and B.

## Compatibility matrix

|  | A | B | C |
|---|---|---|---|
| M1 $c_{\text{dir}}$ | ✓ | ✓ | ✓ |
| M2 $c_{\text{mag}}$ | ✓ | ⚠ degenerate (report only) | ✓ |
| M3 $c_{\text{geo}}$ | (reference-free) | | |
| M4 $c_M$ | ✓ ($\Sigma$ on raw) | ✓ ($\Sigma$ on normalized) | ✓ ($\Sigma$ on raw, ref = $h_{\text{norm}}$) |
| M5 $c_\tau$ | ✓ | ✓ | ✓ |
| M6 $D_{L_2}$ | ✓ | (potentially degenerate) | ✓ |

**14 main cells**: M1 × 3 + M2 × 2 + M3 × 1 + M4 × 3 + M5 × 3 + M6 × 2.

## Diagnostic value of the variant comparisons

| Comparison | Diagnoses |
|---|---|
| **A vs C** | Effect of removing the $\gamma$ asymmetry — does the existing gDTR baseline distort the settling pattern? |
| **B vs C** | Effect of moving to DTR-style $\gamma$-symmetric — does the original NLP DTR formulation transfer cleanly? |
| **A vs B** | Pure $\gamma$ effect within symmetric pairs — does $\gamma$ change which direction settles first, or only inflate distances? |
| **Reference-dependent (A, B, C) vs M3 reference-free** | Trajectory-intrinsic settling vs reference-induced settling. Where do they agree, where do they not? |

These four comparisons constitute the project's RMSNorm ablation. They are the lower-bound deliverable even if the family approach as a whole shows redundancy.

## Implementation note

All reference variants are computed from the same hidden-state cache. Specifically, given the cache containing $\{h_\ell\}_{\ell=0}^{31}$ and the model's RMSNorm parameters $\gamma_{\text{rms}}, \epsilon_{\text{rms}}$:

- Variant A: use $h_{29}$ directly
- Variant B: compute $\mathrm{RMSNorm}(h_\ell) = \gamma_{\text{rms}} \odot h_\ell / \sqrt{\mathrm{mean}(h_\ell^2) + \epsilon_{\text{rms}}}$ for both $h_\ell$ and $h_{29}$ at query time
- Variant C: load $h_{\text{norm}}$ from cache (already post-RMSNorm in the existing Phase 1 caches)

A single shared helper module (`src/tdig/references/`) provides the three reference-vector accessors; each metric calls into it rather than re-implementing the variants.

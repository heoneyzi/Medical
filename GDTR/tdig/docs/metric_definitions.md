# Metric definitions (design v2 — 2026-05-24)

> **Design v2 changes from v1** (response to internal critique 2026-05-24):
> - Settling concept split into 3 explicit definitions (Def 1 / Def 2 / Def 3)
> - M2 renamed from "magnitude settling" → **"Residual accumulation magnitude"**
> - M6 demoted from metric → **consistency check / unit test**
> - Persistence check (rolling window) applied to **all settling metrics**
> - M4 → **M4_set** (Reference-whitened settling distance with Σ_ref shrinkage)
> - M5 Option **B locked** (RMSNormed trajectory for Ref B)
> - α/β ablation for M3 c_geo
> - Production γ from q70 → **γ ablation matrix (50/70/90 percentiles)**

---

## 0. The three "settling" definitions (must be reported separately)

A foundational issue is that "settling depth" has at least **three distinct meanings** in this project, and previous designs implicitly mixed them. Design v2 makes the choice explicit:

| Def | What "settled" means | Measured by | Caveat |
|---|---|---|---|
| **Def 1** *Trajectory stops evolving* | The residual stream is no longer changing in meaningful ways | M3 c_geo (reference-free) | Cleanest definition; "stop" requires both small velocity AND stable direction |
| **Def 2** *Convergence to $h_{29}$* | Hidden state approaches the canonical interpretively-distinct tap (pre-final-block) | M1 Ref A, M5 Ref A, M4_set Ref A | Pre-rotation; $h_{29}$ is an *intermediate* not a final output. Honest framing: "settling into the pre-final representation" |
| **Def 3** *Convergence to $h_{\text{norm}}$* | Hidden state approaches the final post-norm output (what lm_head consumes) | **No metric in current family captures this** | $\cos(h_{28}, h_{\text{norm}}) \approx -0.013$ (gDTR Phase 1) → trajectory is near-orthogonal to $h_{\text{norm}}$ in layers 0–28; **the L30 block rotates everything into the $h_{\text{norm}}$ direction in a single step**. Direction-based metrics cannot detect approach to $h_{\text{norm}}$; magnitude metrics fail because $\|h_{\text{norm}}\|$ is tiny vs raw $\|h_\ell\|$. **Reported as honest limitation.** |

**Paper-level claim**: report each metric explicitly under its Def label. Avoid claiming "M1 Ref A measures settling" without specifying that this is **Def 2 settling**, not Def 1 or Def 3.

---

## 1. Notation

- $h_\ell(t) \in \mathbb{R}^d$ — residual stream at layer $\ell \in \{0, \ldots, 31\}$, token $t$. $d = 4096$.
- $L^\star = 29$ — canonical interpretively-distinct tap (Evo 2 idle-block fix)
- $h_{\text{norm}}(t)$ — post-RMSNorm output ($\approx \text{RMSNorm}(h_{30})$)
- $r = \|h_\ell\| / \|h_{\text{ref}}\|$ — magnitude ratio
- $\gamma$ — settling threshold (calibrated via population quantile)

---

## 2. Persistence check (uniform protocol)

**Locked in design v2**: all non-monotone settling metrics use a **rolling-window persistence** condition. This replaces the single-dip-vulnerable running-min envelope from v1.

$$c(t) = \min \Bigl\{ \ell : D(k, t) \le \gamma \;\text{ for all } k \in [\ell,\; \ell + W - 1] \Bigr\}$$

where $W$ is the persistence window. **Default $W = 3$** (current layer + next 2). Exceptions:
- M3 c_geo: $W = L - 2 - \ell + 1$ (strict "all subsequent layers"; the gauge $g$ is intentionally non-monotone after standardization, so stricter check is meaningful)
- M4_set: $W = 1$ (the metric is **monotone-decreasing by Σ_ref-whitening construction**, no persistence needed)

**Sensitivity ablation requirement**: every metric reports settling depth at $W \in \{1, 3, 5\}$ in supplementary, with $W = 3$ as production. $W = 1$ corresponds to "first crossing" (old v1 behavior).

---

## 3. Metric family

### M1 — Direction settling (cosine)

$$D_{\text{dir}}(\ell, t) = 1 - \cos\bigl(h^{\text{ref-state}}_\ell, h^{\text{ref-state}}_{\text{ref}}\bigr)$$

**Captures**: direction settling under the chosen reference. Three variants per reference:
- Ref A: both raw → measures Def 2 settling (toward $h_{29}$)
- Ref B: both RMSNormed → DTR-style, γ-symmetric Def 2-like
- Ref C: $h_\ell$ raw vs $h_{\text{norm}}$ → asymmetric; expected to fail per Def 3 limitation

**Limitations (acknowledged explicitly)**:
- **Superposition**: $\cos(h_\ell, h_{\text{ref}}) \approx 1$ does NOT imply same semantic content; per-layer basis can differ, so high cosine may be coincidental alignment. **Not solved by this metric alone**.
- **Early-layer instability**: $\|h_\ell\|$ is small for low $\ell$, cosine numerically noisy.

**Persistence**: $W = 3$.

**γ calibration**: $\gamma_{\text{dir}}$ = q-percentile of $D_{\text{dir}}(L^\star - 1, t)$ over chr22 sanity population. **Ablation matrix: $q \in \{50, 70, 90\}$.**

### M2 — Residual accumulation magnitude (renamed; was "magnitude settling")

$$D_{\text{mag}}(\ell, t) = |r - 1|, \quad r = \|h_\ell\| / \|h_{\text{ref}}\|$$

**Renamed because**: Evo 2 is pre-RMSNorm and residual stream magnitude **grows monotonically across layers** by construction. M2 captures *residual accumulation* (a transformer-architectural artifact), not semantic settling. Acknowledged as a transformer-known phenomenon (Brown et al. 2020; Brody et al. 2023 residual scaling analyses).

**Status**: **diagnostic only**, not a primary settling metric. Reported alongside M1 as the "magnitude axis of (M1, M2) signature" but interpreted as residual accumulation.

**Variants**:
- Ref A: $r = \|h_\ell\|/\|h_{29}\|$ → only meaningful variant
- Ref B: trivially $r \approx 1$ by RMSNorm — **degenerate** (reported once for confirmation)
- Ref C: $\|h_{\text{norm}}\|$ is tiny → r explodes; **unusable**

**Persistence**: $W = 3$.

**γ calibration**: q-percentile ablation $\{50, 70, 90\}$.

### M3 — c_geo (reference-free trajectory dynamics) — **Def 1 primary**

Velocity (scale-invariant):
$$v_\ell(t) = \|h_{\ell+1} - h_\ell\| / \|h_\ell\|$$

Curvature (trajectory angle change):
$$\kappa_\ell(t) = 1 - \cos(h_{\ell+1} - h_\ell,\; h_{\ell+2} - h_{\ell+1})$$

Standardize per-population (chr22 sanity):
$$\tilde v_\ell = (v_\ell - \mu_v) / \sigma_v, \qquad \tilde\kappa_\ell = (\kappa_\ell - \mu_\kappa) / \sigma_\kappa$$

Combined gauge:
$$g_\ell = \alpha \tilde v_\ell + \beta \tilde\kappa_\ell$$

Settling (strict — all subsequent):
$$c_{\text{geo}}(t) = \min \Bigl\{ \ell : g_k(t) \le \gamma_{\text{geo}} \;\text{ for all } k \in [\ell, L - 2] \Bigr\}$$

**α/β ablation matrix (paper-grade requirement)**:

| α | β | meaning |
|---|---|---|
| 1 | 0 | velocity-only |
| 0 | 1 | curvature-only |
| 1 | 1 | symmetric combined (default) |
| 1 | 0.5 | velocity-weighted |
| 0.5 | 1 | curvature-weighted |

Each ablation cell reports per-context Cohen's d at the downstream validation step. The cell with strongest discriminant signal is reported as production; all cells are reported in supplementary.

**Why M3 is the primary settling metric in v2**:
- Reference-free (no Def 1/2/3 ambiguity)
- Persistence built into definition (strict "all subsequent")
- Standardized components reduce arbitrary scaling
- Captures Def 1 cleanly

### M4 — M4_set — Reference-whitened settling distance (new; replaces v1 Mahalanobis)

$$D_{M,\text{set}}(\ell, t) = \sqrt{(h_\ell - h_{\text{ref}})^\top \Sigma_{\text{ref}}^{-1} (h_\ell - h_{\text{ref}})}$$

where $\Sigma_{\text{ref}}$ is the **covariance of the reference vector across the chr22 sanity population**, with shrinkage:
$$\Sigma_{\text{ref}} = (1 - \lambda) \Sigma_{\text{emp}} + \lambda \cdot \frac{\text{tr}(\Sigma_{\text{emp}})}{d} I$$

with $\lambda$ from **Ledoit-Wolf data-driven shrinkage** (preferred) or fixed $\lambda = 0.05$.

**Three properties that make this v1→v2's biggest theoretical win**:

1. **Cross-layer comparability**: all layers measured in the same whitened coordinate system (Σ_ref^{-1/2}). Settling depth is now a well-posed quantity (compare D_M at any two ℓ directly).
2. **Naming honesty**: dropped "Mahalanobis" (which is point-to-distribution-mean); now "Reference-whitened settling distance" — exact description.
3. **Monotone decrease**: for typical samples, $\mathbb{E}[x^\top \Sigma^{-1} x] = d \approx 4096$, so $D_{M,\text{set}}(0, t) \approx \sqrt{d} \approx 64$ and $D_{M,\text{set}}(29, t) = 0$. The metric **decreases monotonically** toward 0 as ℓ → 29 → **running-min envelope and persistence not needed**.

**Three reference variants**: $\Sigma_{\text{ref}}^{(A)} = \text{Cov}(h_{29})$, $\Sigma_{\text{ref}}^{(B)} = \text{Cov}(\text{RMSNorm}(h_{29}))$, $\Sigma_{\text{ref}}^{(C)} = \text{Cov}(h_{\text{norm}})$.

**Settling**: $c_{M,\text{set}}(t) = \min\{\ell : D_{M,\text{set}}(\ell, t) \le \gamma_{M,\text{set}}\}$ — direct first-crossing, no persistence (monotone).

**γ calibration**: q-percentile at ℓ = 28 with ablation $\{50, 70, 90\}$.

**Status in v2**: **upgraded from "deferred" (v1) to first-class metric**. Captures Def 2 settling (Ref A) in a way M1 cannot (no superposition ambiguity at the whitened-space level).

### M5 — Path tortuosity — **Option B locked**

$$\tau(\ell, t) = \frac{\sum_{k=\ell}^{L^\star - 1} \|h^{\text{state}}_{k+1} - h^{\text{state}}_k\|}{\|h^{\text{state}}_\ell - h^{\text{state}}_{\text{ref}}\| + \varepsilon}$$

**Option B** (locked v2): path numerator and denominator both use the **reference variant's state representation**:
- Ref A: $h^{\text{state}}_k = h_k$ (raw) — trajectory is raw
- Ref B: $h^{\text{state}}_k = \text{RMSNorm}(h_k)$ — **trajectory is RMSNormed** (was Option A's raw + normed denom mix in v1)
- Ref C: $h^{\text{state}}_k = h_k$ raw, denom uses $h_{\text{norm}}$ — keeps asymmetric (matches M1 Ref C convention)

**Calibration layer**: $\ell = 27$ (since $\tau(28) = 1$ trivially with denominator $\|h_{28} - h_{29}\|$ = numerator).

**ε smoothing**: 1st percentile of denominator distribution from chr22 sanity.

**Monotonicity diagnostic**: required to report fraction of tokens with monotone-non-increasing τ; target ≥ 95%. If below: strengthen ε or escalate.

**Persistence**: $W = 3$.

**Limitation (acknowledged)**: τ ≈ 1 indicates *straight path*, not *arrival*. A trajectory passing through the reference and continuing also has τ ≈ 1 locally. **M5 alone cannot identify settling; must be co-reported with M1 or M4_set**.

### M6 — D_L2 — **consistency check (NOT a metric)**

> Design v2 explicitly: M6 is a **consistency / unit-test** quantity, NOT a settling metric. Removed from "main metric family" listing.

$$D_{L_2}(\ell, t) = \|h_\ell - h_{\text{ref}}\|_2 / \|h_{\text{ref}}\|_2$$

with the algebraic identity (Pythagorean decomposition under angle):
$$D_{L_2}^2 = (r - c)^2 + (1 - c^2), \qquad c = \cos(h_\ell, h_{\text{ref}})$$

**Used to verify**: that M1 and M2 (Ref A) jointly account for D_L2 behavior. If $D_{L_2}^2 \ne (r - c)^2 + (1 - c^2)$ for any (window, layer, token), there's a numerical bug. Used in `scripts/_verify_outputs.py`.

**Also used as antiparallel audit**: if M2 says "settled" (r ≈ 1) but cosine is negative, D_L2 catches it as still-large.

---

## 4. γ calibration ablation matrix

For each (metric × reference) cell, calibrate γ at **3 percentiles**: q50, q70, q90. Report:
- Splice donor vs intron Cohen's $d$ at each γ
- Per-context settling distribution at each γ
- Bootstrap CI on γ (n=100 sanity windows, 200 resamples)

**Production γ** is chosen as the percentile maximizing downstream task signal (Phase Pre-V splice canonical/non-canonical test), not arbitrarily.

---

## 5. Reference variants matrix (updated for v2)

| Cell | M1 dir | M2 mag | M3 geo | M4_set | M5 tau | M6 (check) |
|---|---|---|---|---|---|---|
| Ref A (no-norm) | ✓ | ✓ | n/a (ref-free) | ✓ | ✓ | ✓ |
| Ref B (both-norm DTR) | ✓ | degenerate | n/a | ✓ | ✓ Option B locked | n/a |
| Ref C (asymmetric current gDTR) | ✓ (degenerate per Def 3) | unusable | n/a | ✓ | ✓ | ✓ |

**Total main settling cells** (excluding M6 consistency check):
- M1: 3 cells
- M2: 1 healthy cell + 2 diagnostic
- M3: 1 cell (ref-free) × 5 α/β variants
- M4_set: 3 cells
- M5: 3 cells

**Production cells (where most analysis lives)**: M1×A, M3 (α=β=1), M4_set×A, M5×A. Other cells are reported for ablation honesty.

---

## 6. Downstream validation requirement (Phase Pre-V)

Per design v2: **no full chr22/chr17 forward without first validating metric design on a downstream task** (response to critique D "검증의 부재").

**Phase Pre-V validation experiment** (cheap, uses 70 subset windows already on disk):

> **Splice canonical vs non-canonical settling distribution test**
> 
> - Input: 70 subset windows (from interrupted chr22 forward), `chr22_splice_class_labels.npy` (codebook 1=GT-AG donor, 5=non-canonical donor)
> - For each metric × reference cell, compute settling depth at each splice donor position within the 70 windows
> - Split positions into canonical (class 1) vs non-canonical (class 5)
> - Statistical test: Mann-Whitney U, Cohen's d
> - Expected: canonical donors should have lower SD (settle earlier) under at least one Def-2 or Def-1 metric

**Pass criterion for proceeding to full forward**: at least 1 metric × reference cell shows |d| ≥ 0.2 with p < 0.05 on the splice canonical vs non-canonical test.

Reported in `Phase_PreV_validation_report.json`.

---

## 7. Computation cost summary (updated for v2)

Per chr22 window (6 kb, 6000 positions) with v2 metric set:

| Component | Cost |
|---|---|
| Forward pass (Evo 2 7B) | ~0.3 s |
| M1, M2, M5 scalars | ~0.1 s |
| M3 v + κ + standardization | ~0.05 s |
| **M4_set (matmul Σ⁻¹ × (T, 4096))** | **~0.3 s** (new addition) |
| Per-window Tier 1 settling depths (post-compute) | ~0.02 s |
| Total | **~0.8 s/window** |

For 12,978 chr22 windows: ~170 min wall. Slower than v1 (~85 min) due to M4_set inline computation, but produces a 4th healthy axis (M4_set Ref A) which v1 didn't have.

---

## 8. What's preserved from v1 (no re-forward needed for these)

- `population_stats/per_layer_*.npy` — design-independent population statistics
- `chr22/tier2_scalars_subset.h5` (70/100 windows complete) — raw cos / norm / step per-layer scalars
- `chr22/tier3_raw.h5` (70/100 windows complete) — raw h_ell + RMSNormed + h_norm
- `subset_window_ids.json`
- `window_metadata.parquet`

What needs to be added for v2:
- `population_stats/sigma_ref_*.npy` — 3 full Σ_ref shrinkage estimates (~400 MB total)
- `population_stats/gamma_calibration_v2.json` — γ ablation matrix per metric × ref × {q50, q70, q90}

What needs to be recomputed at full chr22 (v2 forward):
- `chr22/tier1_settling_v2.parquet` — all settling depths under v2 protocol (persistence W=3, M4_set added, M5 Option B)

---

## 9. Open items (v3 future work)

- Superposition disentanglement: dictionary learning / SAE on residual stream (acknowledged but explicit limitation in v2)
- MLM compatibility: reference-free M3 already works; revisit other metrics for NT-v2 / DNABERT-2 in Phase 2
- Σ_ref full rank vs PCA-top-k vs diagonal — Phase 2 sensitivity
- α/β grid: currently 5 cells, could be denser
- W ∈ {1, 3, 5} ablation matrix for persistence; could go finer

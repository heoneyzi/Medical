# T1.2 + T2.4 verification specification

This document defines the verification protocol that runs once the server chain script (`_chain_t12.sh`) completes. It is invoked as the prompt for a single verification agent.

## When to run

Trigger: `/path/to/ICML/results/_chain_t12_done` exists locally (rsync'd from server) OR `<gpu-host>:/path/to/gDTR/results/_chain_t12_done` exists.

## Inputs to verify

### T1.2 outputs
- Server: `/path/to/gDTR/results/tier1_baselines/`
  - `delta_h_features.csv` (10,910 × 39 cols: 7 meta + 32 delta_h_norm_*)
  - `rollout_features.csv` (10,910 × 22 cols: 7 meta + 15 rollout-related — ref/alt/delta × 5 attn layers)
  - `ig_features.csv` (10,910 × 13 cols: 7 meta + 3 ig + 3 config)
  - `baseline_auroc.json` — full AUROC + DeLong + incremental info
  - `baseline_spearman.csv` (4×4 ρ matrix)
  - `delong_pairs.csv` (3 rows: A vs B, A vs C, A vs D)
  - `_done` (final marker with `{"ok": true, "n": 8008}` payload)

### T2.4 outputs
- Server: `/path/to/gDTR/results/tier2_compute/`
  - `cost_benchmark.csv` (4 rows × 6 cols: method, n_variants, total_wall_sec, mean_per_variant_ms, peak_vram_gb, multi_gpu_scalable)
  - `_done`

## Verification protocol

### 1. File integrity
- All `_done` markers present
- All CSVs/JSONs parse without error
- No empty files; no NaN columns

### 2. Schema compliance
- `delta_h_features.csv` columns: `chrom, pos, ref, alt, gene, category, stars, delta_h_norm_0, …, delta_h_norm_31`
- `rollout_features.csv` includes `rollout_3_ref, rollout_3_alt, rollout_3_delta` etc. for layers {3, 10, 17, 24, 31}
- `ig_features.csv` includes `ig_score_ref, ig_score_alt, ig_score_delta, ig_n_steps, ig_target_layer, ig_context_half`
- `baseline_auroc.json` keys: `phase, config, n_train_variants, results_stratified, results_logo, delong_pairs, incremental_info, wall_time_sec`

### 3. Statistical sanity
- All AUROC ∈ [0.5, 1.0]; flag any ∈ [0.5, 0.55] (suspicious)
- Stratified mean AUROC point-estimate ∈ [ci95_lo, ci95_hi]
- Spearman ρ symmetry: `spear[i,j] == spear[j,i]` and diagonal == 1.0
- DeLong: `auc_a − auc_b == delta` exactly; `p_value` matches normal-CDF(z) computation
- Incremental info: `auroc_residual_dD_minus_baseline` should be > 0.5 (if not, ΔD adds nothing beyond that baseline; flag)
- Cost: ms/variant should follow IG > gDTR ≈ rollout > Δh (Δh is forward-only with no extra compute beyond hidden-state extraction)

### 4. Independent re-derivation (CRITICAL)

```python
# Recompute one key number from scratch and compare to claimed value
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ph3 = pd.read_csv("/path/to/gDTR/results/phase3_main/variants_features.csv")
dh = pd.read_csv("/path/to/gDTR/results/tier1_baselines/delta_h_features.csv")

# Merge on (chrom, pos, ref, alt) — both have these
df = ph3.merge(dh[["chrom","pos","ref","alt","gene","category"] + [f"delta_h_norm_{i}" for i in range(32)]],
               on=["chrom","pos","ref","alt"], how="inner",
               suffixes=("","_dup"))
df = df[df["category"].isin(["P_LP","B_LB"])]
y = (df["category"] == "P_LP").astype(int).values

# Re-train ‖Δh‖ vector LR on stratified 10-fold seed=42 — should match B_delta_h.mean_auroc within 1e-3
X = df[[f"delta_h_norm_{i}" for i in range(32)]].values
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
aurocs = []
for tr, te in skf.split(X, y):
    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:,1]
    aurocs.append(roc_auc_score(y[te], p))
my_b_auroc = np.mean(aurocs)

# Compare to claimed B_delta_h
import json
auroc_json = json.load(open("/path/to/gDTR/results/tier1_baselines/baseline_auroc.json"))
claimed = auroc_json["results_stratified"]["B_delta_h"]["mean_auroc"]
assert abs(my_b_auroc - claimed) < 0.001, f"B_delta_h mismatch: my={my_b_auroc:.4f} vs claimed={claimed:.4f}"
print("✓ B_delta_h re-derivation matches")
```

Repeat for one of {C_rollout, D_ig} as a second cross-check.

### 5. Cross-consistency
- Phase 3 cached `variants_features.csv` ΔD_cos AUROC must reproduce 0.844 ± 0.003 in this run (this is the A_dD_cos slot of the new table)
- T1.1's per-layer ablation already showed `argmax_layer_jsd = L29`; the L=29 column should now appear with high LR coefficient when refit (sanity check)

### 6. Cost benchmark sanity
- IG's `mean_per_variant_ms` should be approximately consistent with the IG rate in chain logs (0.59 var/s ≈ 1.7 sec/var ≈ 850 ms per ref+alt pair = 425 ms for one direction, but the benchmark times only one direction or both — verify which by reading `49_t24_cost.py`'s `bench_ig` definition)
- Peak VRAM should be the same order of magnitude as the live `nvidia-smi` reading (19 GB for IG)
- All multi_gpu_scalable values are valid booleans/strings

### 7. Reproducibility
Re-run only the comparison pipeline (`scripts/48_t12_compare_pipeline.py`) — it should be deterministic with seed 42 and produce bit-for-bit identical AUROC values.

## Output

Verification report at `/path/to/ICML/results/_verification/T12_T24_verification.md` covering:
1. File integrity status (per dir)
2. Schema compliance status
3. Statistical sanity
4. Independent re-derivation table (claimed vs computed for ≥ 2 methods)
5. Cross-consistency
6. Cost benchmark sanity
7. Reproducibility test
8. **Verdict**: PASS / N issues / FAIL with reasons
9. List of any flagged values for paper-readiness

## Constraints
- READ-ONLY — do not modify any results or scripts
- Do not occupy GPU (chain script already finished by definition of trigger)
- Use scratch dir `/path/to/gDTR/results/_verification/` for any intermediate compute
- Total budget: < 15 min wall-clock

## Final triggers (next steps if PASS)
- `python scripts/figures/F4.py` — auto-generates F4 (already implemented to use the spec'd schemas)
- Run `scripts/figures/S7.py` for compute cost (TODO — write after T2.4 lands; see TODO list)
- Replace [PLACEHOLDER_X] tokens in `ICML_MANUSCRIPT_DRAFT.md` §4.4 with actual values from `baseline_auroc.json`
- Make commit #4: "Add T1.2 baselines + T2.4 cost; F4 + §4.4 final"
- Push to `origin/main`

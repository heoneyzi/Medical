"""Exp H3 — L28→L29 and L29→L30 SVD mechanism analysis.

Approximate Evo 2's late-layer transitions as linear maps T_ℓ such that
h_{ℓ+1} ≈ T_ℓ @ h_ℓ, fit via OLS over many (window, token) pairs from
tier3, then SVD the T matrices. Compares L28→L29, L29→L30, L30→L31 plus
nearby controls (L25→L26, L15→L16).

Rotation hypothesis: T_29 should have singular values near 1 (orthogonal-like)
indicating no expansion/compression, just direction change.

Outputs: results/L29_svd/
  singular_values.csv        per transition layer: top-K singular values
  svd_summary.png            multi-panel SVD spectra comparison
  alignment_metrics.json     orthogonality + condition number per transition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32
HIDDEN = 4096


def fit_linear_transition(X, Y, ridge=1e-3):
    """OLS h_{l+1} ≈ T @ h_l. X = (N, H), Y = (N, H). Returns T = (H, H)."""
    N, H = X.shape
    XtX = X.T @ X + ridge * np.eye(H, dtype=X.dtype)
    XtY = X.T @ Y
    return np.linalg.solve(XtX, XtY)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/L29_svd"))
    p.add_argument("--transitions", nargs="+", type=int,
                   default=[15, 25, 27, 28, 29, 30],
                   help="layer indices ell s.t. we fit T: h_ell -> h_{ell+1}")
    p.add_argument("--n-samples", type=int, default=50000,
                   help="random tokens for fit")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print(f"[load] tier3 raw_h_ell ...")
    with h5py.File(args.tier3, "r") as h5:
        # raw_h_ell shape (100, 32, 600, 4096) fp32 — load all needed layers
        n_w, n_l, n_t, H = h5["raw_h_ell"].shape
        print(f"  shape ({n_w}, {n_l}, {n_t}, {H})")
        rng = np.random.default_rng(args.seed)
        # Random token indices
        total = n_w * n_t
        n_take = min(args.n_samples, total)
        all_idx = rng.choice(total, size=n_take, replace=False)
        wi = all_idx // n_t
        ti = all_idx % n_t
        print(f"  taking {n_take} tokens (wi, ti pairs)")
        # Load only the needed layers
        needed_layers = sorted(set(args.transitions + [t + 1 for t in args.transitions]))
        print(f"  needed layers: {needed_layers}")
        h_data = {}
        for ell in needed_layers:
            h_data[ell] = h5["raw_h_ell"][:, ell][wi, ti, :].astype(np.float32)  # (n_take, H)
        print(f"  loaded; per-layer matrix shape {h_data[needed_layers[0]].shape}")

    # Fit + SVD per transition
    records = []
    metrics = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, ell in enumerate(args.transitions):
        X = h_data[ell]; Y = h_data[ell + 1]
        # Center
        X_mu = X.mean(0); Y_mu = Y.mean(0)
        Xc = X - X_mu; Yc = Y - Y_mu
        print(f"\n[fit] transition L{ell}→L{ell+1} ridge={args.ridge}")
        T = fit_linear_transition(Xc, Yc, ridge=args.ridge)
        # Norm
        T_fro = float(np.linalg.norm(T, "fro"))
        # SVD — only top-100 singular values to save memory
        print("  SVD ...")
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        # Reconstruction quality
        Y_pred = Xc @ T
        resid = Yc - Y_pred
        r2 = 1.0 - (resid.var() / Yc.var())
        # Orthogonality measure: how close is T to T (i.e. how flat is S?)
        s_norm = S / S.max()
        # rank effective: # of s_i above 0.01 of max
        r_eff = int(np.sum(s_norm > 0.01))
        # condition number
        cond = float(S[0] / max(S[-1], 1e-12))
        # "rotation-like" score: fraction of energy in s in top-K equal to expected for rotation
        # If perfect orthogonal, S would be constant. Measure as: 1 - normalized std.
        rot_score = float(1.0 - np.std(S) / np.mean(S))
        metrics[f"L{ell}_to_L{ell+1}"] = {
            "ridge": args.ridge, "n_samples": n_take,
            "T_frobenius": T_fro,
            "T_R2": float(r2),
            "S_max": float(S.max()), "S_min": float(S.min()),
            "S_median": float(np.median(S)),
            "condition_number": cond,
            "effective_rank_pct_1": r_eff,
            "rotation_score_1_minus_std_over_mean": rot_score,
            "top_10_singular": [float(s) for s in S[:10]],
        }
        print(f"  R² = {r2:.3f}, cond = {cond:.2e}, rotation_score = {rot_score:.3f}")
        for k in range(min(100, len(S))):
            records.append({
                "transition": f"L{ell}_to_L{ell+1}",
                "k": k, "singular_value": float(S[k]),
                "normalized": float(s_norm[k]),
            })
        # Plot per-transition SVD spectrum
        ax = axes[i]
        ax.plot(range(len(S)), S, lw=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("Singular value index"); ax.set_ylabel("Singular value (log)")
        ax.set_title(f"L{ell}→L{ell+1}\nR²={r2:.3f}, cond={cond:.1e}, rot_score={rot_score:.3f}")
        ax.axhline(S.max(), color="r", ls="--", lw=0.6, alpha=0.5, label=f"max={S.max():.2g}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"svd_summary.{ext}", dpi=150)
    plt.close(fig)
    print("[plot] svd_summary saved")

    import pandas as pd
    pd.DataFrame(records).to_csv(args.out_dir / "singular_values.csv", index=False)
    (args.out_dir / "alignment_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[done] outputs at {args.out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

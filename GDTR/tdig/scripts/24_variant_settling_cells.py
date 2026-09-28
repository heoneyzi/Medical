"""Phase 1.1b — Full 17-cell settling computation for ClinVar variants.

Loads variant_h_ell_ref.h5 + variant_h_ell_alt.h5, applies the same 17-cell
v2 settling logic (mirrors 15_chr22_forward.py + 20_gamma_ablation.py fix) to
each (ref, alt) pair separately, then computes Δc = c(alt) − c(ref) per
variant per cell.

Then: per-cell Δc AUROC for P/LP vs B/LB at every layer (no layer to pick;
this is a single Δc summary per variant).

Memory: variant_h_ell_ref/alt each (10910, 32, 4096) fp32 = ~5.4 GB. Both
loaded = ~11 GB. Plus working copies. Need ~25 GB RAM available.

Output: results/variant_settling_cells/
  variant_settling_per_cell.csv     (variant_idx, gene, category, stars, cell, c_ref, c_alt, delta_c)
  cell_auroc.csv                    (cell, AUROC, n_PLP, n_BLB)
  cell_auroc.png                    bar chart
  per_consequence_cell.csv          (cell × consequence × AUROC)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
N_LAYERS = 32
HIDDEN = 4096
ALPHA_BETA_CELLS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]


def settling_persistence(D, gamma, W=3, max_layer=None):
    """Mirror 15_chr22_forward.py settling_persistence (boundary clip)."""
    if D.ndim == 1:
        D = D[:, None]
    L, T = D.shape
    if max_layer is None or max_layer >= L:
        max_layer = L - 1
    below = D <= gamma
    rolling = np.zeros((L, T), dtype=bool)
    for ell in range(max_layer + 1):
        end_k = min(ell + W - 1, max_layer)
        rolling[ell] = below[ell:end_k + 1].all(axis=0)
    c = np.full(T, -1, dtype=np.int32)
    any_set = rolling.any(axis=0)
    c[any_set] = np.argmax(rolling, axis=0)[any_set]
    return c


def settling_monotone_direct(D, gamma):
    if D.ndim == 1:
        D = D[:, None]
    below = D <= gamma
    c = np.full(D.shape[1], -1, dtype=np.int32)
    any_set = below.any(axis=0)
    c[any_set] = np.argmax(below, axis=0)[any_set]
    return c


def compute_17_cells_per_variant(h_ell, h_norm, gamma_v2, sigma_inv,
                                   batch_size=200, persistence_w=3):
    """h_ell shape (N, 32, 4096), h_norm shape (N, 4096).
       Returns dict {cell_name: (N,) int settling layer}.
    """
    N = h_ell.shape[0]
    L = N_LAYERS
    H = HIDDEN
    out = {}
    cell_names = []

    # Process in batches to limit memory of intermediate computations
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        h_b = h_ell[start:end]  # (B, 32, 4096)
        hn_b = h_norm[start:end]  # (B, 4096)
        h29_b = h_b[:, L_STAR]    # (B, 4096)
        B = h_b.shape[0]

        # Norms
        norms_ell = np.linalg.norm(h_b, axis=-1)  # (B, 32)
        norm_h29 = np.linalg.norm(h29_b, axis=-1)  # (B,)
        norm_hnorm = np.linalg.norm(hn_b, axis=-1)  # (B,)

        # Cos refA: cos(h_ell, h_29)
        # dot(h_ell, h_29) / (||h_ell|| * ||h_29||)
        cos_A = (h_b * h29_b[:, None, :]).sum(-1) / (norms_ell * norm_h29[:, None] + 1e-12)
        cos_C = (h_b * hn_b[:, None, :]).sum(-1) / (norms_ell * norm_hnorm[:, None] + 1e-12)
        # For refB: RMSnorm — we don't have it; skip refB cells (mark -1)

        # Step norms (between consecutive layers)
        step_norm_raw = np.linalg.norm(h_b[:, 1:] - h_b[:, :-1], axis=-1)  # (B, L-1)
        step_cos_raw = (h_b[:, 1:] * h_b[:, :-1]).sum(-1) / (
            np.linalg.norm(h_b[:, 1:], axis=-1) * np.linalg.norm(h_b[:, :-1], axis=-1) + 1e-12)
        step_cos = (h_b[:, 2:] - h_b[:, 1:-1] + 1e-12)  # not needed; computed below
        # Use velocity/curvature for M3 — same as forward script
        # v[ell] = step_norm[ell] / (||h_ell||+eps)
        v = step_norm_raw / (norms_ell[:, :L - 1] + 1e-12)  # (B, L-1)
        # kappa[ell] = 1 - cos(step_{ell+1}, step_{ell}) for ell in [0, L-3]
        step_vec = h_b[:, 1:] - h_b[:, :-1]  # (B, L-1, H)
        step_cos_vec = (step_vec[:, 1:] * step_vec[:, :-1]).sum(-1) / (
            np.linalg.norm(step_vec[:, 1:], axis=-1) * np.linalg.norm(step_vec[:, :-1], axis=-1) + 1e-12)
        kappa = 1.0 - step_cos_vec  # (B, L-2)

        # All cells: persistence over (L, B) — need to convert to (L, T_eff=B)
        # ── M1 cells ──
        for ref_label, cos_arr, max_l in [("refA", cos_A, L_STAR),
                                            ("refC", cos_C, L - 1)]:
            D_dir = 1.0 - cos_arr.T  # (L, B)
            gamma = gamma_v2[f"D_dir_{ref_label}_at_28"]["q70"]
            cells_v = settling_persistence(D_dir, gamma, W=persistence_w, max_layer=max_l)
            out.setdefault(f"M1_dir_{ref_label}", []).append(cells_v)
        # refB needs RMSnormed — skip (mark -1)
        out.setdefault("M1_dir_refB", []).append(np.full(B, -1, dtype=np.int32))

        # ── M2 magnitude refA ──
        D_mag = np.abs(norms_ell / (norm_h29[:, None] + 1e-12) - 1.0).T  # (L, B)
        gamma = gamma_v2["D_mag_refA_at_28"]["q70"]
        cells_v = settling_persistence(D_mag, gamma, W=persistence_w, max_layer=L_STAR)
        out.setdefault("M2_mag_refA", []).append(cells_v)
        out.setdefault("M2_mag_refB_diag", []).append(np.full(B, -1, dtype=np.int32))
        out.setdefault("M2_mag_refC_diag", []).append(np.full(B, -1, dtype=np.int32))

        # ── M3 geometry ──
        v_mean = gamma_v2["_geo_pop_stats"]["v_mean"]
        v_std = gamma_v2["_geo_pop_stats"]["v_std"]
        k_mean = gamma_v2["_geo_pop_stats"]["kappa_mean"]
        k_std = gamma_v2["_geo_pop_stats"]["kappa_std"]
        v_use = v[:, :L - 2]  # (B, L-2)
        v_z = (v_use - v_mean) / (v_std + 1e-12)
        k_z = (kappa - k_mean) / (k_std + 1e-12)
        for (alpha, beta) in ALPHA_BETA_CELLS:
            g = alpha * v_z + beta * k_z  # (B, L-2)
            g_pad = np.full((B, L), np.inf, dtype=np.float32)
            g_pad[:, :L - 2] = g
            gamma_g = gamma_v2[f"D_geo_a{alpha}_b{beta}_at_26"]["q70"]
            cells_v = settling_persistence(g_pad.T, gamma_g,
                                            W=persistence_w, max_layer=L - 3)
            out.setdefault(f"M3_geo_a{alpha}_b{beta}", []).append(cells_v)

        # ── M4_set: needs sigma_inv ──
        # D_Mset(ell, t) = sqrt((h_l - h_ref)^T Sigma_inv (h_l - h_ref))
        for ref_label, ref_vec, sig_var in [("refA", h29_b, "A")]:
            sig_inv = sigma_inv[sig_var]  # (4096, 4096) fp32
            diff = h_b - ref_vec[:, None, :]  # (B, L, H)
            diff_flat = diff.reshape(-1, H)
            sig_diff = diff_flat @ sig_inv
            quad = np.maximum((diff_flat * sig_diff).sum(-1), 0.0).reshape(B, L)
            D_set = np.sqrt(quad).T  # (L, B)
            gamma_set = gamma_v2[f"D_Mset_ref{sig_var}_at_28"]["q70"]
            cells_v = settling_monotone_direct(D_set, gamma_set)
            out.setdefault(f"M4_set_{ref_label}", []).append(cells_v)
        # refB/C: degenerate — mark -1
        out.setdefault("M4_set_refB", []).append(np.full(B, -1, dtype=np.int32))
        out.setdefault("M4_set_refC", []).append(np.full(B, -1, dtype=np.int32))

        # ── M5 tau refA, refC ──
        # path_cum_raw[ell] = sum step_norm[0..ell-1]
        path_cum = np.cumsum(step_norm_raw[:, :L_STAR], axis=1)  # (B, L*)
        path_total = path_cum[:, L_STAR - 1]
        remaining = np.zeros((B, L_STAR), dtype=np.float32)
        remaining[:, 0] = path_total
        remaining[:, 1:] = path_total[:, None] - path_cum[:, :L_STAR - 1]
        # res_A = ||h_ell - h_29||
        diff_A = h_b - h29_b[:, None, :]
        res_A = np.linalg.norm(diff_A, axis=-1)  # (B, L)
        diff_C = h_b - hn_b[:, None, :]
        res_C = np.linalg.norm(diff_C, axis=-1)

        tau_A = remaining / (res_A[:, :L_STAR] + 1e-12)  # (B, L*)
        tau_C = remaining / (res_C[:, :L_STAR] + 1e-12)
        tau_max = L_STAR - 1
        for ref_label, tau in [("refA", tau_A), ("refC", tau_C)]:
            tau_pad = np.full((B, L), np.inf, dtype=np.float32)
            tau_pad[:, :L_STAR] = tau
            gamma_tau = gamma_v2[f"tau_{ref_label}_at_27"]["q70"]
            cells_v = settling_persistence(tau_pad.T, gamma_tau,
                                            W=persistence_w, max_layer=tau_max)
            out.setdefault(f"M5_tau_{ref_label}", []).append(cells_v)
        out.setdefault("M5_tau_refB", []).append(np.full(B, -1, dtype=np.int32))

        if (end // batch_size) % 5 == 0:
            print(f"   processed {end}/{N}")

    # Concatenate batches
    out = {k: np.concatenate(v) for k, v in out.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-h5", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_h_ell_ref.h5"))
    p.add_argument("--alt-h5", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_h_ell_alt.h5"))
    p.add_argument("--scalars", type=Path,
                   default=Path("/path/to/TDiG/data/cache/variants/variant_scalars.parquet"))
    p.add_argument("--gamma", type=Path,
                   default=Path("/path/to/TDiG/data/cache/population_stats/gamma_calibration_v2.json"))
    p.add_argument("--sigma-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/population_stats"))
    p.add_argument("--consequence-csv", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_per_consequence/variant_with_consequence.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/variant_settling_cells"))
    p.add_argument("--batch-size", type=int, default=200)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})

    print(f"[load] gamma + sigma + scalars + consequence ...")
    gamma_v2 = json.loads(args.gamma.read_text())
    sigma_inv = {
        "A": np.load(args.sigma_dir / "sigma_ref_inv_A.npy"),
    }
    scalars = pd.read_parquet(args.scalars)
    cons_df = pd.read_csv(args.consequence_csv)[["chrom", "pos", "ref", "alt", "consequence"]]
    cons_df["chrom"] = cons_df.chrom.astype(str)
    scalars["chrom"] = scalars.chrom.astype(str)
    scalars = scalars.merge(cons_df, on=["chrom", "pos", "ref", "alt"], how="left")

    print(f"[load] ref h5 ...")
    with h5py.File(args.ref_h5, "r") as h5:
        h_ell_ref = h5["h_ell"][:].astype(np.float32)
        h_norm_ref = h5["h_norm"][:].astype(np.float32)
    print(f"  ref shape {h_ell_ref.shape}")

    print(f"[load] alt h5 ...")
    with h5py.File(args.alt_h5, "r") as h5:
        h_ell_alt = h5["h_ell"][:].astype(np.float32)
        h_norm_alt = h5["h_norm"][:].astype(np.float32)
    print(f"  alt shape {h_ell_alt.shape}")

    print("\n[compute] settling cells for REF ...")
    cells_ref = compute_17_cells_per_variant(h_ell_ref, h_norm_ref, gamma_v2,
                                               sigma_inv, batch_size=args.batch_size)
    # Free ref memory
    del h_ell_ref, h_norm_ref

    print("\n[compute] settling cells for ALT ...")
    cells_alt = compute_17_cells_per_variant(h_ell_alt, h_norm_alt, gamma_v2,
                                               sigma_inv, batch_size=args.batch_size)
    del h_ell_alt, h_norm_alt

    # Per-variant table: c_ref, c_alt, delta
    print("\n[table] per-variant per-cell table ...")
    long_records = []
    for cell, c_ref in cells_ref.items():
        c_alt = cells_alt[cell]
        for i in range(len(scalars)):
            r = scalars.iloc[i]
            long_records.append({
                "variant_idx": i, "gene": r.gene, "category": r.category,
                "stars": int(r.stars), "consequence": r.consequence,
                "cell": cell, "c_ref": int(c_ref[i]), "c_alt": int(c_alt[i]),
                "delta_c": int(c_alt[i] - c_ref[i]) if c_ref[i] >= 0 and c_alt[i] >= 0 else None,
            })
    long_df = pd.DataFrame(long_records)
    long_df.to_csv(args.out_dir / "variant_settling_per_cell.csv", index=False)
    print(f"[save] variant_settling_per_cell.csv ({len(long_df):,} rows)")

    # AUROC per cell using |delta_c| as score for P_LP vs B_LB
    print("\n[AUROC] per-cell AUROC P_LP vs B_LB using |Δc|")
    auroc_records = []
    bin_df = long_df[long_df.category.isin({"P_LP", "B_LB"})].copy()
    bin_df["y"] = (bin_df.category == "P_LP").astype(int)
    for cell in bin_df.cell.unique():
        sub = bin_df[(bin_df.cell == cell) & bin_df.delta_c.notna()]
        if sub.y.sum() < 10 or (1 - sub.y).sum() < 10:
            continue
        try:
            auroc = roc_auc_score(sub.y.values, np.abs(sub.delta_c.values))
        except Exception:
            continue
        auroc_records.append({
            "cell": cell, "AUROC": float(auroc),
            "n_PLP": int(sub.y.sum()), "n_BLB": int((1 - sub.y).sum()),
        })
        print(f"  {cell:20s}  AUROC={auroc:.3f}  n_PLP={int(sub.y.sum())}  n_BLB={int((1-sub.y).sum())}")
    pd.DataFrame(auroc_records).to_csv(args.out_dir / "cell_auroc.csv", index=False)

    # Per-consequence × cell AUROC
    print("\n[AUROC] per-consequence × cell")
    pc_records = []
    for cell in bin_df.cell.unique():
        for cons in bin_df.consequence.dropna().unique():
            sub = bin_df[(bin_df.cell == cell) & (bin_df.consequence == cons) & bin_df.delta_c.notna()]
            if sub.y.sum() < 5 or (1 - sub.y).sum() < 5:
                continue
            try:
                auroc = roc_auc_score(sub.y.values, np.abs(sub.delta_c.values))
            except Exception:
                continue
            pc_records.append({
                "cell": cell, "consequence": cons, "AUROC": float(auroc),
                "n_PLP": int(sub.y.sum()), "n_BLB": int((1 - sub.y).sum()),
            })
    pd.DataFrame(pc_records).to_csv(args.out_dir / "per_consequence_cell.csv", index=False)
    print(f"[save] per_consequence_cell.csv")

    # Bar plot of cell AUROC
    if auroc_records:
        adf = pd.DataFrame(auroc_records).sort_values("AUROC", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.bar(adf.cell, adf.AUROC, color="#1f77b4", alpha=0.85)
        for i, (c, a) in enumerate(zip(adf.cell, adf.AUROC)):
            ax.text(i, a + 0.003, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.set_ylabel("AUROC (|Δc|, P_LP vs B_LB)")
        ax.set_title("(1.1b) Per-cell variant AUROC using |c(alt) − c(ref)| as score")
        ax.set_ylim(0.4, 1.0)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.out_dir / f"cell_auroc.{ext}", dpi=150)
        plt.close(fig)
        print("[plot] cell_auroc saved")

    print(f"\n[done] outputs at {args.out_dir}")


if __name__ == "__main__":
    main()

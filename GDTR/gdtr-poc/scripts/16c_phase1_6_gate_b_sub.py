"""Phase 1.6 Gate B sub-analyses (post-hoc, CPU-only).

Reuses /path/to/gDTR/results/phase1.6/chr22_cache.h5 and γ_cos = 0.397 (locked).

Three sub-analyses:
  A) Per-gene rank by mean settling depth (chr22 protein-coding, GENCODE v44).
  B) Splice-site fine-grained distance profile (mean c at {0, ±10, ±20, ±50,
     ±100, ±200} bp from nearest donor/acceptor).
  C) Per-context Cohen's d pairwise matrix (7 active contexts; skip 'repeat').

Per-position settling depth is averaged over overlapping windows (3 kb stride,
6 kb width => up to 2× coverage per position).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

from phase0_src.src.stats import mwu_with_effect, cohens_d as ph0_cohens_d  # noqa: E402

PHASE = "phase1.6_gate_b_sub"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase1.6_sub"
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)

SEED = 42
GAMMA_LOCKED = 0.397  # locked at calibration (matches gamma_cos_global_q70 ≈ 0.3966)

CTX = {
    0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
    5: "splice_donor", 6: "splice_acceptor", 7: "repeat",
}
ACTIVE_CTX = [
    "intergenic", "intron", "coding_exon", "5utr", "3utr",
    "splice_donor", "splice_acceptor",
]
DISTANCE_BINS = [-200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200]
DISTANCE_TOL = 1  # ±1 bp around each target distance to give a small bin

CHROM = "chr22"


def settling_depth_per_window_np(D_cos: np.ndarray, gamma: float) -> np.ndarray:
    """[L, T] -> [T] 1-based settling depth (NumPy mirror of phase 0 logic)."""
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


def build_position_c_array(h5_path: Path, gamma: float, chrom_len: int) -> np.ndarray:
    """Build per-position mean settling depth array of length chrom_len.

    Overlapping windows are averaged. Positions never covered receive NaN.
    """
    import h5py

    sum_c = np.zeros(chrom_len, dtype=np.float64)
    cov = np.zeros(chrom_len, dtype=np.uint16)

    with h5py.File(h5_path, "r") as h5:
        N = int(h5["D_cos"].shape[0])
        done_mask = h5["done_mask"][:]
        starts = h5["starts"][:]
        ends = h5["ends"][:]
        for i in range(N):
            if not done_mask[i]:
                continue
            s = int(starts[i]); e = int(ends[i])
            D = h5["D_cos"][i].astype(np.float32)  # [L, T]
            c = settling_depth_per_window_np(D, gamma)  # [T]
            # Clamp to chromosome length just in case
            seg_end = min(e, chrom_len)
            seg_len = seg_end - s
            if seg_len <= 0:
                continue
            sum_c[s:seg_end] += c[:seg_len]
            cov[s:seg_end] += 1
            if (i + 1) % 2000 == 0:
                LOG.info("position-c accumulation: %d/%d windows", i + 1, N)

    LOG.info("coverage: covered=%d / chrom_len=%d (%.2f%%)",
             int((cov > 0).sum()), chrom_len, 100.0 * (cov > 0).mean())
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    nz = cov > 0
    out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# A. Per-gene rank by mean settling depth
# -----------------------------------------------------------------------------
def analysis_a_per_gene_rank(pos_c: np.ndarray) -> Path:
    import gffutils
    import csv

    db = gffutils.FeatureDB(str(ru.GDTR_ROOT / "data" / "annotation" /
                                "gencode.v44.chr17_chr22.gtf.db"))
    rows = []
    n_skipped = 0
    for g in db.features_of_type("gene"):
        if g.seqid != CHROM:
            continue
        gtype = g.attributes.get("gene_type", [None])[0]
        if gtype != "protein_coding":
            continue
        # GTF is 1-based inclusive; numpy slice [start-1:end]
        s = max(int(g.start) - 1, 0)
        e = min(int(g.end), pos_c.shape[0])
        if e <= s:
            n_skipped += 1
            continue
        seg = pos_c[s:e]
        valid = seg[~np.isnan(seg)]
        if valid.size < 10:
            n_skipped += 1
            continue
        gene_name = g.attributes.get("gene_name", [g.id])[0]
        rows.append({
            "gene_id": g.id,
            "gene_name": gene_name,
            "n_pos": int(valid.size),
            "mean_c": float(valid.mean()),
            "median_c": float(np.median(valid)),
        })
    LOG.info("Per-gene: kept %d genes, skipped %d", len(rows), n_skipped)

    rows.sort(key=lambda r: r["mean_c"])  # ascending: deepest thinking first
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    out_csv = PHASE_OUT_DIR / "per_gene_rank.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene_id", "gene_name", "n_pos",
                                            "mean_c", "median_c", "rank"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Figures: top-20 deepest (lowest mean_c) and shallowest (highest mean_c)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _bar(rows_sub, title, fname):
        fig, ax = plt.subplots(figsize=(8, 6))
        names = [r["gene_name"] for r in rows_sub]
        vals = [r["mean_c"] for r in rows_sub]
        ax.barh(range(len(names)), vals, color="#3a7ca5")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("mean settling depth c")
        ax.set_title(title)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(PHASE_OUT_DIR / f"{fname}.{ext}", dpi=150)
        plt.close(fig)

    top20 = rows[:20]
    bot20 = rows[-20:][::-1]
    _bar(top20, "chr22 top-20 deepest-thinking genes (lowest mean c)",
         "F_top20_deepest_genes")
    _bar(bot20, "chr22 top-20 shallowest genes (highest mean c)",
         "F_bot20_shallowest_genes")
    return out_csv


# -----------------------------------------------------------------------------
# B. Splice site fine-grained distance profile
# -----------------------------------------------------------------------------
def analysis_b_splice_profile(pos_c: np.ndarray, labels: np.ndarray) -> Path:
    """For each donor/acceptor center position, compute mean c at offsets."""
    donor_centers = np.flatnonzero(labels == 5)  # splice_donor
    acceptor_centers = np.flatnonzero(labels == 6)  # splice_acceptor
    LOG.info("donor centers=%d, acceptor centers=%d",
             donor_centers.size, acceptor_centers.size)

    chrom_len = pos_c.shape[0]
    profile = {"donor": {}, "acceptor": {}, "distance_bins": DISTANCE_BINS,
               "distance_tol_bp": DISTANCE_TOL}

    for name, centers in (("donor", donor_centers),
                          ("acceptor", acceptor_centers)):
        per_dist = {}
        for d in DISTANCE_BINS:
            # Take all positions at offset d (with ±DISTANCE_TOL window)
            offsets = np.arange(d - DISTANCE_TOL, d + DISTANCE_TOL + 1)
            vals = []
            for off in offsets:
                idx = centers + off
                idx = idx[(idx >= 0) & (idx < chrom_len)]
                v = pos_c[idx]
                v = v[~np.isnan(v)]
                if v.size:
                    vals.append(v)
            if vals:
                merged = np.concatenate(vals)
                per_dist[str(d)] = {
                    "n": int(merged.size),
                    "mean_c": float(merged.mean()),
                    "median_c": float(np.median(merged)),
                    "std_c": float(merged.std(ddof=1)) if merged.size > 1 else None,
                }
            else:
                per_dist[str(d)] = {"n": 0, "mean_c": None, "median_c": None,
                                    "std_c": None}
        profile[name] = per_dist

    # Background reference: intronic mean c
    intron_mask = labels == 1
    intron_c = pos_c[intron_mask]
    intron_c = intron_c[~np.isnan(intron_c)]
    profile["intron_mean_c_background"] = float(intron_c.mean())
    profile["gamma_cos"] = GAMMA_LOCKED
    profile["seed"] = SEED

    out_json = PHASE_OUT_DIR / "splice_distance_profile.json"
    out_json.write_text(json.dumps(profile, indent=2))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in (("donor", "#d1495b"), ("acceptor", "#2e86ab")):
        xs, ys = [], []
        for d in DISTANCE_BINS:
            entry = profile[name][str(d)]
            if entry["mean_c"] is not None:
                xs.append(d); ys.append(entry["mean_c"])
        ax.plot(xs, ys, "o-", label=name, color=color, lw=1.6, ms=5)
    ax.axhline(profile["intron_mean_c_background"], ls="--", color="gray",
               lw=1, label=f"intron bg = {profile['intron_mean_c_background']:.2f}")
    ax.set_xlabel("offset from splice site (bp)")
    ax.set_ylabel("mean settling depth c")
    ax.set_title(f"chr22 splice fine-grained profile (γ={GAMMA_LOCKED})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PHASE_OUT_DIR / f"F_splice_distance_profile.{ext}", dpi=150)
    plt.close(fig)
    return out_json


# -----------------------------------------------------------------------------
# C. Per-context Cohen's d pairwise matrix + MWU pairwise p-values
# -----------------------------------------------------------------------------
def analysis_c_pairwise(pos_c: np.ndarray, labels: np.ndarray) -> Path:
    import csv

    code_for = {v: k for k, v in CTX.items()}
    rng = np.random.default_rng(SEED)
    # Per-context settling-depth values (subsample for tractable MWU on huge n)
    SUBSAMPLE_N = 200_000  # cap per group; preserves rank-based statistics
    ctx_vals = {}
    for ctx in ACTIVE_CTX:
        code = code_for[ctx]
        idx = np.flatnonzero(labels == code)
        v = pos_c[idx]
        v = v[~np.isnan(v)]
        if v.size > SUBSAMPLE_N:
            sel = rng.choice(v.size, SUBSAMPLE_N, replace=False)
            v = v[sel]
        ctx_vals[ctx] = v
        LOG.info("context %s: n=%d (capped at %d)", ctx, v.size, SUBSAMPLE_N)

    n = len(ACTIVE_CTX)
    d_mat = np.full((n, n), np.nan, dtype=np.float64)
    p_mat = np.full((n, n), np.nan, dtype=np.float64)

    n_pairs = n * (n - 1) // 2  # 21
    bonf = float(n_pairs)
    for i, ai in enumerate(ACTIVE_CTX):
        for j, bj in enumerate(ACTIVE_CTX):
            if i == j:
                d_mat[i, j] = 0.0
                p_mat[i, j] = 1.0
                continue
            a = ctx_vals[ai]; b = ctx_vals[bj]
            if a.size < 2 or b.size < 2:
                continue
            d_mat[i, j] = ph0_cohens_d(a, b)  # positive => row larger
            U, p, _r = mwu_with_effect(a, b, alternative="two-sided")
            p_mat[i, j] = min(p * bonf, 1.0)

    # Write CSVs
    def _write_csv(path, mat):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([""] + ACTIVE_CTX)
            for i, row_name in enumerate(ACTIVE_CTX):
                w.writerow([row_name] + [f"{mat[i, j]:.6g}" for j in range(n)])

    d_csv = PHASE_OUT_DIR / "cohens_d_matrix.csv"
    p_csv = PHASE_OUT_DIR / "pairwise_pvalues.csv"
    _write_csv(d_csv, d_mat)
    _write_csv(p_csv, p_mat)

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmax = float(np.nanmax(np.abs(d_mat)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(d_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ACTIVE_CTX, rotation=35, ha="right")
    ax.set_yticklabels(ACTIVE_CTX)
    for i in range(n):
        for j in range(n):
            if np.isfinite(d_mat[i, j]):
                ax.text(j, i, f"{d_mat[i, j]:.2f}", ha="center", va="center",
                        color="black" if abs(d_mat[i, j]) < 0.5 * vmax else "white",
                        fontsize=8)
    ax.set_title("Cohen's d pairwise (row vs col, +ve => row higher c)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PHASE_OUT_DIR / f"F_cohens_d_heatmap.{ext}", dpi=150)
    plt.close(fig)

    # Save summary JSON (top pair, etc.)
    abs_d = np.abs(d_mat).copy()
    np.fill_diagonal(abs_d, 0.0)
    flat_idx = int(np.nanargmax(abs_d))
    i, j = np.unravel_index(flat_idx, abs_d.shape)
    summary = {
        "active_contexts": ACTIVE_CTX,
        "n_pairs": n_pairs,
        "bonferroni_factor": bonf,
        "subsample_per_context": SUBSAMPLE_N,
        "strongest_pair": {
            "row": ACTIVE_CTX[i], "col": ACTIVE_CTX[j],
            "cohens_d": float(d_mat[i, j]),
            "p_bonf": float(p_mat[i, j]),
        },
    }
    (PHASE_OUT_DIR / "pairwise_summary.json").write_text(json.dumps(summary, indent=2))
    return d_csv


# -----------------------------------------------------------------------------
def main() -> None:
    np.random.seed(SEED)
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="gate_b_sub"):
        h5_path = ru.GDTR_ROOT / "results" / "phase1.6" / "chr22_cache.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"missing {h5_path}")

        cal = json.loads((ru.GDTR_ROOT / "results" / "phase1.4" / "calibration.json").read_text())
        gamma_cal = float(cal["gamma_cos_global_q70"])
        LOG.info("γ_cos locked = %.4f (spec) ; calibration q70 = %.4f",
                 GAMMA_LOCKED, gamma_cal)

        labels = np.load(ru.GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy")
        chrom_len = int(labels.shape[0])
        LOG.info("labels: n=%d", chrom_len)

        # Build per-position c array (overlapping windows averaged)
        LOG.info("building per-position settling depth array …")
        pos_c = build_position_c_array(h5_path, GAMMA_LOCKED, chrom_len)
        # Persist for re-use
        np.save(PHASE_OUT_DIR / "chr22_position_c.npy", pos_c)
        LOG.info("saved chr22_position_c.npy (mean=%.3f, NaN=%d)",
                 float(np.nanmean(pos_c)), int(np.isnan(pos_c).sum()))

        # A: per-gene rank
        LOG.info("=== A. per-gene rank ===")
        a_csv = analysis_a_per_gene_rank(pos_c)
        LOG.info("wrote %s", a_csv)

        # B: splice fine profile
        LOG.info("=== B. splice fine-grained profile ===")
        b_json = analysis_b_splice_profile(pos_c, labels)
        LOG.info("wrote %s", b_json)

        # C: pairwise Cohen's d matrix
        LOG.info("=== C. pairwise Cohen's d matrix ===")
        c_csv = analysis_c_pairwise(pos_c, labels)
        LOG.info("wrote %s", c_csv)

        ru.write_done(PHASE, PHASE_OUT_DIR,
                      {"gamma_cos": GAMMA_LOCKED, "seed": SEED,
                       "outputs": {"per_gene_rank": str(a_csv),
                                   "splice_profile": str(b_json),
                                   "cohens_d_matrix": str(c_csv)}},
                      step_name="gate_b_sub")
        LOG.info("Phase 1.6 Gate B sub-analyses done.")


if __name__ == "__main__":
    main()

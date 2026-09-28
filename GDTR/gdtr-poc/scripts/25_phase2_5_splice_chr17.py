"""Phase 2.5 — Splice fine profile for chr17, compared to chr22.

Replicates Phase 1.6_sub splice profile (mean c at offsets ±10, ±20, ±50,
±100, ±200) for chr17 and overlays chr22 baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase2.5"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.5"
LOG = ru.setup_logging(PHASE)

GAMMA_LOCKED = 0.39663
DISTANCE_BINS = [-200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200]
DISTANCE_TOL = 1
SEED = 42


def settling_depth_per_window(D_cos, gamma):
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


def build_pos_c(h5_path: Path, chrom_len: int, gamma: float) -> np.ndarray:
    import h5py
    sum_c = np.zeros(chrom_len, dtype=np.float64)
    cov = np.zeros(chrom_len, dtype=np.uint16)
    with h5py.File(h5_path, "r") as h5:
        N = h5["D_cos"].shape[0]
        done = h5["done_mask"][:]
        starts = h5["starts"][:]; ends = h5["ends"][:]
        for i in range(N):
            if not done[i]:
                continue
            s = int(starts[i]); e = int(ends[i])
            D = h5["D_cos"][i].astype(np.float32)
            c = settling_depth_per_window(D, gamma)
            seg_end = min(e, chrom_len)
            seg_len = seg_end - s
            if seg_len <= 0:
                continue
            sum_c[s:seg_end] += c[:seg_len]
            cov[s:seg_end] += 1
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    nz = cov > 0
    out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
    return out


def splice_profile(pos_c: np.ndarray, labels: np.ndarray) -> dict:
    chrom_len = pos_c.shape[0]
    donor_centers = np.flatnonzero(labels == 5)
    acceptor_centers = np.flatnonzero(labels == 6)
    LOG.info("donor centers=%d, acceptor centers=%d",
             donor_centers.size, acceptor_centers.size)
    profile = {"donor": {}, "acceptor": {},
               "distance_bins": DISTANCE_BINS,
               "distance_tol_bp": DISTANCE_TOL}
    for name, centers in (("donor", donor_centers), ("acceptor", acceptor_centers)):
        per_dist = {}
        for d in DISTANCE_BINS:
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
                per_dist[str(d)] = {"n": 0, "mean_c": None, "median_c": None, "std_c": None}
        profile[name] = per_dist
    intron_mask = labels == 1
    intron_c = pos_c[intron_mask]
    intron_c = intron_c[~np.isnan(intron_c)]
    profile["intron_mean_c_background"] = float(intron_c.mean()) if intron_c.size else None
    return profile


def main() -> None:
    np.random.seed(SEED)
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="splice_chr17"):
        chr17_h5 = ru.GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        chr17_lab = np.load(ru.GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy")

        chr17_pos_c_path = ru.GDTR_ROOT / "results" / "phase2.4" / "chr17_position_c.npy"
        if chr17_pos_c_path.exists():
            LOG.info("reusing chr17_position_c.npy from phase2.4")
            pos17 = np.load(chr17_pos_c_path)
        else:
            LOG.info("building chr17 per-position c ...")
            pos17 = build_pos_c(chr17_h5, chr17_lab.shape[0], GAMMA_LOCKED)
            np.save(PHASE_OUT_DIR / "chr17_position_c.npy", pos17)

        prof17 = splice_profile(pos17, chr17_lab)
        prof17["gamma_cos"] = GAMMA_LOCKED
        prof17["seed"] = SEED
        prof17["chrom"] = "chr17"
        (PHASE_OUT_DIR / "splice_chr17_profile.json").write_text(json.dumps(prof17, indent=2))

        # Reuse chr22 phase1.6_sub profile for overlay
        chr22_prof_path = ru.GDTR_ROOT / "results" / "phase1.6_sub" / "splice_distance_profile.json"
        prof22 = json.loads(chr22_prof_path.read_text()) if chr22_prof_path.exists() else None

        # Plot overlay
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            for ax, name, color17, color22 in (
                (axes[0], "donor", "#d1495b", "#f5a09c"),
                (axes[1], "acceptor", "#2e86ab", "#90c4d8"),
            ):
                xs17, ys17 = [], []
                for d in DISTANCE_BINS:
                    e = prof17[name][str(d)]
                    if e["mean_c"] is not None:
                        xs17.append(d); ys17.append(e["mean_c"])
                ax.plot(xs17, ys17, "o-", color=color17, lw=1.7, label="chr17", ms=5)
                if prof22 is not None and name in prof22:
                    xs22, ys22 = [], []
                    for d in DISTANCE_BINS:
                        e22 = prof22[name].get(str(d), {})
                        if e22.get("mean_c") is not None:
                            xs22.append(d); ys22.append(e22["mean_c"])
                    ax.plot(xs22, ys22, "s--", color=color22, lw=1.4, label="chr22", ms=4)
                if prof17.get("intron_mean_c_background") is not None:
                    ax.axhline(prof17["intron_mean_c_background"], ls=":",
                               color=color17, alpha=0.6,
                               label=f"chr17 intron bg={prof17['intron_mean_c_background']:.2f}")
                if prof22 and prof22.get("intron_mean_c_background") is not None:
                    ax.axhline(prof22["intron_mean_c_background"], ls=":",
                               color=color22, alpha=0.6,
                               label=f"chr22 intron bg={prof22['intron_mean_c_background']:.2f}")
                ax.set_xlabel("offset from splice site (bp)")
                ax.set_title(name)
                ax.grid(alpha=0.3)
                ax.legend(loc="best", fontsize=8)
            axes[0].set_ylabel("mean settling depth c")
            fig.suptitle(f"Phase 2.5 splice fine profile (γ={GAMMA_LOCKED})")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_splice_chr17_vs_chr22.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("figure failed: %s", e)

        # Asymmetry & match summary
        donor_min_c = min(
            (e["mean_c"] for e in prof17["donor"].values() if e["mean_c"] is not None),
            default=None,
        )
        intron_bg = prof17.get("intron_mean_c_background")
        asym_summary = {
            "chr17_donor_min_c": donor_min_c,
            "chr17_intron_bg": intron_bg,
            "donor_below_bg": (donor_min_c is not None and intron_bg is not None
                               and donor_min_c < intron_bg),
        }
        ru.write_done(PHASE, PHASE_OUT_DIR, asym_summary, step_name="splice_chr17")
        LOG.info("asym summary: %s", asym_summary)
        LOG.info("Phase 2.5 done.")


if __name__ == "__main__":
    main()

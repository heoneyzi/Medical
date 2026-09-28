"""Phase 2.2 — Gate B chr17 exon vs intron contrast (CPU on chr17 cache).

Adapted from scripts/16b_phase1_6_gate_b.py. Uses γ_cos = 0.39663 (Phase 1.4 locked).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase2.2"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.2"
LOG = ru.setup_logging(PHASE)

GAMMA_LOCKED = 0.39663

CTX = {
    0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
    5: "splice_donor", 6: "splice_acceptor", 7: "repeat",
}


def settling_depth_per_window(D_cos, gamma):
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.int64)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return (np.mean(x) - np.mean(y)) / sp


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="gate_b_chr17"):
        from scipy.stats import mannwhitneyu
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py required")

        h5_path = ru.GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"missing {h5_path}; run Phase 2.1 first")
        gamma = GAMMA_LOCKED
        LOG.info("gamma_cos = %.5f (locked from Phase 1.4)", gamma)

        labels = np.load(ru.GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy")
        LOG.info("position labels shape=%s dtype=%s", labels.shape, labels.dtype)

        c_per_label = {v: [] for v in CTX.values()}
        n_windows_done = 0
        n_pos_done = 0
        with h5py.File(h5_path, "r") as h5:
            N = h5["D_cos"].shape[0]
            done_mask = h5["done_mask"][:]
            starts = h5["starts"][:]
            ends = h5["ends"][:]
            for i in range(N):
                if not done_mask[i]:
                    continue
                D = h5["D_cos"][i].astype(np.float32)
                s = int(starts[i]); e = int(ends[i])
                T = e - s
                c = settling_depth_per_window(D, gamma)
                lab_slice = labels[s:e]
                for code, name in CTX.items():
                    sel = lab_slice == code
                    if sel.any():
                        c_per_label[name].extend(c[sel].tolist())
                n_windows_done += 1
                n_pos_done += T
                if (i + 1) % 1000 == 0:
                    LOG.info("processed %d/%d windows", i + 1, N)

        sizes = {k: len(v) for k, v in c_per_label.items()}
        LOG.info("per-context sample sizes: %s", sizes)

        exon = np.asarray(c_per_label["coding_exon"], dtype=np.float64)
        intron = np.asarray(c_per_label["intron"], dtype=np.float64)
        if exon.size < 2 or intron.size < 2:
            raise RuntimeError("not enough exon/intron positions")
        u, p = mannwhitneyu(exon, intron, alternative="two-sided")
        d = cohens_d(intron, exon)  # +ve d => intron deeper

        alpha = 1.7e-51
        intron_higher = float(intron.mean()) > float(exon.mean())
        verdict = {
            "U": float(u), "p_two_sided": float(p),
            "alpha_bonf": alpha, "p_pass": bool(p < alpha),
            "cohens_d": float(d), "d_pass": bool(d >= 0.5),
            "intron_higher": intron_higher,
            "overall_pass": bool((p < alpha) and (d >= 0.5) and intron_higher),
            "exon_mean_c": float(exon.mean()), "intron_mean_c": float(intron.mean()),
        }

        per_ctx = {k: {"n": len(v), "mean_c": float(np.mean(v)) if v else None,
                       "median_c": float(np.median(v)) if v else None}
                   for k, v in c_per_label.items()}
        out = {
            "phase": PHASE,
            "chrom": "chr17",
            "gamma_cos": gamma,
            "n_windows_done": n_windows_done,
            "n_positions_total": n_pos_done,
            "per_context": per_ctx,
            "context_counts": sizes,
            "verdict_gate_b": verdict,
            "cohens_d_exon_vs_intron": float(d) if np.isfinite(d) else None,
        }
        (PHASE_OUT_DIR / "gate_b_chr17.json").write_text(json.dumps(out, indent=2))
        LOG.info("Gate B chr17 verdict: %s", verdict)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = ["intergenic", "intron", "coding_exon", "5utr", "3utr",
                     "splice_donor", "splice_acceptor", "repeat"]
            data = [c_per_label[k] for k in order]
            data = [d if len(d) > 0 else [0] for d in data]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.boxplot(data, labels=order, showfliers=False)
            ax.set_ylabel("settling depth c (1-based)")
            ax.set_title(f"Phase 2.2 chr17 Gate B (gamma={gamma:.4f})")
            plt.xticks(rotation=20)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F6_chr17_context_boxplot.{ext}", dpi=150)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            means = [per_ctx[k]["mean_c"] or 0 for k in order]
            ax2.bar(range(len(order)), means)
            ax2.set_xticks(range(len(order)))
            ax2.set_xticklabels(order, rotation=20)
            ax2.set_ylabel("mean settling depth c")
            ax2.set_title("chr17 mean c per context")
            fig2.tight_layout()
            for ext in ("pdf", "png"):
                fig2.savefig(PHASE_OUT_DIR / f"F4_chr17_profile.{ext}", dpi=150)
            plt.close(fig2)
        except Exception as e:
            LOG.warning("figures failed: %s", e)

        ru.write_done(PHASE, PHASE_OUT_DIR, {"verdict": verdict}, step_name="gate_b_chr17")
        LOG.info("Phase 2.2 Gate B chr17 done.")


if __name__ == "__main__":
    main()

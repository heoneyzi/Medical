"""Phase 2.3 — chr17 vs chr22 per-context settling depth comparison (CPU).

Reads the per-context lists from Phase 2.2 (chr17) and Phase 1.6 (chr22) gate JSONs
when available, plus the cached per-position c arrays for the actual KS test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase2.3"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.3"
LOG = ru.setup_logging(PHASE)

GAMMA_LOCKED = 0.39663

CTX = {
    0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
    5: "splice_donor", 6: "splice_acceptor",
}


def settling_depth_per_window(D_cos, gamma):
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.int64)


def collect_per_context(h5_path: Path, labels_path: Path, gamma: float) -> dict:
    """Stream H5 + labels and return {ctx_name: list of c values}."""
    import h5py
    labels = np.load(labels_path)
    out = {v: [] for v in CTX.values()}
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
            c = settling_depth_per_window(D, gamma)
            lab_slice = labels[s:e]
            for code, name in CTX.items():
                sel = lab_slice == code
                if sel.any():
                    out[name].extend(c[sel].tolist())
            if (i + 1) % 2000 == 0:
                LOG.info("  ...processed %d/%d", i + 1, N)
    return out


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="cross_chr"):
        from scipy.stats import ks_2samp

        rng = np.random.default_rng(42)

        chr17_h5 = ru.GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        chr22_h5 = ru.GDTR_ROOT / "results" / "phase1.6" / "chr22_cache.h5"
        chr17_lab = ru.GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy"
        chr22_lab = ru.GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"

        for p in (chr17_h5, chr22_h5, chr17_lab, chr22_lab):
            if not p.exists():
                raise FileNotFoundError(f"missing required input: {p}")

        LOG.info("collecting chr17 per-context...")
        c17 = collect_per_context(chr17_h5, chr17_lab, GAMMA_LOCKED)
        LOG.info("collecting chr22 per-context...")
        c22 = collect_per_context(chr22_h5, chr22_lab, GAMMA_LOCKED)

        # KS test per context (subsample to <=200_000 per side for tractability)
        SUB = 200_000
        per_context = {}
        for ctx in CTX.values():
            a = np.asarray(c17[ctx], dtype=np.float64)
            b = np.asarray(c22[ctx], dtype=np.float64)
            a_n = int(a.size); b_n = int(b.size)
            if a_n > SUB:
                a = a[rng.choice(a_n, SUB, replace=False)]
            if b_n > SUB:
                b = b[rng.choice(b_n, SUB, replace=False)]
            entry = {
                "chr17_n": a_n, "chr22_n": b_n,
                "chr17_mean_c": float(a.mean()) if a.size else None,
                "chr22_mean_c": float(b.mean()) if b.size else None,
                "chr17_median_c": float(np.median(a)) if a.size else None,
                "chr22_median_c": float(np.median(b)) if b.size else None,
                "delta_mean": (float(a.mean()) - float(b.mean()))
                              if (a.size and b.size) else None,
            }
            if a.size > 1 and b.size > 1:
                ks_stat, ks_p = ks_2samp(a, b, alternative="two-sided")
                entry["ks_stat"] = float(ks_stat)
                entry["ks_p"] = float(ks_p)
            else:
                entry["ks_stat"] = None
                entry["ks_p"] = None
            per_context[ctx] = entry
            LOG.info("ctx %s | chr17 n=%d mean=%.3f | chr22 n=%d mean=%.3f | delta=%s",
                     ctx, a_n, entry["chr17_mean_c"] or 0,
                     b_n, entry["chr22_mean_c"] or 0,
                     f"{entry['delta_mean']:+.3f}" if entry["delta_mean"] is not None else "NA")

        # Replication verdict: does intron > exon hold on chr17 like chr22?
        chr17_intron_gt_exon = (
            per_context["intron"]["chr17_mean_c"] is not None
            and per_context["coding_exon"]["chr17_mean_c"] is not None
            and per_context["intron"]["chr17_mean_c"] > per_context["coding_exon"]["chr17_mean_c"]
        )
        chr22_intron_gt_exon = (
            per_context["intron"]["chr22_mean_c"] is not None
            and per_context["coding_exon"]["chr22_mean_c"] is not None
            and per_context["intron"]["chr22_mean_c"] > per_context["coding_exon"]["chr22_mean_c"]
        )
        replication_verdict = {
            "chr17_intron_gt_exon": bool(chr17_intron_gt_exon),
            "chr22_intron_gt_exon": bool(chr22_intron_gt_exon),
            "qualitative_replicates": bool(chr17_intron_gt_exon and chr22_intron_gt_exon),
        }
        out = {
            "phase": PHASE,
            "gamma_cos": GAMMA_LOCKED,
            "subsample_per_context": SUB,
            "per_context": per_context,
            "replication_verdict": replication_verdict,
        }
        (PHASE_OUT_DIR / "cross_chr_comparison.json").write_text(json.dumps(out, indent=2))
        LOG.info("replication verdict: %s", replication_verdict)

        # Figure: chr17 vs chr22 mean c bar plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = ["intergenic", "intron", "coding_exon", "5utr", "3utr",
                     "splice_donor", "splice_acceptor"]
            x = np.arange(len(order))
            w = 0.4
            m17 = [per_context[k]["chr17_mean_c"] or 0 for k in order]
            m22 = [per_context[k]["chr22_mean_c"] or 0 for k in order]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(x - w / 2, m17, w, label="chr17", color="#d1495b")
            ax.bar(x + w / 2, m22, w, label="chr22", color="#2e86ab")
            ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
            ax.set_ylabel("mean settling depth c")
            ax.set_title("chr17 vs chr22 per-context mean settling depth")
            ax.legend()
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_cross_chr_means.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("figure failed: %s", e)

        ru.write_done(PHASE, PHASE_OUT_DIR,
                      {"replication_verdict": replication_verdict},
                      step_name="cross_chr")
        LOG.info("Phase 2.3 cross-chr comparison done.")


if __name__ == "__main__":
    main()

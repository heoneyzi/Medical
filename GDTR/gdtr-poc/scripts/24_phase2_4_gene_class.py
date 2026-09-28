"""Phase 2.4 — Gene-class stratification (cancer driver vs other).

Uses chr17 + chr22 caches. Builds per-position c arrays (overlapping windows
averaged), then aggregates per gene per class:
  - cancer_driver: TP53, BRCA1, ATM (chr17). chr22 has no genes in our 15-gene
    cancer set (Phase 0/1 cancer list excludes chr22 drivers).
  - other_chr17_protein_coding: remaining chr17 protein-coding genes
  - other_chr22_protein_coding: chr22 protein-coding genes

For each gene, mean settling depth = mean over its protein-coding gene span.
Then test cancer_driver vs other (Mann-Whitney + Cohen's d, gene-level).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

from phase0_src.src.stats import cohens_d as ph0_cohens_d  # noqa: E402

PHASE = "phase2.4"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.4"
LOG = ru.setup_logging(PHASE)

GAMMA_LOCKED = 0.39663
CHR17_CANCER_DRIVERS = {"TP53", "BRCA1", "ATM"}
SEED = 42


def settling_depth_per_window(D_cos, gamma):
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


def build_position_c(h5_path: Path, chrom_len: int, gamma: float) -> np.ndarray:
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
            if (i + 1) % 2000 == 0:
                LOG.info("  pos-c accum: %d/%d", i + 1, N)
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    nz = cov > 0
    out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
    return out


def per_gene_stats(pos_c: np.ndarray, genes: list[dict]) -> list[dict]:
    rows = []
    for g in genes:
        s = max(int(g["start"]), 0)
        e = min(int(g["end"]), pos_c.shape[0])
        if e <= s:
            continue
        seg = pos_c[s:e]
        valid = seg[~np.isnan(seg)]
        if valid.size < 10:
            continue
        rows.append({
            "gene_id": g.get("gene_id", g.get("id", "?")),
            "gene_name": g.get("gene_name", g.get("name", "?")),
            "n_pos": int(valid.size),
            "mean_c": float(valid.mean()),
            "median_c": float(np.median(valid)),
        })
    return rows


def main() -> None:
    np.random.seed(SEED)
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="gene_class"):
        from scipy.stats import mannwhitneyu

        chr17_h5 = ru.GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        chr22_h5 = ru.GDTR_ROOT / "results" / "phase1.6" / "chr22_cache.h5"
        chr17_lab = np.load(ru.GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy")
        chr22_lab = np.load(ru.GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy")
        chr17_class_path = ru.GDTR_ROOT / "data" / "baselines" / "chr17_gene_class.json"
        if not chr17_class_path.exists():
            raise FileNotFoundError(f"missing {chr17_class_path}; Phase 2.0 must run first")
        chr17_class = json.loads(chr17_class_path.read_text())

        # Build chr17 + chr22 per-position c
        LOG.info("building chr17 per-position c ...")
        pos17 = build_position_c(chr17_h5, chr17_lab.shape[0], GAMMA_LOCKED)
        np.save(PHASE_OUT_DIR / "chr17_position_c.npy", pos17)
        LOG.info("chr17 pos_c mean=%.3f NaN=%d", float(np.nanmean(pos17)), int(np.isnan(pos17).sum()))

        LOG.info("building chr22 per-position c ...")
        pos22 = build_position_c(chr22_h5, chr22_lab.shape[0], GAMMA_LOCKED)
        np.save(PHASE_OUT_DIR / "chr22_position_c.npy", pos22)
        LOG.info("chr22 pos_c mean=%.3f NaN=%d", float(np.nanmean(pos22)), int(np.isnan(pos22).sum()))

        # Build chr22 gene list using gffutils (no separate JSON)
        import gffutils
        db = gffutils.FeatureDB(str(ru.GDTR_ROOT / "data" / "annotation" /
                                    "gencode.v44.chr17_chr22.gtf.db"))
        chr22_genes = []
        for g in db.region(seqid="chr22", featuretype="gene"):
            gtype = g.attributes.get("gene_type", [None])[0]
            if gtype != "protein_coding":
                continue
            chr22_genes.append({
                "gene_id": g.id,
                "gene_name": g.attributes.get("gene_name", [g.id])[0],
                "start": int(g.start) - 1,
                "end": int(g.end),
                "strand": g.strand,
            })
        LOG.info("chr22 protein_coding genes: %d", len(chr22_genes))

        # Per-gene stats per group
        LOG.info("computing per-gene stats ...")
        cd_rows = per_gene_stats(pos17, chr17_class["cancer_driver"])
        other17_rows = per_gene_stats(pos17, chr17_class["other"])
        other22_rows = per_gene_stats(pos22, chr22_genes)
        LOG.info("cancer_driver(chr17)=%d  other_chr17=%d  other_chr22=%d",
                 len(cd_rows), len(other17_rows), len(other22_rows))

        # Group-level vectors of mean_c
        cd_vec = np.array([r["mean_c"] for r in cd_rows], dtype=np.float64)
        o17_vec = np.array([r["mean_c"] for r in other17_rows], dtype=np.float64)
        o22_vec = np.array([r["mean_c"] for r in other22_rows], dtype=np.float64)
        all_other_vec = np.concatenate([o17_vec, o22_vec])

        def grp(label, v):
            return {
                "label": label,
                "n_genes": int(v.size),
                "mean_c": float(v.mean()) if v.size else None,
                "median_c": float(np.median(v)) if v.size else None,
                "std_c": float(v.std(ddof=1)) if v.size > 1 else None,
            }

        groups = {
            "cancer_driver": grp("cancer_driver (chr17: TP53,BRCA1,ATM)", cd_vec),
            "other_chr17_protein_coding": grp("other chr17 protein_coding", o17_vec),
            "other_chr22_protein_coding": grp("other chr22 protein_coding", o22_vec),
            "all_other_combined": grp("other_chr17 + other_chr22", all_other_vec),
        }

        # Test cancer_driver vs all_other_combined
        if cd_vec.size >= 2 and all_other_vec.size >= 2:
            U, p = mannwhitneyu(cd_vec, all_other_vec, alternative="two-sided")
            d = ph0_cohens_d(cd_vec, all_other_vec)
            test_vs_all = {
                "U": float(U), "p_two_sided": float(p),
                "cohens_d_cd_minus_other": float(d) if np.isfinite(d) else None,
                "cancer_driver_mean": float(cd_vec.mean()),
                "other_mean": float(all_other_vec.mean()),
                "direction": "cd_lower" if cd_vec.mean() < all_other_vec.mean() else "cd_higher",
            }
        else:
            test_vs_all = {"error": "insufficient sample size for MWU"}

        # Test cancer_driver vs other_chr17 (within-chromosome control)
        if cd_vec.size >= 2 and o17_vec.size >= 2:
            U17, p17 = mannwhitneyu(cd_vec, o17_vec, alternative="two-sided")
            d17 = ph0_cohens_d(cd_vec, o17_vec)
            test_vs_chr17 = {
                "U": float(U17), "p_two_sided": float(p17),
                "cohens_d_cd_minus_other17": float(d17) if np.isfinite(d17) else None,
                "cancer_driver_mean": float(cd_vec.mean()),
                "other_chr17_mean": float(o17_vec.mean()),
                "direction": "cd_lower" if cd_vec.mean() < o17_vec.mean() else "cd_higher",
            }
        else:
            test_vs_chr17 = {"error": "insufficient sample size for MWU"}

        out = {
            "phase": PHASE,
            "gamma_cos": GAMMA_LOCKED,
            "seed": SEED,
            "groups": groups,
            "cancer_driver_genes_used": [r["gene_name"] for r in cd_rows],
            "test_cd_vs_all_other": test_vs_all,
            "test_cd_vs_other_chr17": test_vs_chr17,
            "per_gene_cancer_driver": cd_rows,
        }
        (PHASE_OUT_DIR / "gene_class_stratification.json").write_text(json.dumps(out, indent=2))
        LOG.info("test cd vs all_other: %s", test_vs_all)
        LOG.info("test cd vs chr17 only: %s", test_vs_chr17)

        # Figures
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            data = [cd_vec, o17_vec, o22_vec]
            labels = [f"cancer_driver\n(n={cd_vec.size})",
                      f"other_chr17\n(n={o17_vec.size})",
                      f"other_chr22\n(n={o22_vec.size})"]
            ax.boxplot(data, labels=labels, showfliers=True)
            ax.set_ylabel("per-gene mean settling depth c")
            ax.set_title("Phase 2.4 gene-class stratification")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_gene_class_boxplot.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("figure failed: %s", e)

        ru.write_done(PHASE, PHASE_OUT_DIR,
                      {"groups": groups, "test_vs_all_other": test_vs_all},
                      step_name="gene_class")
        LOG.info("Phase 2.4 done.")


if __name__ == "__main__":
    main()

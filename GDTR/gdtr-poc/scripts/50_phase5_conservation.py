"""Phase 5: gDTR-Conservation discordance Q2 analysis on chr22.

Find regions of high settling depth (deep computation) but low evolutionary
conservation. Candidate "recently evolved regulatory" elements that are
functionally important but not conserved across mammalian lineages.

CPU-only. seed=42. HIGHER c = DEEPER computation (high gDTR).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import _runner_utils as ru

PHASE = "phase5"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase5"
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)

SEED = 42
CHR = "chr22"
CHR_LEN = 50_818_468

GDTR_TOP_PCT = 0.25
PHYLOP_BOT_PCT = 0.25
MIN_REGION_BP = 100

CTX_NAME = {
    0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
    5: "splice_donor", 6: "splice_acceptor", 7: "repeat",
}

CONS_DIR = ru.GDTR_ROOT / "data" / "conservation"
PHYLOP_BW = CONS_DIR / "hg38.phyloP100way.bw"
CCRE_BED = CONS_DIR / "GRCh38-cCREs.bed"
RDHS_BED = CONS_DIR / "GRCh38-rDHSs.bed"
RMSK_TXT = CONS_DIR / "rmsk.txt.gz"


def load_phylop_chr22(bw_path: Path, chrom: str, chrom_len: int) -> np.ndarray:
    import pyBigWig
    LOG.info("opening %s", bw_path)
    bw = pyBigWig.open(str(bw_path))
    if chrom not in bw.chroms():
        raise RuntimeError(f"{chrom} not in bigWig (have {list(bw.chroms())[:5]})")
    bw_len = bw.chroms()[chrom]
    LOG.info("%s in bigWig: length=%d (analysis len=%d)", chrom, bw_len, chrom_len)
    n = min(bw_len, chrom_len)
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    chunk = 5_000_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        v = np.asarray(bw.values(chrom, s, e, numpy=True), dtype=np.float32)
        out[s:e] = v
    bw.close()
    return out


def load_bed_chr22_intervals(bed_path: Path, chrom: str) -> np.ndarray:
    starts, ends = [], []
    with open(bed_path, "r") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            try:
                starts.append(int(parts[1])); ends.append(int(parts[2]))
            except ValueError:
                continue
    if not starts:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.empty((len(starts), 2), dtype=np.int64)
    arr[:, 0] = starts; arr[:, 1] = ends
    arr = arr[np.argsort(arr[:, 0])]
    return arr


def load_rmsk_chr22(rmsk_gz: Path, chrom: str):
    by_class = {}
    with gzip.open(rmsk_gz, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13 or parts[5] != chrom:
                continue
            try:
                s = int(parts[6]); e = int(parts[7])
            except ValueError:
                continue
            cls = parts[11]
            by_class.setdefault(cls, []).append((s, e))
    out = {}
    for cls, ivs in by_class.items():
        arr = np.array(ivs, dtype=np.int64)
        arr = arr[np.argsort(arr[:, 0])]
        out[cls] = arr
    return out


def intervals_to_mask(intervals: np.ndarray, chrom_len: int) -> np.ndarray:
    m = np.zeros(chrom_len, dtype=bool)
    if intervals.shape[0] == 0:
        return m
    for s, e in intervals:
        s = max(0, int(s)); e = min(chrom_len, int(e))
        if e > s:
            m[s:e] = True
    return m


def contiguous_runs(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return np.stack([starts, ends], axis=1).astype(np.int64)


def hypergeom_p(k: int, K: int, n: int, N: int) -> float:
    from scipy.stats import hypergeom
    if min(K, n, N) <= 0 or k <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, N, K, n))


def main():
    np.random.seed(SEED)
    ru.write_status(PHASE, "running", {"step": "load"})

    LOG.info("loading per-position c (settling depth)")
    c_arr = np.load(ru.GDTR_ROOT / "results/phase1.6_sub/chr22_position_c.npy")
    LOG.info("loading per-position labels")
    labels = np.load(ru.GDTR_ROOT / "data/annotation/chr22_position_labels.npy")
    assert c_arr.shape[0] == CHR_LEN
    assert labels.shape[0] == CHR_LEN

    nan_c = np.isnan(c_arr)
    LOG.info("c: %d nan / %d (%.2f%%)", int(nan_c.sum()), CHR_LEN, 100.0*float(nan_c.mean()))

    LOG.info("loading PhyloP from %s", PHYLOP_BW)
    phylop = load_phylop_chr22(PHYLOP_BW, CHR, CHR_LEN)
    nan_p = np.isnan(phylop)
    LOG.info("phyloP: %d nan / %d (%.2f%%)", int(nan_p.sum()), CHR_LEN, 100.0*float(nan_p.mean()))

    valid = ~(nan_c | nan_p)
    n_valid = int(valid.sum())
    LOG.info("valid (both): %d (%.2f%%)", n_valid, 100.0*n_valid/CHR_LEN)

    c_valid = c_arr[valid]; p_valid = phylop[valid]
    c_thr = float(np.quantile(c_valid, 1.0 - GDTR_TOP_PCT))
    p_thr = float(np.quantile(p_valid, PHYLOP_BOT_PCT))
    LOG.info("c_threshold (top %.0f%%) = %.4f", GDTR_TOP_PCT*100, c_thr)
    LOG.info("phyloP_threshold (bot %.0f%%) = %.4f", PHYLOP_BOT_PCT*100, p_thr)

    high_g = c_arr >= c_thr
    low_cons = phylop <= p_thr

    q = np.zeros(CHR_LEN, dtype=np.uint8)
    q[valid & high_g & ~low_cons] = 1
    q[valid & high_g & low_cons] = 2
    q[valid & ~high_g & ~low_cons] = 3
    q[valid & ~high_g & low_cons] = 4

    sizes = {f"Q{i}": int((q == i).sum()) for i in (1, 2, 3, 4)}
    sizes["NA"] = int((q == 0).sum())
    sizes_pct = {k: 100.0*v/CHR_LEN for k, v in sizes.items()}
    LOG.info("quadrant sizes (bp): %s", sizes)

    np.save(PHASE_OUT_DIR / "chr22_quadrants.npy", q)

    q2_runs = contiguous_runs(q == 2)
    lens = q2_runs[:, 1] - q2_runs[:, 0]
    keep = lens >= MIN_REGION_BP
    q2_regions = q2_runs[keep]
    LOG.info("Q2 runs: %d total, %d >=%dbp", int(q2_runs.shape[0]), int(q2_regions.shape[0]), MIN_REGION_BP)

    rows = []
    for s, e in q2_regions:
        cseg = c_arr[s:e]; pseg = phylop[s:e]; labseg = labels[s:e]
        mc = float(np.nanmean(cseg)); mp = float(np.nanmean(pseg))
        if labseg.size:
            vals, counts = np.unique(labseg, return_counts=True)
            dom = int(vals[counts.argmax()])
            dom_name = CTX_NAME.get(dom, str(dom))
        else:
            dom_name = "NA"
        rows.append((CHR, int(s), int(e), int(e-s), mc, mp, dom_name))

    bed_path = PHASE_OUT_DIR / "q2_regions.bed"
    with open(bed_path, "w") as f:
        f.write("#chrom\tstart\tend\tlength\tmean_c\tmean_phylop\tdom_context\n")
        for r in rows:
            f.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.4f}" for x in r) + "\n")
    LOG.info("wrote %s (%d regions)", bed_path, len(rows))

    LOG.info("loading annotations for chr22")
    ccre_iv = load_bed_chr22_intervals(CCRE_BED, CHR)
    rdhs_iv = load_bed_chr22_intervals(RDHS_BED, CHR)
    LOG.info("cCREs: %d, rDHSs: %d", ccre_iv.shape[0], rdhs_iv.shape[0])
    rmsk_by_class = load_rmsk_chr22(RMSK_TXT, CHR)
    LOG.info("rmsk classes: %s", {k: int(v.shape[0]) for k, v in rmsk_by_class.items()})

    annotations = {
        "ENCODE_cCRE": intervals_to_mask(ccre_iv, CHR_LEN),
        "ENCODE_rDHS": intervals_to_mask(rdhs_iv, CHR_LEN),
    }
    for cls in ("SINE", "LINE", "LTR", "DNA", "Simple_repeat", "Low_complexity", "Satellite", "Retroposon"):
        if cls in rmsk_by_class:
            annotations[f"rmsk_{cls}"] = intervals_to_mask(rmsk_by_class[cls], CHR_LEN)

    valid_mask = valid
    N = int(valid_mask.sum())
    q2_mask = (q == 2)
    n = int(q2_mask.sum())
    LOG.info("enrichment N=%d, Q2 n=%d", N, n)

    enrichment = {}
    for ann_name, ann_mask in annotations.items():
        K = int((ann_mask & valid_mask).sum())
        k_ = int((ann_mask & q2_mask).sum())
        if N == 0 or n == 0 or K == 0:
            fold = float("nan"); pval = 1.0
        else:
            fold = (k_/n) / (K/N) if K else float("nan")
            pval = hypergeom_p(k_, K, n, N)
        enrichment[ann_name] = {
            "n_overlap_q2": k_, "n_overlap_total_valid": K,
            "n_q2": n, "n_total_valid": N,
            "obs_frac_in_q2": (k_/n) if n else 0.0,
            "exp_frac_genome": (K/N) if N else 0.0,
            "fold_enrichment": fold,
            "hypergeom_pval_oneSided_overrep": pval,
        }
        LOG.info("  %-22s  k=%-7d K=%-9d  fold=%.3f  p=%.3e", ann_name, k_, K, fold, pval)

    ctx_q2 = {}
    for cid, name in CTX_NAME.items():
        m = (labels == cid)
        K = int((m & valid_mask).sum())
        k_ = int((m & q2_mask).sum())
        if N and n and K:
            fold = (k_/n) / (K/N); pval = hypergeom_p(k_, K, n, N)
        else:
            fold, pval = float("nan"), 1.0
        ctx_q2[name] = {"n_overlap_q2": k_, "n_overlap_total_valid": K,
                        "fold_enrichment": fold, "hypergeom_pval_oneSided_overrep": pval}

    out = {
        "phase": PHASE, "seed": SEED, "chrom": CHR, "chrom_len": CHR_LEN,
        "n_valid_positions": N, "frac_valid": N/CHR_LEN,
        "frac_phylop_nan": float(nan_p.mean()), "frac_c_nan": float(nan_c.mean()),
        "c_threshold_top25pct": c_thr, "phylop_threshold_bot25pct": p_thr,
        "quadrant_sizes_bp": sizes, "quadrant_pct_of_chr22": sizes_pct,
        "q2_n_runs_total": int(q2_runs.shape[0]),
        "q2_n_regions_min100bp": int(q2_regions.shape[0]),
        "annotation_enrichment": enrichment,
        "context_enrichment_in_q2": ctx_q2,
    }

    def _safe(x):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    with open(PHASE_OUT_DIR / "q2_enrichment.json", "w") as fp:
        json.dump(out, fp, indent=2, default=_safe)
    LOG.info("wrote q2_enrichment.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(SEED)
        idx_valid = np.flatnonzero(valid_mask)
        sample_n = min(100_000, idx_valid.size)
        sub = rng.choice(idx_valid, size=sample_n, replace=False)
        sub_q = q[sub]; cs = c_arr[sub]; ps = phylop[sub]
        colors = {1: "#1f77b4", 2: "#d62728", 3: "#7f7f7f", 4: "#cccccc"}
        labels_q = {1: "Q1: hi-c hi-cons", 2: "Q2: hi-c lo-cons (recently evolved)",
                    3: "Q3: lo-c hi-cons", 4: "Q4: lo-c lo-cons"}
        fig, ax = plt.subplots(figsize=(7, 6))
        for qi in (4, 3, 1, 2):
            m = sub_q == qi
            if m.any():
                ax.scatter(cs[m], ps[m], s=1, alpha=0.35, c=colors[qi], label=labels_q[qi])
        ax.axvline(c_thr, color="k", lw=0.6, ls="--")
        ax.axhline(p_thr, color="k", lw=0.6, ls="--")
        ax.set_xlabel("settling depth c (per-position, chr22)")
        ax.set_ylabel("PhyloP 100-way (per-position)")
        ax.set_title(f"chr22 gDTR vs PhyloP  (Q2 regions>=100bp = {len(rows)})")
        ax.legend(loc="lower right", fontsize=8, markerscale=4)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(PHASE_OUT_DIR / f"F_quadrant_scatter.{ext}", dpi=150)
        plt.close(fig)

        names = list(enrichment.keys())
        folds = [enrichment[k]["fold_enrichment"] if np.isfinite(enrichment[k]["fold_enrichment"]) else 0.0 for k in names]
        pvals = [enrichment[k]["hypergeom_pval_oneSided_overrep"] for k in names]
        fig, ax = plt.subplots(figsize=(8, 0.4*len(names) + 1.5))
        ypos = np.arange(len(names))
        ax.barh(ypos, folds, color=["#d62728" if f >= 1 else "#1f77b4" for f in folds])
        ax.axvline(1.0, color="k", lw=0.7, ls="--", label="genome-wide expectation")
        ax.set_yticks(ypos); ax.set_yticklabels(names)
        ax.set_xlabel("Fold enrichment in Q2 (vs valid genome positions)")
        ax.set_title("Q2 (high gDTR + low conservation) enrichment, chr22")
        for i, (f, p) in enumerate(zip(folds, pvals)):
            txt = f"  {f:.2f}x  p={p:.2e}" if (np.isfinite(p) and p > 0) else f"  {f:.2f}x"
            ax.text(max(f, 1.0), i, txt, va="center", fontsize=8)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(PHASE_OUT_DIR / f"F_q2_enrichment.{ext}", dpi=150)
        plt.close(fig)
        LOG.info("plots written")
    except Exception as e:
        LOG.warning("plotting failed: %s", e)

    ru.write_status(PHASE, "done", {
        "n_valid_positions": N,
        "q2_regions_min100bp": int(q2_regions.shape[0]),
        "q2_size_bp": int(sizes["Q2"]),
    })
    LOG.info("phase 5 done")


if __name__ == "__main__":
    main()

"""P3B-3 chr17 — multi-chromosome Q2 replication (Plan §3.4.B-3.2).

Mirrors 50b_phase5_smoothed.py but on chr17:
  - load chr17 per-position c (build from chr17_cache.h5 if missing).
  - load chr17 PhyloP-100way (download if missing under
    ~/gDTR/data/conservation/).
  - smooth both with 100-bp box-car.
  - top-25% c × bot-25% PhyloP → Q1..Q4 quadrants.
  - region-level enrichment of cCRE / repeat classes / etc.
  - hypergeometric p, fold enrichment.

Outputs:
    ~/gDTR/results/p3b3_chr17/q2_enrichment_chr17.json
    ~/gDTR/results/p3b3_chr17/F_q2_chr17_vs_chr22.{pdf,png}
    ~/gDTR/results/p3b3_chr17/_status.json
    ~/gDTR/results/p3b3_chr17/_done
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
GAMMA_LOCKED = 0.39663
SMOOTH = 100
GTOP = 0.25
PBOT = 0.25
MIN_REGION = 100

CTX_NAME = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
            5: "splice_donor", 6: "splice_acceptor", 7: "repeat"}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p3b3_chr17")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def settling_depth(D_cos: np.ndarray, gamma: float) -> np.ndarray:
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


def build_pos_c(h5_path: Path, chrom_len: int, gamma: float,
                log: logging.Logger, smoke: bool = False) -> np.ndarray:
    import h5py
    sum_c = np.zeros(chrom_len, dtype=np.float64)
    cov = np.zeros(chrom_len, dtype=np.uint16)
    with h5py.File(h5_path, "r") as h5:
        N = h5["D_cos"].shape[0]
        done = h5["done_mask"][:].astype(bool)
        starts = h5["starts"][:]; ends = h5["ends"][:]
        idx = np.flatnonzero(done)
        if smoke:
            idx = idx[:50]
        for k, i in enumerate(idx):
            i = int(i)
            s = int(starts[i]); e = int(ends[i])
            D = h5["D_cos"][i].astype(np.float32)
            c = settling_depth(D, gamma)
            seg_end = min(e, chrom_len)
            seg_len = seg_end - s
            if seg_len <= 0:
                continue
            sum_c[s:seg_end] += c[:seg_len]
            cov[s:seg_end] += 1
            if (k + 1) % 1000 == 0:
                log.info("  built per-pos c %d windows", k + 1)
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    nz = cov > 0
    out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
    return out


def smooth_box(arr: np.ndarray, w: int) -> np.ndarray:
    finite = np.isfinite(arr)
    a = np.where(finite, arr, 0.0).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    cn = np.concatenate([[0],   np.cumsum(finite.astype(np.int64))])
    half = w // 2
    n = arr.size
    s = np.maximum(0, np.arange(n) - half)
    e = np.minimum(n, np.arange(n) + (w - half))
    sums = cs[e] - cs[s]
    cnts = cn[e] - cn[s]
    out = np.full(n, np.nan, dtype=np.float32)
    valid = cnts >= max(1, w // 4)
    out[valid] = (sums[valid] / cnts[valid]).astype(np.float32)
    return out


def load_phylop_chr(bw_path: Path, chrom: str, chrom_len: int) -> np.ndarray:
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    if chrom not in bw.chroms():
        raise RuntimeError(f"{chrom} not in bigWig")
    n = min(bw.chroms()[chrom], chrom_len)
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    chunk = 5_000_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.asarray(bw.values(chrom, s, e, numpy=True), dtype=np.float32)
    bw.close()
    return out


def load_bed_intervals(bed_path: Path, chrom: str) -> np.ndarray:
    if not bed_path.exists():
        return np.zeros((0, 2), dtype=np.int64)
    starts, ends = [], []
    with bed_path.open() as f:
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
    return arr[np.argsort(arr[:, 0])]


def load_rmsk(rmsk_path: Path, chrom: str) -> dict[str, np.ndarray]:
    if rmsk_path is None or not rmsk_path.exists():
        return {}
    by = {}
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13 or parts[5] != chrom:
                continue
            try:
                s = int(parts[6]); e = int(parts[7])
            except ValueError:
                continue
            cls = parts[11]
            by.setdefault(cls, []).append((s, e))
    out = {}
    for cls, ivs in by.items():
        arr = np.array(ivs, dtype=np.int64)
        out[cls] = arr[np.argsort(arr[:, 0])]
    return out


def to_mask(intervals: np.ndarray, chrom_len: int) -> np.ndarray:
    m = np.zeros(chrom_len, dtype=bool)
    if intervals.shape[0] == 0:
        return m
    for s, e in intervals:
        s = max(0, int(s)); e = min(chrom_len, int(e))
        if e > s:
            m[s:e] = True
    return m


def runs(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    s = np.where(diff == 1)[0]; e = np.where(diff == -1)[0]
    return np.stack([s, e], axis=1).astype(np.int64)


def hypergeom_p(k: int, K: int, n: int, N: int) -> float:
    from scipy.stats import hypergeom
    if min(K, n, N) <= 0 or k <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, N, K, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p3b3_chr17"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p3b3_chr17: _done exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p3b3_chr17", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        # ---- chr17 per-position c ----
        chr17_lab_path = GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy"
        if not chr17_lab_path.exists():
            raise FileNotFoundError(f"missing {chr17_lab_path}")
        chr17_labels = np.load(chr17_lab_path)
        chr17_len = int(chr17_labels.shape[0])

        cached_c = GDTR_ROOT / "results" / "phase2.4" / "chr17_position_c.npy"
        cached_c_alt = GDTR_ROOT / "results" / "phase2.5" / "chr17_position_c.npy"
        cached_c_p1a = GDTR_ROOT / "results" / "p1a" / "chr17_position_c.npy"
        if cached_c.exists():
            chr17_c = np.load(cached_c)
        elif cached_c_alt.exists():
            chr17_c = np.load(cached_c_alt)
        elif cached_c_p1a.exists():
            chr17_c = np.load(cached_c_p1a)
        else:
            h5_path = GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"need cached chr17_position_c.npy or chr17_cache.h5")
            log.info("computing chr17 per-position c from cache")
            chr17_c = build_pos_c(h5_path, chr17_len, GAMMA_LOCKED, log,
                                   smoke=args.smoke)
            np.save(out_dir / "chr17_position_c.npy", chr17_c)
        log.info("chr17 c: %d nan / %d", int(np.isnan(chr17_c).sum()), chr17_c.size)

        # ---- chr17 PhyloP ----
        cons_dir = GDTR_ROOT / "data" / "conservation"
        bw_path = cons_dir / "hg38.phyloP100way.bw"
        if not bw_path.exists():
            raise FileNotFoundError(
                f"missing {bw_path}; this script does not auto-download bigWig"
            )
        log.info("loading chr17 PhyloP from %s", bw_path)
        phylop = load_phylop_chr(bw_path, "chr17", chr17_len)

        # ---- smoothing ----
        log.info("smoothing %dbp", SMOOTH)
        c_s = smooth_box(chr17_c, SMOOTH)
        p_s = smooth_box(phylop, SMOOTH)
        valid = ~(np.isnan(c_s) | np.isnan(p_s))
        N = int(valid.sum())
        log.info("valid positions: %d (%.2f%%)", N, 100.0 * N / chr17_len)

        cv = c_s[valid]; pv = p_s[valid]
        c_thr = float(np.quantile(cv, 1.0 - GTOP))
        p_thr = float(np.quantile(pv, PBOT))

        high_g = c_s >= c_thr
        low_c = p_s <= p_thr
        q = np.zeros(chr17_len, dtype=np.uint8)
        q[valid & high_g & ~low_c] = 1
        q[valid & high_g & low_c] = 2
        q[valid & ~high_g & ~low_c] = 3
        q[valid & ~high_g & low_c] = 4
        sizes = {f"Q{i}": int((q == i).sum()) for i in (1, 2, 3, 4)}
        sizes["NA"] = int((q == 0).sum())
        sizes_pct = {k: 100.0 * v / chr17_len for k, v in sizes.items()}
        log.info("quadrant sizes: %s", sizes)
        np.save(out_dir / "chr17_quadrants.npy", q)

        rs = runs(q == 2)
        L = rs[:, 1] - rs[:, 0]
        rg = rs[L >= MIN_REGION]
        log.info("Q2 regions ≥%dbp = %d (of %d total runs)",
                 MIN_REGION, int(rg.shape[0]), int(rs.shape[0]))

        # ---- annotations ----
        ext_dir = GDTR_ROOT / "data" / "external"
        ann_masks = {}
        ccre = ext_dir / "ccre_els_only_chr17.bed"
        if not ccre.exists():
            ccre = cons_dir / "GRCh38-cCREs.bed"
        ann_masks["ENCODE_cCRE"] = to_mask(load_bed_intervals(ccre, "chr17"), chr17_len)
        rdhs = cons_dir / "GRCh38-rDHSs.bed"
        ann_masks["ENCODE_rDHS"] = to_mask(load_bed_intervals(rdhs, "chr17"), chr17_len)

        rmsk_path = cons_dir / "rmsk.txt.gz"
        if not rmsk_path.exists():
            rmsk_path = ext_dir / "rmsk.txt.gz"
        if rmsk_path.exists():
            rmsk = load_rmsk(rmsk_path, "chr17")
            for cls in ("SINE", "LINE", "LTR", "DNA",
                        "Simple_repeat", "Low_complexity",
                        "Satellite", "Retroposon"):
                if cls in rmsk:
                    ann_masks[f"rmsk_{cls}"] = to_mask(rmsk[cls], chr17_len)

        q2_mask = (q == 2)
        n_q2 = int(q2_mask.sum())
        log.info("N=%d Q2=%d", N, n_q2)
        enrichment = {}
        for nm, mk in ann_masks.items():
            K = int((mk & valid).sum())
            k_ = int((mk & q2_mask).sum())
            if N == 0 or n_q2 == 0 or K == 0:
                fold = float("nan"); pv_ = 1.0
            else:
                fold = (k_ / n_q2) / (K / N) if K else float("nan")
                pv_ = hypergeom_p(k_, K, n_q2, N)
            enrichment[nm] = {
                "n_overlap_q2": k_, "n_overlap_total_valid": K,
                "n_q2": n_q2, "n_total_valid": N,
                "fold_enrichment": fold,
                "hypergeom_pval_oneSided_overrep": pv_,
            }
            log.info("  %-22s k=%-7d K=%-9d fold=%.3f p=%.3e", nm, k_, K, fold, pv_)

        out = {
            "phase": "p3b3_chr17", "seed": 42, "chrom": "chr17",
            "chrom_len": chr17_len, "smoothing_bp": SMOOTH,
            "n_valid_positions": N,
            "c_threshold_top25pct_smoothed": c_thr,
            "phylop_threshold_bot25pct_smoothed": p_thr,
            "quadrant_sizes_bp": sizes,
            "quadrant_pct_of_chr17": sizes_pct,
            "q2_n_runs_total": int(rs.shape[0]),
            "q2_n_regions_min100bp": int(rg.shape[0]),
            "annotation_enrichment": enrichment,
        }

        def _safe(x):
            if isinstance(x, float) and not np.isfinite(x):
                return None
            return x

        (out_dir / "q2_enrichment_chr17.json").write_text(
            json.dumps(out, indent=2, default=_safe)
        )
        log.info("wrote q2_enrichment_chr17.json")

        # Side-by-side figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            chr22_json = GDTR_ROOT / "results" / "phase5" / "q2_enrichment.json"
            chr22 = json.loads(chr22_json.read_text()) if chr22_json.exists() else None

            names = list(enrichment.keys())
            folds17 = [enrichment[k]["fold_enrichment"]
                       if np.isfinite(enrichment[k]["fold_enrichment"]) else 0.0
                       for k in names]
            folds22 = []
            if chr22 is not None:
                ann22 = chr22.get("annotation_enrichment", {})
                for k in names:
                    f = ann22.get(k, {}).get("fold_enrichment")
                    folds22.append(float(f) if f is not None and np.isfinite(float(f)) else 0.0)
            else:
                folds22 = [0.0] * len(names)

            ypos = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.5))
            ax.barh(ypos - 0.2, folds17, height=0.4,
                    color="#d1495b", label="chr17")
            ax.barh(ypos + 0.2, folds22, height=0.4,
                    color="#1f77b4", label="chr22")
            ax.axvline(1.0, color="k", ls="--", lw=0.7,
                       label="genome-wide expectation")
            ax.set_yticks(ypos); ax.set_yticklabels(names)
            ax.set_xlabel("Fold enrichment in Q2")
            ax.set_title("Q2 enrichment chr17 vs chr22")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            for ext_ in ("pdf", "png"):
                fig.savefig(out_dir / f"F_q2_chr17_vs_chr22.{ext_}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        # Verdict (Plan §3.4.B-3.3)
        verdict = "PASS"
        if rg.shape[0] < 100:
            verdict = "REFRAME"

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr17",
            "status": verdict,
            "reason": f"q2_n_regions_min100bp={int(rg.shape[0])}",
            "metrics": {"q2_n_regions_min100bp": int(rg.shape[0]),
                        "q2_pct_chr17": sizes_pct["Q2"]},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p3b3_chr17 DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p3b3_chr17 FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr17",
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 1,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

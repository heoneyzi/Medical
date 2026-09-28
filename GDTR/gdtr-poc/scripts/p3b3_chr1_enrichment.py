"""P3B-3 chr1 enrichment (Plan §3.4.B-3.2 chr1 optional, CPU step).

Reads the chr1 per-position c built by p3b3_chr1_forward.py and runs the
same Q2 quadrant + repeat / cCRE enrichment analysis as p3b3_q2_chr17.py
on the chr1 sub-sample.

Outputs:
    ~/gDTR/results/p3b3_chr1/q2_enrichment_chr1.json
    ~/gDTR/results/p3b3_chr1/_status.json
    ~/gDTR/results/p3b3_chr1/_done
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
SMOOTH = 100
GTOP = 0.25
PBOT = 0.25
MIN_REGION = 100


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p3b3_chr1_enrich")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def smooth_box(arr, w):
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
    n = min(bw.chroms().get(chrom, 0), chrom_len)
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


def to_mask(intervals: np.ndarray, chrom_len: int) -> np.ndarray:
    m = np.zeros(chrom_len, dtype=bool)
    if intervals.shape[0] == 0:
        return m
    for s, e in intervals:
        s = max(0, int(s)); e = min(chrom_len, int(e))
        if e > s:
            m[s:e] = True
    return m


def runs(mask):
    if mask.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    s = np.where(diff == 1)[0]; e = np.where(diff == -1)[0]
    return np.stack([s, e], axis=1).astype(np.int64)


def hypergeom_p(k, K, n, N):
    from scipy.stats import hypergeom
    if min(K, n, N) <= 0 or k <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, N, K, n))


def load_rmsk_class(rmsk_path: Path, chrom: str) -> dict:
    if not rmsk_path.exists():
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
        a = np.array(ivs, dtype=np.int64)
        out[cls] = a[np.argsort(a[:, 0])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p3b3_chr1"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p3b3_chr1_enrichment: _done exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p3b3_chr1_enrichment", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        c_path = out_dir / "chr1_position_c.npy"
        if not c_path.exists():
            raise FileNotFoundError(f"missing {c_path} (run p3b3_chr1_forward.py)")
        c_arr = np.load(c_path)
        chr1_len = c_arr.shape[0]
        log.info("chr1 c: %d nan / %d", int(np.isnan(c_arr).sum()), chr1_len)

        cons_dir = GDTR_ROOT / "data" / "conservation"
        bw_path = cons_dir / "hg38.phyloP100way.bw"
        if not bw_path.exists():
            raise FileNotFoundError(f"missing {bw_path}")
        phylop = load_phylop_chr(bw_path, "chr1", chr1_len)

        c_s = smooth_box(c_arr, SMOOTH)
        p_s = smooth_box(phylop, SMOOTH)
        valid = ~(np.isnan(c_s) | np.isnan(p_s))
        N = int(valid.sum())
        if N == 0:
            raise RuntimeError("no valid positions on chr1 sub-sample")
        cv = c_s[valid]; pv = p_s[valid]
        c_thr = float(np.quantile(cv, 1.0 - GTOP))
        p_thr = float(np.quantile(pv, PBOT))

        high_g = c_s >= c_thr
        low_c = p_s <= p_thr
        q = np.zeros(chr1_len, dtype=np.uint8)
        q[valid & high_g & ~low_c] = 1
        q[valid & high_g & low_c] = 2
        q[valid & ~high_g & ~low_c] = 3
        q[valid & ~high_g & low_c] = 4
        sizes = {f"Q{i}": int((q == i).sum()) for i in (1, 2, 3, 4)}
        sizes_pct = {k: 100.0 * v / chr1_len for k, v in sizes.items()}
        rs = runs(q == 2); L = rs[:, 1] - rs[:, 0]
        rg = rs[L >= MIN_REGION]
        log.info("Q2 regions ≥%dbp = %d", MIN_REGION, int(rg.shape[0]))

        ann_masks = {}
        ext_dir = GDTR_ROOT / "data" / "external"
        ccre = ext_dir / "ccre_els_only_chr1.bed"
        if not ccre.exists():
            ccre = cons_dir / "GRCh38-cCREs.bed"
        ann_masks["ENCODE_cCRE"] = to_mask(load_bed_intervals(ccre, "chr1"), chr1_len)
        rdhs = cons_dir / "GRCh38-rDHSs.bed"
        ann_masks["ENCODE_rDHS"] = to_mask(load_bed_intervals(rdhs, "chr1"), chr1_len)

        rmsk_path = cons_dir / "rmsk.txt.gz"
        if not rmsk_path.exists():
            rmsk_path = ext_dir / "rmsk.txt.gz"
        rmsk = load_rmsk_class(rmsk_path, "chr1") if rmsk_path.exists() else {}
        for cls in ("SINE", "LINE", "LTR", "DNA",
                    "Simple_repeat", "Low_complexity",
                    "Satellite", "Retroposon"):
            if cls in rmsk:
                ann_masks[f"rmsk_{cls}"] = to_mask(rmsk[cls], chr1_len)

        q2 = (q == 2); n_q2 = int(q2.sum())
        enrichment = {}
        for nm, mk in ann_masks.items():
            K = int((mk & valid).sum())
            k = int((mk & q2).sum())
            if N == 0 or n_q2 == 0 or K == 0:
                fold = float("nan"); p = 1.0
            else:
                fold = (k / n_q2) / (K / N) if K else float("nan")
                p = hypergeom_p(k, K, n_q2, N)
            enrichment[nm] = {
                "n_overlap_q2": k, "n_overlap_total_valid": K,
                "n_q2": n_q2, "n_total_valid": N,
                "fold_enrichment": fold,
                "hypergeom_pval_oneSided_overrep": p,
            }
            log.info("  %-22s k=%-7d K=%-9d fold=%.3f p=%.3e", nm, k, K, fold, p)

        out = {
            "phase": "p3b3_chr1",
            "chrom": "chr1",
            "chrom_len_covered": chr1_len,
            "smoothing_bp": SMOOTH,
            "n_valid_positions": N,
            "c_threshold_top25pct_smoothed": c_thr,
            "phylop_threshold_bot25pct_smoothed": p_thr,
            "quadrant_sizes_bp": sizes,
            "quadrant_pct_of_chr1_subsample": sizes_pct,
            "q2_n_regions_min100bp": int(rg.shape[0]),
            "annotation_enrichment": enrichment,
        }

        def _safe(x):
            if isinstance(x, float) and not np.isfinite(x):
                return None
            return x

        (out_dir / "q2_enrichment_chr1.json").write_text(
            json.dumps(out, indent=2, default=_safe)
        )

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr1_enrichment",
            "status": "PASS",
            "reason": f"q2_regions={int(rg.shape[0])}",
            "metrics": {"q2_n_regions_min100bp": int(rg.shape[0])},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p3b3_chr1_enrichment DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p3b3_chr1_enrichment FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr1_enrichment",
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

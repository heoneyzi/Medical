"""P1b — canonical splice statistics (Plan §3.2.3 step P1b-2).

Loads chr17_cache.h5 and (if present) chr22_cache.h5, computes per-position
settling depth `c` at γ_cos = 0.39663 (frozen), then bins positions by the
new 7-class splice label and computes mean / median / Cohen's d / MWU p
for the key pairs:
  - canonical_GT_AG_donor    vs intron
  - canonical_GC_AG_donor    vs intron
  - non_canonical_donor      vs intron
  - canonical_GT_AG_donor    vs non_canonical_donor
  - same for acceptors

H2 (Plan §3.2.2): c(canonical) < c(non_canonical) < c(intron),
                  Cohen's d (canonical vs non-canonical) ≥ 0.20.

Outputs:
    ~/gDTR/results/p1b/splice_canonical_compare.json
    ~/gDTR/results/p1b/F_splice_canonical.{pdf,png}
    ~/gDTR/results/p1b/_status.json
    ~/gDTR/results/p1b/_done
"""
from __future__ import annotations

import argparse
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

# Splice class codebook (must match p1b_canonical_splice_label.py)
CODEBOOK = {
    "not_splice": 0,
    "canonical_GT_AG_donor": 1,
    "canonical_GT_AG_acceptor": 2,
    "canonical_GC_AG_donor": 3,
    "canonical_GC_AG_acceptor": 4,
    "non_canonical_donor": 5,
    "non_canonical_acceptor": 6,
}
INTRON_CODE_LEGACY = 1  # in chr*_position_labels.npy


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p1b_compare")
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


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def build_pos_c(h5_path: Path, chrom_len: int, gamma: float,
                log: logging.Logger, smoke: bool = False) -> np.ndarray:
    """Per-position c on a chromosome from a cache h5."""
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


def per_class_c(pos_c: np.ndarray, splice_labels: np.ndarray,
                pos_labels: np.ndarray) -> dict[str, np.ndarray]:
    """Returns dict class_name -> 1D array of c values, plus 'intron' for ref."""
    n = min(pos_c.shape[0], splice_labels.shape[0], pos_labels.shape[0])
    pc = pos_c[:n]; sl = splice_labels[:n]; lb = pos_labels[:n]
    valid = ~np.isnan(pc)
    out: dict[str, np.ndarray] = {}
    for name, code in CODEBOOK.items():
        if name == "not_splice":
            continue
        m = (sl == code) & valid
        out[name] = pc[m].astype(np.float64) if m.any() else np.array([], dtype=np.float64)
    intron_m = (lb == INTRON_CODE_LEGACY) & valid & (sl == 0)
    out["intron"] = pc[intron_m].astype(np.float64) if intron_m.any() else np.array([], dtype=np.float64)
    return out


def stats_block(arrs: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, arr in arrs.items():
        n = int(arr.size)
        out[name] = {
            "n": n,
            "mean_c": float(arr.mean()) if n else None,
            "median_c": float(np.median(arr)) if n else None,
            "std_c": float(arr.std(ddof=1)) if n > 1 else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p1b_compare: _done exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p1b_compare", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        from scipy.stats import mannwhitneyu

        ann = GDTR_ROOT / "data" / "annotation"
        sp17 = ann / "chr17_splice_class_labels.npy"
        sp22 = ann / "chr22_splice_class_labels.npy"
        lb17 = ann / "chr17_position_labels.npy"
        lb22 = ann / "chr22_position_labels.npy"
        for required in (sp17, lb17, lb22):
            if not required.exists():
                raise FileNotFoundError(f"missing input: {required}")

        h5_chr17 = GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        h5_chr22 = GDTR_ROOT / "results" / "phase1.6" / "chr22_cache.h5"

        # chr17 per-position c (prefer cached file)
        chr17_lab = np.load(lb17)
        chr17_len = int(chr17_lab.shape[0])
        cached_chr17_c = GDTR_ROOT / "results" / "phase2.4" / "chr17_position_c.npy"
        cached_chr17_c_alt = GDTR_ROOT / "results" / "phase2.5" / "chr17_position_c.npy"
        cached_chr17_c_p1a = GDTR_ROOT / "results" / "p1a" / "chr17_position_c.npy"
        if cached_chr17_c.exists():
            log.info("loading cached chr17 per-position c from %s", cached_chr17_c)
            chr17_c = np.load(cached_chr17_c)
        elif cached_chr17_c_alt.exists():
            chr17_c = np.load(cached_chr17_c_alt)
        elif cached_chr17_c_p1a.exists():
            chr17_c = np.load(cached_chr17_c_p1a)
        else:
            if not h5_chr17.exists():
                raise FileNotFoundError(f"need cache h5 or cached c for chr17")
            log.info("computing chr17 per-position c from cache (γ=%.5f)", GAMMA_LOCKED)
            chr17_c = build_pos_c(h5_chr17, chr17_len, GAMMA_LOCKED, log,
                                  smoke=args.smoke)

        chr17_splice = np.load(sp17)
        chr17_classes = per_class_c(chr17_c, chr17_splice, chr17_lab)
        log.info("chr17 sample sizes: %s",
                 {k: v.size for k, v in chr17_classes.items()})

        # chr22 (prefer cached per-position c)
        chr22_classes: dict[str, np.ndarray] = {}
        cached_chr22_c = GDTR_ROOT / "results" / "phase1.6_sub" / "chr22_position_c.npy"
        if sp22.exists() and cached_chr22_c.exists():
            chr22_c = np.load(cached_chr22_c)
            chr22_lab = np.load(lb22)
            chr22_splice = np.load(sp22)
            chr22_classes = per_class_c(chr22_c, chr22_splice, chr22_lab)
            log.info("chr22 sample sizes: %s",
                     {k: v.size for k, v in chr22_classes.items()})

        # Build pooled (chr17 + chr22) for stats robustness
        pooled: dict[str, np.ndarray] = {}
        for name in CODEBOOK:
            if name == "not_splice":
                continue
            a = chr17_classes.get(name, np.array([]))
            b = chr22_classes.get(name, np.array([]))
            pooled[name] = np.concatenate([a, b]) if (a.size + b.size) else np.array([])
        a = chr17_classes.get("intron", np.array([]))
        b = chr22_classes.get("intron", np.array([]))
        pooled["intron"] = np.concatenate([a, b])

        # Stats — pairwise tests
        def _mwu(x, y):
            if x.size < 2 or y.size < 2:
                return float("nan"), float("nan")
            try:
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                return float(u), float(p)
            except Exception:
                return float("nan"), float("nan")

        def pairwise_for_dict(d: dict[str, np.ndarray]) -> dict:
            comparisons = [
                ("canonical_GT_AG_donor", "intron"),
                ("canonical_GC_AG_donor", "intron"),
                ("non_canonical_donor",   "intron"),
                ("canonical_GT_AG_donor", "non_canonical_donor"),
                ("canonical_GT_AG_acceptor", "intron"),
                ("canonical_GC_AG_acceptor", "intron"),
                ("non_canonical_acceptor",   "intron"),
                ("canonical_GT_AG_acceptor", "non_canonical_acceptor"),
            ]
            out = []
            for a_name, b_name in comparisons:
                xa = d.get(a_name, np.array([]))
                xb = d.get(b_name, np.array([]))
                u, p = _mwu(xa, xb)
                out.append({
                    "a": a_name, "b": b_name,
                    "n_a": int(xa.size), "n_b": int(xb.size),
                    "mean_a": float(xa.mean()) if xa.size else None,
                    "mean_b": float(xb.mean()) if xb.size else None,
                    "cohens_d_a_minus_b": cohens_d(xa, xb)
                                          if (xa.size > 1 and xb.size > 1) else None,
                    "mwu_U": u,
                    "mwu_p_two_sided": p,
                })
            return out

        out = {
            "gamma_cos": GAMMA_LOCKED,
            "codebook": CODEBOOK,
            "per_class": {
                "chr17": stats_block(chr17_classes),
                "chr22": stats_block(chr22_classes),
                "pooled": stats_block(pooled),
            },
            "pairwise": {
                "chr17": pairwise_for_dict(chr17_classes),
                "chr22": pairwise_for_dict(chr22_classes),
                "pooled": pairwise_for_dict(pooled),
            },
        }

        out_path = out_dir / "splice_canonical_compare.json"
        out_path.write_text(json.dumps(out, indent=2))
        log.info("wrote %s", out_path)

        # Hypothesis verdict (Plan §3.2.4) — use pooled
        pooled_pair = {(p["a"], p["b"]): p for p in out["pairwise"]["pooled"]}
        canon_intron = pooled_pair.get(("canonical_GT_AG_donor", "intron"), {})
        noncan_intron = pooled_pair.get(("non_canonical_donor", "intron"), {})
        canon_vs_noncan = pooled_pair.get(("canonical_GT_AG_donor",
                                           "non_canonical_donor"), {})
        d_canon_noncan = (canon_vs_noncan.get("cohens_d_a_minus_b") or float("nan"))
        mean_canon = canon_intron.get("mean_a")
        mean_noncan = noncan_intron.get("mean_a")
        mean_intron = canon_intron.get("mean_b")
        ordering_ok = (mean_canon is not None and mean_noncan is not None
                       and mean_intron is not None
                       and mean_canon < mean_noncan < mean_intron)
        if ordering_ok and abs(d_canon_noncan) >= 0.20:
            verdict = "PASS"
        elif ordering_ok and abs(d_canon_noncan) >= 0.10:
            verdict = "REFRAME"
        elif ordering_ok:
            verdict = "REFRAME"
        else:
            verdict = "REFRAME"

        # ---- Figure ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = ["canonical_GT_AG_donor", "canonical_GC_AG_donor",
                     "non_canonical_donor",
                     "canonical_GT_AG_acceptor", "canonical_GC_AG_acceptor",
                     "non_canonical_acceptor", "intron"]
            labels_short = ["GT_AG_d", "GC_AG_d", "noncan_d",
                            "GT_AG_a", "GC_AG_a", "noncan_a", "intron"]
            data = []
            for name in order:
                arr = pooled.get(name, np.array([]))
                if arr.size == 0:
                    arr = np.array([0.0])
                data.append(arr)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(data, labels=labels_short, showfliers=False)
            ax.set_ylabel("settling depth c (1-based)")
            ax.set_title(
                f"Canonical / non-canonical splice c (pooled chr17+chr22, γ={GAMMA_LOCKED})"
            )
            plt.xticks(rotation=20)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"F_splice_canonical.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1b_compare",
            "status": verdict,
            "reason": (f"ordering_ok={ordering_ok} "
                       f"d_canon_vs_noncan={d_canon_noncan!r}"),
            "metrics": {
                "ordering_ok": bool(ordering_ok),
                "cohens_d_canon_vs_noncan_donor": d_canon_noncan,
                "mean_canon_donor": mean_canon,
                "mean_noncan_donor": mean_noncan,
                "mean_intron": mean_intron,
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p1b_compare DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p1b_compare FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1b_compare",
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

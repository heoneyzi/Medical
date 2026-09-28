"""P1a — calibration / validation split (Plan §3.1.3, step P1a-1).

Loads chr17_cache.h5 and chr17 / chr22 position labels, recomputes:
  - chr17 q70 of running-min D_cos at the penultimate layer (L7 for HyenaDNA;
    for Evo 2 our cache is L=N_LAYERS-2=30 — same logic as 14_phase1_4_calibration.py).
  - chr22 q70 (loaded from phase1.4/calibration.json; verified against
    phase1.6_sub/chr22_position_c.npy if available).
  - Per-context mean settling depth `c` on chr17 using frozen γ_cos = 0.39663
    (replicates the running-min logic in 25_phase2_5_splice_chr17.py:26-33).
  - Per-context mean c on chr22 from phase1.6_sub/chr22_position_c.npy.
  - Cohen's d vs intron + two-sided MWU p for each context.

Validation criteria (Plan §3.1.4):
  abs(chr17_q70 - chr22_q70) < 0.05       → PASS
                          0.05..0.15      → REFRAME
                          > 0.15          → FAIL

Outputs:
  ~/gDTR/results/p1a/calib_val_table.csv
  ~/gDTR/results/p1a/chr17_q70_recheck.json
  ~/gDTR/results/p1a/_status.json
  ~/gDTR/results/p1a/_done
"""
from __future__ import annotations

import argparse
import csv
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
N_LAYERS = 32
PENULT_LAYER = N_LAYERS - 2  # = 30 for Evo 2

CTX = {
    0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
    5: "splice_donor", 6: "splice_acceptor", 7: "repeat",
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p1a")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def settling_depth(D_cos: np.ndarray, gamma: float) -> np.ndarray:
    """D_cos [L, T] → c [T]. 1-based. Same as 25_phase2_5_splice_chr17.py:26-33."""
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


def compute_chr17_q70(h5_path: Path, log: logging.Logger,
                      smoke: bool = False) -> tuple[float, bool]:
    """Read chr17_cache.h5, compute q70 of running-min D_cos at penult layer.

    Returns (q70, all_done).
    """
    import h5py
    with h5py.File(h5_path, "r") as h5:
        N = h5["D_cos"].shape[0]
        done_mask = h5["done_mask"][:].astype(bool)
        all_done = bool(done_mask.all())
        n_done = int(done_mask.sum())
        log.info("chr17_cache.h5: %d/%d windows done", n_done, N)
        if smoke:
            target = min(50, n_done)
            sel = np.flatnonzero(done_mask)[:target]
        else:
            sel = np.flatnonzero(done_mask)
        # Stream running-min @ penult layer per window
        rmin_vals = []
        for i in sel:
            D = h5["D_cos"][int(i)].astype(np.float32)  # [L, T]
            rmin = np.minimum.accumulate(D, axis=0)  # [L, T]
            rmin_vals.append(rmin[PENULT_LAYER])
            if len(rmin_vals) % 1000 == 0:
                log.info("  scanned %d windows", len(rmin_vals))
        if not rmin_vals:
            raise RuntimeError("no done windows; cannot compute q70")
        flat = np.concatenate(rmin_vals)
        q70 = float(np.quantile(flat, 0.70))
        log.info("chr17 q70 of running-min D_cos[L=%d] = %.5f", PENULT_LAYER, q70)
    return q70, all_done


def per_context_c(pos_c: np.ndarray, labels: np.ndarray,
                  log: logging.Logger) -> dict[str, np.ndarray]:
    """Bin per-position c by context label."""
    out: dict[str, np.ndarray] = {}
    n = min(pos_c.shape[0], labels.shape[0])
    pc = pos_c[:n]
    lb = labels[:n]
    valid = ~np.isnan(pc)
    for code, name in CTX.items():
        m = (lb == code) & valid
        if m.any():
            out[name] = pc[m].astype(np.float64)
        else:
            out[name] = np.array([], dtype=np.float64)
    return out


def build_chr17_pos_c(h5_path: Path, chrom_len: int, gamma: float,
                      log: logging.Logger, smoke: bool = False) -> np.ndarray:
    """Per-position c on chr17 (NaN where uncovered)."""
    import h5py
    sum_c = np.zeros(chrom_len, dtype=np.float64)
    cov = np.zeros(chrom_len, dtype=np.uint16)
    with h5py.File(h5_path, "r") as h5:
        N = h5["D_cos"].shape[0]
        done = h5["done_mask"][:].astype(bool)
        starts = h5["starts"][:]
        ends = h5["ends"][:]
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
                log.info("  built per-pos c for %d windows", k + 1)
    out = np.full(chrom_len, np.nan, dtype=np.float32)
    nz = cov > 0
    out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run on first 50 windows for fast end-to-end check")
    parser.add_argument("--force", action="store_true",
                        help="override _done marker")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p1a"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p1a: _done exists; skipping (use --force to re-run)")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {
        "step": "p1a",
        "status": "RUNNING",
        "started_at": started_at,
        "smoke": args.smoke,
    }
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        from scipy.stats import mannwhitneyu

        # ---- Inputs ----
        h5_path = GDTR_ROOT / "results" / "phase2.1" / "chr17_cache.h5"
        chr17_lab_path = GDTR_ROOT / "data" / "annotation" / "chr17_position_labels.npy"
        chr22_lab_path = GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"
        chr22_c_path = GDTR_ROOT / "results" / "phase1.6_sub" / "chr22_position_c.npy"
        cal_path = GDTR_ROOT / "results" / "phase1.4" / "calibration.json"
        cached_chr17_c = GDTR_ROOT / "results" / "phase2.4" / "chr17_position_c.npy"
        cached_chr17_c_alt = GDTR_ROOT / "results" / "phase2.5" / "chr17_position_c.npy"

        for required in (h5_path, chr17_lab_path, chr22_lab_path, cal_path):
            if not required.exists():
                raise FileNotFoundError(f"missing input: {required}")

        cal = json.loads(cal_path.read_text())
        chr22_q70 = float(cal.get("gamma_cos_global_q70", GAMMA_LOCKED))
        log.info("chr22 q70 (frozen γ_cos) = %.5f", chr22_q70)

        # ---- chr17 q70 recheck ----
        chr17_q70, all_done = compute_chr17_q70(h5_path, log, smoke=args.smoke)
        abs_diff = abs(chr17_q70 - chr22_q70)
        if abs_diff < 0.05:
            q70_class = "PASS"
        elif abs_diff < 0.15:
            q70_class = "REFRAME"
        else:
            q70_class = "FAIL"
        log.info("|chr17_q70 - chr22_q70| = %.4f (%s)", abs_diff, q70_class)

        recheck = {
            "chr17_q70": chr17_q70,
            "chr22_q70": chr22_q70,
            "abs_diff": abs_diff,
            "pass": q70_class,
            "all_chr17_windows_done": all_done,
            "gamma_used": chr22_q70,
        }
        (out_dir / "chr17_q70_recheck.json").write_text(json.dumps(recheck, indent=2))

        # ---- chr17 per-position c (cached or compute) ----
        chr17_labels = np.load(chr17_lab_path)
        chr17_len = int(chr17_labels.shape[0])
        if cached_chr17_c.exists():
            log.info("loading cached chr17 per-position c from %s", cached_chr17_c)
            chr17_c = np.load(cached_chr17_c)
        elif cached_chr17_c_alt.exists():
            log.info("loading cached chr17 per-position c from %s", cached_chr17_c_alt)
            chr17_c = np.load(cached_chr17_c_alt)
        else:
            log.info("computing chr17 per-position c from cache (γ=%.5f) ...", chr22_q70)
            chr17_c = build_chr17_pos_c(h5_path, chr17_len, chr22_q70, log,
                                         smoke=args.smoke)
            np.save(out_dir / "chr17_position_c.npy", chr17_c)
        ctx_chr17 = per_context_c(chr17_c, chr17_labels, log)

        # ---- chr22 per-position c ----
        chr22_labels = np.load(chr22_lab_path)
        if not chr22_c_path.exists():
            raise FileNotFoundError(
                f"chr22 per-position c not found at {chr22_c_path} — required for "
                "Calibration table baseline"
            )
        chr22_c = np.load(chr22_c_path)
        ctx_chr22 = per_context_c(chr22_c, chr22_labels, log)

        # ---- Build the calib/val table ----
        rows = []
        # Build all 7 (non-repeat) contexts × 2 chromosomes; intron is reference.
        contexts = ["intergenic", "intron", "coding_exon", "5utr", "3utr",
                    "splice_donor", "splice_acceptor"]
        for chrom_name, ctx_dict in (("chr22", ctx_chr22), ("chr17", ctx_chr17)):
            intron_arr = ctx_dict.get("intron", np.array([]))
            for c in contexts:
                arr = ctx_dict.get(c, np.array([]))
                n = int(arr.size)
                mean_c = float(arr.mean()) if n else float("nan")
                med_c = float(np.median(arr)) if n else float("nan")
                if c == "intron":
                    d_vs_intron = 0.0
                    p_mwu = 1.0
                else:
                    if n >= 2 and intron_arr.size >= 2:
                        d_vs_intron = cohens_d(arr, intron_arr)
                        try:
                            _u, p_mwu = mannwhitneyu(arr, intron_arr,
                                                     alternative="two-sided")
                            p_mwu = float(p_mwu)
                        except Exception:
                            p_mwu = float("nan")
                    else:
                        d_vs_intron = float("nan")
                        p_mwu = float("nan")
                rows.append({
                    "chrom": chrom_name,
                    "context": c,
                    "n": n,
                    "mean_c": mean_c,
                    "median_c": med_c,
                    "cohens_d_vs_intron": d_vs_intron,
                    "mwu_p_vs_intron": p_mwu,
                })

        # Validation criteria (Plan §3.1.4) on chr17 donor / exon vs intron
        def find_row(chrom: str, ctx: str):
            for r in rows:
                if r["chrom"] == chrom and r["context"] == ctx:
                    return r
            return None

        chr22_donor = find_row("chr22", "splice_donor") or {}
        chr17_donor = find_row("chr17", "splice_donor") or {}
        chr17_exon = find_row("chr17", "coding_exon") or {}

        d22 = chr22_donor.get("cohens_d_vs_intron", float("nan"))
        d17 = chr17_donor.get("cohens_d_vs_intron", float("nan"))
        chr17_exon_p = chr17_exon.get("mwu_p_vs_intron", float("nan"))
        d_ratio = (abs(d17) / abs(d22)) if (np.isfinite(d22) and abs(d22) > 1e-9
                                            and np.isfinite(d17)) else float("nan")

        if np.isfinite(d_ratio):
            if d_ratio >= 0.80:
                d_class = "PASS"
            elif d_ratio >= 0.50:
                d_class = "REFRAME"
            else:
                d_class = "FAIL"
        else:
            d_class = "REFRAME"

        if np.isfinite(chr17_exon_p):
            if chr17_exon_p < 1.7e-51:
                p_class = "PASS"
            elif chr17_exon_p < 1e-30:
                p_class = "REFRAME"
            else:
                p_class = "FAIL"
        else:
            p_class = "REFRAME"

        # Aggregate verdict
        order = {"PASS": 0, "REFRAME": 1, "FAIL": 2}
        worst = max([q70_class, d_class, p_class], key=lambda x: order[x])

        # Write the CSV
        csv_path = out_dir / "calib_val_table.csv"
        cols = ["chrom", "context", "n", "mean_c", "median_c",
                "cohens_d_vs_intron", "mwu_p_vs_intron"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
        log.info("wrote %s (%d rows)", csv_path, len(rows))

        # Final status JSON
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1a",
            "status": worst,
            "reason": (f"q70_class={q70_class} d_class={d_class} p_class={p_class}; "
                       f"chr17_q70={chr17_q70:.5f} abs_diff={abs_diff:.4f} "
                       f"d_ratio={d_ratio:.3f} chr17_exon_p={chr17_exon_p:.2e}"),
            "metrics": {
                "chr17_q70": chr17_q70,
                "chr22_q70": chr22_q70,
                "abs_diff": abs_diff,
                "chr22_donor_d_vs_intron": d22,
                "chr17_donor_d_vs_intron": d17,
                "d_ratio": d_ratio,
                "chr17_exon_p_vs_intron": chr17_exon_p,
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
            "smoke": args.smoke,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p1a DONE — verdict=%s wall=%.1fs", worst, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p1a FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p1a",
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

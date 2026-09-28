"""P3B-1 — functional positive control (Plan §3.4.B-1.2).

For each functional dataset (cCRE-ELS, GTEx eQTL chr22, GWAS Catalog chr22)
compute:
  - mean c at functional positions (`c_func`)
  - mean c at matched-control positions: 100 random shuffles of the
    functional mask within chr22, matched by per-shuffle bp mass, with
    `bedtools shuffle -chrom -noOverlapping`
  - one-sided MWU p of c_func > c_shuffled
  - effect size = (mean(c_func) - mean(c_shuffled)) / std(c_shuffled)

CPU-only. Plan: HARD GATE for the Q2 path. PASS = all three datasets show
c_func > c_shuffled with MWU p < 1e-10 and effect ≥ 0.30.

Outputs:
    ~/gDTR/results/p3b1/p3b1_func_pos.json
    ~/gDTR/results/p3b1/F_func_pos.{pdf,png}
    ~/gDTR/results/p3b1/_status.json
    ~/gDTR/results/p3b1/_done
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
CHR = "chr22"
CHR_LEN = 50_818_468


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p3b1")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def load_bed_mask(bed_path: Path, chrom: str, chrom_len: int) -> np.ndarray:
    m = np.zeros(chrom_len, dtype=bool)
    if not bed_path.exists():
        return m
    with bed_path.open() as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            s = max(0, s); e = min(chrom_len, e)
            if e > s:
                m[s:e] = True
    return m


def shuffle_mask_python(mask: np.ndarray, chrom_len: int, rng: np.random.Generator,
                       n_iters: int = 5) -> np.ndarray:
    """Python fallback shuffle if bedtools is unavailable.

    Splits mask into contiguous regions, then re-places each region at a
    random non-overlapping position. Approximate (not 100% identical to
    bedtools' algorithm) but adequate for null distribution.
    """
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    n = len(starts)
    if n == 0:
        return np.zeros(chrom_len, dtype=bool)
    out = np.zeros(chrom_len, dtype=bool)
    for L in lengths:
        # try a few times to find a non-overlapping placement
        for _ in range(n_iters):
            new_s = int(rng.integers(0, max(1, chrom_len - L)))
            if not out[new_s:new_s + L].any():
                out[new_s:new_s + L] = True
                break
    return out


def shuffle_with_bedtools(bed_path: Path, genome_file: Path, seed: int) -> np.ndarray:
    """Use bedtools shuffle to get a shuffled BED, return mask."""
    cmd = (f"bedtools shuffle -i {bed_path} -g {genome_file} "
           f"-seed {seed} -chrom -noOverlapping")
    try:
        r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        cmd = (f"bedtools shuffle -i {bed_path} -g {genome_file} "
               f"-seed {seed} -chrom")
        r = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    m = np.zeros(CHR_LEN, dtype=bool)
    for line in r.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3 or parts[0] != CHR:
            continue
        try:
            s = int(parts[1]); e = int(parts[2])
        except ValueError:
            continue
        s = max(0, s); e = min(CHR_LEN, e)
        if e > s:
            m[s:e] = True
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run with 10 shuffles instead of 100")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-shuffles", type=int, default=100)
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p3b1"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done"
    if done_marker.exists() and not args.force:
        log.info("p3b1: _done exists; skipping")
        return

    n_shuf = 10 if args.smoke else args.n_shuffles

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p3b1", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        from scipy.stats import mannwhitneyu

        c_path = GDTR_ROOT / "results" / "phase1.6_sub" / "chr22_position_c.npy"
        if not c_path.exists():
            raise FileNotFoundError(f"missing {c_path}")
        c_arr = np.load(c_path)
        if c_arr.shape[0] != CHR_LEN:
            log.warning("c array length %d != CHR_LEN %d (proceeding)",
                        c_arr.shape[0], CHR_LEN)
        valid_c = ~np.isnan(c_arr)

        ext = GDTR_ROOT / "data" / "external"
        datasets = {
            "cCRE_ELS": ext / "ccre_els_only_chr22.bed",
            "GTEx_eQTL": ext / "gtex_eqtl_chr22.bed",
            "GWAS": ext / "gwas_chr22.bed",
        }
        # genome_file for bedtools shuffle
        genome_file = out_dir / "chr22.genome"
        genome_file.write_text(f"{CHR}\t{CHR_LEN}\n")

        bedtools_ok = shutil.which("bedtools") is not None
        if not bedtools_ok:
            log.warning("bedtools NOT FOUND on PATH — falling back to Python shuffle")

        per_dataset = {}
        rng = np.random.default_rng(42)
        for name, bed in datasets.items():
            if not bed.exists():
                log.warning("missing dataset %s at %s — skipping", name, bed)
                per_dataset[name] = {"available": False}
                continue
            log.info("=== %s ===", name)
            mask = load_bed_mask(bed, CHR, CHR_LEN)
            in_mask = mask & valid_c
            n_in = int(in_mask.sum())
            if n_in == 0:
                per_dataset[name] = {"available": True, "n_func_positions": 0,
                                      "skipped": "no overlap with valid c"}
                continue
            c_func = c_arr[in_mask].astype(np.float64)

            null_means = []
            null_pool = []
            for s in range(n_shuf):
                if bedtools_ok:
                    sh_mask = shuffle_with_bedtools(bed, genome_file, seed=42 + s)
                else:
                    sh_mask = shuffle_mask_python(mask, CHR_LEN, rng)
                sh_in = sh_mask & valid_c
                if sh_in.any():
                    vals = c_arr[sh_in].astype(np.float64)
                    null_means.append(float(vals.mean()))
                    null_pool.append(vals)

            null_means = np.asarray(null_means, dtype=np.float64)
            if null_pool:
                null_concat = np.concatenate(null_pool)
            else:
                null_concat = np.array([])
            mean_c_func = float(c_func.mean())
            mean_null = float(null_means.mean()) if null_means.size else float("nan")
            std_null = float(null_means.std(ddof=1)) if null_means.size > 1 else float("nan")
            effect = ((mean_c_func - mean_null) / std_null
                      if std_null and std_null > 0 else float("nan"))
            try:
                if null_concat.size:
                    _u, p = mannwhitneyu(c_func, null_concat, alternative="greater")
                    p = float(p)
                else:
                    p = float("nan")
            except Exception:
                p = float("nan")

            per_dataset[name] = {
                "available": True,
                "n_func_positions": int(n_in),
                "mean_c_func": mean_c_func,
                "mean_c_shuffled": mean_null,
                "std_c_shuffled": std_null,
                "effect": effect,
                "mwu_p_one_sided_greater": p,
                "n_shuffles": n_shuf,
                "null_means_first10": null_means[:10].tolist(),
            }
            log.info("  n_func=%d mean_c_func=%.3f mean_c_shuffled=%.3f effect=%.3f p=%.2e",
                     n_in, mean_c_func, mean_null, effect, p)

        # Verdict (Plan §3.4.1.B-1.3)
        n_pass = 0
        for v in per_dataset.values():
            if not v.get("available"):
                continue
            p = v.get("mwu_p_one_sided_greater", float("nan"))
            eff = v.get("effect", float("nan"))
            if (np.isfinite(p) and p < 1e-10
                    and np.isfinite(eff) and eff >= 0.30):
                n_pass += 1
        if n_pass == 3:
            verdict = "PASS"
        elif n_pass == 2:
            verdict = "REFRAME"
        else:
            verdict = "FAIL"

        out = {
            "config": {
                "chrom": CHR, "chrom_len": CHR_LEN, "n_shuffles": n_shuf,
                "shuffler": "bedtools" if bedtools_ok else "python",
            },
            "per_dataset": per_dataset,
            "verdict": verdict,
        }
        (out_dir / "p3b1_func_pos.json").write_text(json.dumps(out, indent=2))

        # Figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = [k for k, v in per_dataset.items() if v.get("available")]
            n = max(1, len(names))
            fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, name in zip(axes, names):
                v = per_dataset[name]
                ax.bar(["functional", "shuffled"],
                       [v["mean_c_func"], v["mean_c_shuffled"]],
                       color=["#d1495b", "#7f7f7f"])
                ax.set_title(f"{name}\nn_func={v['n_func_positions']}\n"
                             f"effect={v['effect']:.2f}, p={v['mwu_p_one_sided_greater']:.1e}")
            axes[0].set_ylabel("mean settling depth c")
            fig.suptitle("Q2 functional positive control")
            fig.tight_layout()
            for ext_ in ("pdf", "png"):
                fig.savefig(out_dir / f"F_func_pos.{ext_}", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("figure failed: %s", e)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b1",
            "status": verdict,
            "reason": f"{n_pass}/3 datasets pass strong threshold",
            "metrics": {"n_pass": n_pass},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p3b1 DONE verdict=%s wall=%.1fs", verdict, time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p3b1 FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b1",
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

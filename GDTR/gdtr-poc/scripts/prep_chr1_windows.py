"""prep_chr1_windows.py — chr1 sub-sample windows (Plan §3.4.B-3.2 chr1 optional).

Downloads chr1.fa.gz from UCSC if missing, generates 8000 random
non-overlapping 6 kb windows (covering ~48 Mb of chr1, sufficient for an
enrichment-level test).

Outputs:
    ~/gDTR/data/reference/chr1.fa
    ~/gDTR/data/reference/chr1_windows.tsv
    ~/gDTR/data/annotation/chr1_position_labels.npy   (optional;
                                                       unlabelled = intergenic)
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
WINDOW = 6000
N_FRAC_MAX = 0.01
TARGET_N_WINDOWS = 8000
SEED = 42

UCSC_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr1.fa.gz"


def setup_logging() -> logging.Logger:
    log = logging.getLogger("prep_chr1")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="cap windows to 200 for fast smoke run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-windows", type=int, default=TARGET_N_WINDOWS)
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "data" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = GDTR_ROOT / "data" / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_chr1_windows"
    if done_marker.exists() and not args.force:
        log.info("prep_chr1: _done_chr1_windows exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    target_n = 200 if args.smoke else args.target_windows
    status = {"step": "prep_chr1", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke,
              "target_n_windows": target_n}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        fa_gz = out_dir / "chr1.fa.gz"
        fa = out_dir / "chr1.fa"

        # Download if missing
        if not fa.exists():
            if not fa_gz.exists():
                log.info("downloading %s -> %s", UCSC_URL, fa_gz)
                with urllib.request.urlopen(UCSC_URL, timeout=600) as r, fa_gz.open("wb") as fo:
                    shutil.copyfileobj(r, fo)
                log.info("downloaded %.1f MB", fa_gz.stat().st_size / 1e6)
            log.info("decompressing %s", fa_gz)
            with gzip.open(fa_gz, "rb") as fi, fa.open("wb") as fo:
                shutil.copyfileobj(fi, fo)
            log.info("wrote %s (%.1f MB)", fa, fa.stat().st_size / 1e6)

        # samtools faidx for pysam
        try:
            subprocess.run(["samtools", "faidx", str(fa)], check=False)
        except FileNotFoundError:
            log.warning("samtools not found; pyfaidx will index on demand")

        # Build a chr1 sequence numpy view via pyfaidx (lazy) for window N-fraction
        from pyfaidx import Fasta
        fa_obj = Fasta(str(fa), as_raw=True, sequence_always_upper=True)
        seq_str = str(fa_obj["chr1"][:])
        chrom_len = len(seq_str)
        log.info("chr1 length=%d", chrom_len)

        seq_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        N_BYTE = ord("N")
        cum_n = np.concatenate(([0], np.cumsum((seq_arr == N_BYTE).astype(np.int64))))

        rng = np.random.default_rng(SEED)
        max_start = chrom_len - WINDOW
        windows = []
        # Deterministic shuffle of candidate starts to draw non-overlapping
        # 6 kb windows. We use a simple mark-and-skip approach.
        used = np.zeros(chrom_len, dtype=bool)
        attempts = 0
        max_attempts = target_n * 50
        while len(windows) < target_n and attempts < max_attempts:
            attempts += 1
            start = int(rng.integers(0, max_start))
            end = start + WINDOW
            if used[start:end].any():
                continue
            n_count = int(cum_n[end] - cum_n[start])
            if n_count / WINDOW > N_FRAC_MAX:
                continue
            used[start:end] = True
            windows.append((start, end, n_count))
            if len(windows) % 1000 == 0:
                log.info("  collected %d windows (attempts=%d)", len(windows), attempts)

        log.info("collected %d windows after %d attempts", len(windows), attempts)
        windows.sort()

        # Write TSV
        tsv = out_dir / "chr1_windows.tsv"
        cols = ["window_idx", "chrom", "start", "end", "n_count"]
        with tsv.open("w") as f:
            f.write("\t".join(cols) + "\n")
            for i, (s, e, nc) in enumerate(windows):
                f.write(f"{i}\tchr1\t{s}\t{e}\t{nc}\n")
        log.info("wrote %s (%d rows)", tsv, len(windows))

        # Write a position-labels.npy seeded with intergenic everywhere
        # (chr1 has no GENCODE chr17/chr22-DB-equivalent locally; we use a
        # placeholder so downstream Q2 enrichment scripts can stream).
        labels = np.zeros(chrom_len, dtype=np.uint8)
        np.save(ann_dir / "chr1_position_labels.npy", labels)
        log.info("wrote %s", ann_dir / "chr1_position_labels.npy")

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "prep_chr1",
            "status": "PASS",
            "reason": f"prepared {len(windows)} chr1 windows",
            "metrics": {"n_windows": len(windows),
                        "chrom_len": chrom_len,
                        "fa_path": str(fa)},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True, "n_windows": len(windows)},
                                          indent=2))
        log.info("prep_chr1 DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("prep_chr1 FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "prep_chr1",
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

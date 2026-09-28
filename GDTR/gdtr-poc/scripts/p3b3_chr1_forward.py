"""P3B-3 chr1 (optional GPU PRIORITY 2) — forward 8000 chr1 windows on Evo 2 7B.

Modeled on 21_phase2_1_chr17_forward.py — same forward logic but reads
chr1_windows.tsv produced by prep_chr1_windows.py.

Hard wall stop: 12 hours (per Plan §10.1 row 14). The runner enforces the
wall stop; this script runs as long as it has work.

Outputs:
    ~/gDTR/results/p3b3_chr1/chr1_cache.h5
    ~/gDTR/results/p3b3_chr1/chr1_position_c.npy
    ~/gDTR/results/p3b3_chr1/_status.json
    ~/gDTR/results/p3b3_chr1/_done_forward
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
SEED = 42
GAMMA_LOCKED = 0.39663


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p3b3_chr1_forward")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def read_windows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            parts = line.rstrip().split("\t")
            d = dict(zip(header, parts))
            rows.append(d)
    return rows


def settling_depth(D_cos: np.ndarray, gamma: float) -> np.ndarray:
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="forward only the first 50 windows")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p3b3_chr1"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_forward"
    if done_marker.exists() and not args.force:
        log.info("p3b3_chr1_forward: _done_forward exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p3b3_chr1_forward", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        sys.path.insert(0, str(GDTR_ROOT))
        sys.path.insert(0, str(GDTR_ROOT / "scripts"))
        try:
            import _runner_utils as ru  # noqa: F401
            ru.add_repo_paths()
            ru.patch_safe_globals()
        except Exception:
            pass

        import torch
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import N_LAYERS
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        import pysam
        fa_path = GDTR_ROOT / "data" / "reference" / "chr1.fa"
        if not fa_path.exists():
            raise FileNotFoundError(f"missing {fa_path} (run prep_chr1_windows.py)")
        fasta = pysam.FastaFile(str(fa_path))

        win_path = GDTR_ROOT / "data" / "reference" / "chr1_windows.tsv"
        if not win_path.exists():
            raise FileNotFoundError(f"missing {win_path} (run prep_chr1_windows.py)")
        windows = read_windows(win_path)
        if args.smoke:
            windows = windows[:50]
        N = len(windows)
        log.info("loaded %d chr1 windows from %s", N, win_path)
        T = int(windows[0]["end"]) - int(windows[0]["start"])
        log.info("window width T=%d", T)

        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py required")
        h5_path = out_dir / "chr1_cache.h5"

        ckpt_path = out_dir / "_checkpoint.json"
        start_i = 0
        if h5_path.exists() and ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            start_i = int(ckpt.get("next_window_idx", 0))
            log.info("resuming from window %d", start_i)
        else:
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("D_jsd", shape=(N, N_LAYERS, T), dtype="float16",
                                  compression="gzip", chunks=(1, N_LAYERS, T))
                h5.create_dataset("D_cos", shape=(N, N_LAYERS, T), dtype="float16",
                                  compression="gzip", chunks=(1, N_LAYERS, T))
                h5.create_dataset("starts", shape=(N,), dtype="int64")
                h5.create_dataset("ends", shape=(N,), dtype="int64")
                h5.create_dataset("done_mask", shape=(N,), dtype="uint8")
        bundle = load_evo2()
        log.info("model loaded: %s", bundle.loaded_variant)
        layer_names = all_layer_names()

        save_every = 500
        with h5py.File(h5_path, "a") as h5:
            for i in range(start_i, N):
                w = windows[i]
                chrom = w["chrom"]
                s = int(w["start"]); e = int(w["end"])
                seq = fasta.fetch(chrom, s, e).upper()
                if len(seq) != T:
                    log.warning("window %d width %d != %d (skip)", i, len(seq), T)
                    h5["done_mask"][i] = 0
                    continue
                input_ids = tokenize(seq, bundle, device="cuda")
                hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)
                d_jsd = jsd_lens(hs, bundle, n_layers=N_LAYERS).numpy().astype(np.float16)
                d_cos = cosine_lens(hs, n_layers=N_LAYERS).numpy().astype(np.float16)
                h5["D_jsd"][i] = d_jsd
                h5["D_cos"][i] = d_cos
                h5["starts"][i] = s
                h5["ends"][i] = e
                h5["done_mask"][i] = 1
                del hs
                torch.cuda.empty_cache()
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1 - start_i) / max(1.0, elapsed)
                    eta = (N - i - 1) / max(1e-6, rate)
                    log.info("window %d/%d  rate=%.2f/s  ETA=%.1f min",
                             i + 1, N, rate, eta / 60)
                if (i + 1) % save_every == 0:
                    h5.flush()
                    ckpt_path.write_text(json.dumps({"next_window_idx": i + 1}))

        ckpt_path.write_text(json.dumps({"next_window_idx": N}))

        # Optional: per-position c (small chr1 sub-sample)
        chr1_len_for_c = max(int(w["end"]) for w in windows) + T
        try:
            import h5py
            sum_c = np.zeros(chr1_len_for_c, dtype=np.float64)
            cov = np.zeros(chr1_len_for_c, dtype=np.uint16)
            with h5py.File(h5_path, "r") as h5:
                done = h5["done_mask"][:].astype(bool)
                starts = h5["starts"][:]; ends = h5["ends"][:]
                for i in np.flatnonzero(done):
                    i = int(i)
                    s = int(starts[i]); e = int(ends[i])
                    D = h5["D_cos"][i].astype(np.float32)
                    c = settling_depth(D, GAMMA_LOCKED)
                    seg_end = min(e, chr1_len_for_c)
                    seg_len = seg_end - s
                    sum_c[s:seg_end] += c[:seg_len]
                    cov[s:seg_end] += 1
            out = np.full(chr1_len_for_c, np.nan, dtype=np.float32)
            nz = cov > 0
            out[nz] = (sum_c[nz] / cov[nz]).astype(np.float32)
            np.save(out_dir / "chr1_position_c.npy", out)
            log.info("wrote chr1_position_c.npy (windows-covered)")
        except Exception as e:
            log.warning("per-position c build failed: %s", e)

        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr1_forward",
            "status": "PASS",
            "reason": f"forwarded {N} chr1 windows",
            "metrics": {"n_windows": N},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        done_marker.write_text(json.dumps({"ok": True}, indent=2))
        log.info("p3b3_chr1_forward DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p3b3_chr1_forward FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p3b3_chr1_forward",
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

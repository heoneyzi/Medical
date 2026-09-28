"""Phase 2.1 — chr17 forward over all chr17_windows.

Adapted from scripts/16_phase1_6_chr22_forward.py. SAME forward logic
(extract_hidden_states + jsd_lens + cosine_lens). Save to
/path/to/gDTR/results/phase2.1/chr17_cache.h5.

γ_cos = 0.39663 (locked from Phase 1.4); cosine_lens uses final_key="norm" (FIXED).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase2.1"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase2.1"
LOG = ru.setup_logging(PHASE)


def read_windows(path: Path):
    rows = []
    with path.open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            parts = line.rstrip().split("\t")
            d = dict(zip(header, parts))
            rows.append(d)
    return rows


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="chr17_forward"):
        seed = 42
        torch.manual_seed(seed); np.random.seed(seed)

        from src.constants_evo2 import N_LAYERS
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        try:
            import pysam
        except Exception as e:
            raise RuntimeError(f"pysam required: {e}")
        fasta = pysam.FastaFile(str(ru.GDTR_ROOT / "data" / "reference" / "chr17.fa"))

        win_path = ru.GDTR_ROOT / "data" / "baselines" / "chr17_windows.tsv"
        windows = read_windows(win_path)
        N = len(windows)
        LOG.info("loaded %d chr17 windows from %s", N, win_path)

        T = int(windows[0]["end"]) - int(windows[0]["start"])
        if T <= 0:
            raise RuntimeError("zero-width window detected")
        LOG.info("window width T=%d", T)

        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py required")
        h5_path = PHASE_OUT_DIR / "chr17_cache.h5"

        ckpt_path = PHASE_OUT_DIR / "_checkpoint.json"
        start_i = 0
        if h5_path.exists() and ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            start_i = int(ckpt.get("next_window_idx", 0))
            LOG.info("resuming from window %d", start_i)
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
        LOG.info("model loaded: %s", bundle.loaded_variant)

        layer_names = all_layer_names()
        t0 = time.time()
        save_every = 1000

        with h5py.File(h5_path, "a") as h5:
            for i in range(start_i, N):
                w = windows[i]
                chrom = w["chrom"]
                s = int(w["start"]); e = int(w["end"])
                seq = fasta.fetch(chrom, s, e).upper()
                if len(seq) != T:
                    LOG.warning("window %d width %d != %d (skip)", i, len(seq), T)
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
                    LOG.info("window %d/%d  rate=%.2f/s  ETA=%.1f min", i + 1, N, rate, eta / 60)
                if (i + 1) % save_every == 0:
                    h5.flush()
                    ckpt_path.write_text(json.dumps({"next_window_idx": i + 1}))
                    LOG.info("checkpoint at window %d", i + 1)

        ckpt_path.write_text(json.dumps({"next_window_idx": N}))
        (PHASE_OUT_DIR / "_chr17_cache_done").write_text("ok\n")
        ru.write_done(PHASE, PHASE_OUT_DIR,
                      {"n_windows": N, "h5_path": str(h5_path)},
                      step_name="chr17_forward")
        LOG.info("Phase 2.1 chr17 forward complete: %d windows", N)


if __name__ == "__main__":
    main()

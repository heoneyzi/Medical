"""Cross-architecture forward (HyenaDNA + NT-v2 + DNABERT) on chr22.

Usually invoked via `15_chr22_forward.py --with-crossarch` for concurrent
execution. This standalone script is for re-runs only.

SERVER script. ~10 min standalone, or 0 min when run concurrently with 15.

Outputs:
    /path/to/TDiG/data/cache/crossarch/
        hyenadna_tier1.parquet     (8 layers)
        nt_v2_tier1.parquet        (29 layers)
        dnabert2_tier1.parquet     (12 layers)
        per_model_summary.json     {gamma_q70 per model, splice signal per model, ...}
        _done

Only Tier 1 settling tables are saved for the 3 non-Evo-2 models. Tier 2/3
storage budget reserved for chr22/chr17.

Memory profile (all three concurrent on H200):
    HyenaDNA-large 28M:  ~1 GB
    NT-v2 500M:           ~2 GB
    DNABERT-2 117M:       ~0.5 GB
    Total weights:        ~3.5 GB (negligible)
    Activations per 6kb:  ~3 GB combined
"""

from __future__ import annotations

import argparse
from pathlib import Path


def forward_hyenadna(windows, out_path):
    raise NotImplementedError


def forward_nt_v2(windows, out_path):
    raise NotImplementedError


def forward_dnabert2(windows, out_path):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/path/to/TDiG/data/cache/crossarch"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Sequential per-model (since each model is small enough to do quickly)
    # Or: triple-concurrent via separate CUDA streams (saves a few min)
    raise NotImplementedError("Cross-architecture forward")


if __name__ == "__main__":
    main()

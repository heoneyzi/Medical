"""PHASE C (part 1) — chr17 batched forward.

Thin wrapper around 15_chr22_forward's compute machinery, with chr17-specific
input/output paths. Uses same 14-cell settling + Tier 2 (subset) + Tier 3
(subset raw h_ell) pipeline.

chr17 has 27,586 windows (vs chr22 12,978). At ~2.2 win/s, full chr17 wall
~3.5 hours. Output paths under /path/to/TDiG/data/cache/chr17/.

Sub-selection: 100 chr17 subset windows (selected by 05_select_subset.py
with seed=43 — different from chr22's seed=42 — to avoid forcing the same
position-class patterns).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import everything from 15
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chr22_forward_module", Path(__file__).resolve().parent / "15_chr22_forward.py"
)
chr22_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chr22_mod)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-tsv", type=Path,
                        default=Path("/path/to/gDTR/data/baselines/chr17_windows.tsv"))
    parser.add_argument("--fasta", type=Path,
                        default=Path("/path/to/gDTR/data/reference/chr17.fa"))
    parser.add_argument("--subset-file", type=Path,
                        default=Path("/path/to/TDiG/data/subset_window_ids.json"))
    parser.add_argument("--gamma-file", type=Path,
                        default=Path("/path/to/TDiG/data/cache/population_stats/gamma_calibration_v2.json"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/chr17"))
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--tier3-tokens-per-window", type=int, default=600)
    parser.add_argument("--chr-key", type=str, default="chr17",
                        help="Key in subset_window_ids.json to use")
    args = parser.parse_args()

    # Delegate to the chr22 module's main logic but with chr17 paths.
    # Easiest: patch sys.argv to call chr22 main with our defaults.
    import sys as _sys
    _sys.argv = [
        "16_chr17_forward.py",
        "--windows-tsv", str(args.windows_tsv),
        "--fasta", str(args.fasta),
        "--subset-file", str(args.subset_file),
        "--gamma-file", str(args.gamma_file),
        "--out-dir", str(args.out_dir),
        "--max-windows", str(args.max_windows),
        "--log-every", str(args.log_every),
        "--save-every", str(args.save_every),
        "--tier3-tokens-per-window", str(args.tier3_tokens_per_window),
    ]

    # Monkey-patch the subset key reading
    original_main = chr22_mod.main
    # Patch the subset_ids loading to use args.chr_key
    import json as _json
    _orig_loads = _json.loads
    target_key = args.chr_key
    def patched_loads(s, *a, **k):
        obj = _orig_loads(s, *a, **k)
        if isinstance(obj, dict) and "chr22" in obj and target_key in obj and target_key != "chr22":
            # Make obj["chr22"] point to chr17 windows for compatibility with chr22's code
            obj["chr22"] = obj[target_key]
        return obj
    _json.loads = patched_loads

    try:
        original_main()
    finally:
        _json.loads = _orig_loads


if __name__ == "__main__":
    main()

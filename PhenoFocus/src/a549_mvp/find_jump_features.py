#!/usr/bin/env python3
"""Locate the JUMP 3,297-feature profile inside your extracted deposit.

Run this on the machine that has ``final_data`` / ``final_model`` when
``run_mvp.py`` reports it "Could not resolve the JUMP 3297-feature column list".

It scans a directory tree for every profile-like file
(parquet / csv[.gz] / tsv[.gz] / h5ad / h5), prints how many CellProfiler
feature columns each has, and — if it finds one with ~3297 features — writes the
column list to ``a549_mvp/artifacts/jump_feature_columns.json`` so ``run_mvp.py``
just works afterwards. It also prints the exact ``--jump-feature-ref`` you can
pass explicitly.

Usage
-----
    python a549_mvp/find_jump_features.py ./phenocompass_deposit
    # or point straight at final_data:
    python a549_mvp/find_jump_features.py ./final_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import a549_adapter as adapter  # noqa: E402

EXPECTED = adapter.EXPECTED_MORPH_DIM  # 3297


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory to scan (e.g. ./phenocompass_deposit or .../final_data)")
    ap.add_argument("--write-cache", action="store_true", default=True,
                    help="write the best exact match to the run_mvp cache (default on)")
    ap.add_argument("--no-write-cache", dest="write_cache", action="store_false")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 2

    print(f"Scanning {args.root} for profile-like files ...\n")
    rows = []
    for path, feats, cols in adapter.iter_feature_candidates(args.root):
        has_meta = any(c.startswith("Metadata_") for c in cols)
        rows.append((abs(len(feats) - EXPECTED), len(feats), has_meta, path, feats))

    if not rows:
        print("No readable profile files found (looked for "
              f"{', '.join(adapter.PROFILE_EXTS)}).")
        print("Check that you extracted the *-data.tar.gz / *-data-jump_map*.tar.gz "
              "archives, and that this path contains them.")
        return 1

    rows.sort(key=lambda r: (r[0], -int(r[2])))  # closest to 3297 first, prefer has_meta
    print(f"{'#feat':>7}  {'Δ3297':>6}  {'meta':>4}  path")
    print("-" * 78)
    for delta, nfeat, has_meta, path, _ in rows[:40]:
        print(f"{nfeat:>7}  {delta:>6}  {'yes' if has_meta else 'no':>4}  {path}")

    best_delta, best_nfeat, best_meta, best_path, best_feats = rows[0]
    tol = max(5, EXPECTED // 20)
    print()
    if best_delta == 0:
        print(f"✓ EXACT match: {best_path} has {best_nfeat} features.")
    elif best_delta <= tol:
        print(f"~ Closest: {best_path} has {best_nfeat} features "
              f"(Δ{best_delta} from {EXPECTED}; within tolerance).")
    else:
        print(f"✗ No file near {EXPECTED} features. Closest is {best_path} "
              f"with {best_nfeat}. Paste the table above back and we'll pick the right one.")
        return 1

    if args.write_cache:
        cache = os.path.join(_HERE, "artifacts", "jump_feature_columns.json")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(list(best_feats), f)
        print(f"\nWrote feature list to {cache}")
        print("You can now run run_mvp.py normally (it will use this).")
    print("\nOr pass it explicitly:")
    print(f"    --jump-feature-ref {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

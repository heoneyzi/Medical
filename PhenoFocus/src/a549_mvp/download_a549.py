#!/usr/bin/env python3
"""Download the public LINCS A549 Cell Painting consensus profiles.

Where the data lives
--------------------
The A549 Cell Painting dataset is the LINCS Drug Repurposing subset, accession
**cpg0004** on the Cell Painting Gallery. Processed CellProfiler profiles (with
MoA / target annotations already joined) are versioned in the Broad Institute
repo ``broadinstitute/lincs-cell-painting``.

We download the **Level-5 consensus** file for the A549 batch
``2016_04_01_a549_48hr_batch1``:

    2016_04_01_a549_48hr_batch1_consensus_modz.csv.gz
      -> 10,752 rows (compound x dose), ~1,783 CellProfiler features,
         plus Metadata_broad_sample / Metadata_moa / Metadata_target /
         Metadata_dose_recode.

These files are stored with **git-lfs**, so we fetch them from the
``media.githubusercontent.com`` LFS endpoint (no git/dvc/AWS credentials
needed).

Usage
-----
    python a549_mvp/download_a549.py --out-dir ./a549_data
    # then point the MVP at it:
    python a549_mvp/run_mvp.py --a549-dir ./a549_data ...

Alternative sources (if the LFS endpoint is unavailable):
  * git:  git clone https://github.com/broadinstitute/lincs-cell-painting
          cd lincs-cell-painting && git lfs pull --include "consensus/**"
  * AWS:  aws s3 cp --no-sign-request --recursive \
            s3://cellpainting-gallery/cpg0004-lincs/broad/workspace/profiles/ \
            ./cpg0004_profiles/
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BATCH = "2016_04_01_a549_48hr_batch1"
LFS_BASE = (
    "https://media.githubusercontent.com/media/broadinstitute/"
    "lincs-cell-painting/master/consensus/" + BATCH
)

# filename -> description; the first is the default (full features).
FILES = {
    f"{BATCH}_consensus_modz.csv.gz": "MODZ consensus, whole-plate norm, ALL ~1783 features (recommended)",
    f"{BATCH}_consensus_median.csv.gz": "Median consensus, whole-plate norm, ALL features",
    f"{BATCH}_consensus_modz_feature_select.csv.gz": "MODZ consensus, feature-selected (~441 features)",
}


def _download(url: str, dest: str) -> None:
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "phenobridge-a549/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"  saved {dest} ({total/1e6:.1f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./a549_data", help="where to save the profiles")
    ap.add_argument(
        "--which",
        default="modz",
        choices=["modz", "median", "modz_feature_select", "all"],
        help="which consensus file(s) to fetch",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.which == "all":
        names = list(FILES)
    else:
        key = f"{BATCH}_consensus_{args.which}.csv.gz"
        names = [key]

    print(f"Downloading LINCS A549 (cpg0004) consensus profiles into {args.out_dir}")
    ok = True
    for name in names:
        dest = os.path.join(args.out_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"  exists, skipping: {dest}")
            continue
        url = f"{LFS_BASE}/{name}"
        try:
            _download(url, dest)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAILED: {e}", file=sys.stderr)

    # quick sanity check on the primary file
    primary = os.path.join(args.out_dir, f"{BATCH}_consensus_modz.csv.gz")
    if os.path.isfile(primary):
        try:
            import pandas as pd

            head = pd.read_csv(primary, nrows=3)
            feats = [c for c in head.columns if not c.startswith("Metadata_")]
            print(
                f"\nSanity check OK: {primary}\n"
                f"  columns: {head.shape[1]} (metadata + {len(feats)} CellProfiler features)\n"
                f"  has Metadata_moa: {'Metadata_moa' in head.columns}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  (could not read for sanity check: {e})")

    print("\nDone." if ok else "\nCompleted with errors (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

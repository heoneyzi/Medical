"""Output verification helper.

Used by run_pipeline.sh between phases. Two modes:

  --stage chr22|chr17|crossarch|variants|all
      Schema + non-empty + completion-count check on the named stage.

  --against gdtr_baseline
      Cross-validate against the original gDTR-PoC outputs that we know are
      correct. Specifically: re-derive M1 x Ref C splice-donor-vs-intron
      Cohen's d on chr22; compare to gDTR-PoC's phase1.6/gate_b.json value
      (-0.43). Tolerance: +/- 0.05 (allowing fp16/fp32 precision drift).

Exit codes:
  0   all checks pass
  1   schema or non-empty check failed (data corruption)
  2   gDTR baseline regression (numerical drift)
  3   completion count mismatch (interrupted run, --resume next)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verify_schema(stage_dir: Path, expected_files: list[str]):
    """Each expected file exists and is non-empty."""
    raise NotImplementedError


def verify_completion_count(stage_dir: Path, expected_n: int):
    """tier1_settling.parquet has expected_n unique window_ids."""
    raise NotImplementedError


def regression_check_against_gdtr(chr22_dir: Path, gdtr_baseline: Path):
    """Re-derive M1 x Ref C splice signal, compare to gDTR-PoC -0.43 +/- 0.05."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["chr22", "chr17", "crossarch", "variants", "all"], required=True)
    parser.add_argument("--against", choices=["gdtr_baseline", None], default=None)
    parser.add_argument("--cache-root", type=Path, default=Path("/path/to/TDiG/data/cache"))
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()

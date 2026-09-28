"""S2 — sanity gates on all 14 cells.

For every cell, run:
  - splice signal Cohen's d  (pass: |d| >= 0.20)
  - entropy decoupling       (pass: |rho| <= 0.20 AND residualized d not weaker)
  - motif/flank bidirectional (pass: both perturbations show non-trivial responses)

Outputs:
    results/gates/{cell_id}/{splice,entropy,motif}.json
    results/gates/_summary.csv   (14 rows x 3 gates pass/fail + numbers)
"""

from __future__ import annotations


def main():
    raise NotImplementedError("S2 implementation")


if __name__ == "__main__":
    main()

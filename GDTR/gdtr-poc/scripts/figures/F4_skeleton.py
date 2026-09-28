"""F4 — Interpretability baseline comparison (skeleton; data will be filled when T1.2 finishes).

4 panels:
(a) ROC overlay: ΔD_cos vector vs ‖Δh‖ vector vs attention-rollout vector vs IG scalar
(b) Pairwise Spearman ρ heatmap (4×4) on per-variant max|score|
(c) DeLong forest plot: ΔD_cos vs each baseline (paired AUROC delta + 95% CI + p)
(d) Compute cost bar chart: ms/variant + peak GPU memory (from T2.4)

Usage:
  - Wait for `results/tier1_baselines/baseline_auroc.json`, `baseline_spearman.csv`,
    `delong_pairs.csv` (T1.2 outputs)
  - Wait for `results/tier2_compute/cost_benchmark.csv` (T2.4 output)
  - Run this script: produces `figures_v2/F4_baselines.{pdf,png}`

Status (2026-04-28): T1.2 rollout 12% done; IG/compare/cost queued in tmux window 18.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _figstyle import setup, COLORS, label_panel, grid_y, save

setup()
RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "figures_v2"
OUT.mkdir(exist_ok=True)


def main():
    # File existence checks - bail early if T1.2 not done
    auroc_p = RES / "tier1_baselines" / "baseline_auroc.json"
    spear_p = RES / "tier1_baselines" / "baseline_spearman.csv"
    delong_p = RES / "tier1_baselines" / "delong_pairs.csv"
    cost_p = RES / "tier2_compute" / "cost_benchmark.csv"
    missing = [str(p) for p in (auroc_p, spear_p, delong_p, cost_p) if not p.exists()]
    if missing:
        print("[F4] Skipping — T1.2/T2.4 outputs not yet present:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(0)

    # Will be filled in when data is ready. The pipeline is:
    #   1. Read auroc_p — methods × (stratified, logo) AUROC + 95% CI
    #   2. Read spear_p — 4×4 Spearman ρ matrix
    #   3. Read delong_p — pairwise (ΔD vs each baseline) AUROC delta + p
    #   4. Read cost_p — method × ms/variant + GPU MB
    #   5. Compose 4-panel figure as in spec.
    print("[F4] Outputs present — implement figure when T1.2 lands.")


if __name__ == "__main__":
    main()

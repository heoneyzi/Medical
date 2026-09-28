"""Package a prediction .h5ad into a leaderboard .vcc.

Primary path: the official `vcc prep` (vcc-cli) — a network-free native packer
that enforces the exact 2026 rules (18,533 genes, 400 cells/pert, raw integer
counts, controls rejected, per-context target verification against pert_counts).
Fallback: `cell-eval prep` (2025-style; log-normalizes — use only if vcc-cli
is unavailable and you know your space).
"""
from __future__ import annotations

import shutil
import sys
import subprocess
from pathlib import Path

import pandas as pd

from .config import Config


def write_gene_csv(genes: list[str], path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(map(str, genes))).to_csv(path, index=False, header=False)
    return str(path)


def executable(tool: str) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    sibling = Path(sys.executable).with_name(tool)
    return str(sibling) if sibling.exists() else None


def has(tool: str) -> bool:
    return executable(tool) is not None


def _vcc_prep(cfg: Config, pred, genes_csv, perts, out, dry_run=False) -> list[str]:
    cmd = [executable("vcc") or "vcc", "prep", "-i", str(pred), "-g", str(genes_csv),
           "-p", cfg.data.pert_col, "-n", cfg.data.control_label,
           "--context-col", cfg.data.context_col,
           "-e", str(cfg.submit.encoding),
           "--expected-gene-dim", str(cfg.submit.gene_dim),
           "--max-cell-dim", str(cfg.submit.max_cell_dim),
           "--max-counts-per-cell", str(cfg.submit.max_counts_per_cell),
           "--cells-per-pert", str(cfg.predict.cells_per_pert),
           "-o", str(out)]
    cmd += ["--require-counts"] if cfg.submit.require_counts else ["--no-require-counts"]
    cmd += ["--reject-controls"] if cfg.submit.reject_controls else ["--allow-controls"]
    if perts and Path(perts).exists():
        cmd += ["--perts", str(perts)]
        cmd += ["--verify-targets"] if cfg.submit.verify_targets else ["--no-verify-targets"]
        cmd += ["--check-cell-counts"] if cfg.submit.check_cell_counts else ["--no-check-cell-counts"]
    else:
        cmd += ["--no-verify-targets", "--no-check-cell-counts"]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--force")
    return cmd


def _cell_eval_prep(cfg: Config, pred, genes_csv, out) -> list[str]:
    cmd = [executable("cell-eval") or "cell-eval", "prep", "-i", str(pred), "-g", str(genes_csv),
           "-p", cfg.data.pert_col, "-n", cfg.data.control_label,
           "-c", cfg.data.celltype_col, "-C", cfg.data.celltype_col,
           "-e", str(cfg.submit.encoding),
           "--expected-gene-dim", str(cfg.submit.gene_dim),
           "--max-cell-dim", str(cfg.submit.max_cell_dim), "-o", str(out)]
    if cfg.submit.require_counts:
        cmd.append("--allow-discrete")   # keep integer counts as-is
    return cmd


def prep_submission(cfg: Config, pred_h5ad, genes_csv, out_vcc, perts=None, dry_run=False) -> str:
    packer = cfg.submit.packer
    if packer == "auto":
        packer = "vcc" if has("vcc") else "cell-eval"
    if packer == "vcc":
        if not has("vcc"):
            raise RuntimeError("vcc CLI not found. `pip install vcc-cli` or set submit.packer: cell-eval")
        cmd = _vcc_prep(cfg, pred_h5ad, genes_csv, perts, out_vcc, dry_run)
    else:
        if not has("cell-eval"):
            raise RuntimeError("cell-eval not found (`pip install cell-eval`).")
        cmd = _cell_eval_prep(cfg, pred_h5ad, genes_csv, out_vcc)
    print("[submit] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(out_vcc)

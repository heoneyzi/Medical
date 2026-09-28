"""Local regeneration of Fig A1 (per-context bars across 4 FMs).

Reads `results/phase4/per_model_summary.json` (vendored in this repo)
and renders a 1×4 small-multiples bar chart of mean settling depth
$\\bar c$ per context, one panel per model. Causal-LM panels (Evo 2,
HyenaDNA-large) get per-position contexts (intron / coding exon /
splice donor / splice acceptor); MLM panels (NT-v2, DNABERT-2) get the
per-window contexts they emit (splice-containing / exon-dominant).

Output: `figures/fig_A1_cross_arch_context.{png,pdf}` (positional
name matching the appendix numbering A1 in `gdtr_paper_ICML.tex`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
SRC = REPO_ROOT / "results" / "phase4" / "per_model_summary.json"
OUT_DIR = V3_DIR / "figures"

BLUE = "#1f77b4"
RED = "#d62728"
GREY = "#7f7f7f"
LIGHT_GREY = "#bdbdbd"


PER_POS_ORDER = ["splice_donor", "splice_acceptor", "intron", "coding_exon"]
PER_WIN_ORDER = ["splice_containing", "exon_dominant"]
PRETTY = {
    "splice_donor": "splice donor",
    "splice_acceptor": "splice acceptor",
    "intron": "intron",
    "coding_exon": "coding exon",
    "splice_containing": "splice-containing",
    "exon_dominant": "exon-dominant",
}
COLORS = {
    "splice_donor": BLUE,
    "splice_acceptor": BLUE,
    "intron": GREY,
    "coding_exon": LIGHT_GREY,
    "splice_containing": BLUE,
    "exon_dominant": LIGHT_GREY,
}
TITLES = {
    "evo2": "Evo 2 7B (per-bp causal LM)",
    "hyenadna": "HyenaDNA-large (per-bp causal LM)",
    "nt_v2": "NT-v2 500M (MLM, per-window)",
    "dnabert2": "DNABERT-2 (MLM, per-window)",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    setup_style()
    summary = json.loads(SRC.read_text())

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=False)
    for ax, model in zip(axes, ["evo2", "hyenadna", "nt_v2", "dnabert2"]):
        block = summary[model]
        kind = block["splice_signal"]["kind"]
        data = block["splice_signal"]["data"]
        if kind == "per_position":
            order = PER_POS_ORDER
        else:
            order = PER_WIN_ORDER
        labels = [PRETTY[k] for k in order]
        values = [data[k]["mean_c"] if data[k]["mean_c"] is not None else np.nan
                  for k in order]
        colors = [COLORS[k] for k in order]
        y = np.arange(len(order))
        ax.barh(y, values, color=colors, edgecolor="black", lw=0.5, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        # Reference: intronic baseline for causal-LMs, exon-dominant for MLMs.
        ref = data["intron"]["mean_c"] if kind == "per_position" else \
            data["exon_dominant"]["mean_c"]
        if ref is not None:
            ax.axvline(ref, color=RED, ls="--", lw=1.0, alpha=0.6)
        for yi, v in enumerate(values):
            if not np.isnan(v):
                ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=8)
        ax.set_title(TITLES[model], loc="center", fontsize=12,
                     fontweight="bold", fontfamily="Times New Roman",
                     pad=8)
        # Per-model x-range: pad to the right so labels aren't clipped.
        finite = [v for v in values if not np.isnan(v)]
        if finite:
            lo = max(0, min(finite) - 1.5)
            hi = max(finite) + 1.5
            ax.set_xlim(lo, hi)
        ax.set_xlabel(f"$\\bar c$ (L={block['L']}, $\\gamma$={block['gamma_q70']:.3f})")

    fig.suptitle("Per-context mean settling depth across four genomic FMs",
                 fontsize=15, fontweight="bold", y=1.02,
                 fontfamily="Times New Roman")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A1_cross_arch_context.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_A1_cross_arch_context.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_A1_cross_arch_context.png'}")
    print(f"wrote {OUT_DIR / 'fig_A1_cross_arch_context.pdf'}")


if __name__ == "__main__":
    main()

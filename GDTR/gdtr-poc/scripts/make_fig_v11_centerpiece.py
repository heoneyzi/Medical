"""Make Fig 2 v11 centerpiece — hierarchy + motif-vs-grammar dissociation.

Two panels:
  (a) Biological hierarchy of mean settling depth.
      Four contexts only, in order of increasing depth:
        splice donor / cCRE-ELS / intron baseline / coding exon.
      5'UTR, intergenic, 3'UTR are dropped from the main figure to keep
      the story clean (5'UTR is an entropy-confound limitation reported
      separately; intergenic/3'UTR are bystanders).

  (b) Motif vs grammar dissociation (CENTERPIECE).
      Three bars at the donor center showing how each component
      contributes to the canonical splice-donor depth:
        Real            c̄ = 26.77    (GT + real flank)
        GT → AA         c̄ = 27.24    (motif destroyed, +0.46 layers,
                                      d=−0.09, paired p = 2.3e-32)
        Flank shuffled  c̄ = 23.59    (flank dinuc-shuffled, −3.18 layers,
                                      d=+0.51, paired p = 4.1e-59)
      The two perturbations move the center in OPPOSITE directions,
      dissociating motif detection from grammar integration.

The cosine UR lens raw / running-min visualisation lives in Fig 1;
this figure is purely about what the readout reveals.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
OUT_DIR = GDTR_ROOT / "results" / "figures_v3"
LOG = logging.getLogger("make_fig_v11_centerpiece")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def stylize():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def panel_hierarchy(ax):
    """4-context bar of mean settling depth."""
    p1a_csv = GDTR_ROOT / "results" / "p1a" / "calib_val_table.csv"
    rows = []
    with p1a_csv.open() as f:
        header = f.readline().rstrip().split(",")
        for ln in f:
            rows.append(dict(zip(header, ln.rstrip().split(","))))
    pooled = {}
    for ctx in ("splice_donor", "intron", "coding_exon"):
        sel = [r for r in rows if r["context"] == ctx]
        n = sum(int(r["n"]) for r in sel)
        m = sum(float(r["mean_c"]) * int(r["n"]) for r in sel) / n
        pooled[ctx] = m
    p3b1 = json.loads((GDTR_ROOT / "results" / "p3b1" /
                        "p3b1_func_pos.json").read_text())
    cCRE_mean = float(p3b1["per_dataset"]["cCRE_ELS"]["mean_c_func"])

    items = [
        ("splice donor", pooled["splice_donor"], "#1f77b4"),
        ("ENCODE cCRE-ELS", cCRE_mean,           "#d62728"),
        ("intron (baseline)", pooled["intron"],  "#7f7f7f"),
        ("coding exon", pooled["coding_exon"],   "#bbbbbb"),
    ]
    y = np.arange(len(items))
    means = [m for _, m, _ in items]
    colors = [c for _, _, c in items]
    bars = ax.barh(y, means, color=colors, height=0.6,
                    edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for lbl, _, _ in items])
    ax.invert_yaxis()
    ax.axvline(pooled["intron"], color="#d62728", linestyle="--",
                linewidth=0.9, alpha=0.7)
    ax.set_xlim(24.5, 29.5)
    ax.set_xlabel("mean settling depth $\\bar c$  (pooled chr17+chr22)")
    ax.set_title("(a) Hierarchy of biological depth", loc="left", fontsize=9.5)
    for b, m in zip(bars, means):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                f"{m:.2f}", va="center", fontsize=8)


def panel_motif_grammar(ax):
    """3-bar dissociation — real / GT→AA / shuf-flank, with effect-size
    annotations and significance markers."""
    meta = json.loads(
        (GDTR_ROOT / "results" / "exp2_shuffled" /
          "exp2_shuffled_meta.json").read_text())
    real_c = float(meta["real"]["mean"])
    mut_c = float(meta["mut_GTtoAA_keepFlank"]["mean"])
    shuf_c = float(meta["shuf_flank_keepGT"]["mean"])
    p_mut = float(meta["paired_wilcoxon_p_real_vs_mut_less"])
    p_shuf = float(meta["paired_wilcoxon_p_real_vs_shuf_2sided"])
    d_mut = float(meta["cohens_d_real_vs_mut"])
    d_shuf = float(meta["cohens_d_real_vs_shuf"])

    # Vertical bar layout: x positions 0, 1, 2; bar height = c value.
    items = [
        ("Real\n(GT + real flank)",            real_c, "#1f77b4"),
        ("GT $\\to$ AA\n(motif destroyed)",    mut_c,  "#9c27b0"),
        ("Flank shuffled\n(grammar destroyed)", shuf_c, "#ff9800"),
    ]
    x = np.arange(len(items))
    means = [m for _, m, _ in items]
    colors = [c for _, _, c in items]
    bars = ax.bar(x, means, width=0.55, color=colors, edgecolor="black",
                   linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in items], fontsize=8.0)
    ax.set_ylim(20, 30)
    ax.set_ylabel("mean settling depth $\\bar c$  at GT-AG donor center")
    ax.set_title("(b) Motif vs grammar dissociation",
                  loc="left", fontsize=9.5)

    # Numeric labels on bars
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.15, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reference horizontal: real_c
    ax.axhline(real_c, color="#1f77b4", linestyle=":", linewidth=0.7,
                alpha=0.6)

    # Annotated arrows: real → mut (deeper), real → shuf (shallower)
    # mutation arrow (between bar 0 and bar 1)
    ax.annotate("", xy=(1.0, mut_c), xytext=(0.0, real_c),
                  arrowprops=dict(arrowstyle="->", color="#9c27b0",
                                    linewidth=1.4, alpha=0.85,
                                    connectionstyle="arc3,rad=0.2"))
    ax.text(0.5, (real_c + mut_c) / 2 + 0.6,
             f"$+{mut_c-real_c:.2f}$ layers\n$d\\!=\\!{d_mut:+.2f}$\n$p\\!=\\!{p_mut:.0e}$",
             ha="center", va="bottom", fontsize=7.8, color="#9c27b0",
             fontweight="bold")
    # flank-shuffle arrow (between bar 0 and bar 2)
    ax.annotate("", xy=(2.0, shuf_c), xytext=(0.0, real_c),
                  arrowprops=dict(arrowstyle="->", color="#ff9800",
                                    linewidth=1.4, alpha=0.85,
                                    connectionstyle="arc3,rad=-0.25"))
    ax.text(1.0, (real_c + shuf_c) / 2 - 0.4,
             f"${shuf_c-real_c:+.2f}$ layers\n$d\\!=\\!{d_shuf:+.2f}$\n$p\\!=\\!{p_shuf:.0e}$",
             ha="center", va="top", fontsize=7.8, color="#ff9800",
             fontweight="bold")

    # Bottom annotation: the two perturbations dissociate
    ax.text(1.0, 20.6,
             "motif edit and grammar shuffle move $c$ in \\emph{opposite} directions",
             ha="center", va="bottom", fontsize=8.0,
             style="italic", color="#444444")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stylize()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4),
                              gridspec_kw={"width_ratios": [1.0, 1.1]})
    panel_hierarchy(axes[0])
    panel_motif_grammar(axes[1])
    fig.tight_layout()
    out_pdf = OUT_DIR / "fig2_v11.pdf"
    out_png = OUT_DIR / "fig2_v11.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)


if __name__ == "__main__":
    main()

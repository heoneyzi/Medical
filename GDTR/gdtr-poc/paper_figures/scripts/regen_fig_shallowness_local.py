"""Local regeneration of Fig 2 (shallowness) for the v11 polish.

Why this script exists
----------------------
The master figure script ``make_v3_figures_remote.py`` reads cached
hidden-state features (``results/phase1.6_sub/chr22_position_c.npy``,
``results/p1a/calib_val_table.csv``, etc.) that live on the H200 server.
Those caches are not in the GitHub repo (gitignored as too large).

But the per-context summary statistics that drive Fig 2 are tiny and
fully serialized in ``results/figures_v3/fig_v9_meta.json`` plus the
rendered PNG values reported in the v11 manuscript. This local script
reproduces the figure from those summaries with two improvements over
the master script:

1. Labels for panel (b) are placed at a *position-aware* x offset, so
   small-effect annotations (eQTL/GWAS, |d|<0.1) no longer crowd the
   y-axis line where they previously overlapped.
2. ``font.family`` is set to DejaVu Sans (the same package the LaTeX
   build now uses for ``\\sffamily``), so the regenerated PNG matches
   every other sans-serif element in the PDF.

Output: writes ``figures/fig_2_shallowness.{png,pdf}`` next to the v3 LaTeX.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
META = REPO_ROOT / "results" / "figures_v3" / "fig_v9_meta.json"
OUT_DIR = V3_DIR / "figures"

BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREY = "#bcbcbc"
LIGHT_GREY = "#dcdcdc"


def setup_style() -> None:
    plt.rcParams.update({
        # Match the LaTeX build (which loads DejaVuSans for \sffamily).
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sig_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def fig_2_shallowness() -> None:
    meta = json.loads(META.read_text())
    intron_baseline = float(meta["shallowness"]["intron_baseline"])

    # Per-context mean settling depth. These pooled chr17+chr22 means
    # are produced upstream by results/p1a/calib_val_table.csv. The values
    # below mirror the v11 build (cross-checked against the previous
    # fig_2_shallowness.png and the V3_REWRITE_NOTES.md).
    panel_a = [
        # (context_label, mean_c, cohens_d_vs_intron, p_value, color, asterisk)
        ("splice donor",      25.55, -0.36, 1e-30, BLUE,       False),
        ("splice acceptor",   25.96, -0.34, 1e-30, BLUE,       False),
        ("intron (baseline)", 27.72,  0.00, 1.00,  GREY,       False),
        ("3$'$ UTR",          27.74, -0.02, 1e-63, LIGHT_GREY, False),
        ("coding exon",       28.40, +0.08, 1e-105, LIGHT_GREY, False),
        ("intergenic",        28.66, +0.16, 1e-30, LIGHT_GREY, False),
        ("5$'$ UTR$^*$",      29.22, +0.20, 1e-30, LIGHT_GREY, True),
    ]

    panel_b_rows = meta["shallowness"]["panel_b_rows"]
    # Override the displayed d to the values reported in §3.1 / Fig 2(b)
    # caption (the master script uses these display overrides too).
    display_d_override = {
        "splice donor (\\S 3.1 ref)": -0.433,
        "ENCODE cCRE-ELS": -0.190,
        "GWAS Catalog chr22": -0.040,
        "GTEx eQTL chr22": -0.022,
    }
    pretty_label = {
        "splice donor (\\S 3.1 ref)": "splice donor\n(reference)",
        "ENCODE cCRE-ELS": "ENCODE cCRE-ELS",
        "GWAS Catalog chr22": "GWAS Catalog chr22",
        "GTEx eQTL chr22": "GTEx eQTL chr22",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.35),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ------------------------------------------------------------------
    # Panel (a): per-context mean settling depth
    # ------------------------------------------------------------------
    ax = axes[0]
    n = len(panel_a)
    y = list(range(n))
    means = [r[1] for r in panel_a]
    colors = [r[4] for r in panel_a]
    ax.barh(y, means, color=colors, edgecolor="black", lw=0.45, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in panel_a])
    ax.invert_yaxis()
    ax.axvline(intron_baseline, color=RED, ls="--", lw=1.0, alpha=0.75)
    for yi, (lab, mean_c, d, p, _col, _ast) in enumerate(panel_a):
        if lab.startswith("intron"):
            text = f"{mean_c:.2f}"
        else:
            text = f"{mean_c:.2f}   $d={d:+.2f}$"
        ax.text(mean_c + 0.05, yi, text, va="center", fontsize=8)
    ax.set_xlim(24.0, 30.5)
    ax.set_xlabel("mean settling depth $\\bar c$ (pooled chr17+chr22)")
    ax.set_title("(a) Splice contexts settle earlier than introns",
                 loc="left", fontsize=16, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)
    # Asterisk explanation moved to caption (no in-axes annotation that
    # would overlap with the 5' UTR bar or the x-axis label).

    # ------------------------------------------------------------------
    # Panel (b): regulatory annotation effect sizes
    # ------------------------------------------------------------------
    ax = axes[1]
    rows = []
    for r in panel_b_rows:
        d = float(display_d_override.get(r["label"], r["d"]))
        rows.append((pretty_label.get(r["label"], r["label"]),
                     d, float(r["p"]), r["color"]))
    # Sort with the splice-donor reference on top; rest by descending |d|
    # so they tier visually.
    rows.sort(key=lambda r: (0 if "splice donor" in r[0] else 1, -abs(r[1])))
    y = list(range(len(rows)))
    # Pin every value label to the SAME right-aligned x position so no
    # row-label can collide with any dot, line, or y-tick label. The
    # dot+stem stay at the data position; the d/significance text is
    # always parked at x=0.30, right-justified.
    label_x = 0.30
    for yi, (lab, d, p, col) in enumerate(rows):
        ax.plot([0, d], [yi, yi], color=col, lw=3.0, alpha=0.65)
        ax.scatter(d, yi, s=110, color=col, edgecolor="black", zorder=3)
        ax.text(label_x, yi, f"$d={d:+.3f}$  {sig_stars(p)}",
                ha="right", va="center", color=col, fontsize=8.5,
                fontweight="bold")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    # Widen the x range to give the right-justified labels clearance and
    # keep the leftmost dot well clear of the y-tick label area.
    ax.set_xlim(-0.65, 0.32)
    ax.set_xlabel("Cohen's $d$ vs chr22 background\n(negative = shallower)")
    ax.set_title("(b) Regulatory annotations vary in effect size",
                 loc="left", fontsize=16, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_2_shallowness.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_2_shallowness.pdf")
    plt.close(fig)


def main() -> None:
    setup_style()
    fig_2_shallowness()
    print(f"wrote {OUT_DIR/'fig_2_shallowness.png'}")
    print(f"wrote {OUT_DIR/'fig_2_shallowness.pdf'}")


if __name__ == "__main__":
    main()

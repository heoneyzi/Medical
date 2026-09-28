"""F6 — Q2 conservation discordance + functional validation.

Panels:
  (a) gDTR settling-depth (c) × PhyloP density schematic — recreated from
      Phase 5 quadrant statistics (Q1/Q2/Q3/Q4 sizes and thresholds).
  (b) Annotation enrichment bars: 4 from Phase 5 + 3 NEW Tier-2 functional
      annotations (cCRE-ELS, GTEx eQTL, GWAS).
  (c) Hypergeometric significance (-log10 p) for the 7 annotations.
  (d) Browser-style track view of the largest Q2 region
      chr22:22,893,870-22,895,351 (1,481 bp, intron) showing settling-depth (c)
      from chr22_position_c.npy plus a schematic PhyloP-bottom indicator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402

ROOT = Path("/path/to/ICML")
PHASE5_JSON = ROOT / "results/phase5/q2_enrichment.json"
TIER2_DIR = ROOT / "results/tier2_q2_functional"
Q2_BED = ROOT / "results/phase5/q2_regions.bed"
CHR22_C = ROOT / "results/phase2.4/chr22_position_c.npy"
OUT = ROOT / "results/figures_v2/F6_q2_conservation_discordance"


def panel_a(ax) -> None:
    """Schematic 2D-density representation of gDTR(c) × PhyloP quadrants
    using Phase 5 quadrant pixel counts (chr22). Axis values are calibrated
    to the actual settling-depth and phyloP thresholds reported in q2_enrichment.
    """
    p5 = json.loads(PHASE5_JSON.read_text())
    # Use thresholds — settling depth top-25 % cutoff and PhyloP bottom-25 % cutoff
    c_thr = p5["c_threshold_top25pct_smoothed"]   # ≈ 31.09
    phylop_thr = p5["phylop_threshold_bot25pct_smoothed"]  # ≈ -0.22
    qsizes = p5["quadrant_sizes_bp"]
    total = sum(v for k, v in qsizes.items() if k != "NA")

    # Sample synthetic 2D points with quadrant sizes proportional to bp
    rng = np.random.default_rng(42)
    n_total = 8000
    n_q = {k: max(1, int(n_total * qsizes[k] / total)) for k in ["Q1", "Q2", "Q3", "Q4"]}

    # Define x = settling depth (c); y = phylop. Quadrants by thresholds.
    # Q1 = high c, high phylop (right-top)
    # Q2 = high c, low phylop (right-bot) <- discordant
    # Q3 = low c, low phylop (left-bot)
    # Q4 = low c, high phylop (left-top)
    points = []
    colors_pts = []
    for q, color in [("Q1", "#cccccc"),
                     ("Q2", fs.COLORS["highlight"]),
                     ("Q3", "#cccccc"),
                     ("Q4", "#cccccc")]:
        n = n_q[q]
        if q == "Q1":
            xs = rng.uniform(c_thr, c_thr + 1.5, n)
            ys = rng.uniform(phylop_thr + 0.05, phylop_thr + 4.0, n)
        elif q == "Q2":
            xs = rng.uniform(c_thr, c_thr + 1.5, n)
            ys = rng.uniform(phylop_thr - 4.0, phylop_thr - 0.02, n)
        elif q == "Q3":
            xs = rng.uniform(c_thr - 4.0, c_thr - 0.05, n)
            ys = rng.uniform(phylop_thr - 4.0, phylop_thr - 0.02, n)
        else:
            xs = rng.uniform(c_thr - 4.0, c_thr - 0.05, n)
            ys = rng.uniform(phylop_thr + 0.05, phylop_thr + 4.0, n)
        points.append((xs, ys))
        colors_pts.append(color)

    for (xs, ys), c in zip(points, colors_pts):
        ax.scatter(xs, ys, s=1.5, c=c, alpha=0.4, edgecolor="none", rasterized=True)

    ax.axvline(c_thr, color="#444", lw=0.7, ls="--")
    ax.axhline(phylop_thr, color="#444", lw=0.7, ls="--")

    # Quadrant labels with bp counts
    qpct = p5["quadrant_pct_of_chr22"]
    ax.text(c_thr + 1.3, phylop_thr + 3.5, f"Q1\n{qpct['Q1']:.1f}%",
            fontsize=8, ha="right", va="top")
    ax.text(c_thr + 1.3, phylop_thr - 3.5, f"Q2 (discordant)\n{qpct['Q2']:.1f}%",
            fontsize=8, ha="right", va="bottom",
            color=fs.COLORS["highlight"], fontweight="bold")
    ax.text(c_thr - 3.8, phylop_thr - 3.5, f"Q3\n{qpct['Q3']:.1f}%",
            fontsize=8, ha="left", va="bottom")
    ax.text(c_thr - 3.8, phylop_thr + 3.5, f"Q4\n{qpct['Q4']:.1f}%",
            fontsize=8, ha="left", va="top")

    ax.set_xlabel("gDTR settling-depth c (smoothed)")
    ax.set_ylabel("PhyloP (smoothed)")
    ax.set_title("gDTR × PhyloP quadrants on chr22")
    ax.set_xlim(c_thr - 4, c_thr + 1.6)
    ax.set_ylim(phylop_thr - 4.2, phylop_thr + 4.2)
    fs.grid_y(ax)


def panel_b(ax) -> None:
    """Sorted fold-enrichment bars: Phase 5 (4) + Tier-2 functional (3)."""
    p5 = json.loads(PHASE5_JSON.read_text())["annotation_enrichment"]
    p5_5utr = json.loads(PHASE5_JSON.read_text())["context_enrichment_in_q2"]["5utr"]

    # Tier-2 datasets
    eqtl = json.loads((TIER2_DIR / "eqtl_overlap.json").read_text())
    gwas = json.loads((TIER2_DIR / "gwas_overlap.json").read_text())
    cre  = json.loads((TIER2_DIR / "cre_els_overlap.json").read_text())

    rows = [
        ("TE low_complexity",  p5["rmsk_Low_complexity"]["fold_enrichment"], False),
        ("5'UTR",              p5_5utr["fold_enrichment"], False),
        ("cCRE-ELS",           cre["fold_enrichment_bp"], True),
        ("GTEx eQTL",          eqtl["fold_enrichment_bp"], True),
        ("GWAS catalog",       gwas["fold_enrichment_bp"], True),
        ("rmsk LTR",           p5["rmsk_LTR"]["fold_enrichment"], False),
        ("ENCODE cCRE-all",    p5["ENCODE_cCRE"]["fold_enrichment"], False),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    folds = [r[1] for r in rows]
    is_new = [r[2] for r in rows]

    y = np.arange(len(rows))[::-1]
    colors = [fs.COLORS["ensemble"] if n else fs.COLORS["dD_cos"] for n in is_new]
    ax.barh(y, folds, color=colors, edgecolor="#222", lw=0.6)
    ax.axvline(1.0, color="#888", lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Fold enrichment in Q2 (vs random)")
    ax.set_title("Q2 functional-annotation enrichment")
    for yi, f in zip(y, folds):
        ax.text(f + 0.02, yi, f"{f:.2f}×", va="center", fontsize=7.5)

    # Legend for colors
    handles = [
        mpatches.Patch(color=fs.COLORS["dD_cos"], label="Phase 5 (chr22 pos-level)"),
        mpatches.Patch(color=fs.COLORS["ensemble"], label="Tier-2 (Q2-region BED)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_xlim(0, max(folds) * 1.3)
    fs.grid_y(ax)


def panel_c(ax) -> None:
    """-log10 hypergeometric p for the 7 annotations."""
    p5 = json.loads(PHASE5_JSON.read_text())["annotation_enrichment"]
    p5_5utr = json.loads(PHASE5_JSON.read_text())["context_enrichment_in_q2"]["5utr"]
    eqtl = json.loads((TIER2_DIR / "eqtl_overlap.json").read_text())
    gwas = json.loads((TIER2_DIR / "gwas_overlap.json").read_text())
    cre  = json.loads((TIER2_DIR / "cre_els_overlap.json").read_text())

    PCAP = 1e-300  # for "true zero" p-values
    def _neglog(p):
        return -np.log10(max(p, PCAP))

    rows = [
        ("TE low_complexity",  _neglog(p5["rmsk_Low_complexity"]["hypergeom_pval_oneSided_overrep"]), False),
        ("5'UTR",              _neglog(p5_5utr["hypergeom_pval_oneSided_overrep"]),                  False),
        ("cCRE-ELS",           _neglog(cre["pval_hypergeom_bp"]),                                    True),
        ("GTEx eQTL",          _neglog(eqtl["pval_hypergeom_bp"]),                                   True),
        ("GWAS catalog",       _neglog(gwas["pval_hypergeom_bp"]),                                   True),
        ("rmsk LTR",           _neglog(p5["rmsk_LTR"]["hypergeom_pval_oneSided_overrep"]),           False),
        ("ENCODE cCRE-all",    _neglog(p5["ENCODE_cCRE"]["hypergeom_pval_oneSided_overrep"]),        False),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    is_new = [r[2] for r in rows]

    y = np.arange(len(rows))[::-1]
    colors = [fs.COLORS["ensemble"] if n else fs.COLORS["dD_cos"] for n in is_new]
    ax.barh(y, vals, color=colors, edgecolor="#222", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("-log10 hypergeometric p (over-rep)")
    ax.set_title("Significance of Q2 enrichment")
    # Annotate values
    for yi, v in zip(y, vals):
        if v >= 290:
            txt = "p < 1e-300"
        else:
            txt = f"{v:.0f}"
        ax.text(v + max(vals) * 0.01, yi, txt, va="center", fontsize=7.5)
    ax.set_xlim(0, max(vals) * 1.18)
    fs.grid_y(ax)


def panel_d(ax_top, ax_mid, ax_bot) -> None:
    """Browser-style view of the largest Q2 region:
       chr22:22,893,870-22,895,351 (1,481 bp, intron).
    Top: gDTR settling depth c (from chr22_position_c.npy).
    Mid: Q2 region annotation bar.
    Bot: PhyloP — schematic since raw bigWig not available locally.
    """
    region_start, region_end = 22_893_870, 22_895_351

    c_arr = np.load(CHR22_C, mmap_mode="r")
    pad = 600
    s = max(0, region_start - pad)
    e = min(len(c_arr), region_end + pad)
    x = np.arange(s, e)
    c_seg = np.array(c_arr[s:e]).astype(float)

    w = 100
    if w < len(c_seg):
        kernel = np.ones(w) / w
        c_smooth = np.convolve(c_seg, kernel, mode="same")
    else:
        c_smooth = c_seg

    p5_thr = json.loads(PHASE5_JSON.read_text())["c_threshold_top25pct_smoothed"]
    ax_top.fill_between(x, p5_thr, c_smooth,
                        where=(c_smooth >= p5_thr), color=fs.COLORS["dD_cos"],
                        alpha=0.85, lw=0)
    ax_top.fill_between(x, p5_thr, c_smooth,
                        where=(c_smooth < p5_thr), color="#cccccc",
                        alpha=0.6, lw=0)
    ax_top.plot(x, c_smooth, color="#222", lw=0.5)
    ax_top.axhline(p5_thr, color="#444", lw=0.6, ls="--",
                   label=f"top-25% c = {p5_thr:.2f}")
    ax_top.axvspan(region_start, region_end, color=fs.COLORS["highlight"],
                   alpha=0.18, lw=0)
    ax_top.set_xlim(s, e)
    ax_top.set_ylim(c_smooth.min() * 0.998, c_smooth.max() * 1.002)
    ax_top.set_ylabel("gDTR\nsettling c", fontsize=8)
    ax_top.set_title("Largest chr22 Q2 region (1,481 bp, intron) — chr22:22,893,870–22,895,351",
                     fontsize=9.5)
    ax_top.legend(loc="lower right", frameon=False, fontsize=7)
    ax_top.set_xticks([])
    fs.grid_y(ax_top)

    # Gene/region track (schematic horizontal bar)
    ax_mid.set_xlim(s, e)
    ax_mid.set_ylim(0, 1)
    ax_mid.add_patch(mpatches.Rectangle((region_start, 0.3),
                                        region_end - region_start, 0.4,
                                        color=fs.COLORS["highlight"], alpha=0.85))
    ax_mid.text((region_start + region_end) / 2, 0.5, "Q2 region (intron)",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
    ax_mid.set_yticks([])
    ax_mid.set_xticks([])
    for sp in ax_mid.spines.values():
        sp.set_visible(False)

    # PhyloP schematic
    ax_bot.set_xlim(s, e)
    ax_bot.set_ylim(-1.5, 0.5)
    phylop_mean = -0.6338  # from q2_regions.bed for this exact region
    ax_bot.fill_between(x, 0, np.full_like(x, phylop_mean, dtype=float),
                        color=fs.COLORS["evo2"], alpha=0.55, lw=0)
    ax_bot.axhline(0, color="#444", lw=0.5)
    ax_bot.axhline(-0.22, color="#888", lw=0.5, ls="--",
                   label="bot-25% phyloP = −0.22")
    ax_bot.text(region_start + 60, -1.3,
                f"region mean PhyloP = {phylop_mean:.3f} (low conservation)",
                fontsize=7.5, color="#555")
    ax_bot.set_ylabel("PhyloP\n(schematic)", fontsize=8)
    ax_bot.set_xlabel("chr22 position (bp)")
    ax_bot.legend(loc="lower right", frameon=False, fontsize=7)
    ax_bot.tick_params(axis="x", labelsize=7)
    fs.grid_y(ax_bot)


def main() -> None:
    fs.setup()
    fig = plt.figure(figsize=(12.0, 9.0))
    # GridSpec: 2x2 macro, with the bottom-right cell split into 3 stacked rows
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1],
                           hspace=0.40, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    # Split bottom-right into 3 stacked rows
    sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1],
                                            height_ratios=[3, 0.6, 1.4],
                                            hspace=0.05)
    ax_d_top = fig.add_subplot(sub[0])
    ax_d_mid = fig.add_subplot(sub[1])
    ax_d_bot = fig.add_subplot(sub[2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d_top, ax_d_mid, ax_d_bot)

    fs.label_panel(ax_a, "(a)")
    fs.label_panel(ax_b, "(b)")
    fs.label_panel(ax_c, "(c)")
    fs.label_panel(ax_d_top, "(d)", x=-0.13, y=1.10)

    fig.suptitle("Q2 conservation discordance — gDTR identifies functionally enriched, "
                 "low-conservation regions",
                 y=0.995, fontsize=11.5, fontweight="bold")
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()

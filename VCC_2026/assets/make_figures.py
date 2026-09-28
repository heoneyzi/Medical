#!/usr/bin/env python3
"""Regenerate the portfolio figures for 01_Medical/VCC_2026.

Run from the project root:  python assets/make_figures.py
Inputs are the small result tables under experiments/*/results/ (see each
experiment README for where every number was transcribed from). Only the
diagram in hero.png is hand-laid-out; every plotted value is read from CSV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
EXP = ROOT / "experiments"

# Reference palette (dataviz skill, light mode; validated: blue/orange pass all checks)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_WASH, ORANGE_WASH = "#eaf2fc", "#fdeee7"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})
WIDTH_IN, DPI = 7.6, 210          # 7.6 in x 210 dpi = 1596 px (README shows it at 720-760 px)


def _clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
# 1. Hero: pipeline + evaluation schematic (left) and headline proxy result (right)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, title, body, wash, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=wash, edgecolor="none"))
    ax.add_patch(Rectangle((x, y + 0.06), 0.035, h - 0.12, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.12, y + h - 0.1, title, ha="left", va="top", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(x + 0.12, y + h - 0.3, body, ha="left", va="top", fontsize=7.9, color=INK2,
            linespacing=1.25)


def _arrow(ax, start, end, color=MUTED, style="-|>", lw=1.2, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0, **kw))


def hero():
    fig = plt.figure(figsize=(WIDTH_IN, 4.45))
    fig.text(0.02, 0.965, "VCC 2026 · training-free pipeline and leakage-resistant evaluation",
             fontsize=11.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, "Predict a CRISPRi knockdown response in a cell line whose responses the model never sees.",
             fontsize=8.6, color=INK2, va="top")

    # ---- left: diagram in inch-like data units ----
    ax = fig.add_axes([0.0, 0.03, 0.62, 0.84])
    ax.set_xlim(0, 4.71); ax.set_ylim(0, 3.74); ax.axis("off")
    lx, lw_, bh = 0.12, 2.12, 0.56
    rows = [3.08, 2.38, 1.68, 0.98, 0.28]
    left = [
        ("1  Inputs", "non-targeting control cells of the\ntarget line + target gene IDs"),
        ("2  Context representation", "raw · PCA · frozen STATE-SE, STACK,\nUCE, TranscriptFormer, scPRINT-2"),
        ("3  Effect transfer", "similarity-weighted Replogle or\nsource-line log-fold effects"),
        ("4  Raw-count generator", "multinomial (G0) · Poisson (G1)\n· negative binomial (G2)"),
        ("5  Submission file", "18,533 genes × 400 cells per\ntarget · vcc prep → .vcc"),
    ]
    for (title, body), y in zip(left, rows):
        _box(ax, lx, y, lw_, bh, title, body, BLUE_WASH, BLUE)
    for y_top, y_bot in zip(rows[:-1], rows[1:]):
        _arrow(ax, (lx + lw_ / 2, y_top), (lx + lw_ / 2, y_bot + bh), color=BLUE)

    rx, rw = 2.88, 1.78
    _box(ax, rx, 2.62, rw, 1.02, "Shadow benchmark",
         "public Perturb-seq data;\nJiang24 IFNG: BxPC3 held out\nGSE270828: rep3 held out\n(+ strict zero-shot variant)",
         ORANGE_WASH, ORANGE)
    _box(ax, rx, 0.28, rw, 0.9, "Scoring",
         "public proxy: cell-eval PDS\nexact 2026: cell-eval2 six\nmetrics, local anchors", ORANGE_WASH, ORANGE)
    # controls only -> inputs
    _arrow(ax, (rx, 3.36), (lx + lw_, 3.36), color=ORANGE)
    ax.text((rx + lx + lw_) / 2, 3.41, "controls\nonly", ha="center", va="bottom", fontsize=7.4, color=INK2,
            linespacing=1.1)
    # predictions -> scoring
    _arrow(ax, (lx + lw_, 0.56), (rx, 0.56), color=BLUE)
    ax.text((rx + lx + lw_) / 2, 0.61, "predic-\ntions", ha="center", va="bottom", fontsize=7.4, color=INK2,
            linespacing=1.1)
    # sealed truth path
    _arrow(ax, (rx + rw / 2, 2.62), (rx + rw / 2, 1.18), color=ORANGE, lw=1.4)
    ax.text(rx + rw / 2 + 0.07, 1.9, "sealed truth\n(scorer only)", ha="left", va="center",
            fontsize=7.6, color=INK2, linespacing=1.2)

    # ---- right: headline result, dot plot ----
    df = pd.read_csv(EXP / "E1_jiang24_bxpc3_proxy/results/bxpc3_cell_eval_proxy.csv")
    df["label"] = df.apply(lambda r: "no effect" if r.method == "no_effect" else
                           f"{r.representation.split('-')[0]} · {r.method.replace('gwps_', '')}", axis=1)
    df = df.sort_values("PDS", ascending=True).reset_index(drop=True)
    axr = fig.add_axes([0.745, 0.17, 0.235, 0.53])
    _clean_axes(axr, grid_axis="x")
    axr.grid(False)
    axr.spines["left"].set_visible(False)
    base = float(df.loc[df.method == "no_effect", "PDS"].iloc[0])
    axr.axvline(base, color=AXIS, linewidth=1.0, zorder=1)
    for i, r in df.iterrows():
        hi = r.label == "raw · nearest"
        axr.scatter(r.PDS, i, s=48 if hi else 34, color=BLUE if hi else MUTED, zorder=3,
                    edgecolors=SURFACE, linewidths=1.2)
        if hi:
            axr.text(r.PDS - 0.004, i + 0.42, f"{r.PDS:.4f}", ha="right", va="bottom",
                     fontsize=9, fontweight="bold", color=INK)
    axr.set_yticks(range(len(df)), df.label, fontsize=7.9)
    axr.set_xlim(0.485, 0.59); axr.set_xticks([0.50, 0.55])
    axr.set_ylim(-0.6, len(df) - 0.1)
    axr.set_xlabel("public-proxy PDS (higher = better)", fontsize=7.8)
    axr.text(base + 0.002, -0.55, "no-effect", fontsize=7.2, color=MUTED, ha="left", va="bottom")
    fig.text(0.745, 0.845, "Headline result", fontsize=8.8, fontweight="bold", color=INK, va="top")
    fig.text(0.745, 0.81, "Jiang24 IFNG, BxPC3 held out,\n56 targets × 4,017 genes", fontsize=7.6,
             color=INK2, va="top", linespacing=1.2)
    fig.text(0.02, 0.012, "Schematic drawn for this portfolio from src/vcc_baselines/. Values: docs/SHADOW_VCC.md\n"
             "(legacy cell-eval vcc-profile proxy on a public shadow split, not an official leaderboard score).",
             fontsize=6.9, color=MUTED, va="bottom", linespacing=1.25)
    fig.savefig(OUT / "hero.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. E1 dot plot (full table, both proxy PDS and local rank)
# --------------------------------------------------------------------------- #
def bxpc3_dotplot():
    proxy = pd.read_csv(EXP / "E1_jiang24_bxpc3_proxy/results/bxpc3_cell_eval_proxy.csv")
    local = pd.read_csv(EXP / "E1_jiang24_bxpc3_proxy/results/bxpc3_local_metrics.csv")
    df = proxy.merge(local, on=["representation", "method"])
    df["label"] = df.apply(lambda r: "no effect (raw controls)" if r.method == "no_effect" else
                           f"{r.representation} · {r.method}", axis=1)
    df = df.sort_values("PDS").reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.05), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    panels = [(a1, "PDS", "public-proxy PDS · higher = better", (0.485, 0.59)),
              (a2, "pdisc_norm_rank", "local discrimination rank · lower = better", (0.425, 0.52))]
    for ax, col, xlabel, xlim in panels:
        _clean_axes(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        base = float(df.loc[df.method == "no_effect", col].iloc[0])
        ax.axvline(base, color=AXIS, linewidth=1.0, zorder=1)
        for i, r in df.iterrows():
            hi = r.method == "gwps_nearest" and r.representation == "raw"
            ax.scatter(r[col], i, s=46 if hi else 32, color=BLUE if hi else MUTED, zorder=3,
                       edgecolors=SURFACE, linewidths=1.2)
            if hi:
                ax.text(r[col], i + 0.36, f"{r[col]:.4f}", ha="center", va="bottom", fontsize=8.6,
                        fontweight="bold", color=INK)
        ax.set_xlim(*xlim); ax.set_xlabel(xlabel, fontsize=8)
        ax.text(base, len(df) - 0.35, "no-effect", fontsize=7.2, color=MUTED, ha="center", va="bottom",
                bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    a1.set_yticks(range(len(df)), df.label, fontsize=8)
    a1.set_ylim(-0.6, len(df) + 0.1)
    fig.suptitle("Jiang24 IFNG → held-out BxPC3: raw nearest-context transfer leads; frozen encoders do not beat it",
                 x=0.02, y=0.985, ha="left", fontsize=9.6, fontweight="bold", color=INK)
    fig.text(0.02, 0.012, "Source: docs/SHADOW_VCC.md (56 CRISPRi targets, 4,017 genes, seed 2026). "
             "Public proxy on a shadow split, not a VCC 2026 leaderboard score.", fontsize=6.9, color=MUTED)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.2)
    fig.savefig(OUT / "bxpc3_proxy_pds.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. E2 small multiples: cells per perturbation trade-off
# --------------------------------------------------------------------------- #
def cells_tradeoff():
    df = pd.read_csv(EXP / "E2_six_metric_robustness/results/jiang24_cells_per_pert.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.9), gridspec_kw={"wspace": 0.42})
    panels = [("nmae", "LFC NMAE (scaled)"), ("jac", "Significant-gene Jaccard (scaled)"),
              ("overall", "Overall (mean of six)")]
    colors = {"scPRINT-2": BLUE, "PCA": ORANGE}
    xs = {25: 0, 50: 1, 100: 2}
    for ax, (col, title) in zip(axes, panels):
        _clean_axes(ax)
        for rep, sub in df.groupby("representation", sort=False):
            x = [xs[c] for c in sub.cells_per_pert]
            ax.plot(x, sub[col], color=colors[rep], linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter(x, sub[col], s=34, color=colors[rep], edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.set_xticks([0, 1, 2], ["25", "50", "100"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title, fontsize=8.6, color=INK, loc="left", pad=6)
        ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    axes[1].set_xlabel("predicted cells per perturbation", fontsize=8)
    # direct labels where the series separate: right end of the Overall panel
    axes[2].set_xlim(-0.3, 3.1)
    for rep, sub in df.groupby("representation", sort=False):
        y = float(sub.loc[sub.cells_per_pert == 100, "overall"].iloc[0])
        axes[2].text(2.15, y, rep, fontsize=7.6, color=INK2, va="center", ha="left")
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker="o", markersize=5,
                          markeredgecolor=SURFACE, label=k) for k, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("More cells help magnitude (NMAE) but expose wrong DE sets (Jaccard)",
                 x=0.02, y=0.985, ha="left", fontsize=9.6, fontweight="bold", color=INK)
    fig.text(0.02, 0.012, "Jiang24 IFNG, BxPC3 held out · exact cell-eval2 0.16.0 vcc2026 preset with dataset-local anchors "
             "(0 = generic baseline, 1 = replicate)\nτ 0.1, seed 2026 · higher is better on every panel · "
             "post-hoc sensitivity sweep, not a tuning result · source: docs/ko/03_experiments_and_metrics_ko.md §6.3",
             fontsize=6.7, color=MUTED, linespacing=1.25)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.25)
    fig.savefig(OUT / "cells_per_pert_tradeoff.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. E3 internal-CV magnitude sweep
# --------------------------------------------------------------------------- #
def scale_sweep():
    df = pd.read_csv(EXP / "E3_replogle_only_strict/results/internal_cv/magnitude_sweep.csv")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    _clean_axes(ax)
    series = {"k562_to_rpe1": ("K562 → RPE1", BLUE), "rpe1_to_k562": ("RPE1 → K562", ORANGE)}
    for key, (label, color) in series.items():
        sub = df[df.direction == key]
        ax.plot(sub.scale, sub.nmae, color=color, linewidth=2, zorder=2)
        ax.scatter(sub.scale, sub.nmae, s=34, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        last = sub.iloc[-1]
        ax.text(last.scale + 0.03, last.nmae, label, fontsize=8, color=INK2, va="center", ha="left")
    sel = df[(df.direction == "mean") & (df.scale == 0.25)].iloc[0]
    ax.axvline(0.25, color=AXIS, linewidth=1.0, zorder=1)
    ax.text(0.27, 1.62, f"selected scale 0.25\nmean NMAE {sel.nmae:.3f}", fontsize=8, color=INK, va="center")
    ax.set_xlim(0.2, 1.45); ax.set_xticks([0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_xlabel("global log-fold-change scale applied to the transferred Replogle effect", fontsize=8)
    ax.set_ylabel("normalised MAE (lower = better)", fontsize=8)
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker="o", markersize=5,
                          markeredgecolor=SURFACE, label=l) for l, c in series.values()]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Replogle-only internal CV: cross-line transfer wants a much smaller effect than measured",
                 x=0.02, y=0.975, ha="left", fontsize=9.6, fontweight="bold", color=INK)
    fig.text(0.02, 0.012, "Symmetric K562 ↔ RPE1 cross-source CV on 2,390 dual targets × 7,093 genes · "
             "source: experiments/E3_replogle_only_strict/results/internal_cv/magnitude_sweep.csv",
             fontsize=6.9, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2)
    fig.savefig(OUT / "replogle_scale_sweep.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    hero(); bxpc3_dotplot(); cells_tradeoff(); scale_sweep()
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")

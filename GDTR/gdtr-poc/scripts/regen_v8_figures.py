"""Regenerate the three v8 figures with publication-quality visualisation.

Outputs:
  /path/to/gDTR/results/figures_v3/fig_splice.{pdf,png}        (replaces v7 fig_splice)
  /path/to/gDTR/results/figures_v3/fig_funcshallow.{pdf,png}   (NEW §3.2 headline)
  /path/to/gDTR/results/figures_v3/fig_disruption.{pdf,png}    (replaces v7 fig_disruption)

Design (per user instruction "제일 중요한 내용을, 제일 가시적으로"):
  fig_splice:        2-panel — (a) universality bar plot of mean c per context;
                                (b) canonical / non-canonical breakdown bar plot.
  fig_funcshallow:   2-panel — (a) c-distribution gradient (shuffled-null grey,
                                cCRE-ELS red, splice-donor blue) overlaid KDE;
                                (b) forest plot of Cohen's d for
                                cCRE-ELS / eQTL / GWAS / splice donor (reference).
  fig_disruption:    2-row composite — top: horizontal boxplot of P/LP
                                argmax_layer per class (5 SNV + frameshift indel);
                                bottom: 5 representative ΔD_cos traces (one per class).

Usage:
  python3 scripts/regen_v8_figures.py [--smoke] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR"))
OUT_DIR = GDTR_ROOT / "results" / "figures_v3"
PHASE = "regen_v8_figures"

LOG = logging.getLogger(PHASE)


def setup_log():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def write_status(state: str, extra: dict | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"step": PHASE, "status": state}
    if extra:
        payload.update(extra)
    (OUT_DIR / "_status.json").write_text(json.dumps(payload, indent=2, default=str))


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
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# =====================================================================
# fig_splice — universality + canonical breakdown
# =====================================================================
def fig_splice():
    """2-panel splice figure: universality (a) + canonical breakdown (b)."""
    p1a_csv = GDTR_ROOT / "results" / "p1a" / "calib_val_table.csv"
    p1b_json = GDTR_ROOT / "results" / "p1b" / "splice_canonical_compare.json"

    # --- Panel A: per-context mean c (chr17 + chr22 combined) ---
    rows = []
    with p1a_csv.open() as f:
        header = f.readline().strip().split(",")
        for ln in f:
            d = dict(zip(header, ln.strip().split(",")))
            rows.append({
                "chrom": d["chrom"],
                "context": d["context"],
                "n": int(d["n"]),
                "mean_c": float(d["mean_c"]),
            })
    # Pool chr17 + chr22 by weighted mean for panel A summary
    contexts = ["splice_donor", "splice_acceptor", "3utr", "intron",
                "intergenic", "coding_exon", "5utr"]
    pooled = {}
    for c in contexts:
        sel = [r for r in rows if r["context"] == c]
        n_total = sum(r["n"] for r in sel)
        m_total = sum(r["mean_c"] * r["n"] for r in sel) / n_total
        pooled[c] = {"mean": m_total, "n": n_total}
    intron_baseline = pooled["intron"]["mean"]

    # --- Panel B: canonical / non-canonical breakdown ---
    spc = json.loads(p1b_json.read_text())["per_class"]
    classes_b = ["non_canonical_donor", "non_canonical_acceptor",
                 "canonical_GT_AG_donor", "canonical_GT_AG_acceptor",
                 "canonical_GC_AG_donor", "canonical_GC_AG_acceptor"]
    canon_groups = {
        "non-canonical": ["non_canonical_donor", "non_canonical_acceptor"],
        "GT-AG (canonical)": ["canonical_GT_AG_donor", "canonical_GT_AG_acceptor"],
        "GC-AG (canonical)": ["canonical_GC_AG_donor", "canonical_GC_AG_acceptor"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                              gridspec_kw={"width_ratios": [1.0, 1.2]})

    # --- Panel A ---
    ax = axes[0]
    pretty = {"splice_donor": "splice donor",
              "splice_acceptor": "splice acceptor",
              "3utr": "3' UTR",
              "intron": "intron (baseline)",
              "intergenic": "intergenic",
              "coding_exon": "coding exon",
              "5utr": "5' UTR"}
    order = sorted(contexts, key=lambda c: pooled[c]["mean"])
    means = [pooled[c]["mean"] for c in order]
    colors = []
    for c in order:
        if "splice" in c:
            colors.append("#1f77b4")  # blue
        elif c == "intron":
            colors.append("#7f7f7f")  # grey
        else:
            colors.append("#bbbbbb")  # light grey

    y = np.arange(len(order))
    bars = ax.barh(y, means, color=colors, height=0.7, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[c] for c in order])
    ax.axvline(intron_baseline, color="red", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"intron baseline c̄={intron_baseline:.2f}")
    ax.set_xlim(24.5, 30.0)
    ax.set_xlabel("mean settling depth $c$  (pooled chr17+chr22)")
    ax.set_title("(a) Splice universality: every splice context $<$ intron baseline",
                 loc="left")
    for bi, b in enumerate(bars):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                f"{means[bi]:.2f}", va="center", fontsize=7)
    ax.legend(loc="lower right", frameon=False)

    # --- Panel B ---
    ax = axes[1]
    chr_labels = ["chr17", "chr22"]
    group_labels = list(canon_groups.keys())
    width = 0.36
    x = np.arange(len(group_labels))
    chr17_means, chr22_means = [], []
    for g in group_labels:
        keys = canon_groups[g]
        c17 = sum(spc["chr17"][k]["mean_c"] * spc["chr17"][k]["n"] for k in keys)
        n17 = sum(spc["chr17"][k]["n"] for k in keys)
        c22 = sum(spc["chr22"][k]["mean_c"] * spc["chr22"][k]["n"] for k in keys)
        n22 = sum(spc["chr22"][k]["n"] for k in keys)
        chr17_means.append(c17 / max(n17, 1))
        chr22_means.append(c22 / max(n22, 1))
    chr17_intron = spc["chr17"]["intron"]["mean_c"]
    chr22_intron = spc["chr22"]["intron"]["mean_c"]
    intron_pooled = (chr17_intron + chr22_intron) / 2

    color_map = {"non-canonical": "#d62728",          # red
                 "GT-AG (canonical)": "#1f77b4",       # blue
                 "GC-AG (canonical)": "#3a5fab"}       # darker blue
    bars1 = ax.bar(x - width/2, chr17_means, width, label="chr17",
                    color=[color_map[g] for g in group_labels],
                    edgecolor="black", linewidth=0.4, hatch="")
    bars2 = ax.bar(x + width/2, chr22_means, width, label="chr22",
                    color=[color_map[g] for g in group_labels],
                    edgecolor="black", linewidth=0.4, hatch="//", alpha=0.85)
    ax.axhline(intron_pooled, color="grey", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"intron c̄={intron_pooled:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=10, ha="right")
    ax.set_ylim(24.0, 28.5)
    ax.set_ylabel("mean settling depth $c$")
    ax.set_title("(b) canonical vs non-canonical splice motifs", loc="left")
    # value annotations
    for bset in (bars1, bars2):
        for b in bset:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                    ha="center", fontsize=7)
    legend_h = [
        mpatches.Patch(color=color_map["non-canonical"], label="non-canonical"),
        mpatches.Patch(color=color_map["GT-AG (canonical)"], label="GT-AG canonical"),
        mpatches.Patch(color=color_map["GC-AG (canonical)"], label="GC-AG canonical"),
        Line2D([0], [0], color="grey", linestyle="--", label=f"intron c̄={intron_pooled:.2f}"),
    ]
    ax.legend(handles=legend_h, loc="upper left", frameon=False, ncol=1)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig_splice.pdf"
    out_png = OUT_DIR / "fig_splice.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)


# =====================================================================
# fig_funcshallow — NEW §3.2 headline
# =====================================================================
def fig_funcshallow(smoke: bool = False):
    """2-panel functional-shallowness figure: KDE gradient (a) + forest plot (b)."""
    p3b1_json = GDTR_ROOT / "results" / "p3b1" / "p3b1_func_pos.json"
    p1a_csv = GDTR_ROOT / "results" / "p1a" / "calib_val_table.csv"
    chr22_c_npy = GDTR_ROOT / "results" / "phase1.6_sub" / "chr22_position_c.npy"
    chr22_labels_npy = GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"
    splice_class_npy = GDTR_ROOT / "data" / "annotation" / "chr22_splice_class_labels.npy"
    cCRE_bed = GDTR_ROOT / "data" / "external" / "ccre_els_chr22.bed"

    # --- Build c samples for the three groups ---
    LOG.info("loading chr22 c array (~200 MB)")
    c_arr = np.load(chr22_c_npy)
    if c_arr.dtype != np.float32:
        c_arr = c_arr.astype(np.float32)
    valid_mask = ~np.isnan(c_arr)

    # cCRE-ELS positions: read BED + flag
    LOG.info("building cCRE-ELS mask from BED")
    cCRE_mask = np.zeros(c_arr.shape[0], dtype=bool)
    with cCRE_bed.open() as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            try:
                a = int(parts[1]); b = int(parts[2])
            except ValueError:
                continue
            if 0 <= a < cCRE_mask.size and 0 <= b <= cCRE_mask.size:
                cCRE_mask[a:b] = True
    cCRE_pos = np.where(cCRE_mask & valid_mask)[0]

    # Splice donor positions (canonical GT-AG donor, code=1 in splice_class_labels)
    LOG.info("loading splice class labels")
    sc = np.load(splice_class_npy)
    splice_donor_pos = np.where(((sc == 1) | (sc == 3)) & valid_mask)[0]  # both GT-AG and GC-AG donors

    # Shuffled null: random subset of all valid positions (roughly chr22 background)
    rng = np.random.default_rng(42)
    bg_pos = np.where(valid_mask)[0]
    n_bg_sample = min(2_000_000, len(bg_pos))
    if smoke:
        n_bg_sample = min(50_000, len(bg_pos))
    bg_idx = rng.choice(bg_pos, size=n_bg_sample, replace=False)
    if smoke and len(cCRE_pos) > 50_000:
        cCRE_pos = rng.choice(cCRE_pos, size=50_000, replace=False)

    c_bg = c_arr[bg_idx]
    c_cCRE = c_arr[cCRE_pos]
    c_splice = c_arr[splice_donor_pos]
    LOG.info("samples: bg=%d cCRE=%d splice=%d", len(c_bg), len(c_cCRE), len(c_splice))

    # --- Panel A: KDE / hist overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8),
                              gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    # Use CDF instead of PDF — the c=31 saturation peak dominates a histogram
    # so the shallowness shift among the three groups is invisible. With a CDF
    # we read directly: at any threshold c, the fraction of positions with
    # c <= that threshold; the curve sitting higher is shallower.
    grid = np.linspace(0, 31, 250)
    for arr, lab, color in [
        (c_bg, "chr22 background", "#7f7f7f"),
        (c_cCRE, "cCRE-ELS", "#d62728"),
        (c_splice, "splice donor", "#1f77b4"),
    ]:
        sorted_arr = np.sort(arr)
        # Empirical CDF at the grid
        cdf = np.searchsorted(sorted_arr, grid, side="right") / len(sorted_arr)
        ax.plot(grid, cdf, color=color, linewidth=2.4,
                label=f"{lab}  (c̄={arr.mean():.2f}, n={len(arr):,})")
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("settling depth $c$  (1-indexed layer)")
    ax.set_ylabel("cumulative fraction with $c \\leq x$")
    ax.set_title(
        "(a) Shallowness gradient (CDF): higher curve $=$ shallower distribution",
        loc="left",
    )
    ax.legend(loc="lower right", frameon=False)

    # --- Panel B: forest plot of Cohen's d ---
    ax = axes[1]
    # Compute Cohen's d using per-position std as the natural scale
    bg_std = float(np.std(c_bg, ddof=1))
    func = json.loads(p3b1_json.read_text())["per_dataset"]
    splice_d = (c_splice.mean() - c_bg.mean()) / bg_std
    rows_d = [
        ("splice donor (§3.1)", splice_d, "#1f77b4"),
        ("ENCODE cCRE-ELS",
         (func["cCRE_ELS"]["mean_c_func"] - func["cCRE_ELS"]["mean_c_shuffled"]) / bg_std,
         "#d62728"),
        ("GTEx eQTL chr22",
         (func["GTEx_eQTL"]["mean_c_func"] - func["GTEx_eQTL"]["mean_c_shuffled"]) / bg_std,
         "#ff7f0e"),
        ("GWAS Catalog chr22",
         (func["GWAS"]["mean_c_func"] - func["GWAS"]["mean_c_shuffled"]) / bg_std,
         "#ff7f0e"),
    ]
    y = np.arange(len(rows_d))
    for yi, (label, d, color) in enumerate(rows_d):
        ax.scatter(d, yi, s=110, color=color, zorder=3,
                   edgecolor="black", linewidth=0.7)
        ax.plot([0, d], [yi, yi], color=color, linewidth=2.5, zorder=2, alpha=0.6)
        # place the d label BELOW the marker (after invert_yaxis, +offset = visually down)
        ax.text(d - 0.015, yi + 0.30, f"$d={d:.3f}$", fontsize=8.5,
                ha="right", va="center", color=color, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows_d], fontsize=9)
    ax.set_xlabel("Cohen's $d$  (vs chr22 background; negative $=$ shallower)")
    ax.set_title("(b) Effect size: cCRE-ELS aligns with splice axis", loc="left")
    xmin = min(r[1] for r in rows_d) - 0.10
    ax.set_xlim(xmin, 0.05)
    ax.invert_yaxis()
    fig.tight_layout()
    out_pdf = OUT_DIR / "fig_funcshallow.pdf"
    out_png = OUT_DIR / "fig_funcshallow.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)
    return {
        "splice_d": float(splice_d),
        "cCRE_d": rows_d[1][1],
        "eQTL_d": rows_d[2][1],
        "GWAS_d": rows_d[3][1],
        "bg_std": bg_std,
        "n_bg": int(len(c_bg)),
        "n_cCRE": int(len(c_cCRE)),
        "n_splice": int(len(c_splice)),
    }


# =====================================================================
# fig_disruption — 2-row composite (boxplot + 5 traces)
# =====================================================================
def fig_disruption():
    """2-row figure: P/LP class boxplot (top) + 5 representative traces (bottom)."""
    p2_csv = GDTR_ROOT / "results" / "p2" / "p2_snv_per_class.csv"
    p2_indel_csv = GDTR_ROOT / "results" / "p2_indel" / "p2_indel_per_class.csv"
    p2_indel_features = GDTR_ROOT / "results" / "p2_indel" / "variants_features_indel.csv"
    p2_snv_features = GDTR_ROOT / "results" / "p2" / "variants_features_classed.csv"
    case_json = GDTR_ROOT / "results" / "p2_case" / "case_studies_v8.json"

    # --- Panel top: per-class P/LP argmax_layer boxplot ---
    # Build per-class argmax arrays from features (P/LP only)
    def _read_features_argmax_by_class(path):
        d = {}
        with path.open() as f:
            header = f.readline().strip().split(",")
            cat_i = header.index("category")
            mc_i = header.index("mc_class")
            argmax_i = header.index("argmax_layer")
            for ln in f:
                parts = ln.rstrip().split(",")
                if len(parts) < len(header):
                    continue
                cat = parts[cat_i]
                if cat != "P_LP":
                    continue
                mc = parts[mc_i]
                try:
                    al = int(parts[argmax_i])
                except ValueError:
                    continue
                d.setdefault(mc, []).append(al)
        return d

    snv_argmax = _read_features_argmax_by_class(p2_snv_features)
    indel_argmax = _read_features_argmax_by_class(p2_indel_features)

    classes_order = ["intron", "frameshift", "nonsense", "missense",
                     "canonical_splice", "synonymous"]
    pretty_class = {
        "intron": "intron",
        "frameshift": "frameshift\n(indel forward)",
        "nonsense": "nonsense",
        "missense": "missense",
        "canonical_splice": "canonical splice",
        "synonymous": "synonymous",
    }
    # Merge SNV and indel sources; frameshift only exists in indel
    arg_per_class = {}
    for c in classes_order:
        merged = list(snv_argmax.get(c, []))
        merged.extend(indel_argmax.get(c, []))
        arg_per_class[c] = merged

    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.7, 1.0], hspace=0.55, wspace=0.30)

    # Top: horizontal boxplot spanning all 5 columns
    ax_top = fig.add_subplot(gs[0, :])
    box_data = [arg_per_class[c] for c in classes_order]
    medians = [int(np.median(d)) if d else float("nan") for d in box_data]
    n_per = [len(d) for d in box_data]
    ax_top.boxplot(box_data, vert=False, showfliers=False,
                    tick_labels=[f"{pretty_class[c]}  (n={n_per[i]})"
                                 for i, c in enumerate(classes_order)],
                    patch_artist=True,
                    boxprops=dict(facecolor="#cfd8dc", edgecolor="black"),
                    medianprops=dict(color="#d62728", linewidth=2.0))
    for i, m in enumerate(medians):
        ax_top.text(m + 0.4, i + 1, f"L{m}", va="center", fontsize=9, color="#d62728",
                     fontweight="bold")
    ax_top.set_xlim(0, 32)
    ax_top.set_xlabel("argmax layer (1-indexed) — where $|\\Delta D_{\\cos}|$ peaks per variant")
    ax_top.set_title(
        "Per-class disruption layer (P/LP variants, 15 cancer-associated genes; "
        "Kruskal–Wallis 5-way $p=7.1\\times 10^{-10}$)",
        loc="left", fontsize=10
    )

    # Bottom: 5 representative traces
    cs = json.loads(case_json.read_text())
    rep_classes = ["missense", "nonsense", "canonical_splice", "synonymous", "frameshift"]
    rep_titles = {"missense": "missense", "nonsense": "nonsense",
                   "canonical_splice": "canonical splice",
                   "synonymous": "synonymous", "frameshift": "frameshift"}
    for col, klass in enumerate(rep_classes):
        ax = fig.add_subplot(gs[1, col])
        # find class entry
        entry = next((v for v in cs["variants"] if v["mc_class"] == klass), None)
        if entry is None:
            ax.set_visible(False)
            continue
        median_trace = np.array(entry.get("median_trace_dD_cos", []), dtype=np.float64)
        pv = entry["picked_variant"]
        rep_trace = np.array(pv["dD_cos_per_layer"], dtype=np.float64)
        n_class = entry["n_class"]
        argmax_layer = pv["argmax_layer"]
        gene = pv["gene"]
        chrom = pv["chrom"]; pos = pv["pos"]; ref = pv["ref"]; alt = pv["alt"]
        x = np.arange(1, len(rep_trace) + 1)
        if median_trace.size == len(rep_trace):
            ax.plot(x, median_trace, color="#7f7f7f", linewidth=2.0,
                    label=f"class median (n={n_class})", alpha=0.85)
        ax.plot(x, rep_trace, color="#d62728", linewidth=1.6,
                label=f"L{argmax_layer} representative")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(argmax_layer, color="#d62728", linewidth=0.7, alpha=0.5,
                    linestyle=":")
        ax.set_xlim(1, 32)
        ax.set_xlabel("layer")
        if col == 0:
            ax.set_ylabel("$\\Delta D_{\\cos}$ (alt − ref)")
        ax.set_title(f"{rep_titles[klass]}\n{gene} chr{chrom}:{pos} {ref}>{alt}",
                      fontsize=8.5)
        ax.legend(loc="upper left", frameon=False, fontsize=7)

    out_pdf = OUT_DIR / "fig_disruption.pdf"
    out_png = OUT_DIR / "fig_disruption.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)


def main():
    setup_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    write_status("RUNNING")
    stylize()
    try:
        LOG.info("=== fig_splice ===")
        fig_splice()
        LOG.info("=== fig_funcshallow ===")
        meta = fig_funcshallow(smoke=args.smoke)
        LOG.info("=== fig_disruption ===")
        fig_disruption()
        write_status("PASS", {"wall_sec": time.time() - t0,
                                "fig_funcshallow_meta": meta})
        LOG.info("done in %.1fs", time.time() - t0)
    except Exception as e:
        import traceback
        write_status("FAIL", {"error": str(e), "traceback": traceback.format_exc()})
        LOG.exception("FAIL: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

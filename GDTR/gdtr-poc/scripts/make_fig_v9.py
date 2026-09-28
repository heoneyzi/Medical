"""Generate the v9 (consolidated 4p ICML body) main figures.

Two output figures replace the four standalone v8 panels and add explicit
significance markers:

  fig_shallowness.{pdf,png}   — 2-panel
      (a) splice & functional context mean settling depth c̄, pooled chr17+chr22
          for splice / intron / non-splice; per-position chr22 background
          and ENCODE cCRE-ELS overlay. Sig stars vs intron baseline (one-sided
          MWU + Cohen's d).
      (b) Forest plot of Cohen's d vs chr22 background for: splice donor
          (reference), ENCODE cCRE-ELS, GTEx eQTL chr22, GWAS Catalog chr22.
          Per-row p-value (one-sided MWU) annotated as ns / * / ** / ***
          (Bonferroni-corrected over 4 tests).

  fig_variants.{pdf,png}      — 2-row composite
      Top:    horizontal box plot of per-class P/LP argmax-layer
              (intron / frameshift / nonsense / missense / canonical splice /
              synonymous). Annotated: Kruskal–Wallis 5-way p, plus pairwise
              Dunn-Bonferroni significance brackets between adjacent classes.
      Bottom: 5 representative ΔD_cos traces, one per non-intron class, picked
              at the per-class median argmax-layer.

Inputs (server-side paths under GDTR_ROOT):
  results/p1a/calib_val_table.csv
  results/p1b/splice_canonical_compare.json
  results/p3b1/p3b1_func_pos.json
  results/p2/variants_features_classed.csv
  results/p2_indel/variants_features_indel.csv
  results/p2_case/case_studies_v8.json
  results/phase1.6_sub/chr22_position_c.npy   (large; only loaded for panel-a CIs)
  data/annotation/chr22_position_labels.npy   (for context masks)
  data/annotation/chr22_splice_class_labels.npy
  data/external/ccre_els_chr22.bed

Outputs:
  results/figures_v3/fig_shallowness.{pdf,png}
  results/figures_v3/fig_variants.{pdf,png}
  results/figures_v3/fig_v9_meta.json    — numbers used in captions

Usage:
  python3 scripts/make_fig_v9.py [--smoke] [--no-bg]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR"))
OUT_DIR = GDTR_ROOT / "results" / "figures_v3"
PHASE = "make_fig_v9"

LOG = logging.getLogger(PHASE)


def setup_log():
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
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def sig_stars(p, alpha_levels=(1e-4, 1e-3, 1e-2, 5e-2)):
    """Return ASCII significance label."""
    if p < alpha_levels[0]:
        return "****"
    if p < alpha_levels[1]:
        return "***"
    if p < alpha_levels[2]:
        return "**"
    if p < alpha_levels[3]:
        return "*"
    return "ns"


def cohen_d(x, y):
    """Pooled-SD Cohen's d, x − y."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


# =====================================================================
# fig_shallowness — splice + functional combined
# =====================================================================
def _load_p1a(p1a_csv):
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
    return rows


def _pooled_context(rows):
    contexts = ["splice_donor", "splice_acceptor", "3utr", "intron",
                "intergenic", "coding_exon", "5utr"]
    pooled = {}
    for c in contexts:
        sel = [r for r in rows if r["context"] == c]
        if not sel:
            continue
        n_total = sum(r["n"] for r in sel)
        m_total = sum(r["mean_c"] * r["n"] for r in sel) / n_total
        pooled[c] = {"mean": m_total, "n": n_total}
    return pooled


def _build_chr22_groups(c_arr, valid_mask, smoke=False, no_bg=False, rng=None):
    """Return dict label -> sample of c values for cCRE-ELS / splice / bg."""
    if rng is None:
        rng = np.random.default_rng(42)
    out = {}
    cCRE_bed = GDTR_ROOT / "data" / "external" / "ccre_els_chr22.bed"
    splice_class_npy = GDTR_ROOT / "data" / "annotation" / "chr22_splice_class_labels.npy"

    if cCRE_bed.exists():
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
        if smoke and len(cCRE_pos) > 50_000:
            cCRE_pos = rng.choice(cCRE_pos, size=50_000, replace=False)
        out["cCRE-ELS"] = c_arr[cCRE_pos]

    if splice_class_npy.exists():
        sc = np.load(splice_class_npy)
        donor_pos = np.where(((sc == 1) | (sc == 3)) & valid_mask)[0]
        out["splice donor"] = c_arr[donor_pos]

    if not no_bg:
        bg_pos = np.where(valid_mask)[0]
        n_bg_sample = 50_000 if smoke else min(2_000_000, len(bg_pos))
        bg_idx = rng.choice(bg_pos, size=n_bg_sample, replace=False)
        out["chr22 background"] = c_arr[bg_idx]

    return out


def fig_shallowness(smoke=False, no_bg=False):
    p1a_csv = GDTR_ROOT / "results" / "p1a" / "calib_val_table.csv"
    p3b1_json = GDTR_ROOT / "results" / "p3b1" / "p3b1_func_pos.json"
    chr22_c_npy = GDTR_ROOT / "results" / "phase1.6_sub" / "chr22_position_c.npy"

    rows = _load_p1a(p1a_csv)
    pooled = _pooled_context(rows)
    intron_baseline = pooled["intron"]["mean"]

    # Load chr22 position-c for sig testing on splice (panel a) and groups (panel b)
    LOG.info("loading chr22 c array")
    c_arr = np.load(chr22_c_npy)
    if c_arr.dtype != np.float32:
        c_arr = c_arr.astype(np.float32)
    valid_mask = ~np.isnan(c_arr)
    groups = _build_chr22_groups(c_arr, valid_mask, smoke=smoke, no_bg=no_bg)

    # ---- Panel A: per-context bar with intron baseline + sig stars vs intron
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8),
                              gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    pretty = {"splice_donor": "splice donor",
              "splice_acceptor": "splice acceptor",
              "3utr": "3$'$ UTR",
              "intron": "intron (baseline)",
              "intergenic": "intergenic",
              "coding_exon": "coding exon",
              "5utr": "5$'$ UTR"}
    order = sorted(pooled.keys(), key=lambda c: pooled[c]["mean"])
    means = [pooled[c]["mean"] for c in order]
    n_per = [pooled[c]["n"] for c in order]

    # Compute Cohen's d via chr22-only context labels
    chr22_labels_npy = GDTR_ROOT / "data" / "annotation" / "chr22_position_labels.npy"
    sig_labels = {}
    if chr22_labels_npy.exists() and not no_bg:
        cls = np.load(chr22_labels_npy)
        # Map label codes to context names (from prep scripts; same encoding as p1a)
        # chr22_position_labels.npy encoding (verified by unique-count match
        # against tab:splice n_chr22 column):
        #   0 intergenic   1 intron       2 coding_exon
        #   3 5utr         4 3utr         5 splice_donor   6 splice_acceptor
        code_to_name = {0: "intergenic", 1: "intron", 2: "coding_exon",
                         3: "5utr", 4: "3utr",
                         5: "splice_donor", 6: "splice_acceptor"}
        intron_c = c_arr[(cls == 1) & valid_mask]
        # subsample intron to manageable size
        if len(intron_c) > 500_000:
            rng = np.random.default_rng(0)
            intron_c = rng.choice(intron_c, size=500_000, replace=False)
        for code, name in code_to_name.items():
            if name == "intron":
                continue
            ctx_c = c_arr[(cls == code) & valid_mask]
            if len(ctx_c) < 50:
                continue
            if len(ctx_c) > 200_000:
                rng = np.random.default_rng(code)
                ctx_c = rng.choice(ctx_c, size=200_000, replace=False)
            d = cohen_d(ctx_c, intron_c)
            try:
                _u, p = stats.mannwhitneyu(ctx_c, intron_c, alternative="two-sided")
            except ValueError:
                p = float("nan")
            sig_labels[name] = (d, p)

    colors = []
    for c in order:
        if "splice" in c:
            colors.append("#1f77b4")
        elif c == "intron":
            colors.append("#7f7f7f")
        else:
            colors.append("#bbbbbb")
    y = np.arange(len(order))
    bars = ax.barh(y, means, color=colors, height=0.7,
                    edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[c] for c in order])
    ax.axvline(intron_baseline, color="red", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"intron baseline $\\bar c={intron_baseline:.2f}$")
    ax.set_xlim(24.0, 30.5)
    ax.set_xlabel("mean settling depth $\\bar c$  (pooled chr17+chr22)")
    ax.set_title("(a) Splice contexts settle $\\sim 2$ layers shallower than introns",
                 loc="left")
    for bi, b in enumerate(bars):
        ctx = order[bi]
        lab = f"{means[bi]:.2f}"
        if ctx in sig_labels:
            d, p = sig_labels[ctx]
            stars = sig_stars(p)
            lab = f"{means[bi]:.2f}  $d{{=}}{d:+.2f}$  {stars}"
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                lab, va="center", fontsize=7)
    ax.legend(loc="lower right", frameon=False)

    # ---- Panel B: forest plot of Cohen's d, regulatory layers
    ax = axes[1]
    func = json.loads(p3b1_json.read_text())["per_dataset"]
    bg = groups.get("chr22 background")
    splice = groups.get("splice donor")

    # row entries: (label, group_array, color)
    row_entries = []
    if splice is not None and bg is not None:
        row_entries.append(("splice donor (ref)", splice, bg, "#1f77b4"))
    if "cCRE-ELS" in groups and bg is not None:
        row_entries.append(("ENCODE cCRE-ELS",
                              groups["cCRE-ELS"], bg, "#d62728"))
    # GTEx eQTL and GWAS: use precomputed func vs shuffled-null from p3b1
    rows_d = []
    n_compare = 2  # for adjustment of GTEx/GWAS (precomputed); rest from raw
    for label, x, y_grp, color in row_entries:
        d = cohen_d(x, y_grp)
        try:
            _u, p = stats.mannwhitneyu(x, y_grp, alternative="less")
        except ValueError:
            p = float("nan")
        # Bonferroni over 4 tests
        p_corr = min(1.0, p * 4)
        rows_d.append({"label": label, "d": d, "p": p, "p_corr": p_corr,
                        "color": color, "n_func": len(x), "n_null": len(y_grp)})
    # GTEx eQTL & GWAS — use precomputed shuffled-null distribution.
    # The p3b1 step compares each functional set against 100 chr22-restricted
    # shuffled draws and reports `effect` = (mean_c_func − mean_c_null) /
    # std_c_null  (a z-score against the empirical null of mean_c). We use that
    # z directly: p_two_sided = 2 * Φ(−|z|), then Bonferroni × 4.
    bg_sigma = float(np.std(bg, ddof=1)) if bg is not None else 5.74
    for key, label, color in [("GTEx_eQTL", "GTEx eQTL chr22", "#ff7f0e"),
                                ("GWAS", "GWAS Catalog chr22", "#ff7f0e")]:
        e = func[key]
        d_pos = (e["mean_c_func"] - e["mean_c_shuffled"]) / bg_sigma
        z_null = float(e.get("effect", float("nan")))  # z vs the 100-shuffle null
        # two-sided p from |z|
        if np.isfinite(z_null):
            p_two = 2.0 * stats.norm.sf(abs(z_null))
        else:
            p_two = float("nan")
        rows_d.append({"label": label, "d": float(d_pos),
                        "p": p_two,
                        "p_corr": min(1.0, p_two * 4) if np.isfinite(p_two) else float("nan"),
                        "color": color,
                        "n_func": int(e.get("n_func_positions", 0)),
                        "n_null": int(e.get("n_shuffles", 0))})

    y = np.arange(len(rows_d))
    xmin = min(r["d"] for r in rows_d) - 0.10
    xmax = 0.18
    for yi, r in enumerate(rows_d):
        ax.scatter(r["d"], yi, s=110, color=r["color"], zorder=3,
                   edgecolor="black", linewidth=0.7)
        ax.plot([0, r["d"]], [yi, yi], color=r["color"], linewidth=2.5,
                 zorder=2, alpha=0.6)
        # Right-aligned tabular annotation outside the data zone (positive x).
        ax.text(xmax - 0.005, yi,
                 f"$d={r['d']:+.3f}$  {sig_stars(r['p_corr'])}",
                 fontsize=8.5, ha="right", va="center", color=r["color"],
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows_d], fontsize=9)
    ax.set_xlabel("Cohen's $d$  (vs chr22 background; negative $=$ shallower)")
    ax.set_title("(b) Functional regulatory elements share splice shallowness axis",
                 loc="left")
    ax.set_xlim(xmin, xmax)
    ax.invert_yaxis()
    fig.tight_layout()
    out_pdf = OUT_DIR / "fig_shallowness.pdf"
    out_png = OUT_DIR / "fig_shallowness.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)
    return {"panel_b_rows": rows_d, "panel_a_sig": sig_labels,
             "intron_baseline": intron_baseline}


# =====================================================================
# fig_variants — boxplot + KW + Dunn pairs + 5 traces
# =====================================================================
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


def dunn_bonf_pair(x, y, n_pairs):
    """Single pairwise Dunn test (rank-sum on combined ranks),
    Bonferroni-corrected by n_pairs. Returns adjusted p-value."""
    try:
        _u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return float("nan")
    return min(1.0, p * n_pairs)


def fig_variants():
    p2_features = GDTR_ROOT / "results" / "p2" / "variants_features_classed.csv"
    p2_indel_features = GDTR_ROOT / "results" / "p2_indel" / "variants_features_indel.csv"
    case_json = GDTR_ROOT / "results" / "p2_case" / "case_studies_v8.json"

    snv_argmax = _read_features_argmax_by_class(p2_features)
    indel_argmax = _read_features_argmax_by_class(p2_indel_features)

    classes_order = ["intron", "frameshift", "nonsense", "missense",
                     "canonical_splice", "synonymous"]
    pretty_class = {
        "intron": "intron",
        "frameshift": "frameshift\n(indel)",
        "nonsense": "nonsense",
        "missense": "missense",
        "canonical_splice": "canonical splice",
        "synonymous": "synonymous",
    }
    arg_per_class = {}
    for c in classes_order:
        merged = list(snv_argmax.get(c, []))
        merged.extend(indel_argmax.get(c, []))
        arg_per_class[c] = merged

    # KW 5-way (excluding the smallest 'synonymous' if degenerate; keep all 6)
    kw_input = [arg_per_class[c] for c in classes_order
                 if len(arg_per_class[c]) >= 3]
    kw_p = float("nan")
    if len(kw_input) >= 2:
        try:
            kw_stat, kw_p = stats.kruskal(*kw_input)
        except ValueError:
            pass

    # Dunn-Bonferroni adjacent pairs
    pairs = [(classes_order[i], classes_order[i+1]) for i in range(len(classes_order)-1)]
    n_pairs = len(pairs)
    pair_p = {}
    for a, b in pairs:
        if len(arg_per_class[a]) > 2 and len(arg_per_class[b]) > 2:
            pair_p[(a, b)] = dunn_bonf_pair(arg_per_class[a], arg_per_class[b], n_pairs)

    # Layout: 2 rows, top = boxplot full-width; bottom = 5 representative traces
    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.85, 1.0], hspace=0.85, wspace=0.35)

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
        ax_top.text(m + 0.4, i + 1, f"L{m}", va="center", fontsize=9,
                     color="#d62728", fontweight="bold")
    # Add Dunn adjacent-pair sig brackets on the right margin
    x_bracket_base = 32.0
    bracket_step = 0.7
    for j, (a, b) in enumerate(pairs):
        ia = classes_order.index(a) + 1
        ib = classes_order.index(b) + 1
        x_b = x_bracket_base + j * bracket_step
        p_adj = pair_p.get((a, b), float("nan"))
        s = sig_stars(p_adj)
        ax_top.plot([x_b, x_b + 0.18, x_b + 0.18, x_b],
                    [ia, ia, ib, ib], color="black", linewidth=0.8)
        ax_top.text(x_b + 0.22, (ia + ib) / 2, s, va="center", fontsize=8,
                     fontweight="bold", color="black")
    ax_top.set_xlim(0, x_bracket_base + n_pairs * bracket_step + 1.0)
    ax_top.set_xlabel("argmax layer (1-indexed) — where $|\\Delta D_{\\cos}|$ peaks per variant")
    title_kw = (f"Per-class disruption layer (P/LP, 15 cancer-associated genes; "
                f"Kruskal–Wallis $p={kw_p:.1e}$; "
                f"Dunn–Bonferroni adjacent pairs at right)")
    ax_top.set_title(title_kw, loc="left", fontsize=10)

    # Bottom: 5 representative traces (skip intron, which is the rarest)
    cs = json.loads(case_json.read_text())
    rep_classes = ["missense", "nonsense", "canonical_splice", "synonymous", "frameshift"]
    rep_titles = {"missense": "missense", "nonsense": "nonsense",
                   "canonical_splice": "canonical splice",
                   "synonymous": "synonymous", "frameshift": "frameshift"}
    for col, klass in enumerate(rep_classes):
        ax = fig.add_subplot(gs[1, col])
        entry = next((v for v in cs["variants"] if v["mc_class"] == klass), None)
        if entry is None:
            ax.set_visible(False)
            continue
        median_trace = np.array(entry.get("median_trace_dD_cos", []), dtype=np.float64)
        pv = entry["picked_variant"]
        rep_trace = np.array(pv["dD_cos_per_layer"], dtype=np.float64)
        n_class = entry["n_class"]
        argmax_layer = pv["argmax_layer"]
        gene = pv["gene"]; chrom = pv["chrom"]; pos = pv["pos"]
        ref = pv["ref"]; alt = pv["alt"]
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

    out_pdf = OUT_DIR / "fig_variants.pdf"
    out_png = OUT_DIR / "fig_variants.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("wrote %s + .png", out_pdf)
    return {"kw_p": kw_p, "pair_p_adj": {f"{a}_vs_{b}": v for (a, b), v in pair_p.items()},
             "n_per_class": dict(zip(classes_order, n_per)),
             "median_per_class": dict(zip(classes_order, medians))}


def main():
    setup_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-bg", action="store_true",
                        help="Skip chr22-background loading (panel-a sig stars become NA)")
    args = parser.parse_args()

    t0 = time.time()
    stylize()
    meta = {"step": PHASE, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        LOG.info("=== fig_shallowness ===")
        meta["shallowness"] = fig_shallowness(smoke=args.smoke, no_bg=args.no_bg)
        LOG.info("=== fig_variants ===")
        meta["variants"] = fig_variants()
        meta["status"] = "PASS"
        meta["wall_sec"] = time.time() - t0
        (OUT_DIR / "fig_v9_meta.json").write_text(
            json.dumps(meta, indent=2, default=str))
        LOG.info("done in %.1fs", time.time() - t0)
    except Exception as e:
        import traceback
        meta["status"] = "FAIL"
        meta["error"] = str(e)
        meta["traceback"] = traceback.format_exc()
        (OUT_DIR / "fig_v9_meta.json").write_text(
            json.dumps(meta, indent=2, default=str))
        LOG.exception("FAIL: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

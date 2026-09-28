"""Handoff summary figures A-D (portfolio variant).

Numbers are read from the archived arch_compare result files (../arch_compare/results),
or copied verbatim from the EXP1/EXP2 write-ups where no machine-readable file was
archived (marked SRC_MD). Portfolio edits: the per-panel tags that pointed into an
internal planning map were removed, figure subtitles were shortened, and only PNGs
are written. Run:  python figs.py   (writes figA-D *.png and figure_data.json here)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "arch_compare", "results")   # Handoff/experiments/arch_compare/results
OUT = HERE

# ---- palette (validated: validate_palette.js "#2a78d6,#eb6834" light, all pairs PASS)
BLUE, ORANGE = "#2a78d6", "#eb6834"          # 7B, 40B
GRAY = "#898781"                             # controls / final-state reference
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"

rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"], "font.size": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.labelsize": 6.5, "axes.titlesize": 7, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "legend.fontsize": 5.8,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
})

def style(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)

def tag(ax, text, added):
    return  # portfolio variant: internal planning-map tags removed
    kw = dict(transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6)
    if added:
        ax.text(1.0, 1.02, text, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=WASH, ec="none"), **kw)
    else:
        ax.text(1.0, 1.02, text, color=INK2, **kw)

def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", pad=11, fontweight="normal")

def onset_line(ax, x, label=None, y=0.97, color=GRAY):
    ax.axvline(x, color=color, lw=0.6, zorder=1)
    if label:
        ax.text(x, y, label, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=5.4, color=INK2,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

DATA = {}   # table view of every plotted number, written next to the figures

# ================================================================ data
def ratio(d):
    return json.load(open(f"{R}/{d}/profile.json"))["ratio"]          # ratio[i] = rho_{i+1}
R7 = {c: ratio(f"{c}_evo2_7b_utr") for c in ("chr22", "chr17", "chr19")}
R40 = {"chr22": ratio("chr22_evo2_40b"), "chr17": ratio("chr17_evo2_40b"),
       "chr19": ratio("chr19_evo2_40b_all")}

cos = json.load(open(f"{R}/cos_curves.json"))
def cos_arr(key, metric):
    c = cos[key]["curves"]
    a = np.array([[np.nan if c[r][metric][l][0] is None else c[r][metric][l][0]
                   for l in range(len(c[r][metric]))] for r in c])
    return a        # regions x blocks

# E2-01 (SRC_MD: EXP2 section 3.1), branch-max dNLL per block, locked_A 100 windows
E2_01_blocks = list(range(17, 32))
E2_01 = [.013, .012, .003, .011, .009, .009, .023, .022, .057, .007, .013, 2.153, 2.076, 5.062, 0.000]

a40 = json.load(open(f"{R}/atlas40_ci.json"))["conditions"]
def br_max(b):
    cands = [a40[k] for k in (f"b{b}.mlp", f"b{b}.mix") if k in a40]
    best = max(cands, key=lambda z: z["delta_nll"])
    return best["delta_nll"], best["ci"], best["branch"]

# M17
sd = json.load(open(f"{R}/surviving_dims.json"))["next_base"]
ls = json.load(open(f"{R}/layer_select.json"))
ls22 = [e for e in ls if e["chrom"] == "chr22"][0]["auroc"]["intergenic"]
ler = json.load(open(f"{R}/late_eval_reps.json"))["chr22"]["intergenic"]

# E1-07 (SRC_MD: EXP1 section 9.3), R(28)=U(h28)/U(h27), R(30)=U(h30)/U(h27)
E1_07 = [  # task, group, R28, R30
    ("entropy of model output", "out", 1.47, 1.07),
    ("margin of model output", "out", 1.44, 0.96),
    ("next nucleotide", "out", 1.07, 1.09),
    ("log p of true next base", "out", 1.06, 1.01),
    ("current nucleotide", "seq", 0.99, 0.51),
    ("coding exon vs intron", "seq", 0.97, 0.58),
    ("repeat", "seq", 0.92, 0.52),
    ("phyloP", "seq", 0.89, 0.40),
    ("context (3-way)", "seq", 0.83, 0.44),
    ("GC", "seq", 0.73, 0.31),
    ("intergenic vs intron", "seq", 0.45, 0.28),
    ("donor vs intron", "seq", 0.20, 0.23),
]

# recoverability of the last pre-onset state
e40 = json.load(open(f"{R}/exp1_40b.json"))["chr22_evo2_40b"]
REC7 = {"l-2": 0.937, "l0": 0.808, "l+2": 0.349, "l+2_pub": 0.270}   # E1-06, E2-21 re-analysis
REC40 = {"l-2": e40["recover_b19"], "l0": e40["recover_b21"], "l+1": e40["recover_b22"],
         "l+2": e40["recover_b23"], "b34": e40["recover_b34"], "norm": e40["recover_norm"]}

# alpha sweeps (M13)
al = {}
for f in ("alpha_sweep.json", "alpha_sweep_fine.json"):
    for k, v in json.load(open(f"{R}/{f}")).items():
        al[float(k)] = v["next_base_acc"]

# E2-13 dose (SRC_MD: EXP2 section 6.4a), D_shape
E2_13_a = [0, 0.25, 0.5, 0.9, 1.5, 2, 4]
E2_13 = {"g28": [0.4213, 5.7e-6, 2.6e-6, 1.9e-6, 2.8e-6, 4.7e-6, 2.8e-5],
         "m30": [0.1485, 5.1e-15, 5.1e-15, 3.4e-7, 3.5e-7, 5.1e-15, 5.1e-15],
         "g21": [3.2e-3, 1.9e-3, 7.8e-4, 3.1e-5, 6.0e-4, 2.1e-3, 2.6e-2]}

# M38 40B dose
rad = json.load(open(f"{R}/radial40.json"))["sweep"]
REM = {"b21.mlp": a40["b21.mlp"]["delta_nll"], "b21.mix": a40["b21.mix"]["delta_nll"],
       "b23.mix": a40["b23.mix"]["delta_nll"]}

# E1-17 boundary sharpness (SRC_MD: EXP1 section 14.4)
E1_17 = [("relative L2", "final", 2316), ("relative L2", "block 27", 2108),
         ("magnitude (M2)", "final", 1512), ("magnitude (M2)", "block 27", 1322),
         ("whitened (M4)", "final", 1231), ("whitened (M4)", "block 27", 1061),
         ("velocity (M3)", "none", 164), ("curvature (M3)", "none", 16.5),
         ("direction (M1)", "block 27", 10.6), ("tortuosity (M5)", "final", 9.2),
         ("geodesic (M3)", "none", 4.3), ("direction (M1)", "final", 0.77)]

# ================================================================ helpers
HN = r"$h_{\mathrm{norm}}$"

def header(fig, t1, t2, y1=0.985, y2=0.94, x=0.07):
    fig.text(x, y1, t1, fontsize=7.5, color=INK, fontweight="bold", va="top")
    fig.text(x, y2, t2, fontsize=5.8, color=INK2, va="top", linespacing=1.35)

def zero_marker(ax, x, col, y=1.5e-3):
    ax.plot([x], [y], "o", ms=3.2, mfc="white", mec=col, mew=0.9, zorder=3, clip_on=False)
    ax.text(x, y * 1.7, "0", ha="center", fontsize=5.3, color=INK2)

# ================================================================ Figure A
def fig_A():
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.5),
                            gridspec_kw=dict(width_ratios=[1, 1.05, 1.15]))
    plt.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.09, wspace=0.36, hspace=0.72)
    rows = [("7B", R7, 28, BLUE, "7B_chr22", 27), ("40B", R40, 21, ORANGE, "40B_chr22", 20)]
    DATA["A"] = {}
    YL = "ΔNLL, larger branch zeroed (nats)"
    for i, (nm, RR, on, col, ck, pre) in enumerate(rows):
        ax = axs[i, 0]; style(ax)
        pre_max = max(max(r[:on - 1]) for r in RR.values())
        on_min = min(r[on - 1] for r in RR.values())
        on_max = max(r[on - 1] for r in RR.values())
        ax.axhspan(pre_max, on_min, color=WASH, zorder=0, lw=0)
        for c, r in RR.items():
            ax.plot(range(1, len(r) + 1), r, color=col, lw=0.9, alpha=0.9, zorder=3)
        ax.set_yscale("log")
        ax.plot([on], [RR["chr22"][on - 1]], "o", ms=3.6, color=col, mec="white", mew=0.8, zorder=4)
        ax.text(0.03 if nm == "7B" else 0.5, 0.72, f"empty interval {pre_max:.2f} to {on_min:.0f}:\nany T in it returns b{on}",
                transform=ax.transAxes, fontsize=5.3, color=INK2, va="center")
        ax.text(on - 1.0, RR["chr22"][on - 1] * 2.2, f"b{on}: {on_min:.0f}–{on_max:.0f}×",
                fontsize=5.3, color=INK2, va="bottom", ha="right")
        ax.set_xlabel("block ℓ"); ax.set_ylabel(r"$\Vert h_\ell\Vert\ /\ \Vert h_{\ell-1}\Vert$")
        ax.set_xlim(0, len(RR["chr22"]) + 1)
        title(ax, "ad"[i], f"{nm}: norm ratio, 3 chromosomes")
        DATA["A"][f"{nm}_norm_ratio"] = {"pre_onset_max": pre_max, "onset_min": on_min, "onset_max": on_max,
                                          "ratios": {c: r for c, r in RR.items()}}

    # --- (b) 7B causal atlas, added E2-01 / E2-02
    ax = axs[0, 1]; style(ax)
    xs = [b for b, v in zip(E2_01_blocks, E2_01) if v > 0]
    ys = [v for v in E2_01 if v > 0]
    ax.plot(xs, ys, color=BLUE, lw=0.9, zorder=2)
    ax.plot(xs, ys, "o", ms=3.2, color=BLUE, mec="white", mew=0.7, zorder=3)
    for b in (21, 28):          # phase twins
        v = E2_01[E2_01_blocks.index(b)]
        ax.plot([b], [v], "o", ms=6.2, mfc="none", mec=INK2, mew=0.6, zorder=4)
    zero_marker(ax, 31, BLUE)
    ax.set_yscale("log"); ax.set_ylim(1.2e-3, 40)
    onset_line(ax, 28)
    ax.text(27.6, 0.16, "×165", fontsize=5.8, color=INK, ha="right")
    ax.text(16.8, 28, "ringed: phase twins, one period apart\n"
                      "b28 − b21 MLP  +2.15 (377×)\n"
                      "b28 − b25 MLP  +2.10 (39×)\n"
                      "b27 − b20 mixer  +0.0008 (null)",
            fontsize=5.1, color=INK2, va="top")
    ax.set_xticks(range(17, 32, 2)); ax.set_xlim(16.3, 31.8)
    ax.set_xlabel("block ℓ"); ax.set_ylabel(YL)
    title(ax, "b", "7B: causal atlas, every block")
    DATA["A"]["7B_causal_atlas"] = dict(zip(E2_01_blocks, E2_01))
    DATA["A"]["7B_structural_contrasts_E2-02"] = {   # SRC_MD: EXP2 section 3.2
        "b28.g-b21.g": [2.1474, 2.0688, 2.2239, "377x"], "b28.g-b25.g": [2.0980, 2.0208, 2.1745, "39.1x"],
        "b29.g-b22.g": [2.0695, 1.8107, 2.3581, "415x"], "b30.m-b23.m": [5.0390, 4.9897, 5.0919, "217x"],
        "b27.m-b20.m": [0.0008, -0.0002, 0.0018, "1.1x"], "step_into_b28": 165.1}

    # --- (e) 40B causal atlas (arch_compare atlas40_ci.json)
    ax = axs[1, 1]; style(ax)
    blocks = [18, 19, 20, 21, 22, 23, 30, 34]
    pos = [0, 1, 2, 3, 4, 5, 6.6, 7.6]
    ctrl = {18, 19, 20, 30}
    rec = {}
    for b, p in zip(blocks, pos):
        v, ci, brn = br_max(b)
        rec[b] = (v, ci, brn)
        c = GRAY if b in ctrl else ORANGE
        if v > 0:
            ax.plot([p, p], [ci[0], ci[1]], color=c, lw=0.8, zorder=2)
            ax.plot([p], [v], "o", ms=3.4, zorder=3, mfc="white" if b in ctrl else ORANGE, mec=c, mew=0.9)
        else:
            zero_marker(ax, p, c)
    ax.set_yscale("log"); ax.set_ylim(1.2e-3, 40)
    ax.set_xticks(pos); ax.set_xticklabels([str(b) for b in blocks])
    ax.text(5.8, 1.5e-3, "…", ha="center", fontsize=6, color=INK2)
    ax.set_xlim(-0.6, 8.2)
    ax.text(-0.45, 28, "target vs previous block of its operator class\n"
                       "b21 mixer +0.43 [0.39, 0.46], b23 mixer +0.46 [0.43, 0.49]\n"
                       "b23 MLP 0 (dead); b34 writes the state, moves nothing",
            fontsize=5.1, color=INK2, va="top")
    ax.set_xlabel("block ℓ"); ax.set_ylabel(YL)
    handles = [Line2D([], [], marker="o", ls="", ms=3.4, mfc=ORANGE, mec=ORANGE, label="target"),
               Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=GRAY, label="control")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.08), handletextpad=0.2)
    title(ax, "e", "40B: causal atlas, targets and controls")
    DATA["A"]["40B_causal_atlas_branchmax"] = {b: {"dNLL": v, "ci": ci, "branch": brn} for b, (v, ci, brn) in rec.items()}
    DATA["A"]["40B_paired_contrasts"] = json.load(open(f"{R}/atlas40_ci.json"))["conditions"]["paired"]

    # --- (c, f) direction, added A06
    for i, (nm, RR, on, col, ck, pre) in enumerate(rows):
        ax = axs[i, 2]; style(ax)
        tp = cos_arr(ck, "to_preonset"); th = cos_arr(ck, "to_hnorm"); cs = cos_arr(ck, "consecutive")
        L = tp.shape[1]; x = np.arange(L)
        for arr, c, lab in ((tp, col, f"to b{pre}, last pre-onset state"), (th, GRAY, "to final state " + HN)):
            ax.fill_between(x, np.nanmin(arr, 0), np.nanmax(arr, 0), color=c, alpha=0.18, lw=0, zorder=2)
            ax.plot(x, np.nanmean(arr, 0), color=c, lw=1.1, zorder=3, label=lab)
        onset_line(ax, on)
        adj_on = np.nanmean(cs[:, on]); adj_pre_min = np.nanmin(cs[:, 1:on])
        ax.set_ylim(-0.3, 1.08); ax.set_xlim(-0.5, L - 0.5)
        ax.set_xlabel("block ℓ"); ax.set_ylabel("cosine")
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if nm == "7B":
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.2)
            ax.text(0.5, -0.27, f"adjacent blocks: cos ≥ {adj_pre_min:.2f} before b{on}, {adj_on:.2f} at b{on}",
                    fontsize=5.1, color=INK2)
        else:
            ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), handlelength=1.0, fontsize=5.3)
            ax.text(0.5, -0.27, f"adjacent blocks: cos ≥ {adj_pre_min:.2f} before b{on}, {adj_on:.2f} at b{on}",
                    fontsize=5.1, color=INK2)
            ax.text(0.5, -0.155, "|cos to " + HN + "| < 0.005 at every block before b21", fontsize=5.1, color=INK2)
            ax.text(24.5, 0.6, "re-encoder b23", fontsize=5.1, color=INK2)
        title(ax, "cf"[i], f"{nm} chr22: where the state points")
        DATA["A"][f"{nm}_direction"] = {
            "to_preonset_mean": np.nanmean(tp, 0).round(4).tolist(),
            "to_hnorm_mean": np.nanmean(th, 0).round(4).tolist(),
            "adjacent_at_onset_mean": float(adj_on), "adjacent_pre_onset_min": float(adj_pre_min)}
    header(fig, "Figure A. The same block from activation norms and from causal ablation; the jump turns the state",
           "Row 1: Evo 2 7B.  Row 2: Evo 2 40B.  Norm ratios on chr22, chr17, chr19; (b) EXP2 panel (locked_A, 100 windows); (c, f) chr22.\n"
           "(c, f) are geometry of the same activations as (a, d), not a third instrument.")
    for ext in ("png",):
        fig.savefig(f"{OUT}/figA_existence.{ext}")
    plt.close(fig)

# ================================================================ Figure B
def fig_B():
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.2, 1.0], height_ratios=[1, 1],
                          left=0.07, right=0.985, top=0.74, bottom=0.25, wspace=0.62, hspace=0.25)
    DATA["B"] = {}
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for ax in (ax1, ax2): style(ax)
    lay = [32.6 if l == "norm" else int(l) for l in sd["layers"]]
    ax1.plot(lay, sd["acc"], color=BLUE, lw=1.1)
    ax1.plot(lay, sd["acc"], "o", ms=2.3, color=BLUE, mec="white", mew=0.5)
    ax1.set_ylabel("next-base\nprobe acc.", fontsize=6)
    ax1.set_ylim(0.3, 0.72); ax1.set_yticks([0.3, 0.5, 0.7])
    x_early = list(range(24)); y_early = ls22[:24]
    x_late = list(range(24, 32)) + [32.6]
    keys = [f"h{b}" for b in range(24, 32)] + ["hnorm"]
    y_late = [ler[k][0] for k in keys]; s_late = [ler[k][1] for k in keys]
    ax2.plot(x_early + x_late, y_early + y_late, color=BLUE, lw=1.1)
    ax2.errorbar(x_late, y_late, yerr=s_late, fmt="none", ecolor=BLUE, elinewidth=0.6, capsize=0)
    ax2.set_ylabel("intergenic\nAUROC", fontsize=6); ax2.set_ylim(0.5, 0.86); ax2.set_yticks([0.5, 0.7])
    for ax in (ax1, ax2):
        ax.axvline(28, color=GRAY, lw=0.6, zorder=1)
    ax2.text(27.4, 0.53, "onset", fontsize=5.2, color=INK2, ha="right")
    ax2.set_xticks([0, 8, 16, 24, 32.6]); ax2.set_xticklabels(["0", "8", "16", "24", "norm"])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xlabel("block (7B, chr22)")
    ax1.set_title("a  The output gains what the state loses", loc="left", pad=11)
    DATA["B"]["M17"] = {"next_base": dict(zip(sd["layers"], sd["acc"])),
                        "intergenic_b0_23_fp16": y_early, "intergenic_late_mean_sd": dict(zip(keys, zip(y_late, s_late)))}

    ax = fig.add_subplot(gs[:, 1]); style(ax, grid_y=False)
    ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0); ax.set_axisbelow(True)
    ypos = []; y = 0
    for k in range(len(E1_07)):
        if k == 4: y += 1.2
        ypos.append(y); y += 1
    top = max(ypos); ypos = [top - v for v in ypos]
    ax.axvline(1.0, color=INK2, lw=0.6, zorder=1)
    for (t, g, r28, r30), yy in zip(E1_07, ypos):
        ax.plot([r28, r30], [yy, yy], color=GRID, lw=1.2, zorder=1)
        ax.plot([r28], [yy], "o", ms=3.6, color=BLUE, mec="white", mew=0.6, zorder=3)
        ax.plot([r30], [yy], "s", ms=3.2, mfc="white", mec=BLUE, mew=0.9, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([t for t, *_ in E1_07], fontsize=5.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.6); ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_ylim(-0.7, top + 1.25)
    ax.text(0.02, ypos[0] + 0.75, "quantities the head uses", fontsize=5.3, color=INK2, fontweight="bold")
    ax.text(0.02, ypos[4] + 0.75, "sequence and annotation", fontsize=5.3, color=INK2, fontweight="bold")
    ax.text(1.02, -0.55, "1 = all kept", fontsize=5.1, color=INK2)
    ax.set_xlabel("information relative to block 27 (bits / bits)")
    handles = [Line2D([], [], marker="o", ls="", ms=3.6, color=BLUE, label="after the onset (h28)"),
               Line2D([], [], marker="s", ls="", ms=3.2, mfc="white", mec=BLUE, label="after the re-encoder (h30)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2, handletextpad=0.2,
              columnspacing=1.0)
    ax.set_title("b  What survives: capacity-matched decoders", loc="left", pad=11)
    ax.text(0.0, -0.36, "A kernel decoder that contains the linear one reads ≤ 0.015 bits more at h27, h28, h30 (EXP1 access-gap test).",
            transform=ax.transAxes, fontsize=5.1, color=INK2)
    DATA["B"]["E1-07"] = [{"task": t, "group": g, "R28": a, "R30": b} for t, g, a, b in E1_07]

    ax = fig.add_subplot(gs[:, 2]); style(ax)
    cats = ["ℓ*−2", "ℓ*\nwriter", "ℓ*+1", "ℓ*+2\nre-encoder", "later"]
    ax.axvspan(2.6, 3.4, color=WASH, lw=0, zorder=0)
    x7 = [0, 1, 3]; y7 = [REC7["l-2"], REC7["l0"], REC7["l+2"]]
    ax.plot(x7[:2], y7[:2], color=BLUE, lw=1.0, zorder=2)
    ax.plot(x7[1:], y7[1:], color=BLUE, lw=0.8, ls=(0, (1, 1.5)), zorder=2)   # h29 not measured
    ax.plot(x7, y7, "o", ms=3.8, color=BLUE, mec="white", mew=0.7, zorder=3, label="7B: source → h27")
    ax.text(1.45, 0.42, "7B ℓ*+1\nnot measured", fontsize=4.8, color=INK2, ha="center")
    ax.plot([3], [REC7["l+2_pub"]], "o", ms=3.2, mfc="white", mec=BLUE, mew=0.8, zorder=3)
    ax.annotate("0.270 as first reported", xy=(3, REC7["l+2_pub"]), xytext=(1.75, 0.09), fontsize=5.0, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=GRAY, shrinkA=1, shrinkB=2))
    x40 = [0, 1, 2, 3, 4]; y40 = [REC40["l-2"], REC40["l0"], REC40["l+1"], REC40["l+2"], REC40["b34"]]
    ax.plot(x40, y40, color=ORANGE, lw=1.0, zorder=2)
    ax.plot(x40, y40, "s", ms=3.4, color=ORANGE, mec="white", mew=0.7, zorder=3, label="40B: source → b20")
    ax.plot([4.3], [REC40["norm"]], "s", ms=3.0, mfc="white", mec=ORANGE, mew=0.8, zorder=3)
    ax.text(4.3, REC40["norm"] + 0.05, HN, fontsize=5.0, color=INK2, ha="center")
    ax.text(0.0, y7[0] + 0.04, "0.94", fontsize=5.2, color=INK2, ha="center")
    ax.text(1.0, y7[1] + 0.045, "0.81", fontsize=5.2, color=INK2, ha="center")
    ax.text(3.18, y7[2] + 0.02, "≈0.35", fontsize=5.2, color=INK2)
    ax.text(1.0, y40[1] - 0.09, "0.73", fontsize=5.2, color=INK2, ha="center")
    ax.text(3.18, y40[3] - 0.07, "0.24", fontsize=5.2, color=INK2)
    ax.set_xticks(range(5)); ax.set_xticklabels(cats, fontsize=5.6)
    ax.set_ylim(0, 1.08); ax.set_xlim(-0.4, 4.65)
    ax.set_ylabel("held-out R², last pre-onset state")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), handletextpad=0.2)
    ax.set_title("c  Lost at the re-encoder", loc="left", pad=11)
    DATA["B"]["recoverability"] = {"7B": REC7, "40B_chr22": REC40}
    header(fig, "Figure B. The handoff compresses toward what the head needs; information is lost two blocks later",
           "a: arch_compare chr22, 100 windows.  b: EXP1 chr22 400-window panel.  c: 7B from EXP1 (value re-analysed in EXP2), "
           "40B from arch_compare.\nEstimators differ between panels and between scales in (c): compare shapes and "
           "where the drop falls, not levels.", y2=0.925)
    for ext in ("png",):
        fig.savefig(f"{OUT}/figB_compression.{ext}")
    plt.close(fig)

# ================================================================ Figure C
def fig_C():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.05, 1.1, 1.0]))
    plt.subplots_adjust(left=0.07, right=0.985, top=0.7, bottom=0.19, wspace=0.42)
    DATA["C"] = {}
    ax = axs[0]; style(ax)
    xs = sorted(k for k in al if k > 0); ys = [al[k] for k in xs]
    ax.axvspan(1 / 235, 116 / 7162, color=WASH, lw=0, zorder=0)
    ax.plot(xs, ys, color=BLUE, lw=1.1, zorder=2)
    ax.plot(xs, ys, "o", ms=3.0, color=BLUE, mec="white", mew=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(2.8e-4, 3)
    X0 = 4e-4
    ax.plot([X0], [al[0.0]], "o", ms=3.4, mfc="white", mec=BLUE, mew=0.9, zorder=3, clip_on=False)
    ax.text(X0 * 1.25, al[0.0] + 0.004, "branch removed", fontsize=5.0, color=INK2, va="bottom")
    ax.set_xticks([X0, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.text(6.3e-4, 0.35, "//", fontsize=6, color=INK2, ha="center", va="center", clip_on=False)
    ax.axvline(1, color=GRAY, lw=0.6)
    ax.text(1, 0.365, "native", fontsize=5.0, color=INK2, ha="center",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.text(0.0085, 0.465, r"$\alpha\Vert g\Vert = \Vert r\Vert$" + "\n(host estimates\n1/235 to 1/61.5)",
            fontsize=5.0, color=INK2, ha="center")
    ax.set_ylim(0.35, 0.6); ax.set_xlabel("α (block-28 MLP output × α)")
    ax.set_ylabel("next-base accuracy (7B)")
    title(ax, "a", "Size of the onset branch")
    DATA["C"]["M13"] = {str(k): al[k] for k in sorted(al)}

    ax = axs[1]; style(ax)
    sty = {"g28": ("o", BLUE, BLUE, "b28 MLP (writer)"), "m30": ("s", BLUE, BLUE, "b30 mixer (re-encoder)"),
           "g21": ("o", "white", GRAY, "b21 MLP (phase twin of b28)")}
    for k in ("g28", "m30", "g21"):
        mk, fc, ec, lab = sty[k]
        ax.plot(E2_13_a, E2_13[k], color=ec, lw=0.9, zorder=2)
        ax.plot(E2_13_a, E2_13[k], mk, ms=3.3, mfc=fc, mec=ec if fc == "white" else "white", mew=0.8,
                color=ec, zorder=3, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-16, 1e5)
    ax.axvline(1, color=GRAY, lw=0.6)
    ax.set_xlabel("α (branch output × α)"); ax.set_ylabel("change in output distribution (D_shape)")
    ax.text(1.9, 2e-14, "b30 at 0.25×, 0.5×, 2×, 4×:\nΔNLL exactly 0", fontsize=5.1, color=INK2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
    title(ax, "b", "Why size is inert: step functions")
    DATA["C"]["E2-13"] = {"alpha": E2_13_a, **E2_13}

    ax = axs[2]; style(ax)
    sty40 = {"b21.mlp": ("o", "b21 MLP (writer)"), "b21.mix": ("^", "b21 mixer"), "b23.mix": ("s", "b23 mixer (re-encoder)")}
    for k, (mk, lab) in sty40.items():
        xa = [0, 0.25, 0.5, 1, 2, 4]
        ya = [REM[k], rad[k]["0.25"]["delta"], rad[k]["0.5"]["delta"], 0.0, rad[k]["2.0"]["delta"], rad[k]["4.0"]["delta"]]
        ax.plot(xa, ya, color=ORANGE, lw=0.9, zorder=2)
        ax.plot(xa, ya, mk, ms=3.3, color=ORANGE, mec="white", mew=0.6, zorder=3, label=lab)
        DATA["C"].setdefault("M38", {})[k] = dict(zip(map(str, xa), ya))
    ax.axvline(1, color=GRAY, lw=0.6)
    ax.set_xlabel("scale (0 = removed, 1 = native)"); ax.set_ylabel("ΔNLL (nats, 40B)")
    ax.set_ylim(-0.03, 0.58)
    ax.legend(loc="upper right", handletextpad=0.2)
    ax.text(1.35, 0.17, "b23 at half strength:\n1/82 of its removal cost", fontsize=5.1, color=INK2)
    title(ax, "c", "The same at 40B, less sharply")
    header(fig, "Figure C. The norm locates the handoff; the output depends on the direction written, not its size",
           "Every branch reads an RMS-normalised copy of its input, so once one term dominates the sum its size is divided out:\n"
           "size must stop mattering once " + r"$\alpha\Vert g\Vert \gg \Vert r\Vert$"
           + "; the shading in (a) marks the equal-norm point " + r"$\alpha\Vert g\Vert = \Vert r\Vert$"
           + ".  (b) EXP2 locked split; (a, c) arch_compare, 100 windows.", y2=0.915)
    for ext in ("png",):
        fig.savefig(f"{OUT}/figC_magnitude.{ext}")
    plt.close(fig)

# ================================================================ Figure D
def fig_D():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    plt.subplots_adjust(left=0.37, right=0.95, top=0.76, bottom=0.17)
    style(ax, grid_y=False); ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0, which="major"); ax.set_axisbelow(True)
    labs = [f"{m} · ref {r}" for m, r, v in E1_17]; vals = [v for *_, v in E1_17]
    y = np.arange(len(vals))[::-1]
    cols = [BLUE if (m, r) == ("direction (M1)", "final") else GRAY for m, r, v in E1_17]
    ax.barh(y, vals, height=0.55, color=cols, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(0.3, 9000)
    ax.axvline(1, color=INK2, lw=0.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.6); ax.tick_params(axis="y", length=0)
    for yy, v in zip(y, vals):
        ax.text(max(v * 1.15, 1.12), yy, f"{v:g}", va="center", fontsize=5.2, color=INK2)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("sharpness at the onset\n(|change b27→b28| ÷ median change between blocks)", fontsize=6)
    header(fig, "Figure D. Only the settling-depth metric cannot see the handoff",
           "7B, twelve trajectory metrics × reference.  Direction to the final state (blue),\n"
           "the metric settling depth is built on, changes less at the onset than between\n"
           "ordinary blocks; folding it into one depth loses a further 55× (|d| 1.26 vs 0.023).",
           x=0.03, y2=0.925)
    for ext in ("png",):
        fig.savefig(f"{OUT}/figD_necessity.{ext}")
    plt.close(fig)
    DATA["D"] = {"E1-17": [{"metric": m, "reference": r, "sharpness": v} for m, r, v in E1_17]}

if __name__ == "__main__":
    fig_A(); fig_B(); fig_C(); fig_D()
    json.dump(DATA, open(f"{OUT}/figure_data.json", "w"), indent=1, default=float)
    a = DATA["A"]
    for k in ("7B", "40B"):
        print(k, "adjacent at onset %.4f  pre-onset min %.4f" % (a[f"{k}_direction"]["adjacent_at_onset_mean"],
              a[f"{k}_direction"]["adjacent_pre_onset_min"]),
              " gap %.3f - %.2f (max %.2f)" % (a[f"{k}_norm_ratio"]["pre_onset_max"], a[f"{k}_norm_ratio"]["onset_min"],
                                                a[f"{k}_norm_ratio"]["onset_max"]))

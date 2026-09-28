"""Render assets/hero.png: Evo 2 7B, three lenses on one depth axis.

Every number is read from a result file in this folder:
  (a) GDTR mean settling depth per context  ../gdtr-poc/results/phase1.6/gate_b.json
  (b) TDiG linear-probe AUROC per layer      ../tdig/results/analysis_BD/per_layer_auroc.csv
  (c) block-to-block norm ratio (handoff)    ../handoff/arch_compare/results/chr22_evo2_7b_utr/profile.json
Blocks are 0-based everywhere. GDTR's c(t) is 1-based (gdtr-poc/src/gdtr.py), so the
settling dots are drawn at c - 1. Run:  python make_hero.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS, WASH = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"

rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlesize": 7.8, "axes.titlecolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6, "legend.fontsize": 6.6, "legend.frameon": False,
    "savefig.dpi": 210, "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---------------------------------------------------------------- data
gate = json.loads((ROOT / "gdtr-poc/results/phase1.6/gate_b.json").read_text())["per_context"]
ctx_order = ["splice_donor", "splice_acceptor", "3utr", "intron", "coding_exon", "intergenic", "5utr"]
ctx_label = {"splice_donor": "splice donor", "splice_acceptor": "splice acceptor", "3utr": "3′ UTR",
             "intron": "intron", "coding_exon": "coding exon", "intergenic": "intergenic",
             "5utr": "5′ UTR"}
cbar = {c: gate[c]["mean_c"] for c in ctx_order}

probe = defaultdict(dict)
with open(ROOT / "tdig/results/analysis_BD/per_layer_auroc.csv") as f:
    for r in csv.DictReader(f):
        probe[r["pair"]][int(r["layer"])] = float(r["AUROC"])

ratio = json.loads((ROOT / "handoff/arch_compare/results/chr22_evo2_7b_utr/profile.json").read_text())["ratio"]
ratio_x = list(range(1, len(ratio) + 1))          # ratio[i] = ||h_{i+1}|| / ||h_i||


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.axvspan(27.5, 30.5, color=WASH, lw=0, zorder=0)
    ax.set_xlim(-0.8, 31.8)


fig, axs = plt.subplots(3, 1, figsize=(7.6, 6.0), sharex=True,
                        gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.0]))
plt.subplots_adjust(left=0.13, right=0.965, top=0.855, bottom=0.075, hspace=0.42)

# (a) GDTR settling depth
ax = axs[0]
style(ax)
ys = list(range(len(ctx_order)))[::-1]
for c, y in zip(ctx_order, ys):
    col = BLUE if c.startswith("splice") else GRAY
    ax.plot([cbar[c] - 1], [y], "o", ms=5.2, color=col, mec="white", mew=0.8, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([ctx_label[c] for c in ctx_order])
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.8, len(ctx_order) - 0.2)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color=GRID, lw=0.45, zorder=0)
d, i = cbar["splice_donor"], cbar["intron"]
ax.text(d - 1 - 0.6, ys[0], rf"$\bar{{c}}$ = {d:.2f}", ha="right", va="center", fontsize=6.6, color=INK2)
ax.text(i - 1 - 0.6, ys[3], rf"$\bar{{c}}$ = {i:.2f}", ha="right", va="center", fontsize=6.6, color=INK2)
ax.set_title("a   GDTR (published): mean settling depth per genomic context", loc="left", pad=6)
ax.text(0.03, 0.5, "splice sites settle ~2 layers earlier than intron;\n"
        "all seven context means fall between blocks 24 and 28",
        transform=ax.transAxes, fontsize=6.6, color=INK2, va="center")

# (b) TDiG probe AUROC
ax = axs[1]
style(ax)
L = sorted(probe["splice_donor_vs_intron"])
for key, col, lab in (("splice_donor_vs_intron", BLUE, "splice donor vs intron"),
                      ("coding_exon_vs_intron", ORANGE, "coding exon vs intron")):
    ax.plot(L, [probe[key][l] for l in L], color=col, lw=1.4, zorder=3, label=lab)
ax.set_ylim(0.5, 1.02)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("probe AUROC")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.6)
sd = probe["splice_donor_vs_intron"]
ax.annotate(f"{sd[27]:.3f} → {sd[29]:.3f}\n(L27 → L29)", xy=(29, sd[29]), xytext=(21.3, 0.62),
            fontsize=6.6, color=INK2, arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY))
ax.set_title("b   TDiG (team follow-up): what a linear probe can still read at each layer",
             loc="left", pad=6)

# (c) handoff norm ratio
ax = axs[2]
style(ax)
ax.plot(ratio_x, ratio, color=BLUE, lw=1.4, zorder=3)
ax.plot(ratio_x, ratio, "o", ms=2.6, color=BLUE, mec="white", mew=0.5, zorder=4)
ax.set_yscale("log")
ax.set_ylim(0.5, 5e6)
ax.set_yticks([1, 1e2, 1e4, 1e6])
ax.set_yticklabels(["1", "10²", "10⁴", "10⁶"])
ax.set_ylabel(r"$\Vert h_\ell\Vert\ /\ \Vert h_{\ell-1}\Vert$")
ax.text(27.3, ratio[27], f"b28 ×{ratio[27]:.0f}\nwriter", ha="right", va="center",
        fontsize=6.6, color=INK2)
ax.text(29.7, ratio[29], f"b30 ×{ratio[29] / 1e5:.1f}·10⁵\nre-encoder", ha="right", va="center", fontsize=6.6, color=INK2)
ax.text(31.3, 3.2, "b31\nidle", ha="center", va="bottom", fontsize=6.6, color=INK2)
ax.set_title("c   Handoff line (ongoing): where the residual stream jumps", loc="left", pad=6)
ax.set_xlabel("Evo 2 7B block (0-based)")
ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])

fig.text(0.13, 0.975, "Evo 2 7B, one depth axis: where tokens settle, what probes read, where the stream jumps",
         fontsize=9.2, color=INK, fontweight="bold", va="top")
fig.text(0.13, 0.938, "chr22. Grey band = blocks 28–30 (late-stack handoff). GDTR's c(t) is 1-based, so its dots "
         "sit at " + r"$\bar{c}$" + " − 1 on this 0-based axis.", fontsize=6.8, color=INK2, va="top")
fig.savefig(HERE / "hero.png")
print("wrote", HERE / "hero.png")

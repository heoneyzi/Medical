"""Draw assets/hero.png for the CAFA6 README.

Every number is copied from the printed outputs of the team's EDA notebook
(`../code/eda.ipynb`, cells 4, 8 and 12) on the CAFA 6 training data.
Term names are the standard Gene Ontology labels of the listed GO IDs.
Run:  python make_hero.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ASPECT = {"MF": "#2a78d6", "CC": "#eb6834", "BP": "#1baf7a"}  # validated categorical slots 1-3
ASPECT_NAME = {"MF": "Molecular function", "CC": "Cellular component", "BP": "Biological process"}

# eda.ipynb cell 12: top terms by number of annotated training proteins
TOP = [
    ("GO:0005515", "protein binding", "MF", 33713),
    ("GO:0005634", "nucleus", "CC", 13283),
    ("GO:0005829", "cytosol", "CC", 13040),
    ("GO:0005886", "plasma membrane", "CC", 10150),
    ("GO:0005737", "cytoplasm", "CC", 9442),
    ("GO:0005739", "mitochondrion", "CC", 5807),
    ("GO:0005654", "nucleoplasm", "CC", 5065),
    ("GO:0016020", "membrane", "CC", 3563),
    ("GO:0042802", "identical protein binding", "MF", 3547),
    ("GO:0005576", "extracellular region", "CC", 3241),
    ("GO:0005783", "endoplasmic reticulum", "CC", 2837),
    ("GO:0005615", "extracellular space", "CC", 2391),
    ("GO:0045944", "positive regulation of\ntranscription by RNA pol II", "BP", 2319),
]
# eda.ipynb cells 4 and 8
TILES = [
    ("82,404", "training proteins with known GO labels"),
    ("224,309", "proteins to predict (test superset)"),
    ("26,125", "distinct GO terms used as labels"),
    ("537,027", "protein-term annotations\nBP 250,805 · CC 157,770 · MF 128,452"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(9.5, 4.35), dpi=160, facecolor=SURFACE)
fig.text(0.02, 0.955, "CAFA 6: predict what a protein does - as Gene Ontology terms - from its sequence",
         fontsize=12, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.02, 0.895, "Training-set statistics from the team EDA notebook (Kaggle CAFA 6 data, UniProtKB/Swiss-Prot 2025_03)",
         fontsize=8.5, color=INK2, ha="left", va="top")

# ---- left: stat tiles
tiles = fig.add_axes([0.02, 0.03, 0.29, 0.80])
tiles.set_axis_off()
tiles.set_xlim(0, 1)
tiles.set_ylim(0, 1)
h, gap = 0.225, 0.025
for i, (value, label) in enumerate(TILES):
    y0 = 1 - (i + 1) * h - i * gap
    tiles.add_patch(FancyBboxPatch((0.0, y0), 1.0, h, boxstyle="round,pad=0,rounding_size=0.03",
                                   facecolor="#f4f3ef", edgecolor="none"))
    two = "\n" in label
    tiles.text(0.06, y0 + h * (0.70 if two else 0.64), value, fontsize=17, fontweight="bold", color=INK, va="center")
    tiles.text(0.06, y0 + h * (0.30 if two else 0.28), label, fontsize=7.8, color=INK2,
               va="center", linespacing=1.35)

# ---- right: long-tail bar chart
ax = fig.add_axes([0.535, 0.175, 0.43, 0.60], facecolor=SURFACE)
names = [f"{n}" for _, n, _, _ in TOP][::-1]
vals = [v for *_, v in TOP][::-1]
cols = [ASPECT[a] for _, _, a, _ in TOP][::-1]
ys = [-0.3] + list(range(1, len(TOP)))  # extra room for the two-line label at the bottom
ax.barh(ys, vals, height=0.62, color=cols, edgecolor="none", zorder=3)
for y, v in zip(ys, vals):
    ax.text(v + 450, y, f"{v:,}", va="center", ha="left", fontsize=7.4, color=INK2)
ax.set_ylim(-1.0, len(TOP) - 0.4)
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=7.6, color=INK, linespacing=1.0)
ax.set_xlim(0, 39000)
ax.set_xticks([0, 10000, 20000, 30000])
ax.set_xticklabels(["0", "10,000", "20,000", "30,000"], fontsize=7.4, color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="y", length=0, pad=4)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("training proteins annotated with the term", fontsize=7.8, color=INK2)
fig.text(0.355, 0.815, "A long tail: a few terms are everywhere, most label only a handful of proteins",
         fontsize=9.2, fontweight="bold", color=INK, ha="left", va="center")
handles = [plt.Rectangle((0, 0), 1, 1, color=ASPECT[k]) for k in ("MF", "CC", "BP")]
ax.legend(handles, [ASPECT_NAME[k] for k in ("MF", "CC", "BP")], loc="lower right", frameon=False,
          fontsize=7.4, labelcolor=INK2, handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
fig.text(0.355, 0.03,
         "Top 13 of 26,125 terms shown. Median term labels 4 proteins; 75% of terms label 12 or fewer.",
         fontsize=7.4, color=MUTED, ha="left")
fig.savefig("hero.png", dpi=160, facecolor=SURFACE)
print("saved hero.png")

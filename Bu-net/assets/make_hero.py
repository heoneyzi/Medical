"""Draw assets/hero.png: schematic of the team's Simplified BU-Net.

The structure follows code/bu-net_pytorch/model/SimpleBUnet.py exactly
(channel widths, where the WC and RES blocks sit, 1x1 conv + softmax head).
Run:  python make_hero.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, BLUE_T = "#2a78d6", "#dbe8fb"      # conv blocks
ORANGE, ORANGE_T = "#eb6834", "#fbe0d4"  # WC block
AQUA, AQUA_T = "#1baf7a", "#d3f1e5"      # RES block
GRAY, GRAY_T = "#898781", "#f1f0ec"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(9.5, 4.35), dpi=160, facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.set_axis_off()


def box(x, y, w, h, text, fc, ec, fs=7.6, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=fc, edgecolor=ec, linewidth=1.0))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, linespacing=1.15)


def arrow(p, q, color=GRAY, lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


ax.text(2, 44.2, "Simplified BU-Net: the team's lightweight take on BU-Net for 2-D brain-tumor segmentation",
        fontsize=12, fontweight="bold", color=INK, va="top")
ax.text(2, 40.6, "U-Net encoder-decoder + a wide-context (WC) bottleneck + one residual-extended-skip (RES) block"
        "  ·  code/bu-net_pytorch/model/SimpleBUnet.py", fontsize=8, color=INK2, va="top")

levels = [(33.0, 64, "256²"), (26.0, 128, "128²"), (19.0, 256, "64²"), (12.0, 512, "32²")]
EX, DX, W, H = 13, 55, 12, 4.4
for i, (y, ch, res) in enumerate(levels):
    box(EX, y, W, H, f"Conv×2 · {ch}", BLUE_T, BLUE)
    if i:  # level-1 resolution is given by the input box
        ax.text(EX - 0.8, y, res, ha="right", va="center", fontsize=7, color=MUTED)
    box(DX, y, W, H, f"Conv×2 · {ch}", BLUE_T, BLUE)
    if i < 3:  # encoder max-pool, decoder up-conv
        arrow((EX + W / 2, y - H / 2), (EX + W / 2, levels[i + 1][0] + H / 2))
        arrow((DX + W / 2, levels[i + 1][0] + H / 2), (DX + W / 2, y - H / 2))
    if i != 2:  # plain skip connections (copy + concat)
        arrow((EX + W, y), (DX, y), color=GRAY, lw=0.9)
ax.text(EX + W / 2 + 0.8, 29.6, "max-pool", fontsize=6.6, color=MUTED, va="center")
ax.text(DX + W / 2 + 0.8, 29.6, "up-conv", fontsize=6.6, color=MUTED, va="center")
ax.text(40, 34.0, "skip: copy + concatenate", fontsize=6.8, color=MUTED, ha="center")

# RES on the 64x64 skip path
box(33, 19.0, 14, H, "RES block · 256", AQUA_T, AQUA, weight="bold")
arrow((EX + W, 19.0), (33, 19.0), color=AQUA, lw=1.1)
arrow((47, 19.0), (DX, 19.0), color=AQUA, lw=1.1)

# bottleneck: WC -> 3x3 conv
box(21, 4.6, 15, H, "WC block · 512→1024", ORANGE_T, ORANGE, weight="bold")
box(39, 4.6, 13, H, "Conv 3×3 · 1024", BLUE_T, BLUE)
arrow((EX + W / 2, 12.0 - H / 2), (21, 4.6), rad=0.25)
ax.text(8.2, 6.4, "max-pool\n16²", fontsize=6.6, color=MUTED, ha="left", linespacing=1.1)
arrow((36, 4.6), (39, 4.6))
arrow((52, 4.6), (DX + W / 2, 12.0 - H / 2), rad=0.25)

# input / output
box(1.2, 33.0, 9.6, 5.6, "MRI slice\n1 × 256 × 256", GRAY_T, GRAY, fs=7.2)
arrow((10.8, 33.0), (EX, 33.0))
box(70.5, 33.0, 17.0, 5.6, "tumor mask\n4 classes × 256 × 256", GRAY_T, GRAY, fs=7.2)
arrow((DX + W, 33.0), (70.5, 33.0))
ax.text(79.0, 28.9, "1×1 conv + softmax", fontsize=6.6, color=MUTED, ha="center")

notes = [
    (ORANGE, "WC", "two parallel factorized paths (15×1→1×15 and\n1×15→15×1), concatenated, then a 3×3 conv"),
    (AQUA, "RES", "identity + 15-wide and 9-wide factorized paths,\nthen 3×3 and 1×1 convs (the paper uses four\npaths on every skip; the team kept one block)"),
    (GRAY, "Data", "one MRI modality per 2-D slice (BraTS 2018);\nclasses: background, core, edema, enhancing"),
]
y = 22.5
for c, head, body in notes:
    ax.add_patch(FancyBboxPatch((70.5, y - 0.9), 1.6, 1.8, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor=c, edgecolor="none"))
    ax.text(73.0, y, head, fontsize=7.2, fontweight="bold", color=INK, va="center")
    ax.text(70.5, y - 1.9, body, fontsize=6.5, color=INK2, va="top", linespacing=1.25)
    y -= 7.4
fig.savefig("hero.png", dpi=160, facecolor=SURFACE)
print("saved hero.png")

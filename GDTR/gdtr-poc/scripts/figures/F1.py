"""F1 — Method schematic + Evo 2 calibration.

4 panels:
(a) 4-axis interpretability schematic (where look / what encode / what predict / where think deeply)
(b) gDTR pipeline (NLP DTR → genomic CLM adaptations)
(c) Tuned-lens per-layer recovery (Evo 2 32 layers); annotated peak / worst / canonical / degenerate
(d) Per-block-type raw M2 (running-min absorbs raw violations) — calibration justification
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _figstyle import setup, COLORS, label_panel, grid_y, save

setup()
RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "figures_v2"
OUT.mkdir(exist_ok=True)

recovery = json.loads((RES / "phase1.followup_full" / "recovery_curve.json").read_text())
verdict = json.loads((RES / "phase1.followup_full" / "verdict.json").read_text())
gate_a = json.loads((RES / "phase1.1" / "gate_a_evo_untuned.json").read_text())

fig = plt.figure(figsize=(11.5, 8))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

# ---- (a) 4-axis interpretability schematic -----------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

axes_data = [
    ("Where does the model\nLOOK?", "attention maps", COLORS["dD_jsd"], 0.05, 0.78),
    ("What does the model\nENCODE?", "sparse autoencoders", COLORS["evo2"], 0.55, 0.78),
    ("What does the model\nPREDICT?", "embedding distance,\nsequence likelihood", COLORS["alphamis"], 0.05, 0.42),
    ("Where does the model\nTHINK DEEPLY?  ★", "gDTR (this work)\nlayer-wise settling depth", COLORS["highlight"], 0.55, 0.42),
]
for label, methods, color, x, y in axes_data:
    ax.add_patch(plt.Rectangle((x, y), 0.40, 0.27, facecolor=color, alpha=0.35,
                                edgecolor=color, linewidth=1.5))
    ax.text(x+0.20, y+0.20, label, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(x+0.20, y+0.07, methods, ha="center", va="center", fontsize=7.5, color="#222")

ax.text(0.5, 0.05, "Four axes of mechanistic interpretability for genomic CLMs",
        ha="center", va="center", fontsize=8.5, fontstyle="italic")
label_panel(ax, "(a)", x=-0.03)

# ---- (b) NLP-DTR → gDTR pipeline ---------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Boxes
def box(x, y, w, h, head, body, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.35,
                                edgecolor=color, linewidth=1.4))
    ax.text(x+w/2, y+h-0.04, head, ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(x+w/2, y+0.04, body, ha="center", va="bottom", fontsize=7.5)

box(0.02, 0.62, 0.44, 0.30, "NLP-DTR (Chen et al. 2026)",
    "JSD lens, γ=0.5, |V|≈100k\nGPT-OSS, DeepSeek-R1, Qwen3", COLORS["dD_jsd"])
box(0.54, 0.62, 0.44, 0.30, "gDTR (this work)",
    "JSD + cosine UR lens\nq70 calibration γ_cos=0.40\nEvo 2 / HyenaDNA / NT / DNABERT-2",
    COLORS["highlight"])
ax.annotate("", xy=(0.54, 0.77), xytext=(0.46, 0.77),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))

# Bottom: 3 adaptation challenges
ax.text(0.5, 0.50, "Three adaptation challenges:", ha="center", va="center",
        fontsize=9, fontweight="bold")
challenges = [
    "C1  small vocabulary  |V| ≤ 512  →  q70 calibration",
    "C2  hybrid architectures  →  block-type-aware lens",
    "C3  untied lm_head + idle L31  →  tuned lens, tap = L29",
]
for i, c in enumerate(challenges):
    ax.text(0.5, 0.36 - i*0.08, c, ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f4f4f4", edgecolor="#888", linewidth=0.6))
label_panel(ax, "(b)", x=-0.03)

# ---- (c) Tuned-lens per-layer recovery ---------------------------------------
ax = fig.add_subplot(gs[1, 0])
per_layer = recovery["per_layer"]
layers = sorted([int(k) for k in per_layer.keys()])
recov = [per_layer[str(l)]["recovery_pct"] for l in layers]
init = [per_layer[str(l)]["initial_loss_identity"] for l in layers]

ax.plot(layers, recov, color=COLORS["dD_cos"], linewidth=1.6, marker="o", markersize=3.5,
        label="recovery (1 - final/initial)")
ax.axhline(0.98, linestyle=":", color="grey", alpha=0.7, label="0.98 cutoff")
ax.fill_between(layers, recov, 1.0, where=np.array(recov) >= 0.98, alpha=0.10, color=COLORS["dD_cos"])
ax.set_xlabel("layer index $l$")
ax.set_ylabel("tuned-lens recovery pct")
ax.set_ylim(0.96, 1.005)
ax.set_xticks(range(0, 33, 4))

# Annotate landmarks
ax.axvline(verdict["peak_divergence_layer"], color=COLORS["evo2"], linestyle="--", alpha=0.7)
ax.text(verdict["peak_divergence_layer"]+0.5, 0.965, f"L={verdict['peak_divergence_layer']}\npeak div.",
        fontsize=7, color=COLORS["evo2"])
ax.axvline(verdict["worst_recovery_layer"], color=COLORS["alphamis"], linestyle="--", alpha=0.7)
ax.text(verdict["worst_recovery_layer"]+0.5, 0.965, f"L={verdict['worst_recovery_layer']}\nworst",
        fontsize=7, color=COLORS["alphamis"])
ax.axvline(verdict["canonical_deep_thinking_layer"], color=COLORS["highlight"], linestyle="--", alpha=0.85)
ax.text(verdict["canonical_deep_thinking_layer"]-2.0, 0.965, f"L={verdict['canonical_deep_thinking_layer']}\ncanonical ★",
        fontsize=7, color=COLORS["highlight"], fontweight="bold")
for d in verdict["degenerate_layers"]:
    ax.scatter([d], [recov[d]], marker="x", color="grey", s=40, zorder=5)
ax.text(30, 0.973, "L=30,31\ndegenerate\n(idle block)", fontsize=7, color="grey", ha="center")

ax.legend(fontsize=7.5, loc="lower left")
grid_y(ax)
ax.set_title(f"30/32 layers recover ≥0.98 via single 4096² affine", fontsize=9.5, loc="left")
label_panel(ax, "(c)", x=-0.13)

# ---- (d) Per-block-type raw M2 -----------------------------------------------
ax = fig.add_subplot(gs[1, 1])
M2_jsd = gate_a["per_block_M2_jsd"]
M2_cos = gate_a["per_block_M2_cos"]
blocks = ["attn", "hcs", "hcm", "hcl"]
ys_jsd = [M2_jsd[b] for b in blocks]
ys_cos = [M2_cos[b] for b in blocks]
xs = np.arange(len(blocks))
width = 0.36
ax.bar(xs - width/2, ys_jsd, width, label="JSD lens", color=COLORS["dD_jsd"])
ax.bar(xs + width/2, ys_cos, width, label="cosine UR lens", color=COLORS["dD_cos"])
ax.axhline(0.85, linestyle="--", color=COLORS["highlight"], linewidth=1.2, alpha=0.85,
           label="0.85 raw threshold")
ax.set_xticks(xs)
ax.set_xticklabels(["attn", "hyena-S", "hyena-M", "hyena-L"], fontsize=8)
ax.set_ylabel("raw monotonicity $M_2$ pass rate")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=7.5, loc="upper right")
grid_y(ax)
ax.text(1.5, 0.55, "Raw $M_2 < 0.85$ for all block types\n→ tuned lens + running-min\n   absorbs violations",
        fontsize=7.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.85))
ax.set_title("(d) Calibration: raw lens fails → tuned + running-min", fontsize=9.5, loc="left")
label_panel(ax, "(d)", x=-0.13)

plt.tight_layout()
save(fig, OUT / "F1_method_schematic")
print(f"saved {OUT/'F1_method_schematic.pdf'} and .png")

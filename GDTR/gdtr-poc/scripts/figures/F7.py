"""F7 — Cross-architecture two-tier invariance.

3 panels:
(a) 4×4 Spearman ρ heatmap (per-window mean settling depth, chr22)
(b) Per-model splice signal (donor mean_c minus intron mean_c, normalized by L)
(c) Family annotation: causal-LM-per-bp vs MLM-token, with arrow
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

concord = json.loads((RES / "phase4" / "concordance_matrix.json").read_text())
per_model = json.loads((RES / "phase4" / "per_model_summary.json").read_text())

models = concord["models"]
disp = {"evo2": "Evo 2 7B", "hyenadna": "HyenaDNA-large", "nt_v2": "NT-v2 500M", "dnabert2": "DNABERT-2 117M"}
rho = np.array([[concord["spearman_rho"][m1][m2] for m2 in models] for m1 in models])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# ---- (a) Spearman heatmap ----------------------------------------------------
ax = axes[0]
im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(models)))
ax.set_yticks(range(len(models)))
ax.set_xticklabels([disp[m] for m in models], rotation=30, ha="right", fontsize=8)
ax.set_yticklabels([disp[m] for m in models], fontsize=8)
for i in range(len(models)):
    for j in range(len(models)):
        v = rho[i, j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.5 else "black", fontweight="bold" if i != j else "normal")
plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Spearman ρ")
ax.set_title("Per-window mean $c$ (chr22)", fontsize=10)
label_panel(ax, "(a)")

# ---- (b) Per-model splice signal (donor vs intron, normalized by L) ----------
ax = axes[1]
data = []
for m in models:
    sig = per_model[m]["splice_signal"]
    L = per_model[m]["L"]
    if sig["kind"] == "per_position":
        d = sig["data"]
        intron_mc = d["intron"]["mean_c"]
        donor_mc = d["splice_donor"]["mean_c"]
        acc_mc = d["splice_acceptor"]["mean_c"]
    else:  # token-based (MLM): use splice_containing vs intron_dominant
        d = sig["data"]
        intron_mc = d["intron_dominant"]["mean_c"]
        donor_mc = d["splice_containing"]["mean_c"]  # tokens spanning splice site
        acc_mc = d.get("exon_dominant", {}).get("mean_c", intron_mc)
    data.append((m, L, intron_mc, donor_mc, acc_mc))

xs = np.arange(len(models))
width = 0.35
intron_norm = []
donor_norm = []
for m, L, im_, do_, ac_ in data:
    intron_norm.append(im_ / L if im_ is not None else 0)
    donor_norm.append(do_ / L if do_ is not None else 0)

ax.bar(xs - width/2, intron_norm, width, label="intron / L", color=COLORS["dD_cos"])
ax.bar(xs + width/2, donor_norm, width, label="splice donor / L", color=COLORS["highlight"])
ax.set_xticks(xs)
ax.set_xticklabels([disp[m] for m in models], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("mean $c$ / L (depth-normalized)")
ax.legend(fontsize=8)
grid_y(ax)
ax.set_title("(b) Donor < intron in per-bp models", fontsize=10, loc="left")
# annotate token-based caveat
ax.text(2.5, max(intron_norm)*0.5, "MLM (NT / DNABERT-2):\ntoken-not-bp; direct\ncomparison limited",
        fontsize=7, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.85))
label_panel(ax, "(b)")

# ---- (c) Two-tier diagram ----------------------------------------------------
ax = axes[2]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box(x, y, w, h, label, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.6))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9, fontweight="bold")

box(0.05, 0.62, 0.4, 0.20, "Causal-LM\n(per-bp)\nEvo 2 + HyenaDNA\nρ = +0.516", COLORS["dD_cos"])
box(0.55, 0.62, 0.4, 0.20, "MLM\n(token-based)\nNT-v2 + DNABERT-2\nρ = +0.663", COLORS["evo2"])

ax.annotate("", xy=(0.55, 0.72), xytext=(0.45, 0.72),
            arrowprops=dict(arrowstyle="<->", color=COLORS["highlight"], linewidth=2))
ax.text(0.5, 0.78, "ρ ∈ [-0.29, -0.12]\n(weakly negative)", ha="center", va="bottom", fontsize=8,
        color=COLORS["highlight"], fontweight="bold")

ax.text(0.5, 0.50, "Two-tier architecture invariance:", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(0.5, 0.40, "(i) within architecture family — strong concordance\n(ii) cross-family — weakly negative,\n     suggesting tokenization-level dependence",
        ha="center", va="top", fontsize=8.5)
ax.text(0.5, 0.10, "Top-decile windows: Jaccard = 0\nacross all 4 models", ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["control"], alpha=0.4))
label_panel(ax, "(c)", x=-0.05)

plt.tight_layout()
save(fig, OUT / "F7_cross_architecture")
print(f"saved {OUT/'F7_cross_architecture.pdf'} and .png")

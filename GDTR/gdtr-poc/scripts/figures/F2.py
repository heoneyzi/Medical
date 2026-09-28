"""F2 — Splice deep-thinking universality.

4 panels:
(a) chr22 donor distance profile (Evo 2)
(b) chr22 + chr17 acceptor profile overlay (Evo 2)
(c) chr22 per-context mean_c ranking (Evo 2)
(d) Evo 2 vs HyenaDNA-large splice signal (donor / acceptor / intron)
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


def _profile(p):
    d = json.loads(Path(p).read_text())
    out = {}
    for site in ("donor", "acceptor"):
        if site not in d:
            continue
        keys = sorted(d[site].keys(), key=lambda x: int(x))
        x = np.array([int(k) for k in keys])
        y = np.array([d[site][k]["mean_c"] for k in keys])
        n = np.array([d[site][k]["n"] for k in keys])
        out[site] = (x, y, n)
    return out


prof_chr22 = _profile(RES / "phase1.6_sub" / "splice_distance_profile.json")
prof_chr17 = _profile(RES / "phase2.5" / "splice_chr17_profile.json")

per_model = json.loads((RES / "phase4" / "per_model_summary.json").read_text())
ctx_chr22 = json.loads((RES / "phase1.6" / "gate_b.json").read_text())["per_context"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

# ---- (a) chr22 donor profile -------------------------------------------------
ax = axes[0, 0]
x22, y22, _ = prof_chr22["donor"]
intron_chr22 = ctx_chr22["intron"]["mean_c"]
ax.plot(x22, y22, marker="o", color=COLORS["dD_cos"], linewidth=2, markersize=4, label="chr22 donor")
ax.axhline(intron_chr22, linestyle="--", color="grey", alpha=0.6, label=f"chr22 intron ({intron_chr22:.2f})")
ax.set_xlabel("distance from splice donor (bp)")
ax.set_ylabel("mean settling depth $c$")
ax.set_xlim(-220, 220)
ax.legend(fontsize=8, loc="lower left")
grid_y(ax)
label_panel(ax, "(a)", x=-0.16)

# ---- (b) chr22 + chr17 acceptor overlay --------------------------------------
ax = axes[0, 1]
x22, y22, _ = prof_chr22["acceptor"]
x17, y17, _ = prof_chr17["acceptor"]
ax.plot(x22, y22, marker="o", color=COLORS["dD_cos"], linewidth=2, markersize=4, label="chr22")
ax.plot(x17, y17, marker="s", color=COLORS["evo2"], linewidth=2, markersize=4, label="chr17 (replication)")
ax.set_xlabel("distance from splice acceptor (bp)")
ax.set_ylabel("mean settling depth $c$")
ax.set_xlim(-220, 220)
ax.legend(fontsize=8)
grid_y(ax)
label_panel(ax, "(b)", x=-0.16)

# ---- (c) per-context mean_c (chr22, Evo 2) -----------------------------------
ax = axes[1, 0]
order = ["splice_donor", "splice_acceptor", "3utr", "intron", "coding_exon", "intergenic", "5utr"]
labels = ["splice\ndonor", "splice\nacceptor", "3'UTR", "intron", "coding\nexon", "intergenic", "5'UTR"]
vals = [ctx_chr22[k]["mean_c"] for k in order]
ctx_colors = [COLORS["highlight"], COLORS["highlight"], COLORS["alphamis"],
              COLORS["dD_cos"], COLORS["evo2"], COLORS["control"], COLORS["alphamis"]]
bars = ax.bar(range(len(order)), vals, color=ctx_colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mean settling depth $c$")
ax.set_ylim(min(vals)-0.3, max(vals)+0.3)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
grid_y(ax)
label_panel(ax, "(c)", x=-0.16)

# ---- (d) Evo 2 vs HyenaDNA donor < intron ------------------------------------
ax = axes[1, 1]
models = ["evo2", "hyenadna"]
disp = ["Evo 2 7B\n(32 layers)", "HyenaDNA-large\n(8 layers)"]
intron_vals, donor_vals, acc_vals = [], [], []
for m in models:
    d = per_model[m]["splice_signal"]["data"]
    intron_vals.append(d["intron"]["mean_c"])
    donor_vals.append(d["splice_donor"]["mean_c"])
    acc_vals.append(d["splice_acceptor"]["mean_c"])
xs = np.arange(len(models))
width = 0.27
ax.bar(xs-width, intron_vals, width, label="intron", color=COLORS["dD_cos"])
ax.bar(xs, donor_vals, width, label="splice donor", color=COLORS["highlight"])
ax.bar(xs+width, acc_vals, width, label="splice acceptor", color=COLORS["alphamis"])
ax.set_xticks(xs)
ax.set_xticklabels(disp, fontsize=8)
ax.set_ylabel("mean settling depth $c$ (model-native)")
ax.legend(fontsize=8)
grid_y(ax)
for i, m in enumerate(models):
    delta = intron_vals[i] - donor_vals[i]
    rel = delta / intron_vals[i] * 100
    ax.text(i+width*1.55, max(intron_vals[i], donor_vals[i], acc_vals[i])*0.55,
            f"Δ={delta:.2f}\n({rel:.1f}%)", ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", alpha=0.8))
label_panel(ax, "(d)", x=-0.16)

plt.tight_layout()
save(fig, OUT / "F2_splice_universality")
print(f"saved {OUT/'F2_splice_universality.pdf'} and .png")

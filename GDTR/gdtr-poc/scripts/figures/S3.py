"""S3 — Variant Δ-feature 32-layer heatmap + LR coefficients.

2 panels:
(a) Mean ΔD per layer, stratified by category (P_LP vs B_LB), for cos and jsd lenses.
(b) Logistic regression coefficient magnitudes per layer (which layers does the classifier rely on).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _figstyle import setup, COLORS, label_panel, grid_y, save

setup()
RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "figures_v2"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(RES / "phase3_ensemble" / "variants_features_full.csv")
train = df[df["category"].isin(["P_LP", "B_LB"])].copy()
y = (train["category"] == "P_LP").astype(int).values

L = 32
cos_cols = [f"dD_cos_{l}" for l in range(L)]
jsd_cols = [f"dD_jsd_{l}" for l in range(L)]

# Mean per layer per class
mean_cos_p = np.array([train.loc[y==1, c].mean() for c in cos_cols])
mean_cos_b = np.array([train.loc[y==0, c].mean() for c in cos_cols])
mean_jsd_p = np.array([train.loc[y==1, c].mean() for c in jsd_cols])
mean_jsd_b = np.array([train.loc[y==0, c].mean() for c in jsd_cols])

mean_abs_cos_p = np.array([train.loc[y==1, c].abs().mean() for c in cos_cols])
mean_abs_cos_b = np.array([train.loc[y==0, c].abs().mean() for c in cos_cols])
mean_abs_jsd_p = np.array([train.loc[y==1, c].abs().mean() for c in jsd_cols])
mean_abs_jsd_b = np.array([train.loc[y==0, c].abs().mean() for c in jsd_cols])

# LR coefficient magnitudes
def lr_coefs(cols):
    X = train[cols].values
    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X, y)
    return pipe.named_steps["lr"].coef_.flatten()

coef_cos = lr_coefs(cos_cols)
coef_jsd = lr_coefs(jsd_cols)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- (a) Mean |ΔD| per layer, P vs B, both lenses -----------------------------
ax = axes[0]
xs = np.arange(L)
ax.plot(xs, mean_abs_cos_p, color=COLORS["dD_cos"], linewidth=1.6, marker="o", markersize=3, label="cos · P_LP")
ax.plot(xs, mean_abs_cos_b, color=COLORS["dD_cos"], linewidth=1.2, linestyle="--", marker="o", markersize=2.5, alpha=0.55, label="cos · B_LB")
ax.plot(xs, mean_abs_jsd_p, color=COLORS["dD_jsd"], linewidth=1.6, marker="s", markersize=3, label="jsd · P_LP")
ax.plot(xs, mean_abs_jsd_b, color=COLORS["dD_jsd"], linewidth=1.2, linestyle="--", marker="s", markersize=2.5, alpha=0.55, label="jsd · B_LB")
ax.set_xlabel("layer index $l$")
ax.set_ylabel("mean $|\\Delta D|$ across variants")
ax.set_xticks(range(0, L+1, 4))
ax.legend(fontsize=7.5, ncol=2)
grid_y(ax)
ax.set_title("(a) Class-stratified mean |ΔD| per layer", fontsize=10, loc="left")
ax.axvline(29, linestyle=":", color=COLORS["highlight"], alpha=0.6)
ax.text(29.2, ax.get_ylim()[1]*0.95, "L=29", fontsize=7, color=COLORS["highlight"], fontweight="bold")
label_panel(ax, "(a)", x=-0.12)

# ---- (b) LR coefficient magnitudes per layer ---------------------------------
ax = axes[1]
width = 0.42
ax.bar(xs - width/2, np.abs(coef_cos), width, color=COLORS["dD_cos"], label="cos lens (32-d)")
ax.bar(xs + width/2, np.abs(coef_jsd), width, color=COLORS["dD_jsd"], label="jsd lens (32-d)")
ax.set_xlabel("layer index $l$")
ax.set_ylabel("|LR coefficient|  (standardized)")
ax.set_xticks(range(0, L+1, 4))
ax.legend(fontsize=8)
grid_y(ax)
ax.axvline(29, linestyle=":", color=COLORS["highlight"], alpha=0.6)
ax.text(29.2, ax.get_ylim()[1]*0.95, "L=29", fontsize=7, color=COLORS["highlight"], fontweight="bold")
ax.set_title("(b) Per-layer importance from logistic regression", fontsize=10, loc="left")
label_panel(ax, "(b)", x=-0.12)

plt.tight_layout()
save(fig, OUT / "S3_variant_layer_features")
print(f"saved {OUT/'S3_variant_layer_features.pdf'} and .png")

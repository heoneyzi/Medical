"""F5 — Mechanism case studies.

Panels (a)(b)(c): Per-layer ΔD profiles (ΔD_cos and ΔD_jsd) for one P/LP variant
   contrasted with its matched B/LB control:
      (a) BRCA1 splice (chr17:43076602 G>T) vs BRCA1 LB control (Δ=10 bp)
      (b) TP53 R175H (chr17:7674220 C>T) vs TP53 LB control (Δ=2 bp)
      (c) BRCA1 c.5266 region (chr17:43057063 G>A) vs BRCA1 LB control
Panel (d): max|ΔD_cos| comparison for the 3 P/LP variants and 3 controls,
   side-by-side with argmax_layer annotation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as fs  # noqa: E402

ROOT = Path("/path/to/ICML")
CASES = ROOT / "results/tier1_case_studies/case_studies.json"
OUT = ROOT / "results/figures_v2/F5_mechanism_cases"


def _by_id(records, vid):
    for r in records:
        if r["variant_id"] == vid:
            return r
    raise KeyError(vid)


def panel_profile(ax, p_record, c_record, title: str) -> None:
    L = np.arange(32)
    p_cos = np.array(p_record["dD_cos_per_layer"])
    p_jsd = np.array(p_record["dD_jsd_per_layer"])
    c_cos = np.array(c_record["dD_cos_per_layer"])
    c_jsd = np.array(c_record["dD_jsd_per_layer"])

    ax.plot(L, p_cos, color=fs.COLORS["dD_cos"], lw=1.6, label="P/LP ΔD_cos")
    ax.plot(L, p_jsd, color=fs.COLORS["dD_jsd"], lw=1.4, ls="--",
            label="P/LP ΔD_jsd")
    ax.plot(L, c_cos, color=fs.COLORS["control"], lw=1.0, alpha=0.85,
            label="B/LB ΔD_cos")
    ax.plot(L, c_jsd, color="#bbbbbb", lw=0.9, ls="--", alpha=0.85,
            label="B/LB ΔD_jsd")

    # Mark argmax layer for the P/LP variant
    am = int(p_record["argmax_layer"])
    ax.axvline(am, color=fs.COLORS["highlight"], lw=0.7, ls=":")
    y_max_abs = float(p_record["max_abs_dD"])
    ax.annotate(f"argmax L={am}\nmax|ΔD|={y_max_abs:.3f}",
                xy=(am, p_cos[am] if abs(p_cos[am]) >= abs(p_jsd[am]) else p_jsd[am]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color=fs.COLORS["highlight"])
    ax.axhline(0.0, color="#888", lw=0.6, ls=":")
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("ΔD (alt − ref)")
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, ncol=2)
    fs.grid_y(ax)


def panel_summary(ax, records) -> None:
    """Side-by-side max|ΔD_cos| for P/LP and matched B/LB controls."""
    pairs = [
        ("BRCA1 splice\n(c.5074+1)", "BRCA1_splice_43076602_G_T", "BRCA1_LB_43076592_A_G"),
        ("TP53 R175H",               "TP53_R175H_7674220_C_T",    "TP53_LB_7674222_G_A"),
        ("BRCA1 c.5266",             "BRCA1_5266_43057063_G_A",   "BRCA1_LB_43057061_C_T"),
    ]
    p_vals, c_vals, p_argmax, c_argmax, names = [], [], [], [], []
    for name, pid, cid in pairs:
        p = _by_id(records, pid)
        c = _by_id(records, cid)
        p_vals.append(p["max_abs_dD"])
        c_vals.append(c["max_abs_dD"])
        p_argmax.append(int(p["argmax_layer"]))
        c_argmax.append(int(c["argmax_layer"]))
        names.append(name)

    x = np.arange(len(pairs))
    w = 0.36
    bars_p = ax.bar(x - w / 2, p_vals, w, color=fs.COLORS["dD_cos"],
                    edgecolor="#222", lw=0.6, label="P/LP")
    bars_c = ax.bar(x + w / 2, c_vals, w, color=fs.COLORS["control"],
                    edgecolor="#222", lw=0.6, label="B/LB control")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("max |ΔD| across layers")
    ax.set_title("Pathogenic vs matched-control mechanism strength")
    ax.legend(loc="upper right", frameon=False)

    # Annotate argmax layers above bars
    for xi, v, lay in zip(x - w / 2, p_vals, p_argmax):
        ax.text(xi, v + 0.0015, f"L{lay}", ha="center", va="bottom",
                fontsize=7, color=fs.COLORS["dD_cos"])
    for xi, v, lay in zip(x + w / 2, c_vals, c_argmax):
        ax.text(xi, v + 0.0015, f"L{lay}", ha="center", va="bottom",
                fontsize=7, color="#555")
    ax.set_ylim(0, max(p_vals) * 1.25)
    fs.grid_y(ax)


def main() -> None:
    fs.setup()
    data = json.loads(CASES.read_text())
    records = data["variants"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.5))
    panel_profile(axes[0, 0],
                  _by_id(records, "BRCA1_splice_43076602_G_T"),
                  _by_id(records, "BRCA1_LB_43076592_A_G"),
                  "BRCA1 splice region (chr17:43,076,602 G>T)")
    panel_profile(axes[0, 1],
                  _by_id(records, "TP53_R175H_7674220_C_T"),
                  _by_id(records, "TP53_LB_7674222_G_A"),
                  "TP53 R175H — c.524G>A (chr17:7,674,220 C>T)")
    panel_profile(axes[1, 0],
                  _by_id(records, "BRCA1_5266_43057063_G_A"),
                  _by_id(records, "BRCA1_LB_43057061_C_T"),
                  "BRCA1 c.5266 region (chr17:43,057,063 G>A)")
    panel_summary(axes[1, 1], records)
    for ax, lab in zip(axes.flat, list("abcd")):
        fs.label_panel(ax, f"({lab})")
    fig.suptitle("Mechanism case studies: per-layer ΔD profiles for 3 ClinVar P/LP vs B/LB controls",
                 y=1.01, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fs.save(fig, OUT)
    print(f"saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate the portfolio figures from the saved result files in ../results/.

Every plotted number is read from a saved JSON file listed next to each figure below.
Requires numpy + matplotlib.  Run from anywhere:  python assets/make_figures.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

# --- palette (validated: blue/orange adjacent pair passes CVD + normal-vision floors) ---
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY_DARK, GRAY_LIGHT = "#898781", "#c3c2b7"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DPI = 160

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def load(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def footer(fig, text):
    fig.text(0.01, 0.012, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def dot(ax, x, y, color, marker="o", size=8.5, z=4):
    ax.plot([x], [y], marker=marker, markersize=size, color=color, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=z, linestyle="none")


# ------------------------------------------------------------------ data
def hard_v2_flow():
    arms = ["open_loop", "receding_horizon", "compute_matched_blind_replanning", "receding_horizon_verifier_stop_guard"]
    per_seed = {}
    for s in (17, 29, 43):
        m = load(f"hard_v2/final_reports/state_flow_seed{s}_test.json")["metrics"]
        per_seed[s] = {a: m[a]["goal_completion_rate"] for a in arms}
    return arms, per_seed


def geoacmg_flow():
    r7 = load("geoacmg_final/findings/R7_flow_dev_root_only.json")
    return {"open_loop": r7["open_loop"]["goal_completion_rate"],
            "receding_horizon": r7["receding_horizon"]["goal_completion_rate"],
            "compute_matched_blind_replanning": r7["compute_matched_blind_replanning"]["goal_completion_rate"],
            "receding_horizon_verifier_stop_guard": r7["receding_horizon_verifier_stop_guard"]["goal_completion_rate"]}


def seven_seed_rows():
    r2f = load("geoacmg_working_findings/R2f_readout_7seeds.json")["by_readout"]
    r3c = load("geoacmg_working_findings/R3c_planning_axis_7seeds.json")["axes"]
    rows = []
    for key, label in (("own_energy", "Ordering · own energy"), ("cos_std", "Ordering · standardized cosine"),
                       ("l2", "Ordering · L2"), ("cos_whiten", "Ordering · whitening + cosine")):
        r = r2f[key]
        rows.append((label, r["seed"]["est"], r["seed"]["ci"], r["gene"]["ci"]))
    for key, label in (("policy", "Planning · policy accuracy"), ("joint", "Planning · joint STOP/action"),
                       ("regret@1", "Planning · regret@1 (lower = cosine better)")):
        r = r3c[key]
        rows.append((label, r["seed"]["est"], r["seed"]["ci"], r["gene"]["ci"]))
    return rows


# ------------------------------------------------------------------ figures
def fig_hero():
    arms, per_seed = hard_v2_flow()
    hv = {a: mean(per_seed[s][a] for s in per_seed) for a in arms}
    gv = geoacmg_flow()
    rows = seven_seed_rows()

    fig = plt.figure(figsize=(10, 4.9))
    ax1 = fig.add_axes([0.2, 0.15, 0.255, 0.70])
    ax2 = fig.add_axes([0.665, 0.15, 0.25, 0.70])

    # left: replanning (horizontal bars, labels carry identity; blue = emphasis)
    conds = [("open_loop", "one-shot plan", GRAY_DARK), ("receding_horizon", "observe → replan", BLUE),
             ("compute_matched_blind_replanning", "blind replan (same compute)", GRAY_LIGHT)]
    groups = [("hard-v2 · held-out synthetic test · 48 tasks × 3 seeds", hv),
              ("GeoACMG · ClinGen tasks · 584-task development split", gv)]
    ypos, ylabels, y = [], [], 0.0
    for gname, vals in groups:
        ax1.text(0.005, y + 0.05, gname, ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
        for key, lab, col in conds:
            y += 0.62
            v = vals[key]
            ax1.barh(y, v, height=0.46, color=col, zorder=3)
            ax1.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=9.5,
                     color=INK if key == "receding_horizon" else INK2,
                     fontweight="bold" if key == "receding_horizon" else "normal")
            ypos.append(y); ylabels.append(lab)
        y += 0.95
    ax1.set_yticks(ypos, ylabels, fontsize=9)
    ax1.set_ylim(y - 0.5, -0.4)
    ax1.set_xlim(0, 0.72)
    ax1.set_xlabel("goal-completion rate (task roots)")
    ax1.set_title("Observe → replan is the dependable win", x=-0.66)
    style_axis(ax1, "x")

    # right: 7-seed contrasts (seed-clustered CI)
    short = ["own energy", "standardized cosine", "L2", "whitening + cosine",
             "policy accuracy", "joint STOP/action", "regret@1 (lower is better)"]
    ys = [7.6, 6.6, 5.6, 4.6, 2.6, 1.6, 0.6]
    for yy, (_label, est, sci, _gci) in zip(ys, rows):
        excl = sci[0] > 0 or sci[1] < 0
        col = BLUE if excl else GRAY_DARK
        ax2.plot(sci, [yy, yy], color=col, linewidth=2.2, solid_capstyle="round", zorder=3)
        dot(ax2, est, yy, col)
        ax2.text(1.02, yy, f"{est:+.3f}", va="center", ha="left", fontsize=9, color=INK2,
                 transform=ax2.get_yaxis_transform())
    ax2.axvline(0, color=INK2, linewidth=1, zorder=2)
    ax2.axhline(3.75, color=GRID, linewidth=1)
    ax2.set_yticks(ys, short, fontsize=9)
    for yy, txt in ((8.35, "Ordering accuracy, by readout"), (3.35, "Planning metrics")):
        ax2.text(0.015, yy, txt, ha="left", va="center", fontsize=9, fontweight="bold", zorder=6,
                 transform=ax2.get_yaxis_transform(), bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax2.set_ylim(0, 8.8)
    ax2.set_xlim(-0.1, 0.2)
    ax2.set_xlabel("cosine − Euclidean, 7 seeds (95% seed CI)")
    ax2.set_title("…the 'best' geometry depends on the readout", x=-0.62)
    style_axis(ax2, "x")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", linewidth=2, markeredgecolor=SURFACE, label="CI excludes 0"),
                        Line2D([0], [0], color=GRAY_DARK, marker="o", linewidth=2, markeredgecolor=SURFACE, label="CI includes 0")],
               loc="upper right", fontsize=8.5, ncol=2, handlelength=1.8, bbox_to_anchor=(0.995, 0.995))
    footer(fig, "Drawn by assets/make_figures.py from the saved result JSON. "
                "Left: hard-v2 held-out test and GeoACMG R7. Right: GeoACMG seven-seed contrasts (R2f/R3c).")
    fig.savefig(OUT / "hero.png", dpi=DPI)
    plt.close(fig)


def fig_hard_v2():
    summ = load("hard_v2/final_reports/value_test/summary.json")["runs"]
    sel = [r for r in summ if r["seed"] in (17, 29, 43)]
    ref = [r for r in summ if r["seed"] == 19][0]
    metrics = [("policy_optimal_set_accuracy", "policy acc. ↑"), ("joint_stop_action_accuracy", "joint STOP/action ↑"),
               ("regret_at_1", "regret@1 ↓")]
    arms, per_seed = hard_v2_flow()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.27, wspace=0.28)
    w = 0.34
    for i, (key, lab) in enumerate(metrics):
        vals = [r["metrics"][key] for r in sel]
        m = mean(vals)
        ax1.bar(i - w / 2 - 0.02, m, width=w, color=BLUE, zorder=3)
        for k, v in enumerate(vals):
            dot(ax1, i - w / 2 - 0.02 + (k - 1) * 0.07, v, INK2, size=5.5, z=5)
        ax1.text(i - w / 2 - 0.02, max(vals) + 0.035, f"{m:.3f}", ha="center", fontsize=9.5, fontweight="bold")
        rv = ref["metrics"][key]
        ax1.bar(i + w / 2 + 0.02, rv, width=w, color=ORANGE, zorder=3)
        ax1.text(i + w / 2 + 0.02, rv + 0.02, f"{rv:.3f}", ha="center", fontsize=9.5, color=INK2)
    ax1.set_xticks(range(3), [m[1] for m in metrics])
    ax1.set_ylim(0, 1.0)
    ax1.set_title("State-level test metrics (48 held-out tasks)")
    style_axis(ax1)
    ax1.legend(handles=[Patch(color=BLUE, label="MedCPT-only + cosine + 2× head (mean of 3 seeds; dots = seeds)"),
                        Patch(color=ORANGE, label="pre-registered full + DAgger reference (1 checkpoint)")],
               loc="upper left", bbox_to_anchor=(-0.02, -0.12), fontsize=8.3)

    names = [("open_loop", "one-shot"), ("receding_horizon", "observe →\nreplan"),
             ("compute_matched_blind_replanning", "blind replan\n(same compute)"),
             ("receding_horizon_verifier_stop_guard", "verifier STOP\n(upper control)")]
    for i, (key, lab) in enumerate(names):
        vals = [per_seed[s][key] for s in (17, 29, 43)]
        m = mean(vals)
        col = BLUE if key == "receding_horizon" else (GRAY_LIGHT if "verifier" in key else GRAY_DARK)
        ax2.bar(i, m, width=0.5, color=col, zorder=3)
        for k, v in enumerate(vals):
            dot(ax2, i + (k - 1) * 0.1, v, INK2, size=5.5, z=5)
        ax2.text(i, max(vals) + 0.022, f"{m:.3f}", ha="center", fontsize=9.5,
                 fontweight="bold" if key == "receding_horizon" else "normal")
    ax2.set_xticks(range(4), [n[1] for n in names], fontsize=9)
    ax2.set_ylim(0, 0.6)
    ax2.set_ylabel("goal-completion rate")
    ax2.set_title("State Flow planner, test task roots")
    ax2.text(0, -0.3, "bars = mean of seeds 17/29/43; dots = individual seeds", transform=ax2.transAxes, fontsize=8.3, color=INK2)
    style_axis(ax2)
    footer(fig, "Drawn by assets/make_figures.py from results/hard_v2/final_reports/ (value_test/summary.json, "
                "state_flow_seed*_test.json). Synthetic benchmark, one-time test.")
    fig.savefig(OUT / "hard_v2_test.png", dpi=DPI)
    plt.close(fig)


def fig_geoacmg_dev():
    r2 = load("geoacmg_final/findings/R2_readout_decomposition.json")
    acc = {row["readout"]: row["accuracy_tie_corrected"] for row in r2["readouts"]}
    fams = [("euclidean", "Euclid-\nean"), ("cosine", "cosine"), ("poincare", "Poin-\ncaré"),
            ("directed_quasimetric", "directed\nquasi-\nmetric"), ("pair_mlp", "pair\nMLP")]
    raw = [("raw_cos", "raw\ncosine"), ("raw_l2", "raw\nL2"), ("raw_dot", "raw\ndot")]
    own = {f: mean(acc[f"own_energy_{f}-{s}"] for s in (17, 29, 43)) for f, _ in fams}

    runs = {}
    for p in glob.glob(str(RES / "geoacmg_latest_checkpoints/geometry_comparison/*/seed-*/value_metrics.json")):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.setdefault(d["energy"], []).append(d["metrics"]["dev"]["policy_optimal_set_accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.26, wspace=0.34)
    xs, labs = [], []
    for i, (k, lab) in enumerate(raw):
        ax1.bar(i, acc[k], width=0.55, color=GRAY_LIGHT, zorder=3)
        ax1.text(i, acc[k] + 0.012, f"{acc[k]:.3f}", ha="center", fontsize=9, color=INK2)
        xs.append(i); labs.append(lab)
    for j, (f, lab) in enumerate(fams):
        x = len(raw) + 0.6 + j
        ax1.bar(x, own[f], width=0.55, color=BLUE, zorder=3)
        ax1.text(x, own[f] + 0.012, f"{own[f]:.3f}", ha="center", fontsize=9, fontweight="bold")
        xs.append(x); labs.append(lab)
    ax1.axhline(0.5, color=INK2, linewidth=1, zorder=4)
    ax1.text(2.8, 0.515, "chance", va="bottom", ha="center", fontsize=8.5, color=INK2)
    ax1.set_xticks(xs, labs, fontsize=8.6)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("depth-matched ordering accuracy")
    ax1.set_title("Ordering: raw distances ≈ chance, learned heads vary")
    style_axis(ax1)
    ax1.legend(handles=[Patch(color=GRAY_LIGHT, label="frozen MedCPT space, raw metric"),
                        Patch(color=BLUE, label="learned head, own energy (seeds 17/29/43)")],
               loc="upper left", fontsize=8.3, bbox_to_anchor=(0, 1.0))

    for j, (f, lab) in enumerate(fams):
        vals = runs[f]
        y = j
        ax2.plot([min(vals), max(vals)], [y, y], color=GRAY_LIGHT, linewidth=2, zorder=2)
        for v in vals:
            dot(ax2, v, y, GRAY_DARK, size=6, z=3)
        dot(ax2, mean(vals), y, BLUE, size=10, z=5)
        ax2.text(0.8935, y, f"{mean(vals):.4f}", va="center", ha="right", fontsize=9)
    names = {"euclidean": "Euclidean", "cosine": "cosine", "poincare": "Poincaré",
             "directed_quasimetric": "directed quasimetric", "pair_mlp": "pair MLP"}
    ax2.set_yticks(range(len(fams)), [names[f] for f, _ in fams], fontsize=9)
    ax2.set_xlim(0.83, 0.895)
    ax2.set_ylim(-0.6, len(fams) - 0.4)
    ax2.set_xlabel("dev policy accuracy (584 tasks)")
    ax2.set_title("Planning: policy barely differs")
    style_axis(ax2, "x")
    ax2.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, linestyle="none", markersize=9, markeredgecolor=SURFACE, label="mean of 3 seeds"),
                        Line2D([0], [0], marker="o", color=GRAY_DARK, linestyle="none", markersize=6, markeredgecolor=SURFACE, label="single seed")],
               loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=2, fontsize=8.3)
    footer(fig, "Drawn by assets/make_figures.py from R2_readout_decomposition.json and "
                "geometry_comparison/*/value_metrics.json; 584-task development split.")
    fig.savefig(OUT / "geoacmg_dev_dissociation.png", dpi=DPI)
    plt.close(fig)


def fig_seven_seed():
    rows = seven_seed_rows()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.88, bottom=0.27)
    n = len(rows)
    for i, (label, est, sci, gci) in enumerate(rows):
        y = n - 1 - i
        ax.plot(sci, [y + 0.13, y + 0.13], color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=3)
        dot(ax, est, y + 0.13, BLUE)
        ax.plot(gci, [y - 0.13, y - 0.13], color=ORANGE, linewidth=2.2, solid_capstyle="round", zorder=3)
        dot(ax, est, y - 0.13, ORANGE, marker="D", size=7.5)
        ax.text(0.235, y, f"{est:+.4f}", va="center", ha="right", fontsize=9.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1, zorder=2)
    ax.axhline(2.5, color=GRID, linewidth=1)
    ax.set_yticks(range(n), [r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.1, 0.24)
    ax.set_xlabel("cosine − Euclidean difference (GeoACMG dev, seeds 17/29/43/59/71/83/97)")
    ax.set_title("Seven-seed contrasts: two uncertainty units, two different stories")
    style_axis(ax, "x")
    ax.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", linewidth=2, markeredgecolor=SURFACE, label="95% CI clustered by seed (7): retraining stability"),
                       Line2D([0], [0], color=ORANGE, marker="D", linewidth=2, markeredgecolor=SURFACE, label="95% CI clustered by gene (47): across-gene spread")],
              loc="upper left", bbox_to_anchor=(-0.42, -0.17), ncol=2, fontsize=8.5)
    footer(fig, "Drawn by assets/make_figures.py from R2f_readout_7seeds.json and R3c_planning_axis_7seeds.json "
                "(stored BCa intervals; 584-task development split).")
    fig.savefig(OUT / "geoacmg_7seed_contrasts.png", dpi=DPI)
    plt.close(fig)


def fig_readout_ranking():
    r10 = load("geoacmg_working_findings/R10_readout_dependent_ranking.json")
    fams = [("cosine", "cosine"), ("euclidean", "Euclidean"), ("directed_quasimetric", "directed quasimetric"),
            ("poincare", "Poincaré"), ("pair_mlp", "pair MLP")]
    reads = [("own_energy", "own\nenergy"), ("cos", "cosine"), ("l2", "L2"), ("neg_dot", "−dot"),
             ("cos_std", "standard.\ncosine"), ("cos_whiten", "whitened\ncosine")]
    data = [[r10["by_readout"][r]["accuracies"][f] for r, _ in reads] for f, _ in fams]
    lo, hi = 0.35, 0.85
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.29, right=0.98, top=0.8, bottom=0.2)
    ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for j, (r, _) in enumerate(reads):
        best = r10["by_readout"][r]["best"]
        for i, (f, _) in enumerate(fams):
            v = data[i][j]
            frac = (v - lo) / (hi - lo)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if frac > 0.55 else INK, fontweight="bold" if f == best else "normal")
            if f == best:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.2))
    ax.set_xticks(range(len(reads)), [r[1] for r in reads], fontsize=9)
    ax.set_yticks(range(len(fams)), [f[1] for f in fams], fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Readout decides the 'best' geometry\n(outlined = best per readout; seeds 17/29/43)", fontsize=11.5)
    footer(fig, "From results/geoacmg_working_findings/R10_readout_dependent_ranking.json\n(depth-matched ordering accuracy, development split).")
    fig.savefig(OUT / "geoacmg_readout_ranking.png", dpi=DPI)
    plt.close(fig)


def fig_ablation():
    r11 = load("geoacmg_working_findings/R11_medcpt_ablation.json")["axes"]
    rows = [("policy", "policy accuracy"), ("joint", "joint STOP/action"), ("regret@1", "regret@1 (higher = worse)")]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.83, bottom=0.3)
    n = len(rows)
    for i, (k, lab) in enumerate(rows):
        y = n - 1 - i
        a = r11[k]
        ax.plot(a["run_clustered"]["ci"], [y + 0.12, y + 0.12], color=BLUE, linewidth=2.2, solid_capstyle="round", zorder=3)
        dot(ax, a["run_clustered"]["est"], y + 0.12, BLUE)
        ax.plot(a["gene_clustered"]["ci"], [y - 0.12, y - 0.12], color=ORANGE, linewidth=2.2, solid_capstyle="round", zorder=3)
        dot(ax, a["gene_clustered"]["est"], y - 0.12, ORANGE, marker="D", size=7.5)
        ax.text(0.095, y, f"{a['run_clustered']['est']:+.4f}", va="center", ha="right", fontsize=9.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(n), [r[1] for r in rows][::-1], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.045, 0.097)
    ax.set_xlabel("MedCPT removed − included (dev)")
    ax.set_title("Zeroing the frozen MedCPT view\n(cosine/Euclidean × seeds 17/29, retrained)", fontsize=11.5)
    style_axis(ax, "x")
    ax.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", linewidth=2, markeredgecolor=SURFACE, label="95% CI by run (4)"),
                       Line2D([0], [0], color=ORANGE, marker="D", linewidth=2, markeredgecolor=SURFACE, label="95% CI by gene (47)")],
              loc="upper left", bbox_to_anchor=(-0.5, -0.19), ncol=2, fontsize=8.5)
    footer(fig, "From results/geoacmg_working_findings/R11_medcpt_ablation.json\n(stored BCa intervals; recomputed by verification/verify_portfolio.py).")
    fig.savefig(OUT / "geoacmg_medcpt_ablation.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_hard_v2()
    fig_geoacmg_dev()
    fig_seven_seed()
    fig_readout_ranking()
    fig_ablation()
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")

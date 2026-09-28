"""Converging-tail GDTR ablation.

Streams cached D_cos HDF5 files and writes all new artifacts under
results/tail_ablation without modifying existing phase outputs.

Default run:
  python3 scripts/60_tail_ablation.py

Smoke run:
  python3 scripts/60_tail_ablation.py --max-windows 10

Optional perturbation rerun (GPU/model required):
  python3 scripts/60_tail_ablation.py --run-perturbation
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("GDTR_ROOT", Path(__file__).resolve().parents[1])).expanduser()
OUT_DIR = ROOT / "results" / "tail_ablation"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PERT_DIR = ROOT / "results" / "exp2_shuffled_tail"
SEED = 42
RHO_GRID = (0.80, 0.85, 0.90, 0.95)
CTX = {
    0: "intergenic",
    1: "intron",
    2: "coding_exon",
    3: "5utr",
    4: "3utr",
    5: "splice_donor",
    6: "splice_acceptor",
    7: "repeat",
}
MIN_CONTEXTS = ("splice_donor", "splice_acceptor", "intron", "coding_exon", "intergenic")

LOG = logging.getLogger("tail_ablation")


def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUT_DIR / "tail_ablation.log"),
        ],
    )


def metric_specs(rhos: tuple[float, ...]) -> list[dict[str, Any]]:
    specs = [{"name": "c_first", "kind": "first", "rho": None}]
    specs.append({"name": "c_tail_strict", "kind": "strict", "rho": None})
    for rho in rhos:
        specs.append({
            "name": f"c_tail_soft_rho{int(round(rho * 100)):02d}",
            "kind": "soft",
            "rho": float(rho),
        })
    return specs


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_frozen_gamma() -> float:
    path = ROOT / "results" / "phase1.4" / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen calibration: {path}")
    gamma = read_json(path).get("gamma_cos_global_q70")
    if gamma is None:
        raise RuntimeError(f"missing gamma_cos_global_q70 in {path}")
    return float(gamma)


def compute_recalibrated_gammas(specs: list[dict[str, Any]], quantile: float = 0.70) -> dict[str, float]:
    path = ROOT / "results" / "phase1.1" / "lens_traces.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 1.1 calibration traces: {path}")
    traces = np.load(path, allow_pickle=True)
    D = np.asarray(traces["D_cos"], dtype=np.float32)
    if D.ndim != 3:
        raise ValueError(f"expected D_cos [N,L,T], got {D.shape}")
    L = D.shape[1]
    penult = L - 2
    suffix = D[:, penult:, :]
    out: dict[str, float] = {}
    for spec in specs:
        if spec["kind"] == "first":
            stat = D[:, penult, :]
        elif spec["kind"] == "strict":
            stat = np.max(suffix, axis=1)
        else:
            rho = float(spec["rho"])
            suffix_len = suffix.shape[1]
            kth = max(1, min(suffix_len, int(math.ceil(rho * suffix_len)))) - 1
            stat = np.partition(suffix, kth, axis=1)[:, kth, :]
        out[spec["name"]] = float(np.quantile(stat.reshape(-1), quantile))
    return out


def first_depth_np(D: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    rmin = np.minimum.accumulate(D, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx = below.argmax(axis=0)
    L = D.shape[0]
    c = np.where(any_below, first_idx + 1, L).astype(np.int16)
    return c, ~any_below


def strict_tail_depth_np(D: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    suffix_max = np.maximum.accumulate(D[::-1], axis=0)[::-1]
    ok = suffix_max <= gamma
    any_ok = ok.any(axis=0)
    first_idx = ok.argmax(axis=0)
    L = D.shape[0]
    c = np.where(any_ok, first_idx + 1, L + 1).astype(np.int16)
    return c, ~any_ok


def soft_tail_depth_np(D: np.ndarray, gamma: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0,1], got {rho}")
    inside = (D <= gamma).astype(np.int16)
    suffix_count = np.cumsum(inside[::-1], axis=0)[::-1]
    L = D.shape[0]
    suffix_len = np.arange(L, 0, -1, dtype=np.float32)[:, None]
    ok = (suffix_count / suffix_len) >= rho
    any_ok = ok.any(axis=0)
    first_idx = ok.argmax(axis=0)
    c = np.where(any_ok, first_idx + 1, L + 1).astype(np.int16)
    return c, ~any_ok


def compute_depth(D: np.ndarray, spec: dict[str, Any], gamma: float) -> tuple[np.ndarray, np.ndarray]:
    if spec["kind"] == "first":
        return first_depth_np(D, gamma)
    if spec["kind"] == "strict":
        return strict_tail_depth_np(D, gamma)
    return soft_tail_depth_np(D, gamma, float(spec["rho"]))


def empty_hist(L: int) -> np.ndarray:
    return np.zeros(L + 2, dtype=np.int64)


def add_values_to_hist(hist: np.ndarray, values: np.ndarray) -> None:
    hist += np.bincount(values.astype(np.int64), minlength=hist.size)[: hist.size]


def hist_n(hist: np.ndarray) -> int:
    return int(hist.sum())


def hist_mean(hist: np.ndarray) -> float:
    n = hist_n(hist)
    if n == 0:
        return float("nan")
    values = np.arange(hist.size, dtype=np.float64)
    return float(np.dot(values, hist) / n)


def hist_var_sample(hist: np.ndarray) -> float:
    n = hist_n(hist)
    if n < 2:
        return float("nan")
    values = np.arange(hist.size, dtype=np.float64)
    mean = hist_mean(hist)
    ss = float(np.dot((values - mean) ** 2, hist))
    return ss / (n - 1)


def hist_std(hist: np.ndarray) -> float:
    var = hist_var_sample(hist)
    return float(math.sqrt(var)) if np.isfinite(var) else float("nan")


def hist_quantile(hist: np.ndarray, q: float) -> float:
    n = hist_n(hist)
    if n == 0:
        return float("nan")
    target = max(1, int(math.ceil(q * n)))
    return float(np.searchsorted(np.cumsum(hist), target, side="left"))


def hist_summary(hist: np.ndarray, L: int, unresolved_count: int) -> dict[str, Any]:
    n = hist_n(hist)
    return {
        "n": n,
        "mean": hist_mean(hist),
        "median": hist_quantile(hist, 0.50),
        "std": hist_std(hist),
        "q10": hist_quantile(hist, 0.10),
        "q90": hist_quantile(hist, 0.90),
        "unresolved_fraction": float(unresolved_count / n) if n else float("nan"),
        "frac_final_or_lplus": float(hist[L:].sum() / n) if n else float("nan"),
    }


def cohens_d_hist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = hist_n(a), hist_n(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = hist_var_sample(a), hist_var_sample(b)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0 or not np.isfinite(sp):
        return float("nan")
    return float((hist_mean(a) - hist_mean(b)) / sp)


def mwu_hist(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Histogram Mann-Whitney U with tie-corrected normal p-value.

    Returns U_a, two-sided p, and rank-biserial effect positive when a > b.
    """
    na, nb = hist_n(a), hist_n(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), float("nan")
    combined = a + b
    n = na + nb
    cum_prev = np.concatenate(([0], np.cumsum(combined)[:-1]))
    avg_rank = cum_prev + (combined + 1) / 2.0
    rank_sum_a = float(np.dot(a, avg_rank))
    u_a = rank_sum_a - na * (na + 1) / 2.0
    rank_biserial = (2.0 * u_a / (na * nb)) - 1.0
    if n < 2:
        return float(u_a), float("nan"), float(rank_biserial)
    tie_term = float(np.sum(combined.astype(np.float64) ** 3 - combined.astype(np.float64)))
    variance = na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        p = float("nan")
    else:
        z = (u_a - na * nb / 2.0) / math.sqrt(variance)
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(u_a), float(p), float(rank_biserial)


def spearman_from_pairs(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(xs, ys).statistic)
    except Exception:
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_empty_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def collect_chromosome(
    chrom: str,
    h5_path: Path,
    labels_path: Path,
    specs: list[dict[str, Any]],
    gamma_sets: dict[str, dict[str, float]],
    max_windows: int | None,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    import h5py

    if not h5_path.exists():
        raise FileNotFoundError(f"missing {chrom} cache: {h5_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing {chrom} labels: {labels_path}")
    labels = np.load(labels_path)
    accum: dict[str, Any] = {}
    scatter: list[dict[str, Any]] = []
    examples: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(SEED)

    with h5py.File(h5_path, "r") as h5:
        N, L, _ = h5["D_cos"].shape
        for cal_name, gammas in gamma_sets.items():
            for spec in specs:
                key = (cal_name, spec["name"])
                accum[key] = {
                    "all_hist": empty_hist(L),
                    "all_unresolved": 0,
                    "context_hists": {name: empty_hist(L) for name in CTX.values()},
                    "context_unresolved": {name: 0 for name in CTX.values()},
                    "gamma": gammas[spec["name"]],
                    "L": L,
                }
        done_mask = h5["done_mask"][:] if "done_mask" in h5 else np.ones(N, dtype=np.uint8)
        starts = h5["starts"][:]
        ends = h5["ends"][:]
        n_seen = 0
        for i in range(N):
            if not done_mask[i]:
                continue
            if max_windows is not None and n_seen >= max_windows:
                break
            D = h5["D_cos"][i].astype(np.float32)
            s, e = int(starts[i]), int(ends[i])
            lab_slice = labels[s:e]
            if lab_slice.shape[0] != D.shape[1]:
                lab_slice = lab_slice[: D.shape[1]]
                D = D[:, : lab_slice.shape[0]]
            depths_for_diagnostics: dict[str, np.ndarray] = {}
            unresolved_for_diagnostics: dict[str, np.ndarray] = {}
            for cal_name, gammas in gamma_sets.items():
                for spec in specs:
                    c, unresolved = compute_depth(D, spec, gammas[spec["name"]])
                    item = accum[(cal_name, spec["name"])]
                    add_values_to_hist(item["all_hist"], c)
                    item["all_unresolved"] += int(unresolved.sum())
                    for code, ctx in CTX.items():
                        sel = lab_slice == code
                        if sel.any():
                            add_values_to_hist(item["context_hists"][ctx], c[sel])
                            item["context_unresolved"][ctx] += int(unresolved[sel].sum())
                    if cal_name == "frozen":
                        depths_for_diagnostics[spec["name"]] = c
                        unresolved_for_diagnostics[spec["name"]] = unresolved

            if "c_first" in depths_for_diagnostics and "c_tail_soft_rho85" in depths_for_diagnostics:
                first = depths_for_diagnostics["c_first"]
                soft = depths_for_diagnostics["c_tail_soft_rho85"]
                strict = depths_for_diagnostics.get("c_tail_strict")
                if len(scatter) < sample_limit and chrom == "chr22":
                    remaining = sample_limit - len(scatter)
                    take = min(remaining, max(1, D.shape[1] // 500), D.shape[1])
                    idx = rng.choice(D.shape[1], size=take, replace=False)
                    for j in idx:
                        scatter.append({
                            "context": CTX.get(int(lab_slice[j]), "unknown"),
                            "c_first": int(first[j]),
                            "c_tail_soft_rho85": int(soft[j]),
                        })
                if chrom == "chr22":
                    candidates = {
                        "first_early_tail_late": np.where((first <= 10) & (soft >= min(L, 25)))[0],
                        "first_tail_agree": np.where(np.abs(first.astype(np.int32) - soft.astype(np.int32)) <= 1)[0],
                        "strict_final_or_unresolved": np.where(strict >= L if strict is not None else np.zeros_like(first, dtype=bool))[0],
                    }
                    for name, idxs in candidates.items():
                        if name not in examples and idxs.size:
                            j = int(idxs[0])
                            examples[name] = {
                                "D_cos": D[:, j].copy(),
                                "context": CTX.get(int(lab_slice[j]), "unknown"),
                                "window_idx": np.asarray(i),
                                "position": np.asarray(s + j),
                                "c_first": np.asarray(int(first[j])),
                                "c_tail_soft_rho85": np.asarray(int(soft[j])),
                                "c_tail_strict": np.asarray(int(strict[j])) if strict is not None else np.asarray(-1),
                            }
            n_seen += 1
            if n_seen % 1000 == 0:
                LOG.info("%s processed %d windows", chrom, n_seen)
    return accum, scatter, examples


def build_tables(
    chrom_accums: dict[str, dict[str, Any]],
    specs: list[dict[str, Any]],
    gamma_sets: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dist_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []

    for chrom, accum in chrom_accums.items():
        for cal_name in gamma_sets:
            for spec in specs:
                name = spec["name"]
                item = accum[(cal_name, name)]
                L = item["L"]
                summ = hist_summary(item["all_hist"], L, item["all_unresolved"])
                dist_rows.append({
                    "chrom": chrom,
                    "calibration": cal_name,
                    "metric": name,
                    "gamma": item["gamma"],
                    **summ,
                })
                intron = item["context_hists"]["intron"]
                for ctx in CTX.values():
                    hist = item["context_hists"][ctx]
                    n = hist_n(hist)
                    if n == 0:
                        continue
                    summ_ctx = hist_summary(hist, L, item["context_unresolved"][ctx])
                    d = cohens_d_hist(hist, intron) if ctx != "intron" else 0.0
                    u, p, r = mwu_hist(hist, intron) if ctx != "intron" else (float("nan"), float("nan"), 0.0)
                    context_rows.append({
                        "chrom": chrom,
                        "calibration": cal_name,
                        "metric": name,
                        "gamma": item["gamma"],
                        "context": ctx,
                        **summ_ctx,
                        "cohens_d_vs_intron": d,
                        "mwu_u_vs_intron": u,
                        "mwu_p_vs_intron": p,
                        "rank_biserial_vs_intron": r,
                    })

    ctx_lookup = {
        (r["chrom"], r["calibration"], r["metric"], r["context"]): r
        for r in context_rows
    }
    for cal_name in gamma_sets:
        for spec in specs:
            name = spec["name"]
            shared_contexts = [
                c for c in CTX.values()
                if ("chr22", cal_name, name, c) in ctx_lookup and ("chr17", cal_name, name, c) in ctx_lookup
            ]
            rho = spearman_from_pairs(
                [ctx_lookup[("chr22", cal_name, name, c)]["mean"] for c in shared_contexts],
                [ctx_lookup[("chr17", cal_name, name, c)]["mean"] for c in shared_contexts],
            )
            for ctx in ("splice_donor", "splice_acceptor", "coding_exon"):
                k22 = ("chr22", cal_name, name, ctx)
                k17 = ("chr17", cal_name, name, ctx)
                if k22 not in ctx_lookup or k17 not in ctx_lookup:
                    continue
                d22 = ctx_lookup[k22]["cohens_d_vs_intron"]
                d17 = ctx_lookup[k17]["cohens_d_vs_intron"]
                transfer_rows.append({
                    "calibration": cal_name,
                    "metric": name,
                    "context": ctx,
                    "chr22_effect_d": d22,
                    "chr17_effect_d": d17,
                    "effect_retention_chr17_over_chr22": float(d17 / d22) if d22 not in (0.0, float("nan")) and np.isfinite(d22) else float("nan"),
                    "context_order_spearman": rho,
                    "n_contexts_for_spearman": len(shared_contexts),
                })
    return dist_rows, context_rows, transfer_rows


def plot_outputs(
    dist_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    scatter: list[dict[str, Any]],
    examples: dict[str, dict[str, np.ndarray]],
    specs: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(7, 2.6 * len(examples)), sharex=True)
        if len(examples) == 1:
            axes = [axes]
        for ax, (name, ex) in zip(axes, examples.items()):
            y = ex["D_cos"]
            ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3, lw=1.2)
            ax.axhline(load_frozen_gamma(), color="#666666", ls="--", lw=1.0)
            ax.set_ylabel("D_cos")
            ax.set_title(
                f"{name}: {ex['context']} pos={int(ex['position'])} "
                f"first={int(ex['c_first'])} soft85={int(ex['c_tail_soft_rho85'])} "
                f"strict={int(ex['c_tail_strict'])}"
            )
        axes[-1].set_xlabel("layer")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"tail_ablation_trajectories.{ext}", dpi=160)
        plt.close(fig)

    frozen_chr22 = [r for r in dist_rows if r["chrom"] == "chr22" and r["calibration"] == "frozen"]
    if frozen_chr22:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(frozen_chr22))
        ax.bar(x, [r["mean"] for r in frozen_chr22], color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels([r["metric"] for r in frozen_chr22], rotation=25, ha="right")
        ax.set_ylabel("mean settling depth")
        ax.set_title("chr22 frozen-gamma depth distribution")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"tail_ablation_depth_means.{ext}", dpi=160)
        plt.close(fig)

    if scatter:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        contexts = sorted({r["context"] for r in scatter})
        colors = {ctx: plt.cm.tab10(i % 10) for i, ctx in enumerate(contexts)}
        for ctx in contexts:
            pts = [r for r in scatter if r["context"] == ctx]
            ax.scatter([r["c_first"] for r in pts], [r["c_tail_soft_rho85"] for r in pts],
                       s=8, alpha=0.45, color=colors[ctx], label=ctx)
        ax.set_xlabel("c_first")
        ax.set_ylabel("c_tail_soft_rho85")
        ax.legend(fontsize=7, loc="best", markerscale=2)
        ax.set_title("Sampled chr22 positions")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"tail_ablation_scatter_first_vs_soft85.{ext}", dpi=160)
        plt.close(fig)

    selected = {"c_first", "c_tail_strict", "c_tail_soft_rho85"}
    rows = [
        r for r in context_rows
        if r["chrom"] == "chr22" and r["calibration"] == "frozen" and r["metric"] in selected
        and r["context"] in MIN_CONTEXTS
    ]
    if rows:
        metrics = [s["name"] for s in specs if s["name"] in selected]
        contexts = [c for c in MIN_CONTEXTS if any(r["context"] == c for r in rows)]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharey=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            vals = []
            for ctx in contexts:
                match = [r for r in rows if r["metric"] == metric and r["context"] == ctx]
                vals.append(match[0]["mean"] if match else 0)
            ax.bar(np.arange(len(contexts)), vals, color="#72b7b2")
            ax.set_title(metric)
            ax.set_xticks(np.arange(len(contexts)))
            ax.set_xticklabels(contexts, rotation=25, ha="right")
        axes[0].set_ylabel("mean settling depth")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"tail_ablation_context_bars.{ext}", dpi=160)
        plt.close(fig)


def paired_effect(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size < 2:
        return float("nan")
    sd = float(delta.std(ddof=1))
    return float(delta.mean() / sd) if sd else float("nan")


def dinuc_shuffle(seq: str, rng: random.Random) -> str:
    n = len(seq)
    if n < 3:
        return seq
    chars = list(seq.upper())
    adj = {c: [] for c in set(chars)}
    for i in range(n - 1):
        adj[chars[i]].append(chars[i + 1])
    start = chars[0]
    for _ in range(64):
        adj_copy = {k: list(v) for k, v in adj.items()}
        for k in adj_copy:
            rng.shuffle(adj_copy[k])
        walk = eulerian_walk(adj_copy, start)
        if walk is not None and len(walk) == n:
            return "".join(walk)
    rng.shuffle(chars)
    return "".join(chars)


def eulerian_walk(adj: dict[str, list[str]], start: str) -> list[str] | None:
    stack = [start]
    walk: list[str] = []
    while stack:
        v = stack[-1]
        if adj.get(v):
            stack.append(adj[v].pop())
        else:
            walk.append(stack.pop())
    walk.reverse()
    if any(adj.get(v) for v in adj):
        return None
    return walk


def find_donor_centers(splice_labels: np.ndarray, target_label: int, chrom_seq: str) -> np.ndarray:
    mask = splice_labels == target_label
    if not mask.any():
        return np.array([], dtype=np.int64)
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size]])
    centers = []
    for a, b in zip(starts, ends):
        seg = chrom_seq[int(a):int(b)]
        gt_offsets = [i for i in range(len(seg) - 1) if seg[i:i + 2] == "GT"]
        if gt_offsets:
            mid = (b - a) // 2
            centers.append(int(a) + min(gt_offsets, key=lambda x: abs(x - mid)))
    return np.asarray(centers, dtype=np.int64)


def run_perturbation(
    specs: list[dict[str, Any]],
    gamma_sets: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Rerun the GT/flank perturbation and compute all tail metrics."""
    PERT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    import torch
    import pysam
    from scipy import stats as st
    from src.constants_evo2 import BOS_OFFSET, N_LAYERS
    from src.logit_lens_evo2 import all_layer_names, extract_hidden_states
    from src.model_loader_evo2 import load_evo2, tokenize
    from src.ur_gdtr_evo2 import cosine_lens

    rng_np = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    labels = np.load(ROOT / "data" / "annotation" / "chr22_position_labels.npy")
    fasta = pysam.FastaFile(str(ROOT / "data" / "reference" / "chr22.fa"))
    chr22_seq = fasta.fetch("chr22").upper()
    centers_all = find_donor_centers(labels, 5, chr22_seq)
    valid = (centers_all >= args.ctx_half) & (centers_all + args.ctx_half + 2 < len(chr22_seq))
    centers_valid = centers_all[valid]
    n_pick = min(args.n_donors, centers_valid.size)
    centers = centers_valid[rng_np.choice(centers_valid.size, size=n_pick, replace=False)]

    bundle = load_evo2()
    layer_names = all_layer_names()
    metrics = {
        (cal_name, spec["name"]): {
            "real": np.full(n_pick, np.nan, dtype=np.float32),
            "mut": np.full(n_pick, np.nan, dtype=np.float32),
            "shuf": np.full((n_pick, args.n_shuffles), np.nan, dtype=np.float32),
        }
        for cal_name in gamma_sets
        for spec in specs
    }

    def metric_readouts(seq: str, center_off: int) -> dict[tuple[str, str], float]:
        toks = tokenize(seq, bundle, device="cuda")
        with torch.no_grad():
            hs = extract_hidden_states(bundle, toks, save_layers=layer_names)
            D = cosine_lens(hs, n_layers=N_LAYERS, bos_offset=BOS_OFFSET).float().cpu().numpy()
        del hs
        torch.cuda.empty_cache()
        lo = max(0, center_off - args.readout_half)
        hi = min(D.shape[1], center_off + args.readout_half + 1)
        out = {}
        for cal_name, gammas in gamma_sets.items():
            for spec in specs:
                c, _ = compute_depth(D[:, lo:hi], spec, gammas[spec["name"]])
                out[(cal_name, spec["name"])] = float(np.mean(c))
        return out

    t0 = time.time()
    for k, p0 in enumerate(centers):
        p0 = int(p0)
        s = p0 - args.ctx_half
        e = p0 + args.ctx_half
        seq = chr22_seq[s:e]
        if len(seq) != 2 * args.ctx_half or seq[args.ctx_half:args.ctx_half + 2] != "GT":
            continue
        center_off = args.ctx_half
        flank_n = args.shuffle_half
        real = metric_readouts(seq, center_off)
        mut_seq = seq[:center_off] + "AA" + seq[center_off + 2:]
        mut = metric_readouts(mut_seq, center_off)
        for key, value in real.items():
            metrics[key]["real"][k] = value
        for key, value in mut.items():
            metrics[key]["mut"][k] = value
        left = seq[center_off - flank_n:center_off]
        right = seq[center_off + 2:center_off + 2 + flank_n]
        for j in range(args.n_shuffles):
            sh_seq = (
                seq[:center_off - flank_n]
                + dinuc_shuffle(left, py_rng)
                + seq[center_off:center_off + 2]
                + dinuc_shuffle(right, py_rng)
                + seq[center_off + 2 + flank_n:]
            )
            sh = metric_readouts(sh_seq, center_off)
            for key, value in sh.items():
                metrics[key]["shuf"][k, j] = value
        if (k + 1) % max(1, n_pick // 20) == 0:
            LOG.info("perturbation %d/%d in %.1fs", k + 1, n_pick, time.time() - t0)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"centers": centers}
    for (cal_name, metric), vals in metrics.items():
        real = vals["real"]
        mut = vals["mut"]
        shuf = vals["shuf"]
        valid_real = np.isfinite(real) & np.isfinite(mut) & np.isfinite(shuf).all(axis=1)
        real_v = real[valid_real]
        mut_v = mut[valid_real]
        shuf_v = shuf[valid_real]
        shuf_mean = shuf_v.mean(axis=1) if shuf_v.size else np.array([], dtype=np.float32)
        arrays[f"{cal_name}_{metric}_real"] = real
        arrays[f"{cal_name}_{metric}_mut"] = mut
        arrays[f"{cal_name}_{metric}_shuf"] = shuf
        for arm, arm_vals, delta in (
            ("GT_to_AA", mut_v, mut_v - real_v),
            ("flank_shuffled", shuf_mean, shuf_mean - real_v),
        ):
            p = float(st.wilcoxon(delta, alternative="two-sided").pvalue) if delta.size >= 5 else float("nan")
            rows.append({
                "calibration": cal_name,
                "metric": metric,
                "n": int(real_v.size),
                "arm": arm,
                "real_mean_depth": float(real_v.mean()) if real_v.size else float("nan"),
                "perturbed_mean_depth": float(arm_vals.mean()) if arm_vals.size else float("nan"),
                "delta_vs_real": float(delta.mean()) if delta.size else float("nan"),
                "paired_wilcoxon_p": p,
                "paired_effect_size_delta_over_sd": paired_effect(delta),
            })
    np.savez_compressed(PERT_DIR / "per_donor_tail_metrics.npz", **arrays)
    write_csv(PERT_DIR / "tail_perturbation_table.csv", rows)
    return rows


def write_report(
    dist_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    perturb_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    def find_ctx(metric: str, calibration: str, chrom: str, ctx: str) -> dict[str, Any] | None:
        for row in context_rows:
            if row["metric"] == metric and row["calibration"] == calibration and row["chrom"] == chrom and row["context"] == ctx:
                return row
        return None

    lines = [
        "# Converging-Tail GDTR Ablation Report",
        "",
        "Generated by `scripts/60_tail_ablation.py`.",
        "",
        "## Interpretation Questions",
    ]
    first_donor = find_ctx("c_first", "frozen", "chr22", "splice_donor")
    soft_donor = find_ctx("c_tail_soft_rho85", "frozen", "chr22", "splice_donor")
    strict_dist = next((r for r in dist_rows if r["metric"] == "c_tail_strict" and r["calibration"] == "frozen" and r["chrom"] == "chr22"), None)
    if first_donor and soft_donor:
        preserved = (
            np.isfinite(soft_donor["cohens_d_vs_intron"])
            and np.sign(soft_donor["cohens_d_vs_intron"]) == np.sign(first_donor["cohens_d_vs_intron"])
        )
        lines.append(f"1. Splice donor/acceptor signals preserved under tail metrics: {'yes' if preserved else 'no or mixed'} based on frozen chr22 soft-rho85 donor d={soft_donor['cohens_d_vs_intron']:.4g}.")
    else:
        lines.append("1. Splice donor/acceptor preservation could not be determined from available outputs.")
    if strict_dist:
        lines.append(f"2. Strict tail late-layer collapse: final-or-L+1 fraction is {strict_dist['frac_final_or_lplus']:.4f} on chr22 frozen gamma.")
    else:
        lines.append("2. Strict tail late-layer collapse could not be determined.")
    lines.extend([
        "3. Soft tail stability should be judged from the rho sweep in Table 1 and Table 2; rho85 is the primary compromise metric.",
        "4. Transient-crossing dependence should be judged by comparing c_first vs c_tail_soft_rho85 effect signs, retention, and scatter diagnostics.",
        "5. Keep or reframe the main metric according to the requested decision rule: if soft-tail signs/orderings hold, retain running-min as first-passage; otherwise reword as first-entry/first-contact.",
        "6. Include this ablation in the appendix if Table 2, Table 3, and perturbation directions are qualitatively stable.",
        "",
        "## Output Tables",
        "- Table 1: `tables/table1_distribution.csv`",
        "- Table 2: `tables/table2_context_effects.csv`",
        "- Table 3: `tables/table3_transfer.csv`",
        "- Table 4: `tables/table4_perturbation.csv` when perturbation is run, otherwise `results/exp2_shuffled_tail/tail_perturbation_table.csv`",
    ])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {w}" for w in warnings)
    (OUT_DIR / "tail_ablation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--scatter-sample", type=int, default=50000)
    parser.add_argument("--skip-chr17", action="store_true")
    parser.add_argument("--skip-recalibrated", action="store_true")
    parser.add_argument("--run-perturbation", action="store_true")
    parser.add_argument("--n-donors", type=int, default=1000)
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--ctx-half", type=int, default=3000)
    parser.add_argument("--shuffle-half", type=int, default=100)
    parser.add_argument("--readout-half", type=int, default=10)
    args = parser.parse_args()

    setup_logging()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    specs = metric_specs(RHO_GRID)
    warnings: list[str] = []
    frozen = load_frozen_gamma()
    gamma_sets = {"frozen": {spec["name"]: frozen for spec in specs}}
    if not args.skip_recalibrated:
        try:
            gamma_sets["recalibrated"] = compute_recalibrated_gammas(specs)
        except Exception as e:
            msg = f"recalibrated gamma skipped: {e}"
            warnings.append(msg)
            LOG.warning(msg)

    cal_rows = [
        {
            "calibration": cal_name,
            "metric": spec["name"],
            "gamma": gammas[spec["name"]],
            "quantile": 0.70,
            "source": "phase1.4 gamma_cos_global_q70" if cal_name == "frozen" else "phase1.1 metric-specific q70",
        }
        for cal_name, gammas in gamma_sets.items()
        for spec in specs
    ]
    write_csv(TABLE_DIR / "calibrations.csv", cal_rows)

    chrom_inputs = {
        "chr22": (
            ROOT / "results" / "phase1.6" / "chr22_cache.h5",
            ROOT / "data" / "annotation" / "chr22_position_labels.npy",
        )
    }
    if not args.skip_chr17:
        chrom_inputs["chr17"] = (
            ROOT / "results" / "phase2.1" / "chr17_cache.h5",
            ROOT / "data" / "annotation" / "chr17_position_labels.npy",
        )

    chrom_accums: dict[str, dict[str, Any]] = {}
    all_scatter: list[dict[str, Any]] = []
    all_examples: dict[str, dict[str, np.ndarray]] = {}
    for chrom, (h5_path, labels_path) in chrom_inputs.items():
        try:
            LOG.info("collecting %s from %s", chrom, h5_path)
            accum, scatter, examples = collect_chromosome(
                chrom, h5_path, labels_path, specs, gamma_sets, args.max_windows, args.scatter_sample
            )
            chrom_accums[chrom] = accum
            all_scatter.extend(scatter)
            all_examples.update({k: v for k, v in examples.items() if k not in all_examples})
        except Exception as e:
            msg = f"{chrom} skipped: {e}"
            warnings.append(msg)
            LOG.warning(msg)

    if not chrom_accums:
        raise RuntimeError("no chromosome caches were processed")

    dist_rows, context_rows, transfer_rows = build_tables(chrom_accums, specs, gamma_sets)
    write_csv(TABLE_DIR / "table1_distribution.csv", dist_rows)
    write_csv(TABLE_DIR / "table2_context_effects.csv", context_rows)
    write_csv(TABLE_DIR / "table3_transfer.csv", transfer_rows)

    perturb_rows: list[dict[str, Any]] = []
    if args.run_perturbation:
        perturb_rows = run_perturbation(specs, gamma_sets, args)
        write_csv(TABLE_DIR / "table4_perturbation.csv", perturb_rows)
    else:
        prior = PERT_DIR / "tail_perturbation_table.csv"
        if prior.exists():
            with prior.open() as f:
                perturb_rows = list(csv.DictReader(f))
            write_csv(TABLE_DIR / "table4_perturbation.csv", perturb_rows)
        else:
            warnings.append("perturbation not run; pass --run-perturbation on the GPU/H200 environment")
            write_empty_csv(TABLE_DIR / "table4_perturbation.csv", [
                "calibration",
                "metric",
                "n",
                "arm",
                "real_mean_depth",
                "perturbed_mean_depth",
                "delta_vs_real",
                "paired_wilcoxon_p",
                "paired_effect_size_delta_over_sd",
            ])

    try:
        plot_outputs(dist_rows, context_rows, all_scatter[: args.scatter_sample], all_examples, specs)
    except Exception as e:
        warnings.append(f"plots failed: {e}")
        LOG.warning("plots failed: %s", e)

    summary = {
        "root": str(ROOT),
        "out_dir": str(OUT_DIR),
        "metrics": [s["name"] for s in specs],
        "gamma_sets": gamma_sets,
        "processed_chromosomes": list(chrom_accums.keys()),
        "warnings": warnings,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(dist_rows, context_rows, transfer_rows, perturb_rows, warnings)
    LOG.info("tail ablation complete: %s", OUT_DIR)


if __name__ == "__main__":
    main()

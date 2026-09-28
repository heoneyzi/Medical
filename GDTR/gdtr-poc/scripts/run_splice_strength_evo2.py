"""Run Evo-2 on splice-strength sequences and compute gDTR settling depth c.

Input:
  results/splice_strength_spliceai/
    sequences.fa   — 4096-bp variants at 10 SpliceAI-calibrated levels
    metadata.csv   — target_level, achieved_score, site info per sequence

Pipeline per sequence:
  tokenize → Evo-2 forward → D_cos [N_LAYERS, T]
  → settling depth c [T]  (per-position, using locked γ_cos = 0.397)

Four c metrics, ordered by closeness to 16c methodology:

  c_at_site    ← PRIMARY — c at exactly the splice-site token (T//2 = 2048).
                 Directly mirrors 16c analysis_b: pos_c[donor/acceptor_center].
                 Single-position; high variance per site → aggregate over sites.

  mean_c_motif ← SECONDARY — mean c over the splice motif only.
                 Donor 9-mer: positions T//2-3 … T//2+6  (9 tokens).
                 Acceptor 23-mer: positions T//2-20 … T//2+3  (23 tokens).
                 Captures only the mutated region → best signal-to-noise.

  mean_c_center — mean c over ±CENTER_HALF bp (default ±100 bp, 200 tokens).
                  Includes motif + immediate context; lower variance than c_at_site.

  mean_c_full  — mean c over all 4096 positions.
                 Dominated by ~4073 unmutated tokens → 0.002-0.006× signal.
                 Kept for reference; NOT recommended as primary metric.

Difference vs 16c build_position_c_array:
  16c averages c across overlapping 6 kb windows (up to 2× coverage per position),
  reducing per-position variance.  Here each sequence is unique so no averaging
  across windows is possible; noise is reduced instead by aggregating c_at_site
  across sites (N_SITES=200 per splice type).

After all sequences, aggregate by (splice_type, target_level):
  mean / SEM of each metric across sites → line plot and CSV.

Checkpointing: c_per_sequence.csv is appended after each forward pass so the
run is resumable.  Re-run the script and already-processed seq_ids are skipped.

Output:
  results/splice_strength_evo2/
    c_per_sequence.csv  — one row per sequence (seq_id + all c metrics)
    c_by_level.csv      — per-(splice_type, target_level, metric) aggregated stats
    F_c_vs_level_{site,motif,center,full}.{pdf,png}  — c metric vs. splice-strength level
    F_heatmap_{donor,acceptor}.{pdf,png}  — site × level heatmap of mean_c_motif
    summary.json        — run config and Pearson r per splice type × metric

Usage (server, GPU):
  PYTHONPATH=. python scripts/run_splice_strength_evo2.py

Env-var overrides:
  GDTR_ROOT
  SPLICEAI_OUT_DIR  (default: results/splice_strength_spliceai)
  GAMMA_LOCKED      (default: 0.397)
  CENTER_HALF       (default: 100  — ±100 bp around splice site center)
"""


from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
GDTR_ROOT = ru.GDTR_ROOT

SPLICEAI_OUT_DIR = Path(os.environ.get(
    "SPLICEAI_OUT_DIR",
    str(GDTR_ROOT / "results" / "splice_strength_spliceai"),
))
FA_PATH   = SPLICEAI_OUT_DIR / "sequences.fa"
META_PATH = SPLICEAI_OUT_DIR / "metadata.csv"

OUT_DIR = GDTR_ROOT / "results" / "splice_strength_evo2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_CSV_PATH   = OUT_DIR / "c_per_sequence.csv"
LEVEL_CSV_PATH = OUT_DIR / "c_by_level.csv"
JSON_PATH      = OUT_DIR / "summary.json"

PHASE    = "splice_strength_evo2"
LOG      = ru.setup_logging(PHASE)

GAMMA_LOCKED = float(os.environ.get("GAMMA_LOCKED", "0.397"))
CENTER_HALF  = int(os.environ.get("CENTER_HALF", "100"))   # ±bp around site center

# Motif half-widths (relative to win_center, matching make_splice_strength_spliceai.py)
# Donor 9-mer:    positions [win_center-3 : win_center+6]
# Acceptor 23-mer: positions [win_center-20 : win_center+3]
MOTIF_DONOR_LO    = -3;  MOTIF_DONOR_HI    = 6    # offsets from win_center
MOTIF_ACCEPTOR_LO = -20; MOTIF_ACCEPTOR_HI = 3

# All four metrics; c_at_site and mean_c_motif are the primary ones (see docstring)
METRICS = ["c_at_site", "mean_c_motif", "mean_c_center", "mean_c_full"]

SEQ_CSV_FIELDS = [
    "seq_id", "splice_type", "chrom", "site_pos", "strand",
    "target_level", "target_score", "achieved_score", "ref_score",
    "n_mutations", "applied_mutations",
    # primary (comparable to 16c pos_c[splice_center])
    "c_at_site",
    # secondary (mean over mutated motif only — best signal/noise)
    "mean_c_motif", "n_motif_tokens",
    # supplementary
    "mean_c_center", "median_c_center",
    "mean_c_full",   "median_c_full",
    "window_start", "window_end",
]

# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------
def parse_fasta(path: Path) -> dict[str, str]:
    """Return {seq_id: sequence} from a FASTA file."""
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(buf)
                cur_id = line[1:].split()[0]
                buf = []
            elif cur_id is not None:
                buf.append(line.upper())
    if cur_id is not None:
        seqs[cur_id] = "".join(buf)
    return seqs


# ---------------------------------------------------------------------------
# Settling depth (NumPy, mirrors 16c logic)
# ---------------------------------------------------------------------------
def settling_depth_np(D_cos: np.ndarray, gamma: float) -> np.ndarray:
    """[L, T] float32 → [T] float32 of 1-based settling depths.

    Returns values in {1, …, L}; L means "never reached γ".
    """
    rmin = np.minimum.accumulate(D_cos, axis=0)
    below = rmin <= gamma
    any_below = below.any(axis=0)
    first_idx  = below.argmax(axis=0)
    L = D_cos.shape[0]
    return np.where(any_below, first_idx + 1, L).astype(np.float32)


# ---------------------------------------------------------------------------
# c statistics helper
# ---------------------------------------------------------------------------
def c_stats(c: np.ndarray) -> tuple[float, float, float]:
    """(mean, median, std) of a 1-D settling-depth array."""
    return float(c.mean()), float(np.median(c)), float(c.std(ddof=1)) if c.size > 1 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_done_ids(csv_path: Path) -> set[str]:
    """Return set of seq_ids already written to the checkpoint CSV."""
    if not csv_path.exists():
        return set()
    done: set[str] = set()
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add(row["seq_id"])
    return done


def append_row(csv_path: Path, row: dict, fields: list[str]) -> None:
    """Append one row to the per-sequence CSV; write header if new file."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    with ru.phase_context(PHASE, OUT_DIR, step_name="splice_evo2"):
        _run()


def _run() -> None:
    LOG.info("γ_cos locked = %.4f  center_half = %d bp", GAMMA_LOCKED, CENTER_HALF)

    # ---- Load metadata ----
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {META_PATH}.\n"
            "Run make_splice_strength_spliceai.py first."
        )
    meta: dict[str, dict] = {}
    with open(META_PATH) as fh:
        for row in csv.DictReader(fh):
            meta[row["seq_id"]] = row
    LOG.info("loaded %d sequence records from metadata.csv", len(meta))

    # ---- Load sequences ----
    if not FA_PATH.exists():
        raise FileNotFoundError(f"sequences.fa not found at {FA_PATH}.")
    LOG.info("parsing %s …", FA_PATH)
    seqs = parse_fasta(FA_PATH)
    LOG.info("loaded %d sequences", len(seqs))

    seq_ids = [sid for sid in seqs if sid in meta]
    if not seq_ids:
        raise RuntimeError("No seq_ids matched between FASTA and metadata.")
    LOG.info("%d sequences matched to metadata", len(seq_ids))

    # Order: group by site_pos for cache-friendly processing
    seq_ids.sort(key=lambda sid: (
        meta[sid]["splice_type"],
        int(meta[sid]["site_pos"]),
        meta[sid]["target_level"],
    ))

    # ---- Checkpoint: skip already processed ----
    done_ids = load_done_ids(SEQ_CSV_PATH)
    remaining = [sid for sid in seq_ids if sid not in done_ids]
    LOG.info("already done: %d  remaining: %d", len(done_ids), len(remaining))

    if remaining:
        # ---- Load Evo-2 model ----
        from src.constants_evo2 import N_LAYERS
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import extract_hidden_states, all_layer_names
        from src.ur_gdtr_evo2 import cosine_lens

        bundle = load_evo2()
        LOG.info("Evo-2 loaded: %s", bundle.loaded_variant)
        layer_names = all_layer_names()

        t0 = time.time()
        for idx, seq_id in enumerate(remaining):
            seq  = seqs[seq_id]
            T    = len(seq)
            m    = meta[seq_id]
            stype = m["splice_type"]   # "donor" or "acceptor"

            # Splice site is at position T//2 = 2048 in the 4096-bp output window
            # (verified: out_seq = score_win[SCORE_CTR-OUTPUT_HALF : SCORE_CTR+OUTPUT_HALF]
            #  → splice_site_in_output = SCORE_CTR - (SCORE_CTR - OUTPUT_HALF) = OUTPUT_HALF)
            win_center = T // 2   # = 2048

            # Motif slice (only the mutated bases — best signal for comparison with 16c)
            if stype == "donor":
                m_lo = win_center + MOTIF_DONOR_LO     # 2048 - 3 = 2045
                m_hi = win_center + MOTIF_DONOR_HI     # 2048 + 6 = 2054
            else:
                m_lo = win_center + MOTIF_ACCEPTOR_LO  # 2048 - 20 = 2028
                m_hi = win_center + MOTIF_ACCEPTOR_HI  # 2048 + 3  = 2051
            m_lo = max(0, m_lo); m_hi = min(T, m_hi)

            # Center window (flanking context around motif)
            c_lo = max(0, win_center - CENTER_HALF)
            c_hi = min(T, win_center + CENTER_HALF)

            # ---- Evo-2 forward ----
            try:
                input_ids = tokenize(seq, bundle, device="cuda")
                hs = extract_hidden_states(bundle, input_ids,
                                           save_layers=layer_names)
                D_cos = cosine_lens(hs, n_layers=N_LAYERS).numpy().astype(np.float32)
                # D_cos: [N_LAYERS, T]  — kept as float32 (no float16 round-trip)
                del hs
                torch.cuda.empty_cache()
            except Exception as exc:
                LOG.error("forward failed for %s: %s", seq_id, exc)
                torch.cuda.empty_cache()
                continue

            # ---- Settling depth ----
            c_full = settling_depth_np(D_cos, GAMMA_LOCKED)   # [T], values in {1,…,32}

            # PRIMARY: c at exactly the splice-site token (mirrors 16c pos_c[center])
            c_at_site = float(c_full[win_center])

            # SECONDARY: mean over mutated motif only (best signal-to-noise)
            c_motif  = c_full[m_lo:m_hi]
            n_motif  = c_motif.size
            mean_c_motif = float(c_motif.mean()) if n_motif > 0 else float("nan")

            # Supplementary: center window and full window
            c_center = c_full[c_lo:c_hi]
            mean_c_center, med_c_center, _ = (
                c_stats(c_center) if c_center.size > 0 else (float("nan"),) * 3)
            mean_c_full, med_c_full, _ = c_stats(c_full)

            row = dict(m)
            row.update({
                "c_at_site":       f"{c_at_site:.4f}",
                "mean_c_motif":    f"{mean_c_motif:.4f}",
                "n_motif_tokens":  n_motif,
                "mean_c_center":   f"{mean_c_center:.4f}",
                "median_c_center": f"{med_c_center:.4f}",
                "mean_c_full":     f"{mean_c_full:.4f}",
                "median_c_full":   f"{med_c_full:.4f}",
            })
            append_row(SEQ_CSV_PATH, row, SEQ_CSV_FIELDS)

            elapsed = time.time() - t0
            rate    = (idx + 1) / max(1.0, elapsed)
            eta     = (len(remaining) - idx - 1) / max(1e-6, rate)
            LOG.info(
                "[%d/%d] %s  level=%s  c_at_site=%.1f  mean_c_motif=%.2f"
                "  mean_c_center=%.2f  rate=%.2f/s  ETA=%.1f min",
                idx + 1, len(remaining),
                seq_id, m["target_level"],
                c_at_site, mean_c_motif, mean_c_center, rate, eta / 60,
            )
    else:
        LOG.info("all sequences already processed; loading saved results.")

    # ---- Aggregation ----
    LOG.info("aggregating c by (splice_type, target_level) …")

    from collections import defaultdict

    # {(splice_type, target_level): {metric: [values], ...}}
    _empty = lambda: {m: [] for m in METRICS + ["achieved_score", "ref_score"]}
    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(_empty)

    with open(SEQ_CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            key = (row["splice_type"], row["target_level"])
            try:
                for m in METRICS:
                    agg[key][m].append(float(row[m]))
                agg[key]["achieved_score"].append(float(row["achieved_score"]))
                agg[key]["ref_score"].append(float(row["ref_score"]))
            except (ValueError, KeyError):
                pass

    level_rows: list[dict] = []
    for (stype, level), d in sorted(agg.items()):
        for metric in METRICS:
            arr = np.array(d[metric])
            arr = arr[np.isfinite(arr)]
            n   = arr.size
            if n == 0:
                continue
            level_rows.append({
                "splice_type":         stype,
                "target_level":        level,
                "metric":              metric,
                "n_sites":             n,
                "mean_mean_c":         float(arr.mean()),
                "sem_mean_c":          float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "std_mean_c":          float(arr.std(ddof=1)) if n > 1 else 0.0,
                "median_mean_c":       float(np.median(arr)),
                "mean_achieved_score": float(np.mean(d["achieved_score"])) if d["achieved_score"] else None,
                "mean_ref_score":      float(np.mean(d["ref_score"]))      if d["ref_score"] else None,
            })

    with open(LEVEL_CSV_PATH, "w", newline="") as fh:
        fields = ["splice_type", "target_level", "metric", "n_sites",
                  "mean_mean_c", "sem_mean_c", "std_mean_c", "median_mean_c",
                  "mean_achieved_score", "mean_ref_score"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(level_rows)
    LOG.info("wrote c_by_level.csv  (%d rows)", len(level_rows))

    # ---- Pearson r analysis ----
    # Primary metrics: c_at_site (mirrors 16c), mean_c_motif (cleanest signal)
    pearson_stats: dict[str, dict] = {}
    for stype in ("donor", "acceptor"):
        for metric in METRICS:
            rows_s = [r for r in level_rows
                      if r["splice_type"] == stype and r["metric"] == metric]
            if len(rows_s) < 3:
                continue
            levels_arr = np.array([float(r["target_level"]) for r in rows_s])
            c_arr      = np.array([r["mean_mean_c"] for r in rows_s])
            r_val      = float(np.corrcoef(levels_arr, c_arr)[0, 1])
            pearson_stats[f"{stype}_{metric}"] = {
                "pearson_r":  r_val,
                "n_levels":   len(rows_s),
                "is_primary": metric in ("c_at_site", "mean_c_motif"),
            }
            flag = "★" if metric in ("c_at_site", "mean_c_motif") else " "
            LOG.info("%s Pearson r (%s, %s) = %.3f  (n=%d levels)",
                     flag, stype, metric, r_val, len(rows_s))

    # ---- Figures ----
    _make_figures(level_rows, OUT_DIR)

    # ---- Summary JSON ----
    summary = {
        "gamma_locked":      GAMMA_LOCKED,
        "center_half_bp":    CENTER_HALF,
        "n_sequences_total": len(seq_ids),
        "n_processed":       len(done_ids) + len(remaining),
        "pearson_r":         pearson_stats,
        "paths": {
            "input_fa":      str(FA_PATH),
            "input_meta":    str(META_PATH),
            "seq_csv":       str(SEQ_CSV_PATH),
            "level_csv":     str(LEVEL_CSV_PATH),
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote summary.json")

    ru.write_done(PHASE, OUT_DIR,
                  {"gamma_locked": GAMMA_LOCKED, "summary": str(JSON_PATH)},
                  step_name="splice_evo2")
    LOG.info("all done.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _make_figures(level_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not available; skipping figures")
        return

    COLORS     = {"donor": "#d1495b", "acceptor": "#2e86ab"}
    LINESTYLES = {"donor": "o-",      "acceptor": "s--"}

    # Metric display config: (metric_key, y-axis label, figure filename suffix, is_primary)
    metric_cfg = [
        ("c_at_site",
         f"settling depth c at splice-site token  [mirrors 16c pos_c(center)]",
         "site", True),
        ("mean_c_motif",
         "mean c over splice motif  [donor 9-mer / acceptor 23-mer]",
         "motif", True),
        ("mean_c_center",
         f"mean c over ±{CENTER_HALF} bp center window",
         "center", False),
        ("mean_c_full",
         "mean c over full 4096-bp window  (diluted — reference only)",
         "full", False),
    ]

    for metric, ylabel, suffix, is_primary in metric_cfg:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_plotted = False

        for stype in ("donor", "acceptor"):
            rows_s = sorted(
                [r for r in level_rows
                 if r["splice_type"] == stype and r["metric"] == metric],
                key=lambda r: float(r["target_level"]),
                reverse=True,   # left = strong (1.0), right = weak (0.1)
            )
            if not rows_s:
                continue
            xs  = [float(r["target_level"]) for r in rows_s]
            ys  = [r["mean_mean_c"]         for r in rows_s]
            err = [r["sem_mean_c"]          for r in rows_s]

            ax.errorbar(xs, ys, yerr=err,
                        fmt=LINESTYLES[stype],
                        color=COLORS[stype],
                        lw=1.8 if is_primary else 1.2,
                        ms=5, capsize=3,
                        label=stype)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        star = " ★" if is_primary else ""
        ax.set_xlabel("SpliceAI-calibrated splice-strength level")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"gDTR settling depth c vs. splice strength{star}  (γ={GAMMA_LOCKED})")
        ax.invert_xaxis()
        ax.set_xticks([round(v, 1) for v in np.linspace(0.1, 1.0, 10)])
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fname = f"F_c_vs_level_{suffix}"
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}", dpi=150)
        LOG.info("saved %s%s", fname, star)
        plt.close(fig)

    # ---- Heatmap: site × level matrix of mean_c_center ----
    _make_heatmap(out_dir)


def _make_heatmap(out_dir: Path) -> None:
    """Site × level heatmap of mean_c_center for each splice type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows_all: list[dict] = []
    with open(SEQ_CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            try:
                rows_all.append({
                    "site_pos":      int(row["site_pos"]),
                    "splice_type":   row["splice_type"],
                    "target_level":  float(row["target_level"]),
                    "mean_c_motif":  float(row.get("mean_c_motif", "nan")),
                    "mean_c_center": float(row.get("mean_c_center", "nan")),
                })
            except (ValueError, KeyError):
                pass

    if not rows_all:
        return

    LEVELS_SORTED = sorted({r["target_level"] for r in rows_all}, reverse=True)

    for stype in ("donor", "acceptor"):
        sub = [r for r in rows_all if r["splice_type"] == stype]
        if not sub:
            continue
        sites = sorted({r["site_pos"] for r in sub})
        if len(sites) > 100:
            import random
            random.seed(42)
            sites = sorted(random.sample(sites, 100))

        mat = np.full((len(sites), len(LEVELS_SORTED)), np.nan)
        site_idx = {s: i for i, s in enumerate(sites)}
        lev_idx  = {l: j for j, l in enumerate(LEVELS_SORTED)}

        for r in sub:
            si = site_idx.get(r["site_pos"])
            li = lev_idx.get(r["target_level"])
            if si is not None and li is not None:
                # prefer mean_c_motif; fall back to mean_c_center
                val = r.get("mean_c_motif") or r.get("mean_c_center")
                if val is not None:
                    mat[si, li] = val

        hm_metric = "mean_c_motif"
        fig, ax = plt.subplots(figsize=(10, max(4, len(sites) * 0.12 + 2)))
        vmin = np.nanmin(mat); vmax = np.nanmax(mat)
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(LEVELS_SORTED)))
        ax.set_xticklabels([f"{l:.1f}" for l in LEVELS_SORTED], fontsize=8)
        ax.set_xlabel("splice-strength level")
        ax.set_ylabel(f"{stype} sites (n={len(sites)})")
        ax.set_title(f"{stype}: {hm_metric} per site × level  (γ={GAMMA_LOCKED})")
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03, label=hm_metric)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"F_heatmap_{stype}.{ext}", dpi=150)
        LOG.info("saved F_heatmap_%s", stype)
        plt.close(fig)


if __name__ == "__main__":
    main()

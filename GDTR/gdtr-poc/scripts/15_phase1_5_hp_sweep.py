"""Phase 1.5 — HP sweep over (gamma_cos, rho).

Grid: gamma in {0.4, 0.5, 0.6}, rho in {0.8, 0.85, 0.9}.
Effect-size proxy: Cohen's d of gDTR(sanity_gc) - gDTR(sanity_shuf) — the
larger the separation, the more discriminative the (gamma, rho) pair on the
sanity contrast that mirrors exon vs intron in chr22.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import _runner_utils as ru
ru.add_repo_paths()

PHASE = "phase1.5"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)


def settling_depth_running_min(D, gamma):
    """Return [N, T] discrete settling depth (1-based, L if never crossed)."""
    rmin = np.minimum.accumulate(D, axis=1)  # [N, L, T]
    below = rmin <= gamma
    any_below = below.any(axis=1)  # [N, T]
    first_idx = below.argmax(axis=1)  # [N, T] 0-based
    L = D.shape[1]
    c = np.where(any_below, first_idx + 1, L)
    return c.astype(np.int64)


def gdtr(c, rho, L):
    return float((c.astype(np.float64) > rho * L).mean())


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return (np.mean(x) - np.mean(y)) / sp


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        traces = np.load(ru.GDTR_ROOT / "results" / "phase1.1" / "lens_traces.npz", allow_pickle=True)
        D_cos = traces["D_cos"]  # [N, L, T]
        regions = traces["regions"]
        N, L, T = D_cos.shape

        gc_idx = np.where(regions == "sanity_gc")[0]
        shuf_idx = np.where(regions == "sanity_shuf")[0]
        LOG.info("gc=%d shuf=%d L=%d T=%d", len(gc_idx), len(shuf_idx), L, T)

        gammas = [0.4, 0.5, 0.6]
        rhos = [0.8, 0.85, 0.9]

        rows = []
        d_grid = np.zeros((len(gammas), len(rhos)))
        for gi, g in enumerate(gammas):
            c_all = settling_depth_running_min(D_cos, g)  # [N, T]
            # per-seq gDTR
            for ri, r in enumerate(rhos):
                gdtr_per_seq = (c_all.astype(np.float64) > r * L).mean(axis=1)
                gc_v = gdtr_per_seq[gc_idx]
                shuf_v = gdtr_per_seq[shuf_idx]
                d = cohens_d(gc_v, shuf_v)
                d_grid[gi, ri] = d
                rows.append({
                    "gamma_cos": g, "rho": r,
                    "gDTR_gc_mean": float(gc_v.mean()), "gDTR_gc_std": float(gc_v.std()),
                    "gDTR_shuf_mean": float(shuf_v.mean()), "gDTR_shuf_std": float(shuf_v.std()),
                    "cohens_d": d,
                })

        # CSV
        csv_path = PHASE_DIR / "hp_sweep.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # Heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(d_grid, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(len(rhos))); ax.set_xticklabels(rhos)
            ax.set_yticks(range(len(gammas))); ax.set_yticklabels(gammas)
            ax.set_xlabel("rho"); ax.set_ylabel("gamma_cos")
            for i in range(len(gammas)):
                for j in range(len(rhos)):
                    ax.text(j, i, f"{d_grid[i, j]:.2f}", ha="center", va="center",
                            color="black", fontsize=10)
            fig.colorbar(im, ax=ax, label="Cohen's d (gc vs shuf)")
            ax.set_title("Phase 1.5 HP sweep (sanity proxy)")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F9_hp_heatmap.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("heatmap failed: %s", e)

        # Pick best
        best_i, best_j = np.unravel_index(np.argmax(np.abs(d_grid)), d_grid.shape)
        best = {
            "gamma_cos": gammas[best_i], "rho": rhos[best_j],
            "cohens_d": float(d_grid[best_i, best_j]),
        }
        (PHASE_DIR / "best_hp.json").write_text(json.dumps(best, indent=2))
        LOG.info("best HP (sanity proxy): %s", best)

        ru.write_done(PHASE, PHASE_DIR, {"best": best})


if __name__ == "__main__":
    main()

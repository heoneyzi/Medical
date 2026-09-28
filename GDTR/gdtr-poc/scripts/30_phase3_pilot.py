"""Phase 3 ClinVar Pilot — gDTR DeltaD(l) for cancer variant pathogenicity.

Tests whether per-layer JSD difference (alt - ref) carries pathogenicity
information for TP53 + BRCA1 ClinVar variants. Uses 6kb context centered
on each variant; runs Evo 2 7B forward on both ref and alt, computes
DeltaD_jsd[L], DeltaD_cos[L], dc_interp, max|DeltaD|. Evaluates each
feature with stratified 10-fold logistic regression CV.

Inputs locked from Phase 1:
  - gamma_cos = 0.39663 (Phase 1.4 sanity_global_q70)
  - 32 blocks + post-norm
  - BOS_OFFSET = 0
  - extract_hidden_states already clones (Bug 1 fix applied)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase3_pilot"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase3_pilot"
LOG = ru.setup_logging(PHASE)

# Configuration
GAMMA_COS = 0.39663  # Locked from Phase 1.4 (sanity_global_q70)
SEED = 42
CONTEXT_HALF = 3000   # 6kb total context centered on variant
CAP_PER_CELL = 250    # max 250 per (gene x category) -> 1000 total
N_FOLDS = 10
GENES = ("TP53", "BRCA1")
CATEGORIES = ("P_LP", "B_LB")  # ground-truth (binary)
CHROM = "chr17"


def load_variants(tsv_path: Path) -> list[dict]:
    """Load + filter clinvar TSV. Returns list of dicts (SNVs only,
    target genes + P/LP / B/LB only, balanced cap per cell)."""
    rng = np.random.default_rng(SEED)
    rows_per_cell: dict[tuple, list[dict]] = {(g, c): [] for g in GENES for c in CATEGORIES}
    with tsv_path.open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            parts = line.rstrip().split("\t")
            d = dict(zip(header, parts))
            if d["gene"] not in GENES:
                continue
            if d["category"] not in CATEGORIES:
                continue
            ref, alt = d["ref"], d["alt"]
            if len(ref) != 1 or len(alt) != 1:
                continue  # SNV only for clean substitution
            if ref not in "ACGT" or alt not in "ACGT":
                continue
            d["pos"] = int(d["pos"])
            rows_per_cell[(d["gene"], d["category"])].append(d)
    # Cap per cell
    final = []
    for (g, c), rows in rows_per_cell.items():
        if len(rows) > CAP_PER_CELL:
            idx = rng.choice(len(rows), CAP_PER_CELL, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        LOG.info("  %s %s: %d variants (after cap)", g, c, len(rows))
        final.extend(rows)
    rng.shuffle(final)
    return final


def fetch_window(fasta, chrom: str, pos: int) -> tuple[str, str, int]:
    """Return (ref_seq, alt_seq_template, local_idx) for [pos-3000, pos+3000).
    pos is 1-based. local_idx is the 0-based offset where the variant sits."""
    start = pos - 1 - CONTEXT_HALF  # convert to 0-based
    end = pos - 1 + CONTEXT_HALF
    if start < 0:
        # Pad start with 'N' if too close to chrom start
        pad = -start
        start = 0
        seq = "N" * pad + fasta.fetch(chrom, start, end).upper()
    else:
        seq = fasta.fetch(chrom, start, end).upper()
    local_idx = pos - 1 - (pos - 1 - CONTEXT_HALF)  # = CONTEXT_HALF
    return seq, seq, local_idx


def settling_depth_interp_np(D: np.ndarray, gamma: float) -> tuple[float, bool]:
    """1-position scalar version: interpolated settling depth at variant pos.
    D shape [L] (single column). Returns (c_interp, saturated)."""
    L = D.shape[0]
    rmin = np.minimum.accumulate(D)
    below = rmin <= gamma
    if not below.any():
        return float(L), True
    first = int(below.argmax())
    if first == 0:
        return 1.0, False
    # bracket between layers (first-1, first)
    rm_above = rmin[first - 1]
    rm_below = rmin[first]
    denom = rm_above - rm_below
    if denom < 1e-12:
        return float(first + 1), False
    frac = (rm_above - gamma) / denom
    frac = max(0.0, min(1.0, frac))
    return float(first) + frac, False


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="pilot"):
        t_start = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import N_LAYERS
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        # ----- Load variants -----
        tsv = ru.GDTR_ROOT / "data" / "variants" / "clinvar_15gene.tsv"
        LOG.info("loading variants from %s", tsv)
        variants = load_variants(tsv)
        LOG.info("total variants for pilot: %d", len(variants))
        cell_counts: dict[str, int] = {}
        for v in variants:
            k = f"{v['gene']}_{v['category']}"
            cell_counts[k] = cell_counts.get(k, 0) + 1
        LOG.info("cell counts: %s", cell_counts)

        # ----- FASTA -----
        try:
            import pysam
        except Exception as e:
            raise RuntimeError(f"pysam required: {e}")
        fasta = pysam.FastaFile(str(ru.GDTR_ROOT / "data" / "reference" / "chr17.fa"))

        # ----- Model -----
        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)
        layer_names = all_layer_names()

        # ----- Per-variant forward -----
        rows = []
        n_err = 0
        n_nan = 0
        for i, var in enumerate(variants):
            try:
                ref_seq, _, local_idx = fetch_window(fasta, CHROM, var["pos"])
                T = len(ref_seq)
                if T < 100:
                    LOG.warning("variant %d skipped: short window T=%d", i, T)
                    n_err += 1
                    continue
                # Sanity: ref base at local_idx matches var.ref (allow N at boundary)
                ref_at = ref_seq[local_idx] if local_idx < T else "N"
                if ref_at != var["ref"] and ref_at != "N":
                    LOG.warning(
                        "variant %d ref mismatch at pos=%d: fasta=%s var.ref=%s (skip)",
                        i, var["pos"], ref_at, var["ref"],
                    )
                    n_err += 1
                    continue
                # Build alt seq
                alt_seq = ref_seq[:local_idx] + var["alt"] + ref_seq[local_idx + 1:]

                # Forward ref
                ids_ref = tokenize(ref_seq, bundle, device="cuda")
                hs_ref = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
                d_jsd_ref = jsd_lens(hs_ref, bundle, n_layers=N_LAYERS).numpy()  # [L, T]
                d_cos_ref = cosine_lens(hs_ref, n_layers=N_LAYERS).numpy()       # [L, T]
                del hs_ref, ids_ref
                torch.cuda.empty_cache()

                # Forward alt
                ids_alt = tokenize(alt_seq, bundle, device="cuda")
                hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                d_jsd_alt = jsd_lens(hs_alt, bundle, n_layers=N_LAYERS).numpy()
                d_cos_alt = cosine_lens(hs_alt, n_layers=N_LAYERS).numpy()
                del hs_alt, ids_alt
                torch.cuda.empty_cache()

                # Pull variant-position column
                # local_idx == position in T_real (BOS_OFFSET=0)
                col_jsd_ref = d_jsd_ref[:, local_idx]  # [L]
                col_jsd_alt = d_jsd_alt[:, local_idx]
                col_cos_ref = d_cos_ref[:, local_idx]
                col_cos_alt = d_cos_alt[:, local_idx]

                if not (np.isfinite(col_jsd_ref).all() and np.isfinite(col_jsd_alt).all()
                        and np.isfinite(col_cos_ref).all() and np.isfinite(col_cos_alt).all()):
                    LOG.warning("variant %d NaN in lens (skip)", i)
                    n_nan += 1
                    continue

                dD_jsd = col_jsd_alt - col_jsd_ref      # [L]
                dD_cos = col_cos_alt - col_cos_ref
                abs_dJ = np.abs(dD_jsd)
                arg = int(abs_dJ.argmax())
                signed_argmax = float(dD_jsd[arg])
                max_abs = float(abs_dJ.max())

                # Settling depth dc_interp using cosine running-min
                c_ref, _ = settling_depth_interp_np(col_cos_ref, GAMMA_COS)
                c_alt, _ = settling_depth_interp_np(col_cos_alt, GAMMA_COS)
                dc_interp = c_alt - c_ref

                row = {
                    "chrom": CHROM, "pos": var["pos"],
                    "ref": var["ref"], "alt": var["alt"],
                    "gene": var["gene"], "category": var["category"],
                    "max_abs_dD": max_abs,
                    "signed_argmax": signed_argmax,
                    "argmax_layer": arg + 1,
                    "dc_interp": float(dc_interp),
                }
                for ell in range(N_LAYERS):
                    row[f"dD_jsd_{ell}"] = float(dD_jsd[ell])
                for ell in range(N_LAYERS):
                    row[f"dD_cos_{ell}"] = float(dD_cos[ell])
                rows.append(row)

                if (i + 1) % 25 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / max(elapsed, 1e-6)
                    eta = (len(variants) - i - 1) / max(rate, 1e-9)
                    LOG.info(
                        "var %d/%d  rate=%.2f/s  ETA=%.1f min  errs=%d  nan=%d",
                        i + 1, len(variants), rate, eta / 60, n_err, n_nan,
                    )
            except Exception as e:
                LOG.exception("variant %d failed: %s", i, e)
                n_err += 1
                torch.cuda.empty_cache()
                continue

        LOG.info("forward done: %d rows  err=%d  nan=%d", len(rows), n_err, n_nan)

        # ----- Save CSV -----
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = PHASE_OUT_DIR / "variants_features.csv"
        df.to_csv(csv_path, index=False)
        LOG.info("wrote %s rows=%d cols=%d", csv_path, len(df), len(df.columns))

        # ----- Logistic regression with 10-fold CV -----
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        y = (df["category"] == "P_LP").astype(int).to_numpy()
        feat_jsd = np.array([[r[f"dD_jsd_{l}"] for l in range(N_LAYERS)] for _, r in df.iterrows()])
        feat_cos = np.array([[r[f"dD_cos_{l}"] for l in range(N_LAYERS)] for _, r in df.iterrows()])
        feat_dc = df[["dc_interp"]].to_numpy()
        feat_max = df[["max_abs_dD"]].to_numpy()

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        def cv_auroc(X: np.ndarray, name: str) -> dict:
            aurocs = []
            all_y_test, all_y_score = [], []
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")),
                ])
                pipe.fit(X[tr], y[tr])
                score = pipe.predict_proba(X[te])[:, 1]
                au = roc_auc_score(y[te], score)
                aurocs.append(au)
                all_y_test.append(y[te])
                all_y_score.append(score)
            mean_au = float(np.mean(aurocs))
            std_au = float(np.std(aurocs, ddof=1))
            ci_lo = mean_au - 1.96 * std_au / np.sqrt(N_FOLDS)
            ci_hi = mean_au + 1.96 * std_au / np.sqrt(N_FOLDS)
            # Pooled ROC for plot
            yt = np.concatenate(all_y_test)
            ys = np.concatenate(all_y_score)
            fpr, tpr, _ = roc_curve(yt, ys)
            return {
                "name": name, "fold_aurocs": [float(a) for a in aurocs],
                "mean_auroc": mean_au, "std_auroc": std_au,
                "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
                "n_features": int(X.shape[1]), "n_samples": int(X.shape[0]),
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "pooled_auroc": float(roc_auc_score(yt, ys)),
            }

        results = {
            "dD_jsd_vector": cv_auroc(feat_jsd, "dD_jsd_vector"),
            "dD_cos_vector": cv_auroc(feat_cos, "dD_cos_vector"),
            "dc_interp_scalar": cv_auroc(feat_dc, "dc_interp_scalar"),
            "max_abs_dD_scalar": cv_auroc(feat_max, "max_abs_dD_scalar"),
        }

        # Verdict
        primary = results["dD_jsd_vector"]["mean_auroc"]
        any_above_chance = any(r["mean_auroc"] > 0.50 for r in results.values())
        verdict = {
            "primary_dD_jsd_auroc": primary,
            "primary_passes_0.65": bool(primary >= 0.65),
            "primary_passes_0.55": bool(primary >= 0.55),
            "any_above_chance": bool(any_above_chance),
            "best_feature": max(results.keys(), key=lambda k: results[k]["mean_auroc"]),
            "best_auroc": float(max(r["mean_auroc"] for r in results.values())),
        }

        out = {
            "phase": PHASE,
            "config": {
                "gamma_cos": GAMMA_COS, "seed": SEED, "n_folds": N_FOLDS,
                "context_half": CONTEXT_HALF, "cap_per_cell": CAP_PER_CELL,
                "genes": list(GENES), "categories": list(CATEGORIES),
                "model_variant": bundle.loaded_variant,
            },
            "cell_counts": cell_counts,
            "n_total_variants": int(len(variants)),
            "n_processed_variants": int(len(rows)),
            "n_errors": int(n_err),
            "n_nans": int(n_nan),
            "wall_time_sec": float(time.time() - t_start),
            "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
                        for k, v in results.items()},
            "verdict": verdict,
        }
        json_path = PHASE_OUT_DIR / "pilot_results.json"
        json_path.write_text(json.dumps(out, indent=2))
        LOG.info("wrote %s", json_path)
        LOG.info("VERDICT: %s", verdict)

        # ----- Plot ROC overlay -----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 6))
            for k, r in results.items():
                ax.plot(r["fpr"], r["tpr"],
                        label=f"{k} (AUROC={r['mean_auroc']:.3f} CI[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}])")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.5)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"Phase 3 Pilot ROC — TP53+BRCA1 (n={len(rows)})")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_OUT_DIR / f"F_phase3_auroc.{ext}", dpi=150)
            plt.close(fig)
            LOG.info("ROC figure saved")
        except Exception as e:
            LOG.warning("figure failed: %s", e)

        ru.write_done(PHASE, PHASE_OUT_DIR, verdict, step_name="pilot")
        LOG.info("Phase 3 pilot done in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()

"""Phase 3 main — gDTR ΔD lens + Evo 2 likelihood for 15 cancer genes.

Adapts the Phase 3 pilot (TP53+BRCA1) to all 15 genes across 9 chromosomes.
Per variant we run Evo 2 forward on (ref, alt) windows, extract per-layer
JSD/cosine distances at the variant position, and additionally compute Evo 2
log-likelihood at the variant token (ref vs alt) for the ensemble baseline.

Outputs (incremental):
  /path/to/gDTR/results/phase3_main/variants_features.csv   (resumable)
  /path/to/gDTR/results/phase3_main/_checkpoint.json        (next idx)
  /path/to/gDTR/results/phase3_main/_main_done              (final marker)
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase3_main"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "phase3_main"
LOG = ru.setup_logging(PHASE)

# Configuration (locked from Phase 1.4 / Phase 3 pilot).
GAMMA_COS = 0.39663
SEED = 42
CONTEXT_HALF = 3000  # 6 kb total context, matches pilot
CHECKPOINT_EVERY = 1000

GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)
TRAIN_CATS = ("P_LP", "B_LB")
HOLD_CATS = ("VUS",)
ALL_CATS = TRAIN_CATS + HOLD_CATS

REF_DIR = ru.GDTR_ROOT / "data" / "reference"
TSV_PATH = ru.GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.tsv"
CSV_PATH = PHASE_OUT_DIR / "variants_features.csv"
CKPT_PATH = PHASE_OUT_DIR / "_checkpoint.json"


def load_variants(tsv_path: Path) -> list[dict]:
    rows = []
    with tsv_path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            d = dict(zip(header, parts))
            if d["gene"] not in GENES:
                continue
            if d["category"] not in ALL_CATS:
                continue
            ref = d["ref"]; alt = d["alt"]
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                continue
            d["pos"] = int(d["pos"])
            d["stars"] = int(d.get("stars", 0))
            rows.append(d)
    return rows


def stable_order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["chrom"], r["pos"], r["ref"], r["alt"]))


class FastaCache:
    """Lazy-open chr*.fa and keep one open file per chrom."""

    def __init__(self, ref_dir: Path):
        self.ref_dir = ref_dir
        self._open: dict[str, "pysam.FastaFile"] = {}

    def get(self, chrom: str):
        if chrom in self._open:
            return self._open[chrom]
        # FASTA filename: chrN.fa with header `chrN`
        fa_path = self.ref_dir / (f"chr{chrom}.fa" if not str(chrom).startswith("chr") else f"{chrom}.fa")
        if not fa_path.exists():
            # Some prep scripts strip "chr" prefix
            alt = self.ref_dir / f"{chrom.lstrip('chr')}.fa"
            if alt.exists():
                fa_path = alt
            else:
                raise FileNotFoundError(f"No FASTA for {chrom} at {fa_path}")
        import pysam
        f = pysam.FastaFile(str(fa_path))
        self._open[chrom] = f
        return f


def fetch_window(fa, chrom: str, pos: int) -> tuple[str, int]:
    """Return (ref_seq, local_idx). Pads with N if near 5' end."""
    start = pos - 1 - CONTEXT_HALF
    end = pos - 1 + CONTEXT_HALF
    chrom_len = fa.get_reference_length(chrom)
    if start < 0:
        pad = -start
        seq = "N" * pad + fa.fetch(chrom, 0, end).upper()
    elif end > chrom_len:
        seq = fa.fetch(chrom, start, chrom_len).upper()
        seq = seq + "N" * (end - chrom_len)
    else:
        seq = fa.fetch(chrom, start, end).upper()
    local_idx = CONTEXT_HALF
    return seq, local_idx


def settling_depth_interp_np(D: np.ndarray, gamma: float) -> tuple[float, bool]:
    L = D.shape[0]
    rmin = np.minimum.accumulate(D)
    below = rmin <= gamma
    if not below.any():
        return float(L), True
    first = int(below.argmax())
    if first == 0:
        return 1.0, False
    rm_above = rmin[first - 1]
    rm_below = rmin[first]
    denom = rm_above - rm_below
    if denom < 1e-12:
        return float(first + 1), False
    frac = (rm_above - gamma) / denom
    frac = max(0.0, min(1.0, frac))
    return float(first) + frac, False


def feature_columns(n_layers: int) -> list[str]:
    cols = [
        "chrom", "pos", "ref", "alt", "gene", "category", "stars",
        "max_abs_dD", "signed_argmax", "argmax_layer", "dc_interp",
        "evo2_ref_loglik", "evo2_alt_loglik", "evo2_delta_loglik",
    ]
    for ell in range(n_layers):
        cols.append(f"dD_jsd_{ell}")
    for ell in range(n_layers):
        cols.append(f"dD_cos_{ell}")
    return cols


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name="main"):
        t_start = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import N_LAYERS, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2, tokenize, lm_head_apply
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        # ---- Load + sort variants ----
        LOG.info("loading variants from %s", TSV_PATH)
        all_variants = load_variants(TSV_PATH)
        all_variants = stable_order(all_variants)
        LOG.info("total variants: %d (train+VUS)", len(all_variants))
        cat_counts = {}
        gene_counts = {}
        for v in all_variants:
            cat_counts[v["category"]] = cat_counts.get(v["category"], 0) + 1
            gene_counts[v["gene"]] = gene_counts.get(v["gene"], 0) + 1
        LOG.info("category counts: %s", cat_counts)
        LOG.info("per-gene counts: %s", gene_counts)

        # ---- Resume from checkpoint ----
        start_idx = 0
        if CKPT_PATH.exists():
            ckpt = json.loads(CKPT_PATH.read_text())
            start_idx = int(ckpt.get("next_idx", 0))
            LOG.info("resuming from idx=%d", start_idx)
        cols = feature_columns(N_LAYERS)
        # Open csv for append (or write header if new)
        write_header = not CSV_PATH.exists() or start_idx == 0
        if start_idx == 0 and CSV_PATH.exists():
            CSV_PATH.unlink()
            write_header = True

        # ---- Load Evo 2 ----
        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)
        layer_names = all_layer_names()

        fastas = FastaCache(REF_DIR)
        n_err = 0
        n_nan = 0
        n_done = 0

        f_csv = CSV_PATH.open("a", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=cols)
        if write_header:
            writer.writeheader(); f_csv.flush()

        # Tokenizer mapping for likelihood computation:
        # We compute log P(token at variant pos | left context) via the FINAL
        # post-norm logits. Token indices for ACGT are obtained via the
        # tokenizer once.
        tok = bundle.tokenizer
        # tokenize returns list[uint8], length 1 per single char
        token_id = {}
        for ch in "ACGT":
            ids = tok.tokenize(ch)
            token_id[ch] = int(np.asarray(ids, dtype=np.int64)[0])
        LOG.info("token_ids ACGT = %s", token_id)

        try:
            for i in range(start_idx, len(all_variants)):
                var = all_variants[i]
                try:
                    cc = var["chrom"]
                    canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
                    fa = fastas.get(cc)
                    ref_seq, local_idx = fetch_window(fa, canon, var["pos"])
                    T = len(ref_seq)
                    if T < 100:
                        n_err += 1; continue
                    ref_at = ref_seq[local_idx] if local_idx < T else "N"
                    if ref_at != var["ref"] and ref_at != "N":
                        LOG.warning(
                            "var %d ref mismatch %s:%d fa=%s clinvar=%s — skip",
                            i, var["chrom"], var["pos"], ref_at, var["ref"],
                        )
                        n_err += 1; continue
                    alt_seq = ref_seq[:local_idx] + var["alt"] + ref_seq[local_idx + 1:]

                    # ---- Forward ref ----
                    ids_ref = tokenize(ref_seq, bundle, device="cuda")
                    hs_ref = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
                    d_jsd_ref = jsd_lens(hs_ref, bundle, n_layers=N_LAYERS).numpy()
                    d_cos_ref = cosine_lens(hs_ref, n_layers=N_LAYERS).numpy()
                    # likelihood: norm tap → unembed → log-softmax @ pos local_idx-1
                    h_norm_ref = hs_ref["norm"]  # [1, T, H]
                    with torch.no_grad():
                        logits_ref = lm_head_apply(h_norm_ref, bundle).float()
                    # Predict token_t from logits at position t-1 (causal).
                    # Use position local_idx-1's logits as the predictor of
                    # token at local_idx (next-token likelihood).
                    if local_idx == 0:
                        pred_pos = 0
                    else:
                        pred_pos = local_idx - 1
                    log_p_at_var_ref = F.log_softmax(
                        logits_ref[0, pred_pos, :VOCAB_SIZE], dim=-1
                    ).cpu().numpy()
                    ref_loglik = float(log_p_at_var_ref[token_id[var["ref"]]])
                    del hs_ref, ids_ref, logits_ref, h_norm_ref
                    torch.cuda.empty_cache()

                    # ---- Forward alt ----
                    ids_alt = tokenize(alt_seq, bundle, device="cuda")
                    hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                    d_jsd_alt = jsd_lens(hs_alt, bundle, n_layers=N_LAYERS).numpy()
                    d_cos_alt = cosine_lens(hs_alt, n_layers=N_LAYERS).numpy()
                    h_norm_alt = hs_alt["norm"]
                    with torch.no_grad():
                        logits_alt = lm_head_apply(h_norm_alt, bundle).float()
                    log_p_at_var_alt = F.log_softmax(
                        logits_alt[0, pred_pos, :VOCAB_SIZE], dim=-1
                    ).cpu().numpy()
                    alt_loglik = float(log_p_at_var_alt[token_id[var["alt"]]])
                    del hs_alt, ids_alt, logits_alt, h_norm_alt
                    torch.cuda.empty_cache()

                    col_jsd_ref = d_jsd_ref[:, local_idx]
                    col_jsd_alt = d_jsd_alt[:, local_idx]
                    col_cos_ref = d_cos_ref[:, local_idx]
                    col_cos_alt = d_cos_alt[:, local_idx]

                    if not (np.isfinite(col_jsd_ref).all() and np.isfinite(col_jsd_alt).all()
                            and np.isfinite(col_cos_ref).all() and np.isfinite(col_cos_alt).all()
                            and np.isfinite(ref_loglik) and np.isfinite(alt_loglik)):
                        n_nan += 1; continue

                    dD_jsd = col_jsd_alt - col_jsd_ref
                    dD_cos = col_cos_alt - col_cos_ref
                    abs_dJ = np.abs(dD_jsd)
                    arg = int(abs_dJ.argmax())
                    signed_argmax = float(dD_jsd[arg])
                    max_abs = float(abs_dJ.max())
                    c_ref, _ = settling_depth_interp_np(col_cos_ref, GAMMA_COS)
                    c_alt, _ = settling_depth_interp_np(col_cos_alt, GAMMA_COS)
                    dc_interp = float(c_alt - c_ref)

                    row = {
                        "chrom": var["chrom"], "pos": var["pos"],
                        "ref": var["ref"], "alt": var["alt"],
                        "gene": var["gene"], "category": var["category"],
                        "stars": var["stars"],
                        "max_abs_dD": max_abs,
                        "signed_argmax": signed_argmax,
                        "argmax_layer": arg + 1,
                        "dc_interp": dc_interp,
                        "evo2_ref_loglik": ref_loglik,
                        "evo2_alt_loglik": alt_loglik,
                        "evo2_delta_loglik": alt_loglik - ref_loglik,
                    }
                    for ell in range(N_LAYERS):
                        row[f"dD_jsd_{ell}"] = float(dD_jsd[ell])
                    for ell in range(N_LAYERS):
                        row[f"dD_cos_{ell}"] = float(dD_cos[ell])
                    writer.writerow(row)
                    n_done += 1

                    if (i + 1) % 25 == 0:
                        f_csv.flush()
                        elapsed = time.time() - t_start
                        rate = max(n_done, 1) / max(elapsed, 1e-6)
                        eta = (len(all_variants) - i - 1) / max(rate, 1e-9)
                        LOG.info(
                            "var %d/%d  done=%d  rate=%.2f/s  ETA=%.1f min  err=%d  nan=%d",
                            i + 1, len(all_variants), n_done, rate, eta / 60, n_err, n_nan,
                        )
                    if (i + 1) % CHECKPOINT_EVERY == 0:
                        f_csv.flush()
                        CKPT_PATH.write_text(json.dumps({"next_idx": i + 1}))
                        LOG.info("checkpoint at idx=%d", i + 1)
                except Exception as e:  # noqa: BLE001
                    LOG.exception("var %d failed: %s", i, e)
                    n_err += 1
                    torch.cuda.empty_cache()
                    continue
        finally:
            f_csv.flush(); f_csv.close()

        CKPT_PATH.write_text(json.dumps({"next_idx": len(all_variants)}))
        out = {
            "phase": PHASE,
            "config": {
                "gamma_cos": GAMMA_COS, "seed": SEED, "context_half": CONTEXT_HALF,
                "genes": list(GENES),
                "train_categories": list(TRAIN_CATS),
                "hold_categories": list(HOLD_CATS),
                "model_variant": bundle.loaded_variant,
            },
            "n_total_variants": int(len(all_variants)),
            "n_processed_variants": int(n_done),
            "n_errors": int(n_err),
            "n_nans": int(n_nan),
            "category_counts": cat_counts,
            "per_gene_counts": gene_counts,
            "wall_time_sec": float(time.time() - t_start),
        }
        ru.write_done(PHASE, PHASE_OUT_DIR, out, step_name="main")
        LOG.info("Phase 3 main forward done in %.1f min  done=%d err=%d nan=%d",
                 (time.time() - t_start) / 60, n_done, n_err, n_nan)


if __name__ == "__main__":
    main()

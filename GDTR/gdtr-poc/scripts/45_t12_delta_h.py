"""T1.2 Baseline A: per-layer ‖Δh_l‖₂ on the same 10,910 ClinVar variants.

Reuses the exact Evo 2 forward + tokenization + variant-position alignment
from scripts/31_phase3_main.py. For each variant we forward Evo 2 on
(ref, alt) windows, extract hidden states for blocks.0..blocks.31 at the
variant position, and compute Euclidean norm of the per-layer difference:

  delta_h_norm[l] = || h_l_alt[var_pos] - h_l_ref[var_pos] ||_2  for l ∈ 0..31

Output (resumable):
  /path/to/gDTR/results/tier1_baselines/delta_h_features.csv
  /path/to/gDTR/results/tier1_baselines/_delta_h_checkpoint.json
  /path/to/gDTR/results/tier1_baselines/_delta_h_done

Companion T2.4 timing rows are written by 49_t24_cost.py (separate run).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "tier1_baselines_delta_h"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_baselines"
LOG = ru.setup_logging(PHASE)

# Match Phase 3 main exactly.
SEED = 42
CONTEXT_HALF = 3000
CHECKPOINT_EVERY = 500

GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)
TRAIN_CATS = ("P_LP", "B_LB")
HOLD_CATS = ("VUS",)
ALL_CATS = TRAIN_CATS + HOLD_CATS

REF_DIR = ru.GDTR_ROOT / "data" / "reference"
TSV_PATH = ru.GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.tsv"
CSV_PATH = PHASE_OUT_DIR / "delta_h_features.csv"
CKPT_PATH = PHASE_OUT_DIR / "_delta_h_checkpoint.json"
DONE_NAME = "delta_h"


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
    def __init__(self, ref_dir: Path):
        self.ref_dir = ref_dir
        self._open: dict[str, "pysam.FastaFile"] = {}

    def get(self, chrom: str):
        if chrom in self._open:
            return self._open[chrom]
        fa_path = self.ref_dir / (
            f"chr{chrom}.fa" if not str(chrom).startswith("chr") else f"{chrom}.fa"
        )
        if not fa_path.exists():
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


def feature_columns(n_layers: int) -> list[str]:
    cols = ["chrom", "pos", "ref", "alt", "gene", "category", "stars"]
    for ell in range(n_layers):
        cols.append(f"delta_h_norm_{ell}")
    return cols


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=DONE_NAME):
        t_start = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import N_LAYERS
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

        all_variants = stable_order(load_variants(TSV_PATH))
        LOG.info("total variants: %d", len(all_variants))

        start_idx = 0
        if CKPT_PATH.exists():
            ckpt = json.loads(CKPT_PATH.read_text())
            start_idx = int(ckpt.get("next_idx", 0))
            LOG.info("resuming from idx=%d", start_idx)
        cols = feature_columns(N_LAYERS)
        write_header = not CSV_PATH.exists() or start_idx == 0
        if start_idx == 0 and CSV_PATH.exists():
            CSV_PATH.unlink()
            write_header = True

        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)
        # Extract only blocks.* (no norm needed for this baseline).
        layer_names = [f"blocks.{i}" for i in range(N_LAYERS)] + ["norm"]

        fastas = FastaCache(REF_DIR)
        n_err = 0; n_nan = 0; n_done = 0

        f_csv = CSV_PATH.open("a", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=cols)
        if write_header:
            writer.writeheader(); f_csv.flush()

        try:
            for i in range(start_idx, len(all_variants)):
                var = all_variants[i]
                try:
                    cc = var["chrom"]
                    canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
                    fa = fastas.get(cc)
                    ref_seq, local_idx = fetch_window(fa, canon, var["pos"])
                    if len(ref_seq) < 100:
                        n_err += 1; continue
                    ref_at = ref_seq[local_idx] if local_idx < len(ref_seq) else "N"
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
                    h_ref_var = []
                    for ell in range(N_LAYERS):
                        h_ref_var.append(hs_ref[f"blocks.{ell}"][0, local_idx, :].float().cpu())
                    del hs_ref, ids_ref
                    torch.cuda.empty_cache()

                    # ---- Forward alt ----
                    ids_alt = tokenize(alt_seq, bundle, device="cuda")
                    hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                    h_alt_var = []
                    for ell in range(N_LAYERS):
                        h_alt_var.append(hs_alt[f"blocks.{ell}"][0, local_idx, :].float().cpu())
                    del hs_alt, ids_alt
                    torch.cuda.empty_cache()

                    deltas = []
                    bad = False
                    for ell in range(N_LAYERS):
                        d = (h_alt_var[ell] - h_ref_var[ell]).norm(p=2).item()
                        if not np.isfinite(d):
                            bad = True; break
                        deltas.append(d)
                    if bad:
                        n_nan += 1; continue

                    row = {
                        "chrom": var["chrom"], "pos": var["pos"],
                        "ref": var["ref"], "alt": var["alt"],
                        "gene": var["gene"], "category": var["category"],
                        "stars": var["stars"],
                    }
                    for ell in range(N_LAYERS):
                        row[f"delta_h_norm_{ell}"] = float(deltas[ell])
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
                except Exception as e:
                    LOG.exception("var %d failed: %s", i, e)
                    n_err += 1
                    torch.cuda.empty_cache()
                    continue
        finally:
            f_csv.flush(); f_csv.close()

        CKPT_PATH.write_text(json.dumps({"next_idx": len(all_variants)}))
        out = {
            "phase": PHASE,
            "config": {"seed": SEED, "context_half": CONTEXT_HALF,
                       "model_variant": bundle.loaded_variant,
                       "n_layers": N_LAYERS},
            "n_total_variants": int(len(all_variants)),
            "n_processed_variants": int(n_done),
            "n_errors": int(n_err),
            "n_nans": int(n_nan),
            "wall_time_sec": float(time.time() - t_start),
        }
        ru.write_done(PHASE, PHASE_OUT_DIR, out, step_name=DONE_NAME)
        LOG.info("delta_h done in %.1f min  done=%d err=%d nan=%d",
                 (time.time() - t_start) / 60, n_done, n_err, n_nan)


if __name__ == "__main__":
    main()

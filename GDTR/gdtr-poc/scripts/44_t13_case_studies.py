"""Tier-1 case study: per-layer ΔD trace for 6 hand-picked variants.

3 pathogenic + 3 matched benign (same gene, smallest |Δpos|), all chr17.
Goal: produce 32-d ΔD signatures per variant to show "deep-layer disruption"
mechanism behind the headline ΔD AUROC = 0.844.

Outputs (server):
  /path/to/gDTR/results/tier1_case_studies/case_studies.json
  /path/to/gDTR/results/tier1_case_studies/case_studies.npz
  /path/to/gDTR/results/tier1_case_studies/_done

Reproducibility check: re-runs Phase 3 main's exact ΔD extraction logic at
the variant token, and compares max|ΔD| against
results/phase3_main/variants_features.csv for each (chrom,pos,ref,alt).
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

PHASE = "tier1_case_studies"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_case_studies"
LOG = ru.setup_logging(PHASE)

CONTEXT_HALF = 3000  # match Phase 3
SEED = 42

REF_DIR = ru.GDTR_ROOT / "data" / "reference"
PHASE3_CSV = ru.GDTR_ROOT / "results" / "phase3_main" / "variants_features.csv"

# Hand-picked variant set (chr17 GRCh38, ClinVar). See context note in script.
# Each entry: (variant_id, chrom, pos, ref, alt, gene, category, c_notation,
#              clinvar_stars, control_pair_id_or_None)
VARIANTS = [
    # --- pathogenic ---
    ("BRCA1_splice_43076602_G_T", "17", 43076602, "G", "T", "BRCA1", "P_LP",
     "BRCA1 splice region (near c.5074+1G>A)", 3, "BRCA1_LB_43076592_A_G"),
    ("TP53_R175H_7674220_C_T",    "17",  7674220, "C", "T", "TP53",  "P_LP",
     "TP53 c.524G>A (p.R175H), genomic C>T (negative strand)", 3, "TP53_LB_7674222_G_A"),
    ("BRCA1_5266_43057063_G_A",   "17", 43057063, "G", "A", "BRCA1", "P_LP",
     "BRCA1 c.5266 region (substitution stand-in for c.5266dupC; indels filtered)", 3,
     "BRCA1_LB_43057061_C_T"),
    # --- matched benign controls (same gene, smallest |Δpos|) ---
    ("BRCA1_LB_43076592_A_G",     "17", 43076592, "A", "G", "BRCA1", "B_LB",
     "BRCA1 LB control near splice donor (Δ=10bp)", 3, None),
    ("TP53_LB_7674222_G_A",       "17",  7674222, "G", "A", "TP53",  "B_LB",
     "TP53 B/LB control near R175H (Δ=2bp)", 2, None),
    ("BRCA1_LB_43057061_C_T",     "17", 43057061, "C", "T", "BRCA1", "B_LB",
     "BRCA1 LB control near c.5266 region (Δ=2bp)", 3, None),
]


def fetch_window(fa, chrom_canonical: str, pos: int):
    start = pos - 1 - CONTEXT_HALF
    end = pos - 1 + CONTEXT_HALF
    chrom_len = fa.get_reference_length(chrom_canonical)
    if start < 0:
        seq = "N" * (-start) + fa.fetch(chrom_canonical, 0, end).upper()
    elif end > chrom_len:
        seq = fa.fetch(chrom_canonical, start, chrom_len).upper()
        seq = seq + "N" * (end - chrom_len)
    else:
        seq = fa.fetch(chrom_canonical, start, end).upper()
    return seq, CONTEXT_HALF


def load_phase3_max_abs(chrom: str, pos: int, ref: str, alt: str):
    """Return (max_abs_dD, argmax_layer, dJsd_vec[32]) from Phase 3 cache, or None."""
    if not PHASE3_CSV.exists():
        return None
    with PHASE3_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (str(row["chrom"]) == str(chrom) and int(row["pos"]) == int(pos)
                    and row["ref"] == ref and row["alt"] == alt):
                vec = np.array([float(row[f"dD_jsd_{l}"]) for l in range(32)], dtype=np.float64)
                return (float(row["max_abs_dD"]), int(row["argmax_layer"]), vec,
                        float(row["evo2_delta_loglik"]))
    return None


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR):
        t0 = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import N_LAYERS, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2, tokenize, lm_head_apply
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        import pysam

        LOG.info("loading Evo 2 ...")
        bundle = load_evo2()
        LOG.info("model: %s  N_LAYERS=%d  VOCAB=%d", bundle.loaded_variant, N_LAYERS, VOCAB_SIZE)
        layer_names = all_layer_names()

        # Token IDs for ACGT (for delta-loglik sanity)
        tok = bundle.tokenizer
        token_id = {ch: int(np.asarray(tok.tokenize(ch), dtype=np.int64)[0]) for ch in "ACGT"}
        LOG.info("token_ids ACGT = %s", token_id)

        # FASTA
        fa_path = REF_DIR / "chr17.fa"
        fa = pysam.FastaFile(str(fa_path))

        traces = []
        dD_jsd_arr = np.zeros((len(VARIANTS), N_LAYERS), dtype=np.float64)
        dD_cos_arr = np.zeros((len(VARIANTS), N_LAYERS), dtype=np.float64)
        variant_ids = []

        gpu_peak_mb = 0

        for vi, (vid, chrom, pos, ref, alt, gene, cat, cnot, stars, ctrl) in enumerate(VARIANTS):
            LOG.info("[%d/%d] %s  %s:%d  %s>%s  cat=%s",
                     vi + 1, len(VARIANTS), vid, chrom, pos, ref, alt, cat)
            canon = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
            ref_seq, local_idx = fetch_window(fa, canon, pos)
            assert len(ref_seq) == 2 * CONTEXT_HALF, f"window len {len(ref_seq)}"
            ref_at = ref_seq[local_idx]
            if ref_at != ref:
                raise RuntimeError(
                    f"FASTA ref mismatch for {vid}: fa={ref_at} expected={ref}"
                )
            alt_seq = ref_seq[:local_idx] + alt + ref_seq[local_idx + 1:]

            # ---- forward ref ----
            t_fwd = time.time()
            ids_ref = tokenize(ref_seq, bundle, device="cuda")
            hs_ref = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
            d_jsd_ref = jsd_lens(hs_ref, bundle, n_layers=N_LAYERS).numpy()
            d_cos_ref = cosine_lens(hs_ref, n_layers=N_LAYERS).numpy()
            h_norm_ref = hs_ref["norm"]
            with torch.no_grad():
                logits_ref = lm_head_apply(h_norm_ref, bundle).float()
            pred_pos = max(0, local_idx - 1)
            log_p_ref = F.log_softmax(logits_ref[0, pred_pos, :VOCAB_SIZE], dim=-1).cpu().numpy()
            ref_loglik = float(log_p_ref[token_id[ref]])
            del hs_ref, ids_ref, logits_ref, h_norm_ref
            torch.cuda.empty_cache()

            # ---- forward alt ----
            ids_alt = tokenize(alt_seq, bundle, device="cuda")
            hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
            d_jsd_alt = jsd_lens(hs_alt, bundle, n_layers=N_LAYERS).numpy()
            d_cos_alt = cosine_lens(hs_alt, n_layers=N_LAYERS).numpy()
            h_norm_alt = hs_alt["norm"]
            with torch.no_grad():
                logits_alt = lm_head_apply(h_norm_alt, bundle).float()
            log_p_alt = F.log_softmax(logits_alt[0, pred_pos, :VOCAB_SIZE], dim=-1).cpu().numpy()
            alt_loglik = float(log_p_alt[token_id[alt]])
            del hs_alt, ids_alt, logits_alt, h_norm_alt
            mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_peak_mb = max(gpu_peak_mb, mb)
            torch.cuda.empty_cache()
            LOG.info("  fwd time %.1fs  GPU peak %.0f MB", time.time() - t_fwd, mb)

            # variant token only
            col_jsd_ref = d_jsd_ref[:, local_idx]
            col_jsd_alt = d_jsd_alt[:, local_idx]
            col_cos_ref = d_cos_ref[:, local_idx]
            col_cos_alt = d_cos_alt[:, local_idx]

            assert np.isfinite(col_jsd_ref).all() and np.isfinite(col_jsd_alt).all()
            assert np.isfinite(col_cos_ref).all() and np.isfinite(col_cos_alt).all()
            assert np.isfinite(ref_loglik) and np.isfinite(alt_loglik)

            dD_jsd = col_jsd_alt - col_jsd_ref
            dD_cos = col_cos_alt - col_cos_ref
            abs_dJ = np.abs(dD_jsd)
            argmax = int(abs_dJ.argmax())
            max_abs = float(abs_dJ.max())

            dD_jsd_arr[vi] = dD_jsd
            dD_cos_arr[vi] = dD_cos
            variant_ids.append(vid)

            # Reproducibility check vs Phase 3 cache
            cached = load_phase3_max_abs(chrom, pos, ref, alt)
            cached_max = cached[0] if cached else None
            cached_argmax = cached[1] if cached else None
            cached_vec = cached[2] if cached else None
            cached_dll = cached[3] if cached else None
            rel_err = None
            if cached_vec is not None:
                # Compare full vector
                num = np.abs(dD_jsd - cached_vec).max()
                den = max(np.abs(cached_vec).max(), 1e-12)
                rel_err = float(num / den)
                LOG.info("  Phase3 cached max|ΔD|=%.6e (argmax L%d); new max|ΔD|=%.6e (argmax L%d); rel_err=%.2e",
                         cached_max, cached_argmax, max_abs, argmax + 1, rel_err)

            traces.append({
                "variant_id": vid,
                "gene": gene,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "category": cat,
                "clinvar_stars": stars,
                "c_notation": cnot,
                "control_pair_id": ctrl,
                "dD_jsd_per_layer": [float(x) for x in dD_jsd],
                "dD_cos_per_layer": [float(x) for x in dD_cos],
                "evo2_ref_loglik": ref_loglik,
                "evo2_alt_loglik": alt_loglik,
                "evo2_delta_loglik": alt_loglik - ref_loglik,
                "max_abs_dD": max_abs,
                "argmax_layer": argmax + 1,  # 1-indexed (matches Phase 3 convention)
                "phase3_cached_max_abs_dD": cached_max,
                "phase3_cached_argmax_layer": cached_argmax,
                "phase3_cached_evo2_delta_loglik": cached_dll,
                "phase3_rel_err": rel_err,
            })

        # Attach control_max_abs_dD into pathogenic entries for convenience
        by_id = {t["variant_id"]: t for t in traces}
        for t in traces:
            if t["control_pair_id"] is not None and t["control_pair_id"] in by_id:
                t["control_max_abs_dD"] = by_id[t["control_pair_id"]]["max_abs_dD"]
                t["control_argmax_layer"] = by_id[t["control_pair_id"]]["argmax_layer"]
            else:
                t["control_max_abs_dD"] = None
                t["control_argmax_layer"] = None

        # ---- Confirmation table ----
        LOG.info("")
        LOG.info("=== Reproducibility check vs Phase 3 cache ===")
        LOG.info("%-32s  %-15s  %-15s  %s", "variant", "Phase3 max|ΔD|", "New max|ΔD|", "match?")
        all_match = True
        for t in traces:
            cached = t["phase3_cached_max_abs_dD"]
            new = t["max_abs_dD"]
            ok = (cached is not None) and (t["phase3_rel_err"] is not None) and (t["phase3_rel_err"] < 1e-4)
            if not ok:
                all_match = False
            LOG.info("%-32s  %-15.6e  %-15.6e  %s",
                     t["variant_id"],
                     cached if cached is not None else float("nan"),
                     new,
                     "YES" if ok else f"NO (rel_err={t['phase3_rel_err']})")

        # Mechanism observation
        path_traces = [t for t in traces if t["category"] == "P_LP"]
        deep_count = sum(1 for t in path_traces if t["argmax_layer"] >= 16)
        LOG.info("")
        LOG.info("Pathogenic variants with argmax_layer >= 16: %d / %d", deep_count, len(path_traces))
        for t in path_traces:
            ctrl_max = t["control_max_abs_dD"]
            ctrl_arg = t["control_argmax_layer"]
            LOG.info("  %s: P max|ΔD|=%.4e (L%d) vs control %.4e (L%s)",
                     t["variant_id"], t["max_abs_dD"], t["argmax_layer"],
                     ctrl_max if ctrl_max is not None else float("nan"),
                     ctrl_arg if ctrl_arg is not None else "?")

        # ---- Write outputs ----
        out_json = {
            "phase": PHASE,
            "config": {
                "context_half": CONTEXT_HALF,
                "seed": SEED,
                "model_variant": bundle.loaded_variant,
                "n_layers": N_LAYERS,
            },
            "all_phase3_match": bool(all_match),
            "n_pathogenic_deep_argmax": int(deep_count),
            "wall_time_sec": float(time.time() - t0),
            "gpu_peak_mb": float(gpu_peak_mb),
            "variants": traces,
        }
        (PHASE_OUT_DIR / "case_studies.json").write_text(json.dumps(out_json, indent=2))
        np.savez(
            PHASE_OUT_DIR / "case_studies.npz",
            dD_jsd=dD_jsd_arr,
            dD_cos=dD_cos_arr,
            variant_ids=np.array(variant_ids, dtype=object),
        )
        ru.write_done(PHASE, PHASE_OUT_DIR, {
            "n_variants": len(traces),
            "all_phase3_match": all_match,
            "wall_time_sec": time.time() - t0,
            "gpu_peak_mb": gpu_peak_mb,
        })
        LOG.info("DONE in %.1f min  GPU peak %.0f MB", (time.time() - t0) / 60, gpu_peak_mb)


if __name__ == "__main__":
    main()

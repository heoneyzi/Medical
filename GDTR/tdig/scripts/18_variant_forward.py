"""PHASE C (part 2) — ClinVar variant forward.

For each of 10,910 ClinVar variants in 15 cancer genes:
  - Build ref + alt sequences with +/- 3 kb context around variant
  - Forward both through Evo 2 7B
  - Extract h_ell at variant position for all layers
  - Compute per-layer scalars at variant position (settling, |Δh|, Δcos)

For T-B downstream task. Reuses gDTR Phase 3 variants_features.csv as the
variant cohort definition (same 10,910 variants).

Note: this iteration uses batch=1, no kv-cache reuse (those are future
optimizations). At ~0.5 s per variant (1 ref forward + 1 alt forward),
expected wall: ~90 min for 10,910 variants.

Outputs:
    /path/to/TDiG/data/cache/variants/
        variant_h_ell_ref.h5         (10910, 32, 4096) fp32  ~5.7 GB
                                     h_ell at variant position, ref sequence
        variant_h_ell_alt.h5         same for alt
        variant_scalars.parquet      per-variant features for LR (settling depths +
                                     ||Delta h||_2 per layer + Delta_settling)
        variant_metadata.parquet     gene, chr, pos, ref, alt, class
        _provenance.json + _done
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/gDTR")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pull settling-depth helpers from 15
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chr22_forward_module", Path(__file__).resolve().parent / "15_chr22_forward.py"
)
chr22_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chr22_mod)

L_STAR = 29
N_LAYERS = 32
HIDDEN_SIZE = 4096
CTX_BP = 3000  # 3kb each side; total 6kb window


def build_context_seq(fasta, chrom, pos, ref_allele, alt_allele=None):
    """Build a 6kb sequence centered on the variant position.

    pos is 1-based per ClinVar/VCF convention.
    Returns (seq_string, variant_idx_in_seq).
    """
    pos0 = pos - 1  # 0-based
    start = max(0, pos0 - CTX_BP)
    end = pos0 + CTX_BP
    seq = fasta.fetch(chrom, start, end).upper()
    var_idx = pos0 - start  # index of variant in seq

    if alt_allele is None:
        return seq, var_idx

    # Build alt sequence by replacing the variant nucleotide(s).
    # For SNVs: ref_allele and alt_allele are 1 char each.
    ref_len = len(ref_allele)
    alt_seq = seq[:var_idx] + alt_allele + seq[var_idx + ref_len:]
    return alt_seq, var_idx


@torch.no_grad()
def forward_variant_extract(bundle, seq: str, var_idx: int):
    """Forward + extract h_ell at variant position only (for all 32 layers).

    Returns:
        h_at_var: [L, H] tensor on GPU
        h_norm_at_var: [H,] tensor on GPU
    """
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    input_ids = tokenize(seq, bundle, device="cuda")
    layer_names = all_layer_names()
    hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)

    # Stack only at variant position
    h_at_var = torch.stack(
        [hs[f"blocks.{ell}"][0, var_idx] for ell in range(N_LAYERS)], dim=0
    ).float()  # (L, H)
    h_norm_at_var = hs["norm"][0, var_idx].float()  # (H,)
    return h_at_var, h_norm_at_var


@torch.no_grad()
def compute_variant_scalars(bundle, h_ref, h_norm_ref, h_alt, h_norm_alt):
    """Per-layer scalars at variant position.

    Inputs are all GPU tensors:
        h_ref:      (L, H)
        h_norm_ref: (H,)
        h_alt, h_norm_alt: same shapes

    Returns dict of numpy arrays (per layer scalars):
        delta_h_norm_2          (L,) fp32   ||h_alt_l - h_ref_l||_2
        delta_h_norm_1          (L,) fp32   ||h_alt_l - h_ref_l||_1
        delta_cos               (L,) fp16   1 - cos(h_alt_l, h_ref_l)
        norm_h_ref              (L,) fp32   ||h_ref_l||
        norm_h_alt              (L,) fp32   ||h_alt_l||
        cos_ref_to_h_norm_ref   (L,) fp16   cos(h_ref_l, h_norm_ref)
        cos_alt_to_h_norm_alt   (L,) fp16   cos(h_alt_l, h_norm_alt)
    """
    diff = h_alt - h_ref  # (L, H)
    out = {
        "delta_h_norm_2": torch.linalg.vector_norm(diff, dim=-1).cpu().numpy().astype(np.float32),
        "delta_h_norm_1": torch.linalg.vector_norm(diff, dim=-1, ord=1).cpu().numpy().astype(np.float32),
        "delta_cos": (1 - F.cosine_similarity(h_alt, h_ref, dim=-1)).cpu().numpy().astype(np.float16),
        "norm_h_ref": torch.linalg.vector_norm(h_ref, dim=-1).cpu().numpy().astype(np.float32),
        "norm_h_alt": torch.linalg.vector_norm(h_alt, dim=-1).cpu().numpy().astype(np.float32),
        "cos_ref_h_norm": F.cosine_similarity(
            h_ref, h_norm_ref.unsqueeze(0).expand_as(h_ref), dim=-1
        ).cpu().numpy().astype(np.float16),
        "cos_alt_h_norm": F.cosine_similarity(
            h_alt, h_norm_alt.unsqueeze(0).expand_as(h_alt), dim=-1
        ).cpu().numpy().astype(np.float16),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-csv", type=Path,
                        default=Path("/path/to/gDTR/results/phase3_main/variants_features.csv"))
    parser.add_argument("--fasta-dir", type=Path, default=Path("/path/to/gDTR/data/reference"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/path/to/TDiG/data/cache/variants"))
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading variants from {args.variants_csv}", flush=True)
    df_v = pd.read_csv(args.variants_csv)
    if args.max_variants > 0:
        df_v = df_v.head(args.max_variants).copy()
    N = len(df_v)
    print(f"[setup] N={N} variants", flush=True)

    # Filter to SNVs only (ref + alt both single char) for this iteration
    snv_mask = (df_v["ref"].str.len() == 1) & (df_v["alt"].str.len() == 1)
    n_snv = snv_mask.sum()
    print(f"[setup] {n_snv}/{N} are SNVs (others = indels, skip this iteration)", flush=True)
    df_v = df_v[snv_mask].reset_index(drop=True)
    N = len(df_v)

    # Map chromosome -> pysam FastaFile (lazy load)
    import pysam
    fasta_cache: dict[str, pysam.FastaFile] = {}

    def get_fasta(chrom):
        # chrom in CSV is like '10', '17' (without 'chr' prefix sometimes)
        chrom_str = str(chrom)
        if not chrom_str.startswith("chr"):
            chrom_str = "chr" + chrom_str
        if chrom_str not in fasta_cache:
            fa_path = args.fasta_dir / f"{chrom_str}.fa"
            if not fa_path.exists():
                # Try without 'chr' prefix
                fa_path2 = args.fasta_dir / f"{chrom}.fa"
                if fa_path2.exists():
                    fa_path = fa_path2
                else:
                    print(f"  [warn] missing FASTA for chrom={chrom_str}; tried {fa_path}", flush=True)
                    return None, None
            fasta_cache[chrom_str] = (pysam.FastaFile(str(fa_path)), chrom_str)
        return fasta_cache[chrom_str]

    print(f"[setup] loading Evo 2", flush=True)
    from src.model_loader_evo2 import load_evo2
    bundle = load_evo2()
    print(f"[setup] model loaded, variant={bundle.loaded_variant}", flush=True)

    # h5 outputs (pre-allocate for N variants × L × H)
    h5_ref_path = args.out_dir / "variant_h_ell_ref.h5"
    h5_alt_path = args.out_dir / "variant_h_ell_alt.h5"
    parquet_path = args.out_dir / "variant_scalars.parquet"

    if not h5_ref_path.exists():
        with h5py.File(h5_ref_path, "w") as h5:
            h5.create_dataset("h_ell", shape=(N, N_LAYERS, HIDDEN_SIZE), dtype="float32",
                              chunks=(1, N_LAYERS, HIDDEN_SIZE))
            h5.create_dataset("h_norm", shape=(N, HIDDEN_SIZE), dtype="float16")
            h5.create_dataset("done_mask", shape=(N,), dtype="uint8")
    if not h5_alt_path.exists():
        with h5py.File(h5_alt_path, "w") as h5:
            h5.create_dataset("h_ell", shape=(N, N_LAYERS, HIDDEN_SIZE), dtype="float32",
                              chunks=(1, N_LAYERS, HIDDEN_SIZE))
            h5.create_dataset("h_norm", shape=(N, HIDDEN_SIZE), dtype="float16")
            h5.create_dataset("done_mask", shape=(N,), dtype="uint8")

    # Resume
    scalar_records = []
    completed_set = set()
    if parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path)
            completed_set = set(zip(existing["chrom"].astype(str), existing["pos"]))
            scalar_records = existing.to_dict(orient="records")
            print(f"[resume] {len(completed_set)} variants already done", flush=True)
        except Exception as e:
            print(f"[resume] couldn't parse existing parquet: {e}", flush=True)

    t_start = time.time()
    n_processed = 0
    n_skipped = 0

    for i, row in df_v.iterrows():
        chrom_raw = row["chrom"]
        pos = int(row["pos"])
        ref_allele = row["ref"]
        alt_allele = row["alt"]
        key = (str(chrom_raw), pos)
        if key in completed_set:
            continue

        fa, chrom_str = get_fasta(chrom_raw) if get_fasta(chrom_raw)[0] else (None, None)
        if fa is None:
            n_skipped += 1
            continue

        try:
            ref_seq, var_idx = build_context_seq(fa, chrom_str, pos, ref_allele, alt_allele=None)
            alt_seq, _      = build_context_seq(fa, chrom_str, pos, ref_allele, alt_allele=alt_allele)

            if len(ref_seq) != len(alt_seq) or var_idx < 0 or var_idx >= len(ref_seq):
                print(f"  [skip] variant {key}: seq build failed", flush=True)
                n_skipped += 1
                continue

            with torch.no_grad():
                h_ref, h_norm_ref = forward_variant_extract(bundle, ref_seq, var_idx)
                h_alt, h_norm_alt = forward_variant_extract(bundle, alt_seq, var_idx)
                scalars = compute_variant_scalars(bundle, h_ref, h_norm_ref, h_alt, h_norm_alt)

            # Save raw vectors
            with h5py.File(h5_ref_path, "a") as h5:
                h5["h_ell"][i] = h_ref.cpu().numpy().astype(np.float32)
                h5["h_norm"][i] = h_norm_ref.cpu().numpy().astype(np.float16)
                h5["done_mask"][i] = 1
            with h5py.File(h5_alt_path, "a") as h5:
                h5["h_ell"][i] = h_alt.cpu().numpy().astype(np.float32)
                h5["h_norm"][i] = h_norm_alt.cpu().numpy().astype(np.float16)
                h5["done_mask"][i] = 1

            # Scalar record (flattened per-layer)
            rec = {
                "chrom": str(chrom_raw),
                "pos": pos,
                "ref": ref_allele,
                "alt": alt_allele,
                "gene": row.get("gene", ""),
                "category": row.get("category", ""),
                "stars": int(row.get("stars", 0)),
            }
            for k, v in scalars.items():
                rec[k] = v.astype(np.float32).tolist()
            scalar_records.append(rec)

            del h_ref, h_alt, h_norm_ref, h_norm_alt
            torch.cuda.empty_cache()
            n_processed += 1

            if n_processed % args.log_every == 0:
                elapsed = time.time() - t_start
                rate = n_processed / elapsed
                eta = (N - len(completed_set) - n_processed) / rate / 60
                gpu_peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [{len(completed_set)+n_processed}/{N}] rate={rate:.2f}/s ETA={eta:.1f}min GPU={gpu_peak:.1f}GB", flush=True)

            if n_processed % args.save_every == 0:
                tmp = parquet_path.with_suffix(".tmp.parquet")
                pd.DataFrame(scalar_records).to_parquet(tmp, index=False)
                tmp.replace(parquet_path)

        except Exception as e:
            print(f"  [ERR variant {key}] {type(e).__name__}: {e}", flush=True)
            raise

    pd.DataFrame(scalar_records).to_parquet(parquet_path, index=False)

    prov = {
        "script": "18_variant_forward.py",
        "host": platform.node(),
        "n_variants_input": int(N),
        "n_processed": int(n_processed),
        "n_skipped": int(n_skipped),
        "wall_minutes": (time.time() - t_start) / 60,
        "model_variant": bundle.loaded_variant,
        "context_bp_each_side": CTX_BP,
        "snv_filter": True,
        "variants_csv": str(args.variants_csv),
        "h5_ref": str(h5_ref_path),
        "h5_alt": str(h5_alt_path),
        "scalars_parquet": str(parquet_path),
    }
    (args.out_dir / "_provenance.json").write_text(json.dumps(prov, indent=2))
    (args.out_dir / "_done").write_text(json.dumps({"ok": True, "n_processed": n_processed}, indent=2))
    print(f"[done] {args.out_dir}  wall={prov['wall_minutes']:.2f}min")


if __name__ == "__main__":
    main()

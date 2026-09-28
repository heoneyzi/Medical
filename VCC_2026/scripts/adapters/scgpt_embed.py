#!/usr/bin/env python3
"""Thin, official-API scGPT cell-embedding wrapper."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from scgpt.tasks import embed_data

    data = ad.read_h5ad(args.input)
    data.var["feature_name"] = data.var_names.astype(str)
    embedded = embed_data(
        data,
        args.model_dir,
        gene_col="feature_name",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        use_fast_transformer=False,
        return_new_adata=False,
    )
    values = np.asarray(embedded.obsm["X_scGPT"])
    if values.shape[0] != data.n_obs or not np.isfinite(values).all():
        raise RuntimeError("scGPT returned an invalid cell embedding matrix")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    embedded.write_h5ad(output, compression="gzip")
    print(f"[scgpt-adapter] cells={values.shape[0]} dim={values.shape[1]} output={output}")


if __name__ == "__main__":
    main()

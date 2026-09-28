#!/usr/bin/env python3
"""Build a raw-count H5AD from export_gse270828_rds.R binary arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def properties(path: Path) -> dict[str, str]:
    return dict(line.rstrip("\n").split("=", 1) for line in path.read_text().splitlines())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", required=True)
    p.add_argument("--source-rds", required=True)
    p.add_argument("--feature-readme", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf")
    args = p.parse_args()

    root, source, readme = Path(args.export_dir), Path(args.source_rds), Path(args.feature_readme)
    props = properties(root / "matrix.properties")
    n_genes, n_cells, nnz = map(int, (props["n_genes"], props["n_cells"], props["nnz"]))
    genes = (root / "genes.txt").read_text().splitlines()
    obs = pd.read_csv(root / "metadata.csv", dtype={"cell_id": str}).set_index("cell_id")
    if len(genes) != n_genes or len(obs) != n_cells:
        raise ValueError(f"dimension mismatch: genes={len(genes)}/{n_genes}, cells={len(obs)}/{n_cells}")

    indices = np.memmap(root / "indices.int32.bin", mode="r", dtype="<i4", shape=(nnz,))
    indptr = np.memmap(root / "indptr.int32.bin", mode="r", dtype="<i4", shape=(n_cells + 1,))
    data = np.memmap(root / "data.float32.bin", mode="r", dtype="<f4", shape=(nnz,))
    if indptr[0] != 0 or indptr[-1] != nnz or np.any(data < 0) or not np.all(data == np.rint(data)):
        raise ValueError("invalid or non-integer sparse count arrays")

    # The R object is genes x cells CSC. Its transpose is directly CSR in the
    # cells x genes orientation expected by AnnData, with no dense expansion.
    X = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells), copy=False).transpose().tocsr(copy=False)
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs["source_accession"] = "GSE270828"
    out.uns["provenance"] = {
        "geo_accession": "GSE270828",
        "bioproject": "PRJNA1128583",
        "matrix_origin": "Seurat RNA counts",
        "raw_integer_counts": True,
        "regulatory_target_column": "har",
        "har_to_gene_mapping_used": False,
        "source_rds_sha256": sha256(source),
        "feature_readme_sha256": sha256(readme),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    out.write_h5ad(out_path, compression=compression)

    audit = {
        "shape": [out.n_obs, out.n_vars], "nnz": int(out.X.nnz),
        "n_controls": int((out.obs["har"].astype(str) == "Non-Targeting").sum()),
        "n_regulatory_targets": int(out.obs.loc[out.obs["har"].astype(str) != "Non-Targeting", "har"].nunique()),
        "batches": out.obs["rep"].astype(str).value_counts().sort_index().to_dict(),
        "raw_integer_counts": bool(np.all(out.X.data == np.rint(out.X.data))),
        "har_to_gene_mapping_used": False,
        "h5ad": str(out_path.resolve()),
    }
    out_path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

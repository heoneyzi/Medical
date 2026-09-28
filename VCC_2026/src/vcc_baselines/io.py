"""I/O helpers: load contexts, target genes, expected gene list; align genes."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


_KNOWN_GENE_COLS = ("gene", "gene_name", "gene_symbol", "target_gene", "var_names", "names")


def read_gene_list(path: str | Path) -> list[str]:
    """Read gene/target names (and order).

    Accepts a CSV with a recognised header column (gene, gene_name, target_gene, ...),
    a header-less single-column CSV, or a plain newline-delimited TXT.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in _KNOWN_GENE_COLS:
            if cand in cols_lower:
                return df[cols_lower[cand]].astype(str).tolist()
        # No recognised header: if single column, the first row was likely data.
        if df.shape[1] == 1 and str(df.columns[0]).lower() not in _KNOWN_GENE_COLS:
            s = pd.read_csv(path, header=None).iloc[:, 0]
            return s.astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    text = path.read_text().strip()
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def read_targets(path: str | Path) -> list[str]:
    """Read the list of target genes to predict."""
    return read_gene_list(path)


def to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def align_to_genes(adata: ad.AnnData, genes: list[str], fill: float = 0.0) -> ad.AnnData:
    """Reindex `adata` onto the exact `genes` order.

    Missing genes are inserted as `fill`; extra genes are dropped. This makes the
    output var-space identical to the challenge gene list, which cell-eval
    requires for a valid submission.
    """
    genes = list(map(str, genes))
    var_names = adata.var_names.astype(str)
    idx = pd.Index(var_names).get_indexer(genes)  # -1 where missing
    X = adata.X
    n = adata.n_obs
    present = idx >= 0
    if sp.issparse(X):
        # Preserve sparsity: real perturbation atlases can contain millions of
        # cells and cannot be materialized as a dense cells-by-genes matrix.
        sub = X.tocsr()[:, idx[present]].tocoo()
        target_cols = np.flatnonzero(present)
        out = sp.csr_matrix(
            (sub.data.astype(np.float32), (sub.row, target_cols[sub.col])),
            shape=(n, len(genes)), dtype=np.float32,
        )
        if fill != 0 and (~present).any():
            out = out.toarray()
            out[:, ~present] = fill
    else:
        X = np.asarray(X, dtype=np.float32)
        out = np.full((n, len(genes)), fill, dtype=np.float32)
        out[:, present] = X[:, idx[present]]
    new = ad.AnnData(X=out, obs=adata.obs.copy())
    new.var_names = genes
    n_missing = int((~present).sum())
    if n_missing:
        new.uns["n_genes_filled"] = n_missing
    return new


def pseudobulk(adata: ad.AnnData) -> np.ndarray:
    """Mean expression vector across cells (dense, float32)."""
    return to_dense(adata.X).astype(np.float32).mean(axis=0)


def list_contexts(contexts_dir: str | Path, control_h5ad: str) -> list[str]:
    """Return the names of context sub-folders that contain a control file."""
    contexts_dir = Path(contexts_dir)
    out = []
    for p in sorted(contexts_dir.iterdir()) if contexts_dir.exists() else []:
        if p.is_dir() and (p / control_h5ad).exists():
            out.append(p.name)
    return out


def load_controls(contexts_dir: str | Path, context: str, control_h5ad: str) -> ad.AnnData:
    return ad.read_h5ad(Path(contexts_dir) / context / control_h5ad)


def load_truth(contexts_dir: str | Path, context: str, truth_h5ad: str) -> ad.AnnData | None:
    p = Path(contexts_dir) / context / truth_h5ad
    return ad.read_h5ad(p) if p.exists() else None


def save_prediction(adata: ad.AnnData, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # cell-eval2's sampling-noise correction requires canonical sparse
    # coordinates.  H5AD slices may be mathematically valid while retaining an
    # unsorted CSR index, so normalize the representation before persistence.
    if sp.issparse(adata.X):
        X = adata.X.tocsr(copy=True)
        X.sum_duplicates()
        X.eliminate_zeros()
        X.sort_indices()
        adata.X = X
    # Ensure obs dtypes are submission-friendly (string perturbation labels).
    adata.write_h5ad(path)

#!/usr/bin/env python3
"""Build an explicit symbol-to-Ensembl map; never choose ambiguous aliases."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vcc_baselines.config import Config
from vcc_baselines.external_frozen import context_inputs


ROOT = Path(__file__).resolve().parents[1]


def tokens(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgnc", type=Path,
                        default=ROOT / "data/mappings/hgnc_complete_set.txt")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "manifests/gene_map.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/gene_mapping_summary.json")
    args = parser.parse_args()
    if not args.hgnc.is_file():
        raise SystemExit(
            f"AUDIT_REQUIRED: missing HGNC table {args.hgnc}; run "
            "scripts/setup_external_frozen_models.sh gene-map")

    hgnc = pd.read_csv(args.hgnc, sep="\t", dtype=str)
    required = {"symbol", "ensembl_gene_id", "alias_symbol", "prev_symbol"}
    missing = required - set(hgnc.columns)
    if missing:
        raise SystemExit(f"AUDIT_REQUIRED: HGNC table lacks {sorted(missing)}")
    exact: dict[str, set[tuple[str, str]]] = defaultdict(set)
    alias: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in hgnc.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        ensembl = getattr(row, "ensembl_gene_id")
        if pd.isna(ensembl) or not str(ensembl).strip():
            continue
        pair = (symbol, str(ensembl).split(".")[0])
        exact[symbol].add(pair)
        for name in tokens(getattr(row, "alias_symbol")) + tokens(getattr(row, "prev_symbol")):
            alias[name].add(pair)

    datasets: dict[str, set[str]] = {}
    all_genes: set[str] = set()
    for config_path in args.config:
        config_path = str(Path(config_path).resolve())
        genes: set[str] = set()
        for input_path in context_inputs(config_path).values():
            import anndata as ad
            data = ad.read_h5ad(input_path, backed="r")
            genes.update(map(str, data.var_names))
            data.file.close()
        name = Path(config_path).parent.name
        datasets[name] = genes
        all_genes.update(genes)

    rows = []
    for gene in sorted(all_genes):
        candidates = exact.get(gene, set())
        status = "exact"
        note = "HGNC approved symbol"
        if not candidates:
            candidates = alias.get(gene, set())
            status = "alias"
            note = "unique HGNC alias/previous symbol"
        if len(candidates) == 1:
            approved, ensembl = next(iter(candidates))
        elif len(candidates) > 1:
            approved, ensembl, status = "", "", "ambiguous"
            note = ";".join(f"{symbol}:{ens}" for symbol, ens in sorted(candidates))
        else:
            approved, ensembl, status = "", "", "missing"
            note = "not found in HGNC approved/alias/previous symbols"
        rows.append({
            "vcc_symbol": gene, "ensembl_id": ensembl, "hgnc_symbol": approved,
            "replogle_symbol": approved or gene, "uce_symbol": approved or gene,
            "transcriptformer_id": ensembl, "geneformer_id": ensembl,
            "scgpt_symbol": approved or gene, "scfoundation_symbol": approved or gene,
            "scprint_id": ensembl, "mapping_status": status, "mapping_note": note,
        })
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    report = {
        "hgnc_source": str(args.hgnc.resolve()),
        "n_union_genes": len(frame),
        "status": frame["mapping_status"].value_counts().to_dict(),
        "datasets": {
            name: {
                "n_genes": len(genes),
                "mapped_exact_or_alias": int(frame[
                    frame["vcc_symbol"].isin(genes) &
                    frame["mapping_status"].isin(["exact", "alias"])
                ].shape[0]),
            } for name, genes in datasets.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

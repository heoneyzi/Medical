"""Phase 3 main prep: stratify ClinVar 15-gene variants for the main run.

Reads `/path/to/gDTR/data/variants/clinvar_15gene.tsv` and produces
`clinvar_15gene_stratified.tsv` with up to 350 variants per (gene, category)
cell. SNVs only. P/LP and B/LB go into the binary classifier; VUS is kept
separately for post-hoc ranking.

Outputs:
  /path/to/gDTR/data/variants/clinvar_15gene_stratified.tsv
  /path/to/gDTR/data/variants/clinvar_15gene_stratified.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GDTR_ROOT = Path("/path/to/gDTR")
IN_TSV = GDTR_ROOT / "data" / "variants" / "clinvar_15gene.tsv"
OUT_TSV = GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.tsv"
OUT_JSON = GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.json"

GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)
CATEGORIES_TRAIN = ("P_LP", "B_LB")  # for binary classifier
CATEGORIES_HOLD = ("VUS",)           # for post-hoc ranking
CAP_PER_CELL = 350                   # 15 genes x 2 cats x 350 = 10,500 max
CAP_VUS_PER_GENE = 200               # cap VUS too for downstream ranking
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows_per_cell: dict[tuple, list[dict]] = {}
    rows_vus: dict[str, list[dict]] = {g: [] for g in GENES}
    n_total = 0
    n_snv = 0
    n_kept_train = 0
    n_kept_vus = 0
    skipped_indel = 0

    with IN_TSV.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            d = dict(zip(header, parts))
            n_total += 1
            if d["gene"] not in GENES:
                continue
            ref = d["ref"]; alt = d["alt"]
            # SNV only — clean substitution
            if len(ref) != 1 or len(alt) != 1:
                skipped_indel += 1
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                skipped_indel += 1
                continue
            n_snv += 1
            d["pos"] = int(d["pos"])
            d["stars"] = int(d.get("stars", 0))
            cat = d["category"]
            if cat in CATEGORIES_TRAIN:
                key = (d["gene"], cat)
                rows_per_cell.setdefault(key, []).append(d)
            elif cat in CATEGORIES_HOLD:
                rows_vus[d["gene"]].append(d)

    # Stratify train cells
    out_rows = []
    cell_counts = {}
    for g in GENES:
        for c in CATEGORIES_TRAIN:
            key = (g, c)
            rows = rows_per_cell.get(key, [])
            if len(rows) > CAP_PER_CELL:
                # Prefer higher star + a deterministic shuffle
                rows = sorted(rows, key=lambda r: (-r["stars"], r["pos"]))
                # Take top-CAP_PER_CELL by stars (then pos)
                rows = rows[:CAP_PER_CELL]
            cell_counts[f"{g}_{c}"] = len(rows)
            out_rows.extend(rows)
            n_kept_train += len(rows)

    # Stratify VUS
    vus_rows = []
    vus_counts = {}
    for g in GENES:
        rows = rows_vus[g]
        if len(rows) > CAP_VUS_PER_GENE:
            rows = sorted(rows, key=lambda r: (-r["stars"], r["pos"]))
            rows = rows[:CAP_VUS_PER_GENE]
        vus_counts[g] = len(rows)
        vus_rows.extend(rows)
        n_kept_vus += len(rows)

    # Write combined TSV (train + VUS rows; downstream filters by category)
    cols = ["chrom", "pos", "ref", "alt", "gene", "clnsig", "clnrevstat",
            "stars", "category", "context_5kb_start", "context_5kb_end"]
    with OUT_TSV.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in out_rows + vus_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Sidecar
    sidecar = {
        "input_tsv": str(IN_TSV),
        "n_input": n_total,
        "n_snv": n_snv,
        "n_skipped_indel_or_nonacgt": skipped_indel,
        "cap_per_cell_train": CAP_PER_CELL,
        "cap_per_gene_vus": CAP_VUS_PER_GENE,
        "n_kept_train": n_kept_train,
        "n_kept_vus": n_kept_vus,
        "n_total_kept": n_kept_train + n_kept_vus,
        "cell_counts_train": cell_counts,
        "vus_counts": vus_counts,
        "categories_train": list(CATEGORIES_TRAIN),
        "categories_hold": list(CATEGORIES_HOLD),
        "seed": SEED,
        "selection_rule": (
            "If cell N > cap, sort by (-stars, pos) and take top-cap. "
            "Deterministic — no rng.choice."
        ),
    }
    OUT_JSON.write_text(json.dumps(sidecar, indent=2))
    print(f"[stratify] wrote {OUT_TSV} (train={n_kept_train}, VUS={n_kept_vus})")
    print(f"[stratify] sidecar {OUT_JSON}")


if __name__ == "__main__":
    main()

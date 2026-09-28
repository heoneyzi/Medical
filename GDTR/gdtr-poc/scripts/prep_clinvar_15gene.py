"""Phase 3 prep: ClinVar 15-gene filter, stratified by P/LP, B/LB, VUS with star cutoffs.

Outputs:
    /path/to/gDTR/data/variants/clinvar_15gene.tsv
    /path/to/gDTR/data/variants/clinvar_15gene.json
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CLINVAR_VCF = "/path/to/gDTR/data/variants/clinvar.vcf.gz"
GTF_GZ = "/path/to/gDTR/data/annotation/gencode.v44.annotation.gtf.gz"

OUT_TSV = Path("/path/to/gDTR/data/variants/clinvar_15gene.tsv")
OUT_JSON = Path("/path/to/gDTR/data/variants/clinvar_15gene.json")

GENES = [
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
]

# Chromosome lengths (GRCh38, primary assembly), for context_5kb clamping.
# Pulled from GRCh38.p14 sequence report.
CHROM_LEN = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
    "MT": 16569, "M": 16569,
}

# Star levels per ClinVar review status convention.
# https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/
STAR_MAP = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "criteria_provided,_single_submitter": 1,
    "no_assertion_criteria_provided": 0,
    "no_assertion_provided": 0,
    "no_classification_provided": 0,
    "no_classifications_from_unflagged_records": 0,
}


def clinvar_stars(revstat: str) -> int:
    if not revstat:
        return 0
    key = revstat.strip().lower()
    return STAR_MAP.get(key, 0)


PLP_RE = re.compile(r"\b(Pathogenic|Likely_pathogenic)\b", re.IGNORECASE)
BLB_RE = re.compile(r"\b(Benign|Likely_benign)\b", re.IGNORECASE)
VUS_RE = re.compile(r"\bUncertain_significance\b", re.IGNORECASE)


def categorize(clnsig: str, stars: int) -> str | None:
    if not clnsig:
        return None
    has_plp = bool(PLP_RE.search(clnsig))
    has_blb = bool(BLB_RE.search(clnsig))
    if has_plp and has_blb:
        # Conflicting Pathogenic+Benign — drop
        return None
    if has_plp and stars >= 2:
        return "P_LP"
    if has_blb and stars >= 2:
        return "B_LB"
    if VUS_RE.search(clnsig) and stars >= 1 and not (has_plp or has_blb):
        return "VUS"
    return None


def parse_info(info: str) -> dict:
    out: dict[str, str] = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
        else:
            out[kv] = "1"
    return out


def gene_coords_from_gtf() -> dict[str, tuple[str, int, int, str]]:
    """Return {gene_symbol: (chrom_no_prefix, start_1based, end, strand)} for our 15 genes.

    Picks the widest gene record per symbol (in case of multiple Ensembl entries).
    """
    targets = set(GENES)
    coords: dict[str, tuple[str, int, int, str]] = {}
    print(f"[clinvar] scanning GTF for {len(targets)} genes...", flush=True)
    pat_name = re.compile(r'gene_name "([^"]+)"')
    pat_type = re.compile(r'gene_type "([^"]+)"')
    n_lines = 0
    with gzip.open(GTF_GZ, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n_lines += 1
            # Cheap reject — skip non-gene records before any regex
            if "\tgene\t" not in line:
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            mname = pat_name.search(attrs)
            if mname is None:
                continue
            sym = mname.group(1)
            if sym not in targets:
                continue
            mtype = pat_type.search(attrs)
            gtype = mtype.group(1) if mtype else ""
            # Prefer protein_coding records when picking
            chrom = parts[0].lstrip("chr")
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            cur = coords.get(sym)
            if cur is None:
                coords[sym] = (chrom, start, end, strand)
            else:
                # Pick widest span (or extend if same chrom)
                cchrom, cstart, cend, cstrand = cur
                if cchrom == chrom:
                    coords[sym] = (chrom, min(cstart, start), max(cend, end), strand)
                # else keep first
    missing = sorted(targets - set(coords))
    if missing:
        print(f"[clinvar] WARNING missing genes from GTF: {missing}", flush=True)
    for sym, c in sorted(coords.items()):
        print(f"  {sym}: {c}", flush=True)
    return coords


def main() -> None:
    t0 = time.time()
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)

    coords = gene_coords_from_gtf()
    if not coords:
        raise RuntimeError("No gene coordinates parsed from GENCODE GTF.")

    # Build per-chromosome interval list for fast lookup.
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for sym, (chrom, s, e, _) in coords.items():
        by_chrom[chrom].append((s, e, sym))
    for c in by_chrom:
        by_chrom[c].sort()

    # Parse ClinVar VCF, restricting to chromosomes that contain our genes.
    relevant_chroms = set(by_chrom.keys())
    print(f"[clinvar] chromosomes of interest: {sorted(relevant_chroms)}", flush=True)

    rows: list[dict] = []
    cat_counts: Counter = Counter()
    gene_cat_counts: dict[str, Counter] = {g: Counter() for g in GENES}
    clinvar_filedate = None
    n_records = 0

    with gzip.open(CLINVAR_VCF, "rt") as f:
        for line in f:
            if line.startswith("##"):
                if line.startswith("##fileDate="):
                    clinvar_filedate = line.strip().split("=", 1)[1]
                continue
            if line.startswith("#"):
                continue
            n_records += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _id, ref, alt, _qual, _filt, info = parts[:8]
            if chrom not in relevant_chroms:
                continue
            pos_i = int(pos)
            # Find which gene (if any) contains this position
            matched_gene = None
            for s, e, sym in by_chrom[chrom]:
                if s <= pos_i <= e:
                    matched_gene = sym
                    break
                if s > pos_i:
                    break
            if matched_gene is None:
                continue
            info_d = parse_info(info)
            clnsig = info_d.get("CLNSIG", "")
            revstat = info_d.get("CLNREVSTAT", "")
            stars = clinvar_stars(revstat)
            category = categorize(clnsig, stars)
            if category is None:
                continue
            chrom_full = chrom
            chrom_len = CHROM_LEN.get(chrom_full, pos_i + 5000)
            ctx_start = max(0, pos_i - 5000)
            ctx_end = min(chrom_len, pos_i + 5000)
            rows.append({
                "chrom": chrom_full,
                "pos": pos_i,
                "ref": ref,
                "alt": alt,
                "gene": matched_gene,
                "clnsig": clnsig,
                "clnrevstat": revstat,
                "stars": stars,
                "category": category,
                "context_5kb_start": ctx_start,
                "context_5kb_end": ctx_end,
            })
            cat_counts[category] += 1
            gene_cat_counts[matched_gene][category] += 1

    print(f"[clinvar] scanned {n_records:,} records, kept {len(rows):,}", flush=True)
    print(f"[clinvar] category counts: {dict(cat_counts)}", flush=True)

    # Write TSV
    cols = ["chrom", "pos", "ref", "alt", "gene", "clnsig", "clnrevstat",
            "stars", "category", "context_5kb_start", "context_5kb_end"]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"[clinvar] wrote {OUT_TSV}", flush=True)

    sidecar = {
        "clinvar_release_fileDate": clinvar_filedate,
        "source_vcf": CLINVAR_VCF,
        "n_total_kept": len(rows),
        "category_counts": dict(cat_counts),
        "per_gene_counts": {
            g: dict(gene_cat_counts[g]) for g in GENES
        },
        "gene_coords": {g: list(coords[g]) for g in coords},
        "missing_genes_from_gtf": sorted(set(GENES) - set(coords)),
        "star_definition": "Per ClinVar review_status convention; see STAR_MAP in script.",
        "categories": {
            "P_LP": "Pathogenic|Likely_pathogenic AND stars >= 2",
            "B_LB": "Benign|Likely_benign AND stars >= 2",
            "VUS": "Uncertain_significance AND stars >= 1",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[clinvar] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

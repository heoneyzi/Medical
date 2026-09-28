#!/usr/bin/env python3
"""Build ``a549_compounds.csv`` (SMILES + MoA) for the structure MVP.

Two MoA sources (choose with ``--moa-source``; ``auto`` = ``lincs`` if
``--a549-dir`` is given, else ``deposit``):

* **lincs** (A549-native, richer): MoA from the LINCS A549 consensus
  (``Metadata_moa``, standard Drug-Repurposing-Hub strings), SMILES joined from
  the deposit's ``jump_target2_annotations.csv`` via ``broad_sample``.
* **deposit**: MoA + SMILES from
  ``compound_annotations/Data_S2_jump_annotated_compounds.tsv`` (``optarg_MoA``).

Anchor compounds (``anchor_compounds.tsv``) are excluded from the query set.
Prints MoA distributions so you can see coverage. No external download needed.

Usage
-----
    python prepare_a549_compounds.py \
        --deposit ./final_data \
        --a549-dir ./a549_data \
        --out ./a549_compounds.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import moa_mapping as mm  # noqa: E402


def _read(path, **kw):
    return pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".tsv.gz")) else ",", **kw)


def _pick(cols, aliases):
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a in cols:
            return a
        if a.lower() in low:
            return low[a.lower()]
    return None


def _norm_broad(x):
    if not isinstance(x, str):
        return None
    m = re.match(r"(BRD[-_][A-Za-z0-9]+)", x.strip())
    return m.group(1).upper() if m else x.strip().upper()


def _find_lincs(a549_dir):
    if os.path.isfile(a549_dir):
        return a549_dir
    hits = glob.glob(os.path.join(a549_dir, "**", "*a549*consensus*modz*.csv.gz"), recursive=True)
    hits = [h for h in hits if "feature_select" not in os.path.basename(h)]
    return sorted(hits)[0] if hits else None


def build_from_lincs(a549_dir, deposit):
    """A549-native: LINCS MoA + SMILES from jump_target2_annotations."""
    lincs_path = _find_lincs(a549_dir)
    if not lincs_path:
        print(f"[lincs] no consensus found under {a549_dir}; falling back to deposit")
        return None
    lincs = pd.read_csv(lincs_path, usecols=lambda c: c.startswith("Metadata_"))
    bcol = _pick(lincs.columns, ["Metadata_broad_sample"])
    mcol = _pick(lincs.columns, ["Metadata_moa"])
    if not bcol or not mcol:
        print("[lincs] missing broad_sample/moa columns; falling back to deposit")
        return None
    lincs = lincs[[bcol, mcol]].dropna(subset=[bcol]).copy()
    lincs["_core"] = lincs[bcol].map(_norm_broad)
    lincs = lincs.drop_duplicates("_core")

    t2p = os.path.join(deposit, "compound_annotations", "jump_target2_annotations.csv")
    if not os.path.isfile(t2p):
        print(f"[lincs] {t2p} not found; falling back to deposit")
        return None
    t2 = _read(t2p)
    tb, ts = _pick(t2.columns, ["broad_sample"]), _pick(t2.columns, ["smiles", "SMILES"])
    tk = _pick(t2.columns, ["InChIKey", "IKey"])
    if not tb or not ts:
        print("[lincs] jump_target2 missing broad_sample/smiles; falling back to deposit")
        return None
    t2 = t2[[c for c in [tb, ts, tk] if c]].dropna(subset=[tb, ts]).copy()
    t2["_core"] = t2[tb].map(_norm_broad)
    t2 = t2.drop_duplicates("_core")

    merged = lincs.merge(t2, on="_core", how="inner")
    print(f"[lincs] A549 compounds: {len(lincs)} | with SMILES via jump_target2: {len(merged)}")
    return pd.DataFrame({
        "id": merged["_core"],
        "smiles": merged[ts].astype(str),
        "moa_raw": merged[mcol].astype(str),
        "ikey": merged[tk].astype(str) if tk else "",
    })


def build_from_deposit(deposit):
    p = os.path.join(deposit, "compound_annotations", "Data_S2_jump_annotated_compounds.tsv")
    if not os.path.isfile(p):
        raise SystemExit(f"Not found: {p}\nCheck --deposit points at final_data.")
    df = _read(p)
    scol = _pick(df.columns, ["SMILES", "SMILES_JUMP_canonical", "canonical_SMILES", "smiles"])
    mcol = _pick(df.columns, ["optarg_MoA", "MOA", "moa", "cluster_moa"])
    kcol = _pick(df.columns, ["IKey", "InChIKey"])
    icol = _pick(df.columns, ["pert_iname", "name", "IKey"])
    if not scol or not mcol:
        raise SystemExit(f"{p} needs SMILES + MoA; found {list(df.columns)[:15]}")
    return pd.DataFrame({
        "id": df[icol].astype(str) if icol else range(len(df)),
        "smiles": df[scol].astype(str),
        "moa_raw": df[mcol].astype(str),
        "ikey": df[kcol].astype(str) if kcol else "",
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit", required=True, help="path to final_data")
    ap.add_argument("--a549-dir", help="LINCS A549 consensus dir (enables A549-native MoA)")
    ap.add_argument("--moa-source", default="auto", choices=["auto", "lincs", "deposit"])
    ap.add_argument("--out", default="./a549_compounds.csv")
    ap.add_argument("--a549-only", action="store_true",
                    help="(deposit source) keep only compounds also profiled in A549")
    args = ap.parse_args(argv)

    source = args.moa_source
    if source == "auto":
        source = "lincs" if args.a549_dir else "deposit"

    base = None
    if source == "lincs":
        if not args.a549_dir:
            raise SystemExit("--moa-source lincs requires --a549-dir")
        base = build_from_lincs(args.a549_dir, args.deposit)
    if base is None:
        source = "deposit"
        base = build_from_deposit(args.deposit)
    print(f"[source] using MoA source: {source}")

    base = base[base["smiles"].str.len() > 3].copy()

    # diagnostics: raw MoA distribution (top 15)
    vc = base["moa_raw"].value_counts().head(15)
    print("[moa] top raw MoA strings:")
    for k, v in vc.items():
        print(f"      {v:>5}  {k}")

    base["moa_cluster"] = mm.annotate_clusters(base["moa_raw"])
    mapped = base[base["moa_cluster"].notna()].drop_duplicates("smiles").reset_index(drop=True)
    print(f"[map] {len(base)} rows -> {len(mapped)} in the 6 clusters | "
          f"{mm.summarize_mapping(base['moa_raw'])}")

    # exclude anchors
    anchors = os.path.join(args.deposit, "compound_annotations", "anchor_compounds.tsv")
    if os.path.isfile(anchors):
        adf = _read(anchors)
        acol = _pick(adf.columns, ["canonical_SMILES", "SMILES", "smiles"])
        if acol:
            aset = set(adf[acol].astype(str))
            before = len(mapped)
            mapped = mapped[~mapped["smiles"].isin(aset)].reset_index(drop=True)
            print(f"[anchors] excluded {before - len(mapped)} anchor compounds")

    # A549 grounding flag (deposit source only; lincs source is already A549)
    mapped["in_a549"] = True if source == "lincs" else pd.NA
    if source == "deposit" and args.a549_dir:
        keys = _a549_inchikeys(args.a549_dir, os.path.join(args.deposit, "compound_annotations", "jump_target2_annotations.csv"))
        if keys:
            mapped["in_a549"] = mapped["ikey"].isin(keys)
            print(f"[a549] {int(mapped['in_a549'].sum())}/{len(mapped)} also profiled in A549")
            if args.a549_only:
                mapped = mapped[mapped["in_a549"]].reset_index(drop=True)
                print(f"[a549] --a549-only: kept {len(mapped)}")

    if len(mapped) < 30:
        print("WARNING: few compounds — metrics will be noisy. Try --moa-source lincs "
              "(with --a549-dir), or check the deposit paths.")

    mapped[["id", "smiles", "moa_raw", "moa_cluster", "in_a549"]].to_csv(args.out, index=False)
    print(f"\nWrote {len(mapped)} compounds -> {args.out}")
    return 0


def _a549_inchikeys(a549_dir, target2_path):
    lincs_path = _find_lincs(a549_dir)
    if not lincs_path:
        return set()
    lincs = pd.read_csv(lincs_path, usecols=lambda c: c.startswith("Metadata_"))
    bcol = _pick(lincs.columns, ["Metadata_broad_sample"])
    if not bcol:
        return set()
    lincs_broad = {_norm_broad(x) for x in lincs[bcol].dropna()}
    keys = set()
    if os.path.isfile(target2_path):
        t2 = _read(target2_path)
        tb, tk = _pick(t2.columns, ["broad_sample"]), _pick(t2.columns, ["InChIKey", "IKey"])
        if tb and tk:
            t2 = t2[[tb, tk]].dropna()
            t2["_c"] = t2[tb].map(_norm_broad)
            keys |= set(t2[t2["_c"].isin(lincs_broad)][tk].astype(str))
    return keys


if __name__ == "__main__":
    raise SystemExit(main())

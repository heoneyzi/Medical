#!/usr/bin/env python3
"""PhenoBridge A549 MVP — Direction A (structure-based search engine).

Demonstrates the PhenoCompass *product* on A549-relevant compounds: score
chemical **structures** against the U2OS-derived MoA anchors and show that

  (1) each compound's own MoA is recovered from structure alone
      (cross-context MoA recovery), and
  (2) given a few A549 hits of a MoA as "customer anchors", the truly related
      compounds rank at the top — including **structurally independent**
      chemotypes (low Tanimoto, high score). This is the PhenoBridge Expand loop.

Structure embeddings are cell-line independent, so this runs today with only the
checkpoints (no morphology input). The A549 morphology / DINO cross-line test is
Direction B (see ROADMAP_B_morphology.md).

Run (real)::

    python a549_mvp/run_mvp.py \
        --final-model-dir ./final_model \
        --a549-compounds  ./a549_compounds.csv \
        --out-dir         ./a549_mvp_out

``a549_compounds.csv`` needs a SMILES column and a MoA column (any of:
smiles/SMILES/canonical_smiles and moa/Metadata_moa/MoA). Build it with
``prepare_a549_compounds.py``.

Dry run (no checkpoints/torch, synthetic + watermarked)::

    python a549_mvp/run_mvp.py --mock --synthetic-compounds --out-dir ./out_mock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mvp_core as core  # noqa: E402
import moa_mapping as mm  # noqa: E402
import plots  # noqa: E402

SMILES_ALIASES = ["smiles", "SMILES", "canonical_smiles", "canonical_SMILES", "Metadata_SMILES", "Metadata_smiles"]
MOA_ALIASES = ["moa", "moa_raw", "MoA", "Metadata_moa", "MOA", "optarg_MoA", "cluster_moa"]
CLUSTER_ALIASES = ["moa_cluster"]
ID_ALIASES = ["id", "Metadata_broad_sample", "broad_sample", "pert_iname", "name"]


def _pick(cols, aliases):
    for a in aliases:
        if a in cols:
            return a
    return None


def load_compounds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t" if path.endswith((".tsv", ".tsv.gz")) else ",")
    scol = _pick(df.columns, SMILES_ALIASES)
    mcol = _pick(df.columns, MOA_ALIASES)
    ccol = _pick(df.columns, CLUSTER_ALIASES)  # pre-mapped cluster (from prepare_a549_compounds.py)
    if scol is None or (mcol is None and ccol is None):
        raise SystemExit(
            f"{path} must have a SMILES column ({SMILES_ALIASES}) and a MoA column "
            f"({MOA_ALIASES}) or a moa_cluster column. Found: {list(df.columns)[:12]}"
        )
    icol = _pick(df.columns, ID_ALIASES)
    out = pd.DataFrame({
        "id": df[icol].astype(str) if icol else [f"cpd_{i}" for i in range(len(df))],
        "smiles": df[scol].astype(str),
        "moa_raw": (df[mcol] if mcol else df[ccol]).astype(str),
    })
    if ccol is not None:
        # trust the already-mapped cluster; blank/"nan"/other -> unmapped
        out["moa_cluster"] = [c if c in mm.ANCHOR_MOAS else None for c in df[ccol].astype(str)]
    else:
        out["moa_cluster"] = mm.annotate_clusters(out["moa_raw"])
    out = out[out["smiles"].str.len() > 0].drop_duplicates("smiles").reset_index(drop=True)
    return out


def make_synthetic_compounds(n_per_moa: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moa_str = {"mtor_pi3k": "PI3K inhibitor", "hsp90": "HSP90 inhibitor",
               "jak_rock": "JAK inhibitor", "hdac": "HDAC inhibitor",
               "mapk": "MEK inhibitor", "cdk": "CDK inhibitor"}
    rows = []
    for m in mm.ANCHOR_MOAS:
        for j in range(n_per_moa):
            # fake but syntactically-valid-ish SMILES tokens (only for mock wiring)
            rows.append({"id": f"{m}_{j}", "smiles": f"C{'C'*(j%8)}O_{m}_{j}",
                         "moa_raw": moa_str[m]})
    df = pd.DataFrame(rows)
    df["moa_cluster"] = mm.annotate_clusters(df["moa_raw"])
    return df


def _enable_legacy_checkpoint_loading():
    """Make PyTorch>=2.6 load the released Lightning checkpoints.

    Two incompatibilities with the checkpoints (your own, trusted files):
      * torch 2.6 defaults ``weights_only=True`` -> trips on the omegaconf
        configs stored in ``hyper_parameters``. We force ``weights_only=False``.
      * the checkpoints were pickled by the original training package
        ``multimodal_contrastive`` (since renamed to ``phenocompass``); we alias
        that module name to ``phenocompass`` so unpickling resolves the real
        classes (falling back to a harmless stub for any missing submodule).
    """
    import importlib
    import types
    import importlib.abc
    import importlib.machinery

    import torch

    class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            top = name.split(".")[0]
            if top == "multimodal_contrastive":
                return importlib.machinery.ModuleSpec(name, self)
            return None

        _SEARCH = [
            "phenocompass.models.components", "phenocompass.models.contrastive",
            "phenocompass.models.loss", "phenocompass.models.clr",
            "phenocompass.models.utils", "phenocompass.models",
            "phenocompass.data.dataset", "phenocompass.data.featurization",
        ]

        def create_module(self, spec):
            # Return a proxy module whose attribute lookup resolves a class by
            # NAME against phenocompass (the renamed package), regardless of the
            # original multimodal_contrastive sub-path. Falls back to the aliased
            # sub-path, then to a harmless dummy for truly-unused references.
            target = spec.name.replace("multimodal_contrastive", "phenocompass", 1)
            proxy = types.ModuleType(spec.name)
            proxy.__path__ = []

            def _resolve(name):
                for modname in [target] + self._SEARCH:
                    try:
                        mod = importlib.import_module(modname)
                    except Exception:
                        continue
                    if hasattr(mod, name):
                        return getattr(mod, name)
                return type(name, (object,), {"__init__": lambda s, *a, **k: None})

            proxy.__getattr__ = _resolve
            return proxy

        def exec_module(self, module):
            pass

    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())

    if not getattr(torch.load, "_weights_only_forced", False):
        _orig = torch.load

        def _patched(*a, **k):
            k["weights_only"] = False
            return _orig(*a, **k)

        _patched._weights_only_forced = True
        torch.load = _patched

    # belt-and-suspenders: allowlist omegaconf globals for weights_only paths
    try:
        import omegaconf
        torch.serialization.add_safe_globals([
            omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
        ])
    except Exception:
        pass


def load_real_phenocompass(final_model_dir, anchor_modality, device, verbose):
    src = os.path.join(_REPO, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from phenocompass import PhenoCompass
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not import phenocompass. Run `pip install -e {_REPO}` "
            f"(needs torch, torch_geometric, rdkit), or use --mock.\nError: {e}")
    _enable_legacy_checkpoint_loading()
    return PhenoCompass(anchor_modality=anchor_modality, device=device,
                        verbose=verbose, final_model_dir=final_model_dir)


def anchor_smiles_for(pc, moa) -> List[str]:
    d = pc.anchor_embeddings_dict[moa]
    first = next(iter(d.values()))
    return [str(s) for s in first.index.tolist()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-model-dir", help="dir with model_ckpts/ and anchors/")
    ap.add_argument("--a549-compounds", help="CSV/TSV of A549 compounds (SMILES + MoA)")
    ap.add_argument("--out-dir", default="./a549_mvp_out")
    ap.add_argument("--anchor-modality", default="joint", choices=["joint", "struct", "morph"],
                    help="anchor space to score structures against (joint carries the U2OS morphology signature)")
    ap.add_argument("--n-anchor", type=int, default=3, help="customer anchors per MoA (Demo 2)")
    ap.add_argument("--k", type=int, default=10, help="precision@k for Demo 2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--synthetic-compounds", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    moa_list = mm.ANCHOR_MOAS

    if args.mock:
        print("=" * 72)
        print("  ⚠  MOCK MODE — scores are SYNTHETIC placeholders. Re-run without")
        print("     --mock (real checkpoints) for presentable results.")
        print("=" * 72)

    # --- compounds ----------------------------------------------------------
    if args.synthetic_compounds or (args.mock and not args.a549_compounds):
        cdf = make_synthetic_compounds()
        print("[compounds] synthetic preview set")
    else:
        if not args.a549_compounds:
            raise SystemExit("--a549-compounds is required (build it with prepare_a549_compounds.py), "
                             "or use --synthetic-compounds for a mock preview.")
        cdf = load_compounds(args.a549_compounds)
    print(f"[compounds] {len(cdf)} unique SMILES | mapped MoA: {mm.summarize_mapping(cdf['moa_raw'])}")

    smiles = cdf["smiles"].tolist()
    true_moa = list(cdf["moa_cluster"])

    # --- model --------------------------------------------------------------
    if args.mock:
        from mock_model import MockPhenoCompass
        pc = MockPhenoCompass(moa_list)
        pc.prime_true_moa(smiles, true_moa)
    else:
        if not args.final_model_dir:
            raise SystemExit("--final-model-dir required (or --mock).")
        pc = load_real_phenocompass(args.final_model_dir, args.anchor_modality, args.device, args.verbose)

    # --- Demo 1: cross-context MoA recovery ---------------------------------
    print("[demo1] scoring A549 structures vs U2OS MoA anchors ...")
    raw = pc.score_against_anchors(smiles, moa_list=moa_list)
    anchor_scores = core.strip_score_suffix(raw)
    anchor_scores.index = range(len(anchor_scores))  # positional, SMILES may repeat-safe
    cross = core.evaluate_cross_line(anchor_scores, true_moa, moa_list, top_k=1)
    print(f"[demo1] macro one-vs-rest AUROC = {cross.macro_auroc:.3f} | "
          f"rank-1 accuracy = {cross.accuracy:.3f} | "
          f"top-1 MoA enrichment = {cross.topk_enrichment:.1f}x")

    # --- Demo 2: customer-anchor few-shot expand (structure) ----------------
    print("[demo2] few-shot expand within A549 compound set ...")
    struct_latent = core.ensemble_average(pc.compute_struct_embeddings(smiles))
    retr = core.evaluate_fewshot_retrieval(struct_latent, true_moa, moa_list,
                                           n_anchor=args.n_anchor, k=args.k)
    print(f"[demo2] macro retrieval AUROC = {retr.macro_auroc:.3f} | "
          f"mean precision@{args.k} = {retr.mean_precision_at_k:.3f}")

    # --- structural independence (novel chemotype) --------------------------
    labelled = cross.score_df[cross.score_df["true_moa"].notna()].copy()
    lbl_idx = labelled.index.to_numpy()
    lbl_smiles = [smiles[i] for i in lbl_idx]
    true_lbl = labelled["true_moa"].to_numpy()
    score_true = np.array([labelled.loc[i, labelled.loc[i, "true_moa"]] for i in lbl_idx])
    correct = (labelled["true_moa"] == labelled["pred_moa"]).to_numpy()
    maxtani = np.full(len(lbl_idx), np.nan)
    for moa in moa_list:
        m = true_lbl == moa
        if m.sum():
            a_smi = anchor_smiles_for(pc, moa)
            maxtani[m] = core.max_tanimoto_to_refs([lbl_smiles[i] for i in np.where(m)[0]], a_smi)
    tani_ok = ~np.isnan(maxtani)
    novel = tani_ok & correct & (maxtani < 0.4)
    if tani_ok.any():
        print(f"[novelty] correctly-recovered compounds with Tanimoto<0.4 to anchors: "
              f"{int(novel.sum())}/{int((tani_ok & correct).sum())} "
              f"(median Tanimoto {np.nanmedian(maxtani):.2f})")

    # --- outputs ------------------------------------------------------------
    cdf.assign(**{f"{m}_score": anchor_scores[m] for m in moa_list},
               pred_moa=cross.score_df["pred_moa"].values).to_csv(
        os.path.join(args.out_dir, "a549_anchor_scores.csv"), index=False)
    cross.auroc_per_moa.to_csv(os.path.join(args.out_dir, "a549_crosscontext_auroc.csv"), index=False)
    cross.confusion.to_csv(os.path.join(args.out_dir, "a549_confusion.csv"))
    retr.per_moa.to_csv(os.path.join(args.out_dir, "a549_expand_retrieval.csv"), index=False)

    summary = {
        "mock": args.mock, "direction": "A (structure-based)",
        "anchor_modality": args.anchor_modality,
        "n_compounds": int(len(cdf)),
        "moa_mapped_counts": mm.summarize_mapping(cdf["moa_raw"]),
        "demo1_cross_context": {"macro_ovr_auroc": cross.macro_auroc,
                                "rank1_accuracy": cross.accuracy,
                                "top1_moa_recovery_enrichment": cross.topk_enrichment},
        "demo2_fewshot_expand": {"n_anchor": args.n_anchor, "k": args.k,
                                 "macro_auroc": retr.macro_auroc,
                                 "mean_precision_at_k": retr.mean_precision_at_k},
        "structural_independence": {
            "n_with_tanimoto": int(tani_ok.sum()),
            "n_correct_novel_lt0.4": int(novel.sum()) if tani_ok.any() else None,
            "median_max_tanimoto": float(np.nanmedian(maxtani)) if tani_ok.any() else None,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- figures ------------------------------------------------------------
    plots.plot_auroc_bar(cross.auroc_per_moa, "ovr_auroc",
                         os.path.join(fig_dir, "fig1_crosscontext_auroc"),
                         "A549 cross-context MoA recovery (one-vs-rest AUROC)",
                         macro=cross.macro_auroc, mock=args.mock)
    plots.plot_score_heatmap(cross.score_df, moa_list,
                             os.path.join(fig_dir, "fig2_anchor_score_heatmap"),
                             "A549 structure → MoA anchor score (diagonal = correct)",
                             mock=args.mock)
    plots.plot_auroc_bar(retr.per_moa, "auroc",
                         os.path.join(fig_dir, "fig3_expand_retrieval_auroc"),
                         f"Customer-anchor Expand ({args.n_anchor}-shot) AUROC",
                         macro=retr.macro_auroc, mock=args.mock)
    if tani_ok.any():
        plots.plot_score_vs_tanimoto(maxtani, score_true, correct,
                                     os.path.join(fig_dir, "fig4_structural_independence"),
                                     "Independent chemotypes: high MoA score at low Tanimoto",
                                     mock=args.mock)

    print(f"\nDone. Metrics + figures in {args.out_dir}")
    if args.mock:
        print("Reminder: MOCK outputs are synthetic placeholders.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(f"\nSetup error: {e}", file=sys.stderr)
        raise SystemExit(2)

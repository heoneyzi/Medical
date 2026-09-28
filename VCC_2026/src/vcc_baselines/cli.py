"""CLI: python -m vcc_baselines <cmd>

  make-synth  generate synthetic raw-count VCC-shaped data (offline)
  predict     build a raw-count prediction (all contexts, no controls) + optional vcc prep
  prep        wrap `vcc prep` -> .vcc
  sample      wrap `vcc sample` -> a random valid dummy .vcc (A1 pipeline check)
  evaluate    score a prediction vs truth (shadow-CV) via cell-eval / local
  ladder      3-rung ladder across shadow-CV contexts
  coverage    per coverage-group reach of the GWPS approach
  fm-cmd      print the external STATE/STACK command
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from . import io, ladder as L, predict as P, submit as S
from .config import Config
from .methods import METHODS, build_method


def _cfg(path):
    return Config.load(path).resolve(Path(path).parent)


def cmd_make_synth(a):
    from . import synthetic
    info = synthetic.make(a.out, n_genes=a.n_genes, n_contexts=a.n_contexts,
                          n_targets=a.n_targets, cells_per_pert=a.cells_per_pert, seed=a.seed)
    print(f"[make-synth] {a.out}: genes={info['n_genes']} contexts={info['contexts']} "
          f"targets={len(info['targets'])} (GWPS-measured {len(info['measured'])}), "
          f"cells/pert={info['cells_per_pert']}")


def cmd_predict(a):
    cfg = _cfg(a.config)
    if a.method:
        cfg.predict.method = a.method
    if a.fm_pred:
        cfg._fm_pred = a.fm_pred
    genes = io.read_gene_list(cfg.data.gene_list)
    targets = io.read_targets(cfg.data.targets)
    lib = L.get_library(cfg, genes)
    method = build_method(cfg, lib, genes)
    contexts = a.contexts or io.list_contexts(cfg.data.contexts_dir, cfg.data.control_h5ad)
    if not contexts:
        raise SystemExit(f"no contexts under {cfg.data.contexts_dir}")
    ctx_pbs = L._context_pbs(cfg, contexts, genes)
    embedder = P.make_embedder(cfg, lib, ctx_pbs)

    preds, covs = [], []
    for c in contexts:
        controls = io.load_controls(cfg.data.contexts_dir, c, cfg.data.control_h5ad)
        pr, cov = P.build_prediction(cfg, controls, genes, targets, method, embedder, c,
                                     lib=lib, include_controls=a.include_controls)
        preds.append(pr); cov["context"] = c; covs.append(cov)
        print(f"[predict] {c}: {pr.n_obs} cells ({cfg.predict.cells_per_pert}/pert x {len(targets)}), "
              f"covered {int(cov['covered'].sum())}/{len(cov)}")
    combined = ad.concat(preds, join="outer"); combined.var_names = list(map(str, genes))
    combined = P.cap_cells(combined, cfg.submit.max_cell_dim, cfg.predict.seed)
    outdir = Path(cfg.output_dir) / cfg.predict.method; outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "prediction.h5ad"; io.save_prediction(combined, pred_path)
    pd.concat(covs).to_csv(outdir / "coverage.csv", index=False)
    print(f"[predict] {combined.n_obs} raw-count cells -> {pred_path}")
    if a.prep or a.dry_run:
        gcsv = S.write_gene_csv(genes, outdir / "gene_names.csv")
        S.prep_submission(cfg, pred_path, gcsv, outdir / "submission.prep.vcc",
                          perts=cfg.data.perts, dry_run=a.dry_run)
        if not a.dry_run:
            print(f"[predict] submission -> {outdir/'submission.prep.vcc'}")


def cmd_prep(a):
    cfg = _cfg(a.config)
    genes = io.read_gene_list(cfg.data.gene_list)
    gcsv = S.write_gene_csv(genes, Path(a.pred).with_name("gene_names.csv"))
    out = a.out or str(Path(a.pred).with_suffix(".prep.vcc"))
    S.prep_submission(cfg, a.pred, gcsv, out, perts=cfg.data.perts, dry_run=a.dry_run)


def cmd_sample(a):
    """A1 baseline: official random valid dummy submission."""
    cfg = _cfg(a.config)
    if not shutil.which("vcc"):
        raise SystemExit("vcc CLI not found (`pip install vcc-cli`).")
    genes = io.read_gene_list(cfg.data.gene_list)
    gcsv = S.write_gene_csv(genes, Path(cfg.output_dir) / "gene_names.csv")
    out = a.out or str(Path(cfg.output_dir) / "A1_sample.vcc")
    cmd = [S.executable("vcc") or "vcc", "sample", "-g", gcsv, "-p", cfg.data.perts, "-o", out, "--full", "--force"]
    print("[sample] $", " ".join(cmd)); subprocess.run(cmd, check=True)


def cmd_evaluate(a):
    cfg = _cfg(a.config)
    from . import evaluate
    pred, truth = ad.read_h5ad(a.pred), ad.read_h5ad(a.truth)
    if a.engine == "cell-eval":
        od = Path(cfg.output_dir) / "evaluate"
        io.save_prediction(pred, od / "pred.h5ad"); io.save_prediction(truth, od / "truth.h5ad")
        res = evaluate.evaluate_cell_eval(od / "pred.h5ad", od / "truth.h5ad", cfg, od)
        fp = evaluate.fingerprint(res)
        print("[fingerprint]", {k: round(v, 4) for k, v in fp.items()})
    else:
        res = evaluate.evaluate_local(pred, truth, cfg)
    for k, v in res.items():
        print(f"  {k}: {v}")


def cmd_ladder(a):
    cfg = _cfg(a.config)
    methods = a.methods.split(",") if a.methods else None
    df = L.run_ladder(cfg, methods, engine=a.engine)
    out = Path(cfg.output_dir) / "ladder_results.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print("\n=== 3-RUNG LADDER (mean over contexts) ===\n")
    print(L.to_markdown(df))
    print(f"\n[ladder] full table -> {out}")


def cmd_coverage(a):
    cfg = _cfg(a.config)
    df = L.coverage_report(cfg, a.method)
    agg = df.groupby("group")[["pdisc_norm_rank", "n_targets"]].mean()
    print("\n=== COVERAGE STRATIFICATION (local pdisc; lower=better) ===\n")
    try:
        print(agg.round(4).to_markdown())
    except ImportError:
        # pandas exposes to_markdown even when its optional tabulate dependency
        # is absent. Keep diagnostics usable in a minimal installation.
        print(agg.round(4).to_string())
    out = Path(cfg.output_dir) / "coverage_report.csv"; df.to_csv(out, index=False)
    print(f"\n[coverage] -> {out}")


def cmd_fm_cmd(a):
    from .adapters import command_template
    print(command_template(a.method))


def cmd_compact_shadow(a):
    from . import shadow
    shadow.compact_source(a.input, a.out, shadow._parse_where(a.where), a.matrix)


def cmd_prepare_shadow(a):
    from . import shadow
    if a.input:
        if a.source or a.holdout:
            raise SystemExit("use either --input/--context-col/--holdout-context or --source/--holdout")
        refs, holdout, filters = shadow.specs_from_single_file(
            a.input, a.context_col, a.holdout_context, a.where)
    else:
        if not a.source or not a.holdout:
            raise SystemExit("paired mode requires --source NAME=PATH and --holdout NAME=PATH")
        refs = [shadow.parse_named_path(v) for v in a.source]
        holdout = shadow.parse_named_path(a.holdout)
        filters = shadow._parse_where(a.where)
    shadow.prepare_shadow(
        sources=refs, holdout=holdout, out_dir=a.out, filters=filters,
        pert_col=a.pert_col, control_label=a.control_label, single_col=a.single_col,
        matrix=a.matrix, min_genes=a.min_genes, min_counts=a.min_counts,
        max_percent_mito=a.max_percent_mito, min_cells=a.min_cells,
        max_targets=a.max_targets, max_genes=a.max_genes, max_controls=a.max_controls,
        max_cells_per_pert=a.max_cells_per_pert, cells_per_pert=a.cells_per_pert, seed=a.seed,
        target_gene_map_path=a.target_gene_map)


def cmd_prepare_zero_shot(a):
    from . import shadow
    shadow.prepare_zero_shot(
        input_path=a.input, context_name=a.context_name, out_dir=a.out,
        filters=shadow._parse_where(a.where), pert_col=a.pert_col,
        control_label=a.control_label, single_col=a.single_col, matrix=a.matrix,
        min_genes=a.min_genes, min_counts=a.min_counts,
        max_percent_mito=a.max_percent_mito, max_genes=a.max_genes,
        max_controls=a.max_controls, max_cells_per_pert=a.max_cells_per_pert,
        cells_per_pert=a.cells_per_pert, seed=a.seed)


def cmd_score_zero_shot(a):
    from . import shadow
    frame = shadow.score_zero_shot(
        a.config, a.pred, a.truth, a.out, a.engine, a.profile, a.method_name,
        a.target_gene_map, a.allow_cpu_scorer)
    print(frame.to_string(index=False))


def cmd_shadow_benchmark(a):
    from . import shadow
    methods = a.methods.split(",") if a.methods else None
    out = Path(a.out or (Path(a.config).parent / "shadow_benchmark.csv"))
    df = shadow.benchmark(
        a.config, a.embedding, a.fm_pred, methods, a.engine, a.profile,
        a.ceiling, checkpoint_path=str(out), resume=a.resume,
        six_only=a.six_only, target_gene_map=a.target_gene_map,
        allow_cpu_scorer=a.allow_cpu_scorer,
        artifact_root=a.artifact_root)
    out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    print(df.to_string(index=False)); print(f"[shadow] benchmark -> {out}")


def cmd_pool_embeddings(a):
    from . import shadow
    shadow.pool_embeddings(a.input, a.obsm_key, a.out)


def build_parser():
    p = argparse.ArgumentParser("vcc_baselines")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synth"); s.add_argument("--out", default="data_synth")
    s.add_argument("--n-genes", type=int, default=400); s.add_argument("--n-contexts", type=int, default=3)
    s.add_argument("--n-targets", type=int, default=40); s.add_argument("--cells-per-pert", type=int, default=50)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(func=cmd_make_synth)

    s = sub.add_parser("predict"); s.add_argument("--config", required=True)
    s.add_argument("--method", choices=METHODS); s.add_argument("--contexts", nargs="*")
    s.add_argument("--fm-pred"); s.add_argument("--prep", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--include-controls", action="store_true", help="include allowed controls for local blind scoring")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("prep"); s.add_argument("--config", required=True)
    s.add_argument("--pred", required=True); s.add_argument("--out"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_prep)

    s = sub.add_parser("sample"); s.add_argument("--config", required=True); s.add_argument("--out")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("evaluate"); s.add_argument("--config", required=True)
    s.add_argument("--pred", required=True); s.add_argument("--truth", required=True)
    s.add_argument("--engine", choices=["local", "cell-eval"], default="local"); s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("ladder"); s.add_argument("--config", required=True)
    s.add_argument("--methods"); s.add_argument("--engine", choices=["local", "cell-eval", "cell-eval2"], default="local")
    s.set_defaults(func=cmd_ladder)

    s = sub.add_parser("coverage"); s.add_argument("--config", required=True)
    s.add_argument("--method", default="gwps_weighted"); s.set_defaults(func=cmd_coverage)

    s = sub.add_parser("fm-cmd"); s.add_argument("--method", choices=["state", "stack"], required=True)
    s.set_defaults(func=cmd_fm_cmd)

    s = sub.add_parser("compact-shadow", help="filter a huge H5AD into a counts-only shadow source")
    s.add_argument("--input", required=True); s.add_argument("--out", required=True)
    s.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE")
    s.add_argument("--matrix", choices=["auto", "X", "counts"], default="counts")
    s.set_defaults(func=cmd_compact_shadow)

    s = sub.add_parser("prepare-shadow", help="build a leakage-resistant real-data LOCO benchmark")
    s.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    s.add_argument("--holdout", metavar="NAME=PATH")
    s.add_argument("--input", help="one multi-context H5AD (e.g. Jiang24)")
    s.add_argument("--context-col"); s.add_argument("--holdout-context")
    s.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE")
    s.add_argument("--out", required=True); s.add_argument("--pert-col", default="perturbation")
    s.add_argument("--control-label", default="control"); s.add_argument("--single-col", default="nperts")
    s.add_argument("--matrix", choices=["auto", "X", "counts"], default="auto")
    s.add_argument("--min-genes", type=int, default=200); s.add_argument("--min-counts", type=int, default=500)
    s.add_argument("--max-percent-mito", type=float, default=20.0)
    s.add_argument("--min-cells", type=int, default=30); s.add_argument("--max-targets", type=int, default=100)
    s.add_argument("--max-genes", type=int, default=4000); s.add_argument("--max-controls", type=int, default=500)
    s.add_argument("--max-cells-per-pert", type=int, default=100)
    s.add_argument("--cells-per-pert", type=int, default=50); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--target-gene-map", help="CSV/JSON map used only to retain target genes for official metric exclusion")
    s.set_defaults(func=cmd_prepare_shadow)

    s = sub.add_parser("prepare-zero-shot", help="build a truth-sealed benchmark with no response references")
    s.add_argument("--input", required=True); s.add_argument("--context-name", required=True)
    s.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE")
    s.add_argument("--out", required=True); s.add_argument("--pert-col", default="perturbation")
    s.add_argument("--control-label", default="control"); s.add_argument("--single-col", default="")
    s.add_argument("--matrix", choices=["auto", "X", "counts"], default="auto")
    s.add_argument("--min-genes", type=int, default=200); s.add_argument("--min-counts", type=int, default=500)
    s.add_argument("--max-percent-mito", type=float, default=20.0)
    s.add_argument("--max-genes", type=int, default=4000); s.add_argument("--max-controls", type=int, default=500)
    s.add_argument("--max-cells-per-pert", type=int, default=100)
    s.add_argument("--cells-per-pert", type=int, default=50); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_prepare_zero_shot)

    s = sub.add_parser("score-zero-shot", help="evaluate an existing prediction against sealed truth")
    s.add_argument("--config", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--truth", required=True); s.add_argument("--out", required=True)
    s.add_argument("--engine", choices=["local", "cell-eval", "cell-eval2"], default="cell-eval2")
    s.add_argument("--profile", choices=["vcc", "minimal", "full", "pds"], default="full")
    s.add_argument("--method-name", help="label for an externally generated frozen prediction")
    s.add_argument("--target-gene-map", help="evaluator-only CSV/JSON label-to-gene map")
    s.add_argument("--allow-cpu-scorer", action="store_true", help="allow CPU DE instead of requiring gpudge/CUDA")
    s.set_defaults(func=cmd_score_zero_shot)

    s = sub.add_parser("shadow-benchmark", help="compare raw/PCA/frozen representations")
    s.add_argument("--config", required=True); s.add_argument("--embedding", action="append", default=[], metavar="MODEL=NPZ")
    s.add_argument("--fm-pred", action="append", default=[], metavar="state|stack=H5AD")
    s.add_argument("--methods"); s.add_argument("--engine", choices=["local", "cell-eval", "cell-eval2"], default="local")
    s.add_argument("--profile", choices=["vcc", "minimal", "full", "pds"], default="vcc")
    s.add_argument("--ceiling", action="store_true")
    s.add_argument("--resume", action="store_true", help="skip completed rows already present in --out")
    s.add_argument("--six-only", action="store_true", help="with profile=full, compute only the six fingerprint metrics")
    s.add_argument("--target-gene-map", help="evaluator-only CSV/JSON label-to-gene map")
    s.add_argument("--allow-cpu-scorer", action="store_true", help="allow CPU DE instead of requiring gpudge/CUDA")
    s.add_argument("--artifact-root", help="separate detailed outputs by representation and share one verified scale")
    s.add_argument("--out"); s.set_defaults(func=cmd_shadow_benchmark)

    s = sub.add_parser("pool-embeddings", help="pool frozen model cell embeddings by context")
    s.add_argument("--input", action="append", required=True, metavar="NAME=H5AD")
    s.add_argument("--obsm-key", required=True); s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_pool_embeddings)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

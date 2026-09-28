"""Command line for the eight steps of the Experiment 1 protocol.

    python -m exp1 selftest                     # no GPU, no data
    python -m exp1 step0-motifs acquire         # all obtainable motifs, from your data
    python -m exp1 step1-blockmap               # operator map + T2 (config only)
    python -m exp1 step2-onset --model 7b       # norm-ratio onset + rotation
    python -m exp1 step3-fidelity --model 7b    # dtype ledger + reconstruction gate
    python -m exp1 step4-panel --chrom chr22    # 400-window panel
    python -m exp1 step5-uref --model 7b --chrom chr22
    python -m exp1 step5-extract --model 7b --chrom chr22
    python -m exp1 step6-prereg
    python -m exp1 step7-analysis --model 7b --chrom chr22
    python -m exp1 step8-stage2 --model 7b --chrom chr22

Steps 1-3 are cheap and must pass before anything else runs; step 6 must be
frozen before step 7 touches a contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import analysis as A
from . import blockmap as B
from . import prereg as P
from . import stats as S
from .config import (ADJUST_COVARIATES, CONTEXTS, ONSET_THRESHOLD, ROOT, cache_path,
                     load_models, load_panel, result_path)
from .io import load_table, save_json, save_table


# --------------------------------------------------------------------------

def _model(args):
    spec = load_models()[args.model]
    from .modelio import Evo2Runner

    return spec, Evo2Runner(spec, device=args.device,
                            shuffle_seed=args.shuffle_seed if args.shuffle_seed >= 0 else None)


def _tracks(chrom: str):
    from .labels import load_chrom_tracks

    ps = load_panel()
    return load_chrom_tracks(chrom, ps.fasta, ps.gtf, ps.phylop_bw)


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def step1_blockmap(args) -> int:
    """Operator map for every configured model, plus table T2 if onsets are known."""
    models = load_models()
    rows = []
    records = []
    for key, spec in models.items():
        if args.models and key not in args.models.split(","):
            continue
        bm = B.blockmap_from_period(spec.n_blocks, key)
        cfg_path = Path(ROOT) / "configs" / f"{key}_model_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            bm = B.blockmap_from_config(cfg, spec.n_blocks, key)
        chk = B.check_hcs_prefix(bm)
        print(f"[{key}] blocks={bm.n_blocks} source={bm.source} hcs_prefix_match={chk['match']}")
        if not chk["match"]:
            print(f"    got  {chk['got']}\n    want {chk['want']}")
        rows.extend(bm.as_rows())
        onset_known = {"evo2_7b": 28, "evo2_40b": 21}.get(spec.hf_name)
        rot_known = {"evo2_7b": 30, "evo2_40b": 23, "evo2_1b_base": 23}.get(spec.hf_name)
        records.append(dict(model=key, n_blocks=bm.n_blocks, blockmap=bm,
                            onset=onset_known, rotation=rot_known))
    save_table(pd.DataFrame(rows), result_path("00_blockmap", "blockmap"))
    t2 = B.operator_coordinate_table([r for r in records if r["onset"] is not None])
    if not t2.empty:
        save_table(t2, result_path("00_blockmap", "T2_operator_coordinates"))
        verdict = B.operator_hypothesis_verdict(t2)
        save_json(verdict, result_path("00_blockmap", "T2_verdict.json"))
        print(t2.to_string(index=False))
        print(json.dumps(verdict, indent=2))
    return 0


def step2_onset(args) -> int:
    """Measure the norm-ratio onset and the cosine rotation on a handful of windows."""
    import torch

    from .modelio import CaptureConfig
    from .variants import to_str

    spec, runner = _model(args)
    tracks = _tracks(args.chrom)
    panel = load_table(cache_path("panel", f"{args.chrom}_panel"))

    runner.attach(CaptureConfig(block_outputs=True, keep_device=True))
    n_blocks = runner.anatomy.n_blocks
    sum_norm = np.zeros(n_blocks + 1)
    sum_cos = np.zeros(n_blocks + 1)
    count = 0
    for _, row in panel.head(args.n_windows).iterrows():
        wbp = int(row.get("window_bp", 6000) or 6000)
        seq = to_str(tracks.sequence[int(row["start"]): int(row["start"]) + wbp])
        runner.taps.clear()
        runner.forward(runner.tokenize(seq))
        hn = runner.taps["hnorm"][0].to(torch.float64)
        hn_u = hn / torch.clamp(torch.linalg.vector_norm(hn, dim=-1, keepdim=True), min=1e-30)
        for i in range(n_blocks):
            h = runner.taps[f"h{i}"][0].to(torch.float64)
            sum_norm[i] += float(torch.linalg.vector_norm(h, dim=-1).mean())
            hu = h / torch.clamp(torch.linalg.vector_norm(h, dim=-1, keepdim=True), min=1e-30)
            sum_cos[i] += float((hu * hn_u).sum(-1).mean())
        sum_norm[n_blocks] += float(torch.linalg.vector_norm(hn, dim=-1).mean())
        sum_cos[n_blocks] += 1.0
        count += 1
    runner.detach()

    mean_norm = (sum_norm / max(count, 1)).tolist()
    mean_cos = (sum_cos / max(count, 1)).tolist()
    onset = B.detect_onset(mean_norm[:n_blocks], ONSET_THRESHOLD)
    rot = B.detect_rotation(mean_cos[:n_blocks])
    stab = B.onset_threshold_stability(mean_norm[:n_blocks])
    bm = B.blockmap_from_period(n_blocks, args.model)

    out = dict(model=args.model, chrom=args.chrom, n_windows=count,
               mean_norm=mean_norm, mean_cos=mean_cos,
               onset=onset, rotation=rot, threshold_stability=stab,
               onset_op=bm.type_of(onset["onset"]) if onset["onset"] is not None else None,
               rotation_op=bm.type_of(rot["rotation"]),
               onset_mod7=(onset["onset"] % 7) if onset["onset"] is not None else None,
               rotation_mod7=rot["rotation"] % 7)
    save_json(out, result_path("01_onset", f"{args.model}_{args.chrom}_onset.json"))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("mean_norm", "mean_cos", "onset", "rotation")}, indent=2))
    print(f"onset={onset['onset']} ({out['onset_op']}, mod7={out['onset_mod7']})  "
          f"rotation={rot['rotation']} ({out['rotation_op']}, mod7={out['rotation_mod7']})  "
          f"interior peak: block {rot['interior_peak_block']} = {rot['interior_peak_value']:.3f}")
    return 0


def step3_fidelity(args) -> int:
    """Reconstruction gate + dtype ledger.  Nothing downstream runs until this passes."""
    from .modelio import CaptureConfig, reconstruction_error
    from .variants import to_str

    spec, runner = _model(args)
    tracks = _tracks(args.chrom)
    panel = load_table(cache_path("panel", f"{args.chrom}_panel"))
    blocks = list(range(24, runner.anatomy.n_blocks))

    runner.attach(CaptureConfig(block_outputs=True, block_inputs_for=blocks,
                               branches_for=blocks, keep_device=True))
    rows = []
    for _, row in panel.head(args.n_windows).iterrows():
        wbp = int(row.get("window_bp", 6000) or 6000)
        seq = to_str(tracks.sequence[int(row["start"]): int(row["start"]) + wbp])
        runner.taps.clear()
        runner.forward(runner.tokenize(seq))
        rows.extend(reconstruction_error(runner, blocks))
    runner.detach()

    df = pd.DataFrame(rows)
    agg = df.groupby("block")[["rel_err_mean", "rel_err_max"]].max().reset_index()
    save_table(agg, result_path("02_fidelity", f"{args.model}_reconstruction"))
    save_json(runner.dtype_ledger(), result_path("02_fidelity", f"{args.model}_dtype_ledger.json"))

    worst = float(agg["rel_err_max"].max())
    print(agg.to_string(index=False))
    print(f"\nworst relative reconstruction error: {worst:.3e}")
    if not np.isfinite(worst) or worst > args.tol:
        print("\nGATE FAILED. The discovered branch modules are probably not the ones added "
              "to the residual stream. Module tree:\n")
        print(runner.module_tree())
        print("\nPin `mixer_attr` / `mlp_attr` in configs/models.yaml and re-run.")
        return 2
    print("GATE PASSED")
    return 0


def step4_panel(args) -> int:
    from .panels import build_panel, panel_summary

    ps = load_panel()
    if args.n_windows is not None:
        ps.n_windows = args.n_windows
    print(f"[panel] {args.chrom}: asking for {ps.n_windows} windows of "
          f"{ps.window_bp} bp (scored {ps.scored_bp})")
    tracks = _tracks(args.chrom)
    legacy = None
    if ps.legacy_windows and Path(ps.legacy_windows).exists():
        legacy = pd.read_csv(ps.legacy_windows)
        legacy = legacy.loc[legacy["chrom"] == args.chrom, "start"].astype(int).tolist()
        print(f"[panel] carrying {len(legacy)} legacy windows so the published subset is reproducible")
    panel = build_panel(tracks, ps, legacy_starts=legacy)
    save_table(panel, cache_path("panel", f"{args.chrom}_panel"))
    summary = panel_summary(panel, tracks, ps)
    save_json(summary, result_path("03_panel", f"{args.chrom}_summary.json"))
    print(json.dumps(summary, indent=2))
    return 0


def step5_uref(args) -> int:
    from .extract import estimate_u

    spec, runner = _model(args)
    tracks = _tracks(args.chrom)
    panel = load_table(cache_path("panel", f"{args.chrom}_panel"))
    u = estimate_u(runner, tracks, panel, n_windows=args.n_windows)
    np.save(cache_path("uref", f"{args.model}_{args.chrom}_u.npy"), u)
    print(f"u estimated on {args.n_windows} windows, dim={u.shape[0]}, ||u||={np.linalg.norm(u):.6f}")
    return 0


def step5_extract(args) -> int:
    from .extract import ExtractConfig, run_extraction

    spec, runner = _model(args)
    tracks = _tracks(args.chrom)
    panel = load_table(cache_path("panel", f"{args.chrom}_panel"))
    u_path = cache_path("uref", f"{args.model}_{args.chrom}_u.npy")
    if not u_path.exists():
        print(f"missing {u_path}; run step5-uref first")
        return 2
    u = np.load(u_path)

    cfg = ExtractConfig(
        variants=args.variants.split(","),
        max_windows=args.max_windows,
        offload_cpu=args.offload_cpu,
        raw_state_per_context=args.raw_per_context,
        perturb=args.perturb,
        perturb_families=args.perturb_families.split(","),
        perturb_max_sites=args.perturb_sites,
        perturb_every_n_windows=args.perturb_every,
        motif_manifest=(args.motifs or (str(_motif_manifest_path())
                                        if _motif_manifest_path().exists() else None)),
    )
    if cfg.motif_manifest:
        print(f"[extract] motif models: {cfg.motif_manifest}")
    else:
        print("[extract] no motif manifest -- measurement columns only, no model scores. "
              "Run `python -m exp1 step0-motifs build` first if you want motif strata.")
    tag = args.tag or ("shuffled" if args.shuffle_seed >= 0 else "main")
    path = run_extraction(runner, tracks, panel, u, cfg, model_key=args.model, tag=tag)
    print(f"wrote {path}")
    return 0


def step6_prereg(args) -> int:
    res = P.freeze(Path(ROOT), force=args.force)
    print(json.dumps(res, indent=2))
    print(json.dumps(P.verify(Path(ROOT)), indent=2))
    return 0


def step7_analysis(args) -> int:
    ver = P.verify(Path(ROOT))
    if not ver["ok"] and not args.allow_unfrozen:
        print("pre-registration not frozen (step6). Re-run step6 or pass --allow-unfrozen "
              "for an exploratory pass that must not be reported as confirmatory.")
        return 2

    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    onset, rotation = args.onset, args.rotation

    prof = A.f1_profile(df)
    branch = A.f1_branch_decomposition(df)
    save_table(prof, result_path("07_f1", f"{args.model}_{args.chrom}_profile"))
    save_table(branch, result_path("07_f1", f"{args.model}_{args.chrom}_branch"))
    # No classification. The exact decomposition, with intervals, and each
    # pre-registered point prediction tested against its own interval.
    decomp = A.f1_decomposition(df, onset)
    hyp = A.f1_hypothesis_check(decomp)
    rot = A.f1_rotation_share(df, rotation)
    save_json(dict(onset_decomposition=decomp, rotation_share=rot),
              result_path("07_f1", f"{args.model}_{args.chrom}_decomposition.json"))
    if not hyp.empty:
        save_table(hyp, result_path("07_f1", f"{args.model}_{args.chrom}_hypotheses"))
    A.plot_f1(prof, result_path("07_f1", f"{args.model}_{args.chrom}_F1.png"),
              onset=onset, rotation=rotation, title=f"F1 {args.model} {args.chrom}")

    if decomp.get("available"):
        print("F1 onset decomposition (exact identity, window-bootstrap CIs):")
        for k in ("d_log_cos", "d_log_numerator", "d_log_denominator", "d_p",
                  "branch_mixer", "branch_mlp"):
            t = decomp.get(k)
            if isinstance(t, dict):
                print(f"    {k:20s} {t['mean']:+.5g}  [{t['ci_lo']:+.5g}, {t['ci_hi']:+.5g}]"
                      f"  n_win={t['n_windows']}")
        print(f"    identity check  log: {decomp['identity_log_error']:.2e}"
              + (f"   branch: {decomp['identity_branch_rel_error']:.2e}"
                 if "identity_branch_rel_error" in decomp else ""))
        print("    " + str(decomp["reading"]))
    if not hyp.empty:
        print("\npre-registered point predictions:")
        print(hyp[["hypothesis", "quantity", "mean", "ci_lo", "ci_hi",
                   "predicted", "consistent"]].to_string(index=False))
    print("\nrotation share:", json.dumps(rot, indent=2, default=str))

    ch = A.f2_channel_profile(df)
    save_table(ch, result_path("08_f2", f"{args.model}_{args.chrom}_channels"))
    A.plot_f2(ch, result_path("08_f2", f"{args.model}_{args.chrom}_F2.png"),
              title=f"F2 {args.model} {args.chrom}")

    pool = [f"{i:02d}" for i in range(0, rotation)]
    for pref, name in (("a", "a_pool"), ("au", "au_pool"), ("av", "av_pool")):
        df = A.pool_max(df, pref, pool, name)

    t1 = A.t1_context_table(df, {"a": "a_pool", "a_u": "au_pool", "a_v": "av_pool"},
                            covariates=[c for c in ADJUST_COVARIATES if c in df.columns])
    save_table(t1, result_path("09_t1", f"{args.model}_{args.chrom}_T1"))
    print(t1.to_string(index=False))

    yard = {k: A.complexity_yardstick(df, v)
            for k, v in dict(a="a_pool", a_u="au_pool", a_v="av_pool").items()}
    save_json(yard, result_path("09_t1", f"{args.model}_{args.chrom}_yardstick.json"))

    od = S.overdispersion(df.assign(_ind=(df["a_pool"] > df["a_pool"].median()).astype(int)), "_ind")
    save_json(od, result_path("09_t1", f"{args.model}_{args.chrom}_overdispersion.json"))
    print("overdispersion:", od)
    return 0


def step8_stage2(args) -> int:
    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    onset, rotation = args.onset, args.rotation

    s2 = A.stage2_table(df, onset, rotation)
    save_table(s2, result_path("10_stage2", f"{args.model}_{args.chrom}_stage2"))
    print(s2.to_string(index=False))

    feats = A.stage2_position_features(df, onset, rotation)
    pool = [f"{i:02d}" for i in range(0, rotation)]
    for pref, name in (("av", "av_pool"), ("au", "au_pool")):
        feats = A.pool_max(feats, pref, pool, name)
    save_table(feats[[c for c in feats.columns if not c.startswith(("p_", "q_", "logit"))]],
               cache_path("stage2", f"{args.model}_{args.chrom}_features"))

    outcome = "r_D" if "r_D" in feats.columns else None
    if outcome is None:
        print("no stage-2 outcome available (need kl_ columns)")
        return 0
    audit = S.covariate_audit(feats, ADJUST_COVARIATES)
    print("[covariates] " + json.dumps(audit["finite_fraction"]))
    if audit["all_missing"]:
        print(f"[covariates] dropped (never finite): {audit['all_missing']} -- "
              "adjusted estimates are computed WITHOUT them; say so in the paper")
    print(f"[covariates] positions retained jointly: {audit['joint_retained_fraction']:.3f}")
    save_json(audit, result_path("11_bridge", f"{args.model}_{args.chrom}_covariates.json"))

    res = {}
    for pred in ("av_pool", "au_pool", "entropy_final"):
        if pred in feats.columns:
            res[pred] = A.bridge_test(feats, predictor=pred, outcome=outcome)
    save_json(res, result_path("11_bridge", f"{args.model}_{args.chrom}_bridge.json"))
    A.plot_bridge(feats, "av_pool", outcome,
                  result_path("11_bridge", f"{args.model}_{args.chrom}_F4.png"))
    for k, v in res.items():
        print(k, json.dumps(v["fit"], indent=2))
    return 0



def step7b_regimes(args) -> int:
    """Per-regime features, per-regime context effects, and the R1-vs-R3 contrast."""
    from . import regimes as R

    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    df = df[df["variant"] == args.variant]
    reg = R.Regimes(onset=args.onset, rotation=args.rotation,
                    n_blocks=int(args.n_blocks))
    save_json(reg.as_dict(), result_path("12_regimes", f"{args.model}_{args.chrom}_regimes.json"))

    feats = R.regime_features(df, reg)
    keep = [c for c in feats.columns
            if any(k in c for k in ("_r1_", "_r2_", "_r3_"))]
    save_table(feats[["window_id", "pos", "context", "entropy_final", "gc", "repeat",
                      "phylop", *[c for c in ("motif_best", "splice_canonical",
                                              "motif_best_score") if c in feats.columns],
                      *keep]],
               cache_path("regimes", f"{args.model}_{args.chrom}_features"))

    contexts = [c for c in feats["context"].unique() if c != "intron"]
    table = R.regime_effect_table(
        feats, features=keep, contexts=contexts,
        covariates=[c for c in ADJUST_COVARIATES if c in feats.columns])
    save_table(table, result_path("12_regimes", f"{args.model}_{args.chrom}_regime_effects"))

    contrasts = []
    for prefix in ("av", "au", "a"):
        a, b = f"{prefix}_r1_end", f"{prefix}_r3_end"
        if a in feats.columns and b in feats.columns:
            for ctx in contexts:
                contrasts.append(R.regime_contrast(feats, a, b, target=ctx))
    if contrasts:
        save_table(pd.DataFrame(contrasts),
                   result_path("12_regimes", f"{args.model}_{args.chrom}_regime_contrast"))
        print(pd.DataFrame(contrasts).to_string(index=False))

    transfers = {}
    for r1f, r3f in (("av_r1_end", "kl_r3_end"), ("av_r1_peak", "sctr_r3_end"),
                     ("au_r1_end", "kl_r3_end")):
        if r1f in feats.columns and r3f in feats.columns:
            transfers[f"{r1f}->{r3f}"] = R.regime_transfer(feats, r1f, r3f)
    save_json(transfers, result_path("12_regimes", f"{args.model}_{args.chrom}_transfer.json"))
    print(json.dumps(transfers, indent=2))

    for prefix in ("av", "au"):
        prof = R.regime_profile_by_group(feats, prefix, reg)
        if prof.empty:
            continue
        save_table(prof, result_path("12_regimes", f"{args.model}_{args.chrom}_{prefix}_profile"))
        R.plot_regime_profiles(prof, result_path(
            "12_regimes", f"{args.model}_{args.chrom}_{prefix}_regimes.png"), reg,
            title=f"{prefix} by context across R1/R2/R3")
    return 0


def step7c_motifs(args) -> int:
    """Motif-stratified contrasts and paired perturbation effects."""
    from . import analysis as AA
    from . import motifs as MO
    from . import perturb as PB

    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    base = df[df["variant"] == "real"].copy()

    man = _motif_manifest_path(getattr(args, "motifs", "") or "")
    if man.exists():
        ms = MO.MotifSet.load(str(man))
        print(f"[motifs] strata from {len(ms)} model(s): " + ", ".join(
            f"{n} (n={p.provenance.n_sites or int(p.n_observations)}, {p.provenance.source})"
            for n, p in ms.pwms.items()))
    elif not any(c.startswith("motif_") for c in base.columns):
        print("[motifs] this extraction carries no model scores. Canonical/non-canonical "
              "and the other measurement strata still run; motif-strength strata do not.")

    pool = [f"{i:02d}" for i in range(0, args.rotation)]
    for pref, name in (("a", "a_pool"), ("au", "au_pool"), ("av", "av_pool")):
        base = AA.pool_max(base, pref, pool, name)

    out_rows = []
    if "splice_canonical" in base.columns:
        sp = base[base["context"].isin(["splice_donor", "splice_acceptor"])].copy()
        sp["context"] = np.where(sp["splice_canonical"] > 0.5, "canonical", "noncanonical")
        for value in ("a_pool", "au_pool", "av_pool"):
            eff = S.window_effect(sp, value=value, target="noncanonical", baseline="canonical")
            out_rows.append(dict(kind="canonical_vs_noncanonical", **eff.as_row()))

    if "motif_best_score" in base.columns:
        base["strength"] = MO.strength_quartiles(base["motif_best_score"].to_numpy())
        for value in ("a_pool", "av_pool"):
            for q in ("q4", "q1"):
                eff = S.window_effect(base, value=value, target=q, baseline="q2",
                                      context_col="strength")
                out_rows.append(dict(kind=f"motif_strength_{q}_vs_q2", **eff.as_row()))

    if "motif_best" in base.columns:
        for value in ("av_pool",):
            for m in [m for m in base["motif_best"].unique() if m not in ("none", "")][:6]:
                eff = S.window_effect(base, value=value, target=m, baseline="none",
                                      context_col="motif_best")
                out_rows.append(dict(kind=f"motif_{m}_vs_none", **eff.as_row()))

    if out_rows:
        tbl = pd.DataFrame(out_rows)
        save_table(tbl, result_path("13_motifs", f"{args.model}_{args.chrom}_motif_effects"))
        print(tbl.to_string(index=False))

    if "pert_family" in df.columns and (df["pert_family"] != "none").any():
        pert = df[df["variant"] == "perturb"].copy()
        for pref, name in (("a", "a_pool"), ("av", "av_pool")):
            pert = AA.pool_max(pert, pref, pool, name)
        res = []
        for fam in [f for f in pert["pert_family"].unique() if f != "reference"]:
            for value in ("a_pool", "av_pool", "kl_norm", "entropy_final"):
                if value not in pert.columns:
                    continue
                m = PB.paired_effect_frame(pert, value=value, family=fam)
                if m.empty:
                    continue
                near = m[np.abs(m["offset"] - m["offset"].median()) < 1e9]
                per_window = near.groupby("window_id")["delta"].mean()
                bs = S.cluster_bootstrap_mean(per_window.to_numpy())
                res.append(dict(family=fam, value=value, n_pairs=int(len(m)), **bs))
        if res:
            rt = pd.DataFrame(res)
            save_table(rt, result_path("13_motifs", f"{args.model}_{args.chrom}_perturbation"))
            print(rt.to_string(index=False))
    else:
        print("no perturbation rows in this extraction "
              "(re-run step5-extract with --perturb to get them)")
    return 0

# --------------------------------------------------------------------------
# step 0 -- motif models
# --------------------------------------------------------------------------

MOTIF_MANIFEST = "motifs/motif_set.json"


def _motif_manifest_path(name: str = "") -> Path:
    p = Path(name) if name else (ROOT / MOTIF_MANIFEST)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def step0_motifs(args) -> int:
    """Build, fetch, verify or show the motif models.

    Nothing downstream ships a matrix, so this step is what makes any
    motif-stratified number possible -- and what makes it citable, since the
    manifest it writes records every matrix's source, sample size and digest.
    """
    from . import motifs as MO

    out = _motif_manifest_path(args.out)

    if args.action == "plan":
        from .motifsrc import catalog_table

        tbl = pd.DataFrame(catalog_table())
        print(tbl.to_string(index=False))
        print("\nNothing was fetched or counted. `acquire` does the work.")
        return 0

    if args.action == "acquire":
        from .motifsrc import CATALOG, acquire

        ps = load_panel()
        panel_chroms = [c.strip() for c in
                        (args.panel_chroms or ",".join(ps.chroms)).split(",") if c.strip()]
        chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
        if not ps.fasta or not ps.gtf:
            print("[motifs] configs/panel.yaml needs `fasta` and `gtf`")
            return 2
        recipes = CATALOG
        if args.only:
            keep = {t.strip() for t in args.only.split(",") if t.strip()}
            recipes = tuple(r for r in CATALOG if r.name in keep or r.method in keep)
        ms, report = acquire(ps.fasta, ps.gtf, chroms, panel_chroms=panel_chroms,
                             recipes=recipes, jaspar_dir=args.jaspar_dir or None,
                             allow_network=not args.no_network,
                             out_dir=str(out.parent / "jaspar"))
        ms.save(str(out))
        save_json(report, result_path("00_motifs", "acquisition_report.json"))

        print(pd.DataFrame(report["obtained"])[
            [c for c in ("motif", "method", "n_sites", "consensus", "total_ic_bits", "source")
             if any(c in r for r in report["obtained"])]
        ].to_string(index=False) if report["obtained"] else "[motifs] nothing obtained")
        if report["skipped"]:
            print("\nnot obtained:")
            for row in report["skipped"]:
                print(f"  - {row['motif']}: {row.get('reason', '')}")
        if report["requires_external"]:
            print("\nyou must supply these yourself (nothing is substituted):")
            for row in report["requires_external"]:
                print(f"  - {row['motif']}: {row['source']}")
        print(f"\n[motifs] wrote {out} with {len(ms)} model(s); "
              f"full report in results/00_motifs/acquisition_report.json")
        return 0

    if args.action == "build":
        ps = load_panel()
        panel_chroms = [c.strip() for c in (args.panel_chroms or ",".join(ps.chroms)).split(",") if c.strip()]
        chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
        if not ps.fasta or not ps.gtf:
            print("[motifs] configs/panel.yaml needs `fasta` and `gtf` to derive splice motifs")
            return 2
        ms, report = MO.derive_splice_pwms(
            ps.fasta, ps.gtf, chroms, panel_chroms=panel_chroms,
            canonical_only=not args.include_noncanonical,
            min_sites=args.min_sites)

        if args.jaspar:
            ids = [t.strip() for t in args.jaspar.split(",") if t.strip()]
            fetched = MO.fetch_jaspar(ids, str(out.parent / "jaspar"))
            for mid, rec in fetched.items():
                for pwm in MO.load_jaspar(rec["path"]):
                    ms.add(pwm)
            report["jaspar"] = fetched

        ms.save(str(out))
        save_json(report, result_path("00_motifs", "derivation_report.json"))
        print(json.dumps(report, indent=2))
        print(f"\n[motifs] wrote {out} with {len(ms)} model(s)")
        print("[motifs] consensus check passed: donor core GT, acceptor core AG")
        return 0

    if args.action == "fetch-jaspar":
        if not args.jaspar:
            print("[motifs] pass --jaspar MA0108.3,MA0079.5 (version-pinned ids)")
            return 2
        ids = [t.strip() for t in args.jaspar.split(",") if t.strip()]
        fetched = MO.fetch_jaspar(ids, str(out.parent / "jaspar"))
        ms = MO.MotifSet.load(str(out)) if out.exists() else MO.MotifSet()
        for mid, rec in fetched.items():
            for pwm in MO.load_jaspar(rec["path"]):
                if pwm.name not in ms.pwms:
                    ms.add(pwm)
        ms.save(str(out))
        print(json.dumps(fetched, indent=2))
        return 0

    if args.action in ("verify", "show"):
        if not out.exists():
            print(f"[motifs] no manifest at {out}; run `step0-motifs build` first")
            return 2
        ms = MO.MotifSet.load(str(out))          # re-checks every digest
        rows = []
        for name, pwm in ms.pwms.items():
            ic = pwm.information_content(ms.background)
            rows.append(dict(
                motif=name, length=pwm.length, consensus=pwm.consensus(),
                core=pwm.consensus()[pwm.core_offset:pwm.core_offset + pwm.core_len],
                n_sites=pwm.provenance.n_sites or int(pwm.n_observations),
                total_ic_bits=round(float(ic.sum()), 3),
                source=pwm.provenance.source, digest=pwm.digest()[:12],
                detail=pwm.provenance.detail))
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\n[motifs] {len(ms)} matrices verified; every digest matches the manifest")
        print(f"[motifs] measurement columns (no model): {', '.join(MO.MEASUREMENT_COLUMNS)}")
        if args.action == "verify":
            try:
                MO.maxentscan_scorer()
                print("[motifs] maxentpy present: MaxEntScan splice strength available")
            except ImportError as exc:
                print(f"[motifs] MaxEntScan unavailable -- {exc.args[0].splitlines()[0]}")
        return 0

    print(f"[motifs] unknown action {args.action}")
    return 2


def step7d_dualref(args) -> int:
    """Settling against two references, joined at the handoff."""
    from . import dualref as DR
    from . import sitedefs as SD

    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    base_all = df
    base = df[df["variant"] == args.variant].copy()
    dr = DR.DualRef(onset=args.onset, rotation=args.rotation, n_blocks=args.n_blocks)

    # Before any depth is reported: is either curve flat enough that a
    # threshold on it is meaningless?  This is the construction's main exposure.
    diag = DR.curve_diagnostics(base, dr)
    print("[dualref] curve diagnostics")
    for k in ("to_pre", "to_pre_perp", "to_output"):
        v = diag.get(k, {})
        if v.get("available"):
            print(f"    {k:12s} {v['first']:+.4f} -> {v['last']:+.4f}  "
                  f"range {v['dynamic_range']:.5f}  "
                  f"{'thresholdable' if v['thresholdable'] else 'FLAT - do not threshold'}")
    print(f"    verdict: {diag['verdict']}")

    out, report = DR.dual_settle(base, dr, gamma_quantile=args.gamma_quantile)
    report["curve_diagnostics"] = diag

    # The relative approach is computed always, and becomes the PRIMARY
    # pre-handoff readout whenever the raw curve is flat -- which the rehearsal
    # shows is the expected case for a stream with a large persistent component.
    out = DR.relative_approach(out, dr, perp=False)
    if diag.get("to_pre_perp", {}).get("available"):
        out = DR.relative_approach(out, dr, perp=True)
    primary = ("a_pre" if diag.get("to_pre", {}).get("thresholdable") else "auc_pre")
    report["primary_pre_readout"] = primary
    print(f"[dualref] primary pre-handoff readout: {primary}")
    print(json.dumps(report, indent=2, default=str))

    prof = DR.dual_profile(out, dr)
    ctl = DR.self_reference_control(prof, dr)
    print("\nself-reference control:", json.dumps(ctl, indent=2))

    audit = S.covariate_audit(out, ADJUST_COVARIATES)
    if audit["all_missing"]:
        print(f"[covariates] dropped (never finite): {audit['all_missing']}")
    print(f"[covariates] positions retained jointly: {audit['joint_retained_fraction']:.3f}")

    tbl = DR.dual_context_table(out, dr, CONTEXTS)
    # and the same contrast on the relative readout, which is scale-free
    for value in ("auc_pre", "c_pre_rel_90"):
        if value in out.columns:
            rows = []
            for ctx in CONTEXTS:
                if ctx == "intron":
                    continue
                eff = S.window_effect(out, value=value, target=ctx, baseline="intron")
                rows.append(dict(regime="pre_relative", reference=f"h{dr.pre_ref}",
                                 context=ctx, **eff.as_row()))
            if rows:
                tbl = pd.concat([tbl, pd.DataFrame(rows)], ignore_index=True)
    save_table(tbl, result_path("14_dualref", f"{args.model}_{args.chrom}_context"))
    print("\n" + tbl.to_string(index=False))

    save_table(prof, result_path("14_dualref", f"{args.model}_{args.chrom}_profile"))
    DR.plot_dual(DR.dual_profile(out, dr, group_col="context"), dr,
                 result_path("14_dualref", f"{args.model}_{args.chrom}_dual.png"),
                 title=f"{args.model} {args.chrom}: settling against two references")

    bridge = DR.dual_bridge(out, dr, controls=ADJUST_COVARIATES)
    print("\nbridge (pre-handoff settling -> output commitment):")
    print(json.dumps(bridge, indent=2))

    # the same two regimes, split by how the site was defined
    sd_null = base_all[base_all["variant"] == "shuffle_di"] \
        if "variant" in base_all.columns else None
    sd, sd_report = SD.site_definitions(out, null_df=sd_null)
    if "site_agreement" in sd.columns:
        for value in (f"a_pre", f"a_post"):
            dis = SD.discordance_contrast(sd, value=value)
            if not dis.empty:
                save_table(dis, result_path("14_dualref",
                                            f"{args.model}_{args.chrom}_{value}_by_agreement"))
                print(f"\n{value} by site-definition agreement:")
                print(dis.to_string(index=False))

    save_json(dict(report=report, self_reference_control=ctl, bridge=bridge,
                   site_definitions=sd_report),
              result_path("14_dualref", f"{args.model}_{args.chrom}_summary.json"))
    return 0


def step7e_sitedefs(args) -> int:
    """GENCODE-annotated sites against motif-called sites, and their disagreement."""
    from . import analysis as AA
    from . import sitedefs as SD

    df = load_table(cache_path("extract", f"{args.model}_{args.chrom}_{args.tag}"))
    base = df[df["variant"] == args.variant].copy()
    pool = [f"{i:02d}" for i in range(0, args.rotation)]
    for pref, name in (("a", "a_pool"), ("av", "av_pool")):
        base = AA.pool_max(base, pref, pool, name)

    # the composition-matched null is the shuffle_di variant of the same panel
    null_df = df[df["variant"] == "shuffle_di"] if "variant" in df.columns else None
    if null_df is None or null_df.empty:
        print("[sitedefs] no shuffle_di variant in this extraction; the motif call "
              "cannot be FDR-calibrated. Re-run step5-extract including shuffle_di.")
    sd, reports = SD.site_definitions(base, target_fdr=args.fdr, null_df=null_df)
    print(json.dumps(reports, indent=2, default=str))
    if "site_agreement" not in sd.columns:
        return 2

    conc = SD.concordance_table(sd)
    save_table(conc, result_path("15_sitedefs", f"{args.model}_{args.chrom}_concordance"))
    print("\n" + conc.to_string(index=False))

    for value in ("a_pool", "av_pool"):
        if value not in sd.columns:
            continue
        both = SD.definition_contrast(sd, value=value)
        if not both.empty:
            save_table(both, result_path("15_sitedefs",
                                         f"{args.model}_{args.chrom}_{value}_definitions"))
            print(f"\nsame contrast under each definition ({value}):")
            print(both.to_string(index=False))
        dis = SD.discordance_contrast(sd, value=value)
        if not dis.empty:
            save_table(dis, result_path("15_sitedefs",
                                        f"{args.model}_{args.chrom}_{value}_agreement"))
            print(f"\nagreement cells against background ({value}):")
            print(dis.to_string(index=False))

    tags = [f"{i:02d}" for i in range(args.n_blocks)] + ["norm"]
    prof = SD.discordance_profile(sd, "a", tags)
    save_table(prof, result_path("15_sitedefs", f"{args.model}_{args.chrom}_profile"))
    save_json(dict(definitions=reports), result_path("15_sitedefs",
                                                     f"{args.model}_{args.chrom}_summary.json"))
    return 0


def step9_profile(args) -> int:
    """Linear decodability of biology at every block."""
    from . import probesweep as PS
    from . import sitedefs as SD

    npz = cache_path("raw", f"{args.model}_{args.chrom}_{args.tag}_states.npz")
    if not Path(npz).exists():
        print(f"missing {npz}; run step5-extract with raw_state_blocks set")
        return 2
    states, meta = PS.load_states(
        str(npz), lambda: load_table(cache_path("raw", f"{args.model}_{args.chrom}_{args.tag}_meta")))
    meta, _ = SD.site_definitions(meta)

    # motif_best_score is the max ACROSS motifs, not a motif: probing it as one
    # would put a derived summary in the same table as the models it summarises.
    motif_names = sorted({c[len("motif_"):-len("_score")] for c in meta.columns
                          if c.startswith("motif_") and c.endswith("_score")}
                         - {"best"})
    specs = PS.default_specs(motif_names if args.motif_probes else ())
    if args.labels:
        want = {t.strip() for t in args.labels.split(",") if t.strip()}
        specs = [sp for sp in specs if sp.name in want]

    sw = PS.sweep(states, meta, specs,
                  with_shuffled_control=not args.no_control, seed=42)
    save_table(sw, result_path("16_profile", f"{args.model}_{args.chrom}_sweep"))

    ret = PS.retention_table(sw, onset=args.onset, rotation=args.rotation,
                             n_blocks=args.n_blocks)
    save_table(ret, result_path("16_profile", f"{args.model}_{args.chrom}_retention"))
    print(ret.to_string(index=False))

    PS.plot_sweep(sw, result_path("16_profile", f"{args.model}_{args.chrom}_sweep.png"),
                  onset=args.onset, rotation=args.rotation,
                  title=f"{args.model} {args.chrom}: linear decodability by block")
    print("\nNote: this is linear decodability, not information. A drop across the "
          "handoff is consistent with summarisation AND with non-linear re-encoding; "
          "separating those needs the non-linear probe comparison, not this curve.")
    return 0


def step_dryrun(args) -> int:
    """Rehearse the whole pipeline on a synthetic genome and a rehearsal model."""
    from .dryrun import main as dryrun_main

    return dryrun_main(keep=args.keep or None, n_windows=args.n_windows,
                       window_bp=args.window_bp, scored_bp=args.scored_bp,
                       motif_units=args.motif_units,
                       stop_on_fail=args.stop_on_fail, verbose=args.verbose)


def step_audit_thresholds(args) -> int:
    """Every cut-off in the package, its kind, and what stands behind it."""
    from . import thresholds as TH

    df = TH.audit()
    pd.set_option("display.max_colwidth", 96)
    for kind in ("derived", "separated", "inherited", "removed", "VIOLATION"):
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        print(f"\n== {kind} ({len(sub)}) ==")
        for _, r in sub.iterrows():
            print(f"  {r['name']}  [{r['where']}]")
            if r["value"] != "-":
                print(f"      value : {r['value']}")
            print(f"      basis : {r['basis']}")
            if r["sweep"]:
                print(f"      sweep : {r['sweep']}")
            if r["citation"]:
                print(f"      cite  : {r['citation']}")
    s = TH.summary()
    print(f"\n{s['n']} entries: {s['by_kind']}")
    if s["violations"]:
        print(f"!! {s['violations']} VIOLATION(s): a removed constant is still defined")
        return 1
    print("no violations: every live cut-off is derived, separated or inherited")
    return 0


def step_selftest(args) -> int:
    from .selftest import main as selftest_main

    return selftest_main()


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="exp1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model_default="7b"):
        p.add_argument("--model", default=model_default)
        p.add_argument("--chrom", default="chr22")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--shuffle-seed", type=int, default=-1,
                       help="if >=0, shuffle weights within tensors (untrained control)")
        return p

    p = sub.add_parser("selftest"); p.set_defaults(fn=step_selftest)

    p = sub.add_parser("audit-thresholds",
                       help="every cut-off, its kind, and what stands behind it")
    p.set_defaults(fn=step_audit_thresholds)

    p = sub.add_parser("dryrun",
                       help="rehearse every step end to end on synthetic data (no GPU)")
    p.add_argument("--keep", default="", help="directory to build in and leave behind")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--window-bp", type=int, default=2000)
    p.add_argument("--scored-bp", type=int, default=1000)
    p.add_argument("--motif-units", type=int, default=1100)
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=step_dryrun)

    p = sub.add_parser("step0-motifs",
                       help="build / fetch / verify the motif models (no matrix ships with this package)")
    p.add_argument("action",
                   choices=["plan", "acquire", "build", "fetch-jaspar", "verify", "show"],
                   help="plan: print the catalogue without touching data. "
                        "acquire: obtain everything obtainable (the normal choice). "
                        "build: splice donor/acceptor only.")
    p.add_argument("--chroms", default="chr1,chr2,chr3",
                   help="chromosomes to COUNT splice sites on; must not be panel chromosomes")
    p.add_argument("--panel-chroms", default="",
                   help="defaults to the panel chromosomes in configs/panel.yaml")
    p.add_argument("--jaspar", default="",
                   help="comma-separated version-pinned JASPAR ids, e.g. MA0108.3")
    p.add_argument("--include-noncanonical", action="store_true",
                   help="count non-GT-AG introns too (default: U2 canonical only)")
    p.add_argument("--min-sites", type=int, default=500)
    p.add_argument("--only", default="",
                   help="restrict to these recipe names or methods, e.g. intron,anchor")
    p.add_argument("--jaspar-dir", default="",
                   help="directory of JASPAR files already downloaded (used before the network)")
    p.add_argument("--no-network", action="store_true",
                   help="never reach out; JASPAR entries are reported as missing instead")
    p.add_argument("--out", default="")
    p.set_defaults(fn=step0_motifs)

    p = sub.add_parser("step1-blockmap"); p.add_argument("--models", default="")
    p.set_defaults(fn=step1_blockmap)

    p = common(sub.add_parser("step2-onset")); p.add_argument("--n-windows", type=int, default=10)
    p.set_defaults(fn=step2_onset)

    p = common(sub.add_parser("step3-fidelity"))
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-2)
    p.set_defaults(fn=step3_fidelity)

    p = sub.add_parser("step4-panel")
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--n-windows", type=int, default=None,
                   help="override configs/panel.yaml; omitted, the config wins")
    p.set_defaults(fn=step4_panel)

    p = common(sub.add_parser("step5-uref")); p.add_argument("--n-windows", type=int, default=40)
    p.set_defaults(fn=step5_uref)

    p = common(sub.add_parser("step5-extract"))
    p.add_argument("--variants", default="real,random,polyA,shuffle_di")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--raw-per-context", type=int, default=300)
    p.add_argument("--offload-cpu", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--perturb", action="store_true",
                   help="also run paired motif edits at annotated splice sites")
    p.add_argument("--perturb-families", default="core_mut,flank_shuffle,rescue")
    p.add_argument("--perturb-sites", type=int, default=2)
    p.add_argument("--perturb-every", type=int, default=5)
    p.add_argument("--motifs", default="",
                   help="motif manifest from step0-motifs; without it the run carries "
                        "measurement columns only and no model scores")
    p.set_defaults(fn=step5_extract)

    p = sub.add_parser("step6-prereg"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=step6_prereg)

    p = common(sub.add_parser("step7b-regimes"))
    p.add_argument("--tag", default="main")
    p.add_argument("--variant", default="real")
    p.add_argument("--onset", type=int, default=28)
    p.add_argument("--rotation", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=32)
    p.set_defaults(fn=step7b_regimes)

    p = common(sub.add_parser("step7d-dualref"))
    p.add_argument("--tag", default="main")
    p.add_argument("--variant", default="real")
    p.add_argument("--onset", type=int, default=28)
    p.add_argument("--rotation", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=32)
    p.add_argument("--gamma-quantile", type=float, default=0.70)
    p.set_defaults(fn=step7d_dualref)

    p = common(sub.add_parser("step7e-sitedefs"))
    p.add_argument("--tag", default="main")
    p.add_argument("--variant", default="real")
    p.add_argument("--rotation", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=32)
    p.add_argument("--fdr", type=float, default=0.05,
                   help="motif call = the cut whose false-discovery rate against the "
                        "dinucleotide-shuffled null is at most this")
    p.set_defaults(fn=step7e_sitedefs)

    p = common(sub.add_parser("step9-profile"))
    p.add_argument("--tag", default="main")
    p.add_argument("--onset", type=int, default=28)
    p.add_argument("--rotation", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=32)
    p.add_argument("--labels", default="", help="restrict to these probe names")
    p.add_argument("--motif-probes", action="store_true")
    p.add_argument("--no-control", action="store_true",
                   help="skip the shuffled-label floor (not recommended)")
    p.set_defaults(fn=step9_profile)

    p = common(sub.add_parser("step7c-motifs"))
    p.add_argument("--tag", default="main")
    p.add_argument("--motifs", default="")
    p.add_argument("--onset", type=int, default=28)
    p.add_argument("--rotation", type=int, default=30)
    p.set_defaults(fn=step7c_motifs)

    for name, fn in (("step7-analysis", step7_analysis), ("step8-stage2", step8_stage2)):
        p = common(sub.add_parser(name))
        p.add_argument("--tag", default="main")
        p.add_argument("--onset", type=int, default=28)
        p.add_argument("--rotation", type=int, default=30)
        p.add_argument("--allow-unfrozen", action="store_true")
        p.set_defaults(fn=fn)

    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

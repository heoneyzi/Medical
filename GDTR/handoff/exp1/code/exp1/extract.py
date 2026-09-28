"""Step 5: the extraction pass.

What is stored, and why it is more than Experiment 1 needs
----------------------------------------------------------
Forward passes are the expensive part; re-running them for every follow-up
question is what makes an interpretability project slow.  So each position gets
a wide record that supports later experiments without a new pass:

per position, once
    identity, annotation context, covariates, current/next base and token id
per position, from the unmodified head
    full-vocab entropy, log p of the realised base, top-1, margin, ACGT mass,
    the four ACGT logits and probabilities
per position, per block (0..L-1 and the post-norm state)
    p, n, q, a                     numerator / denominator decomposition
    a_u, a_v                       carrier and content channels
    sctr                           energy fraction in the ACGT contrast space
    kl, ent, top1, acgt_mass       logit-lens readouts against the final head
    logit_{A,C,G,T}                so any later re-derivation is offline
per position, for the branch blocks (default 24..31)
    mixer/mlp update projection and norm against the same reference

and, for a stratified subsample of positions, the raw hidden states at a few
selected blocks, so probes / SAEs / transcoders can be trained later.

Everything is computed in float64 on device and written as float32.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .config import ModelSpec, PanelSpec, cache_path, result_path
from .io import save_json, save_table
from .labels import ChromTracks
from .panels import window_labels


def window_bp_of(row, default: int = 6000) -> int:
    """The window length this panel row was built with.

    Read from the panel rather than hardcoded: the two are the same number for
    the real 6 kb panel, but a hardcoded literal silently mis-slices any other
    panel -- including every rehearsal and any future window-length sweep.
    """
    try:
        v = int(row["window_bp"])
        return v if v > 0 else default
    except (KeyError, TypeError, ValueError):
        return default


def load_motif_set(manifest: Optional[str]):
    """Load a motif manifest, or an explicitly empty set.

    An empty set is a legitimate state -- it means the run carries measurement
    columns and no model scores -- but it is never silently filled in.
    """
    from .motifs import MotifSet

    if not manifest:
        return MotifSet()
    return MotifSet.load(manifest)
from .variants import make_variant, to_str


@dataclass
class ExtractConfig:
    variants: List[str] = field(default_factory=lambda: ["real"])
    branch_blocks: List[int] = field(default_factory=lambda: list(range(24, 32)))
    lens_blocks: Optional[List[int]] = None       # default: every block
    store_block_logits: bool = True               # ACGT logits per block
    # Every block by default: the all-layer information profile needs the full
    # state at every tap, and a stratified subsample is what makes that affordable.
    raw_state_blocks: List[int] = field(default_factory=lambda: list(range(32)))
    raw_state_variants: List[str] = field(default_factory=lambda: ["real"])
    raw_state_per_context: int = 300              # positions kept per context per chromosome
    offload_cpu: bool = False
    seed: int = 42
    max_windows: Optional[int] = None
    # motif perturbations: one paired edit per annotated splice site in the window
    perturb: bool = False
    perturb_families: List[str] = field(
        default_factory=lambda: ["core_mut", "flank_shuffle", "rescue"])
    perturb_max_sites: int = 2
    perturb_flank: int = 100
    perturb_every_n_windows: int = 5      # perturb a subset; edits are paired, not independent
    # Motif models.  None means: emit measurement columns only and no model
    # scores.  There is no default matrix set, so a motif-stratified analysis
    # must state which models it used.
    motif_manifest: Optional[str] = None
    # Set by run_extraction so the raw-state quota can be spread over the panel
    # instead of exhausted on its first windows.
    _n_windows_planned: int = 0
    # Dual reference.  Blocks 0..onset-1 are near-orthogonal to the output frame,
    # so alignment to h_norm measures little there; the pre-handoff endpoint is
    # the reference those blocks are actually approaching.  Both are stored, at
    # every block, so the two regimes can be read against the right target and
    # against each other.  The second entry is a HELD-OUT control: if the
    # approach were an artefact of including the endpoint in its own reference,
    # the two curves would differ in shape, and they can be compared directly.
    pre_ref_blocks: List[int] = field(default_factory=lambda: [27, 26])
    store_pre_perp: bool = True    # also store the u-removed version


# --------------------------------------------------------------------------
# reference direction u
# --------------------------------------------------------------------------

def estimate_u(runner, tracks: ChromTracks, panel: pd.DataFrame, n_windows: int = 40) -> np.ndarray:
    """Global carrier direction: normalise each position's post-norm state, average, renormalise.

    Estimated on a subsample and then **frozen**, so that every window is
    measured against the same u and the channel split is comparable across the
    panel.  This mirrors the handoff manuscript's definition of u as the
    normalised mean direction.
    """
    import torch

    from .modelio import CaptureConfig

    runner.attach(CaptureConfig(block_outputs=False, keep_device=not False))
    acc = None
    count = 0
    for _, row in panel.head(n_windows).iterrows():
        wbp = window_bp_of(row)
        seq = to_str(tracks.sequence[int(row["start"]): int(row["start"]) + wbp])
        ids = runner.tokenize(seq)
        runner.forward(ids)
        h = runner.taps["hnorm"][0].to(torch.float64)
        h = h / torch.clamp(torch.linalg.vector_norm(h, dim=-1, keepdim=True), min=1e-30)
        s = h.sum(dim=0)
        acc = s if acc is None else acc + s
        count += h.shape[0]
        runner.taps.clear()
    runner.detach()
    u = acc / max(count, 1)
    u = u / torch.clamp(torch.linalg.vector_norm(u), min=1e-30)
    return u.cpu().numpy()


# --------------------------------------------------------------------------
# main extraction
# --------------------------------------------------------------------------

def extract_window(
    runner,
    seq: str,
    u_np: np.ndarray,
    cfg: ExtractConfig,
    scored_slice: slice,
    acgt_ids: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Run one sequence and reduce every tap to per-position scalars."""
    import torch

    dev = runner.taps and next(iter(runner.taps.values())).device or runner.device
    ids = runner.tokenize(seq)
    logits_final = runner.forward(ids)
    if logits_final.dim() == 3:
        logits_final = logits_final[0]

    taps = runner.taps
    hnorm = taps["hnorm"][0].to(torch.float64)
    ref_unit = hnorm / torch.clamp(torch.linalg.vector_norm(hnorm, dim=-1, keepdim=True), min=1e-30)
    u = torch.as_tensor(u_np, dtype=torch.float64, device=hnorm.device)

    # -- the pre-handoff references ---------------------------------------
    pre_units: Dict[str, "torch.Tensor"] = {}
    pre_perp_units: Dict[str, "torch.Tensor"] = {}
    u_unit = u / torch.clamp(torch.linalg.vector_norm(u), min=1e-30)
    for b in cfg.pre_ref_blocks:
        key = f"h{b}"
        if key not in taps:
            continue
        hp = taps[key][0].to(torch.float64)
        pre_units[f"{b:02d}"] = hp / torch.clamp(
            torch.linalg.vector_norm(hp, dim=-1, keepdim=True), min=1e-30)
        if cfg.store_pre_perp:
            hp_perp = hp - torch.sum(hp * u_unit, dim=-1, keepdim=True) * u_unit
            pre_perp_units[f"{b:02d}"] = hp_perp / torch.clamp(
                torch.linalg.vector_norm(hp_perp, dim=-1, keepdim=True), min=1e-30)

    E = runner.unembed_weight().to(torch.float64)
    acgt_idx = torch.as_tensor(list(acgt_ids), dtype=torch.long, device=E.device)
    basis = M.acgt_contrast_basis(E, acgt_idx)

    final_logp = M.log_softmax(logits_final.to(torch.float64))
    out: Dict[str, np.ndarray] = {}
    prev_lp = None

    # index of the base actually realised at t+1, so per-block rank and
    # log-probability of the true answer can be stored at every tap
    ids_flat = ids[0] if ids.dim() == 2 else ids
    true_next_idx = torch.cat([ids_flat[1:], ids_flat[-1:]]).to(torch.long)

    sl = scored_slice

    def put(name: str, t):
        arr = t[sl].detach().to(torch.float32).cpu().numpy() if hasattr(t, "detach") else np.asarray(t)[sl]
        out[name] = arr

    # ---- final head ------------------------------------------------------
    fm = M.lens_metrics(logits_final.to(torch.float64), final_logp, acgt_idx)
    put("entropy_final", fm.entropy)
    put("acgt_mass_final", fm.acgt_mass)
    top2 = torch.topk(final_logp, k=2, dim=-1).values
    put("margin_final", top2[:, 0] - top2[:, 1])
    put("top1_final", torch.argmax(final_logp, dim=-1).to(torch.float64))
    for j, base in enumerate("ACGT"):
        put(f"logit{base}_final", logits_final.to(torch.float64)[:, acgt_idx[j]])

    # ---- per block -------------------------------------------------------
    n_blocks = runner.anatomy.n_blocks
    lens_blocks = cfg.lens_blocks if cfg.lens_blocks is not None else list(range(n_blocks))
    norm_mod = runner.final_norm
    param_dtype = next(runner.torch_model.parameters()).dtype

    tap_items = [(f"{i:02d}", taps[f"h{i}"][0]) for i in range(n_blocks) if f"h{i}" in taps]
    tap_items.append(("norm", hnorm))

    prev_h = None
    for tag, h_raw in tap_items:
        h = h_raw.to(torch.float64)
        p, n, q, a = M.pnq(h, ref_unit)
        split = M.channel_split(h, ref_unit, u)
        put(f"p_{tag}", p)
        put(f"n_{tag}", n)
        put(f"q_{tag}", q)
        put(f"a_{tag}", a)
        put(f"au_{tag}", split.a_u)
        put(f"av_{tag}", split.a_v)

        # cosine to each pre-handoff reference.  a_{tag} is the cosine to the
        # output frame, so the two together give both regimes at every block.
        hn = torch.clamp(torch.linalg.vector_norm(h, dim=-1, keepdim=True), min=1e-30)
        h_unit = h / hn
        for rb, ru in pre_units.items():
            put(f"apre{rb}_{tag}", torch.sum(h_unit * ru, dim=-1))
        if pre_perp_units:
            h_perp = h - torch.sum(h * u_unit, dim=-1, keepdim=True) * u_unit
            h_perp = h_perp / torch.clamp(
                torch.linalg.vector_norm(h_perp, dim=-1, keepdim=True), min=1e-30)
            for rb, ru in pre_perp_units.items():
                put(f"apreperp{rb}_{tag}", torch.sum(h_perp * ru, dim=-1))
            del h_perp
        del h_unit
        if tag == "norm":
            put("alpha_ref", split.alpha)
            put("beta_ref", split.beta)
            ident = float(M.channel_identity_error(split))
            out["_identity_err"] = np.float32(ident)

        # --- trajectory geometry: how this block moved the stream ----------
        # Cheap, and it is the part a later experiment cannot reconstruct from
        # summaries: step size, turning angle, and the carrier/content split of
        # the MAGNITUDE (the cosines alone lose the scale).
        if prev_h is not None:
            step = h - prev_h
            sn = torch.linalg.vector_norm(step, dim=-1)
            put(f"stepn_{tag}", sn)
            pn = torch.linalg.vector_norm(prev_h, dim=-1)
            put(f"turn_{tag}", torch.sum(h * prev_h, dim=-1) /
                torch.clamp(hn.squeeze(-1) * pn, min=1e-30))
            put(f"stepalign_{tag}", torch.sum(step * ref_unit, dim=-1) /
                torch.clamp(sn, min=1e-30))
            del step
        pu = torch.sum(h * u_unit, dim=-1)
        put(f"nu_{tag}", pu)                                   # carrier magnitude
        put(f"nperp_{tag}", torch.sqrt(torch.clamp(
            torch.sum(h * h, dim=-1) - pu * pu, min=0.0)))     # content magnitude
        prev_h = h

        want_lens = (tag == "norm") or (int(tag) in lens_blocks if tag != "norm" else True)
        if want_lens:
            z = h if tag == "norm" else norm_mod(h_raw.to(param_dtype)).to(torch.float64)
            put(f"sctr_{tag}", M.subspace_energy_fraction(z, basis))
            lg = z @ E.T
            lm = M.lens_metrics(lg, final_logp, acgt_idx)
            put(f"kl_{tag}", lm.kl_to_final)
            put(f"ent_{tag}", lm.entropy)
            put(f"top1_{tag}", lm.top1_match)
            put(f"acgtmass_{tag}", lm.acgt_mass)

            # --- what the block's own distribution says, not only its distance
            # to the final one.  Margin, the realised base's log-probability and
            # its RANK are what a later calibration or early-exit experiment
            # needs, and none of them is recoverable from KL afterwards.
            blp = M.log_softmax(lg)
            acgt_lp = blp[:, acgt_idx]
            top2 = torch.topk(blp, k=2, dim=-1).values
            put(f"margin_{tag}", top2[:, 0] - top2[:, 1])
            put(f"acgtent_{tag}", -(torch.softmax(acgt_lp, dim=-1)
                                    * torch.log_softmax(acgt_lp, dim=-1)).sum(-1))
            if true_next_idx is not None:
                lp_true = torch.gather(blp, 1, true_next_idx[:, None]).squeeze(-1)
                put(f"logptrue_{tag}", lp_true)
                put(f"ranktrue_{tag}", (blp > lp_true[:, None]).sum(-1).to(torch.float64))
                put(f"acgtargmax_{tag}", torch.argmax(acgt_lp, dim=-1).to(torch.float64))
            if prev_lp is not None:
                put(f"klprev_{tag}", torch.sum(
                    torch.exp(blp) * (blp - prev_lp), dim=-1))
            prev_lp = blp
            del blp, acgt_lp
            if cfg.store_block_logits:
                for j, base in enumerate("ACGT"):
                    put(f"logit{base}_{tag}", lg[:, acgt_idx[j]])
            del lg
        del h
    # ---- branch updates --------------------------------------------------
    for i in cfg.branch_blocks:
        for key, short in ((f"m{i}", "mx"), (f"g{i}", "mlp")):
            if key not in taps:
                continue
            b = taps[key][0].to(torch.float64)
            bp = torch.sum(b * ref_unit, dim=-1)
            bn = torch.linalg.vector_norm(b, dim=-1)
            put(f"{short}p_{i:02d}", bp)
            put(f"{short}n_{i:02d}", bn)
            del b
        # branch geometry: are the two branches doing the same thing?
        if f"m{i}" in taps and f"g{i}" in taps:
            bm = taps[f"m{i}"][0].to(torch.float64)
            bg = taps[f"g{i}"][0].to(torch.float64)
            nm = torch.linalg.vector_norm(bm, dim=-1)
            ng = torch.linalg.vector_norm(bg, dim=-1)
            put(f"branchcos_{i:02d}", torch.sum(bm * bg, dim=-1) /
                torch.clamp(nm * ng, min=1e-30))
            put(f"mlpshare_{i:02d}", ng / torch.clamp(nm + ng, min=1e-30))
            del bm, bg
        if f"xin{i}" in taps:
            xin = taps[f"xin{i}"][0].to(torch.float64)
            put(f"xinp_{i:02d}", torch.sum(xin * ref_unit, dim=-1))
            put(f"xinn_{i:02d}", torch.linalg.vector_norm(xin, dim=-1))
            del xin

    return out


def run_extraction(
    runner,
    tracks: ChromTracks,
    panel: pd.DataFrame,
    u_np: np.ndarray,
    cfg: ExtractConfig,
    model_key: str,
    tag: str = "main",
) -> Path:
    """Extract the whole panel for one model and one chromosome."""
    import torch

    from .modelio import CaptureConfig

    acgt = runner.acgt_ids
    acgt_ids = [acgt[b] for b in "ACGT"]

    cap = CaptureConfig(
        block_outputs=True,
        block_inputs_for=tuple(cfg.branch_blocks),
        branches_for=tuple(cfg.branch_blocks),
        keep_device=not cfg.offload_cpu,
    )
    runner.attach(cap)

    rows: List[pd.DataFrame] = []
    raw_keep: Dict[str, List[np.ndarray]] = {f"h{b}": [] for b in cfg.raw_state_blocks}
    raw_meta: List[pd.DataFrame] = []
    per_window_summary: List[Dict[str, object]] = []
    seen_per_context: Dict[str, int] = {}

    motif_set = load_motif_set(cfg.motif_manifest)
    if cfg.perturb:
        motif_set.require("motif perturbation (--perturb)")

    todo = panel if cfg.max_windows is None else panel.head(cfg.max_windows)
    cfg._n_windows_planned = len(todo)

    for wi, row in todo.reset_index(drop=True).iterrows():
        start = int(row["start"])
        wbp = window_bp_of(row)
        real = tracks.sequence[start : start + wbp]
        lab = window_labels(tracks, int(row["scored_start"]), int(row["scored_end"]),
                            window_start=start, motif_set=motif_set)
        scored_lo = int(row["scored_start"]) - start
        scored_hi = int(row["scored_end"]) - start
        sl = slice(scored_lo, scored_hi)

        for variant in cfg.variants:
            seed = cfg.seed + 1000 * wi + hash(variant) % 997
            arr = make_variant(variant, real, seed)
            seq = to_str(arr)

            runner.taps.clear()
            res = extract_window(runner, seq, u_np, cfg, sl, acgt_ids)
            ident = float(res.pop("_identity_err", np.float32(0.0)))

            df = pd.DataFrame({k: v for k, v in res.items()})
            df.insert(0, "model", model_key)
            df.insert(1, "variant", variant)
            df.insert(2, "chrom", tracks.chrom)
            df.insert(3, "window_id", row["window_id"])
            df.insert(4, "legacy_window", bool(row["legacy"]))
            df["pos"] = lab["pos"].to_numpy()
            df["offset"] = np.arange(scored_lo, scored_hi)
            for c in lab.columns:
                if c != "pos":
                    df[c] = lab[c].to_numpy()

            base_now = arr[sl]
            base_next = np.concatenate([arr[scored_lo + 1 : scored_hi + 1], [ord("N")]])[: len(base_now)]
            df["base"] = [chr(int(b)) for b in base_now]
            df["base_next"] = [chr(int(b)) for b in base_next]
            id_of = {b: acgt[b] for b in "ACGT"}
            df["logp_true_final"] = [
                float(df.loc[i, f"logit{df.loc[i, 'base_next']}_final"])
                if df.loc[i, "base_next"] in id_of else np.nan
                for i in range(len(df))
            ]
            df["identity_err"] = ident
            df["pert_family"] = "none"
            df["pert_site"] = -1
            df["pert_motif"] = ""
            df["pert_param"] = ""
            df["pert_pair_id"] = ""
            rows.append(df)

            if variant in cfg.raw_state_variants and cfg.raw_state_blocks:
                _collect_raw(runner, df, sl, cfg, raw_keep, raw_meta, seen_per_context)

            per_window_summary.append(dict(
                window_id=row["window_id"], variant=variant,
                identity_err=ident,
                **{f"mean_n_{k.split('_')[1]}": float(np.mean(v))
                   for k, v in res.items() if k.startswith("n_")},
                **{f"mean_a_{k.split('_')[1]}": float(np.mean(v))
                   for k, v in res.items() if k.startswith("a_")},
            ))

        if cfg.perturb and (wi % max(cfg.perturb_every_n_windows, 1) == 0):
            rows.extend(_perturbation_rows(
                runner, tracks, row, real, lab, sl, scored_lo, scored_hi,
                u_np, cfg, acgt, acgt_ids, model_key, wi, motif_set))

        runner.taps.clear()
        if wi % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"[extract] {model_key} {tracks.chrom} window {wi+1}/{len(todo)}")

    runner.detach()

    table = pd.concat(rows, axis=0, ignore_index=True)
    base = cache_path("extract", f"{model_key}_{tracks.chrom}_{tag}")
    path = save_table(table, base, meta=dict(
        model=model_key, chrom=tracks.chrom, tag=tag,
        variants=cfg.variants, branch_blocks=cfg.branch_blocks,
        dtype_ledger=runner.dtype_ledger(),
    ))

    summary = pd.DataFrame(per_window_summary)
    save_table(summary, cache_path("extract", f"{model_key}_{tracks.chrom}_{tag}_windows"))

    if raw_meta:
        meta = pd.concat(raw_meta, ignore_index=True)
        arrays = {k: np.concatenate(v, axis=0) for k, v in raw_keep.items() if v}
        np.savez_compressed(
            cache_path("raw", f"{model_key}_{tracks.chrom}_{tag}_states.npz"),
            **arrays,
        )
        save_table(meta, cache_path("raw", f"{model_key}_{tracks.chrom}_{tag}_meta"))

    return path


def _collect_raw(runner, df: pd.DataFrame, sl: slice, cfg: ExtractConfig,
                 raw_keep: Dict[str, List[np.ndarray]], raw_meta: List[pd.DataFrame],
                 seen: Dict[str, int]) -> None:
    """Keep a stratified subsample of full hidden states for later probing work."""
    import torch

    # Spread the quota ACROSS windows, and sample inside a window at random.
    #
    # Filling greedily from the first window that has room looks harmless and is
    # not: a common context fills its whole quota in one window, the probe sweep
    # then groups folds by window and finds a single group, and every curve for
    # that class is undefined or fitted on one locus.  The rehearsal showed
    # coding_exon, intron, intergenic and three_utr each coming from exactly one
    # window.  Taking the FIRST positions is the second half of the same bug:
    # they are the 5' edge of the scored region, not a sample of it.
    quota = cfg.raw_state_per_context
    n_planned = max(int(cfg._n_windows_planned or 1), 1)
    per_window = max(quota // n_planned, 1)
    rng = np.random.default_rng(cfg.seed + 104729)
    take_idx: List[int] = []
    for ctx, g in df.groupby("context"):
        used = seen.get(ctx, 0)
        if used >= quota:
            continue
        room = min(quota - used, per_window)
        pool = g.index.to_numpy()
        if len(pool) > room:
            pool = rng.choice(pool, size=room, replace=False)
        take_idx.extend(int(i) for i in pool)
        seen[ctx] = used + len(pool)
    if not take_idx:
        return
    local = np.asarray(take_idx) - df.index[0]
    for b in cfg.raw_state_blocks:
        key = f"h{b}"
        if key not in runner.taps:
            continue
        h = runner.taps[key][0][sl][local].to(torch.float32).cpu().numpy()
        raw_keep[key].append(h)
    keep = ["model", "chrom", "window_id", "pos", "context", "base", "base_next"]
    keep += [c for c in df.columns
             if c.startswith("motif_") or c in ("splice_core", "splice_canonical",
                                                "ppt_fraction", "cpg_oe",
                                                "kmer3_entropy",
                                                "repeat", "gc", "phylop",
                                                "boundary_dist", "feature_len",
                                                "entropy_final", "logp_true_final")]
    raw_meta.append(df.loc[take_idx, [c for c in dict.fromkeys(keep)
                                      if c in df.columns]].copy())


def _perturbation_rows(runner, tracks, row, real, lab, sl, scored_lo, scored_hi,
                       u_np, cfg: ExtractConfig, acgt, acgt_ids, model_key, wi,
                       motif_set):
    """Run the paired motif edits for one window and return their rows.

    Every edited sequence is emitted together with its own untouched reference
    under the same ``pert_pair_id``, so the downstream comparison is a paired
    difference at the same position in the same window -- composition, context
    and window identity are held fixed by construction rather than adjusted for.
    """
    from .perturb import build_perturbations, sites_from_context

    cores = (lab["splice_core"].to_numpy() if "splice_core" in lab.columns else None)
    sites = sites_from_context(
        lab["context"].to_numpy(), int(row["scored_start"]), int(row["start"]),
        motif_set, cores=cores,
        max_sites=cfg.perturb_max_sites, flank=cfg.perturb_flank,
        window_bp=window_bp_of(row))
    if not sites:
        return []

    plans = build_perturbations(real, sites, motif_set,
                                families=cfg.perturb_families,
                                seed=cfg.seed + 7919 * wi)
    out = []
    for plan in plans:
        runner.taps.clear()
        res = extract_window(runner, to_str(plan.seq), u_np, cfg, sl, acgt_ids)
        ident = float(res.pop("_identity_err", np.float32(0.0)))
        df = pd.DataFrame({k: v for k, v in res.items()})
        df.insert(0, "model", model_key)
        df.insert(1, "variant", "perturb")
        df.insert(2, "chrom", tracks.chrom)
        df.insert(3, "window_id", row["window_id"])
        df.insert(4, "legacy_window", bool(row["legacy"]))
        df["pos"] = lab["pos"].to_numpy()
        df["offset"] = np.arange(scored_lo, scored_hi)
        for c in lab.columns:
            if c != "pos":
                df[c] = lab[c].to_numpy()
        base_now = plan.seq[sl]
        base_next = np.concatenate([plan.seq[scored_lo + 1 : scored_hi + 1],
                                    [ord("N")]])[: len(base_now)]
        df["base"] = [chr(int(b)) for b in base_now]
        df["base_next"] = [chr(int(b)) for b in base_next]
        df["identity_err"] = ident
        # Carry EVERY key the plan emitted, not a hand-listed subset: the
        # paired-effect frame joins on pert_label, and a fixed list silently
        # drops whatever a new perturbation family adds.
        for k in ("pert_family", "pert_site", "pert_motif", "pert_param",
                  "pert_pair_id", "pert_label"):
            df[k] = plan.meta.get(k, "")
        for k, v in plan.meta.items():
            if k not in df.columns:
                df[k] = v
        df["pert_core_offset"] = plan.meta.get("pert_core_offset", -1)
        df["pert_dist_to_core"] = df["offset"] - df["pert_core_offset"]
        out.append(df)
    return out

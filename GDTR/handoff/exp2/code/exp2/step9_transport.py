"""Step 9 — non-trivial proof that b28/b29 information reaches b30 and out.

Upstream having an effect and downstream having an effect does not prove
transport: both could act on the output independently. So one FIXED partial
mediator must reproduce the upstream effect, block it, and rescue it.

F is everything after x31 -- b31 attention m31, b31 MLP g31, the final
RMSNorm and the unembedding. A success therefore means b30 -> b31 -> readout
transport, not "b30 goes straight to the readout"; Step 9-3 audits b31.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import solve_beta_shape
from .manifest import Manifest
from .stats import (
    Estimate, Row, cluster_bootstrap, intersection_union, paired_contrast,
    required_cluster_count,
)
from .taps import Edit, Evo2Runner, replace_with
from .step6_alpha import FREEZE_SETS, _freeze_edits


CONDITIONS = ("Y0", "YU", "YD", "YB", "YA", "YR")

CORE_CONTROL_FAMILIES = (
    "rank_matched_random_subspace",
    "wrong_pair_donor",
    "shifted_position_donor",
)


@dataclass
class TransportRow:
    pair_id: str
    condition: str
    b31: str                 # "free" | "freeze_all" | "freeze_attn" | "freeze_mlp"
    mediator: str            # "true" | control family name
    d_shape: float
    log_beta: float
    signed_following: float  # projection onto the pre-specified Y_U direction
    ood_score: float
    dose: float = 1.0
    patch_mode: str = "target"     # target | context | window | all
    carrier_shift: float = float("nan")
    content_shift: float = float("nan")
    projector_hash: str = ""
    state_source: str = "same_recipient"  # natural_donor is generalisation only
    dependency_keys: tuple[str, ...] = ()


def _split(x31: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
    Pm = P.to(x31.dtype).to(x31.device)
    M = x31 @ Pm.T
    return M, x31 - M


def _compose(M: Tensor, Q: Tensor) -> Tensor:
    return M + Q


def projector_fingerprint(P: Tensor) -> str:
    """Stable content hash stored beside every locked mediator result."""
    raw = P.detach().double().cpu().contiguous().numpy().tobytes()
    return __import__("hashlib").sha256(raw).hexdigest()


def freeze_projector_metadata(P: Tensor, *, source_split: str,
                              source_role: str = "discovery") -> dict:
    """Create the metadata that must be sealed before locked analysis."""
    _, rank = validate_frozen_projector(P, locked=False)
    return {"sha256": projector_fingerprint(P), "rank": rank,
            "source_split": source_split, "source_role": source_role,
            "frozen": True}


def validate_frozen_projector(P: Tensor, metadata: Optional[dict] = None,
                              *, locked: bool = False,
                              tolerance: float = 1e-7) -> tuple[str, int]:
    """Validate shape, finiteness, idempotence, rank, hash, and provenance."""
    Pd = P.detach().double()
    if Pd.ndim != 2 or Pd.shape[0] != Pd.shape[1]:
        raise ValueError(f"P must be a square [d,d] matrix, got {tuple(Pd.shape)}")
    if not torch.isfinite(Pd).all():
        raise ValueError("P contains non-finite values")
    sym = float((Pd - Pd.T).norm())
    idem = float((Pd @ Pd - Pd).norm())
    rank = int(torch.linalg.matrix_rank(Pd))
    if sym >= tolerance or idem >= 10 * tolerance:
        raise ValueError(f"P is not an orthogonal projector: sym={sym:.2e}, idem={idem:.2e}")
    if not (0 < rank < Pd.shape[0]):
        raise ValueError(f"P must be non-trivial: rank={rank}, d={Pd.shape[0]}")
    digest = projector_fingerprint(Pd)
    if locked:
        if metadata is None:
            raise ValueError("locked transport requires frozen projector metadata")
        required = {"sha256", "rank", "source_split", "source_role", "frozen"}
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(f"projector metadata missing {missing}")
        if not metadata["frozen"] or metadata["source_role"] != "discovery":
            raise ValueError("P must be frozen from the discovery split")
        if metadata["sha256"] != digest or int(metadata["rank"]) != rank:
            raise ValueError("P hash/rank differs from the sealed discovery projector")
    elif metadata is not None:
        if metadata.get("sha256") not in (None, digest):
            raise ValueError("P hash differs from supplied metadata")
    return digest, rank


def _patch_positions(native: Tensor, desired: Tensor, mode: str,
                     context_window: Optional[tuple[int, int]]) -> Tensor:
    if native.shape != desired.shape:
        raise ValueError(f"state shapes differ: {tuple(native.shape)} vs {tuple(desired.shape)}")
    out = native.clone()
    seq = out.shape[-2]
    a = out.reshape(-1, seq, out.shape[-1])
    b = desired.reshape(-1, seq, desired.shape[-1]).to(out.dtype).to(out.device)
    if mode == "target":
        a[:, -1] = b[:, -1]
    elif mode == "context":
        if seq > 1:
            a[:, :-1] = b[:, :-1]
    elif mode == "all":
        a[:] = b
    elif mode == "window":
        if context_window is None:
            raise ValueError("patch_mode='window' requires context_window=(start,end)")
        lo, hi = context_window
        lo = lo if lo >= 0 else seq + lo
        hi = hi if hi >= 0 else seq + hi
        if not (0 <= lo < hi <= seq):
            raise ValueError(f"invalid context window {context_window} for length {seq}")
        a[:, lo:hi] = b[:, lo:hi]
    else:
        raise ValueError(f"unknown patch mode {mode!r}")
    return out


def _carrier_content_shift(x: Tensor, x0: Tensor,
                           carrier_axis: Optional[Tensor]) -> tuple[float, float]:
    if carrier_axis is None:
        return float("nan"), float("nan")
    u = carrier_axis.double()
    u = u / u.norm().clamp_min(1e-30)
    d = (x.double() - x0.double()).reshape(-1, x.shape[-1])
    c = d @ u
    carrier = c.norm()
    content = (d - c[:, None] * u[None, :]).norm()
    scale = x0.double().norm().clamp_min(1e-30)
    return float(carrier / scale), float(content / scale)


def _assert_strictly_upstream(edits: Sequence[Edit], *, label: str) -> None:
    """Prevent a purported upstream intervention from editing x31/b31."""
    forbidden = {N.x(31), N.m(31), N.r(31), N.g(31), N.x(32),
                 N.FINAL_PRE_NORM}
    hit = sorted({e.tap for e in edits if e.tap in forbidden})
    if hit:
        raise ValueError(f"{label} must act strictly upstream of x31; forbidden taps={hit}")


@torch.no_grad()
def six_conditions(
    runner: Evo2Runner,
    pairs: Sequence[tuple[str, Tensor, Tensor]],    # (pair_id, recipient_ids, donor_or_upstream_ids)
    man: Manifest,
    *,
    P: Tensor,
    upstream_ablation: Callable[[Tensor], list[Edit]],
    controls: dict[str, Callable[[str], Tensor]],
    b31_modes: Sequence[str] = ("free", "freeze_all", "freeze_attn", "freeze_mlp"),
    ood_score: Callable[[Tensor], float] = lambda t: float("nan"),
    upstream_intervention: Optional[Callable[[Tensor], list[Edit]]] = None,
    projector_metadata: Optional[dict] = None,
    locked: bool = False,
    doses: Sequence[float] = (1.0, 0.25),
    patch_modes: Sequence[str] = ("target",),
    context_window: Optional[tuple[int, int]] = None,
    carrier_axis: Optional[Tensor] = None,
    include_natural_donor_generalization: bool = True,
    dependency_keys_of_pair: Optional[Callable[[str], tuple[str, ...]]] = None,
) -> list[TransportRow]:
    """Y0 / YU / YD / YB / YA / YR with a FIXED P, plus the b31 gate audit.

    Forbidden and not reachable through this API: P = I, a per-pair re-learned
    P, a pair-specific full Delta x31 as the mediator, and choosing the tap or
    rank after seeing the output.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold", "delta_path", "delta_spec")
    if upstream_intervention is None:
        raise ValueError(
            "YU must be produced by an intervention on the SAME recipient; "
            "a natural donor state is generalisation evidence, not the primary YU")
    missing_controls = sorted(set(CORE_CONTROL_FAMILIES) - set(controls))
    extra_controls = sorted(set(controls) - set(CORE_CONTROL_FAMILIES))
    if missing_controls or extra_controls:
        raise ValueError(f"transport controls incomplete; missing={missing_controls}, "
                         f"extra={extra_controls}")
    if not doses or 1.0 not in doses or any(not (0 < float(d) <= 1) for d in doses):
        raise ValueError("doses must lie in (0,1] and include the primary dose 1.0")
    if locked and not any(float(d) < 1 for d in doses):
        raise ValueError("locked transport requires at least one pre-registered small dose")
    bad_modes = sorted(set(b31_modes) - {"free", "freeze_all", "freeze_attn", "freeze_mlp"})
    if bad_modes:
        raise ValueError(f"unknown b31 modes: {bad_modes}")
    P_hash, _ = validate_frozen_projector(P, projector_metadata, locked=locked)
    rows: list[TransportRow] = []
    tap31 = N.x(31)

    def suffix_logits(x31: Tensor, ids: Tensor, b31: str, base_updates: dict,
                      patch_mode: str) -> Tensor:
        def fn(t: Tensor) -> Tensor:
            return _patch_positions(t, x31, patch_mode, context_window)
        edits = [Edit(tap31, "state", fn, "x31 compose")]
        if b31 == "freeze_all":
            edits += _freeze_edits("b31", base_updates)
        elif b31 == "freeze_attn":
            edits += _freeze_edits("b31_attn_only", base_updates)
        elif b31 == "freeze_mlp":
            edits += _freeze_edits("b31_mlp_only", base_updates)
        z = runner.run(ids, taps=(), edits=edits).logits
        return z.reshape(-1, z.shape[-1])[-1]

    for pid, rec_ids, up_ids in pairs:
        want = {tap31, N.m(31), N.g(31)}
        rec = runner.run(rec_ids, taps=want)
        up_edits = list(upstream_intervention(rec_ids))
        abl_edits = list(upstream_ablation(rec_ids))
        _assert_strictly_upstream(up_edits, label=f"{pid}:upstream_intervention")
        _assert_strictly_upstream(abl_edits, label=f"{pid}:upstream_ablation")
        dep_keys = ((dependency_keys_of_pair(pid) if dependency_keys_of_pair else ())
                    or (f"pair:{pid}",))
        up = runner.run(rec_ids, taps=want, edits=up_edits)
        donor = (runner.run(up_ids, taps=want)
                 if include_natural_donor_generalization else None)
        base_updates = {N.m(31): rec[N.m(31)], N.g(31): rec[N.g(31)]}

        x0, x1 = rec[tap31], up[tap31]
        M0, Q0 = _split(x0, P)
        abl = runner.run(rec_ids, taps=want, edits=abl_edits)

        for dose in sorted({float(d) for d in doses}, reverse=True):
            xd = x0 + dose * (x1 - x0)
            Md, Qd = _split(xd, P)
            xa = x0 + dose * (abl[tap31] - x0)
            MAd, QAd = _split(xa, P)
            for patch_mode in patch_modes:
                for b31 in b31_modes:
                    zY0 = suffix_logits(_compose(M0, Q0), rec_ids, b31,
                                        base_updates, patch_mode)
                    zYU = suffix_logits(_compose(Md, Qd), rec_ids, b31,
                                        base_updates, patch_mode)
                    ref_dir = ((zYU.double() - zYU.double().mean())
                               - (zY0.double() - zY0.double().mean()))
                    ref_n2 = ref_dir.norm().square().clamp_min(1e-30)

                    variants = {
                        "Y0": _compose(M0, Q0),
                        "YU": _compose(Md, Qd),
                        "YD": _compose(Md, Q0),
                        "YB": _compose(M0, Qd),
                        "YA": _compose(MAd, QAd),
                        "YR": _compose(Md, QAd),
                    }
                    for cond, x in variants.items():
                        z = suffix_logits(x, rec_ids, b31, base_updates, patch_mode)
                        r = solve_beta_shape(
                            z, zY0, beta_min=man.margins.beta_min,
                            beta_max=man.margins.beta_max,
                            flat_logit_threshold=man.margins.flat_logit_threshold)
                        dz = ((z.double() - z.double().mean())
                              - (zY0.double() - zY0.double().mean()))
                        cs, ns = _carrier_content_shift(x, x0, carrier_axis)
                        rows.append(TransportRow(
                            pid, cond, b31, "true", r.d_shape, r.log_beta,
                            float(torch.dot(dz, ref_dir) / ref_n2), ood_score(x),
                            dose=dose, patch_mode=patch_mode,
                            carrier_shift=cs, content_shift=ns,
                            projector_hash=P_hash, state_source="same_recipient",
                            dependency_keys=tuple(dep_keys)))

                    for fam in CORE_CONTROL_FAMILIES:
                        control_state = controls[fam](pid)
                        if control_state.shape != x0.shape:
                            raise ValueError(f"{pid}:{fam}: control state shape mismatch")
                        Mc_raw, _ = _split(control_state, P)
                        Mc = M0 + dose * (Mc_raw - M0)
                        for cond, x in (("YD", _compose(Mc, Q0)),
                                        ("YR", _compose(Mc, QAd))):
                            z = suffix_logits(x, rec_ids, b31, base_updates, patch_mode)
                            r = solve_beta_shape(
                                z, zY0, beta_min=man.margins.beta_min,
                                beta_max=man.margins.beta_max,
                                flat_logit_threshold=man.margins.flat_logit_threshold)
                            dz = ((z.double() - z.double().mean())
                                  - (zY0.double() - zY0.double().mean()))
                            cs, ns = _carrier_content_shift(x, x0, carrier_axis)
                            rows.append(TransportRow(
                                pid, cond, b31, fam, r.d_shape, r.log_beta,
                                float(torch.dot(dz, ref_dir) / ref_n2), ood_score(x),
                                dose=dose, patch_mode=patch_mode,
                                carrier_shift=cs, content_shift=ns,
                                projector_hash=P_hash, state_source="control",
                                dependency_keys=tuple(dep_keys)))

                    # A natural donor is deliberately outside the primary six
                    # states.  It assesses generalisation only and is excluded
                    # from the transport IUT.
                    if donor is not None:
                        MD_raw, _ = _split(donor[tap31], P)
                        MD = M0 + dose * (MD_raw - M0)
                        for cond, x in (("YD", _compose(MD, Q0)),
                                        ("YR", _compose(MD, QAd))):
                            z = suffix_logits(x, rec_ids, b31, base_updates, patch_mode)
                            r = solve_beta_shape(
                                z, zY0, beta_min=man.margins.beta_min,
                                beta_max=man.margins.beta_max,
                                flat_logit_threshold=man.margins.flat_logit_threshold)
                            dz = ((z.double() - z.double().mean())
                                  - (zY0.double() - zY0.double().mean()))
                            cs, ns = _carrier_content_shift(x, x0, carrier_axis)
                            rows.append(TransportRow(
                                pid, cond, b31, "natural_donor_generalization",
                                r.d_shape, r.log_beta,
                                float(torch.dot(dz, ref_dir) / ref_n2), ood_score(x),
                                dose=dose, patch_mode=patch_mode,
                                carrier_shift=cs, content_shift=ns,
                                projector_hash=P_hash, state_source="natural_donor",
                                dependency_keys=tuple(dep_keys)))
    return rows


def _pick(rows: Sequence[TransportRow], cond: str, med: str, b31: str,
          *, dose: float = 1.0, patch_mode: str = "target",
          attribute: str = "signed_following") -> list[Row]:
    """Backward-compatible selector with explicit dose/window filtering."""
    return [Row(float(getattr(r, attribute)),
                keys=(r.dependency_keys or (f"pair:{r.pair_id}",)))
            for r in rows if r.condition == cond and r.mediator == med
            and r.b31 == b31 and abs(r.dose - dose) < 1e-12
            and r.patch_mode == patch_mode]


def _pick_map(rows: Sequence[TransportRow], cond: str, med: str, b31: str,
              *, dose: float, patch_mode: str,
              attribute: str = "signed_following") -> dict[str, Row]:
    out = {}
    for r in rows:
        if (r.condition == cond and r.mediator == med and r.b31 == b31
                and abs(r.dose - dose) < 1e-12 and r.patch_mode == patch_mode):
            if r.pair_id in out:
                raise ValueError(f"duplicate transport cell for {r.pair_id}, {cond}, {med}, "
                                 f"{b31}, dose={dose}, patch={patch_mode}")
            out[r.pair_id] = Row(float(getattr(r, attribute)),
                                 keys=(r.dependency_keys
                                       or (f"pair:{r.pair_id}",)))
    return out


def _map_contrast(a: dict[str, Row], b: dict[str, Row], *, min_clusters: int,
                  n_boot: int, alpha: float = 0.05) -> Estimate:
    keys = sorted(set(a) & set(b))
    if not keys:
        raise RuntimeError("no aligned pairs for the transport contrast")
    return paired_contrast([a[k] for k in keys], [b[k] for k in keys],
                           min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)


def _difference_map(a: dict[str, Row], b: dict[str, Row]) -> dict[str, Row]:
    keys = sorted(set(a) & set(b))
    return {k: Row(a[k].value - b[k].value,
                   keys=tuple(set(a[k].keys) | set(b[k].keys))) for k in keys}


def transport_verdict(rows: Sequence[TransportRow], man: Manifest, *,
                      b31: str = "free", dose: float = 1.0,
                      patch_mode: str = "target",
                      small_dose: Optional[float] = None,
                      small_dose_margin: Optional[float] = None,
                      expected_channel: str = "content",
                      selectivity_margin: Optional[float] = None,
                      rescue_margin: Optional[float] = None,
                      ood_threshold: Optional[float] = None,
                      min_clusters: Optional[int] = None,
                      n_boot: int = 10_000,
                      alpha: float = 0.05) -> dict:
    """Step 9-5 intersection-union.

    Every control FAMILY must be beaten on its own margin; heterogeneous
    controls are never pooled, so a weak random-subspace control cannot mask a
    failed wrong-pair one. A recovery ratio, if reported at all, is descriptive
    and is NOT called a causal mediation fraction.
    """
    man.margins.require("delta_path", "delta_spec")
    if expected_channel not in {"carrier", "content"}:
        raise ValueError("expected_channel must be 'carrier' or 'content'")
    if min_clusters is None:
        min_clusters = required_cluster_count(man, "step9_transport")
    need = int(min_clusters)
    required_spec = set(CORE_CONTROL_FAMILIES) | {
        "transport_small_dose", "transport_selectivity",
        "transport_rescue", "transport_ood_max",
    }
    missing_spec = sorted(required_spec - set(man.margins.delta_spec))
    if missing_spec:
        raise RuntimeError(
            "Step 9 transport is missing development-calibrated delta_spec "
            f"entries: {missing_spec}"
        )
    if small_dose is None:
        raise RuntimeError(
            "Step 9 causal verdict requires the pre-registered small_dose; "
            "it is not selected from locked rows"
        )
    small_dose_margin = float(
        man.margins.delta_spec["transport_small_dose"]
        if small_dose_margin is None else small_dose_margin
    )
    selectivity_margin = float(
        man.margins.delta_spec["transport_selectivity"]
        if selectivity_margin is None else selectivity_margin
    )
    rescue_margin = float(
        man.margins.delta_spec["transport_rescue"]
        if rescue_margin is None else rescue_margin
    )
    ood_threshold = float(
        man.margins.delta_spec["transport_ood_max"]
        if ood_threshold is None else ood_threshold
    )
    if any(not np.isfinite(v) or v < 0 for v in
           (small_dose_margin, selectivity_margin, rescue_margin, ood_threshold)):
        raise ValueError("Step 9 margins and OOD threshold must be finite and non-negative")
    hashes = {r.projector_hash for r in rows if r.projector_hash}
    if len(hashes) != 1:
        raise ValueError(f"transport rows must share one frozen projector hash, got {hashes}")
    if any(r.state_source != "same_recipient" for r in rows
           if r.mediator == "true"):
        raise ValueError("primary Y0/YU/YD/YB/YA/YR rows must all come from the same recipient")
    present = set(controls_present(rows))
    missing = sorted(set(CORE_CONTROL_FAMILIES) - present)
    if missing:
        raise RuntimeError(f"missing mandatory transport controls: {missing}")

    Y0 = _pick_map(rows, "Y0", "true", b31, dose=dose, patch_mode=patch_mode)
    YU = _pick_map(rows, "YU", "true", b31, dose=dose, patch_mode=patch_mode)
    YD = _pick_map(rows, "YD", "true", b31, dose=dose, patch_mode=patch_mode)
    YB = _pick_map(rows, "YB", "true", b31, dose=dose, patch_mode=patch_mode)
    YA = _pick_map(rows, "YA", "true", b31, dose=dose, patch_mode=patch_mode)
    YR = _pick_map(rows, "YR", "true", b31, dose=dose, patch_mode=patch_mode)

    direct = _map_contrast(YD, Y0, min_clusters=need, n_boot=n_boot, alpha=alpha)
    block = _map_contrast(YU, YB, min_clusters=need, n_boot=n_boot, alpha=alpha)
    loss = _map_contrast(YU, YA, min_clusters=need, n_boot=n_boot, alpha=alpha)
    rescue = _map_contrast(YR, YA, min_clusters=need, n_boot=n_boot, alpha=alpha)

    spec = {}
    true_rescue = _difference_map(YR, YA)
    for fam in CORE_CONTROL_FAMILIES:
        cd = _pick_map(rows, "YD", fam, b31, dose=dose, patch_mode=patch_mode)
        cr = _pick_map(rows, "YR", fam, b31, dose=dose, patch_mode=patch_mode)
        ctrl_rescue = _difference_map(cr, YA)
        spec[f"direct::{fam}"] = (
            _map_contrast(YD, cd, min_clusters=need, n_boot=n_boot, alpha=alpha),
            float(man.margins.delta_spec[fam]))
        spec[f"rescue::{fam}"] = (
            _map_contrast(true_rescue, ctrl_rescue, min_clusters=need,
                          n_boot=n_boot, alpha=alpha),
            float(man.margins.delta_spec[fam]))
    iut = intersection_union(spec)

    # Carrier/content selectivity is evaluated on the same YD rows.  Both
    # values are state-change magnitudes normalised by ||x0||, so their
    # contrast has a coherent unit.
    intended_attr = "content_shift" if expected_channel == "content" else "carrier_shift"
    leakage_attr = "carrier_shift" if expected_channel == "content" else "content_shift"
    intended = _pick_map(rows, "YD", "true", b31, dose=dose,
                         patch_mode=patch_mode, attribute=intended_attr)
    leakage = _pick_map(rows, "YD", "true", b31, dose=dose,
                        patch_mode=patch_mode, attribute=leakage_attr)
    finite_keys = [k for k in sorted(set(intended) & set(leakage))
                   if np.isfinite(intended[k].value) and np.isfinite(leakage[k].value)]
    selectivity = None
    if finite_keys:
        selectivity = paired_contrast(
            [intended[k] for k in finite_keys], [leakage[k] for k in finite_keys],
            min_clusters=need, n_boot=n_boot, alpha=alpha)

    available_small = sorted({r.dose for r in rows if r.dose < dose})
    sd = float(small_dose)
    if not any(abs(d - sd) < 1e-12 for d in available_small):
        raise RuntimeError(
            f"pre-registered small_dose={sd} is absent from transport rows; "
            f"available={available_small}"
        )
    small = {}
    small_pass = False
    if sd is not None:
        s0 = _pick_map(rows, "Y0", "true", b31, dose=sd, patch_mode=patch_mode)
        su = _pick_map(rows, "YU", "true", b31, dose=sd, patch_mode=patch_mode)
        sd_ = _pick_map(rows, "YD", "true", b31, dose=sd, patch_mode=patch_mode)
        sb = _pick_map(rows, "YB", "true", b31, dose=sd, patch_mode=patch_mode)
        sa = _pick_map(rows, "YA", "true", b31, dose=sd, patch_mode=patch_mode)
        sr = _pick_map(rows, "YR", "true", b31, dose=sd, patch_mode=patch_mode)
        esd = _map_contrast(sd_, s0, min_clusters=need, n_boot=n_boot, alpha=alpha)
        esb = _map_contrast(su, sb, min_clusters=need, n_boot=n_boot, alpha=alpha)
        esr = _map_contrast(sr, sa, min_clusters=need, n_boot=n_boot, alpha=alpha)
        small = {"dose": sd, "direct": esd.as_row(), "block": esb.as_row(),
                 "rescue": esr.as_row()}
        small_pass = min(esd.lo, esb.lo, esr.lo) > small_dose_margin

    relevant = [r for r in rows if r.b31 == b31 and r.patch_mode == patch_mode
                and (abs(r.dose - dose) < 1e-12
                     or (sd is not None and abs(r.dose - sd) < 1e-12))
                and r.mediator != "natural_donor_generalization"]
    ood_values = [r.ood_score for r in relevant]
    ood_pass = (ood_threshold is not None and bool(ood_values)
                and all(np.isfinite(v) and v <= ood_threshold for v in ood_values))

    checks = {
        "1_direct_sufficiency": direct.lo > man.margins.delta_path,
        "2_block": block.lo > man.margins.delta_path,
        "3_ablation_loss_and_rescue": (
            loss.lo > man.margins.delta_path and rescue.lo > rescue_margin
        ),
        "4_specificity_IUT": bool(iut.passed),
        "5_carrier_content_selectivity": (selectivity is not None
                                           and selectivity.lo > selectivity_margin),
        "6_small_dose_and_in_distribution": small_pass and ood_pass,
    }
    aligned = sorted(set(Y0) & set(YU) & set(YD) & set(YB) & set(YA) & set(YR))
    per_unit = [{
        "pair_id": k, "state_source": "same_recipient",
        "dose": dose, "patch_mode": patch_mode,
        "direct_YD_minus_Y0": YD[k].value - Y0[k].value,
        "block_YU_minus_YB": YU[k].value - YB[k].value,
        "ablation_loss_YU_minus_YA": YU[k].value - YA[k].value,
        "rescue_YR_minus_YA": YR[k].value - YA[k].value,
        "intended_minus_leakage": (intended[k].value - leakage[k].value
                                    if k in intended and k in leakage else None),
    } for k in aligned]
    return {
        "estimates": {"YD-Y0": direct.as_row(), "YU-YB": block.as_row(),
                      "YU-YA": loss.as_row(), "YR-YA": rescue.as_row()},
        "specificity": {k: (v[0].as_row(), v[1]) for k, v in spec.items()},
        "iut_failures": iut.failures(),
        "selectivity": selectivity.as_row() if selectivity is not None else None,
        "small_dose": small,
        "ood": {"threshold": ood_threshold,
                "max": max(ood_values) if ood_values else None,
                "passed": ood_pass},
        "projector_hash": next(iter(hashes)),
        "per_unit": per_unit,
        "checks": checks,
        "claim": ("causal transport" if all(checks.values())
                  else "layerwise causal sensitivity"),
        "note": ("F includes b31 attention and MLP, so a pass means b30 -> b31 -> "
                 "readout transport. Run Step 9-3 before naming b30 the final mediator."),
    }


def controls_present(rows: Sequence[TransportRow]) -> list[str]:
    return sorted({r.mediator for r in rows
                   if r.mediator not in {"true", "natural_donor_generalization"}})


def b31_gate_audit(rows: Sequence[TransportRow], man: Manifest, *,
                   dose: float = 1.0, patch_mode: str = "target",
                   equivalence_margin: Optional[float] = None,
                   min_clusters: Optional[int] = None,
                   n_boot: int = 10_000,
                   alpha: float = 0.05) -> dict:
    """Paired b31 interaction and equivalence, never visual curve matching."""
    man.margins.require("delta_path", "delta_spec")
    need = (int(min_clusters) if min_clusters is not None
            else required_cluster_count(man, "step9_b31"))
    if equivalence_margin is None:
        if "b31_equivalence" not in man.margins.delta_spec:
            raise RuntimeError(
                "Step 9 b31 audit needs delta_spec['b31_equivalence']"
            )
        equivalence_margin = man.margins.delta_spec["b31_equivalence"]
    margin = float(equivalence_margin)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("b31 equivalence margin must be finite and positive")
    out, effects = {}, {}
    for mode in ("free", "freeze_all", "freeze_attn", "freeze_mlp"):
        y0 = _pick_map(rows, "Y0", "true", mode, dose=dose, patch_mode=patch_mode)
        yd = _pick_map(rows, "YD", "true", mode, dose=dose, patch_mode=patch_mode)
        yu = _pick_map(rows, "YU", "true", mode, dose=dose, patch_mode=patch_mode)
        yb = _pick_map(rows, "YB", "true", mode, dose=dose, patch_mode=patch_mode)
        if not y0 or not yd or not yu or not yb:
            continue
        direct_map = _difference_map(yd, y0)
        block_map = _difference_map(yu, yb)
        direct = _map_contrast(yd, y0, min_clusters=need, n_boot=n_boot, alpha=alpha)
        block = _map_contrast(yu, yb, min_clusters=need, n_boot=n_boot, alpha=alpha)
        out[mode] = {"direct": direct.as_row(), "block": block.as_row()}
        effects[mode] = {"direct": direct_map, "block": block_map}

    required_modes = {"free", "freeze_all", "freeze_attn", "freeze_mlp"}
    missing_modes = sorted(required_modes - set(out))
    if missing_modes:
        raise RuntimeError(f"b31 audit is missing required paired modes: {missing_modes}")

    interactions = {}
    interaction_alpha = alpha / 6.0
    if "free" in effects:
        for mode in ("freeze_all", "freeze_attn", "freeze_mlp"):
            if mode not in effects:
                continue
            interactions[mode] = {}
            for endpoint in ("direct", "block"):
                est = _map_contrast(effects["free"][endpoint], effects[mode][endpoint],
                                    min_clusters=need, n_boot=n_boot,
                                    alpha=interaction_alpha)
                interactions[mode][endpoint] = {
                    "free_minus_frozen": est.as_row(),
                    "equivalent": est.lo > -margin and est.hi < margin,
                    "positive_interaction": est.lo > margin,
                }
    reading = {
        "free ~= freeze_all": "b31 is largely a transparent carrier",
        "changes under freeze_attn only": "b31 attention does active routing",
        "changes under freeze_mlp only": "b31 MLP does an active readout transform",
        "block/rescue only with b31 free": "name the whole chain a b30->b31 mechanism",
    }
    transparent = all(
        interactions["freeze_all"][endpoint]["equivalent"]
        for endpoint in ("direct", "block")
    )
    return {"by_mode": out, "paired_interactions": interactions,
            "equivalence_margin": margin,
            "transparent_b31_for_path": transparent,
            "multiplicity": {
                "method": "Bonferroni over 3 freeze modes x 2 endpoints",
                "family_alpha": alpha, "per_contrast_alpha": interaction_alpha,
            },
            "reading": reading}


@dataclass
class LayerRole2x2Row:
    locus: str
    condition: str                # native | b28_block | b30_block | both_block | rescues
    b28_blocked: bool
    b30_blocked: bool
    d_shape: float
    log_beta: float
    x29_signal: float
    x31_signal: float
    b28_dose: float = 0.0
    b30_dose: float = 0.0
    doses_matched: bool = False
    P29_hash: str = ""
    P31_hash: str = ""


def _projected_strength(x: Tensor, P: Tensor) -> float:
    xx = x.double().reshape(-1, x.shape[-1])
    pp = P.double().to(xx.device)
    return float((xx @ pp.T).norm() / xx.norm().clamp_min(1e-30))


@torch.no_grad()
def b28_b30_matched_2x2(
    runner: Evo2Runner,
    loci: Sequence[tuple[str, Tensor]],
    man: Manifest,
    *,
    b28_block: Callable[[Tensor], list[Edit]],
    b30_block: Callable[[Tensor], list[Edit]],
    P29: Tensor,
    P31: Tensor,
    P29_metadata: Optional[dict] = None,
    P31_metadata: Optional[dict] = None,
    b28_rescue: Optional[Callable[[Tensor], list[Edit]]] = None,
    b30_rescue: Optional[Callable[[Tensor], list[Edit]]] = None,
    b28_tap: str = "g28",
    b30_tap: str = "m30",
    dose_tolerance: Optional[float] = None,
    locked: bool = True,
) -> list[LayerRole2x2Row]:
    """Direct b28-block x b30-block factorial on one endpoint and locus set.

    Intervention dose is the relative change at the edited tap, computed
    against the matching upstream background (10 vs 00 for b28; 11 vs 10 for
    b30).  This avoids calling two raw tensor norms at different layers
    "matched".  The primary main-effect cells and both natural-component
    rescue cells are returned together; a locked run refuses missing rescues
    or an unmatched normalised dose.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    h29, _ = validate_frozen_projector(P29, P29_metadata, locked=locked)
    h31, _ = validate_frozen_projector(P31, P31_metadata, locked=locked)
    if locked and (b28_rescue is None or b30_rescue is None):
        raise ValueError("locked b28xb30 factorial requires both rescue factories")
    if dose_tolerance is None:
        if "b28_b30_dose_match" not in man.margins.delta_spec:
            raise RuntimeError(
                "locked b28xb30 factorial needs development-calibrated "
                "delta_spec['b28_b30_dose_match']"
            )
        dose_tolerance = float(man.margins.delta_spec["b28_b30_dose_match"])
    if not np.isfinite(dose_tolerance) or not (0 < dose_tolerance <= 1):
        raise ValueError("dose_tolerance must be a finite fraction in (0,1]")
    taps = {N.x(29), N.x(31), b28_tap, b30_tap}
    rows: list[LayerRole2x2Row] = []

    for locus, ids in loci:
        e28, e30 = list(b28_block(ids)), list(b30_block(ids))
        conditions: list[tuple[str, bool, bool, list[Edit]]] = [
            ("native", False, False, []),
            ("b28_block", True, False, e28),
            ("b30_block", False, True, e30),
            ("both_block", True, True, e28 + e30),
        ]
        if b28_rescue is not None:
            conditions.append(("b28_rescue", False, False, list(b28_rescue(ids))))
        if b30_rescue is not None:
            conditions.append(("b30_rescue", False, False, list(b30_rescue(ids))))
        traces = {name: runner.run(ids, taps=taps, edits=edits)
                  for name, _, _, edits in conditions}
        z0 = traces["native"].logits.reshape(-1, traces["native"].logits.shape[-1])[-1]

        def relative_delta(a: Tensor, b: Tensor) -> float:
            return float((a.double() - b.double()).norm()
                         / b.double().norm().clamp_min(1e-30))

        d28_10 = relative_delta(traces["b28_block"][b28_tap], traces["native"][b28_tap])
        d28_11 = relative_delta(traces["both_block"][b28_tap], traces["b30_block"][b28_tap])
        d30_01 = relative_delta(traces["b30_block"][b30_tap], traces["native"][b30_tap])
        d30_11 = relative_delta(traces["both_block"][b30_tap], traces["b28_block"][b30_tap])
        dose28 = 0.5 * (d28_10 + d28_11)
        dose30 = 0.5 * (d30_01 + d30_11)
        matched = abs(dose28 - dose30) <= dose_tolerance * max(dose28, dose30, 1e-30)
        if locked and not matched:
            raise RuntimeError(
                f"{locus}: b28/b30 intervention doses are not matched "
                f"({dose28:.4g} vs {dose30:.4g}, tolerance={dose_tolerance:.3g})")

        flags = {name: (b28_on, b30_on) for name, b28_on, b30_on, _ in conditions}
        for name, ts in traces.items():
            z = ts.logits.reshape(-1, ts.logits.shape[-1])[-1]
            shape = solve_beta_shape(
                z, z0, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold)
            b28_on, b30_on = flags[name]
            rows.append(LayerRole2x2Row(
                locus=locus, condition=name, b28_blocked=b28_on,
                b30_blocked=b30_on, d_shape=shape.d_shape,
                log_beta=shape.log_beta,
                x29_signal=_projected_strength(ts[N.x(29)], P29),
                x31_signal=_projected_strength(ts[N.x(31)], P31),
                b28_dose=dose28, b30_dose=dose30,
                doses_matched=matched, P29_hash=h29, P31_hash=h31))
    return rows


def b28_b30_role_verdict(
    rows: Sequence[LayerRole2x2Row],
    *,
    signal_margin: Optional[float] = None,
    output_margin: Optional[float] = None,
    equivalence_margin: Optional[float] = None,
    rescue_margin: Optional[float] = None,
    man: Optional[Manifest] = None,
    min_clusters: Optional[int] = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    require_projector_hashes: bool = True,
) -> dict:
    """Double-dissociation verdict for ``b28 writer -> b30 re-encoder``."""
    margin_names = {
        "signal_margin": "b28_b30_signal",
        "output_margin": "b28_b30_output",
        "equivalence_margin": "b28_b30_equivalence",
        "rescue_margin": "b28_b30_rescue",
    }
    values = {
        "signal_margin": signal_margin,
        "output_margin": output_margin,
        "equivalence_margin": equivalence_margin,
        "rescue_margin": rescue_margin,
    }
    if man is not None:
        man.margins.require("delta_spec")
    resolved = {}
    for arg, key in margin_names.items():
        value = values[arg]
        if value is None and man is not None:
            value = man.margins.delta_spec.get(key)
        if value is None:
            raise RuntimeError(
                f"b28xb30 role verdict needs development-calibrated {arg} "
                f"(delta_spec[{key!r}])"
            )
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{arg} must be finite and non-negative")
        resolved[arg] = value
    signal_margin = resolved["signal_margin"]
    output_margin = resolved["output_margin"]
    equivalence_margin = resolved["equivalence_margin"]
    rescue_margin = resolved["rescue_margin"]
    if equivalence_margin <= 0:
        raise ValueError("equivalence_margin must be positive")
    if min_clusters is None:
        if man is None:
            raise RuntimeError(
                "b28xb30 role verdict needs min_clusters explicitly or calibrated "
                "manifest power key 'step9_b28_b30'"
            )
        min_clusters = required_cluster_count(man, "step9_b28_b30")
    by_cond: dict[str, dict[str, LayerRole2x2Row]] = {}
    for r in rows:
        if r.locus in by_cond.setdefault(r.condition, {}):
            raise ValueError(f"duplicate 2x2 row {r.condition}:{r.locus}")
        by_cond[r.condition][r.locus] = r
    required = {"native", "b28_block", "b30_block", "both_block",
                "b28_rescue", "b30_rescue"}
    missing = sorted(required - set(by_cond))
    if missing:
        raise RuntimeError(f"2x2/rescue panel incomplete: {missing}")
    h29 = {r.P29_hash for r in rows if r.P29_hash}
    h31 = {r.P31_hash for r in rows if r.P31_hash}
    if len(h29) > 1 or len(h31) > 1:
        raise ValueError("2x2 rows mix projector versions")
    if require_projector_hashes and (len(h29) != 1 or len(h31) != 1):
        raise ValueError("2x2 causal verdict requires one frozen P29 and P31 hash")

    def attr(cond: str, name: str) -> dict[str, Row]:
        return {l: Row(float(getattr(r, name)), keys=(f"locus:{l}",))
                for l, r in by_cond[cond].items()}

    native29 = attr("native", "x29_signal")
    block28_29 = attr("b28_block", "x29_signal")
    block30_29 = attr("b30_block", "x29_signal")
    native31 = attr("native", "x31_signal")
    block30_31 = attr("b30_block", "x31_signal")
    z = {l: Row(0.0, keys=v.keys) for l, v in attr("native", "d_shape").items()}
    out28 = attr("b28_block", "d_shape")
    out30 = attr("b30_block", "d_shape")
    out_both = attr("both_block", "d_shape")
    out_r28 = attr("b28_rescue", "d_shape")
    out_r30 = attr("b30_rescue", "d_shape")

    e_writer = _map_contrast(native29, block28_29, min_clusters=min_clusters,
                             n_boot=n_boot, alpha=alpha)
    e_upstream_preserved = _map_contrast(block30_29, native29,
                                         min_clusters=min_clusters,
                                         n_boot=n_boot, alpha=alpha)
    e_reencode = _map_contrast(native31, block30_31, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    e_out28 = _map_contrast(out28, z, min_clusters=min_clusters,
                           n_boot=n_boot, alpha=alpha)
    e_out30 = _map_contrast(out30, z, min_clusters=min_clusters,
                           n_boot=n_boot, alpha=alpha)
    e_r28 = _map_contrast(out28, out_r28, min_clusters=min_clusters,
                         n_boot=n_boot, alpha=alpha)
    e_r30 = _map_contrast(out30, out_r30, min_clusters=min_clusters,
                         n_boot=n_boot, alpha=alpha)

    # Difference-in-differences on the common D_shape endpoint.  It is
    # reported rather than required: serial mechanisms can be additive or
    # interactive without changing the writer/re-encoder classification.
    common = sorted(set(z) & set(out28) & set(out30) & set(out_both))
    if not common:
        raise RuntimeError("2x2 cells have no common loci")
    interaction_rows = [Row(out_both[k].value - out28[k].value
                            - out30[k].value + z[k].value,
                            keys=(f"locus:{k}",)) for k in common]
    interaction = cluster_bootstrap(interaction_rows, min_clusters=min_clusters,
                                    n_boot=n_boot, alpha=alpha)
    dose_ok = all(r.doses_matched for r in rows)
    checks = {
        "b28_writes_x29_mode": e_writer.lo > signal_margin,
        "b30_preserves_upstream_x29": (e_upstream_preserved.lo > -equivalence_margin
                                       and e_upstream_preserved.hi < equivalence_margin),
        "b30_changes_x31_mode": e_reencode.lo > signal_margin,
        "both_blocks_affect_common_output": (e_out28.lo > output_margin
                                             and e_out30.lo > output_margin),
        "natural_component_rescues": (e_r28.lo > rescue_margin
                                      and e_r30.lo > rescue_margin),
        "matched_intervention_dose": dose_ok,
    }
    common_units = sorted(set(by_cond["native"]) & set(by_cond["b28_block"])
                          & set(by_cond["b30_block"]) & set(by_cond["both_block"])
                          & set(by_cond["b28_rescue"]) & set(by_cond["b30_rescue"]))
    per_unit = [{
        "locus": k,
        "b28_x29_loss": (by_cond["native"][k].x29_signal
                          - by_cond["b28_block"][k].x29_signal),
        "b30_x29_change": (by_cond["b30_block"][k].x29_signal
                            - by_cond["native"][k].x29_signal),
        "b30_x31_loss": (by_cond["native"][k].x31_signal
                          - by_cond["b30_block"][k].x31_signal),
        "b28_output_damage": by_cond["b28_block"][k].d_shape,
        "b30_output_damage": by_cond["b30_block"][k].d_shape,
        "b28_rescue_gain": (by_cond["b28_block"][k].d_shape
                             - by_cond["b28_rescue"][k].d_shape),
        "b30_rescue_gain": (by_cond["b30_block"][k].d_shape
                             - by_cond["b30_rescue"][k].d_shape),
        "b28_dose": by_cond["native"][k].b28_dose,
        "b30_dose": by_cond["native"][k].b30_dose,
    } for k in common_units]
    return {
        "checks": checks,
        "estimates": {
            "native_x29_minus_b28_block": e_writer.as_row(),
            "b30_block_x29_minus_native": e_upstream_preserved.as_row(),
            "native_x31_minus_b30_block": e_reencode.as_row(),
            "b28_block_output": e_out28.as_row(),
            "b30_block_output": e_out30.as_row(),
            "b28_rescue": e_r28.as_row(),
            "b30_rescue": e_r30.as_row(),
            "block_interaction": interaction.as_row(),
        },
        "per_unit": per_unit,
        "projector_hashes": {"P29": next(iter(h29)) if h29 else None,
                             "P31": next(iter(h31)) if h31 else None},
        "role": ("b28 writer -> b30 re-encoder/integrator"
                 if all(checks.values()) else "roles not causally identified"),
        "note": "b31 activity is classified separately by b31_gate_audit",
    }


def role_from_transport(same_subspace: bool, gain_only: bool, stable_new_subspace: bool,
                        mediator_lost_on_upstream_ablation: bool,
                        only_full_state_works: bool, wrong_donor_equal: bool) -> str:
    if wrong_donor_equal:
        return "generic state-energy effect"
    if only_full_state_works:
        return "partial mechanism not established"
    if mediator_lost_on_upstream_ablation:
        return "writer -> mediator chain"
    if same_subspace and gain_only:
        return "amplifier"
    if stable_new_subspace:
        return "re-encoder"
    return "undetermined"

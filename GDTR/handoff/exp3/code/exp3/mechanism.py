"""EXP3 Phase 1 -- resolve b29's subtype and the scale boundary.

This module accepts EXP2's b28-writer/b30-reencoder logic as established and
consumes its frozen runner/tap/endpoint contracts without re-running or
copying EXP2 experiments.  It answers three narrower extension questions.

1. Does ``m28`` matter mainly because it prepares the input on which ``g28``
   is written?  The diagnostic is ``m28=0`` followed by an exact ``g28``
   clamp and, when available, a held-out predicted-delta rescue.
2. Does b29 merely grow the b28 mode, or does it write a new orthogonal
   component?  A frozen b28 basis separates the *response* of b29 into
   parallel and perpendicular parts.  Each part is then blocked on the same
   recipient; held-out predicted parts are tested for sufficiency.
3. Do radial interventions share a host-normalized transition and sustained
   plateau across the preregistered late-stack sites/models, while equal-norm
   angular interventions remain output-distinct?

Large norms never determine either label.  The labels are returned only by
the corresponding verdict functions, which use paired dependency-cluster
intervals and explicit equivalence margins.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .artifacts import ArtifactRef, ArtifactStore

from .exp2_api import (
    N, Edit, Evo2Runner, Manifest, Row, branch_geometry, cluster_bootstrap,
    paired_contrast, replace_with, solve_beta_shape,
)


def _last(x: Tensor) -> Tensor:
    """Last sequence position without assuming a batch rank."""
    return x.reshape(-1, x.shape[-1])[-1]


def _centre(x: Tensor) -> Tensor:
    x = x.double()
    return x - x.mean()


def _cosine(a: Tensor, b: Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _relative_delta(x: Tensor, x0: Tensor) -> float:
    return float((x.double() - x0.double()).norm()
                 / x0.double().norm().clamp_min(1e-30))


def _row_keys(locus: str, metadata: Mapping | None = None) -> tuple[str, ...]:
    keys = tuple((metadata or {}).get("keys", ()))
    return keys or (f"locus:{locus}",)


def _unpack_locus(item) -> tuple[str, Tensor, Optional[int], dict]:
    if len(item) == 2:
        locus, ids = item
        return str(locus), ids, None, {}
    if len(item) == 3:
        locus, ids, third = item
        if isinstance(third, Mapping):
            return str(locus), ids, None, dict(third)
        return str(locus), ids, third, {}
    if len(item) == 4:
        locus, ids, true_token, metadata = item
        return str(locus), ids, true_token, dict(metadata)
    raise ValueError("locus entries must be (locus, ids[, true_token][, metadata])")


# ---------------------------------------------------------------------------
# Frozen subspace helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSubspace:
    """A compact orthonormal basis, rather than a dense d x d projector."""

    basis: Tensor                    # [d, k], orthonormal columns
    sha256: str
    source_split: str
    source_role: str = "discovery"
    frozen: bool = True


def _tensor_hash(x: Tensor) -> str:
    raw = x.detach().double().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def freeze_subspace(
    basis: Tensor,
    *,
    source_split: str,
    source_role: str = "discovery",
    tolerance: float = 1e-7,
) -> FrozenSubspace:
    """Validate and freeze a non-trivial ``[d,k]`` orthonormal basis.

    A square idempotent projector is also accepted and converted to its
    positive-eigenvalue basis.  This keeps the public Step-9 projector format
    interoperable without allocating another dense projector in new code.
    """
    B = basis.detach().double()
    if B.ndim != 2 or min(B.shape) == 0 or not torch.isfinite(B).all():
        raise ValueError("basis must be a finite, non-empty matrix")
    if B.shape[0] == B.shape[1]:
        sym = float((B - B.T).norm())
        idem = float((B @ B - B).norm())
        if sym < tolerance and idem < 10 * tolerance:
            ev, U = torch.linalg.eigh(B)
            B = U[:, ev > 0.5]
    gram = B.T @ B
    if float((gram - torch.eye(gram.shape[0], dtype=gram.dtype)).norm()) >= 10 * tolerance:
        raise ValueError("basis columns must be orthonormal (QR it in discovery)")
    if not (0 < B.shape[1] < B.shape[0]):
        raise ValueError(f"subspace must be non-trivial: rank={B.shape[1]}, d={B.shape[0]}")
    if source_role != "discovery":
        raise ValueError("causal subspaces must be selected on the discovery split")
    return FrozenSubspace(B, _tensor_hash(B), str(source_split), source_role, True)


def validate_subspace(space: FrozenSubspace, width: int, *, locked: bool) -> Tensor:
    if not isinstance(space, FrozenSubspace):
        raise TypeError("expected a FrozenSubspace; call freeze_subspace in discovery")
    B = space.basis.detach().double()
    if B.shape[0] != width:
        raise ValueError(f"subspace width {B.shape[0]} != activation width {width}")
    if _tensor_hash(B) != space.sha256:
        raise RuntimeError("frozen subspace tensor changed after sealing")
    if locked and (not space.frozen or space.source_role != "discovery"):
        raise RuntimeError("locked analysis requires a discovery-frozen subspace")
    gram = B.T @ B
    if float((gram - torch.eye(B.shape[1], dtype=B.dtype)).norm()) >= 1e-6:
        raise RuntimeError("frozen subspace is no longer orthonormal")
    return B


def project_parallel(x: Tensor, basis: Tensor) -> Tensor:
    """Project the final feature axis onto a compact orthonormal basis."""
    B = basis.to(x.dtype).to(x.device)
    if x.shape[-1] != B.shape[0]:
        raise ValueError(f"activation width {x.shape[-1]} != basis width {B.shape[0]}")
    return (x @ B) @ B.T


def parallel_perpendicular(x: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    par = project_parallel(x, basis)
    return par, x - par


# ---------------------------------------------------------------------------
# m28 preconditioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionLineage:
    """Bind a prospective activation delta to a frozen train/dev model."""

    model_artifact: ArtifactRef
    method: str
    train_source_sha256: str
    dev_source_sha256: str
    train_unit_order_sha256: str
    dev_unit_order_sha256: str
    locked_input_sha256: str
    locked_unit_id: str
    prediction_sha256: str

    def validate(self, values: Tensor, *, method: str, locked_input: Tensor,
                 locked_unit_id: str) -> None:
        ArtifactStore.verify(self.model_artifact)
        if self.model_artifact.source_role not in {"discovery", "development"}:
            raise RuntimeError("prediction model was not frozen before locked use")
        metadata = self.model_artifact.metadata
        required_metadata = {
            "fit_role": "train_dev_cross_fitted",
            "train_source_sha256": self.train_source_sha256,
            "dev_source_sha256": self.dev_source_sha256,
            "train_unit_order_sha256": self.train_unit_order_sha256,
            "dev_unit_order_sha256": self.dev_unit_order_sha256,
        }
        if any(metadata.get(key) != value
               for key, value in required_metadata.items()):
            raise RuntimeError("prediction model artifact lacks cross-fit provenance")
        for value in (
            self.train_source_sha256, self.dev_source_sha256,
            self.train_unit_order_sha256, self.dev_unit_order_sha256,
            self.locked_input_sha256, self.prediction_sha256,
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("prediction lineage hashes must be lowercase SHA-256")
        if method != self.method or locked_unit_id != self.locked_unit_id:
            raise RuntimeError("prediction method or ordered locked unit changed")
        if _tensor_hash(locked_input) != self.locked_input_sha256:
            raise RuntimeError("locked prediction input changed after commitment")
        expected = hashlib.sha256("\x1f".join((
            _tensor_hash(values), self.model_artifact.sha256, self.method,
            self.locked_input_sha256, self.locked_unit_id,
        )).encode("utf-8")).hexdigest()
        if expected != self.prediction_sha256:
            raise RuntimeError("predicted delta is not bound to the frozen model/input")


def seal_prediction_lineage(
    values: Tensor,
    *,
    model_artifact: ArtifactRef,
    method: str,
    locked_input: Tensor,
    locked_unit_id: str,
) -> PredictionLineage:
    metadata = model_artifact.metadata
    required = (
        "train_source_sha256", "dev_source_sha256",
        "train_unit_order_sha256", "dev_unit_order_sha256",
    )
    if metadata.get("fit_role") != "train_dev_cross_fitted" \
            or any(key not in metadata for key in required):
        raise ValueError("model artifact is not a sealed train/dev cross-fit")
    input_hash = _tensor_hash(locked_input)
    prediction_hash = hashlib.sha256("\x1f".join((
        _tensor_hash(values), model_artifact.sha256, method,
        input_hash, locked_unit_id,
    )).encode("utf-8")).hexdigest()
    lineage = PredictionLineage(
        model_artifact=model_artifact, method=method,
        train_source_sha256=str(metadata["train_source_sha256"]),
        dev_source_sha256=str(metadata["dev_source_sha256"]),
        train_unit_order_sha256=str(metadata["train_unit_order_sha256"]),
        dev_unit_order_sha256=str(metadata["dev_unit_order_sha256"]),
        locked_input_sha256=input_hash, locked_unit_id=locked_unit_id,
        prediction_sha256=prediction_hash,
    )
    lineage.validate(
        values, method=method, locked_input=locked_input,
        locked_unit_id=locked_unit_id)
    return lineage


@dataclass(frozen=True)
class DeltaPrediction:
    """A rescue delta with enough provenance to reject target leakage."""

    delta: Tensor
    method: str
    source_split: str
    held_out: bool
    target_derived: bool = False
    artifact_sha256: str = ""
    source_role: str = "development"
    lineage: Optional[PredictionLineage] = None

    def validate(self, reference: Tensor, *, locked: bool,
                 locked_input: Optional[Tensor] = None,
                 locked_unit_id: str = "") -> Tensor:
        d = self.delta
        if d.shape != reference.shape:
            raise ValueError(
                f"predicted delta shape {tuple(d.shape)} != target {tuple(reference.shape)}")
        if not torch.isfinite(d).all():
            raise ValueError("predicted delta contains non-finite values")
        if not self.method or not self.source_split:
            raise ValueError("predicted rescue requires method and source_split provenance")
        if self.source_role not in {"discovery", "development"}:
            raise ValueError("prediction artifact must originate in discovery/development")
        digest = _tensor_hash(d)
        if self.artifact_sha256 and self.artifact_sha256 != digest:
            raise RuntimeError("predicted delta differs from its recorded hash")
        if locked and (not self.held_out or self.target_derived):
            raise RuntimeError(
                "locked rescue must be held-out and may not use the target delta")
        if locked:
            if self.lineage is None or locked_input is None or not locked_unit_id:
                raise RuntimeError(
                    "locked rescue requires fitted-model and locked-input lineage")
            self.lineage.validate(
                d, method=self.method, locked_input=locked_input,
                locked_unit_id=locked_unit_id)
        return d


@dataclass
class PreconditioningRow:
    locus: str
    condition: str
    d_shape: float
    log_beta: float
    nll: Optional[float]
    g28_norm_ratio: float
    g28_cos_native: float
    g28_relative_delta: float
    factor_a_relative_delta: float
    factor_b_relative_delta: float
    product_relative_delta: float
    rescue_method: str = ""
    rescue_held_out: bool = False
    rescue_target_derived: bool = False
    dependency_keys: tuple[str, ...] = ()


@torch.no_grad()
def m28_preconditioning_panel(
    runner: Evo2Runner,
    loci: Sequence[tuple],
    man: Manifest,
    *,
    prediction: Optional[Callable[[str, Tensor], DeltaPrediction]] = None,
    locked: bool = False,
) -> list[PreconditioningRow]:
    """Run native / m28-zero / g28-clamp / predicted-delta rescue.

    The exact clamp is a path-isolation diagnostic: if it repairs the output,
    the damage caused by deleting m28 travelled primarily through g28.  It is
    *not* counted as held-out sufficiency.  The latter requires ``prediction``
    to return a :class:`DeltaPrediction` that passes the locked provenance
    checks.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    taps = {N.m(28), N.r(28), N.mlp_part(28, "z"), N.g(28), N.x(29)}
    rows: list[PreconditioningRow] = []

    for item in loci:
        locus, ids, true_token, md = _unpack_locus(item)
        native = runner.run(ids, taps=taps)
        z0 = _last(native.logits)
        zero_m = Edit(N.m(28), "update", lambda t: torch.zeros_like(t), "m28=0")
        zero = runner.run(ids, taps=taps, edits=[zero_m])
        lost = native[N.g(28)] - zero[N.g(28)]

        conditions: list[tuple[str, object, str, bool, bool]] = [
            ("native", native, "", False, False),
            ("m28_zero_recompute", zero, "", False, False),
        ]
        clamp = runner.run(
            ids, taps=taps,
            edits=[zero_m, Edit(N.g(28), "update", replace_with(native[N.g(28)]),
                                "m28=0; clamp natural g28")])
        conditions.append(("m28_zero_g28_clamp", clamp, "exact_clamp", False, True))

        if prediction is not None:
            pred = prediction(locus, ids)
            delta = pred.validate(
                zero[N.g(28)], locked=locked, locked_input=ids,
                locked_unit_id=locus)
            rescue = runner.run(
                ids, taps=taps,
                edits=[zero_m, Edit(
                    N.g(28), "update",
                    lambda t, d=delta: t + d.to(t.dtype).to(t.device),
                    f"predicted lost-g28 rescue:{pred.method}")])
            conditions.append(("m28_zero_g28_predicted_rescue", rescue, pred.method,
                               pred.held_out, pred.target_derived))
        elif locked:
            raise ValueError("locked m28 preconditioning requires a held-out delta predictor")

        native_parts = runner.bilinear_parts(28, native[N.mlp_part(28, "z")])
        for cond, trace, method, held_out, target_derived in conditions:
            logits = _last(trace.logits)
            shape = solve_beta_shape(
                logits, z0, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold)
            nll = (None if true_token is None
                   else float(-torch.log_softmax(logits.double(), -1)[int(true_token)]))
            parts = runner.bilinear_parts(28, trace[N.mlp_part(28, "z")])
            rows.append(PreconditioningRow(
                locus=locus, condition=cond, d_shape=shape.d_shape,
                log_beta=shape.log_beta, nll=nll,
                g28_norm_ratio=float(trace[N.g(28)].double().norm()
                                     / native[N.g(28)].double().norm().clamp_min(1e-30)),
                g28_cos_native=_cosine(trace[N.g(28)], native[N.g(28)]),
                g28_relative_delta=_relative_delta(trace[N.g(28)], native[N.g(28)]),
                factor_a_relative_delta=_relative_delta(parts["a"], native_parts["a"]),
                factor_b_relative_delta=_relative_delta(parts["b"], native_parts["b"]),
                product_relative_delta=_relative_delta(parts["p"], native_parts["p"]),
                rescue_method=method, rescue_held_out=held_out,
                rescue_target_derived=target_derived,
                dependency_keys=_row_keys(locus, md)))
    return rows


def _condition_map(rows, condition: str, attr: str) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for r in rows:
        if r.condition != condition:
            continue
        if r.locus in out:
            raise ValueError(f"duplicate row for {condition}:{r.locus}")
        out[r.locus] = Row(float(getattr(r, attr)), keys=(r.dependency_keys
                                                          or (f"locus:{r.locus}",)))
    return out


def _paired_maps(a: Mapping[str, Row], b: Mapping[str, Row], *, min_clusters: int,
                 n_boot: int, alpha: float):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise RuntimeError("no aligned loci for paired causal contrast")
    return paired_contrast([a[k] for k in keys], [b[k] for k in keys],
                           min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)


def _equivalent_to_zero(values: Mapping[str, Row], *, margin: float,
                        min_clusters: int, n_boot: int, alpha: float):
    zero = {k: Row(0.0, keys=v.keys) for k, v in values.items()}
    est = _paired_maps(values, zero, min_clusters=min_clusters,
                       n_boot=n_boot, alpha=alpha)
    return est, est.lo > -margin and est.hi < margin


def m28_preconditioning_verdict(
    rows: Sequence[PreconditioningRow],
    *,
    damage_margin: float,
    rescue_margin: float,
    equivalence_margin: float,
    product_change_margin: float,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    require_predicted_rescue: bool = True,
) -> dict:
    """Conjunctive verdict; a failed clause downgrades rather than disappears."""
    for name, value in {
        "damage_margin": damage_margin, "rescue_margin": rescue_margin,
        "equivalence_margin": equivalence_margin,
        "product_change_margin": product_change_margin,
    }.items():
        if not np.isfinite(value) or value < 0 or (name == "equivalence_margin" and value == 0):
            raise ValueError(f"{name} must be finite and positive/non-negative as appropriate")
    zero = _condition_map(rows, "m28_zero_recompute", "d_shape")
    clamp = _condition_map(rows, "m28_zero_g28_clamp", "d_shape")
    native = {k: Row(0.0, keys=v.keys) for k, v in zero.items()}
    e_damage = _paired_maps(zero, native, min_clusters=min_clusters,
                            n_boot=n_boot, alpha=alpha)
    e_clamp_recovery = _paired_maps(zero, clamp, min_clusters=min_clusters,
                                    n_boot=n_boot, alpha=alpha)
    e_clamp_null, clamp_eq = _equivalent_to_zero(
        clamp, margin=equivalence_margin, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    product = _condition_map(rows, "m28_zero_recompute", "product_relative_delta")
    e_product = _paired_maps(product, native, min_clusters=min_clusters,
                             n_boot=n_boot, alpha=alpha)

    predicted = _condition_map(rows, "m28_zero_g28_predicted_rescue", "d_shape")
    pred_stats = None
    pred_ok = not require_predicted_rescue
    provenance_ok = not require_predicted_rescue
    if predicted:
        e_pred_recovery = _paired_maps(zero, predicted, min_clusters=min_clusters,
                                       n_boot=n_boot, alpha=alpha)
        e_pred_null, pred_eq = _equivalent_to_zero(
            predicted, margin=equivalence_margin, min_clusters=min_clusters,
            n_boot=n_boot, alpha=alpha)
        selected = [r for r in rows if r.condition == "m28_zero_g28_predicted_rescue"]
        provenance_ok = all(r.rescue_held_out and not r.rescue_target_derived for r in selected)
        pred_ok = e_pred_recovery.lo > rescue_margin and pred_eq and provenance_ok
        pred_stats = {
            "recovery": e_pred_recovery.as_row(),
            "residual_vs_native": e_pred_null.as_row(),
            "equivalent_to_native": pred_eq,
            "heldout_non_target_derived": provenance_ok,
        }

    checks = {
        "m28_zero_damages_output": e_damage.lo > damage_margin,
        "m28_zero_changes_bilinear_product": e_product.lo > product_change_margin,
        "natural_g28_clamp_recovers": e_clamp_recovery.lo > rescue_margin and clamp_eq,
        "heldout_predicted_delta_recovers": pred_ok,
    }
    supported = all(checks.values())
    damage_eq = e_damage.lo > -equivalence_margin and e_damage.hi < equivalence_margin
    product_eq = e_product.lo > -equivalence_margin and e_product.hi < equivalence_margin
    if supported:
        status, conclusion_code = "supported", "m28_preconditions_writer"
    elif damage_eq and product_eq:
        status, conclusion_code = "equivalent", "m28_bounded_local_null"
    elif (checks["m28_zero_damages_output"]
          and checks["m28_zero_changes_bilinear_product"]
          and not (checks["natural_g28_clamp_recovers"]
                   or checks["heldout_predicted_delta_recovers"])):
        status, conclusion_code = "refuted", "m28_alternative_input_path"
    elif any(checks.values()):
        status, conclusion_code = "mixed", "m28_redundant_inputs"
    else:
        status, conclusion_code = "unresolved", "m28_unresolved"
    return {
        "status": status,
        "conclusion_code": conclusion_code,
        "checks": checks,
        "estimates": {
            "m28_zero_damage": e_damage.as_row(),
            "bilinear_product_change": e_product.as_row(),
            "exact_g28_clamp_recovery": e_clamp_recovery.as_row(),
            "exact_g28_clamp_residual": e_clamp_null.as_row(),
            "predicted_rescue": pred_stats,
        },
        "claim": ("m28 preconditions the g28 write"
                  if supported else "m28/g28 path only partially identified"),
        "note": ("The exact clamp isolates the path but is not held-out sufficiency; "
                 "the predicted-delta clause supplies that stronger test."),
        "margins_used": {
            "damage_margin": damage_margin,
            "rescue_margin": rescue_margin,
            "equivalence_margin": equivalence_margin,
            "product_change_margin": product_change_margin,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
    }


# ---------------------------------------------------------------------------
# b29 amplifier versus co-writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B29Prediction:
    parallel: Tensor
    perpendicular: Tensor
    method: str
    source_split: str
    held_out: bool
    target_derived: bool = False
    source_role: str = "development"
    lineage: Optional[PredictionLineage] = None

    def validate(self, reference: Tensor, *, locked: bool,
                 locked_input: Optional[Tensor] = None,
                 locked_unit_id: str = "") -> None:
        for name, value in (("parallel", self.parallel),
                            ("perpendicular", self.perpendicular)):
            if value.shape != reference.shape or not torch.isfinite(value).all():
                raise ValueError(f"predicted b29 {name} has invalid shape/value")
        if not self.method or not self.source_split:
            raise ValueError("b29 prediction needs method/source split provenance")
        if self.source_role not in {"discovery", "development"}:
            raise ValueError("b29 predictor must be frozen before locked evaluation")
        if locked and (not self.held_out or self.target_derived):
            raise RuntimeError("locked b29 sufficiency must be held-out and target-free")
        if locked:
            if self.lineage is None or locked_input is None or not locked_unit_id:
                raise RuntimeError(
                    "locked b29 prediction requires fitted-model lineage")
            joined = torch.cat((
                self.parallel.detach().reshape(-1),
                self.perpendicular.detach().reshape(-1),
            ))
            self.lineage.validate(
                joined, method=self.method, locked_input=locked_input,
                locked_unit_id=locked_unit_id)


@dataclass(frozen=True)
class IndependentB29Event:
    """A discovery/development-selected event sealed before locked evaluation."""

    ids: Tensor
    event_id: str
    source_split: str
    source_role: str
    selection_method: str
    dependency_keys: tuple[str, ...]
    held_out: bool
    target_derived: bool = False
    selection_sha256: str = ""

    def compute_hash(self) -> str:
        payload = "\x1f".join((
            self.event_id, self.source_split, self.source_role,
            self.selection_method, *self.dependency_keys,
            str(bool(self.held_out)), str(bool(self.target_derived)),
            _tensor_hash(self.ids),
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self, native_ids: Tensor, *, locked: bool) -> None:
        if (not self.event_id.strip() or not self.source_split.strip()
                or not self.selection_method.strip() or not self.dependency_keys):
            raise ValueError("independent b29 event lacks sealed selection provenance")
        if self.source_role not in {"discovery", "development"}:
            raise ValueError("independent b29 event must be selected before locked analysis")
        if self.ids.shape == native_ids.shape and torch.equal(self.ids, native_ids):
            raise ValueError("independent b29 event is identical to the native sequence")
        if self.ids.dtype.is_floating_point or self.ids.dtype.is_complex:
            raise TypeError("independent b29 event ids must be integer token ids")
        if self.selection_sha256 != self.compute_hash():
            raise RuntimeError("independent b29 event differs from its selection commitment")
        if locked and (not self.held_out or self.target_derived):
            raise RuntimeError(
                "locked independent event must be held-out and target-free")


def seal_independent_b29_event(
    ids: Tensor,
    *,
    event_id: str,
    source_split: str,
    source_role: str,
    selection_method: str,
    dependency_keys: Sequence[str],
    held_out: bool,
    target_derived: bool = False,
) -> IndependentB29Event:
    event = IndependentB29Event(
        ids=ids.detach().clone(), event_id=event_id,
        source_split=source_split, source_role=source_role,
        selection_method=selection_method,
        dependency_keys=tuple(str(value) for value in dependency_keys),
        held_out=held_out, target_derived=target_derived,
    )
    return IndependentB29Event(**{
        **event.__dict__, "selection_sha256": event.compute_hash(),
    })


@dataclass
class B29RoleRow:
    locus: str
    condition: str
    d_shape: float
    log_beta: float
    signed_upstream_following: float
    parallel_response_norm: float
    perpendicular_response_norm: float
    perpendicular_fraction: float
    realized_parallel_gain: float
    realized_perpendicular_gain: float
    prediction_method: str = ""
    prediction_held_out: bool = False
    prediction_target_derived: bool = False
    subspace_hash: str = ""
    dependency_keys: tuple[str, ...] = ()
    parallel_dose: Optional[float] = None
    parallel_q: Optional[float] = None
    parallel_alpha_star: Optional[float] = None
    independent_event: bool = False
    independent_event_id: str = ""
    independent_source_split: str = ""
    independent_source_role: str = ""
    independent_selection_method: str = ""
    independent_selection_sha256: str = ""
    independent_held_out: bool = False
    independent_target_derived: bool = False


@torch.no_grad()
def b29_parallel_perp_panel(
    runner: Evo2Runner,
    loci: Sequence[tuple],
    man: Manifest,
    *,
    b28_space: FrozenSubspace,
    upstream_intervention: Callable[[str, Tensor], Sequence[Edit]],
    prediction: Optional[Callable[[str, Tensor], B29Prediction]] = None,
    independent_inputs: Optional[
        Callable[[str, Tensor], IndependentB29Event]
    ] = None,
    parallel_doses: Sequence[float] = tuple(np.logspace(-5, 0, 21)),
    response_tap: str = "g29",
    locked: bool = False,
) -> list[B29RoleRow]:
    """Measure the b29 *response* to a same-recipient b28 intervention.

    Blocking uses the realised response and is a valid necessity test because
    the basis, tap and intervention are frozen.  ``native_plus_*`` is called
    sufficiency only when its component comes from a held-out predictor.
    """
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    allowed = {N.m(29), N.g(29)}
    if response_tap not in allowed:
        raise ValueError(f"response_tap must be one of {sorted(allowed)}")
    if (not parallel_doses or any(not np.isfinite(d) or d <= 0 or d > 1
                                  for d in parallel_doses)
            or not any(abs(float(d) - 1.0) < 1e-12 for d in parallel_doses)):
        raise ValueError("parallel_doses must lie in (0,1] and include 1.0")
    taps = {N.x(29), N.r(29), response_tap, N.x(30)}
    rows: list[B29RoleRow] = []

    for item in loci:
        locus, ids, _, md = _unpack_locus(item)
        native = runner.run(ids, taps=taps)
        upstream_edits = list(upstream_intervention(locus, ids))
        if not upstream_edits:
            raise ValueError(f"{locus}: upstream intervention returned no edits")
        forbidden = {N.m(29), N.r(29), N.g(29), N.x(30), N.x(31), N.x(32)}
        hit = sorted({e.tap for e in upstream_edits if e.tap in forbidden})
        if hit:
            raise ValueError(f"{locus}: b28 intervention is not upstream of b29: {hit}")
        upstream = runner.run(ids, taps=taps, edits=upstream_edits)
        delta = upstream[response_tap] - native[response_tap]
        B = validate_subspace(b28_space, delta.shape[-1], locked=locked)
        par, perp = parallel_perpendicular(delta, B)
        total = delta.double().norm().clamp_min(1e-30)
        host_norm = upstream[N.r(29)].double().norm().clamp_min(1e-30)
        alpha_star = float(host_norm / par.double().norm().clamp_min(1e-30))

        dx29 = upstream[N.x(29)] - native[N.x(29)]
        in_par, in_perp = parallel_perpendicular(dx29, B)
        par_gain = float(par.double().norm()
                         / in_par.double().norm().clamp_min(1e-30))
        perp_gain = float(perp.double().norm()
                          / in_perp.double().norm().clamp_min(1e-30))

        z0 = _last(native.logits)
        zU = _last(upstream.logits)
        ref = _centre(zU) - _centre(z0)
        ref_n2 = ref.norm().square().clamp_min(1e-30)
        conditions: list[tuple[str, object, str, bool, bool]] = [
            ("native", native, "", False, False),
            ("upstream", upstream, "", False, False),
            ("upstream_block_parallel", runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, d=par: t - d.to(t.dtype).to(t.device),
                    "block b29 response parallel to frozen b28 space")]),
             "", False, False),
            ("upstream_block_perpendicular", runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, d=perp: t - d.to(t.dtype).to(t.device),
                    "block b29 response perpendicular to frozen b28 space")]),
             "", False, False),
            ("upstream_block_all", runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, d=delta: t - d.to(t.dtype).to(t.device),
                    "block complete b29 response")]),
             "", False, False),
        ]

        # Explicit remove/only panel.  These duplicate two algebraic cells in
        # value but not in interpretation: the names keep downstream tables
        # from silently treating a removal test as held-out sufficiency.
        conditions.extend([
            ("upstream_parallel_only", runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, d=delta, p=par:
                    t - d.to(t.dtype).to(t.device) + p.to(t.dtype).to(t.device),
                    "retain only b29 response parallel to U28")]),
             "realized_component_diagnostic", False, True),
            ("upstream_perpendicular_only", runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, d=delta, p=perp:
                    t - d.to(t.dtype).to(t.device) + p.to(t.dtype).to(t.device),
                    "retain only b29 response perpendicular to U28")]),
             "realized_component_diagnostic", False, True),
        ])

        if prediction is not None:
            pred = prediction(locus, ids)
            pred.validate(
                delta, locked=locked, locked_input=ids,
                locked_unit_id=locus)
            for name, d in (("native_plus_predicted_parallel", pred.parallel),
                            ("native_plus_predicted_perpendicular", pred.perpendicular),
                            ("native_plus_predicted_full", pred.parallel + pred.perpendicular)):
                trace = runner.run(ids, taps=taps, edits=[Edit(
                    response_tap, "update",
                    lambda t, dd=d: t + dd.to(t.dtype).to(t.device),
                    f"{name}:{pred.method}")])
                conditions.append((name, trace, pred.method, pred.held_out,
                                   pred.target_derived))
            for name, d in (
                ("upstream_block_all_predicted_parallel_rescue", pred.parallel),
                ("upstream_block_all_predicted_perpendicular_rescue", pred.perpendicular),
                ("upstream_block_all_predicted_full_rescue",
                 pred.parallel + pred.perpendicular),
            ):
                trace = runner.run(ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, observed=delta, dd=d:
                    t - observed.to(t.dtype).to(t.device) + dd.to(t.dtype).to(t.device),
                    f"{name}:{pred.method}")])
                conditions.append((name, trace, pred.method, pred.held_out,
                                   pred.target_derived))
        elif locked:
            raise ValueError("locked b29 role test requires held-out predicted components")

        # Dose the realised parallel response on the upstream trajectory.  It
        # is a scale/plateau experiment, never counted as prediction
        # sufficiency.  q=dose/alpha_star is the dimensionless dominance dose.
        dose_traces: list[tuple[float, object]] = []
        for dose in sorted({float(d) for d in parallel_doses}):
            dose_traces.append((dose, runner.run(
                ids, taps=taps, edits=upstream_edits + [Edit(
                    response_tap, "update",
                    lambda t, observed=delta, p=par, a=dose:
                    t - observed.to(t.dtype).to(t.device)
                    + a * p.to(t.dtype).to(t.device),
                    f"parallel response dose={dose:g}")])))

        # An independently chosen sequence event (typically a motif edit)
        # tests whether the perpendicular mode has meaning beyond the event
        # used to define U28.  This clause is mandatory before the stronger
        # word "co-writer/editor" is returned.
        independent_conditions: list[tuple[str, object]] = []
        independent_event: Optional[IndependentB29Event] = None
        if independent_inputs is not None:
            independent_event = independent_inputs(locus, ids)
            if not isinstance(independent_event, IndependentB29Event):
                raise TypeError(
                    "independent_inputs must return a sealed IndependentB29Event")
            independent_event.validate(ids, locked=locked)
            event_ids = independent_event.ids
            event = runner.run(event_ids, taps=taps)
            event_delta = event[response_tap] - native[response_tap]
            _, event_perp = parallel_perpendicular(event_delta, B)
            independent_conditions = [
                ("independent_event", event),
                ("independent_event_block_perpendicular", runner.run(
                    event_ids, taps=taps, edits=[Edit(
                        response_tap, "update",
                        lambda t, d=event_perp: t - d.to(t.dtype).to(t.device),
                        "block event-induced b29 perpendicular response")]))
            ]

        for cond, trace, method, held_out, target_derived in conditions:
            z = _last(trace.logits)
            shape = solve_beta_shape(
                z, z0, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold)
            dz = _centre(z) - _centre(z0)
            rows.append(B29RoleRow(
                locus=locus, condition=cond, d_shape=shape.d_shape,
                log_beta=shape.log_beta,
                signed_upstream_following=float(torch.dot(dz, ref) / ref_n2),
                parallel_response_norm=float(par.double().norm()),
                perpendicular_response_norm=float(perp.double().norm()),
                perpendicular_fraction=float(perp.double().norm() / total),
                realized_parallel_gain=par_gain,
                realized_perpendicular_gain=perp_gain,
                prediction_method=method, prediction_held_out=held_out,
                prediction_target_derived=target_derived,
                subspace_hash=b28_space.sha256,
                dependency_keys=_row_keys(locus, md)))
        for dose, trace in dose_traces:
            z = _last(trace.logits)
            shape = solve_beta_shape(
                z, z0, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold)
            dz = _centre(z) - _centre(z0)
            rows.append(B29RoleRow(
                locus=locus, condition="parallel_dose", d_shape=shape.d_shape,
                log_beta=shape.log_beta,
                signed_upstream_following=float(torch.dot(dz, ref) / ref_n2),
                parallel_response_norm=float(par.double().norm()),
                perpendicular_response_norm=float(perp.double().norm()),
                perpendicular_fraction=float(perp.double().norm() / total),
                realized_parallel_gain=par_gain,
                realized_perpendicular_gain=perp_gain,
                subspace_hash=b28_space.sha256,
                dependency_keys=_row_keys(locus, md),
                parallel_dose=dose, parallel_q=dose / alpha_star,
                parallel_alpha_star=alpha_star))

        if independent_conditions:
            z_event = _last(independent_conditions[0][1].logits)
            event_ref = _centre(z_event) - _centre(z0)
            event_n2 = event_ref.norm().square().clamp_min(1e-30)
            assert independent_event is not None
            for name, trace in independent_conditions:
                z = _last(trace.logits)
                shape = solve_beta_shape(
                    z, z0, beta_min=man.margins.beta_min,
                    beta_max=man.margins.beta_max,
                    flat_logit_threshold=man.margins.flat_logit_threshold)
                dz = _centre(z) - _centre(z0)
                rows.append(B29RoleRow(
                    locus=locus, condition=name, d_shape=shape.d_shape,
                    log_beta=shape.log_beta,
                    signed_upstream_following=float(torch.dot(dz, event_ref) / event_n2),
                    parallel_response_norm=float(par.double().norm()),
                    perpendicular_response_norm=float(perp.double().norm()),
                    perpendicular_fraction=float(perp.double().norm() / total),
                    realized_parallel_gain=par_gain,
                    realized_perpendicular_gain=perp_gain,
                    subspace_hash=b28_space.sha256,
                    dependency_keys=independent_event.dependency_keys,
                    independent_event=True,
                    independent_event_id=independent_event.event_id,
                    independent_source_split=independent_event.source_split,
                    independent_source_role=independent_event.source_role,
                    independent_selection_method=independent_event.selection_method,
                    independent_selection_sha256=(
                        independent_event.selection_sha256),
                    independent_held_out=independent_event.held_out,
                    independent_target_derived=(
                        independent_event.target_derived)))
    return rows


def b29_role_verdict(
    rows: Sequence[B29RoleRow],
    *,
    effect_margin: float,
    equivalence_margin: float,
    perpendicular_fraction_margin: float,
    independent_effect_margin: Optional[float] = None,
    parallel_plateau_gate: Optional[Mapping[str, object]] = None,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Return amplifier/co-writer only after necessity and sufficiency agree."""
    if equivalence_margin <= 0 or not (0 <= perpendicular_fraction_margin <= 1):
        raise ValueError("invalid equivalence/perpendicular-fraction margin")
    hashes = {r.subspace_hash for r in rows if r.subspace_hash}
    if len(hashes) != 1:
        raise ValueError("b29 rows must use exactly one frozen b28 subspace")

    attr = "signed_upstream_following"
    upstream = _condition_map(rows, "upstream", attr)
    block_par = _condition_map(rows, "upstream_block_parallel", attr)
    block_perp = _condition_map(rows, "upstream_block_perpendicular", attr)
    zero = {k: Row(0.0, keys=v.keys) for k, v in upstream.items()}
    e_up = _paired_maps(upstream, zero, min_clusters=min_clusters,
                        n_boot=n_boot, alpha=alpha)
    e_par_need = _paired_maps(upstream, block_par, min_clusters=min_clusters,
                              n_boot=n_boot, alpha=alpha)
    e_perp_need = _paired_maps(upstream, block_perp, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    perp_fraction = _condition_map(rows, "upstream", "perpendicular_fraction")
    e_perp_fraction = _paired_maps(perp_fraction, zero, min_clusters=min_clusters,
                                   n_boot=n_boot, alpha=alpha)

    pred_par = _condition_map(rows, "native_plus_predicted_parallel", attr)
    pred_perp = _condition_map(rows, "native_plus_predicted_perpendicular", attr)
    pred_full = _condition_map(rows, "native_plus_predicted_full", attr)
    if not pred_par or not pred_perp or not pred_full:
        raise RuntimeError("b29 role verdict requires all held-out predicted component cells")
    e_par_suff = _paired_maps(pred_par, zero, min_clusters=min_clusters,
                              n_boot=n_boot, alpha=alpha)
    e_perp_suff = _paired_maps(pred_perp, zero, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    e_full_suff = _paired_maps(pred_full, zero, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    par_only = _condition_map(rows, "upstream_parallel_only", attr)
    if not par_only:
        raise RuntimeError("b29 role verdict requires the realized parallel-only cell")
    e_par_only_residual = _paired_maps(
        upstream, par_only, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    e_par_prediction_residual = _paired_maps(
        upstream, pred_par, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    e_full_prediction_residual = _paired_maps(
        upstream, pred_full, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    block_all = _condition_map(rows, "upstream_block_all", attr)
    par_rescue = _condition_map(
        rows, "upstream_block_all_predicted_parallel_rescue", attr)
    perp_rescue = _condition_map(
        rows, "upstream_block_all_predicted_perpendicular_rescue", attr)
    full_rescue = _condition_map(
        rows, "upstream_block_all_predicted_full_rescue", attr)
    if not block_all or not par_rescue or not perp_rescue or not full_rescue:
        raise RuntimeError("b29 role verdict requires predicted remove/only/rescue cells")
    e_par_rescue = _paired_maps(par_rescue, block_all, min_clusters=min_clusters,
                                n_boot=n_boot, alpha=alpha)
    e_perp_rescue = _paired_maps(perp_rescue, block_all, min_clusters=min_clusters,
                                 n_boot=n_boot, alpha=alpha)
    e_full_rescue = _paired_maps(full_rescue, block_all, min_clusters=min_clusters,
                                 n_boot=n_boot, alpha=alpha)
    e_par_rescue_residual = _paired_maps(
        upstream, par_rescue, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    e_full_rescue_residual = _paired_maps(
        upstream, full_rescue, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    predicted_rows = [
        r for r in rows
        if r.condition.startswith("native_plus_predicted_")
        or "_predicted_" in r.condition
    ]
    provenance = bool(predicted_rows) and all(
        r.prediction_held_out and not r.prediction_target_derived
        and bool(r.prediction_method)
        for r in predicted_rows)

    perp_need_null = (e_perp_need.lo > -equivalence_margin
                      and e_perp_need.hi < equivalence_margin)
    perp_suff_null = (e_perp_suff.lo > -equivalence_margin
                      and e_perp_suff.hi < equivalence_margin)
    fraction_small = e_perp_fraction.hi < perpendicular_fraction_margin
    plateau_ok = bool(parallel_plateau_gate and parallel_plateau_gate.get("passed"))
    par_only_matches_full = (
        e_par_only_residual.lo > -equivalence_margin
        and e_par_only_residual.hi < equivalence_margin)
    par_prediction_matches_full = (
        e_par_prediction_residual.lo > -equivalence_margin
        and e_par_prediction_residual.hi < equivalence_margin)
    full_prediction_matches_full = (
        e_full_prediction_residual.lo > -equivalence_margin
        and e_full_prediction_residual.hi < equivalence_margin)
    par_rescue_matches_full = (
        e_par_rescue_residual.lo > -equivalence_margin
        and e_par_rescue_residual.hi < equivalence_margin)
    full_rescue_matches_full = (
        e_full_rescue_residual.lo > -equivalence_margin
        and e_full_rescue_residual.hi < equivalence_margin)

    independent = _condition_map(rows, "independent_event", attr)
    independent_block = _condition_map(
        rows, "independent_event_block_perpendicular", attr)
    independent_est = None
    independent_ok = False
    independent_null = False
    independent_provenance = False
    if independent and independent_block:
        independent_est = _paired_maps(
            independent, independent_block, min_clusters=min_clusters,
            n_boot=n_boot, alpha=alpha)
        if independent_effect_margin is None:
            raise RuntimeError(
                "independent motif/event rows require a pre-registered effect margin")
        independent_ok = independent_est.lo > independent_effect_margin
        independent_null = (
            independent_est.lo > -equivalence_margin
            and independent_est.hi < equivalence_margin
        )
        event_rows = [r for r in rows if r.independent_event]
        event_groups: dict[str, list[B29RoleRow]] = {}
        for row in event_rows:
            event_groups.setdefault(row.independent_event_id, []).append(row)
        independent_provenance = (
            bool(event_groups) and "" not in event_groups
            and all(
                len({r.independent_selection_sha256 for r in group}) == 1
                and bool(group[0].independent_selection_sha256)
                and {r.condition for r in group} == {
                    "independent_event",
                    "independent_event_block_perpendicular",
                }
                for group in event_groups.values()
            )
            and all(
                r.independent_held_out and not r.independent_target_derived
                and r.independent_source_role in {"discovery", "development"}
                and bool(r.independent_event_id)
                and bool(r.independent_source_split)
                and bool(r.independent_selection_method)
                for r in event_rows
            )
        )
    common = {
        "upstream_effect_exists": e_up.lo > effect_margin,
        "parallel_component_is_necessary": e_par_need.lo > effect_margin,
        "parallel_prediction_is_sufficient": e_par_suff.lo > effect_margin,
        "parallel_prediction_rescues_blocked_response": e_par_rescue.lo > effect_margin,
        "full_prediction_is_sufficient": e_full_suff.lo > effect_margin,
        "full_prediction_rescues_blocked_response": e_full_rescue.lo > effect_margin,
        "prediction_is_heldout_and_target_free": provenance,
        "realized_parallel_only_matches_full_effect": par_only_matches_full,
        "predicted_parallel_matches_full_effect": par_prediction_matches_full,
        "predicted_parallel_rescue_matches_full_effect": par_rescue_matches_full,
        "predicted_full_matches_full_effect": full_prediction_matches_full,
        "predicted_full_rescue_matches_full_effect": full_rescue_matches_full,
        "parallel_dose_plateaus": plateau_ok,
    }
    amplifier_checks = {
        **common,
        "perpendicular_energy_equivalent_small": fraction_small,
        "perpendicular_component_output_null": perp_need_null and perp_suff_null,
        "independent_event_perpendicular_output_null": (
            bool(independent and independent_block) and independent_null
            and independent_provenance),
    }
    cowriter_checks = {
        **common,
        "perpendicular_response_is_nontrivial": e_perp_fraction.lo > perpendicular_fraction_margin,
        "perpendicular_component_is_necessary": e_perp_need.lo > effect_margin,
        "perpendicular_prediction_is_sufficient": e_perp_suff.lo > effect_margin,
        "perpendicular_prediction_rescues_blocked_response": e_perp_rescue.lo > effect_margin,
        "independent_event_uses_perpendicular_mode": independent_ok,
        "independent_event_was_preselected_and_heldout": independent_provenance,
    }
    editor_checks = {
        "upstream_effect_exists": e_up.lo > effect_margin,
        "perpendicular_component_is_necessary": e_perp_need.lo > effect_margin,
        "independent_event_uses_perpendicular_mode": independent_ok,
        "independent_event_was_preselected_and_heldout": independent_provenance,
        "full_prediction_matches_full_effect": full_prediction_matches_full,
        "full_prediction_rescues_full_effect": full_rescue_matches_full,
        "prediction_is_heldout_and_target_free": provenance,
    }
    upstream_null = e_up.lo > -equivalence_margin and e_up.hi < equivalence_margin
    if all(amplifier_checks.values()):
        role = "b29 amplifier of the frozen b28 mode"
        status, conclusion_code = "supported", "b29_amplifier"
    elif all(cowriter_checks.values()):
        role = "b29 co-writer (adds an output-causal orthogonal mode)"
        status, conclusion_code = "supported", "b29_second_writer"
    elif all(editor_checks.values()):
        role = "b29 editor (orthogonal mode conditionally modifies the write)"
        status, conclusion_code = "supported", "b29_editor"
    elif upstream_null:
        role = "bounded local null for the tested b29 response"
        status, conclusion_code = "equivalent", "b29_bounded_null"
    else:
        role = "mixed or not causally identified"
        if all(common.values()):
            status, conclusion_code = "mixed", "b29_mixed_role"
        elif e_up.lo > effect_margin and not (
                e_par_need.lo > effect_margin or e_perp_need.lo > effect_margin):
            status, conclusion_code = "refuted", "amplifier_refuted"
        elif any(amplifier_checks.values()) or any(cowriter_checks.values()):
            status, conclusion_code = "mixed", "b29_mixed_role"
        else:
            status, conclusion_code = "unresolved", "b29_unresolved"
    return {
        "status": status,
        "conclusion_code": conclusion_code,
        "role": role,
        "amplifier_checks": amplifier_checks,
        "cowriter_checks": cowriter_checks,
        "editor_checks": editor_checks,
        "estimates": {
            "upstream_effect": e_up.as_row(),
            "parallel_necessity": e_par_need.as_row(),
            "perpendicular_necessity": e_perp_need.as_row(),
            "perpendicular_fraction": e_perp_fraction.as_row(),
            "predicted_parallel_sufficiency": e_par_suff.as_row(),
            "predicted_perpendicular_sufficiency": e_perp_suff.as_row(),
            "predicted_full_sufficiency": e_full_suff.as_row(),
            "predicted_parallel_rescue": e_par_rescue.as_row(),
            "predicted_perpendicular_rescue": e_perp_rescue.as_row(),
            "predicted_full_rescue": e_full_rescue.as_row(),
            "realized_parallel_only_residual_to_full": (
                e_par_only_residual.as_row()),
            "predicted_parallel_residual_to_full": (
                e_par_prediction_residual.as_row()),
            "predicted_full_residual_to_full": (
                e_full_prediction_residual.as_row()),
            "predicted_parallel_rescue_residual_to_full": (
                e_par_rescue_residual.as_row()),
            "predicted_full_rescue_residual_to_full": (
                e_full_rescue_residual.as_row()),
            "independent_event_perpendicular_effect": (
                None if independent_est is None else independent_est.as_row()),
        },
        "parallel_plateau_gate": dict(parallel_plateau_gate or {}),
        "alpha_doses": list((parallel_plateau_gate or {}).get("alpha_doses", ())),
        "margins_used": {
            "effect_margin": effect_margin,
            "equivalence_margin": equivalence_margin,
            "perpendicular_fraction_margin": perpendicular_fraction_margin,
            "independent_effect_margin": independent_effect_margin,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
            "parallel_plateau": dict(
                (parallel_plateau_gate or {}).get("margins_used", {})),
        },
        "subspace_hash": next(iter(hashes)),
        "note": "realized causal gain is reported in rows; raw norm ratio is never the role test",
    }


def b29_parallel_plateau_gate(
    rows: Sequence[B29RoleRow],
    *,
    effect_margin: float,
    increment_equivalence_margin: float,
    log_q_boundary_margin: float = 0.50,
    plateau_remaining_fraction: float = 0.10,
    plateau_boundary_spread: float = 0.50,
    expected_doses: Optional[Sequence[float]] = None,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Bounded 0-1 verdict: test whether the U28-parallel response saturates.

    The last two log-spaced doses are compared within locus.  This is a
    deliberately conservative operational plateau: a large response at dose
    1, negligible last increment, finite ``alpha*``, and monotone median
    response are all required.  Downstream code can inspect every gate field.
    """
    selected = [r for r in rows if r.condition == "parallel_dose"]
    expected = np.asarray(
        tuple(np.logspace(-5, 0, 21)) if expected_doses is None
        else tuple(float(x) for x in expected_doses), dtype=float)
    if (expected.ndim != 1 or len(expected) < 8
            or not np.isfinite(expected).all() or np.any(expected <= 0)
            or np.any(np.diff(expected) <= 0)):
        raise ValueError("expected_doses must be a finite increasing positive grid")
    by_locus: dict[str, list[B29RoleRow]] = {}
    for r in selected:
        if r.parallel_dose is None or r.parallel_q is None \
                or r.parallel_alpha_star is None:
            raise ValueError("parallel dose row lacks dose/q/alpha_star")
        by_locus.setdefault(r.locus, []).append(r)
    full_rows, increment_rows = [], []
    midpoint_rows, boundary_rows = [], []
    monotone = True
    alpha_star_finite = True
    per_locus = {}
    for locus, rs in by_locus.items():
        rs = sorted(rs, key=lambda r: float(r.parallel_dose))
        observed = np.asarray([float(r.parallel_dose) for r in rs])
        if observed.shape != expected.shape or not np.allclose(
                observed, expected, rtol=1e-12, atol=1e-15):
            raise RuntimeError(f"{locus}: parallel dose grid is incomplete")
        vals = np.asarray([r.signed_upstream_following for r in rs], dtype=float)
        # Tiny numerical reversals are tolerated only up to the declared
        # equivalence margin; a visibly non-monotone curve is not a plateau.
        monotone &= bool(np.all(np.diff(vals) >= -increment_equivalence_margin))
        alpha_star_finite &= bool(np.isfinite(rs[-1].parallel_alpha_star)
                                  and rs[-1].parallel_alpha_star > 0)
        keys = rs[-1].dependency_keys or (f"locus:{locus}",)
        full_rows.append(Row(float(vals[-1]), keys=keys))
        increment_rows.append(Row(float(vals[-1] - vals[-2]), keys=keys))
        total = float(vals[-1] - vals[0])
        q50 = None
        first_plateau_q = None
        if abs(total) > 1e-12:
            normalized = (vals - vals[0]) / total
            crossing = np.flatnonzero(normalized >= 0.5)
            if len(crossing):
                j = int(crossing[0])
                if j == 0:
                    log_q50 = float(np.log10(rs[0].parallel_q))
                else:
                    y0, y1 = normalized[j - 1], normalized[j]
                    weight = (0.5 - y0) / max(float(y1 - y0), 1e-30)
                    x0 = np.log10(rs[j - 1].parallel_q)
                    x1 = np.log10(rs[j].parallel_q)
                    log_q50 = float(x0 + np.clip(weight, 0.0, 1.0) * (x1 - x0))
                q50 = float(10 ** log_q50)
                midpoint_rows.append(Row(log_q50, keys=keys))
            for j in range(len(rs)):
                tail = normalized[j:]
                if (np.all(np.abs(1.0 - tail) <= plateau_remaining_fraction)
                        and (len(tail) < 2 or np.all(
                            np.abs(np.diff(tail)) <= plateau_remaining_fraction))):
                    first_plateau_q = float(rs[j].parallel_q)
                    boundary_rows.append(Row(
                        float(np.log10(first_plateau_q)), keys=keys))
                    break
        per_locus[locus] = {
            "alpha_star": rs[-1].parallel_alpha_star,
            "last_q": rs[-1].parallel_q,
            "last_increment": float(vals[-1] - vals[-2]),
            "doses": [r.parallel_dose for r in rs],
            "following": vals.tolist(),
            "q50": q50,
            "first_sustained_plateau_q": first_plateau_q,
        }
    if not full_rows:
        raise RuntimeError("no parallel-dose rows")
    e_full = cluster_bootstrap(full_rows, min_clusters=min_clusters,
                               n_boot=n_boot, alpha=alpha)
    e_inc = cluster_bootstrap(increment_rows, min_clusters=min_clusters,
                              n_boot=n_boot, alpha=alpha)
    e_midpoint = (cluster_bootstrap(
        midpoint_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        if len(midpoint_rows) == len(by_locus) else None)
    e_boundary = (cluster_bootstrap(
        boundary_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        if len(boundary_rows) == len(by_locus) else None)
    midpoint_values = [float(row.value) for row in midpoint_rows]
    boundary_values = [float(row.value) for row in boundary_rows]
    midpoint_max_abs = (
        float(max(abs(value) for value in midpoint_values))
        if midpoint_values else float("inf"))
    boundary_max_spread = (
        float(max(boundary_values) - min(boundary_values))
        if boundary_values else float("inf"))
    increment_equiv = (e_inc.lo > -increment_equivalence_margin
                       and e_inc.hi < increment_equivalence_margin)
    checks = {
        "log_grid_complete": all(
            len(v["doses"]) == len(expected) for v in per_locus.values()),
        "alpha_star_finite": alpha_star_finite,
        "parallel_response_nonzero": e_full.lo > effect_margin,
        "last_increment_equivalent_zero": increment_equiv,
        "dose_response_monotone": monotone,
        "q50_is_near_dominance_boundary": bool(
            e_midpoint is not None
            and e_midpoint.lo > -log_q_boundary_margin
            and e_midpoint.hi < log_q_boundary_margin
            and midpoint_max_abs <= log_q_boundary_margin),
        "first_sustained_plateau_is_shared": bool(
            e_boundary is not None
            and boundary_max_spread <= plateau_boundary_spread),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "full_dose_effect": e_full.as_row(),
        "last_increment": e_inc.as_row(),
        "log10_q50": None if e_midpoint is None else e_midpoint.as_row(),
        "log10_first_sustained_plateau_q": (
            None if e_boundary is None else e_boundary.as_row()),
        "boundary_heterogeneity": {
            "q50_max_abs_log10_distance_from_one": midpoint_max_abs,
            "first_plateau_max_log10_spread": boundary_max_spread,
        },
        "per_locus": per_locus,
        "alpha_doses": expected.tolist(),
        "margins_used": {
            "effect_margin": effect_margin,
            "increment_equivalence_margin": increment_equivalence_margin,
            "log_q_boundary_margin": log_q_boundary_margin,
            "plateau_remaining_fraction": plateau_remaining_fraction,
            "plateau_boundary_spread": plateau_boundary_spread,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
        "normalization": "q = dose / alpha_star; alpha_star = ||r29|| / ||delta_g29_parallel||",
    }


# ---------------------------------------------------------------------------
# Scale-plateau normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleGeometryPoint:
    alpha: float
    q: float
    theta: float
    theta_fraction: float
    dominance_fraction: float
    cancellation: float
    sign_crossing: bool
    normalized_effect: Optional[float] = None


def normalize_scale_plateau(
    r: Tensor,
    g: Tensor,
    alphas: Sequence[float],
    *,
    effects: Optional[Sequence[float]] = None,
) -> list[ScaleGeometryPoint]:
    """Express alpha sweeps in geometry shared across loci.

    ``q=alpha||g||/||r||`` removes arbitrary branch units;
    ``theta_fraction`` reports how much of the asymptotic direction change has
    occurred.  Effects, when supplied, are divided by their largest absolute
    value only for visual comparison--never for a causal verdict.
    """
    if not alphas or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("alphas must be a non-empty finite non-negative grid")
    if effects is not None and len(effects) != len(alphas):
        raise ValueError("effects and alphas must have equal length")
    rr, gg = _last(r).double(), _last(g).double()
    if float(rr.norm()) == 0 or float(gg.norm()) == 0:
        raise ValueError("r and g must be non-zero")
    cos = torch.clamp(torch.dot(rr, gg) / (rr.norm() * gg.norm()), -1.0, 1.0)
    theta_inf = float(torch.arccos(cos))
    effect_scale = (max(abs(float(v)) for v in effects)
                    if effects is not None else None)
    points = []
    for i, a in enumerate(alphas):
        geom = branch_geometry(rr, gg, float(a))
        frac = (0.0 if theta_inf < 1e-12
                else float(np.clip(abs(geom.theta) / theta_inf, 0.0, 1.0)))
        q = geom.q
        points.append(ScaleGeometryPoint(
            alpha=float(a), q=q, theta=geom.theta, theta_fraction=frac,
            dominance_fraction=float(q / (1.0 + q)),
            cancellation=geom.cancellation, sign_crossing=geom.sign_crossing,
            normalized_effect=(None if effects is None or effect_scale in (None, 0)
                               else float(effects[i]) / effect_scale)))
    return points


def first_scale_plateau(
    points: Sequence[ScaleGeometryPoint],
    *,
    angular_remaining_tolerance: float,
    effect_increment_tolerance: Optional[float] = None,
) -> Optional[float]:
    """First pre-registered dose after which direction/effect stay flat."""
    if not (0 < angular_remaining_tolerance < 1):
        raise ValueError("angular_remaining_tolerance must lie in (0,1)")
    pts = sorted(points, key=lambda p: p.q)
    for i, p in enumerate(pts):
        tail = pts[i:]
        if any(1.0 - q.theta_fraction > angular_remaining_tolerance for q in tail):
            continue
        if effect_increment_tolerance is not None:
            vals = [q.normalized_effect for q in tail]
            if any(v is None for v in vals):
                raise ValueError("effect tolerance requested without normalized effects")
            if max(vals) - min(vals) > effect_increment_tolerance:
                continue
        return p.q
    return None


# ---------------------------------------------------------------------------
# Mandatory 0-2 multi-layer / 1B scale scan
# ---------------------------------------------------------------------------


DEFAULT_LOG_SCALE_GRID = tuple(float(x) for x in np.logspace(-5, 0, 21))


@dataclass(frozen=True)
class ScaleScanSite:
    """One residual update and the state it is added to."""

    label: str
    block: int
    update_tap: str
    host_tap: str

    def validate(self) -> None:
        valid = {
            (N.g(self.block), N.r(self.block)),
            (N.m(self.block), N.x(self.block)),
        }
        if (self.update_tap, self.host_tap) not in valid:
            raise ValueError(
                f"{self.label}: update/host must be g{self.block}/r{self.block} "
                f"or m{self.block}/x{self.block}")


def standard_7b_scale_sites() -> tuple[ScaleScanSite, ...]:
    return (
        ScaleScanSite("b28_g", 28, N.g(28), N.r(28)),
        ScaleScanSite("b29_g", 29, N.g(29), N.r(29)),
        ScaleScanSite("b30_m", 30, N.m(30), N.x(30)),
    )


@dataclass(frozen=True)
class ScaleScanModel:
    name: str
    runner: Evo2Runner
    manifest: Manifest
    loci: Sequence[tuple]
    sites: Sequence[ScaleScanSite]


@dataclass
class ScaleScanRow:
    model: str
    locus: str
    site: str
    block: int
    update_tap: str
    control: str                 # radial | equal_norm_angular
    alpha: float
    alpha_star: float
    q: float
    path_fraction: float
    d_shape_from_natural: float
    log_beta_from_natural: float
    theta: float
    theta_fraction: float
    update_norm: float
    host_norm: float
    control_angle: float
    dependency_keys: tuple[str, ...] = ()


def _default_angular_control(update: Tensor, seed_text: str) -> Tensor:
    """Deterministic orthogonal diagnostic; locked runs require natural controls."""
    digest = hashlib.sha256(seed_text.encode()).digest()
    gen = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(digest[:8], "little") % (2**63 - 1))
    rnd = torch.randn(update.shape, dtype=torch.float64, generator=gen).to(update.device)
    u = update.double()
    # Orthogonalise each feature vector, not the flattened sequence tensor.
    rnd = rnd - (rnd * u).sum(-1, keepdim=True) \
        / u.square().sum(-1, keepdim=True).clamp_min(1e-30) * u
    bad = rnd.norm(dim=-1, keepdim=True) < 1e-10
    idx = u.abs().argmin(-1, keepdim=True)
    fallback = torch.zeros_like(u).scatter_(-1, idx, 1.0)
    fallback = fallback - (fallback * u).sum(-1, keepdim=True) \
        / u.square().sum(-1, keepdim=True).clamp_min(1e-30) * u
    return torch.where(bad, fallback, rnd).to(update.dtype)


@torch.no_grad()
def multi_model_log_scale_scan(
    models: Sequence[ScaleScanModel],
    *,
    alphas: Sequence[float] = DEFAULT_LOG_SCALE_GRID,
    angular_control: Optional[
        Callable[[str, str, ScaleScanSite, Tensor], Tensor]
    ] = None,
    locked: bool = False,
) -> list[ScaleScanRow]:
    """Log-spaced ``1e-5..1`` scans for b28/b29/b30 and a 1B control.

    The caller supplies the actual 1B layer mapping as ``ScaleScanSite``
    objects read from that runtime; this function never copies the 7B map.
    ``path_fraction`` is the centred-logit movement from alpha=0 to the
    natural alpha=1 output.  It is therefore already dimensionless and can be
    compared after the independent hidden-state normalization
    ``q=alpha/alpha_star``.
    """
    grid = np.asarray(alphas, dtype=float)
    if (grid.ndim != 1 or len(grid) < 3 or np.any(~np.isfinite(grid))
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
            or abs(grid[0] - 1e-5) > 1e-12 or abs(grid[-1] - 1.0) > 1e-12):
        raise ValueError("primary scale grid must be strictly increasing 1e-5..1")
    log_steps = np.diff(np.log10(grid))
    if float(log_steps.max() - log_steps.min()) > 1e-8:
        raise ValueError("primary scale grid must be log-spaced")
    if locked and angular_control is None:
        raise ValueError(
            "locked scan requires a discovery-frozen local-covariance angular control")
    if len({m.name for m in models}) != len(models):
        raise ValueError("scale-scan model names must be unique")

    rows: list[ScaleScanRow] = []
    for model in models:
        model.manifest.margins.require("beta_min", "beta_max", "flat_logit_threshold")
        if not model.sites:
            raise ValueError(f"{model.name}: no scale sites")
        for site in model.sites:
            site.validate()
            if not 0 <= site.block < model.runner.arch.n_layers:
                raise ValueError(f"{model.name}:{site.label}: block outside runtime")
        for item in model.loci:
            locus, ids, _, md = _unpack_locus(item)
            keys = _row_keys(locus, md)
            for site in model.sites:
                native = model.runner.run(ids, taps={site.host_tap, site.update_tap})
                host, update = native[site.host_tap], native[site.update_tap]
                h, u = _last(host).double(), _last(update).double()
                hnorm, unorm = h.norm(), u.norm()
                if float(hnorm) == 0 or float(unorm) == 0:
                    raise RuntimeError(f"{model.name}:{locus}:{site.label}: zero host/update")
                alpha_star = float(hnorm / unorm)
                natural_logits = _last(native.logits)
                zero = model.runner.run(ids, taps=(), edits=[Edit(
                    site.update_tap, "update", lambda t: torch.zeros_like(t),
                    f"{site.label}:alpha=0 anchor")])
                zero_logits = _last(zero.logits)
                output_direction = _centre(natural_logits) - _centre(zero_logits)
                output_n2 = output_direction.norm().square().clamp_min(1e-30)
                max_angle = float(torch.arccos(torch.clamp(
                    torch.dot(h, h + u) / (hnorm * (h + u).norm()).clamp_min(1e-30),
                    -1.0, 1.0)))

                raw_control = (angular_control(model.name, locus, site, update)
                               if angular_control is not None else
                               _default_angular_control(
                                   update, f"{model.name}:{locus}:{site.label}"))
                if raw_control.shape != update.shape or not torch.isfinite(raw_control).all():
                    raise ValueError(f"{model.name}:{locus}:{site.label}: bad angular control")
                angular_dir = raw_control / raw_control.norm(
                    dim=-1, keepdim=True).clamp_min(1e-30)
                natural_dir = update / update.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                control_angle = float(torch.arccos(torch.clamp(
                    (_last(angular_dir).double() * _last(natural_dir).double()).sum(),
                    -1.0, 1.0)))
                if control_angle < 1e-4:
                    raise ValueError("angular control is indistinguishable from natural direction")

                for a in grid:
                    radial_edit = Edit(
                        site.update_tap, "update",
                        lambda t, aa=float(a): t * aa,
                        f"{site.label}:radial alpha={a:g}")
                    angular_replacement = angular_dir.to(update) \
                        * update.norm(dim=-1, keepdim=True) * float(a)
                    angular_edit = Edit(
                        site.update_tap, "update", replace_with(angular_replacement),
                        f"{site.label}:equal-norm angular alpha={a:g}")
                    for control, edit in (("radial", radial_edit),
                                          ("equal_norm_angular", angular_edit)):
                        trace = model.runner.run(ids, taps=(), edits=[edit])
                        z = _last(trace.logits)
                        shape = solve_beta_shape(
                            z, natural_logits,
                            beta_min=model.manifest.margins.beta_min,
                            beta_max=model.manifest.margins.beta_max,
                            flat_logit_threshold=model.manifest.margins.flat_logit_threshold)
                        dz = _centre(z) - _centre(zero_logits)
                        path_fraction = float(torch.dot(dz, output_direction) / output_n2)
                        geom = branch_geometry(h, u if control == "radial"
                                               else _last(angular_replacement).double()
                                               / float(a), float(a))
                        rows.append(ScaleScanRow(
                            model=model.name, locus=locus, site=site.label,
                            block=site.block, update_tap=site.update_tap,
                            control=control, alpha=float(a),
                            alpha_star=alpha_star, q=float(a) / alpha_star,
                            path_fraction=path_fraction,
                            d_shape_from_natural=shape.d_shape,
                            log_beta_from_natural=shape.log_beta,
                            theta=geom.theta,
                            theta_fraction=(0.0 if max_angle < 1e-12 else
                                            abs(geom.theta) / max_angle),
                            update_norm=float(unorm), host_norm=float(hnorm),
                            control_angle=control_angle, dependency_keys=keys))
    return rows


def scale_curve_collapse_gate(
    rows: Sequence[ScaleScanRow],
    *,
    required_models: Sequence[str],
    required_sites: Mapping[str, Sequence[str]],
    plateau_increment_margin: float,
    angular_specificity_margin: float,
    collapse_rmse_margin: float,
    transition_effect_margin: float = 0.10,
    log_q_boundary_margin: float = 0.50,
    plateau_remaining_fraction: float = 0.10,
    plateau_boundary_spread: float = 0.50,
    expected_alphas: Optional[Sequence[float]] = None,
    min_clusters: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Bounded 0-2 verdict with fields safe for adaptive downstream logic."""
    if not rows:
        raise RuntimeError("no scale-scan rows")
    expected = np.asarray(
        DEFAULT_LOG_SCALE_GRID if expected_alphas is None
        else tuple(float(x) for x in expected_alphas), dtype=float)
    if (expected.ndim != 1 or len(expected) < 8
            or not np.isfinite(expected).all() or np.any(expected <= 0)
            or np.any(np.diff(expected) <= 0)):
        raise ValueError("expected_alphas must be a finite increasing positive grid")
    observed_models = {r.model for r in rows}
    missing_models = sorted(set(required_models) - observed_models)
    missing_sites = {
        model: sorted(set(sites) - {r.site for r in rows if r.model == model})
        for model, sites in required_sites.items()
    }
    missing_sites = {k: v for k, v in missing_sites.items() if v}

    curves: dict[tuple[str, str, str, str], list[ScaleScanRow]] = {}
    for r in rows:
        curves.setdefault((r.model, r.site, r.locus, r.control), []).append(r)
    radial_curves = {k: sorted(v, key=lambda r: r.alpha)
                     for k, v in curves.items() if k[-1] == "radial"}
    angular_curves = {k[:-1]: sorted(v, key=lambda r: r.alpha)
                      for k, v in curves.items() if k[-1] == "equal_norm_angular"}
    grid_complete = True
    alpha_star_finite = True
    increments: list[Row] = []
    angular_gaps: list[Row] = []
    increments_by_site: dict[tuple[str, str], list[Row]] = {}
    angular_by_site: dict[tuple[str, str], list[Row]] = {}
    transitions: list[Row] = []
    midpoints: list[Row] = []
    transitions_by_site: dict[tuple[str, str], list[Row]] = {}
    midpoints_by_site: dict[tuple[str, str], list[Row]] = {}
    midpoint_complete = True
    response_monotone = True
    plateau_boundaries: list[Row] = []
    plateau_boundaries_by_site: dict[tuple[str, str], list[Row]] = {}
    plateau_boundary_complete = True
    per_curve = {}
    for key, curve in radial_curves.items():
        model, site, locus, _ = key
        grid = np.asarray([r.alpha for r in curve])
        grid_complete &= (
            grid.shape == expected.shape
            and np.allclose(grid, expected, rtol=1e-12, atol=1e-15))
        alpha_star_finite &= all(np.isfinite(r.alpha_star) and r.alpha_star > 0
                                 and abs(r.q - r.alpha / r.alpha_star) < 1e-10
                                 for r in curve)
        last, prev = curve[-1], curve[-2]
        keys = last.dependency_keys or (f"locus:{locus}",)
        y = np.asarray([r.path_fraction for r in curve], dtype=float)
        total_effect = float(y[-1] - y[0])
        transitions.append(Row(abs(total_effect), keys=keys))
        transitions_by_site.setdefault((model, site), []).append(
            Row(abs(total_effect), keys=keys))
        q_midpoint = None
        q_plateau = None
        if abs(total_effect) > 1e-12 and all(r.q > 0 for r in curve):
            normalized = (y - y[0]) / total_effect
            response_monotone &= bool(np.all(
                np.diff(normalized) >= -plateau_increment_margin))
            crossing = np.flatnonzero(normalized >= 0.5)
            if len(crossing):
                j = int(crossing[0])
                if j == 0:
                    log_q50 = float(np.log10(curve[0].q))
                else:
                    lo_y, hi_y = normalized[j - 1], normalized[j]
                    weight = (0.5 - lo_y) / max(float(hi_y - lo_y), 1e-30)
                    lo_q, hi_q = np.log10(curve[j - 1].q), np.log10(curve[j].q)
                    log_q50 = float(lo_q + np.clip(weight, 0.0, 1.0) * (hi_q - lo_q))
                q_midpoint = float(10 ** log_q50)
                midpoint = Row(log_q50, keys=keys)
                midpoints.append(midpoint)
                midpoints_by_site.setdefault((model, site), []).append(midpoint)
            for j in range(len(curve)):
                tail = normalized[j:]
                if (np.all(np.abs(1.0 - tail) <= plateau_remaining_fraction)
                        and (len(tail) < 2 or np.all(
                            np.abs(np.diff(tail)) <= plateau_remaining_fraction))):
                    log_q_plateau = float(np.log10(curve[j].q))
                    q_plateau = float(curve[j].q)
                    boundary = Row(log_q_plateau, keys=keys)
                    plateau_boundaries.append(boundary)
                    plateau_boundaries_by_site.setdefault(
                        (model, site), []).append(boundary)
                    break
        midpoint_complete &= q_midpoint is not None
        plateau_boundary_complete &= q_plateau is not None
        increment_row = Row(last.path_fraction - prev.path_fraction, keys=keys)
        increments.append(increment_row)
        increments_by_site.setdefault((model, site), []).append(increment_row)
        angular = angular_curves.get((model, site, locus))
        if angular is None or len(angular) != len(curve):
            grid_complete = False
        else:
            angular_row = Row(
                last.path_fraction - angular[-1].path_fraction, keys=keys)
            angular_gaps.append(angular_row)
            angular_by_site.setdefault((model, site), []).append(angular_row)
        per_curve[f"{model}:{site}:{locus}"] = {
            "alpha_star": last.alpha_star,
            "q": [r.q for r in curve],
            "radial_path_fraction": [r.path_fraction for r in curve],
            "angular_path_fraction": ([] if angular is None else
                                      [r.path_fraction for r in angular]),
            "transition_effect": abs(total_effect),
            "q50": q_midpoint,
            "first_sustained_plateau_q": q_plateau,
        }
    if not increments or not angular_gaps:
        raise RuntimeError("scale gate lacks radial/angular paired curves")
    e_inc = cluster_bootstrap(increments, min_clusters=min_clusters,
                              n_boot=n_boot, alpha=alpha)
    e_angle = cluster_bootstrap(angular_gaps, min_clusters=min_clusters,
                                n_boot=n_boot, alpha=alpha)
    e_transition = cluster_bootstrap(
        transitions, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
    e_midpoint = (cluster_bootstrap(
        midpoints, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        if midpoint_complete and midpoints else None)
    e_plateau_boundary = (cluster_bootstrap(
        plateau_boundaries, min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
        if plateau_boundary_complete and plateau_boundaries else None)
    boundary_common = bool(
        e_midpoint is not None
        and e_midpoint.lo > -log_q_boundary_margin
        and e_midpoint.hi < log_q_boundary_margin)
    plateau = (e_inc.lo > -plateau_increment_margin
               and e_inc.hi < plateau_increment_margin)
    site_gates = {}
    site_plateau_all = True
    site_angular_all = True
    site_transition_all = True
    site_boundary_all = True
    site_plateau_boundary_estimates = {}
    expected_site_keys = {
        (model, site) for model, sites in required_sites.items() for site in sites
    }
    for model, site in sorted(expected_site_keys):
        inc_rows = increments_by_site.get((model, site), [])
        ang_rows = angular_by_site.get((model, site), [])
        trans_rows = transitions_by_site.get((model, site), [])
        midpoint_rows = midpoints_by_site.get((model, site), [])
        plateau_boundary_rows = plateau_boundaries_by_site.get((model, site), [])
        if (not inc_rows or not ang_rows or not trans_rows or not midpoint_rows
                or not plateau_boundary_rows):
            site_gates[f"{model}:{site}"] = {
                "present": False, "plateau": False,
                "angular_specificity": False, "transition": False,
                "q50_near_one": False}
            site_plateau_all = site_angular_all = False
            site_transition_all = site_boundary_all = False
            continue
        inc_est = cluster_bootstrap(
            inc_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        ang_est = cluster_bootstrap(
            ang_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        trans_est = cluster_bootstrap(
            trans_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        midpoint_est = cluster_bootstrap(
            midpoint_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        plateau_boundary_est = cluster_bootstrap(
            plateau_boundary_rows, min_clusters=min_clusters,
            n_boot=n_boot, alpha=alpha)
        site_plateau_boundary_estimates[(model, site)] = plateau_boundary_est
        inc_ok = (inc_est.lo > -plateau_increment_margin
                  and inc_est.hi < plateau_increment_margin)
        ang_ok = ang_est.lo > angular_specificity_margin
        trans_ok = trans_est.lo > transition_effect_margin
        midpoint_ok = (
            midpoint_est.lo > -log_q_boundary_margin
            and midpoint_est.hi < log_q_boundary_margin)
        site_plateau_all &= inc_ok
        site_angular_all &= ang_ok
        site_transition_all &= trans_ok
        site_boundary_all &= midpoint_ok
        site_gates[f"{model}:{site}"] = {
            "present": True, "plateau": inc_ok,
            "angular_specificity": ang_ok,
            "transition": trans_ok,
            "q50_near_one": midpoint_ok,
            "last_increment": inc_est.as_row(),
            "radial_minus_equal_norm_angular": ang_est.as_row(),
            "transition_effect": trans_est.as_row(),
            "log10_q50": midpoint_est.as_row(),
            "log10_first_sustained_plateau_q": plateau_boundary_est.as_row(),
        }

    shared_plateau_boundary = bool(e_plateau_boundary is not None)
    if e_plateau_boundary is not None:
        for estimate in site_plateau_boundary_estimates.values():
            deviation_bound = max(
                abs(estimate.lo - e_plateau_boundary.point),
                abs(estimate.hi - e_plateau_boundary.point),
            )
            shared_plateau_boundary &= deviation_bound <= plateau_boundary_spread
    plateau_boundary_values = [float(row.value) for row in plateau_boundaries]
    midpoint_values = [float(row.value) for row in midpoints]
    plateau_boundary_max_spread = (
        float(max(plateau_boundary_values) - min(plateau_boundary_values))
        if plateau_boundary_values else float("inf"))
    midpoint_max_abs = (
        float(max(abs(value) for value in midpoint_values))
        if midpoint_values else float("inf"))
    shared_plateau_boundary &= (
        plateau_boundary_max_spread <= plateau_boundary_spread)

    # Interpolate every radial curve on the common log-q support and compare
    # each to the grand master curve.  This tests collapse; it does not force
    # curves with disjoint dose support to appear comparable.
    log_ranges = [(np.log10(c[0].q), np.log10(c[-1].q))
                  for c in radial_curves.values() if c[0].q > 0]
    lo = max(a for a, _ in log_ranges)
    hi = min(b for _, b in log_ranges)
    collapse_available = bool(np.isfinite(lo) and np.isfinite(hi) and lo < hi)
    rmse = float("inf")
    rmse_interval = None
    common_q = []
    if collapse_available:
        common_logq = np.linspace(lo, hi, 25)
        matrix = []
        for curve in radial_curves.values():
            x = np.log10([r.q for r in curve])
            y = np.asarray([r.path_fraction for r in curve], dtype=float)
            matrix.append(np.interp(common_logq, x, y))
        M = np.asarray(matrix)
        master = M.mean(0)
        rmse = float(np.sqrt(np.mean((M - master[None, :]) ** 2)))
        rmse_rows = []
        for curve, values in zip(radial_curves.values(), M):
            locus = curve[-1].locus
            keys = curve[-1].dependency_keys or (f"locus:{locus}",)
            rmse_rows.append(Row(
                float(np.sqrt(np.mean((values - master) ** 2))), keys=keys))
        rmse_interval = cluster_bootstrap(
            rmse_rows, min_clusters=min_clusters, n_boot=n_boot, alpha=alpha)
        common_q = (10 ** common_logq).tolist()
    collapse = bool(
        collapse_available and rmse <= collapse_rmse_margin
        and rmse_interval is not None and rmse_interval.hi <= collapse_rmse_margin)
    checks = {
        "required_models_present": not missing_models,
        "required_sites_present": not missing_sites,
        "log_spaced_grid_complete": bool(grid_complete),
        "alpha_star_finite_and_q_exact": alpha_star_finite,
        "radial_curve_plateaus": plateau and site_plateau_all,
        "radial_transition_is_nontrivial": (
            e_transition.lo > transition_effect_margin and site_transition_all),
        "q50_is_near_dominance_boundary": (
            midpoint_complete and response_monotone
            and boundary_common and site_boundary_all
            and midpoint_max_abs <= log_q_boundary_margin),
        "first_sustained_plateau_is_shared": (
            plateau_boundary_complete and shared_plateau_boundary),
        "equal_norm_angular_is_different": (
            e_angle.lo > angular_specificity_margin and site_angular_all),
        "normalized_curves_collapse": collapse,
    }
    structural = (
        checks["required_models_present"]
        and checks["required_sites_present"]
        and checks["log_spaced_grid_complete"]
        and checks["alpha_star_finite_and_q_exact"]
    )
    transition_bounded_null = e_transition.hi < transition_effect_margin
    if all(checks.values()):
        status, conclusion_code = "supported", "shared_normalized_transition"
    elif not structural:
        status, conclusion_code = "unresolved", "scale_measurement_incomplete"
    elif transition_bounded_null:
        status, conclusion_code = "equivalent", "no_common_scale_effect"
    elif (checks["radial_curve_plateaus"]
          and checks["equal_norm_angular_is_different"]):
        status, conclusion_code = "mixed", "site_specific_transition"
    else:
        status, conclusion_code = "refuted", "no_shared_plateau"
    return {
        "passed": all(checks.values()),
        "status": status,
        "conclusion_code": conclusion_code,
        "checks": checks,
        "missing_models": missing_models, "missing_sites": missing_sites,
        "required_models": list(required_models),
        "required_sites": {
            str(model): list(sites)
            for model, sites in sorted(required_sites.items())
        },
        "plateau_last_increment": e_inc.as_row(),
        "radial_minus_equal_norm_angular": e_angle.as_row(),
        "curve_collapse": {
            "available": collapse_available, "rmse": rmse,
            "rmse_interval": (
                None if rmse_interval is None else rmse_interval.as_row()),
            "margin": collapse_rmse_margin, "common_q": common_q,
        },
        "transition_effect": e_transition.as_row(),
        "log10_q50": None if e_midpoint is None else e_midpoint.as_row(),
        "log10_first_sustained_plateau_q": (
            None if e_plateau_boundary is None else
            e_plateau_boundary.as_row()),
        "boundary_heterogeneity": {
            "q50_max_abs_log10_distance_from_one": midpoint_max_abs,
            "first_plateau_max_log10_spread": plateau_boundary_max_spread,
        },
        "per_model_site_gates": site_gates,
        "per_curve": per_curve,
        "alpha_doses": expected.tolist(),
        "margins_used": {
            "plateau_increment_margin": plateau_increment_margin,
            "angular_specificity_margin": angular_specificity_margin,
            "collapse_rmse_margin": collapse_rmse_margin,
            "transition_effect_margin": transition_effect_margin,
            "log_q_boundary_margin": log_q_boundary_margin,
            "plateau_remaining_fraction": plateau_remaining_fraction,
            "plateau_boundary_spread": plateau_boundary_spread,
            "min_clusters": min_clusters,
            "n_boot": n_boot,
            "alpha": alpha,
        },
        "normalization": {
            "alpha_star": "||host|| / ||update||",
            "q": "alpha / alpha_star = alpha||update||/||host||",
            "output": "centred-logit path fraction from alpha=0 to alpha=1",
        },
        "downstream_contract": (
            "Do not open motif/SAE/application waves unless passed is true, "
            "or explicitly record which failed claim was downgraded."),
    }

"""EXP3 Phase 5 — learned *delta* transport after raw characterization.

This module is intentionally downstream of Steps 1--10.  It never learns to
reconstruct an absolute hidden state.  Every target is an intervention delta,
and the primary evidence is prediction on a provenance-locked held-out split
followed by a real predicted-delta rescue.  Re-inserting the cached actual
delta is explicitly rejected as primary evidence.

The module is model agnostic: callers collect tensors with ``Evo2Runner`` and
pass them here.  Consequently all mathematics can be tested without loading an
Evo 2 checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .exp2_api import Row, cluster_bootstrap


SplitRole = Literal["train", "dev", "locked"]
ControlFamily = Literal[
    "dose_rank_matched_random",
    "shuffled_prediction",
    "wrong_layer",
    "wrong_pair",
]

REQUIRED_PREDICTED_DELTA_CONTROLS: tuple[ControlFamily, ...] = (
    "dose_rank_matched_random",
    "shuffled_prediction",
    "wrong_layer",
)


# ---------------------------------------------------------------------------
# Provenance: every learned map has train/dev/locked separation
# ---------------------------------------------------------------------------


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(_digest({"shape": list(array.shape)}).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _ordered_units_sha256(unit_ids: Sequence[str]) -> str:
    return _digest({"ordered_unit_ids": list(map(str, unit_ids))})


@dataclass(frozen=True)
class SplitRecord:
    name: str
    role: SplitRole
    unit_ids: tuple[str, ...]
    source_sha256: str
    target_sha256: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("split name is empty")
        if self.role not in {"train", "dev", "locked"}:
            raise ValueError(f"invalid split role {self.role!r}")
        if not self.unit_ids or len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError(f"{self.role} unit ids must be non-empty and unique")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef"
                                                for c in self.source_sha256.lower()):
            raise ValueError(f"{self.role} split requires a full SHA-256 provenance hash")
        if len(self.target_sha256) != 64 or any(c not in "0123456789abcdef"
                                                for c in self.target_sha256.lower()):
            raise ValueError(f"{self.role} split requires a target SHA-256 commitment")


@dataclass(frozen=True)
class TransportProvenance:
    train: SplitRecord
    dev: SplitRecord
    locked: SplitRecord
    feature_spec: str
    target_spec: str
    sealed_before_locked: bool = True
    provenance_sha256: str = ""

    def __post_init__(self) -> None:
        for expected, rec in (("train", self.train), ("dev", self.dev),
                              ("locked", self.locked)):
            rec.validate()
            if rec.role != expected:
                raise ValueError(f"expected {expected} record, got {rec.role}")
        sets = [set(self.train.unit_ids), set(self.dev.unit_ids), set(self.locked.unit_ids)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("train/dev/locked unit ids must be disjoint")
        if not self.sealed_before_locked:
            raise ValueError("feature/target specification must be sealed before locked access")
        forbidden = ("absolute_state", "absolute hidden", "cached_actual_delta",
                     "cached actual delta")
        text = f"{self.feature_spec} {self.target_spec}".lower()
        if any(term in text for term in forbidden):
            raise ValueError("EXP3 Phase 5 permits delta transport only, not absolute/cached targets")
        computed = self.compute_sha256()
        if self.provenance_sha256 and self.provenance_sha256 != computed:
            raise ValueError("transport provenance hash mismatch")
        object.__setattr__(self, "provenance_sha256", computed)

    def compute_sha256(self) -> str:
        return _digest({
            "train": self.train.__dict__, "dev": self.dev.__dict__,
            "locked": self.locked.__dict__, "feature_spec": self.feature_spec,
            "target_spec": self.target_spec, "sealed": self.sealed_before_locked,
        })

    def assert_units(self, role: SplitRole, unit_ids: Sequence[str]) -> None:
        record = getattr(self, role)
        observed = tuple(map(str, unit_ids))
        if len(set(observed)) != len(observed):
            raise ValueError(f"duplicate units in {role} batch")
        if observed != record.unit_ids:
            missing = sorted(set(record.unit_ids) - set(observed))[:5]
            extra = sorted(set(observed) - set(record.unit_ids))[:5]
            mismatch = next(
                (i for i, pair in enumerate(zip(observed, record.unit_ids))
                 if pair[0] != pair[1]),
                min(len(observed), len(record.unit_ids)),
            )
            raise ValueError(
                f"{role} batch/provenance order mismatch at row {mismatch}; "
                f"missing={missing}, extra={extra}"
            )


# ---------------------------------------------------------------------------
# Randomised perturbation batches and radial/tangential geometry
# ---------------------------------------------------------------------------


def intervention_source_sha256(unit_ids: Sequence[str], base: np.ndarray,
                               delta_in: np.ndarray) -> str:
    return _digest({
        "kind": "intervention_input_v1",
        "ordered_unit_ids": list(map(str, unit_ids)),
        "base_sha256": _array_sha256(base),
        "delta_in_sha256": _array_sha256(delta_in),
    })


def intervention_target_sha256(delta_out: np.ndarray,
                               output_base: Optional[np.ndarray] = None) -> str:
    return _digest({
        "kind": "intervention_target_v1",
        "delta_out_sha256": _array_sha256(delta_out),
        "output_base_sha256": (
            _array_sha256(output_base) if output_base is not None else ""
        ),
    })


@dataclass
class InterventionBatch:
    unit_ids: tuple[str, ...]
    role: SplitRole
    base: np.ndarray
    delta_in: np.ndarray
    delta_out: np.ndarray
    randomization_seed: int
    intervention_id: tuple[str, ...] = ()
    output_base: Optional[np.ndarray] = None

    def validate(self, provenance: TransportProvenance) -> None:
        provenance.assert_units(self.role, self.unit_ids)
        self.base = _matrix(self.base, "base")
        self.delta_in = _matrix(self.delta_in, "delta_in")
        self.delta_out = _matrix(self.delta_out, "delta_out")
        n = len(self.unit_ids)
        if any(x.shape[0] != n for x in (self.base, self.delta_in, self.delta_out)):
            raise ValueError("batch tensors and unit_ids have different row counts")
        if self.base.shape != self.delta_in.shape:
            raise ValueError("base and delta_in must have the same shape")
        if not np.isfinite(self.base).all() or not np.isfinite(self.delta_in).all() \
                or not np.isfinite(self.delta_out).all():
            raise ValueError("non-finite intervention batch")
        if np.any(np.linalg.norm(self.base, axis=1) <= 1e-12):
            raise ValueError("radial/tangential decomposition needs non-zero base states")
        if np.any(np.linalg.norm(self.delta_in, axis=1) <= 1e-12):
            raise ValueError("zero perturbations do not identify a transport map")
        if self.intervention_id and len(self.intervention_id) != n:
            raise ValueError("intervention_id must align with rows")
        if not isinstance(self.randomization_seed, (int, np.integer)):
            raise ValueError("randomized intervention batch requires an explicit integer seed")
        if self.output_base is not None:
            self.output_base = _matrix(self.output_base, "output_base")
            if self.output_base.shape != self.delta_out.shape:
                raise ValueError("output_base must align with delta_out for output geometry")
        record = getattr(provenance, self.role)
        actual_source = intervention_source_sha256(
            self.unit_ids, self.base, self.delta_in)
        actual_target = intervention_target_sha256(
            self.delta_out, self.output_base)
        if record.source_sha256 != actual_source:
            raise ValueError(f"{self.role} intervention input/source hash mismatch")
        if record.target_sha256 != actual_target:
            raise ValueError(f"{self.role} intervention target hash mismatch")


def _matrix(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be [n,d], got {a.shape}")
    return a


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product without macOS NumPy/Accelerate's spurious matmul warnings."""
    return np.einsum("...i,ij->...j", a, b, optimize=True)


@dataclass(frozen=True)
class RadialTangential:
    radial: np.ndarray
    tangential: np.ndarray
    radial_coefficient: np.ndarray


def radial_tangential(base: np.ndarray, delta: np.ndarray) -> RadialTangential:
    """Orthogonally split delta relative to the native residual ray."""
    b, d = _matrix(base, "base"), _matrix(delta, "delta")
    if b.shape != d.shape:
        raise ValueError("base/delta shape mismatch")
    norm = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("cannot define a residual ray at zero norm")
    u = b / norm
    coeff = np.sum(d * u, axis=1, keepdims=True)
    radial = coeff * u
    tangent = d - radial
    return RadialTangential(radial, tangent, coeff[:, 0])


@dataclass(frozen=True)
class SphericalTransition:
    log_radius_ratio: np.ndarray
    tangent_log: np.ndarray
    angle: np.ndarray
    excluded: np.ndarray


def spherical_log_map(base: np.ndarray, target: np.ndarray,
                      *, antipodal_tol: float = 1e-6) -> SphericalTransition:
    """Log map on the unit sphere plus the separate log-radius transition.

    The tangent vector lies at ``base/||base||`` and has norm equal to the
    geodesic angle.  Near-antipodal pairs are non-identifiable and marked
    excluded rather than assigned an arbitrary great circle.
    """
    x, y = _matrix(base, "base"), _matrix(target, "target")
    if not 0 < antipodal_tol < 1:
        raise ValueError("antipodal_tol must lie in (0,1)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("spherical transition contains non-finite states")
    if x.shape != y.shape:
        raise ValueError("spherical transition shape mismatch")
    rx = np.linalg.norm(x, axis=1, keepdims=True)
    ry = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(rx <= 1e-12) or np.any(ry <= 1e-12):
        raise ValueError("spherical log map is undefined at zero radius")
    u, v = x / rx, y / ry
    dot = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)
    excluded = (np.pi - angle) < antipodal_tol
    orth = v - dot[:, None] * u
    orth_norm = np.linalg.norm(orth, axis=1, keepdims=True)
    tangent = np.zeros_like(orth)
    usable = (~excluded) & (orth_norm[:, 0] > 1e-12)
    tangent[usable] = orth[usable] / orth_norm[usable] * angle[usable, None]
    tangent[excluded] = np.nan
    return SphericalTransition(
        np.log(ry[:, 0] / rx[:, 0]), tangent, angle, excluded)


def randomized_delta_design(base: np.ndarray, *, radial_sd: float,
                            tangential_sd: float, seed: int) -> np.ndarray:
    """Generate a reproducible full-rank perturbation design around each ray."""
    x = _matrix(base, "base")
    if radial_sd <= 0 or tangential_sd <= 0:
        raise ValueError("both radial and tangential randomisation scales must be positive")
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) <= 1e-12):
        raise ValueError("randomized perturbations require finite non-zero base states")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=x.shape)
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    z = rng.normal(size=(len(x), 1))
    tangent = noise - np.sum(noise * u, axis=1, keepdims=True) * u
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-12)
    return radial_sd * z * u + tangential_sd * rng.normal(size=(len(x), 1)) * tangent


# ---------------------------------------------------------------------------
# Reduced-rank delta regression and system identification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReducedRankRegression:
    coef: np.ndarray
    intercept: np.ndarray
    rank: int
    ridge: float
    train_roles: tuple[str, ...] = ("train", "dev")

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = _matrix(x, "features")
        if xx.shape[1] != self.coef.shape[0]:
            raise ValueError("feature width does not match fitted delta map")
        return _mm(xx, self.coef) + self.intercept


def fit_reduced_rank(x: np.ndarray, y: np.ndarray, *, rank: int,
                     ridge: float) -> ReducedRankRegression:
    """Ridge OLS followed by the classical reduced-rank response projection."""
    xx, yy = _matrix(x, "x"), _matrix(y, "y")
    if xx.shape[0] != yy.shape[0] or xx.shape[0] < 2:
        raise ValueError("reduced-rank fit needs aligned rows and n>=2")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    xm, ym = xx.mean(0), yy.mean(0)
    xc, yc = xx - xm, yy - ym
    if xx.shape[1] == 0:
        return ReducedRankRegression(np.zeros((0, yy.shape[1])), ym, 0, ridge)
    gram = np.einsum("ni,nj->ij", xc, xc, optimize=True) \
        + ridge * np.eye(xx.shape[1])
    cross = np.einsum("ni,nj->ij", xc, yc, optimize=True)
    try:
        bols = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        # ``ridge=0`` is allowed for a pre-registered grid.  Rank-deficient
        # discovery features therefore use the deterministic minimum-norm OLS
        # solution instead of failing according to the linked LAPACK build.
        bols = np.linalg.lstsq(gram, cross, rcond=None)[0]
    fitted = _mm(xc, bols)
    _, _, vh = np.linalg.svd(fitted, full_matrices=False)
    r = min(max(int(rank), 1), vh.shape[0], yy.shape[1])
    vr = vh[:r].T
    coef = _mm(_mm(bols, vr), vr.T)
    return ReducedRankRegression(coef, ym - _mm(xm[None, :], coef)[0], r, float(ridge))


def delta_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, p = _matrix(y_true, "y_true"), _matrix(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError("metric shape mismatch")
    den = float(np.sum((y - y.mean(0)) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / max(den, 1e-30))


def _select_rrr(train_x: np.ndarray, train_y: np.ndarray,
                dev_x: np.ndarray, dev_y: np.ndarray,
                ranks: Sequence[int], ridges: Sequence[float]) -> ReducedRankRegression:
    candidates = []
    for ridge in ridges:
        for rank in ranks:
            model = fit_reduced_rank(train_x, train_y, rank=int(rank), ridge=float(ridge))
            candidates.append((delta_r2(dev_y, model.predict(dev_x)), rank, ridge, model))
    if not candidates:
        raise ValueError("empty rank/ridge grid")
    # Accuracy first, then the simpler rank, then the stronger regularizer.
    candidates.sort(key=lambda z: (-z[0], z[1], -z[2]))
    return candidates[0][3]


@dataclass(frozen=True)
class DeltaTransportSystem:
    model: ReducedRankRegression
    input_width: int
    provenance_sha256: str
    model_sha256: str
    train_design_rank: int
    dev_design_rank: int
    design_width: int
    train_condition_number: float
    dev_condition_number: float

    def features(self, base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        rt = radial_tangential(base, delta)
        return np.concatenate([rt.radial, rt.tangential], axis=1)

    def predict_delta(self, base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return self.model.predict(self.features(base, delta))

    def predict_components(self, base: np.ndarray, delta: np.ndarray) -> dict[str, np.ndarray]:
        if self.train_design_rank != self.design_width \
                or self.dev_design_rank != self.design_width:
            raise RuntimeError(
                "radial/tangential components were not independently identified"
            )
        feat = self.features(base, delta)
        d = self.input_width
        zero = np.zeros_like(feat)
        radial = zero.copy(); radial[:, :d] = feat[:, :d]
        tangent = zero.copy(); tangent[:, d:] = feat[:, d:]
        # Subtract the intercept once so radial+tangential+intercept equals total.
        pr = self.model.predict(radial) - self.model.intercept
        pt = self.model.predict(tangent) - self.model.intercept
        return {"radial": pr, "tangential": pt,
                "total": pr + pt + self.model.intercept}


def fit_randomized_transport(train: InterventionBatch, dev: InterventionBatch,
                             provenance: TransportProvenance, *,
                             ranks: Sequence[int], ridges: Sequence[float],
                             max_condition_number: float = 1e8
                             ) -> DeltaTransportSystem:
    if train.role != "train" or dev.role != "dev":
        raise ValueError("system identification requires train then dev batches")
    train.validate(provenance); dev.validate(provenance)
    if train.delta_in.shape[1] != dev.delta_in.shape[1] \
            or train.delta_out.shape[1] != dev.delta_out.shape[1]:
        raise ValueError("train/dev transport widths differ")
    if max_condition_number <= 1 or not np.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than one")
    tr = radial_tangential(train.base, train.delta_in)
    dv = radial_tangential(dev.base, dev.delta_in)
    tx = np.concatenate([tr.radial, tr.tangential], 1)
    dx = np.concatenate([dv.radial, dv.tangential], 1)

    def diagnostics(design: np.ndarray, label: str) -> tuple[int, float]:
        centered = design - design.mean(0, keepdims=True)
        width = centered.shape[1]
        if centered.shape[0] - 1 < width:
            raise ValueError(
                f"{label} split has too few independent rows to identify the "
                f"{width}-column radial/tangential design"
            )
        singular = np.linalg.svd(centered, compute_uv=False)
        rank = int(np.linalg.matrix_rank(centered))
        condition = float(singular[0] / max(singular[-1], 1e-30))
        if rank != width:
            raise ValueError(
                f"{label} radial/tangential design is rank deficient "
                f"({rank}/{width}) even if raw delta is full-rank"
            )
        if condition > max_condition_number:
            raise ValueError(
                f"{label} radial/tangential design is ill-conditioned "
                f"({condition:.3g} > {max_condition_number:.3g})"
            )
        return rank, condition

    train_rank, train_condition = diagnostics(tx, "train")
    dev_rank, dev_condition = diagnostics(dx, "development")
    model = _select_rrr(tx, train.delta_out, dx, dev.delta_out, ranks, ridges)
    mid = _digest({"coef": model.coef.tolist(), "intercept": model.intercept.tolist(),
                   "rank": model.rank, "ridge": model.ridge,
                   "provenance": provenance.provenance_sha256})
    return DeltaTransportSystem(
        model, train.delta_in.shape[1], provenance.provenance_sha256, mid,
        train_rank, dev_rank, tx.shape[1], train_condition, dev_condition,
    )


@dataclass(frozen=True)
class HeldoutDeltaMetrics:
    r2: float
    cosine_mean: float
    relative_error_mean: float
    radial_r2: float
    tangential_r2: float
    n_units: int


def evaluate_locked_transport(system: DeltaTransportSystem, locked: InterventionBatch,
                              provenance: TransportProvenance) -> HeldoutDeltaMetrics:
    if locked.role != "locked":
        raise ValueError("held-out metrics require the locked split")
    locked.validate(provenance)
    if system.provenance_sha256 != provenance.provenance_sha256:
        raise ValueError("model/provenance mismatch")
    pred = system.predict_delta(locked.base, locked.delta_in)
    y = locked.delta_out
    dot = np.sum(pred * y, axis=1)
    cos = dot / (np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1)).clip(1e-30)
    rel = np.linalg.norm(pred - y, axis=1) / np.linalg.norm(y, axis=1).clip(1e-30)
    # Decompose the OUTPUT delta only when its own native output base was
    # supplied.  The input base is never reused merely because widths happen
    # to match.
    if locked.output_base is not None:
        yt = radial_tangential(locked.output_base, y)
        pt = radial_tangential(locked.output_base, pred)
        rr = delta_r2(yt.radial, pt.radial)
        tr = delta_r2(yt.tangential, pt.tangential)
    else:
        rr = tr = float("nan")
    return HeldoutDeltaMetrics(delta_r2(y, pred), float(np.mean(cos)),
                               float(np.mean(rel)), rr, tr, len(y))


# ---------------------------------------------------------------------------
# Native-factor M0--M4: b29 amplifier versus co-writer
# ---------------------------------------------------------------------------


def native_factor_source_sha256(
    unit_ids: Sequence[str], incoming_delta: np.ndarray, mixer_delta: np.ndarray,
    factor_a_delta: np.ndarray, factor_b_delta: np.ndarray,
    product_delta: np.ndarray,
) -> str:
    return _digest({
        "kind": "native_factor_input_v1",
        "ordered_unit_ids": list(map(str, unit_ids)),
        "incoming_delta_sha256": _array_sha256(incoming_delta),
        "mixer_delta_sha256": _array_sha256(mixer_delta),
        "factor_a_delta_sha256": _array_sha256(factor_a_delta),
        "factor_b_delta_sha256": _array_sha256(factor_b_delta),
        "product_delta_sha256": _array_sha256(product_delta),
    })


def native_factor_target_sha256(outgoing_delta: np.ndarray) -> str:
    return _array_sha256(outgoing_delta)


@dataclass
class NativeFactorBatch:
    unit_ids: tuple[str, ...]
    role: SplitRole
    incoming_delta: np.ndarray
    mixer_delta: np.ndarray
    factor_a_delta: np.ndarray
    factor_b_delta: np.ndarray
    product_delta: np.ndarray
    outgoing_delta: np.ndarray

    def validate(self, provenance: TransportProvenance) -> None:
        provenance.assert_units(self.role, self.unit_ids)
        arrays = [self.incoming_delta, self.mixer_delta, self.factor_a_delta,
                  self.factor_b_delta, self.product_delta, self.outgoing_delta]
        arrays = [_matrix(a, "native factor") for a in arrays]
        if any(a.shape[0] != len(self.unit_ids) for a in arrays):
            raise ValueError("native factor rows do not align")
        (self.incoming_delta, self.mixer_delta, self.factor_a_delta,
         self.factor_b_delta, self.product_delta, self.outgoing_delta) = arrays
        record = getattr(provenance, self.role)
        actual_source = native_factor_source_sha256(
            self.unit_ids, self.incoming_delta, self.mixer_delta,
            self.factor_a_delta, self.factor_b_delta, self.product_delta)
        if record.source_sha256 != actual_source:
            raise ValueError(f"{self.role} native-factor input/source hash mismatch")
        if record.target_sha256 != native_factor_target_sha256(self.outgoing_delta):
            raise ValueError(f"{self.role} native-factor target hash mismatch")

    def features(self) -> dict[str, np.ndarray]:
        n = len(self.unit_ids)
        one = np.ones((n, 1), dtype=np.float64)
        m1 = np.concatenate([one, self.incoming_delta], 1)
        m2 = np.concatenate([m1, self.mixer_delta], 1)
        m3 = np.concatenate([m2, self.factor_a_delta, self.factor_b_delta], 1)
        m4 = np.concatenate([m3, self.product_delta], 1)
        return {"M0": one, "M1": m1, "M2": m2, "M3": m3, "M4": m4}


@dataclass(frozen=True)
class NativeFactorLadder:
    models: Mapping[str, ReducedRankRegression]
    dev_r2: Mapping[str, float]
    provenance_sha256: str

    def locked_scores(self, batch: NativeFactorBatch,
                      provenance: TransportProvenance) -> dict:
        if batch.role != "locked":
            raise ValueError("native-factor claims require locked data")
        batch.validate(provenance)
        if self.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("native-factor model/provenance mismatch")
        feats = batch.features()
        scores = {name: delta_r2(batch.outgoing_delta, model.predict(feats[name]))
                  for name, model in self.models.items()}
        return {
            "r2": scores,
            "amplifier_increment_M2_minus_M1": scores["M2"] - scores["M1"],
            "factor_increment_M3_minus_M2": scores["M3"] - scores["M2"],
            # M3 already contains both factor main effects.  Only M4-M3 is
            # attributable to the registered product/co-write term.  M4-M2
            # would silently count the factor main effects as co-writing.
            "product_cowriter_increment_M4_minus_M3": scores["M4"] - scores["M3"],
            "expanded_increment_M4_minus_M2": scores["M4"] - scores["M2"],
            "reading": "observational held-out attribution; causal co-writer requires predicted-delta rescue",
        }


def native_factor_verdict(ladder: NativeFactorLadder, locked: NativeFactorBatch,
                          provenance: TransportProvenance, *,
                          amplifier_margin: float, cowriter_margin: float,
                          m4_min_r2: float) -> dict:
    """Locked M0--M4 reading; causal language remains gated by rescue."""
    if min(amplifier_margin, cowriter_margin, m4_min_r2) < 0:
        raise ValueError("native-factor margins must be development-frozen and non-negative")
    score = ladder.locked_scores(locked, provenance)
    checks = {
        "b29_amplifier_increment":
            score["amplifier_increment_M2_minus_M1"] > amplifier_margin,
        "b29_product_cowriter_increment":
            score["product_cowriter_increment_M4_minus_M3"] > cowriter_margin,
        "M4_absolute_fidelity": score["r2"]["M4"] > m4_min_r2,
    }
    return {
        **score, "checks": checks,
        "observational_role": (
            "amplifier+co-writer candidate" if all(checks.values())
            else "native-factor ladder does not separate both roles"),
        "causal_role": "withheld until the same predicted component passes locked rescue",
    }


def fit_native_factor_ladder(train: NativeFactorBatch, dev: NativeFactorBatch,
                             provenance: TransportProvenance, *,
                             ranks: Sequence[int], ridges: Sequence[float]) -> NativeFactorLadder:
    if train.role != "train" or dev.role != "dev":
        raise ValueError("M0--M4 selection requires train/dev")
    train.validate(provenance); dev.validate(provenance)
    tf, df = train.features(), dev.features()
    models, scores = {}, {}
    for name in ("M0", "M1", "M2", "M3", "M4"):
        model = _select_rrr(tf[name], train.outgoing_delta, df[name], dev.outgoing_delta,
                            ranks, ridges)
        models[name] = model
        scores[name] = delta_r2(dev.outgoing_delta, model.predict(df[name]))
    return NativeFactorLadder(models, scores, provenance.provenance_sha256)


# ---------------------------------------------------------------------------
# One-sided HCL impulse/lag system identification
# ---------------------------------------------------------------------------


def lag_source_sha256(unit_ids: Sequence[str], delta_in: np.ndarray) -> str:
    return _digest({
        "kind": "lag_input_v1",
        "ordered_unit_ids": list(map(str, unit_ids)),
        "delta_in_sha256": _array_sha256(delta_in),
    })


def lag_target_sha256(delta_out: np.ndarray) -> str:
    return _array_sha256(delta_out)


@dataclass
class LagBatch:
    unit_ids: tuple[str, ...]
    role: SplitRole
    delta_in: np.ndarray       # [n, length, d_in]
    delta_out: np.ndarray      # [n, length, d_out]

    def validate(self, provenance: TransportProvenance) -> None:
        provenance.assert_units(self.role, self.unit_ids)
        self.delta_in = np.asarray(self.delta_in, dtype=np.float64)
        self.delta_out = np.asarray(self.delta_out, dtype=np.float64)
        if self.delta_in.ndim != 3 or self.delta_out.ndim != 3:
            raise ValueError("lag batches must be [n,length,d]")
        if self.delta_in.shape[:2] != self.delta_out.shape[:2]:
            raise ValueError("lag input/output sequence axes differ")
        if self.delta_in.shape[0] != len(self.unit_ids):
            raise ValueError("lag rows do not align to provenance")
        if not np.isfinite(self.delta_in).all() or not np.isfinite(self.delta_out).all():
            raise ValueError("lag tensors must be finite")
        record = getattr(provenance, self.role)
        if record.source_sha256 != lag_source_sha256(self.unit_ids, self.delta_in):
            raise ValueError(f"{self.role} lag input/source hash mismatch")
        if record.target_sha256 != lag_target_sha256(self.delta_out):
            raise ValueError(f"{self.role} lag target hash mismatch")


def _causal_lag_design(x: np.ndarray, max_lag: int) -> np.ndarray:
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    n, length, d = x.shape
    out = np.zeros((n, length, (max_lag + 1) * d), dtype=np.float64)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[:, :, :d] = x
        elif lag < length:
            out[:, lag:, lag * d:(lag + 1) * d] = x[:, :length - lag]
    return out


@dataclass(frozen=True)
class CausalImpulseKernel:
    model: ReducedRankRegression
    max_lag: int
    d_in: int
    d_out: int
    provenance_sha256: str
    model_sha256: str

    @property
    def kernel(self) -> np.ndarray:
        return self.model.coef.reshape(self.max_lag + 1, self.d_in, self.d_out)

    def predict(self, delta_in: np.ndarray) -> np.ndarray:
        x = np.asarray(delta_in, dtype=np.float64)
        design = _causal_lag_design(x, self.max_lag)
        return self.model.predict(design.reshape(-1, design.shape[-1])).reshape(
            x.shape[0], x.shape[1], self.d_out)

    def lag_energy(self) -> np.ndarray:
        return np.linalg.norm(self.kernel, axis=(1, 2))


def fit_hcl_impulse_kernel(train: LagBatch, dev: LagBatch,
                           provenance: TransportProvenance, *, max_lag: int,
                           ranks: Sequence[int], ridges: Sequence[float]) -> CausalImpulseKernel:
    if train.role != "train" or dev.role != "dev":
        raise ValueError("HCL kernel selection requires train/dev")
    train.validate(provenance); dev.validate(provenance)
    tx = _causal_lag_design(train.delta_in, max_lag)
    dx = _causal_lag_design(dev.delta_in, max_lag)
    model = _select_rrr(tx.reshape(-1, tx.shape[-1]),
                        train.delta_out.reshape(-1, train.delta_out.shape[-1]),
                        dx.reshape(-1, dx.shape[-1]),
                        dev.delta_out.reshape(-1, dev.delta_out.shape[-1]),
                        ranks, ridges)
    model_sha = _digest({
        "coef": model.coef.tolist(), "intercept": model.intercept.tolist(),
        "rank": model.rank, "ridge": model.ridge, "max_lag": max_lag,
        "d_in": train.delta_in.shape[-1], "d_out": train.delta_out.shape[-1],
        "provenance": provenance.provenance_sha256,
    })
    return CausalImpulseKernel(
        model, max_lag, train.delta_in.shape[-1], train.delta_out.shape[-1],
        provenance.provenance_sha256, model_sha,
    )


def evaluate_locked_hcl(kernel: CausalImpulseKernel, locked: LagBatch,
                        provenance: TransportProvenance) -> dict:
    if locked.role != "locked":
        raise ValueError("HCL impulse evidence requires locked data")
    locked.validate(provenance)
    if kernel.provenance_sha256 != provenance.provenance_sha256:
        raise ValueError("HCL kernel/provenance mismatch")
    pred = kernel.predict(locked.delta_in)
    per_unit = [delta_r2(locked.delta_out[i], pred[i]) for i in range(len(pred))]
    return {"delta_r2": delta_r2(locked.delta_out.reshape(-1, locked.delta_out.shape[-1]),
                                  pred.reshape(-1, pred.shape[-1])),
            "mean_unit_delta_r2": float(np.mean(per_unit)),
            "per_unit_delta_r2": per_unit,
            "lag_energy": kernel.lag_energy().tolist(),
            "causal_lags_only": list(range(kernel.max_lag + 1))}


# ---------------------------------------------------------------------------
# Minimal sparse DELTA transcoder
# ---------------------------------------------------------------------------


class SparseDeltaTranscoder(nn.Module):
    """Bias-free sparse map Δinput -> Δoutput; it cannot reconstruct a state."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, delta: Tensor) -> tuple[Tensor, Tensor]:
        if delta.ndim != 2:
            raise ValueError("delta transcoder input must be [n,d]")
        code = torch.relu(self.encoder(delta))
        return self.decoder(code), code


@dataclass(frozen=True)
class FittedSparseDeltaTranscoder:
    model: SparseDeltaTranscoder
    dev_mse: float
    dev_r2: float
    provenance_sha256: str
    artifact_sha256: str

    def predict_delta(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            y, _ = self.model(torch.as_tensor(x, dtype=torch.float32))
        return y.cpu().numpy().astype(np.float64)


def _state_dict_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(_digest({"shape": list(array.shape)}).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def fit_sparse_delta_transcoder(
    train: InterventionBatch, dev: InterventionBatch,
    provenance: TransportProvenance, *, hidden: int, l1: float = 1e-3,
    lr: float = 3e-3, epochs: int = 500, seed: int = 0,
) -> FittedSparseDeltaTranscoder:
    if train.role != "train" or dev.role != "dev":
        raise ValueError("delta transcoder fitting requires train/dev")
    train.validate(provenance); dev.validate(provenance)
    if hidden <= 0 or epochs <= 0 or l1 < 0:
        raise ValueError("invalid sparse delta transcoder hyperparameters")
    # Do not perturb the RNG stream of the enclosing Evo 2 experiment.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SparseDeltaTranscoder(train.delta_in.shape[1], hidden,
                                      train.delta_out.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        x = torch.as_tensor(train.delta_in, dtype=torch.float32)
        y = torch.as_tensor(train.delta_out, dtype=torch.float32)
        for _ in range(epochs):
            pred, code = model(x)
            loss = torch.mean((pred - y) ** 2) + l1 * torch.mean(torch.abs(code))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            dp, _ = model(torch.as_tensor(dev.delta_in, dtype=torch.float32))
    pred = dp.cpu().numpy().astype(np.float64)
    mse = float(np.mean((pred - dev.delta_out) ** 2))
    return FittedSparseDeltaTranscoder(
        model, mse, delta_r2(dev.delta_out, pred), provenance.provenance_sha256,
        _state_dict_sha256(model))


# ---------------------------------------------------------------------------
# Locked predicted-delta intervention and rescue verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportModelArtifact:
    source: str
    model_sha256: str
    provenance_sha256: str
    train_source_sha256: str
    dev_source_sha256: str
    train_unit_order_sha256: str
    dev_unit_order_sha256: str
    artifact_sha256: str

    def validate(self, provenance: TransportProvenance) -> None:
        if self.source not in {
            "cross_fitted_rrr", "cross_fitted_sparse_delta_transcoder",
            "cross_fitted_hcl_kernel",
        }:
            raise ValueError("transport artifact source is not an approved fitted model")
        if self.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("transport artifact/provenance mismatch")
        expected_fields = {
            "train_source_sha256": provenance.train.source_sha256,
            "dev_source_sha256": provenance.dev.source_sha256,
            "train_unit_order_sha256": _ordered_units_sha256(provenance.train.unit_ids),
            "dev_unit_order_sha256": _ordered_units_sha256(provenance.dev.unit_ids),
        }
        for name, expected in expected_fields.items():
            if getattr(self, name) != expected:
                raise ValueError(f"transport artifact has stale {name}")
        if not _valid_sha256(self.model_sha256):
            raise ValueError("transport artifact requires a model SHA-256")
        expected_artifact = _digest({
            "source": self.source, "model_sha256": self.model_sha256,
            "provenance_sha256": self.provenance_sha256,
            **expected_fields,
        })
        if self.artifact_sha256 != expected_artifact:
            raise ValueError("transport model artifact commitment mismatch")


def _seal_transport_model_artifact(system, source: str,
                                   provenance: TransportProvenance
                                   ) -> TransportModelArtifact:
    if source == "cross_fitted_rrr":
        if not isinstance(system, DeltaTransportSystem):
            raise TypeError("RRR artifact requires DeltaTransportSystem")
        expected_model_sha = _digest({
            "coef": system.model.coef.tolist(),
            "intercept": system.model.intercept.tolist(),
            "rank": system.model.rank, "ridge": system.model.ridge,
            "provenance": provenance.provenance_sha256,
        })
        if system.model_sha256 != expected_model_sha:
            raise ValueError("RRR parameters changed after fitting")
        model_sha = expected_model_sha
    elif source == "cross_fitted_sparse_delta_transcoder":
        if not isinstance(system, FittedSparseDeltaTranscoder):
            raise TypeError("sparse artifact requires a fitted SparseDeltaTranscoder")
        model_sha = _state_dict_sha256(system.model)
        if model_sha != system.artifact_sha256:
            raise ValueError("sparse delta transcoder changed after fitting")
    elif source == "cross_fitted_hcl_kernel":
        if not isinstance(system, CausalImpulseKernel):
            raise TypeError("HCL artifact requires CausalImpulseKernel")
        model_sha = _digest({
            "coef": system.model.coef.tolist(),
            "intercept": system.model.intercept.tolist(),
            "rank": system.model.rank, "ridge": system.model.ridge,
            "max_lag": system.max_lag, "d_in": system.d_in,
            "d_out": system.d_out,
            "provenance": system.provenance_sha256,
        })
        if model_sha != system.model_sha256:
            raise ValueError("HCL kernel changed after fitting")
    else:
        raise ValueError("use a fitted cross-validated model")
    fields = {
        "source": source, "model_sha256": model_sha,
        "provenance_sha256": provenance.provenance_sha256,
        "train_source_sha256": provenance.train.source_sha256,
        "dev_source_sha256": provenance.dev.source_sha256,
        "train_unit_order_sha256": _ordered_units_sha256(provenance.train.unit_ids),
        "dev_unit_order_sha256": _ordered_units_sha256(provenance.dev.unit_ids),
    }
    artifact = TransportModelArtifact(**fields, artifact_sha256=_digest(fields))
    artifact.validate(provenance)
    return artifact


@dataclass(frozen=True)
class PredictedDelta:
    values: np.ndarray
    unit_ids: tuple[str, ...]
    source: str
    model_sha256: str
    provenance_sha256: str
    model_artifact: Optional[TransportModelArtifact] = None
    prediction_input_sha256: str = ""
    prediction_sha256: str = ""

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        values.setflags(write=False)
        unit_ids = tuple(map(str, self.unit_ids))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "unit_ids", unit_ids)
        computed = self.compute_sha256()
        if self.prediction_sha256 and self.prediction_sha256 != computed:
            raise ValueError("predicted-delta commitment hash mismatch")
        object.__setattr__(self, "prediction_sha256", computed)

    def compute_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(self.values, dtype=np.float64, order="C").tobytes())
        h.update(_digest({
            "shape": list(self.values.shape),
            "unit_ids": list(self.unit_ids),
            "source": self.source,
            "model_sha256": self.model_sha256,
            "provenance_sha256": self.provenance_sha256,
            "model_artifact_sha256": (
                self.model_artifact.artifact_sha256 if self.model_artifact else ""
            ),
            "prediction_input_sha256": self.prediction_input_sha256,
        }).encode())
        return h.hexdigest()

    def validate(self, provenance: TransportProvenance) -> None:
        if self.source not in {"cross_fitted_rrr", "cross_fitted_sparse_delta_transcoder",
                               "cross_fitted_hcl_kernel"}:
            raise ValueError(
                "primary rescue requires a cross-fitted prediction; cached/actual deltas are forbidden"
            )
        if self.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("predicted delta/provenance mismatch")
        if self.model_artifact is None:
            raise ValueError("predicted delta is not bound to a fitted-model artifact")
        self.model_artifact.validate(provenance)
        if self.source != self.model_artifact.source \
                or self.model_sha256 != self.model_artifact.model_sha256:
            raise ValueError("predicted delta model labels do not match its sealed artifact")
        provenance.assert_units("locked", self.unit_ids)
        values = _matrix(self.values, "predicted delta")
        if values.shape[0] != len(self.unit_ids) or not np.isfinite(values).all():
            raise ValueError("invalid predicted delta values")
        if len(self.model_sha256) != 64 or any(c not in "0123456789abcdef"
                                               for c in self.model_sha256.lower()):
            raise ValueError("predicted delta needs a sealed model SHA-256")
        if not _valid_sha256(self.prediction_input_sha256):
            raise ValueError("predicted delta requires a sealed locked-input hash")
        if self.compute_sha256() != self.prediction_sha256:
            raise ValueError("predicted-delta values or ordered unit binding changed after sealing")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


@dataclass(frozen=True)
class RescueControlEndpoint:
    """Sealed specificity-control endpoint with an ordered unit binding."""

    family: ControlFamily
    values: np.ndarray
    unit_ids: tuple[str, ...]
    source_sha256: str
    dose_matched: bool
    rank_matched: bool
    injected_delta: Optional[np.ndarray] = None
    prediction_sha256: str = ""
    source_layer: str = ""
    target_layer: str = ""
    permutation_sha256: str = ""
    permutation: tuple[int, ...] = ()
    endpoint_sha256: str = ""

    def __post_init__(self) -> None:
        if self.family not in {
            "dose_rank_matched_random", "shuffled_prediction",
            "wrong_layer", "wrong_pair",
        }:
            raise ValueError(f"unknown predicted-delta control family {self.family!r}")
        values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("control endpoint must be a finite [n,d] matrix")
        values.setflags(write=False)
        if self.injected_delta is None:
            raise ValueError("control endpoint requires the actual injected delta")
        injected = np.array(
            self.injected_delta, dtype=np.float64, copy=True, order="C")
        if injected.ndim != 2 or not np.isfinite(injected).all():
            raise ValueError("control injected delta must be a finite [n,d] matrix")
        injected.setflags(write=False)
        unit_ids = tuple(map(str, self.unit_ids))
        if values.shape[0] != len(unit_ids) or injected.shape[0] != len(unit_ids) \
                or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("control endpoint rows require unique ordered unit ids")
        if not _valid_sha256(self.source_sha256):
            raise ValueError("control endpoint requires a source SHA-256")
        if self.family in {"dose_rank_matched_random", "wrong_layer", "wrong_pair"} \
                and not (self.dose_matched and self.rank_matched):
            raise ValueError(f"{self.family} must be dose- and rank-matched")
        if self.family == "shuffled_prediction":
            if not _valid_sha256(self.prediction_sha256) or not self.permutation:
                raise ValueError(
                    "shuffled control must bind the sealed prediction and permutation"
                )
        if self.family == "wrong_layer" and (
                not self.source_layer or not self.target_layer
                or self.source_layer == self.target_layer):
            raise ValueError("wrong-layer control must identify two distinct layers")
        if self.family == "wrong_pair" and not self.permutation:
            raise ValueError("wrong-pair control requires a sealed pairing permutation")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "injected_delta", injected)
        object.__setattr__(self, "unit_ids", unit_ids)
        permutation = tuple(int(index) for index in self.permutation)
        if permutation:
            if sorted(permutation) != list(range(len(unit_ids))):
                raise ValueError("control permutation must contain every row exactly once")
            computed_permutation_sha = _digest({"permutation": list(permutation)})
            if self.permutation_sha256 \
                    and self.permutation_sha256 != computed_permutation_sha:
                raise ValueError("control permutation hash mismatch")
            object.__setattr__(self, "permutation_sha256", computed_permutation_sha)
        object.__setattr__(self, "permutation", permutation)
        computed = self.compute_sha256()
        if self.endpoint_sha256 and self.endpoint_sha256 != computed:
            raise ValueError("control endpoint commitment hash mismatch")
        object.__setattr__(self, "endpoint_sha256", computed)

    def compute_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(self.values, dtype=np.float64, order="C").tobytes())
        h.update(np.asarray(self.injected_delta, dtype=np.float64, order="C").tobytes())
        h.update(_digest({
            "shape": list(self.values.shape),
            "injected_shape": list(self.injected_delta.shape),
            "injected_delta_sha256": _array_sha256(self.injected_delta),
            "family": self.family,
            "unit_ids": list(self.unit_ids), "source_sha256": self.source_sha256,
            "dose_matched": self.dose_matched, "rank_matched": self.rank_matched,
            "prediction_sha256": self.prediction_sha256,
            "source_layer": self.source_layer, "target_layer": self.target_layer,
            "permutation_sha256": self.permutation_sha256,
            "permutation": list(self.permutation),
        }).encode())
        return h.hexdigest()

    def validate(self, predicted: PredictedDelta,
                 provenance: TransportProvenance) -> None:
        provenance.assert_units("locked", self.unit_ids)
        if self.unit_ids != predicted.unit_ids:
            raise ValueError("control endpoint order differs from sealed prediction")
        if self.values.shape[0] != predicted.values.shape[0]:
            raise ValueError("control endpoint row count differs from prediction")
        if self.injected_delta.shape != predicted.values.shape:
            raise ValueError("control injected delta shape differs from prediction")
        predicted_dose = float(np.linalg.norm(predicted.values))
        control_dose = float(np.linalg.norm(self.injected_delta))
        predicted_unit_dose = np.linalg.norm(predicted.values, axis=1)
        control_unit_dose = np.linalg.norm(self.injected_delta, axis=1)
        if self.dose_matched and (
                not np.isclose(control_dose, predicted_dose, rtol=1e-6, atol=1e-10)
                or not np.allclose(control_unit_dose, predicted_unit_dose,
                                   rtol=1e-6, atol=1e-10)):
            raise ValueError(
                "control is labelled dose-matched but per-unit numerical dose differs"
            )
        if self.rank_matched and np.linalg.matrix_rank(self.injected_delta) \
                != np.linalg.matrix_rank(predicted.values):
            raise ValueError("control is labelled rank-matched but numerical rank differs")
        if self.family in {"shuffled_prediction", "wrong_pair"}:
            expected = predicted.values[np.asarray(self.permutation, dtype=np.int64)]
            expected_norm = np.linalg.norm(expected, axis=1, keepdims=True).clip(1e-30)
            injected_norm = np.linalg.norm(
                self.injected_delta, axis=1, keepdims=True).clip(1e-30)
            if not np.allclose(
                    self.injected_delta / injected_norm,
                    expected / expected_norm, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    f"{self.family} injected direction does not match its sealed permutation"
                )
        if self.family == "dose_rank_matched_random" \
                and np.array_equal(self.injected_delta, predicted.values):
            raise ValueError("random control cannot reuse the predicted delta")
        if self.compute_sha256() != self.endpoint_sha256:
            raise ValueError("control endpoint changed after sealing")
        if self.family == "shuffled_prediction" \
                and self.prediction_sha256 != predicted.prediction_sha256:
            raise ValueError("shuffled control came from a different prediction")


@dataclass(frozen=True)
class RescueRuntimeContract:
    checkpoint_sha256: str
    architecture_sha256: str
    code_sha256: str
    handoff_sha256: str
    config_sha256: str
    tap_spec_sha256: str
    edit_spec_sha256: str
    intervention_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def validate(self) -> None:
        for name, value in self.as_dict().items():
            if not _valid_sha256(value):
                raise ValueError(f"runtime contract requires full {name}")

    @property
    def contract_sha256(self) -> str:
        self.validate()
        return _digest(self.as_dict())


@dataclass(frozen=True)
class RescueRecord:
    """Immutable receipt for the real suffix execution used as rescue evidence."""

    unit_ids: tuple[str, ...]
    dependency_keys: tuple[tuple[str, ...], ...]
    prediction_sha256: str
    model_artifact_sha256: str
    prediction_input_sha256: str
    injected_delta_sha256: str
    true_delta_sha256: str
    native_endpoint_sha256: str
    ablated_endpoint_sha256: str
    predicted_endpoint_sha256: str
    control_endpoint_sha256: tuple[tuple[str, str], ...]
    checkpoint_sha256: str
    architecture_sha256: str
    code_sha256: str
    handoff_sha256: str
    config_sha256: str
    tap_spec_sha256: str
    edit_spec_sha256: str
    intervention_sha256: str
    generated_by_suffix_runner: bool
    record_sha256: str

    def _payload(self) -> dict:
        return {
            "unit_ids": list(self.unit_ids),
            "dependency_keys": [list(keys) for keys in self.dependency_keys],
            "prediction_sha256": self.prediction_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "prediction_input_sha256": self.prediction_input_sha256,
            "injected_delta_sha256": self.injected_delta_sha256,
            "true_delta_sha256": self.true_delta_sha256,
            "native_endpoint_sha256": self.native_endpoint_sha256,
            "ablated_endpoint_sha256": self.ablated_endpoint_sha256,
            "predicted_endpoint_sha256": self.predicted_endpoint_sha256,
            "control_endpoint_sha256": [list(item) for item in self.control_endpoint_sha256],
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture_sha256": self.architecture_sha256,
            "code_sha256": self.code_sha256,
            "handoff_sha256": self.handoff_sha256,
            "config_sha256": self.config_sha256,
            "tap_spec_sha256": self.tap_spec_sha256,
            "edit_spec_sha256": self.edit_spec_sha256,
            "intervention_sha256": self.intervention_sha256,
            "generated_by_suffix_runner": self.generated_by_suffix_runner,
        }

    def validate(
        self, predicted: PredictedDelta, provenance: TransportProvenance, *,
        true_delta: np.ndarray, native_endpoint: np.ndarray,
        ablated_endpoint: np.ndarray, predicted_rescue_endpoint: np.ndarray,
        controls: Mapping[str, RescueControlEndpoint],
        expected_runtime_contract: Optional[RescueRuntimeContract] = None,
    ) -> None:
        predicted.validate(provenance)
        provenance.assert_units("locked", self.unit_ids)
        if self.unit_ids != predicted.unit_ids:
            raise ValueError("rescue receipt unit order differs from prediction")
        if len(self.dependency_keys) != len(self.unit_ids) \
                or any(not keys for keys in self.dependency_keys):
            raise ValueError("every rescue unit requires dependency-cluster keys")
        if not self.generated_by_suffix_runner:
            raise ValueError("primary rescue must be emitted by the suffix runner")
        if predicted.model_artifact is None:
            raise ValueError("prediction lacks fitted-model artifact")
        expected = {
            "prediction_sha256": predicted.prediction_sha256,
            "model_artifact_sha256": predicted.model_artifact.artifact_sha256,
            "prediction_input_sha256": predicted.prediction_input_sha256,
            "injected_delta_sha256": _array_sha256(predicted.values),
            "true_delta_sha256": _array_sha256(true_delta),
            "native_endpoint_sha256": _array_sha256(native_endpoint),
            "ablated_endpoint_sha256": _array_sha256(ablated_endpoint),
            "predicted_endpoint_sha256": _array_sha256(predicted_rescue_endpoint),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"rescue receipt {name} does not match supplied evidence")
        control_hashes = tuple(sorted(
            (name, control.endpoint_sha256) for name, control in controls.items()
        ))
        if self.control_endpoint_sha256 != control_hashes:
            raise ValueError("rescue control endpoints differ from the sealed receipt")
        for name in (
                "checkpoint_sha256", "architecture_sha256", "code_sha256",
                "handoff_sha256", "config_sha256", "tap_spec_sha256",
                "edit_spec_sha256", "intervention_sha256"):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"rescue receipt requires full {name}")
        if expected_runtime_contract is not None:
            expected_runtime_contract.validate()
            for name, expected_value in expected_runtime_contract.as_dict().items():
                if getattr(self, name) != expected_value:
                    raise ValueError(
                        f"rescue receipt runtime mismatch for {name}"
                    )
        if self.record_sha256 != _digest(self._payload()):
            raise ValueError("rescue receipt commitment hash mismatch")


def seal_rescue_record(
    predicted: PredictedDelta,
    provenance: TransportProvenance,
    *,
    dependency_keys: Sequence[Sequence[str]],
    true_delta: np.ndarray,
    native_endpoint: np.ndarray,
    ablated_endpoint: np.ndarray,
    predicted_rescue_endpoint: np.ndarray,
    control_rescue_endpoints: Mapping[str, RescueControlEndpoint],
    runtime_contract: RescueRuntimeContract,
    generated_by_suffix_runner: bool = True,
) -> RescueRecord:
    """Create the receipt at suffix-execution time, before statistical scoring."""
    predicted.validate(provenance)
    runtime_contract.validate()
    keys = tuple(tuple(map(str, item)) for item in dependency_keys)
    for control in control_rescue_endpoints.values():
        control.validate(predicted, provenance)
    if predicted.model_artifact is None:  # narrowed by validate; keeps type checkers honest
        raise ValueError("prediction lacks fitted-model artifact")
    fields = {
        "unit_ids": predicted.unit_ids,
        "dependency_keys": keys,
        "prediction_sha256": predicted.prediction_sha256,
        "model_artifact_sha256": predicted.model_artifact.artifact_sha256,
        "prediction_input_sha256": predicted.prediction_input_sha256,
        "injected_delta_sha256": _array_sha256(predicted.values),
        "true_delta_sha256": _array_sha256(true_delta),
        "native_endpoint_sha256": _array_sha256(native_endpoint),
        "ablated_endpoint_sha256": _array_sha256(ablated_endpoint),
        "predicted_endpoint_sha256": _array_sha256(predicted_rescue_endpoint),
        "control_endpoint_sha256": tuple(sorted(
            (name, control.endpoint_sha256)
            for name, control in control_rescue_endpoints.items()
        )),
        **runtime_contract.as_dict(),
        "generated_by_suffix_runner": generated_by_suffix_runner,
    }
    provisional = RescueRecord(**fields, record_sha256="")
    record = RescueRecord(**fields, record_sha256=_digest(provisional._payload()))
    record.validate(
        predicted, provenance, true_delta=true_delta,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_rescue_endpoint,
        controls=control_rescue_endpoints,
    )
    return record


def predict_locked_delta(system, locked_input: np.ndarray, unit_ids: Sequence[str],
                         provenance: TransportProvenance, *, source: str,
                         base: Optional[np.ndarray] = None) -> PredictedDelta:
    provenance.assert_units("locked", unit_ids)
    if source == "cross_fitted_hcl_kernel":
        actual_locked_source = lag_source_sha256(unit_ids, locked_input)
    else:
        if base is None:
            raise ValueError(
                "locked intervention source verification requires its native base states"
            )
        actual_locked_source = intervention_source_sha256(
            unit_ids, base, locked_input)
    if actual_locked_source != provenance.locked.source_sha256:
        raise ValueError(
            "locked model input/base tensors do not match the provenance source hash"
        )
    artifact = _seal_transport_model_artifact(system, source, provenance)
    if source == "cross_fitted_rrr":
        if not isinstance(system, DeltaTransportSystem) or base is None:
            raise TypeError("RRR prediction requires DeltaTransportSystem and locked base")
        if system.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("RRR model was fitted under different provenance")
        values = system.predict_delta(base, locked_input)
        model_hash = artifact.model_sha256
    elif source == "cross_fitted_sparse_delta_transcoder":
        if not isinstance(system, FittedSparseDeltaTranscoder):
            raise TypeError("sparse source requires a fitted SparseDeltaTranscoder")
        if system.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("sparse delta model was fitted under different provenance")
        values = system.predict_delta(locked_input)
        model_hash = artifact.model_sha256
    elif source == "cross_fitted_hcl_kernel":
        if not isinstance(system, CausalImpulseKernel):
            raise TypeError("HCL source requires CausalImpulseKernel")
        if system.provenance_sha256 != provenance.provenance_sha256:
            raise ValueError("HCL kernel was fitted under different provenance")
        values = system.predict(locked_input)[:, -1, :]
        model_hash = artifact.model_sha256
    else:
        raise ValueError("use a fitted cross-validated model; direct values are not accepted")
    input_commitment = _digest({
        "locked_input_sha256": _array_sha256(np.asarray(locked_input)),
        "base_sha256": _array_sha256(np.asarray(base)) if base is not None else "",
        "ordered_unit_ids_sha256": _ordered_units_sha256(unit_ids),
        "model_artifact_sha256": artifact.artifact_sha256,
    })
    pred = PredictedDelta(
        np.asarray(values), tuple(map(str, unit_ids)), source,
        model_hash, provenance.provenance_sha256,
        model_artifact=artifact, prediction_input_sha256=input_commitment,
    )
    pred.validate(provenance)
    return pred


@dataclass(frozen=True)
class RescueMargins:
    prediction_relative_energy: float
    recovery_fraction: float
    control_superiority: float
    max_relative_error: float
    prediction_cosine: float = 0.5


def _cluster_mean_ci(values: np.ndarray,
                     dependency_keys: Sequence[Sequence[str]], *,
                     unit_ids: Sequence[str], seed: int, n_boot: int,
                     min_clusters: int, alpha: float = 0.05):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("bootstrap needs at least two finite unit-level values")
    if len(dependency_keys) != len(x) or len(unit_ids) != len(x):
        raise ValueError("cluster keys/unit ids must align with rescue values")
    return cluster_bootstrap(
        [Row(float(value), keys=tuple(map(str, keys)), unit_id=str(unit_id))
         for value, keys, unit_id in zip(x, dependency_keys, unit_ids)],
        seed=seed, n_boot=n_boot, min_clusters=min_clusters, alpha=alpha,
    )


def predicted_delta_rescue_verdict(
    predicted: PredictedDelta,
    provenance: TransportProvenance,
    *,
    fitted_system,
    locked_model_input: np.ndarray,
    locked_model_base: Optional[np.ndarray] = None,
    locked_output_base: Optional[np.ndarray] = None,
    locked_full_target: Optional[np.ndarray] = None,
    true_delta: np.ndarray,
    native_endpoint: np.ndarray,
    ablated_endpoint: np.ndarray,
    predicted_rescue_endpoint: np.ndarray,
    rescue_record: RescueRecord,
    control_rescue_endpoints: Mapping[str, RescueControlEndpoint],
    expected_runtime_contract: RescueRuntimeContract,
    margins: RescueMargins,
    paired_design: bool = True,
    required_control_families: Optional[Sequence[str]] = None,
    seed: int = 42,
    n_boot: int = 10_000,
    min_clusters: int = 2,
    alpha: float = 0.05,
) -> dict:
    """Primary Step-15 evidence: held-out prediction plus real suffix rescue.

    ``true_delta`` is used only to score the pre-committed prediction after it
    has been generated.  It is never patched.  The patched endpoint must come
    from running the model suffix with ``predicted.values``.  Every control
    family is tested separately.
    """
    predicted.validate(provenance)
    margin_values = (
        margins.prediction_relative_energy, margins.recovery_fraction,
        margins.control_superiority, margins.max_relative_error,
        margins.prediction_cosine,
    )
    if not np.isfinite(margin_values).all() or margins.max_relative_error < 0 \
            or not -1 <= margins.prediction_cosine <= 1:
        raise ValueError("invalid pre-registered predicted-rescue margins")
    regenerated = predict_locked_delta(
        fitted_system, locked_model_input, predicted.unit_ids, provenance,
        source=predicted.source, base=locked_model_base,
    )
    if regenerated.prediction_sha256 != predicted.prediction_sha256 \
            or not np.array_equal(regenerated.values, predicted.values):
        raise ValueError(
            "supplied prediction was not generated by the sealed model and locked input"
        )
    if min_clusters < 2 or n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("invalid cluster-inference configuration")
    rescue_record.validate(
        predicted, provenance, true_delta=true_delta,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_rescue_endpoint,
        controls=control_rescue_endpoints,
        expected_runtime_contract=expected_runtime_contract,
    )
    y, p = _matrix(true_delta, "true_delta"), _matrix(predicted.values, "prediction")
    if y.shape != p.shape:
        raise ValueError("true/predicted delta shape mismatch")
    if predicted.source == "cross_fitted_hcl_kernel":
        if locked_full_target is None:
            raise ValueError("HCL rescue scoring requires the sealed full lag target")
        full_target = np.asarray(locked_full_target, dtype=np.float64)
        if full_target.ndim != 3 or not np.array_equal(y, full_target[:, -1, :]):
            raise ValueError("HCL rescue target is not the final slice of its sealed target")
        actual_target_sha = _array_sha256(full_target)
    else:
        actual_target_sha = intervention_target_sha256(y, locked_output_base)
    if actual_target_sha != provenance.locked.target_sha256:
        raise ValueError("locked true-delta target differs from its provenance commitment")
    native = _matrix(native_endpoint, "native_endpoint")
    ablated = _matrix(ablated_endpoint, "ablated_endpoint")
    rescued = _matrix(predicted_rescue_endpoint, "predicted_rescue_endpoint")
    if not (native.shape == ablated.shape == rescued.shape) or native.shape[0] != len(y):
        raise ValueError("endpoint arrays must align with locked units")
    mandatory = set(REQUIRED_PREDICTED_DELTA_CONTROLS)
    if paired_design:
        mandatory.add("wrong_pair")
    if required_control_families is not None:
        registered = set(map(str, required_control_families))
        if not mandatory.issubset(registered):
            missing = sorted(mandatory - registered)
            raise ValueError(
                "registered control plan cannot weaken mandatory specificity families: "
                f"{missing}"
            )
        mandatory = registered
    unknown = sorted(set(control_rescue_endpoints) - {
        "dose_rank_matched_random", "shuffled_prediction", "wrong_layer", "wrong_pair",
    })
    if unknown:
        raise ValueError(f"unknown predicted-delta rescue controls: {unknown}")
    missing_controls = sorted(mandatory - set(control_rescue_endpoints))
    if missing_controls:
        raise ValueError(f"predicted-delta rescue control families missing: {missing_controls}")

    damage = np.linalg.norm(ablated - native, axis=1)
    remaining = np.linalg.norm(rescued - native, axis=1)
    eligible = damage > 1e-12
    if not np.all(eligible):
        raise ValueError("zero-damage units cannot identify rescue")
    recovery = 1.0 - remaining / damage
    control_recovery = {}
    control_hashes = {}
    for name in sorted(control_rescue_endpoints):
        endpoint = control_rescue_endpoints[name]
        if not isinstance(endpoint, RescueControlEndpoint):
            raise TypeError(
                "control_rescue_endpoints must contain sealed RescueControlEndpoint objects"
            )
        if endpoint.family != name:
            raise ValueError(f"control key/family mismatch for {name!r}")
        endpoint.validate(predicted, provenance)
        c = _matrix(endpoint.values, f"control {name}")
        if c.shape != native.shape:
            raise ValueError(f"control {name} endpoint shape mismatch")
        control_recovery[name] = 1.0 - np.linalg.norm(c - native, axis=1) / damage
        control_hashes[name] = endpoint.endpoint_sha256

    pred_r2 = delta_r2(y, p)
    true_norm = np.linalg.norm(y, axis=1).clip(1e-30)
    pred_norm = np.linalg.norm(p, axis=1).clip(1e-30)
    rel = np.linalg.norm(p - y, axis=1) / true_norm
    cosine = np.sum(p * y, axis=1) / (pred_norm * true_norm)
    relative_energy_explained = 1.0 - (
        np.linalg.norm(p - y, axis=1) ** 2 / (true_norm ** 2)
    )
    prediction_energy_ci = _cluster_mean_ci(
        relative_energy_explained, rescue_record.dependency_keys,
        unit_ids=rescue_record.unit_ids, seed=seed + 10_001,
        n_boot=n_boot, min_clusters=min_clusters, alpha=alpha,
    )
    prediction_cosine_ci = _cluster_mean_ci(
        cosine, rescue_record.dependency_keys,
        unit_ids=rescue_record.unit_ids, seed=seed + 10_002,
        n_boot=n_boot, min_clusters=min_clusters, alpha=alpha,
    )
    prediction_relative_error_ci = _cluster_mean_ci(
        rel, rescue_record.dependency_keys,
        unit_ids=rescue_record.unit_ids, seed=seed + 10_003,
        n_boot=n_boot, min_clusters=min_clusters, alpha=alpha,
    )
    recovery_ci = _cluster_mean_ci(
        recovery, rescue_record.dependency_keys,
        unit_ids=rescue_record.unit_ids, seed=seed, n_boot=n_boot,
        min_clusters=min_clusters, alpha=alpha,
    )
    specificity = {}
    for i, (name, ctrl) in enumerate(sorted(control_recovery.items())):
        specificity[name] = _cluster_mean_ci(
            recovery - ctrl, rescue_record.dependency_keys,
            unit_ids=rescue_record.unit_ids, seed=seed + 1 + i,
            n_boot=n_boot, min_clusters=min_clusters, alpha=alpha,
        )
    checks = {
        "heldout_prediction_relative_energy":
            prediction_energy_ci.lo > margins.prediction_relative_energy,
        "heldout_prediction_cosine":
            prediction_cosine_ci.lo > margins.prediction_cosine,
        "heldout_prediction_relative_error":
            prediction_relative_error_ci.hi < margins.max_relative_error,
        "predicted_delta_rescues": recovery_ci.lo > margins.recovery_fraction,
        "beats_every_control": all(ci.lo > margins.control_superiority
                                   for ci in specificity.values()),
    }
    return {
        "prediction": {"delta_r2": pred_r2,
                       "relative_error_mean": float(np.mean(rel)),
                       "relative_energy_cluster_estimate":
                           prediction_energy_ci.as_row(),
                       "cosine_cluster_estimate": prediction_cosine_ci.as_row(),
                       "relative_error_cluster_estimate":
                           prediction_relative_error_ci.as_row(),
                       "source": predicted.source,
                       "model_sha256": predicted.model_sha256,
                       "prediction_sha256": predicted.prediction_sha256,
                       "ordered_unit_ids": list(predicted.unit_ids)},
        "rescue": {"recovery_cluster_estimate": recovery_ci.as_row(),
                   "specificity_cluster_estimates": {
                       name: estimate.as_row() for name, estimate in specificity.items()
                   },
                   "control_endpoint_sha256": control_hashes,
                   "required_control_families": sorted(mandatory),
                   "rescue_record_sha256": rescue_record.record_sha256},
        "checks": checks,
        "predicted_delta_rescue": all(checks.values()),
        "claim_guard": (
            "A pass supports learned delta transport. It does not license absolute-state "
            "reconstruction, and an actual/cached delta patch is not accepted as evidence."
        ),
    }

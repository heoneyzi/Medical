"""EXP3 Phase 6 — applications that reuse the characterized late stack.

Nothing in this file discovers a layer, direction, channel or motif from task
labels.  It consumes the objects frozen by Steps 7--15 and asks whether they
explain variants/tasks, enable stage-specific repair, calibrate confidence,
guide compression, or design sequences.  This separation is essential: a
useful predictor is not automatically a mechanism, and a causal mechanism is
not automatically a useful predictor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .exp2_api import Row, cluster_bootstrap


STAGES = ("W", "A", "R", "K")  # writer, amplifier/co-writer, re-encoder, calibration
FINGERPRINT_AXES = (
    "b28_content_write_magnitude",
    "b28_direction_stability",
    "b29_parallel_perp_ratio",
    "b29_gain",
    "scale_q",
    "scale_plateau_position",
    "b30_mediation_fraction",
    "b30_predicted_rescue_fraction",
    "x31_carrier_sensitivity",
    "offpath_leakage",
)


def _is_sha256(value: str) -> bool:
    return (len(value) == 64 and
            all(char in "0123456789abcdef" for char in value.lower()))


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_variant_cohort_ids(
    variant_to_cohort: Mapping[str, str],
) -> tuple[str, str]:
    """Seal both the variant set and its fixed cohort assignment."""
    if not variant_to_cohort:
        raise ValueError("cannot seal an empty variant/cohort table")
    pairs = []
    for variant_id, cohort_id in variant_to_cohort.items():
        variant_id, cohort_id = str(variant_id), str(cohort_id)
        if not variant_id or not cohort_id:
            raise ValueError("variant and cohort IDs must be non-empty")
        pairs.append((variant_id, cohort_id))
    if len({variant_id for variant_id, _ in pairs}) != len(pairs):
        raise ValueError("variant IDs are not unique after normalization")
    pairs.sort()
    return (_canonical_sha256([variant_id for variant_id, _ in pairs]),
            _canonical_sha256(pairs))


def _vec(x: Tensor) -> Tensor:
    x = x.detach().double().reshape(-1)
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("activation vector is empty or non-finite")
    return x


def _angle(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a, b = _vec(a), _vec(b)
    if a.shape != b.shape:
        raise ValueError("vectors have different widths")
    if a.device != b.device:
        raise ValueError("vectors are on different devices")
    if a.norm() <= eps or b.norm() <= eps:
        raise ValueError("angle is undefined for a zero vector")
    c = torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps)
    return float(torch.acos(c.clamp(-1.0, 1.0)))


@dataclass(frozen=True)
class VariantGeometry:
    variant_id: str
    delta_log_norm_g28: float
    angular_g28: float
    angular_m30_carrier_removed: float
    delta_carrier: float
    transport_ratio: float


def variant_geometry(variant_id: str, *, g28_ref: Tensor, g28_alt: Tensor,
                     m30_ref: Tensor, m30_alt: Tensor, carrier_axis: Tensor,
                     eps: float = 1e-12) -> VariantGeometry:
    """Compute the label-free mechanistic vector used before labels are opened."""
    gr, ga, mr, ma, c = map(_vec, (g28_ref, g28_alt, m30_ref, m30_alt, carrier_axis))
    if not (gr.shape == ga.shape == mr.shape == ma.shape == c.shape):
        raise ValueError("all variant activations and carrier_axis need the same width")
    if len({x.device for x in (gr, ga, mr, ma, c)}) != 1:
        raise ValueError("all variant activations and carrier_axis need the same device")
    if c.norm() <= eps:
        raise ValueError("carrier_axis must be non-zero")
    c = c / c.norm().clamp_min(eps)
    qmr, qma = mr - torch.dot(mr, c) * c, ma - torch.dot(ma, c) * c
    a28 = _angle(gr, ga, eps)
    a30 = _angle(qmr, qma, eps)
    return VariantGeometry(
        variant_id=variant_id,
        delta_log_norm_g28=float(torch.log(ga.norm().clamp_min(eps) /
                                           gr.norm().clamp_min(eps))),
        angular_g28=a28,
        angular_m30_carrier_removed=a30,
        delta_carrier=float(torch.dot(ma - mr, c)),
        transport_ratio=float(a30 / max(a28, eps)),
    )


@dataclass(frozen=True)
class VariantCausalRow:
    """Effects expressed as a fraction of the native ref→alt output effect."""
    variant_id: str
    cohort_id: str
    direction_transfer: float
    reverse_direction_rescue: float
    norm_only: float
    b30_block_remaining: float
    predicted_delta_m30_rescue: float
    random_direction: float
    wrong_layer: float
    off_target_kl: float
    prediction_method: str
    prediction_sha256: str
    prediction_source_split: str
    prediction_held_out: bool
    prediction_target_derived: bool
    dependency_keys: tuple[str, ...]


def _boot(rows: Sequence[VariantCausalRow], attr: str, *, n_boot: int,
          seed: int, min_clusters: int) -> object:
    return cluster_bootstrap(
        [Row(float(getattr(r, attr)), keys=r.dependency_keys or
             (f"variant:{r.variant_id}",)) for r in rows],
        n_boot=n_boot, seed=seed, min_clusters=min_clusters,
    )


def variant_causal_verdict(rows: Sequence[VariantCausalRow], *,
                           rescue_min: float, null_max: float,
                           off_target_max: float, min_clusters: int,
                           expected_prediction_sha256: str,
                           expected_prediction_source_split: str,
                           expected_variant_set_sha256: str,
                           expected_variant_cohort_sha256: str,
                           n_boot: int = 5000, seed: int = 42) -> dict:
    """Conjunction for causal variant use with a sealed held-out predictor.

    The prediction commitment identifies the model/artifact that generated the
    m30 rescue delta.  It is deliberately checked independently of whether the
    rescue happened to succeed.  Thus a target-derived rescue cannot be
    relabelled as a prospective application after seeing the target response.
    """
    if not rows:
        raise ValueError("variant causal table is empty")
    if not _is_sha256(expected_prediction_sha256):
        raise ValueError("expected_prediction_sha256 is not a SHA-256 digest")
    if not expected_prediction_source_split:
        raise ValueError("expected_prediction_source_split is empty")
    if (min(rescue_min, null_max, off_target_max) < 0 or min_clusters < 2 or
            n_boot < 100):
        raise ValueError("invalid variant causal margins/bootstrap")
    if (not _is_sha256(expected_variant_set_sha256) or
            not _is_sha256(expected_variant_cohort_sha256)):
        raise ValueError("variant/cohort commitment is not a SHA-256 digest")
    if len({r.variant_id for r in rows}) != len(rows):
        raise ValueError("variant_id must be unique")
    observed_variant_sha, observed_cohort_sha = seal_variant_cohort_ids(
        {row.variant_id: row.cohort_id for row in rows})
    if observed_variant_sha != expected_variant_set_sha256:
        raise RuntimeError("variant set differs from the sealed locked set")
    if observed_cohort_sha != expected_variant_cohort_sha256:
        raise RuntimeError("variant/cohort assignment differs from the sealed locked set")
    methods = {r.prediction_method for r in rows}
    for row in rows:
        numeric = (
            row.direction_transfer, row.reverse_direction_rescue, row.norm_only,
            row.b30_block_remaining, row.predicted_delta_m30_rescue,
            row.random_direction, row.wrong_layer, row.off_target_kl,
        )
        if (not row.variant_id or not row.cohort_id or not row.dependency_keys or
                not np.isfinite(numeric).all()):
            raise ValueError(f"invalid variant causal row {row.variant_id!r}")
        if row.off_target_kl < 0:
            raise ValueError("off_target_kl must be non-negative")
        if not row.prediction_method:
            raise ValueError("prediction_method must be non-empty")
        if row.prediction_sha256 != expected_prediction_sha256:
            raise RuntimeError("variant rescue uses a different prediction commitment")
        if row.prediction_source_split != expected_prediction_source_split:
            raise RuntimeError("variant rescue uses a different prediction source split")
        if not row.prediction_held_out or row.prediction_target_derived:
            raise RuntimeError(
                "strong variant rescue requires held-out, target-free predictions")
    if len(methods) != 1:
        raise RuntimeError("one frozen prediction method is required per locked table")
    estimates = {a: _boot(rows, a, n_boot=n_boot, seed=seed + i,
                          min_clusters=min_clusters)
                 for i, a in enumerate((
                     "direction_transfer", "reverse_direction_rescue", "norm_only",
                     "b30_block_remaining", "predicted_delta_m30_rescue",
                     "random_direction", "wrong_layer", "off_target_kl"))}
    if min(e.n_clusters for e in estimates.values()) < min_clusters:
        raise RuntimeError("variant analysis has too few dependency clusters")
    checks = {
        "prediction_commitment_matches": True,
        "prediction_is_held_out": True,
        "prediction_is_target_free": True,
        "direction_transfers": estimates["direction_transfer"].lo >= rescue_min,
        "reverse_direction_rescues": estimates["reverse_direction_rescue"].lo >= rescue_min,
        "norm_only_null": max(abs(estimates["norm_only"].lo),
                              abs(estimates["norm_only"].hi)) <= null_max,
        "b30_blocks_transport": max(abs(estimates["b30_block_remaining"].lo),
                                    abs(estimates["b30_block_remaining"].hi)) <= null_max,
        "predicted_delta_rescues": estimates["predicted_delta_m30_rescue"].lo >= rescue_min,
        "random_null": max(abs(estimates["random_direction"].lo),
                           abs(estimates["random_direction"].hi)) <= null_max,
        "wrong_layer_null": max(abs(estimates["wrong_layer"].lo),
                                abs(estimates["wrong_layer"].hi)) <= null_max,
        "off_target_bounded": estimates["off_target_kl"].hi <= off_target_max,
    }
    evidence_payload = {
        "rows": [
            {
                **asdict(row),
                "dependency_keys": list(row.dependency_keys),
            }
            for row in sorted(rows, key=lambda item: item.variant_id)
        ],
        "decision_contract": {
            "rescue_min": float(rescue_min),
            "null_equivalence": float(null_max),
            "off_target_max": float(off_target_max),
            "min_clusters": int(min_clusters),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
    }
    return {
        "claim": ("direction-mediated causal variant transport"
                  if all(checks.values()) else "claim not established"),
        "checks": checks,
        "estimates": {k: v.as_row() for k, v in estimates.items()},
        "prediction_method": next(iter(methods)),
        "prediction_sha256": expected_prediction_sha256,
        "prediction_source_split": expected_prediction_source_split,
        "variant_set_sha256": expected_variant_set_sha256,
        "variant_cohort_sha256": expected_variant_cohort_sha256,
        "provenance_valid": True,
        "causal_evidence_sha256": _canonical_sha256(evidence_payload),
        "decision_contract": evidence_payload["decision_contract"],
    }


@dataclass(frozen=True)
class MechanismFingerprint:
    task_id: str
    task_family: str
    b28_content_write_magnitude: float
    b28_direction_stability: float
    b29_parallel_perp_ratio: float
    b29_gain: float
    scale_q: float
    scale_plateau_position: float
    b30_mediation_fraction: float
    b30_predicted_rescue_fraction: float
    x31_carrier_sensitivity: float
    offpath_leakage: float
    target: float
    source_split: str
    source_role: str
    source_artifact_sha256: str
    features_frozen_before_labels: bool
    feature_row_sha256: str
    baselines: tuple[float, ...] = ()

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FINGERPRINT_AXES],
                          dtype=np.float64)

    def _feature_payload(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "axes": {name: float(getattr(self, name)) for name in FINGERPRINT_AXES},
            "baselines": [float(value) for value in self.baselines],
            "source_split": self.source_split,
            "source_role": self.source_role,
            "source_artifact_sha256": self.source_artifact_sha256,
            "features_frozen_before_labels": self.features_frozen_before_labels,
        }

    def verify_feature_commitment(self) -> None:
        if not self.task_id or not self.task_family:
            raise ValueError("task_id and task_family must be non-empty")
        if self.source_role not in {"discovery", "development"}:
            raise RuntimeError("fingerprint features must come from discovery/development")
        if not self.source_split:
            raise ValueError("fingerprint source_split is empty")
        if not _is_sha256(self.source_artifact_sha256):
            raise ValueError("source_artifact_sha256 is not a SHA-256 digest")
        if not self.features_frozen_before_labels:
            raise RuntimeError("fingerprint features were not frozen before labels")
        if _canonical_sha256(self._feature_payload()) != self.feature_row_sha256:
            raise RuntimeError("mechanism fingerprint changed after feature sealing")

    @classmethod
    def create(cls, *, task_id: str, task_family: str,
               b28_content_write_magnitude: float,
               b28_direction_stability: float,
               b29_parallel_perp_ratio: float, b29_gain: float,
               scale_q: float, scale_plateau_position: float,
               b30_mediation_fraction: float,
               b30_predicted_rescue_fraction: float,
               x31_carrier_sensitivity: float, offpath_leakage: float,
               target: float, source_split: str, source_role: str,
               source_artifact_sha256: str,
               features_frozen_before_labels: bool,
               baselines: Sequence[float] = ()) -> "MechanismFingerprint":
        values = dict(
            task_id=str(task_id), task_family=str(task_family),
            b28_content_write_magnitude=float(b28_content_write_magnitude),
            b28_direction_stability=float(b28_direction_stability),
            b29_parallel_perp_ratio=float(b29_parallel_perp_ratio),
            b29_gain=float(b29_gain), scale_q=float(scale_q),
            scale_plateau_position=float(scale_plateau_position),
            b30_mediation_fraction=float(b30_mediation_fraction),
            b30_predicted_rescue_fraction=float(b30_predicted_rescue_fraction),
            x31_carrier_sensitivity=float(x31_carrier_sensitivity),
            offpath_leakage=float(offpath_leakage), target=float(target),
            source_split=str(source_split), source_role=str(source_role),
            source_artifact_sha256=str(source_artifact_sha256),
            features_frozen_before_labels=bool(features_frozen_before_labels),
            baselines=tuple(float(value) for value in baselines),
        )
        provisional = cls(feature_row_sha256="0" * 64, **values)
        return cls(feature_row_sha256=_canonical_sha256(
            provisional._feature_payload()), **values)


def seal_mechanism_fingerprints(rows: Sequence[MechanismFingerprint]) -> str:
    """Commit the complete pre-label feature table, excluding outcome targets."""
    if not rows:
        raise ValueError("cannot seal an empty mechanism fingerprint table")
    if len({row.task_id for row in rows}) != len(rows):
        raise ValueError("task_id must be unique")
    for row in rows:
        row.verify_feature_commitment()
    return _canonical_sha256([
        row._feature_payload() for row in sorted(rows, key=lambda item: item.task_id)
    ])


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                   ridge: float) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    mu, sd = train_x.mean(0), train_x.std(0)
    sd[sd < 1e-12] = 1.0
    X = (train_x - mu) / sd
    T = (test_x - mu) / sd
    X = np.column_stack([np.ones(len(X)), X])
    T = np.column_stack([np.ones(len(T)), T])
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    system = X.T @ X + penalty
    rhs = X.T @ train_y
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        # ``ridge=0`` is part of the public contract.  A duplicated or
        # constant frozen axis must therefore have the well-defined minimum-
        # norm OLS solution rather than failing according to BLAS details.
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return T @ coef


def leave_family_out_fingerprint(rows: Sequence[MechanismFingerprint], *,
                                 ridge: float, labels_opened: bool,
                                 expected_feature_sha256: str,
                                 include_baselines: bool = False,
                                 drop_axis: str | None = None) -> dict:
    """Predict held-out task families with axes fixed before labels were opened."""
    if not labels_opened:
        raise RuntimeError("task labels are closed; seal raw-mechanism features first")
    if not rows:
        raise ValueError("mechanism fingerprint table is empty")
    if not _is_sha256(expected_feature_sha256):
        raise ValueError("expected_feature_sha256 is not a SHA-256 digest")
    if seal_mechanism_fingerprints(rows) != expected_feature_sha256:
        raise RuntimeError("fingerprint feature table differs from pre-label commitment")
    if len({r.task_id for r in rows}) != len(rows):
        raise ValueError("task_id must be unique")
    if not all(np.isfinite(np.r_[r.vector(), r.target, r.baselines]).all() for r in rows):
        raise ValueError("mechanism fingerprint table contains non-finite values")
    if drop_axis is not None and drop_axis not in FINGERPRINT_AXES:
        raise KeyError(drop_axis)
    families = sorted({r.task_family for r in rows})
    if len(families) < 3:
        raise ValueError("leave-family-out evaluation needs at least three task families")
    baseline_widths = {len(r.baselines) for r in rows}
    if len(baseline_widths) != 1:
        raise ValueError("baseline vector width differs across tasks")

    def features(r: MechanismFingerprint) -> np.ndarray:
        x = r.vector()
        if drop_axis is not None:
            x = np.delete(x, FINGERPRINT_AXES.index(drop_axis))
        if include_baselines:
            x = np.r_[x, np.asarray(r.baselines, dtype=np.float64)]
        return x

    pred = np.empty(len(rows), dtype=np.float64)
    fold = []
    for fam in families:
        te = np.asarray([i for i, r in enumerate(rows) if r.task_family == fam])
        tr = np.asarray([i for i, r in enumerate(rows) if r.task_family != fam])
        if len(tr) <= len(features(rows[0])):
            raise RuntimeError(f"family {fam!r} leaves too few training tasks")
        Xtr = np.stack([features(rows[i]) for i in tr])
        ytr = np.asarray([rows[i].target for i in tr], dtype=np.float64)
        Xte = np.stack([features(rows[i]) for i in te])
        pred[te] = _ridge_predict(Xtr, ytr, Xte, ridge)
        fold.extend({"task_id": rows[i].task_id, "family": fam,
                     "observed": rows[i].target, "predicted": float(pred[i])}
                    for i in te)
    y = np.asarray([r.target for r in rows], dtype=np.float64)
    sse = float(np.square(y - pred).sum())
    sst = float(np.square(y - y.mean()).sum())
    return {
        "r2": (float(1 - sse / sst) if sst > 0 else float("nan")),
        "mae": float(np.abs(y - pred).mean()),
        "predictions": fold,
        "axes": [a for a in FINGERPRINT_AXES if a != drop_axis],
        "includes_baselines": include_baselines,
        "ridge": ridge,
        "feature_sha256": expected_feature_sha256,
        "features_frozen_before_labels": True,
    }


@dataclass(frozen=True)
class DiagnosticSignals:
    case_id: str
    writer_deficit: float
    amplifier_deficit: float
    reencoder_deficit: float
    calibration_deficit: float


def diagnose_stage(signals: DiagnosticSignals, *, minimum_deficit: float,
                   uniqueness_gap: float) -> str:
    values = dict(zip(STAGES, (signals.writer_deficit, signals.amplifier_deficit,
                              signals.reencoder_deficit, signals.calibration_deficit)))
    order = sorted(values, key=values.get, reverse=True)
    if values[order[0]] < minimum_deficit or values[order[0]] - values[order[1]] < uniqueness_gap:
        return "unresolved"
    return order[0]


@dataclass(frozen=True)
class RepairOutcome:
    case_id: str
    predicted_stage: str
    repair_stage: str
    recovered_fraction: float
    off_target_kl: float
    diagnosis_sha256: str
    validation_split: str
    held_out: bool
    dependency_keys: tuple[str, ...]


@dataclass(frozen=True)
class SealedDiagnosisTable:
    case_stages: tuple[tuple[str, str], ...]
    source_split: str
    source_sha256: str
    diagnosis_config_sha256: str
    sha256: str

    def verify(self) -> None:
        if (not self.case_stages or not self.source_split or
                not _is_sha256(self.source_sha256) or
                not _is_sha256(self.diagnosis_config_sha256)):
            raise ValueError("invalid sealed diagnosis provenance")
        cases = [case_id for case_id, _ in self.case_stages]
        if len(set(cases)) != len(cases) or any(not case_id for case_id in cases):
            raise ValueError("diagnosis case IDs must be unique and non-empty")
        if any(stage not in STAGES for _, stage in self.case_stages):
            raise ValueError("sealed diagnosis stage must be W/A/R/K")
        payload = {
            "case_stages": [list(item) for item in self.case_stages],
            "source_split": self.source_split,
            "source_sha256": self.source_sha256,
            "diagnosis_config_sha256": self.diagnosis_config_sha256,
        }
        if _canonical_sha256(payload) != self.sha256:
            raise RuntimeError("diagnosis table changed after sealing")


def seal_diagnosis_table(case_stages: Mapping[str, str], *, source_split: str,
                         source_sha256: str,
                         diagnosis_config_sha256: str) -> SealedDiagnosisTable:
    items = tuple(sorted((str(case_id), str(stage))
                         for case_id, stage in case_stages.items()))
    payload = {
        "case_stages": [list(item) for item in items],
        "source_split": str(source_split),
        "source_sha256": str(source_sha256),
        "diagnosis_config_sha256": str(diagnosis_config_sha256),
    }
    table = SealedDiagnosisTable(
        case_stages=items, source_split=str(source_split),
        source_sha256=str(source_sha256),
        diagnosis_config_sha256=str(diagnosis_config_sha256),
        sha256=_canonical_sha256(payload),
    )
    table.verify()
    return table


def debugger_verdict(rows: Sequence[RepairOutcome], *,
                     diagnosis: SealedDiagnosisTable,
                     expected_validation_split: str, recovery_min: float,
                     specificity_gap: float, off_target_max: float,
                     min_clusters: int, n_boot: int = 5000,
                     alpha: float = 0.05, seed: int = 42) -> dict:
    """Blind 4-way crossover bound to a sealed pre-repair diagnosis table."""
    if not rows:
        raise ValueError("repair table is empty")
    diagnosis.verify()
    if not expected_validation_split:
        raise ValueError("expected_validation_split is empty")
    if min(recovery_min, specificity_gap, off_target_max) < 0:
        raise ValueError("debugger decision margins must be non-negative")
    if min_clusters < 2 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError("invalid debugger bootstrap configuration")
    sealed = dict(diagnosis.case_stages)
    by_case: dict[str, list[RepairOutcome]] = {}
    for row in rows:
        if row.predicted_stage not in STAGES or row.repair_stage not in STAGES:
            raise ValueError("diagnostic/repair stage must be W/A/R/K")
        if (row.diagnosis_sha256 != diagnosis.sha256 or not row.held_out or
                row.validation_split != expected_validation_split or
                not row.dependency_keys):
            raise RuntimeError(
                "repair outcome is not bound to the sealed held-out diagnosis panel")
        if row.case_id not in sealed or row.predicted_stage != sealed[row.case_id]:
            raise RuntimeError("repair outcome changed a pre-repair diagnosis")
        if not np.isfinite((row.recovered_fraction, row.off_target_kl)).all():
            raise ValueError("repair outcome contains non-finite values")
        if row.off_target_kl < 0:
            raise ValueError("repair off_target_kl must be non-negative")
        by_case.setdefault(row.case_id, []).append(row)
    if set(by_case) != set(sealed):
        raise RuntimeError("repair case set differs from the sealed diagnosis table")
    case_rows = []
    for cid, rs in sorted(by_case.items()):
        if {r.repair_stage for r in rs} != set(STAGES) or len(rs) != len(STAGES):
            raise RuntimeError(f"case {cid!r} needs exactly one blind outcome per repair")
        if len({r.predicted_stage for r in rs}) != 1:
            raise RuntimeError(f"case {cid!r} changed diagnosis after repair outcomes")
        predicted = rs[0].predicted_stage
        chosen = next(r for r in rs if r.repair_stage == predicted)
        others = [r for r in rs if r.repair_stage != predicted]
        keys = set(row.dependency_keys for row in rs)
        if len(keys) != 1:
            raise RuntimeError(f"case {cid!r} changed dependency keys across repairs")
        best_wrong = max(r.recovered_fraction for r in others)
        case_rows.append({
            "case_id": cid, "predicted_stage": predicted,
            "selected_recovery": chosen.recovered_fraction,
            "best_wrong_recovery": best_wrong,
            "specificity_gap": chosen.recovered_fraction - best_wrong,
            "off_target_kl": chosen.off_target_kl,
            "keys": chosen.dependency_keys,
        })

    def boot(field: str, offset: int):
        return cluster_bootstrap(
            [Row(float(item[field]), keys=item["keys"], unit_id=item["case_id"])
             for item in case_rows],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )

    recovery = boot("selected_recovery", 0)
    gap = boot("specificity_gap", 1)
    off_target = boot("off_target_kl", 2)
    checks = {
        "selected_repair_recovers": recovery.lo >= recovery_min,
        "selected_beats_best_wrong": gap.lo >= specificity_gap,
        "off_target_bounded": off_target.hi <= off_target_max,
    }
    return {
        "claim": "stage-specific debugger repair" if all(checks.values()) else
                 "debugger repair claim not established",
        "checks": checks, "n_cases": len(case_rows),
        "all_passed": all(checks.values()), "cases": case_rows,
        "selected_recovery": recovery.as_row(),
        "selected_vs_best_wrong": gap.as_row(),
        "off_target_kl": off_target.as_row(),
        "diagnosis_sha256": diagnosis.sha256,
        "validation_split": expected_validation_split,
    }


@dataclass(frozen=True)
class CarrierDoseRow:
    unit_id: str
    split: str
    source_sha256: str
    dose: float
    log_beta: float
    d_shape: float
    rank_changed: bool
    nll: float
    ece: float
    dependency_keys: tuple[str, ...]


def _carrier_table_sha256(rows: Sequence[CarrierDoseRow]) -> str:
    return _canonical_sha256([
        {**asdict(row), "dependency_keys": list(row.dependency_keys)}
        for row in sorted(rows, key=lambda item: (item.unit_id, item.dose))
    ])


def _validate_carrier_grid(rows: Sequence[CarrierDoseRow]) -> tuple[tuple[str, ...],
                                                                    tuple[float, ...]]:
    if not rows:
        raise ValueError("carrier dose table is empty")
    pairs = [(row.unit_id, float(row.dose)) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("carrier dose table contains duplicate unit/dose cells")
    units = tuple(sorted({row.unit_id for row in rows}))
    doses = tuple(sorted({float(row.dose) for row in rows}))
    expected = {(unit_id, dose) for unit_id in units for dose in doses}
    if set(pairs) != expected:
        raise RuntimeError("carrier dose table is not a complete unit x dose grid")
    for row in rows:
        numeric = (row.dose, row.log_beta, row.d_shape, row.nll, row.ece)
        if (not row.unit_id or not row.split or not row.dependency_keys or
                not _is_sha256(row.source_sha256) or
                not np.isfinite(numeric).all()):
            raise ValueError("invalid carrier dose row")
        if row.d_shape < 0 or row.nll < 0 or not 0 <= row.ece <= 1:
            raise ValueError("carrier shape/NLL/ECE values are outside their domains")
    return units, doses


@dataclass(frozen=True)
class CarrierDoseSelection:
    selected_dose: float
    dose_grid: tuple[float, ...]
    development_split: str
    development_unit_ids: tuple[str, ...]
    development_source_sha256: str
    development_table_sha256: str
    d_shape_max: float
    sha256: str

    def verify(self) -> None:
        payload = {
            "selected_dose": self.selected_dose,
            "dose_grid": list(self.dose_grid),
            "development_split": self.development_split,
            "development_unit_ids": list(self.development_unit_ids),
            "development_source_sha256": self.development_source_sha256,
            "development_table_sha256": self.development_table_sha256,
            "d_shape_max": self.d_shape_max,
        }
        if (not self.development_split or not self.development_unit_ids or
                not _is_sha256(self.development_source_sha256) or
                not _is_sha256(self.development_table_sha256)):
            raise ValueError("invalid carrier dose selection provenance")
        if (len(set(self.development_unit_ids)) != len(self.development_unit_ids) or
                any(not value for value in self.development_unit_ids) or
                len(self.dose_grid) < 2 or
                tuple(sorted(set(self.dose_grid))) != self.dose_grid or
                self.selected_dose not in self.dose_grid or
                self.d_shape_max < 0 or not np.isfinite(self.d_shape_max)):
            raise ValueError("invalid sealed carrier dose/grid selection")
        if _canonical_sha256(payload) != self.sha256:
            raise RuntimeError("carrier dose selection changed after sealing")


def select_carrier_dose_development(
    rows: Sequence[CarrierDoseRow], *, d_shape_max: float,
    development_split: str, expected_source_sha256: str,
) -> CarrierDoseSelection:
    """Select and seal one scalar using the development grid only."""
    units, doses = _validate_carrier_grid(rows)
    if not development_split or not _is_sha256(expected_source_sha256):
        raise ValueError("invalid development split/source commitment")
    if (any(row.split != development_split for row in rows) or
            any(row.source_sha256 != expected_source_sha256 for row in rows)):
        raise RuntimeError("carrier selection includes a non-development source")
    if d_shape_max < 0 or not np.isfinite(d_shape_max):
        raise ValueError("d_shape_max must be finite and non-negative")
    by_dose: dict[float, list[CarrierDoseRow]] = {}
    for r in rows:
        by_dose.setdefault(float(r.dose), []).append(r)
    eligible = []
    for dose, rs in by_dose.items():
        if max(r.d_shape for r in rs) <= d_shape_max and not any(r.rank_changed for r in rs):
            eligible.append((float(np.mean([r.nll for r in rs])),
                             float(np.mean([r.ece for r in rs])), abs(dose), dose))
    if not eligible:
        raise RuntimeError("no carrier dose preserves content within the frozen margin")
    selected = float(min(eligible)[-1])
    table_sha = _carrier_table_sha256(rows)
    payload = {
        "selected_dose": selected, "dose_grid": list(doses),
        "development_split": development_split,
        "development_unit_ids": list(units),
        "development_source_sha256": expected_source_sha256,
        "development_table_sha256": table_sha,
        "d_shape_max": float(d_shape_max),
    }
    result = CarrierDoseSelection(
        selected_dose=selected, dose_grid=doses,
        development_split=development_split, development_unit_ids=units,
        development_source_sha256=expected_source_sha256,
        development_table_sha256=table_sha, d_shape_max=float(d_shape_max),
        sha256=_canonical_sha256(payload),
    )
    result.verify()
    return result


def carrier_calibration_verdict(
    rows: Sequence[CarrierDoseRow], *, selection: CarrierDoseSelection,
    expected_locked_split: str, expected_locked_source_sha256: str,
    minimum_beta_span: float, monotonic_fraction_min: float,
    rank_change_equivalence: float, min_clusters: int,
    n_boot: int = 5000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    selection.verify()
    units, doses = _validate_carrier_grid(rows)
    if (not expected_locked_split or
            not _is_sha256(expected_locked_source_sha256)):
        raise ValueError("invalid locked carrier split/source commitment")
    if expected_locked_split == selection.development_split:
        raise RuntimeError("carrier validation split equals development split")
    if expected_locked_source_sha256 == selection.development_source_sha256:
        raise RuntimeError("carrier validation reuses the development source")
    if set(units) & set(selection.development_unit_ids):
        raise RuntimeError("carrier validation reuses development unit IDs")
    if doses != selection.dose_grid:
        raise RuntimeError("locked carrier dose grid differs from development")
    if (any(row.split != expected_locked_split for row in rows) or
            any(row.source_sha256 != expected_locked_source_sha256 for row in rows)):
        raise RuntimeError("carrier validation contains an uncommitted source")
    if (minimum_beta_span < 0 or not 0 <= monotonic_fraction_min <= 1 or
            rank_change_equivalence < 0 or min_clusters < 2 or n_boot < 100 or
            not 0 < alpha < 1):
        raise ValueError("invalid carrier validation thresholds/bootstrap")
    selected_dose = selection.selected_dose
    selected = [r for r in rows if abs(r.dose - selected_dose) < 1e-12]
    if not selected:
        raise ValueError("selected carrier dose is absent")
    curves: dict[str, list[CarrierDoseRow]] = {}
    for r in rows:
        curves.setdefault(r.unit_id, []).append(r)
    unit_summaries = []
    for unit_id, rs in curves.items():
        rs = sorted(rs, key=lambda r: r.dose)
        beta = np.asarray([r.log_beta for r in rs])
        keys = {row.dependency_keys for row in rs}
        if len(keys) != 1:
            raise RuntimeError(f"unit {unit_id!r} changes dependency keys across dose")
        unit_summaries.append({
            "unit_id": unit_id, "keys": rs[0].dependency_keys,
            "monotone": float(np.all(np.diff(beta) >= -1e-12) or
                              np.all(np.diff(beta) <= 1e-12)),
            "beta_span": float(beta.max() - beta.min()),
        })

    def boot_unit(field: str, offset: int):
        return cluster_bootstrap(
            [Row(float(item[field]), keys=item["keys"], unit_id=item["unit_id"])
             for item in unit_summaries],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )

    def boot_selected(field: str, offset: int):
        return cluster_bootstrap(
            [Row(float(getattr(row, field)), keys=row.dependency_keys,
                 unit_id=row.unit_id) for row in selected],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )

    monotone = boot_unit("monotone", 0)
    span = boot_unit("beta_span", 1)
    shape = boot_selected("d_shape", 2)
    rank_change = cluster_bootstrap(
        [Row(float(row.rank_changed), keys=row.dependency_keys, unit_id=row.unit_id)
         for row in selected], n_boot=n_boot, alpha=alpha, seed=seed + 3,
        min_clusters=min_clusters)
    nll = boot_selected("nll", 4)
    ece = boot_selected("ece", 5)
    checks = {
        "dose_is_monotone": monotone.lo >= monotonic_fraction_min,
        "beta_span_nontrivial": span.lo >= minimum_beta_span,
        "shape_preserved": shape.hi <= selection.d_shape_max,
        "ranking_preserved": rank_change.hi <= rank_change_equivalence,
    }
    return {"claim": "internal calibration knob" if all(checks.values()) else
            "calibration-only claim not established", "checks": checks,
            "selected_dose": selected_dose,
            "selection_sha256": selection.sha256,
            "locked_table_sha256": _carrier_table_sha256(rows),
            "locked_split": expected_locked_split,
            "estimates": {
                "monotonic_fraction": monotone.as_row(),
                "beta_span": span.as_row(), "d_shape": shape.as_row(),
                "rank_change_fraction": rank_change.as_row(),
                "nll": nll.as_row(), "ece": ece.as_row(),
            }}


@dataclass(frozen=True)
class PathUseRow:
    unit_id: str
    distance_bp: int
    task_family: str
    long_path_loss: float
    direct_path_loss: float
    both_path_loss: float
    dependency_keys: tuple[str, ...]


def long_context_path_verdict(rows: Sequence[PathUseRow], *, local_max_bp: int,
                              long_min_bp: int, local_equivalence: float,
                              long_superiority: float,
                              joint_noninferiority: float,
                              min_clusters: int, n_boot: int = 5000,
                              alpha: float = 0.05, seed: int = 42) -> dict:
    """Test distance-specific path necessity using ablation-loss contrasts.

    ``long_path_loss`` is the loss after ablating the long/HCL path and
    ``direct_path_loss`` after ablating the direct path.  A positive
    ``long_path_loss - direct_path_loss`` therefore means the long path is more
    necessary, not better-performing by itself.
    """
    local = [r for r in rows if r.distance_bp <= local_max_bp]
    distant = [r for r in rows if r.distance_bp >= long_min_bp]
    if not local or not distant:
        raise ValueError("both local and long-distance strata are required")
    if (local_max_bp >= long_min_bp or min(local_equivalence, long_superiority,
                                           joint_noninferiority) < 0 or
            min_clusters < 2 or n_boot < 100 or not 0 < alpha < 1):
        raise ValueError("invalid path strata/margins/bootstrap")
    if len({row.unit_id for row in rows}) != len(rows):
        raise ValueError("path-use unit_id must be unique")
    if any(not row.dependency_keys or not np.isfinite((
            row.long_path_loss, row.direct_path_loss, row.both_path_loss)).all()
           for row in rows):
        raise ValueError("invalid path-use row")

    def boot(source: Sequence[PathUseRow], value, offset: int):
        return cluster_bootstrap(
            [Row(float(value(row)), keys=row.dependency_keys,
                 unit_id=row.unit_id, stratum=row.task_family) for row in source],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters)

    local_gap = boot(local, lambda row: row.long_path_loss - row.direct_path_loss, 0)
    distant_gap = boot(
        distant, lambda row: row.long_path_loss - row.direct_path_loss, 1)
    joint_gap = boot(
        distant,
        lambda row: row.both_path_loss -
        max(row.long_path_loss, row.direct_path_loss), 2)
    interaction_lo = distant_gap.lo - local_gap.hi
    interaction_hi = distant_gap.hi - local_gap.lo
    checks = {
        "local_paths_redundant": (
            local_gap.lo >= -local_equivalence and
            local_gap.hi <= local_equivalence),
        "long_path_necessary_at_distance": distant_gap.lo >= long_superiority,
        "distance_interaction": interaction_lo >= long_superiority,
        "joint_ablation_not_weaker": joint_gap.lo >= -joint_noninferiority,
    }
    return {"claim": "distance-specific HCL path use" if all(checks.values()) else
            "path heterogeneity not established", "checks": checks,
            "local_ablation_loss_gap": local_gap.as_row(),
            "distant_ablation_loss_gap": distant_gap.as_row(),
            "joint_noninferiority_gap": joint_gap.as_row(),
            "interaction_ci": {"ci_lo": interaction_lo,
                               "ci_hi": interaction_hi}}


@dataclass(frozen=True)
class CompressionRow:
    unit_id: str
    method: str
    flops_removed: float
    d_shape: float
    nll_delta: float
    task_drop: float
    dependency_keys: tuple[str, ...]


@dataclass(frozen=True)
class MechanismCompressionPlan:
    """An evidence-gated, ordered compression recipe.

    This object is deliberately separate from ``compression_comparison``:
    the latter evaluates a finished model, while this object records exactly
    which frozen causal results authorised each destructive transform.
    """

    dropped_paths: tuple[str, ...]
    kept_b28_channels: tuple[int, ...]
    kept_b29_channels: tuple[int, ...]
    b30_residual_rank: int
    canonical_branch_norm: float
    hcl_gate_threshold: float
    evidence_sha256: str
    ordered_operations: tuple[str, ...] = (
        "drop_verified_equivalent_paths",
        "reduce_b28_b29_to_frozen_causal_channels",
        "preserve_b30_causal_subspace_then_low_rank_residual",
        "canonicalize_only_norm-equivalent_branches",
        "gate_hcl_only_with_frozen_router",
    )


def build_mechanism_compression_plan(*, evidence: Mapping[str, bool],
                                     dropped_paths: Sequence[str],
                                     kept_b28_channels: Sequence[int],
                                     kept_b29_channels: Sequence[int],
                                     b30_residual_rank: int,
                                     canonical_branch_norm: float,
                                     hcl_gate_threshold: float) -> MechanismCompressionPlan:
    """Build the requested compression sequence only after every causal gate.

    A path-specific key (for example ``drop:g30``) is required for every path
    to be removed.  This prevents an exploratory null from silently becoming
    model surgery.
    """
    paths = tuple(dict.fromkeys(str(x) for x in dropped_paths))
    required = {
        "b28_noncausal_channels_equivalent",
        "b29_noncausal_channels_equivalent",
        "b30_causal_subspace_sufficient",
        "norm_canonicalization_equivalent",
        "hcl_router_generalizes",
        *(f"drop:{path}" for path in paths),
    }
    missing = sorted(k for k in required if evidence.get(k) is not True)
    if missing:
        raise RuntimeError(f"compression plan lacks passed causal gates: {missing}")

    def channels(values: Sequence[int], label: str) -> tuple[int, ...]:
        out = tuple(sorted(set(int(v) for v in values)))
        if not out or out[0] < 0:
            raise ValueError(f"{label} needs non-negative frozen channel indices")
        return out

    b28 = channels(kept_b28_channels, "b28")
    b29 = channels(kept_b29_channels, "b29")
    if b30_residual_rank < 0 or canonical_branch_norm <= 0 or not np.isfinite(
            (canonical_branch_norm, hcl_gate_threshold)).all():
        raise ValueError("invalid compression rank, canonical norm, or router threshold")
    payload = "|".join([
        *paths, *(f"b28:{x}" for x in b28), *(f"b29:{x}" for x in b29),
        f"rank:{b30_residual_rank}", f"norm:{canonical_branch_norm:.17g}",
        f"gate:{hcl_gate_threshold:.17g}",
        *(f"{k}:{int(bool(evidence[k]))}" for k in sorted(required)),
    ])
    return MechanismCompressionPlan(
        dropped_paths=paths, kept_b28_channels=b28, kept_b29_channels=b29,
        b30_residual_rank=int(b30_residual_rank),
        canonical_branch_norm=float(canonical_branch_norm),
        hcl_gate_threshold=float(hcl_gate_threshold),
        evidence_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )


@dataclass(frozen=True)
class ReducedBilinearWeights:
    W1: Tensor
    W2: Tensor
    W3: Tensor
    kept_channels: tuple[int, ...]


def reduce_bilinear_channels(W1: Tensor, W2: Tensor, W3: Tensor,
                             kept_channels: Sequence[int]) -> ReducedBilinearWeights:
    """Physically gather frozen causal channels from a bilinear MLP.

    For input ``z``, the reduced module computes exactly the contribution of
    the selected native channels: ``((z@W1.T)*(z@W2.T))@W3.T``.
    """
    if W1.ndim != 2 or W2.shape != W1.shape or W3.ndim != 2 \
            or W3.shape[1] != W1.shape[0]:
        raise ValueError("incompatible bilinear weight shapes")
    keep = tuple(sorted(set(int(k) for k in kept_channels)))
    if not keep or keep[0] < 0 or keep[-1] >= W1.shape[0]:
        raise IndexError("kept channel is empty or outside the hidden width")
    index = torch.as_tensor(keep, dtype=torch.long, device=W1.device)
    if W2.device != W1.device or W3.device != W1.device:
        raise ValueError("bilinear weights are on different devices")
    return ReducedBilinearWeights(
        W1=W1.index_select(0, index).clone(),
        W2=W2.index_select(0, index).clone(),
        W3=W3.index_select(1, index).clone(),
        kept_channels=keep,
    )


@dataclass(frozen=True)
class SubspacePreservingWeight:
    weight: Tensor
    causal_rank: int
    residual_rank: int
    relative_error: float
    causal_projection_error: float


@torch.no_grad()
def compress_weight_preserve_output_subspace(weight: Tensor, basis: Tensor, *,
                                             residual_rank: int) -> SubspacePreservingWeight:
    """Low-rank a weight while preserving its frozen causal output subspace.

    ``basis`` contains orthonormal output-space columns.  The returned matrix
    preserves ``P @ weight`` up to floating-point roundoff and only truncates
    the orthogonal residual; it therefore cannot trade causal b30 directions
    for a lower reconstruction error elsewhere.
    """
    if weight.ndim != 2 or basis.ndim != 2 or basis.shape[0] != weight.shape[0]:
        raise ValueError("weight must be [d_out,d_in] and basis [d_out,k]")
    if weight.device != basis.device:
        raise ValueError("weight and basis are on different devices")
    if residual_rank < 0:
        raise ValueError("residual_rank must be non-negative")
    wd, U = weight.detach().double(), basis.detach().double()
    if not torch.isfinite(wd).all() or not torch.isfinite(U).all():
        raise ValueError("weight or basis is non-finite")
    gram = U.T @ U
    if U.shape[1] == 0 or not torch.allclose(
            gram, torch.eye(U.shape[1], device=U.device, dtype=U.dtype),
            atol=1e-8, rtol=1e-6):
        raise ValueError("basis columns must be non-empty and orthonormal")
    causal = U @ (U.T @ wd)
    residual = wd - causal
    max_rank = min(residual.shape)
    rank = min(int(residual_rank), max_rank)
    if rank:
        L, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        approx_residual = (L[:, :rank] * S[:rank]) @ Vh[:rank]
        # Remove numerical spill-back into U exactly in the working dtype.
        approx_residual = approx_residual - U @ (U.T @ approx_residual)
    else:
        approx_residual = torch.zeros_like(residual)
    compressed = causal + approx_residual
    projection_error = float((U.T @ (compressed - wd)).norm() /
                             (U.T @ wd).norm().clamp_min(1e-30))
    relative_error = float((compressed - wd).norm() / wd.norm().clamp_min(1e-30))
    return SubspacePreservingWeight(
        weight=compressed.to(dtype=weight.dtype), causal_rank=U.shape[1],
        residual_rank=rank, relative_error=relative_error,
        causal_projection_error=projection_error,
    )


def canonicalize_update_norm(update: Tensor, target_norm: float,
                             *, eps: float = 1e-12) -> Tensor:
    """Replace only the last-axis norm, preserving every update direction."""
    if target_norm <= 0 or not np.isfinite(target_norm) or update.ndim < 1:
        raise ValueError("target_norm must be finite and positive")
    norms = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
    if torch.any(norms <= eps):
        raise ValueError("cannot canonicalize a zero update")
    return update * (target_norm / norms)


def conditional_path_output(path_output: Tensor, gate: Tensor) -> Tensor:
    """Apply a frozen boolean HCL router without changing active outputs."""
    if gate.dtype != torch.bool or tuple(gate.shape) != tuple(path_output.shape[:-1]):
        raise ValueError("gate must be boolean and match every non-feature axis")
    if gate.device != path_output.device:
        raise ValueError("gate and path_output are on different devices")
    return path_output * gate.unsqueeze(-1)


def compression_comparison(rows: Sequence[CompressionRow], *,
                           mechanism_method: str = "mechanism",
                           matched_tolerance: float = 0.01,
                           minimum_task_gain: float = 0.0,
                           minimum_shape_gain: float = 0.0,
                           minimum_nll_gain: float = 0.0,
                           min_clusters: int, n_boot: int = 5000,
                           alpha: float = 0.05, seed: int = 42) -> dict:
    """Paired cluster inference against every equal-compute baseline."""
    methods = sorted({r.method for r in rows})
    if mechanism_method not in methods or len(methods) < 3:
        raise ValueError("need mechanism, random and magnitude-like methods")
    if not rows or not all(np.isfinite((r.flops_removed, r.d_shape, r.nll_delta,
                                        r.task_drop)).all() for r in rows):
        raise ValueError("compression table is empty or non-finite")
    if (min(matched_tolerance, minimum_task_gain, minimum_shape_gain,
            minimum_nll_gain) < 0 or min_clusters < 2 or n_boot < 100 or
            not 0 < alpha < 1):
        raise ValueError("invalid compression margins/bootstrap")
    by_method = {m: [r for r in rows if r.method == m] for m in methods}
    for method, rs in by_method.items():
        ids = [r.unit_id for r in rs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate unit_id within compression method {method!r}")
        if any(not row.unit_id or not row.dependency_keys for row in rs):
            raise ValueError("compression rows need unit IDs and dependency keys")
    reference_ids = {r.unit_id for r in by_method[mechanism_method]}
    unmatched_units = [m for m, rs in by_method.items()
                       if {r.unit_id for r in rs} != reference_ids]
    if unmatched_units:
        raise RuntimeError(
            "compression methods were not evaluated on identical units: "
            f"{unmatched_units}"
        )
    by_method_unit = {method: {row.unit_id: row for row in method_rows}
                      for method, method_rows in by_method.items()}
    for unit_id in reference_ids:
        dependency_sets = {
            by_method_unit[method][unit_id].dependency_keys for method in methods}
        if len(dependency_sets) != 1:
            raise RuntimeError(
                f"compression unit {unit_id!r} changes dependency keys across methods")
    means = {}
    for m in methods:
        rs = by_method[m]
        means[m] = {"flops_removed": float(np.mean([r.flops_removed for r in rs])),
                    "d_shape": float(np.mean([r.d_shape for r in rs])),
                    "nll_delta": float(np.mean([r.nll_delta for r in rs])),
                    "task_drop": float(np.mean([r.task_drop for r in rs]))}
    paired = {}
    for method_index, method in enumerate(m for m in methods if m != mechanism_method):
        estimates = {}
        for field_index, (name, field) in enumerate((
            ("flops_difference", "flops_removed"),
            ("task_retention_gain", "task_drop"),
            ("shape_retention_gain", "d_shape"),
            ("nll_retention_gain", "nll_delta"),
        )):
            contrast_rows = []
            for unit_id in sorted(reference_ids):
                baseline = by_method_unit[method][unit_id]
                mechanism = by_method_unit[mechanism_method][unit_id]
                value = (getattr(baseline, field) - getattr(mechanism, field))
                contrast_rows.append(Row(
                    float(value), keys=mechanism.dependency_keys, unit_id=unit_id))
            estimates[name] = cluster_bootstrap(
                contrast_rows, n_boot=n_boot, alpha=alpha,
                seed=seed + method_index * 4 + field_index,
                min_clusters=min_clusters)
        checks = {
            "compute_equivalent": (
                estimates["flops_difference"].lo >= -matched_tolerance and
                estimates["flops_difference"].hi <= matched_tolerance),
            "task_retention_better": (
                estimates["task_retention_gain"].lo >= minimum_task_gain),
            "shape_retention_better": (
                estimates["shape_retention_gain"].lo >= minimum_shape_gain),
            "nll_retention_better": (
                estimates["nll_retention_gain"].lo >= minimum_nll_gain),
        }
        paired[method] = {
            "checks": checks, "passed": all(checks.values()),
            "estimates": {name: estimate.as_row()
                          for name, estimate in estimates.items()},
        }
    all_passed = bool(paired) and all(item["passed"] for item in paired.values())
    return {"methods": means, "paired_baselines": paired,
            "mechanism_best_task_retention": all(
                item["checks"]["task_retention_better"] for item in paired.values()),
            "mechanism_best_shape_retention": all(
                item["checks"]["shape_retention_better"] for item in paired.values()),
            "all_paired_claims_passed": all_passed,
            "bootstrap": {"min_clusters": min_clusters, "n_boot": n_boot,
                          "alpha": alpha, "seed": seed}}


@dataclass(frozen=True)
class SteeringScore:
    sequence: str
    target_alignment: float
    output_effect: float
    off_target_penalty: float

    @property
    def objective(self) -> float:
        return self.target_alignment + self.output_effect - self.off_target_penalty


def beam_search_sequence(seed_sequence: str, *,
                         score_batch: Callable[[Sequence[str]], Sequence[SteeringScore]],
                         editable_positions: Sequence[int], alphabet: str = "ACGT",
                         max_mutations: int = 3, beam_width: int = 16) -> list[SteeringScore]:
    """Gradient-free steering; final causal verification remains mandatory."""
    if not seed_sequence or beam_width < 1 or max_mutations < 1:
        raise ValueError("invalid sequence-search configuration")
    pos = sorted(set(int(p) for p in editable_positions))
    if any(p < 0 or p >= len(seed_sequence) for p in pos):
        raise IndexError("editable position outside sequence")
    beam = [seed_sequence]
    for _ in range(max_mutations):
        candidates = set(beam)
        for seq in beam:
            for p in pos:
                for base in alphabet:
                    if base != seq[p]:
                        candidates.add(seq[:p] + base + seq[p + 1:])
        scored = list(score_batch(sorted(candidates)))
        returned = [s.sequence for s in scored]
        if len(returned) != len(candidates) or set(returned) != candidates:
            raise RuntimeError("score_batch must return each candidate exactly once")
        scored.sort(key=lambda s: (-s.objective, s.sequence))
        beam = [s.sequence for s in scored[:beam_width]]
    return sorted(score_batch(beam), key=lambda s: (-s.objective, s.sequence))


@dataclass(frozen=True)
class SteeringValidation:
    unit_id: str
    sequence: str
    validation_split: str
    validation_source_sha256: str
    selection_sha256: str
    held_out: bool
    natural_alignment_gain: float
    output_gain: float
    channel_ablation_remaining: float
    component_rescue_fraction: float
    b30_block_remaining: float
    wrong_direction_effect: float
    random_direction_effect: float
    off_target_kl: float
    dependency_keys: tuple[str, ...]


@dataclass(frozen=True)
class SteeringSelection:
    sequence: str
    development_split: str
    development_unit_ids: tuple[str, ...]
    development_source_sha256: str
    search_config_sha256: str
    sha256: str

    def verify(self) -> None:
        if (not self.sequence or not self.development_split or
                not self.development_unit_ids or
                len(set(self.development_unit_ids)) != len(self.development_unit_ids) or
                any(not value for value in self.development_unit_ids) or
                not _is_sha256(self.development_source_sha256) or
                not _is_sha256(self.search_config_sha256)):
            raise ValueError("invalid steering selection provenance")
        payload = {
            "sequence": self.sequence,
            "development_split": self.development_split,
            "development_unit_ids": list(self.development_unit_ids),
            "development_source_sha256": self.development_source_sha256,
            "search_config_sha256": self.search_config_sha256,
        }
        if _canonical_sha256(payload) != self.sha256:
            raise RuntimeError("steering selection changed after sealing")


def seal_steering_selection(*, sequence: str, development_split: str,
                            development_unit_ids: Sequence[str],
                            development_source_sha256: str,
                            search_config_sha256: str) -> SteeringSelection:
    """Commit the development-selected sequence before locked validation."""
    ids = tuple(str(value) for value in development_unit_ids)
    payload = {
        "sequence": str(sequence), "development_split": str(development_split),
        "development_unit_ids": list(ids),
        "development_source_sha256": str(development_source_sha256),
        "search_config_sha256": str(search_config_sha256),
    }
    selection = SteeringSelection(
        sequence=str(sequence), development_split=str(development_split),
        development_unit_ids=ids,
        development_source_sha256=str(development_source_sha256),
        search_config_sha256=str(search_config_sha256),
        sha256=_canonical_sha256(payload),
    )
    selection.verify()
    return selection


def steering_verdict(rows: Sequence[SteeringValidation], *, effect_min: float,
                     null_max: float, rescue_min: float, off_target_max: float,
                     selection: SteeringSelection,
                     expected_validation_split: str,
                     expected_validation_source_sha256: str,
                     min_clusters: int,
                     n_boot: int = 5000, alpha: float = 0.05,
                     seed: int = 42) -> dict:
    """Validate one frozen search winner on a matched, held-out causal panel."""
    if not rows:
        raise ValueError("steering validation panel is empty")
    selection.verify()
    if (not expected_validation_split or
            not _is_sha256(expected_validation_source_sha256)):
        raise ValueError("invalid validation split/source commitment")
    if expected_validation_split == selection.development_split:
        raise RuntimeError("steering validation split equals the search split")
    if expected_validation_source_sha256 == selection.development_source_sha256:
        raise RuntimeError("steering validation reuses the search source")
    if min(effect_min, null_max, rescue_min, off_target_max) < 0:
        raise ValueError("steering margins must be non-negative")
    if min_clusters < 2 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError("invalid steering bootstrap configuration")
    sequences = {row.sequence for row in rows}
    if sequences != {selection.sequence}:
        raise RuntimeError("locked steering sequence differs from the sealed winner")
    if len({row.unit_id for row in rows}) != len(rows):
        raise ValueError("steering validation unit_id must be unique")
    if {row.unit_id for row in rows} & set(selection.development_unit_ids):
        raise RuntimeError("steering validation reuses development unit IDs")
    fields = (
        "natural_alignment_gain", "output_gain", "channel_ablation_remaining",
        "component_rescue_fraction", "b30_block_remaining",
        "wrong_direction_effect", "random_direction_effect", "off_target_kl",
    )
    for row in rows:
        if (not row.unit_id or not row.dependency_keys or not row.held_out or
                row.selection_sha256 != selection.sha256 or
                row.validation_split != expected_validation_split or
                row.validation_source_sha256 != expected_validation_source_sha256):
            raise RuntimeError(
                "steering row is not from the committed held-out matched panel")
        if not np.isfinite([getattr(row, field) for field in fields]).all():
            raise ValueError("steering validation row contains non-finite values")
        if row.off_target_kl < 0:
            raise ValueError("steering off_target_kl must be non-negative")

    estimates = {}
    for offset, field in enumerate(fields):
        estimates[field] = cluster_bootstrap(
            [Row(float(getattr(row, field)), keys=row.dependency_keys,
                 unit_id=row.unit_id) for row in rows],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )
    equivalent = lambda estimate: (
        estimate.lo >= -null_max and estimate.hi <= null_max)
    checks = {
        "natural_forward_alignment": estimates["natural_alignment_gain"].lo >= effect_min,
        "desired_output": estimates["output_gain"].lo >= effect_min,
        "causal_channel_needed": equivalent(estimates["channel_ablation_remaining"]),
        "component_rescues": estimates["component_rescue_fraction"].lo >= rescue_min,
        "b30_transports": equivalent(estimates["b30_block_remaining"]),
        "wrong_direction_equivalent": equivalent(estimates["wrong_direction_effect"]),
        "random_direction_equivalent": equivalent(estimates["random_direction_effect"]),
        "off_target_bounded": estimates["off_target_kl"].hi <= off_target_max,
    }
    return {"sequence": next(iter(sequences)), "checks": checks,
            "claim": "mechanism-guided sequence" if all(checks.values()) else
            "candidate only",
            "estimates": {field: estimate.as_row()
                          for field, estimate in estimates.items()},
            "selection_sha256": selection.sha256,
            "validation_split": expected_validation_split,
            "validation_source_sha256": expected_validation_source_sha256,
            "held_out": True}


# ---------------------------------------------------------------------------
# External biological assay validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalAssayRow:
    """One prediction frozen before an external assay label was opened.

    ``baseline_prediction`` is a predeclared non-mechanistic baseline evaluated
    on the same variant. ``matched_control_effect`` is the assay effect of the
    sequence-matched negative control.  Dependency keys must include every
    shared locus/donor/window so the bootstrap cannot treat correlated rows as
    independent.
    """

    variant_id: str
    cohort_id: str
    assay_name: str
    predicted_effect: float
    observed_effect: float
    observed_se: float
    baseline_prediction: float
    matched_control_effect: float
    prediction_sha256: str
    dependency_keys: tuple[str, ...]
    source_role: str = "external"


def seal_external_predictions(predictions: Mapping[str, float]) -> str:
    """Return a deterministic commitment for predictions made before labels.

    The hash deliberately contains only variant IDs and signed predictions;
    assay outcomes must not enter this artifact.
    """
    if not predictions:
        raise ValueError("cannot seal an empty prediction table")
    clean = {}
    for key, value in predictions.items():
        if not key or not np.isfinite(float(value)):
            raise ValueError("prediction IDs and values must be finite and non-empty")
        clean[str(key)] = float(value)
    if len(clean) != len(predictions):
        raise ValueError("prediction IDs are not unique after string normalization")
    payload = "\n".join(f"{key}\t{clean[key]:.17g}" for key in sorted(clean))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def external_assay_verdict(
    rows: Sequence[ExternalAssayRow], *, expected_prediction_sha256: str,
    expected_variant_set_sha256: str, expected_variant_cohort_sha256: str,
    labels_opened: bool, resolvable_z: float, sign_concordance_min: float,
    error_gain_min: float, specificity_gap_min: float, min_clusters: int,
    n_boot: int = 5000, seed: int = 42,
) -> dict:
    """Validate frozen mechanism predictions on blinded external assay data.

    This is intentionally an *external predictive* result.  It becomes a
    mechanistic biological claim only when combined with the model-internal
    necessity/rescue result via :func:`external_causal_triangulation`.
    """
    if not labels_opened:
        raise RuntimeError("external assay labels are closed")
    if not rows:
        raise ValueError("external assay table is empty")
    if resolvable_z <= 0 or not 0.5 <= sign_concordance_min <= 1:
        raise ValueError("invalid external-assay decision thresholds")
    if not _is_sha256(expected_prediction_sha256):
        raise ValueError("expected_prediction_sha256 is not a SHA-256 digest")
    if (not _is_sha256(expected_variant_set_sha256) or
            not _is_sha256(expected_variant_cohort_sha256)):
        raise ValueError("variant/cohort commitment is not a SHA-256 digest")
    if len({(r.assay_name, r.variant_id) for r in rows}) != len(rows):
        raise ValueError("duplicate assay/variant row")
    committed_predictions: dict[str, float] = {}
    committed_cohorts: dict[str, str] = {}
    for r in rows:
        values = (r.predicted_effect, r.observed_effect, r.observed_se,
                  r.baseline_prediction, r.matched_control_effect)
        if (not r.variant_id or not r.cohort_id or not r.assay_name or
                not r.dependency_keys or
                not np.isfinite(values).all() or r.observed_se <= 0):
            raise ValueError(f"invalid external assay row {r.variant_id!r}")
        if r.source_role != "external":
            raise RuntimeError("external assay rows must have source_role='external'")
        if r.prediction_sha256 != expected_prediction_sha256:
            raise RuntimeError("external result does not match the sealed prediction hash")
        previous = committed_predictions.setdefault(r.variant_id, float(r.predicted_effect))
        if previous != float(r.predicted_effect):
            raise RuntimeError(
                f"variant {r.variant_id!r} has different predictions across assays")
        previous_cohort = committed_cohorts.setdefault(r.variant_id, r.cohort_id)
        if previous_cohort != r.cohort_id:
            raise RuntimeError(
                f"variant {r.variant_id!r} changed cohort across assays")
    if seal_external_predictions(committed_predictions) != expected_prediction_sha256:
        raise RuntimeError(
            "external prediction values/IDs differ from the sealed pre-label table")
    observed_variant_sha, observed_cohort_sha = seal_variant_cohort_ids(
        committed_cohorts)
    if observed_variant_sha != expected_variant_set_sha256:
        raise RuntimeError("external variant set differs from the sealed locked set")
    if observed_cohort_sha != expected_variant_cohort_sha256:
        raise RuntimeError("external cohort assignment differs from the sealed locked set")

    resolvable = [r for r in rows
                  if abs(r.observed_effect) >= resolvable_z * r.observed_se
                  and abs(r.predicted_effect) > 1e-12]
    if not resolvable:
        raise RuntimeError("no assay-resolvable effects for signed validation")

    def boot(source: Sequence[ExternalAssayRow], value, offset: int):
        return cluster_bootstrap(
            [Row(float(value(r)), keys=r.dependency_keys,
                 stratum=r.assay_name, unit_id=r.variant_id) for r in source],
            n_boot=n_boot, seed=seed + offset, min_clusters=min_clusters,
        )

    sign = boot(resolvable,
                lambda r: float(np.sign(r.predicted_effect) ==
                                np.sign(r.observed_effect)), 0)
    error_gain = boot(
        rows,
        lambda r: abs(r.baseline_prediction - r.observed_effect) -
                  abs(r.predicted_effect - r.observed_effect),
        1,
    )
    specificity = boot(
        rows,
        lambda r: abs(r.observed_effect) - abs(r.matched_control_effect),
        2,
    )
    pred = np.asarray([r.predicted_effect for r in rows], dtype=float)
    obs = np.asarray([r.observed_effect for r in rows], dtype=float)
    correlation = (float(np.corrcoef(pred, obs)[0, 1])
                   if np.std(pred) > 0 and np.std(obs) > 0 else float("nan"))
    checks = {
        "signed_effect_generalizes": sign.lo >= sign_concordance_min,
        "beats_predeclared_baseline": error_gain.lo >= error_gain_min,
        "matched_controls_are_specific": specificity.lo >= specificity_gap_min,
    }
    return {
        "claim": ("blinded external assay prediction supported"
                  if all(checks.values()) else
                  "external assay prediction not established"),
        "scope": "external association; not model-mechanism causality by itself",
        "checks": checks,
        "sign_concordance": sign.as_row(),
        "absolute_error_gain": error_gain.as_row(),
        "matched_control_gap": specificity.as_row(),
        "pearson_descriptive": correlation,
        "n_assay_resolvable": len(resolvable),
        "prediction_sha256": expected_prediction_sha256,
        "variant_set_sha256": expected_variant_set_sha256,
        "variant_cohort_sha256": expected_variant_cohort_sha256,
    }


def external_causal_triangulation(
    external: Mapping[str, object], causal: Mapping[str, object], *,
    expected_causal_evidence_sha256: str,
    expected_prediction_sha256: str,
    expected_variant_set_sha256: str,
    expected_variant_cohort_sha256: str,
) -> dict:
    """Join independent external validity and in-model causal transport.

    Passing only one arm never upgrades the result to a biological mechanism.
    """
    if not _is_sha256(expected_causal_evidence_sha256):
        raise ValueError("expected_causal_evidence_sha256 is not a SHA-256 digest")
    if causal.get("causal_evidence_sha256") != expected_causal_evidence_sha256:
        raise RuntimeError("causal result does not match the committed evidence table")
    for label, expected in (
        ("prediction_sha256", expected_prediction_sha256),
        ("variant_set_sha256", expected_variant_set_sha256),
        ("variant_cohort_sha256", expected_variant_cohort_sha256),
    ):
        if not _is_sha256(expected):
            raise ValueError(f"expected {label} is not a SHA-256 digest")
        if external.get(label) != expected or causal.get(label) != expected:
            raise RuntimeError(
                f"external and internal evidence do not share {label}")
    external_pass = (
        external.get("claim") == "blinded external assay prediction supported" and
        all(bool(value) for value in dict(external.get("checks", {})).values())
    )
    causal_pass = (
        causal.get("claim") == "direction-mediated causal variant transport" and
        causal.get("provenance_valid") is True and
        all(bool(value) for value in dict(causal.get("checks", {})).values())
    )
    checks = {
        "external_assay": external_pass,
        "model_causal_path": causal_pass,
        "causal_evidence_committed": True,
    }
    return {
        "claim": ("external biological effect and model causal path triangulated"
                  if all(checks.values()) else "triangulation incomplete"),
        "checks": checks,
        "external_prediction_sha256": external.get("prediction_sha256"),
        "causal_evidence_sha256": expected_causal_evidence_sha256,
        "prediction_sha256": expected_prediction_sha256,
        "variant_set_sha256": expected_variant_set_sha256,
        "variant_cohort_sha256": expected_variant_cohort_sha256,
    }

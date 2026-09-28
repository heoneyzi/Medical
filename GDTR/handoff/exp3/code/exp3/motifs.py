"""EXP3 Phase 4 -- motif/variant causal replay and bilinear grammar.

The functions here turn a reverse-traced candidate into a falsifiable causal
claim.  They intentionally require separate evidence for writing at g28,
transport through b29/b30, output effect, block, rescue, specificity, and
natural support.  A motif logo, an enrichment p-value, or a large activation
alone cannot satisfy the conjunction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from .reverse_trace import _finite_tensor, _unit_axis


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _contrast_payload(value: "LockedContrast") -> dict[str, object]:
    return {
        "point": float(value.point), "lo": float(value.lo),
        "hi": float(value.hi), "margin": float(value.margin),
        "relation": value.relation,
    }


@dataclass(frozen=True)
class LockedContrast:
    """A pre-specified confidence interval and its scientific decision rule.

    ``relation='greater'`` means the entire interval must exceed ``margin``.
    ``relation='equivalent'`` means it must lie strictly inside
    ``[-margin,+margin]``.  A non-significant difference is therefore never
    silently converted into evidence of equivalence.
    """

    point: float
    lo: float
    hi: float
    margin: float
    relation: str = "greater"

    def __post_init__(self) -> None:
        vals = torch.tensor([self.point, self.lo, self.hi, self.margin], dtype=torch.float64)
        if not torch.isfinite(vals).all():
            raise ValueError("contrast values and margin must be finite")
        if self.lo > self.point or self.point > self.hi:
            raise ValueError("contrast must satisfy lo <= point <= hi")
        if self.margin < 0:
            raise ValueError("contrast margin must be non-negative")
        if self.relation not in {"greater", "equivalent"}:
            raise ValueError("relation must be 'greater' or 'equivalent'")

    @property
    def passed(self) -> bool:
        if self.relation == "greater":
            return bool(self.lo > self.margin)
        return bool(self.lo > -self.margin and self.hi < self.margin)


MOTIF_POSITIVE_CRITERIA = (
    "g28_direction_change",
    "b29_propagation",
    "m30_propagation",
    "output_effect",
    "path_block_loss",
    "heldout_predicted_delta_rescue_gain",
)

MOTIF_EQUIVALENCE_CRITERIA = ("rescue_to_native",)

VARIANT_POSITIVE_CRITERIA = (
    "g28_direction_change",
    "b29_propagation",
    "m30_propagation",
    "output_effect",
    "mechanism_prediction",
    "causal_rescue_gain",
)

VARIANT_EQUIVALENCE_CRITERIA = (
    "rescue_to_reference",
    "off_target_equivalence",
)


@dataclass(frozen=True)
class CausalCriteriaVerdict:
    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    claim: str
    note: str = "all required conditions are conjunctive; no missing condition is dropped"
    evidence_sha256: str = ""
    sealed_payload_json: str = field(default="", repr=False)
    upstream_evidence_sha256: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def verify_seal(self, *, expected_sha256: str | None = None) -> None:
        """Reject mutation or substitution of a causal verdict."""
        if not _SHA256.fullmatch(self.evidence_sha256):
            raise RuntimeError("causal verdict has no valid evidence seal")
        if expected_sha256 is not None:
            if not _SHA256.fullmatch(expected_sha256):
                raise ValueError("expected verdict digest is not a SHA-256")
            if self.evidence_sha256 != expected_sha256:
                raise RuntimeError("causal verdict differs from the expected sealed verdict")
        observed = hashlib.sha256(self.sealed_payload_json.encode("utf-8")).hexdigest()
        if observed != self.evidence_sha256:
            raise RuntimeError("causal verdict payload changed after sealing")
        try:
            payload = json.loads(self.sealed_payload_json)
        except json.JSONDecodeError as error:
            raise RuntimeError("causal verdict seal does not contain canonical JSON") from error
        result = payload.get("result", {})
        expected_result = {
            "passed": bool(self.passed),
            "checks": {str(k): bool(v) for k, v in sorted(self.checks.items())},
            "failures": list(self.failures),
            "claim": self.claim,
        }
        if result != expected_result:
            raise RuntimeError("causal verdict result changed after sealing")
        if payload.get("upstream_evidence_sha256", {}) != dict(
                sorted(self.upstream_evidence_sha256.items())):
            raise RuntimeError("causal verdict upstream evidence changed after sealing")
        if payload.get("provenance", {}) != dict(self.provenance):
            raise RuntimeError("causal verdict provenance changed after sealing")


def _criteria_verdict(
    *,
    positive: Mapping[str, LockedContrast],
    equivalence: Mapping[str, LockedContrast],
    specificity: Mapping[str, LockedContrast],
    required_positive: Sequence[str],
    required_equivalence: Sequence[str],
    booleans: Mapping[str, bool],
    min_specificity_families: int,
    success_claim: str,
    failure_claim: str,
    upstream_evidence_sha256: Mapping[str, str] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> CausalCriteriaVerdict:
    missing_p = sorted(set(required_positive) - set(positive))
    extra_p = sorted(set(positive) - set(required_positive))
    missing_e = sorted(set(required_equivalence) - set(equivalence))
    extra_e = sorted(set(equivalence) - set(required_equivalence))
    if missing_p or extra_p or missing_e or extra_e:
        raise ValueError(
            "causal criteria do not match the frozen schema; "
            f"missing_positive={missing_p}, extra_positive={extra_p}, "
            f"missing_equivalence={missing_e}, extra_equivalence={extra_e}"
        )
    if len(specificity) < min_specificity_families:
        raise ValueError(
            f"need at least {min_specificity_families} pre-registered specificity "
            f"families, got {len(specificity)}"
        )
    wrong_p = [name for name, c in positive.items() if c.relation != "greater"]
    wrong_e = [name for name, c in equivalence.items() if c.relation != "equivalent"]
    wrong_s = [name for name, c in specificity.items() if c.relation != "greater"]
    if wrong_p or wrong_e or wrong_s:
        raise ValueError(
            f"wrong contrast relation: positive={wrong_p}, equivalence={wrong_e}, "
            f"specificity={wrong_s}"
        )

    checks: dict[str, bool] = {}
    checks.update({name: contrast.passed for name, contrast in positive.items()})
    checks.update({name: contrast.passed for name, contrast in equivalence.items()})
    checks.update({f"specificity::{name}": contrast.passed
                   for name, contrast in specificity.items()})
    checks.update({name: bool(value) for name, value in booleans.items()})
    failures = tuple(name for name, passed in checks.items() if not passed)
    passed = not failures
    claim = success_claim if passed else failure_claim
    upstream = dict(sorted((upstream_evidence_sha256 or {}).items()))
    for name, digest in upstream.items():
        if not name or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid upstream evidence digest {name!r}")
    provenance_payload = dict(provenance or {})
    payload = {
        "schema_version": 1,
        "positive": {name: _contrast_payload(value)
                     for name, value in sorted(positive.items())},
        "equivalence": {name: _contrast_payload(value)
                        for name, value in sorted(equivalence.items())},
        "specificity": {name: _contrast_payload(value)
                        for name, value in sorted(specificity.items())},
        "required_positive": list(required_positive),
        "required_equivalence": list(required_equivalence),
        "booleans": {str(k): bool(v) for k, v in sorted(booleans.items())},
        "minimum_specificity_families": int(min_specificity_families),
        "upstream_evidence_sha256": upstream,
        "provenance": provenance_payload,
        "result": {
            "passed": passed,
            "checks": {str(k): bool(v) for k, v in sorted(checks.items())},
            "failures": list(failures),
            "claim": claim,
        },
    }
    sealed_payload = _canonical_json(payload)
    digest = hashlib.sha256(sealed_payload.encode("utf-8")).hexdigest()
    return CausalCriteriaVerdict(
        passed=passed,
        checks=checks,
        failures=failures,
        claim=claim,
        evidence_sha256=digest,
        sealed_payload_json=sealed_payload,
        upstream_evidence_sha256=upstream,
        provenance=provenance_payload,
    )


def _require_sealed_serial_path(
    verdict: CausalCriteriaVerdict | None,
    *,
    expected_sha256: str | None,
) -> tuple[str, bool]:
    if not isinstance(verdict, CausalCriteriaVerdict):
        raise ValueError(
            "a sealed ordered_serial_path_verdict is required for a serial claim")
    verdict.verify_seal(expected_sha256=expected_sha256)
    if verdict.provenance.get("verdict_kind") != "ordered_g28_b29_m30":
        raise RuntimeError("upstream verdict is not an ordered g28->b29->m30 path verdict")
    # A well-formed negative serial-path result is scientific non-support, not
    # a software error.  It enters the downstream conjunction as ``False`` so
    # the strong claim fails cleanly and remains auditable.
    return verdict.evidence_sha256, verdict.passed


def motif_causal_verdict(
    *,
    positive: Mapping[str, LockedContrast],
    equivalence: Mapping[str, LockedContrast],
    specificity: Mapping[str, LockedContrast],
    candidate_consensus: bool,
    direction_frozen_before_test: bool,
    natural_support: bool,
    predicted_rescue_held_out: bool,
    predicted_rescue_target_derived: bool,
    edit_modes_tested: Sequence[str],
    ordered_serial_path: CausalCriteriaVerdict | None = None,
    expected_ordered_serial_path_sha256: str | None = None,
    required_edit_modes: Sequence[str] = ("insertion", "deletion"),
    min_specificity_families: int = 3,
) -> CausalCriteriaVerdict:
    """Conjunctive definition of a motif using the serial late-stack circuit."""

    modes = {str(x) for x in edit_modes_tested}
    needed = {str(x) for x in required_edit_modes}
    serial_sha, serial_passed = _require_sealed_serial_path(
        ordered_serial_path, expected_sha256=expected_ordered_serial_path_sha256)
    return _criteria_verdict(
        positive=positive,
        equivalence=equivalence,
        specificity=specificity,
        required_positive=MOTIF_POSITIVE_CRITERIA,
        required_equivalence=MOTIF_EQUIVALENCE_CRITERIA,
        booleans={
            "candidate_method_consensus": candidate_consensus,
            "direction_frozen_before_test": direction_frozen_before_test,
            "natural_support": natural_support,
            "predicted_rescue_held_out": predicted_rescue_held_out,
            "predicted_rescue_target_free": not predicted_rescue_target_derived,
            "required_edit_modes": needed.issubset(modes),
            "ordered_immediate_serial_path": serial_passed,
        },
        min_specificity_families=min_specificity_families,
        success_claim="serial-circuit causal motif",
        failure_claim="candidate motif; serial causal claim not established",
        upstream_evidence_sha256={"ordered_g28_b29_m30": serial_sha},
        provenance={"verdict_kind": "motif_causal_verdict"},
    )


def variant_causal_verdict(
    *,
    positive: Mapping[str, LockedContrast],
    equivalence: Mapping[str, LockedContrast],
    specificity: Mapping[str, LockedContrast],
    direction_frozen_before_test: bool,
    held_out_variant: bool,
    natural_support: bool,
    predicted_rescue_held_out: bool,
    predicted_rescue_target_derived: bool,
    ordered_serial_path: CausalCriteriaVerdict | None = None,
    expected_ordered_serial_path_sha256: str | None = None,
    min_specificity_families: int = 3,
) -> CausalCriteriaVerdict:
    """Require prediction and selective rescue for a mechanism-level variant claim."""

    serial_sha, serial_passed = _require_sealed_serial_path(
        ordered_serial_path, expected_sha256=expected_ordered_serial_path_sha256)
    return _criteria_verdict(
        positive=positive,
        equivalence=equivalence,
        specificity=specificity,
        required_positive=VARIANT_POSITIVE_CRITERIA,
        required_equivalence=VARIANT_EQUIVALENCE_CRITERIA,
        booleans={
            "direction_frozen_before_test": direction_frozen_before_test,
            "held_out_variant": held_out_variant,
            "natural_support": natural_support,
            "predicted_rescue_held_out": predicted_rescue_held_out,
            "predicted_rescue_target_free": not predicted_rescue_target_derived,
            "ordered_immediate_serial_path": serial_passed,
        },
        min_specificity_families=min_specificity_families,
        success_claim="held-out variant explained and selectively repaired by the serial circuit",
        failure_claim="variant association; mechanism-level repair claim not established",
        upstream_evidence_sha256={"ordered_g28_b29_m30": serial_sha},
        provenance={"verdict_kind": "variant_causal_verdict"},
    )


@dataclass(frozen=True)
class BilinearCellFactors:
    factor_a: Tensor
    factor_b: Tensor


@dataclass(frozen=True)
class BilinearEpistasis:
    product_interaction: Tensor
    channel_interactions: Tensor
    total_interaction: Tensor
    reconstructed_interaction: Tensor
    relative_residual: float
    factor_a_interaction: Tensor
    factor_b_interaction: Tensor
    cell_projections: Mapping[str, Tensor]
    target_direction: Tensor


BILINEAR_CELLS = ("WT", "A", "B", "AB")


def exact_bilinear_epistasis(
    cells: Mapping[str, BilinearCellFactors],
    W3: Tensor,
    target_direction: Tensor,
    *,
    carrier_axis: Optional[Tensor] = None,
    normalize_direction: bool = True,
) -> BilinearEpistasis:
    """Compute exact 2x2 interaction in the b28 bilinear product and output.

    This distinguishes two sequence elements that merely add from a pair that
    jointly changes the native ``(W1 z)*(W2 z)`` code.  It is exact at g28;
    downstream motif interaction still needs the held-out path intervention.
    """

    missing = sorted(set(BILINEAR_CELLS) - set(cells))
    extra = sorted(set(cells) - set(BILINEAR_CELLS))
    if missing or extra:
        raise ValueError(f"bilinear cells incomplete; missing={missing}, extra={extra}")
    C = _finite_tensor("W3", W3).double()
    if C.ndim != 2:
        raise ValueError("W3 must be [d_out, hidden]")
    q = _finite_tensor("target_direction", target_direction).double().reshape(-1)
    if q.numel() != C.shape[0]:
        raise ValueError(f"target direction has width {q.numel()}, expected {C.shape[0]}")
    if carrier_axis is not None:
        u = _unit_axis(carrier_axis, width=q.numel(), name="carrier_axis").to(q.device)
        q = q - torch.dot(q, u) * u
    if normalize_direction:
        norm = q.norm()
        if float(norm) <= 0:
            raise ValueError("target direction is zero after carrier removal")
        q = q / norm

    first = cells["WT"]
    a0 = _finite_tensor("cells['WT'].factor_a", first.factor_a).double()
    b0 = _finite_tensor("cells['WT'].factor_b", first.factor_b).double()
    if a0.shape != b0.shape or a0.shape[-1] != C.shape[1]:
        raise ValueError("WT factor shapes are incompatible with W3")
    a: dict[str, Tensor] = {}
    b: dict[str, Tensor] = {}
    p: dict[str, Tensor] = {}
    for name in BILINEAR_CELLS:
        ai = _finite_tensor(f"cells[{name!r}].factor_a", cells[name].factor_a).double()
        bi = _finite_tensor(f"cells[{name!r}].factor_b", cells[name].factor_b).double()
        if ai.shape != a0.shape or bi.shape != b0.shape:
            raise ValueError(f"cell {name} factor shapes differ from WT")
        ai = ai.to(a0.device)
        bi = bi.to(a0.device)
        a[name], b[name], p[name] = ai, bi, ai * bi

    C = C.to(a0.device)
    q = q.to(a0.device)
    delta_p = p["AB"] - p["A"] - p["B"] + p["WT"]
    readout = C.T @ q
    channels = delta_p * readout
    reconstructed = channels.sum(dim=-1)
    direct = (delta_p @ C.T) @ q
    residual = (reconstructed - direct).norm() / direct.norm().clamp_min(1e-30)
    cell_proj = {name: (p[name] @ C.T) @ q for name in BILINEAR_CELLS}
    return BilinearEpistasis(
        product_interaction=delta_p,
        channel_interactions=channels,
        total_interaction=direct,
        reconstructed_interaction=reconstructed,
        relative_residual=float(residual),
        factor_a_interaction=a["AB"] - a["A"] - a["B"] + a["WT"],
        factor_b_interaction=b["AB"] - b["A"] - b["B"] + b["WT"],
        cell_projections=cell_proj,
        target_direction=q,
    )


@dataclass(frozen=True)
class OrderedBypassVerdict:
    passed: bool
    checks: Mapping[str, bool]
    ratios: Mapping[str, float]
    failures: tuple[str, ...]
    claim: str


def ordered_block_rescue_bypass_verdict(
    *,
    native_effect: float,
    b28_block_effect: float,
    b28_block_plus_m30_delta_rescue: float,
    b30_block_effect: float,
    b30_block_plus_g28_rescue: float,
    max_block_fraction: float,
    min_rescue_fraction: float,
    max_wrong_order_fraction: float,
) -> OrderedBypassVerdict:
    """Diagnose only the coarse ordering ``g28`` before ``m30``.

    A downstream ``delta m30`` may bypass an upstream b28 block.  An upstream
    g28 rescue must *not* bypass a downstream m30 block.  This asymmetry is
    stronger than showing that both blocks matter separately, but there is no
    b29 intervention in this scalar panel.  It therefore cannot establish the
    full ``g28 -> b29 -> m30`` chain.  Use :func:`ordered_serial_path_verdict`
    for that claim.  All effects must be signed toward one pre-specified native
    endpoint.
    """

    vals = torch.tensor([
        native_effect, b28_block_effect, b28_block_plus_m30_delta_rescue,
        b30_block_effect, b30_block_plus_g28_rescue, max_block_fraction,
        min_rescue_fraction, max_wrong_order_fraction,
    ], dtype=torch.float64)
    if not torch.isfinite(vals).all():
        raise ValueError("ordered bypass inputs must be finite")
    if native_effect <= 0:
        raise ValueError("native_effect must be positive in the pre-specified direction")
    for name, value in (
        ("max_block_fraction", max_block_fraction),
        ("min_rescue_fraction", min_rescue_fraction),
        ("max_wrong_order_fraction", max_wrong_order_fraction),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0,1]")

    ratios = {
        "b28_block": b28_block_effect / native_effect,
        "downstream_rescue": b28_block_plus_m30_delta_rescue / native_effect,
        "b30_block": b30_block_effect / native_effect,
        "wrong_order_rescue": b30_block_plus_g28_rescue / native_effect,
    }
    checks = {
        "b28_block_loses_effect": ratios["b28_block"] <= max_block_fraction,
        "m30_delta_bypasses_b28_block": ratios["downstream_rescue"] >= min_rescue_fraction,
        "b30_block_loses_effect": ratios["b30_block"] <= max_block_fraction,
        "g28_cannot_bypass_b30_block": ratios["wrong_order_rescue"] <= max_wrong_order_fraction,
    }
    failures = tuple(k for k, v in checks.items() if not v)
    return OrderedBypassVerdict(
        passed=not failures,
        checks=checks,
        ratios=ratios,
        failures=failures,
        claim=("g28-before-m30 bypass asymmetry diagnostic" if not failures else
               "coarse ordering not established; retain only individual necessity results"),
    )


SERIAL_PATH_POSITIVE_CRITERIA = (
    "b28_block_loss",
    "b29_rescue_after_b28_block",
    "b29_block_loss",
    "m30_rescue_after_b29_block",
    "b30_block_loss",
)

SERIAL_PATH_EQUIVALENCE_CRITERIA = (
    "g28_rescue_under_b29_block",
    "b29_rescue_under_b30_block",
)


def ordered_serial_path_verdict(
    *,
    positive: Mapping[str, LockedContrast],
    equivalence: Mapping[str, LockedContrast],
    direction_frozen_before_test: bool,
    b29_rescue_held_out: bool,
    b29_rescue_target_derived: bool,
    m30_rescue_held_out: bool,
    m30_rescue_target_derived: bool,
    evidence_artifact_sha256: str,
    direction_artifact_sha256: str,
    source_split: str,
    source_role: str,
    source_unit_ids: Sequence[str],
    dependency_keys: Sequence[str],
) -> CausalCriteriaVerdict:
    """Establish ``g28 -> b29 -> m30`` with intervened intermediate links.

    Each downstream stage must rescue the immediately upstream block, whereas
    an upstream rescue must remain ineffective under the next downstream
    block.  Confidence intervals, rather than scalar point estimates, decide
    all positive and equivalence relations.  The b29 and m30 rescue deltas must
    be held-out predictions made without the target response.
    """
    if not _SHA256.fullmatch(evidence_artifact_sha256):
        raise ValueError("evidence_artifact_sha256 is not a full lowercase SHA-256")
    if not _SHA256.fullmatch(direction_artifact_sha256):
        raise ValueError("direction_artifact_sha256 is not a full lowercase SHA-256")
    units = tuple(str(value) for value in source_unit_ids)
    dependencies = tuple(str(value) for value in dependency_keys)
    if (not source_split.strip() or source_role != "locked" or not units
            or not dependencies or any(not value for value in units + dependencies)):
        raise ValueError(
            "serial-path provenance needs a locked split, unit ids, and dependency keys")
    if len(set(units)) != len(units):
        raise ValueError("serial-path source_unit_ids must be unique")
    return _criteria_verdict(
        positive=positive,
        equivalence=equivalence,
        specificity={},
        required_positive=SERIAL_PATH_POSITIVE_CRITERIA,
        required_equivalence=SERIAL_PATH_EQUIVALENCE_CRITERIA,
        booleans={
            "direction_frozen_before_test": bool(direction_frozen_before_test),
            "b29_rescue_held_out": bool(b29_rescue_held_out),
            "b29_rescue_target_free": not bool(b29_rescue_target_derived),
            "m30_rescue_held_out": bool(m30_rescue_held_out),
            "m30_rescue_target_free": not bool(m30_rescue_target_derived),
        },
        min_specificity_families=0,
        success_claim="ordered g28->b29->m30 serial path",
        failure_claim="full serial path not established",
        provenance={
            "verdict_kind": "ordered_g28_b29_m30",
            "evidence_artifact_sha256": evidence_artifact_sha256,
            "direction_artifact_sha256": direction_artifact_sha256,
            "source_split": source_split,
            "source_role": source_role,
            "source_unit_ids": list(units),
            "dependency_keys": list(dependencies),
        },
    )


@dataclass(frozen=True)
class ContextWindowSaturation:
    saturated: bool
    minimal_width: Optional[int]
    full_width: int
    full_effect: float
    ratios: Mapping[int, float]
    plateau_deviation: Mapping[int, float]
    note: str = "widths and thresholds must be frozen before locked evaluation"


def context_window_saturation(
    effects_by_width: Mapping[int, float],
    *,
    target_fraction: float,
    plateau_tolerance_fraction: float,
) -> ContextWindowSaturation:
    """Find the smallest context whose effect is stably equivalent to full context."""

    if len(effects_by_width) < 2:
        raise ValueError("context saturation needs at least two window widths")
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must lie in (0,1]")
    if not 0 <= plateau_tolerance_fraction < 1:
        raise ValueError("plateau_tolerance_fraction must lie in [0,1)")
    widths = sorted(int(w) for w in effects_by_width)
    if len(widths) != len(set(widths)) or widths[0] <= 0:
        raise ValueError("context widths must be unique positive integers")
    effects = {int(w): float(effects_by_width[w]) for w in effects_by_width}
    if not torch.isfinite(torch.tensor(list(effects.values()), dtype=torch.float64)).all():
        raise ValueError("context effects must be finite")
    full_width = widths[-1]
    full = effects[full_width]
    if full == 0:
        raise ValueError("full-context effect is zero; a saturation fraction is undefined")
    ratios = {w: effects[w] / full for w in widths}
    deviations = {w: abs(effects[w] - full) / abs(full) for w in widths}
    selected: Optional[int] = None
    for i, width in enumerate(widths):
        same_sign_and_large = ratios[width] >= target_fraction
        persistent = all(deviations[w] <= plateau_tolerance_fraction for w in widths[i:])
        if same_sign_and_large and persistent:
            selected = width
            break
    return ContextWindowSaturation(
        saturated=selected is not None,
        minimal_width=selected,
        full_width=full_width,
        full_effect=full,
        ratios=ratios,
        plateau_deviation=deviations,
    )


@dataclass(frozen=True)
class ConfoundRecord:
    match_id: str
    arm: str                       # target | control
    target_token: str
    kmer_log_likelihood: float
    gc_fraction: float
    repeat_class: str
    relative_position: float
    orientation: str
    token_phase: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.match_id:
            raise ValueError("confound record match_id is empty")
        if self.arm not in {"target", "control"}:
            raise ValueError("confound record arm must be 'target' or 'control'")
        continuous = torch.tensor(
            [self.kmer_log_likelihood, self.gc_fraction, self.relative_position],
            dtype=torch.float64,
        )
        if not torch.isfinite(continuous).all():
            raise ValueError("confound record contains non-finite continuous values")
        if not 0 <= self.gc_fraction <= 1:
            raise ValueError("gc_fraction must lie in [0,1]")
        if self.token_phase < 0:
            raise ValueError("token_phase must be non-negative")


@dataclass(frozen=True)
class ConfoundMatchReport:
    passed: bool
    n_matches: int
    n_controls: int
    max_absolute_difference: Mapping[str, float]
    failures: tuple[str, ...]
    exact_fields: tuple[str, ...]
    tolerances: Mapping[str, float]


def validate_confound_matches(
    records: Sequence[ConfoundRecord],
    *,
    continuous_tolerances: Mapping[str, float],
    exact_fields: Sequence[str] = (
        "target_token", "repeat_class", "orientation", "token_phase"
    ),
    strict: bool = True,
) -> ConfoundMatchReport:
    """Validate matched motif controls before any causal endpoint is opened.

    The mandatory continuous fields prevent the common failure mode of
    rediscovering next-token identity, local sequence likelihood, GC content,
    or position instead of a late-stack motif.  Each match has exactly one
    target and one or more controls.
    """

    required_continuous = {"kmer_log_likelihood", "gc_fraction", "relative_position"}
    missing = sorted(required_continuous - set(continuous_tolerances))
    extra = sorted(set(continuous_tolerances) - required_continuous)
    if missing or extra:
        raise ValueError(
            f"continuous tolerance schema mismatch; missing={missing}, extra={extra}"
        )
    tolerances = {name: float(continuous_tolerances[name]) for name in required_continuous}
    if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in tolerances.values()):
        raise ValueError("continuous tolerances must be finite and non-negative")
    exact = tuple(str(x) for x in exact_fields)
    allowed_exact = {"target_token", "repeat_class", "orientation", "token_phase"}
    if set(exact) != allowed_exact:
        raise ValueError(
            "exact fields must include target_token, repeat_class, orientation, and token_phase"
        )
    if not records:
        raise ValueError("no confound matches supplied")
    groups: dict[str, list[ConfoundRecord]] = {}
    for record in records:
        record.validate()
        groups.setdefault(record.match_id, []).append(record)

    failures: list[str] = []
    maxima = {name: 0.0 for name in required_continuous}
    n_controls = 0
    for match_id, group in sorted(groups.items()):
        targets = [r for r in group if r.arm == "target"]
        controls = [r for r in group if r.arm == "control"]
        if len(targets) != 1 or not controls:
            failures.append(
                f"{match_id}: expected exactly one target and >=1 control, "
                f"got {len(targets)} target/{len(controls)} control"
            )
            continue
        target = targets[0]
        n_controls += len(controls)
        for ci, control in enumerate(controls):
            for name in exact:
                if getattr(target, name) != getattr(control, name):
                    failures.append(
                        f"{match_id}/control{ci}: exact confound {name} differs "
                        f"({getattr(target, name)!r} vs {getattr(control, name)!r})"
                    )
            for name, tolerance in tolerances.items():
                difference = abs(float(getattr(target, name)) - float(getattr(control, name)))
                maxima[name] = max(maxima[name], difference)
                if difference > tolerance:
                    failures.append(
                        f"{match_id}/control{ci}: {name} difference {difference:.6g} "
                        f"> tolerance {tolerance:.6g}"
                    )
    report = ConfoundMatchReport(
        passed=not failures,
        n_matches=len(groups),
        n_controls=n_controls,
        max_absolute_difference=maxima,
        failures=tuple(failures),
        exact_fields=exact,
        tolerances=tolerances,
    )
    if strict and failures:
        preview = "; ".join(failures[:5])
        raise ValueError(f"confound matching failed: {preview}")
    return report

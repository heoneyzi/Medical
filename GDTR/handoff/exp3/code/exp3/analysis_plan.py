"""Hash-sealed confirmatory analysis contracts for EXP3 Phases 3--8.

The adaptive router decides *which* downstream analysis becomes relevant, but
adaptivity must not also permit thresholds, hyperparameters, controls, or the
decision rule to be chosen after the locked split is inspected.  This module
therefore seals one declaration for every possible Phase 3--8 node before a
scientific run starts.  Unselected declarations remain harmless, documented
alternatives in the plan.

The seal is an integrity and run-binding mechanism.  ``sealed_before_locked``
is an investigator attestation, not a trusted timestamp; projects requiring
independent proof should archive the resulting plan hash in a timestamped
registry before opening the locked data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .adaptive import AdaptiveRouter


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

CONFIRMATORY_PHASES = frozenset({
    "phase3_reverse_trace",
    "phase4_motif_validation",
    "phase5_learned_transport",
    "phase6_applications",
    "phase7_generalization",
    "phase8_optional_features",
})


def required_analysis_experiment_ids(router: "AdaptiveRouter") -> tuple[str, ...]:
    """List every possible Phase 3--8 node in a research graph."""
    return tuple(sorted(
        experiment_id
        for experiment_id, record in router.records.items()
        if record.spec.phase in CONFIRMATORY_PHASES
    ))


def _normalise_json(value: Any, *, path: str) -> Any:
    """Return an isolated JSON value and reject lossy/non-finite content."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"{path} keys must be nonblank strings")
            out[key] = _normalise_json(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class AnalysisDeclaration:
    """Pre-locked contract for one possible confirmatory experiment.

    Empty threshold or hyperparameter mappings are allowed only when the
    corresponding explanation is explicit.  This supports genuinely
    parameter-free analyses without turning an omitted analysis choice into a
    silent degree of freedom.
    """

    experiment_id: str
    estimand: str
    decision_rule: str
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    control_families: tuple[str, ...] = ()
    control_plan: str = ""
    data_roles: Mapping[str, str] = field(default_factory=dict)
    input_artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    no_thresholds_reason: str = ""
    no_hyperparameters_reason: str = ""
    declaration_sha256: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.experiment_id):
            raise ValueError(f"invalid experiment_id {self.experiment_id!r}")
        for name, value in (
            ("estimand", self.estimand),
            ("decision_rule", self.decision_rule),
            ("control_plan", self.control_plan),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        for name, value in (
            ("no_thresholds_reason", self.no_thresholds_reason),
            ("no_hyperparameters_reason", self.no_hyperparameters_reason),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")

        thresholds = _normalise_json(self.thresholds, path="thresholds")
        hyperparameters = _normalise_json(
            self.hyperparameters, path="hyperparameters")
        data_roles = _normalise_json(self.data_roles, path="data_roles")
        artifacts = _normalise_json(
            self.input_artifact_sha256, path="input_artifact_sha256")
        if not isinstance(thresholds, dict) or not isinstance(hyperparameters, dict):
            raise TypeError("thresholds and hyperparameters must be mappings")
        if not isinstance(data_roles, dict) or not data_roles:
            raise ValueError("data_roles must explicitly bind at least one split role")
        if any(not isinstance(value, str) or not value.strip()
               for value in data_roles.values()):
            raise ValueError("data_roles values must be nonblank strings")
        if not isinstance(artifacts, dict):
            raise TypeError("input_artifact_sha256 must be a mapping")
        if any(not isinstance(value, str) or not _HEX64.fullmatch(value)
               for value in artifacts.values()):
            raise ValueError("input artifact commitments must be lowercase SHA-256")

        if isinstance(self.control_families, (str, bytes)):
            raise TypeError("control_families must be a sequence of names")
        controls = tuple(self.control_families)
        if (len(set(controls)) != len(controls)
                or any(not isinstance(value, str) or not value.strip()
                       for value in controls)):
            raise ValueError("control_families must contain unique nonblank names")
        if not thresholds and not self.no_thresholds_reason.strip():
            raise ValueError(
                "an empty threshold map requires no_thresholds_reason")
        if thresholds and self.no_thresholds_reason.strip():
            raise ValueError(
                "no_thresholds_reason must be empty when thresholds are declared")
        if not hyperparameters and not self.no_hyperparameters_reason.strip():
            raise ValueError(
                "an empty hyperparameter map requires no_hyperparameters_reason")
        if hyperparameters and self.no_hyperparameters_reason.strip():
            raise ValueError(
                "no_hyperparameters_reason must be empty when hyperparameters are declared")

        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "hyperparameters", hyperparameters)
        object.__setattr__(self, "control_families", controls)
        object.__setattr__(self, "data_roles", data_roles)
        object.__setattr__(self, "input_artifact_sha256", artifacts)
        expected = self.compute_hash()
        if self.declaration_sha256 and self.declaration_sha256 != expected:
            raise RuntimeError(
                f"analysis declaration {self.experiment_id!r} was modified")
        object.__setattr__(self, "declaration_sha256", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "estimand": self.estimand,
            "decision_rule": self.decision_rule,
            "thresholds": dict(self.thresholds),
            "hyperparameters": dict(self.hyperparameters),
            "control_families": list(self.control_families),
            "control_plan": self.control_plan,
            "data_roles": dict(self.data_roles),
            "input_artifact_sha256": dict(self.input_artifact_sha256),
            "no_thresholds_reason": self.no_thresholds_reason,
            "no_hyperparameters_reason": self.no_hyperparameters_reason,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self._payload())).hexdigest()

    def validate(self) -> None:
        if self.compute_hash() != self.declaration_sha256:
            raise RuntimeError(
                f"analysis declaration {self.experiment_id!r} changed after sealing")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._payload(), "declaration_sha256": self.declaration_sha256}

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "AnalysisDeclaration":
        return AnalysisDeclaration(
            experiment_id=raw["experiment_id"],
            estimand=raw["estimand"],
            decision_rule=raw["decision_rule"],
            thresholds=raw.get("thresholds", {}),
            hyperparameters=raw.get("hyperparameters", {}),
            control_families=raw.get("control_families", ()),
            control_plan=raw["control_plan"],
            data_roles=raw.get("data_roles", {}),
            input_artifact_sha256=raw.get("input_artifact_sha256", {}),
            no_thresholds_reason=raw.get("no_thresholds_reason", ""),
            no_hyperparameters_reason=raw.get(
                "no_hyperparameters_reason", ""),
            declaration_sha256=raw.get("declaration_sha256", ""),
        )


@dataclass(frozen=True)
class ConfirmatoryAnalysisPlan:
    """Complete, sealed set of possible Phase 3--8 analysis declarations."""

    plan_id: str
    config_sha256: str
    locked_split: str
    declarations: tuple[AnalysisDeclaration, ...]
    sealed_before_locked_data: bool = True
    sealed_at_utc: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat())
    registration_reference: str = ""
    schema_version: int = 1
    plan_sha256: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.plan_id):
            raise ValueError("plan_id must be a lowercase slug")
        if not _HEX64.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256")
        if not isinstance(self.locked_split, str) or not self.locked_split.strip():
            raise ValueError("locked_split must be nonblank")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported confirmatory analysis plan schema")
        if self.sealed_before_locked_data is not True:
            raise ValueError(
                "confirmatory plan must be sealed before locked data are opened")
        if not isinstance(self.sealed_at_utc, str):
            raise TypeError("sealed_at_utc must be a string")
        if not isinstance(self.registration_reference, str):
            raise TypeError("registration_reference must be a string")
        try:
            sealed_at = datetime.fromisoformat(self.sealed_at_utc)
        except ValueError as error:
            raise ValueError("sealed_at_utc must be ISO-8601") from error
        if sealed_at.tzinfo is None:
            raise ValueError("sealed_at_utc must include a timezone")
        declarations = tuple(self.declarations)
        if not declarations:
            raise ValueError("confirmatory plan needs at least one declaration")
        if any(not isinstance(item, AnalysisDeclaration)
               for item in declarations):
            raise TypeError("declarations must contain AnalysisDeclaration values")
        ids = [item.experiment_id for item in declarations]
        if len(ids) != len(set(ids)):
            raise ValueError("confirmatory plan contains duplicate experiment ids")
        object.__setattr__(self, "declarations", declarations)
        expected = self.compute_hash()
        if self.plan_sha256 and self.plan_sha256 != expected:
            raise RuntimeError("confirmatory analysis plan was modified")
        object.__setattr__(self, "plan_sha256", expected)

    @property
    def by_experiment(self) -> dict[str, AnalysisDeclaration]:
        return {item.experiment_id: item for item in self.declarations}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "config_sha256": self.config_sha256,
            "locked_split": self.locked_split,
            "sealed_before_locked_data": self.sealed_before_locked_data,
            "sealed_at_utc": self.sealed_at_utc,
            "registration_reference": self.registration_reference,
            "declarations": [
                item.to_dict()
                for item in sorted(
                    self.declarations, key=lambda value: value.experiment_id)
            ],
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self._payload())).hexdigest()

    def validate(
        self,
        *,
        required_experiment_ids: Iterable[str] | None = None,
        expected_config_sha256: str | None = None,
        expected_locked_split: str | None = None,
    ) -> None:
        for declaration in self.declarations:
            declaration.validate()
        if self.compute_hash() != self.plan_sha256:
            raise RuntimeError("confirmatory analysis plan changed after sealing")
        if (expected_config_sha256 is not None
                and self.config_sha256 != expected_config_sha256):
            raise RuntimeError("analysis plan does not match the sealed EXP3 config")
        if (expected_locked_split is not None
                and self.locked_split != expected_locked_split):
            raise RuntimeError("analysis plan does not match the locked split")
        if required_experiment_ids is not None:
            required = set(required_experiment_ids)
            observed = set(self.by_experiment)
            missing = sorted(required - observed)
            extra = sorted(observed - required)
            if missing or extra:
                raise RuntimeError(
                    "analysis plan must declare exactly every possible Phase 3--8 "
                    f"node; missing={missing}, extra={extra}")

    def attestation(self, experiment_id: str) -> dict[str, Any]:
        """Exact values a provider must report as actually used."""
        self.validate()
        try:
            declaration = self.by_experiment[experiment_id]
        except KeyError as error:
            raise KeyError(
                f"no confirmatory declaration for {experiment_id!r}") from error
        return {
            "plan_sha256": self.plan_sha256,
            "declaration_sha256": declaration.declaration_sha256,
            "estimand_used": declaration.estimand,
            "decision_rule_used": declaration.decision_rule,
            "thresholds_used": _normalise_json(
                declaration.thresholds, path="thresholds"),
            "hyperparameters_used": _normalise_json(
                declaration.hyperparameters, path="hyperparameters"),
            "control_families_used": list(declaration.control_families),
            "control_plan_used": declaration.control_plan,
            "data_roles_used": _normalise_json(
                declaration.data_roles, path="data_roles"),
            "input_artifact_sha256_used": _normalise_json(
                declaration.input_artifact_sha256,
                path="input_artifact_sha256"),
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {**self._payload(), "plan_sha256": self.plan_sha256}
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @staticmethod
    def load(path: str | Path) -> "ConfirmatoryAnalysisPlan":
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        return ConfirmatoryAnalysisPlan(
            plan_id=raw["plan_id"],
            config_sha256=raw["config_sha256"],
            locked_split=raw["locked_split"],
            declarations=tuple(
                AnalysisDeclaration.from_dict(value)
                for value in raw["declarations"]
            ),
            sealed_before_locked_data=raw["sealed_before_locked_data"],
            sealed_at_utc=raw["sealed_at_utc"],
            registration_reference=raw.get("registration_reference", ""),
            schema_version=raw["schema_version"],
            plan_sha256=raw.get("plan_sha256", ""),
        )


__all__ = [
    "AnalysisDeclaration", "CONFIRMATORY_PHASES", "ConfirmatoryAnalysisPlan",
    "required_analysis_experiment_ids",
]

"""Auditable, outcome-adaptive orchestration for the standalone EXP3 program.

EXP2 is not executed here.  Its accepted conclusions and selected artifacts
enter only through a hash-sealed :class:`~exp3.inputs.Exp2Handoff`.  Scientific
non-support in a new EXP3 question is recorded as evidence and activates the
corresponding alternative branch.  Invalid measurement/provenance/leakage is
the only scientific hard failure; implementation exceptions remain explicit
retry requirements and can never masquerade as a negative result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .adaptive import (
    AdaptiveRouter,
    EvidenceStatus,
    ExecutionContext,
    ExecutionState,
    ExperimentOutcome,
    RetryDirective,
    ScientificDecision,
    ValidityReport,
    build_exp3_research_router,
)
from .analysis_plan import (
    ConfirmatoryAnalysisPlan,
    required_analysis_experiment_ids,
)
from .artifacts import ArtifactRef, ArtifactStore
from .config import Exp3Config
from .data import TokenPanelRef, load_token_panel_refs
from .decisions import validate_decision_schema
from .inputs import Exp2Handoff
from .runtime import RuntimeManifest


ExperimentResult = ExperimentOutcome | RetryDirective


@dataclass(frozen=True)
class Exp3RunContext:
    """Immutable context passed to one EXP3 experiment implementation."""

    experiment_id: str
    execution: ExecutionContext
    handoff: Exp2Handoff
    config: Exp3Config
    run_dir: Path
    analysis_plan: ConfirmatoryAnalysisPlan | None = None
    token_panel_refs: tuple[TokenPanelRef, ...] = ()

    @property
    def fixed_exp2_foundations(self) -> Mapping[str, bool]:
        return MappingProxyType(dict(self.handoff.accepted_exp2_foundations))

    @property
    def expected_runtime_provenance(self) -> Mapping[str, str]:
        return MappingProxyType({
            "checkpoint_sha256": self.handoff.checkpoint_sha256,
            "architecture_sha256": self.handoff.architecture_sha256,
            "code_sha256": self.handoff.code_sha256,
        })

    @property
    def expected_confirmatory_analysis(self) -> Mapping[str, Any] | None:
        """Return the sealed Phase 3--8 declaration this node must use.

        Phase 1--2 nodes and model-free software fixtures have no downstream
        declaration and return ``None``.  A scientific provider must place the
        returned mapping in ``decision.diagnostics['confirmatory_analysis']``
        after actually using those values.
        """
        if (self.analysis_plan is None
                or self.experiment_id not in self.analysis_plan.by_experiment):
            return None
        return MappingProxyType(
            self.analysis_plan.attestation(self.experiment_id))

    @property
    def token_panel_artifacts(self) -> tuple[ArtifactRef, ...]:
        """Return the run-bound discovery/development/locked panel artifacts."""
        return tuple(reference.artifact_ref()
                     for reference in self.token_panel_refs)


ExperimentCallback = Callable[[Exp3RunContext], ExperimentResult]

NODE_REQUIRED_HANDOFF_INPUTS: dict[str, tuple[str, ...]] = {
    "b29_decomposition": ("u28",),
    "scale_content_carrier_cube": ("u28", "carrier_axis_x31"),
    "cube_b30_mediation": ("b30_subspace",),
    "bidirectional_direction_transfer": (
        "u28", "b30_subspace", "reference_directions"),
}


def _source_sha256(value: Any) -> str:
    target = value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
    source = inspect.getsourcefile(target)
    if source and Path(source).is_file():
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    code = getattr(value, "__code__", None)
    if code is None and inspect.ismethod(value):
        code = getattr(value.__func__, "__code__", None)
    if code is None:
        raise ValueError("callback/provider implementation has no hashable source")
    return hashlib.sha256(code.co_code).hexdigest()


def _callback_contract(callback: ExperimentCallback) -> dict[str, Any]:
    """Bind resume state to the concrete provider implementation and setup."""
    target = callback
    module = getattr(target, "__module__", type(target).__module__)
    qualname = getattr(target, "__qualname__", type(target).__qualname__)
    state_owner = getattr(target, "__self__", None) or target
    raw_state = getattr(state_owner, "__dict__", {})
    state = {
        str(key): value for key, value in raw_state.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    identity: dict[str, Any] = {
        "module": module,
        "qualname": qualname,
        "source_sha256": _source_sha256(target),
        "primitive_state": state,
    }
    provider_contract = getattr(callback, "_exp3_provider_contract", None)
    if provider_contract is not None:
        identity["bound_provider"] = provider_contract
    # configured_callback is only a bridge; bind the actual configured
    # provider module as well so changing EXP3_PROVIDER_FACTORY cannot resume
    # into an existing ledger unnoticed.
    if module == "exp3.callbacks" and qualname.endswith("configured_callback"):
        spec = os.environ.get("EXP3_PROVIDER_FACTORY", "").strip()
        identity["provider_factory"] = spec
        if spec and ":" in spec:
            provider_module = importlib.import_module(spec.rsplit(":", 1)[0])
            source = inspect.getsourcefile(provider_module)
            if source and Path(source).is_file():
                identity["provider_source_sha256"] = hashlib.sha256(
                    Path(source).read_bytes()).hexdigest()
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    identity["provider_id"] = str(
        (provider_contract or {}).get("explicit_provider_id")
        or identity["identity_sha256"])
    return identity


def callback_contract(callback: ExperimentCallback) -> Mapping[str, Any]:
    """Return the immutable identity used to attest a scientific provider.

    This public, read-only view exists so a real provider can create its
    :class:`~exp3.runtime.RuntimeManifest` *before* starting a run without
    duplicating the orchestration's hashing rules.  The same contract is
    recomputed inside :func:`execute_program` and sealed into
    ``run_contract.json``.
    """
    return MappingProxyType(_callback_contract(callback))


def _exp3_tree_sha256() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        relative = source.relative_to(package).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _router_schema_sha256(router: AdaptiveRouter) -> str:
    payload = {
        "experiments": {
            key: {
                "purpose": record.spec.purpose,
                "phase": record.spec.phase,
                "depends_on": list(record.spec.depends_on),
                "initially_active": record.spec.initially_active,
                "tags": list(record.spec.tags),
            }
            for key, record in sorted(router.records.items())
        },
        "routes": {
            key: {
                "when": {
                    source: {
                        "statuses": sorted(x.value for x in matcher.statuses),
                        "codes": sorted(matcher.conclusion_codes),
                    }
                    for source, matcher in sorted(rule.when.items())
                },
                "activate": list(rule.activate),
                "rationale": rule.rationale,
            }
            for key, rule in sorted(router.rules.items())
        },
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_callback(spec: str) -> ExperimentCallback:
    if ":" not in spec:
        raise ValueError("callback must be written as package.module:function")
    module_name, attribute = spec.rsplit(":", 1)
    if not module_name or not attribute:
        raise ValueError("callback must be written as package.module:function")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"callback {spec!r} is not callable")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _contract_payload(*, run_id: str, handoff: Exp2Handoff,
                      config: Exp3Config, run_mode: str,
                      callback_contract: Mapping[str, Any],
                      runtime_manifest: RuntimeManifest | None,
                      analysis_plan: ConfirmatoryAnalysisPlan | None,
                      token_panel_refs: Sequence[TokenPanelRef],
                      exp3_code_sha256: str,
                      graph_schema_sha256: str) -> dict[str, Any]:
    """Bind resumable state to one exact EXP2 handoff and EXP3 config."""
    return {
        "schema_version": 3,
        "program": "EXP3",
        "run_id": run_id,
        "handoff_sha256": handoff.handoff_sha256,
        "exp2_run_id": handoff.exp2_run_id,
        "checkpoint_sha256": handoff.checkpoint_sha256,
        "architecture_sha256": handoff.architecture_sha256,
        "exp2_code_sha256": handoff.code_sha256,
        "input_sha256": {
            name: value.sha256 for name, value in sorted(handoff.inputs.items())
        },
        "accepted_exp2_foundations": dict(sorted(
            handoff.accepted_exp2_foundations.items())),
        "exp3_config_sha256": config.config_sha256,
        "run_mode": run_mode,
        "callback": dict(callback_contract),
        "runtime_manifest_sha256": (
            None if runtime_manifest is None else runtime_manifest.manifest_sha256),
        "confirmatory_analysis_plan_sha256": (
            None if analysis_plan is None else analysis_plan.plan_sha256),
        "token_panels": [reference.as_dict()
                         for reference in token_panel_refs],
        "exp3_code_sha256": exp3_code_sha256,
        "graph_schema_sha256": graph_schema_sha256,
    }


def _seal_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _load_contract(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    observed = raw.pop("contract_sha256", "")
    expected = _seal_contract(raw)["contract_sha256"]
    if observed != expected:
        raise RuntimeError("EXP3 run contract was modified")
    raw["contract_sha256"] = observed
    return raw


def _prepare_run_contract(*, run_dir: Path, resume_path: Path | None,
                          expected: Mapping[str, Any]) -> Path:
    """Create or verify the immutable context of a resumable run."""
    target = run_dir / "run_contract.json"
    sealed_expected = _seal_contract(expected)
    if resume_path is None:
        if (run_dir / "adaptive_state.json").exists():
            raise RuntimeError(
                "run already has state; pass resume_path or choose a new run_id")
        if target.exists():
            existing = _load_contract(target)
            if existing != sealed_expected:
                raise RuntimeError(
                    "run directory belongs to a different EXP2 handoff or EXP3 config")
        else:
            _atomic_json(target, sealed_expected)
        return target

    source = resume_path.parent / "run_contract.json"
    if not source.is_file():
        raise RuntimeError("resume state has no immutable run_contract.json")
    existing = _load_contract(source)
    if existing != sealed_expected:
        raise RuntimeError(
            "resume contract does not match this EXP2 handoff, config, or run_id")
    if target.exists() and _load_contract(target) != existing:
        raise RuntimeError("output directory has a conflicting EXP3 run contract")
    if not target.exists():
        _atomic_json(target, existing)
    return target


def _program_status(router: AdaptiveRouter) -> str:
    states = [record.state for record in router.records.values()]
    if ExecutionState.HARD_FAILED in states or ExecutionState.BLOCKED_INVALID in states:
        return "invalid"
    if ExecutionState.RETRY_REQUIRED in states:
        return "retry_required"
    if ExecutionState.RUNNING in states:
        return "running"
    if ExecutionState.READY in states:
        return "ready"
    # Dormant nodes are unselected alternatives, not unfinished work.
    return "complete"


def _split_mentions_chromosome(source_split: str, chromosome: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(chromosome)}(?![A-Za-z0-9])"
    return re.search(pattern, source_split, flags=re.IGNORECASE) is not None


def _validate_config_bound_diagnostics(
    experiment_id: str, outcome: ExperimentOutcome, config: Exp3Config,
    *, software_fixture: bool,
) -> None:
    diagnostics = outcome.decision.diagnostics
    if diagnostics.get("config_sha256") != config.config_sha256:
        raise ValueError(
            f"{experiment_id} decision is not bound to the sealed EXP3 config")
    margins = config.margins
    expected_margins: dict[str, Mapping[str, Any]] = {
        "m28_preconditioning": {
            "damage_margin": margins.m28_damage,
            "rescue_margin": margins.rescue,
            "equivalence_margin": margins.equivalence,
            "product_change_margin": margins.m28_product_change,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
        },
        "b29_decomposition": {
            "effect_margin": margins.b29_effect,
            "equivalence_margin": margins.equivalence,
            "perpendicular_fraction_margin": margins.b29_perpendicular_fraction,
            "independent_effect_margin": margins.b29_independent_effect,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
            "parallel_plateau": {
                "effect_margin": margins.b29_effect,
                "increment_equivalence_margin": margins.b29_plateau_increment,
                "log_q_boundary_margin": margins.b29_log_q_boundary,
                "plateau_remaining_fraction": (
                    margins.b29_plateau_remaining_fraction),
                "plateau_boundary_spread": margins.b29_plateau_boundary_spread,
                "min_clusters": margins.min_dependency_clusters,
                "n_boot": margins.bootstrap_replicates,
                "alpha": margins.alpha,
            },
        },
        "scale_plateau": {
            "plateau_increment_margin": margins.scale_plateau_increment,
            "angular_specificity_margin": margins.angular_specificity,
            "collapse_rmse_margin": margins.curve_collapse_rmse,
            "transition_effect_margin": margins.scale_transition_effect,
            "log_q_boundary_margin": margins.scale_log_q_boundary,
            "plateau_remaining_fraction": (
                margins.scale_plateau_remaining_fraction),
            "plateau_boundary_spread": margins.scale_plateau_boundary_spread,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
        },
        "scale_content_carrier_cube": {
            "effect_margins": {
                "scale": margins.cube_scale_effect,
                "content": margins.cube_content_effect,
                "carrier": margins.cube_carrier_effect,
            },
            "equivalence_margin": margins.cube_axis_equivalence,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
        },
        "cube_b30_mediation": {
            "effect_margin": margins.b30_mediation_effect,
            "rescue_margin": margins.b30_mediation_rescue,
            "equivalence_margin": margins.b30_mediation_equivalence,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
        },
        "bidirectional_direction_transfer": {
            "effect_margin": margins.direction_transfer_effect,
            "specificity_margins": {
                name: margins.direction_specificity for name in (
                    "norm_only", "wrong_layer", "wrong_pair", "random_matched")
            },
            "equivalence_margin": margins.direction_transfer_equivalence,
            "rescue_margin": margins.direction_transfer_rescue,
            "min_clusters": margins.min_dependency_clusters,
            "n_boot": margins.bootstrap_replicates,
            "alpha": margins.alpha,
        },
    }
    # Software-only wiring fixtures are explicitly marked and never become
    # scientific evidence.  Real root-panel decisions must expose the exact
    # values passed to the verdict, not merely copy the config alongside it.
    if (experiment_id in expected_margins
            and not software_fixture
            and diagnostics.get("margins_used") != expected_margins[experiment_id]):
        raise ValueError(
            f"{experiment_id} did not use the preregistered EXP3 margins")
    if experiment_id not in {"b29_decomposition", "scale_plateau"}:
        return
    observed = diagnostics.get("alpha_doses")
    if (not isinstance(observed, (list, tuple))
            or len(observed) != len(config.alpha_doses)):
        raise ValueError(
            f"{experiment_id} must record the exact sealed config alpha_doses")
    for actual, expected in zip(observed, config.alpha_doses):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isclose(float(actual), float(expected),
                                    rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError(
                f"{experiment_id} alpha_doses differ from the sealed EXP3 config")
    if experiment_id == "scale_plateau":
        expected_models = list(config.comparison_models)
        expected_sites = {
            model: list(sites) for model, sites in config.required_scale_sites
        }
        if diagnostics.get("required_models") != expected_models \
                or diagnostics.get("required_sites") != expected_sites:
            raise ValueError(
                "scale_plateau did not use the sealed cross-model/site panel")


def _validate_runtime_provenance(
    experiment_id: str, outcome: ExperimentOutcome, handoff: Exp2Handoff
) -> None:
    expected = {
        "checkpoint_sha256": handoff.checkpoint_sha256,
        "architecture_sha256": handoff.architecture_sha256,
        "code_sha256": handoff.code_sha256,
    }
    observed = outcome.decision.diagnostics.get("runtime_provenance")
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise ValueError(
            f"{experiment_id} did not attest the actually loaded runtime "
            "against the sealed EXP2 handoff")


def _confirmatory_experiment_ids(router: AdaptiveRouter) -> tuple[str, ...]:
    """Return every possible Phase 3--8 node, including dormant branches."""
    return required_analysis_experiment_ids(router)


def _validate_confirmatory_analysis(
    experiment_id: str,
    outcome: ExperimentOutcome,
    analysis_plan: ConfirmatoryAnalysisPlan | None,
) -> None:
    """Require exact provider attestation of the pre-locked declaration."""
    if analysis_plan is None:
        return
    declaration = analysis_plan.by_experiment.get(experiment_id)
    if declaration is None:
        return
    expected = analysis_plan.attestation(experiment_id)
    observed = outcome.decision.diagnostics.get("confirmatory_analysis")
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise ValueError(
            f"{experiment_id} did not use the sealed confirmatory analysis "
            "thresholds, hyperparameters, controls, and decision rule")


def execute_program(
    *,
    handoff_path: str | Path,
    output_dir: str | Path,
    callback: ExperimentCallback,
    config_path: str | Path | None = None,
    run_id: str = "exp3",
    resume_path: str | Path | None = None,
    max_nodes: int | None = None,
    software_fixture: bool = False,
    runtime_manifest_path: str | Path | None = None,
    analysis_plan_path: str | Path | None = None,
    token_panel_paths: Sequence[str | Path] = (),
) -> tuple[Path, dict[str, Any], AdaptiveRouter]:
    """Run all currently reachable EXP3 nodes and atomically persist state.

    ``callback`` receives only EXP3 node ids.  The handoff is fully verified
    before any callback can see data.  A resumed router must have the same
    run id and fixed foundation contract; its own hash-chained ledger is also
    verified during loading.
    """

    if not run_id.strip():
        raise ValueError("run_id is required")
    if max_nodes is not None and max_nodes <= 0:
        raise ValueError("max_nodes must be positive")

    handoff_source = Path(handoff_path).expanduser().resolve()
    handoff = Exp2Handoff.load(handoff_source, verify_files=True)
    callback_contract = _callback_contract(callback)
    runtime_manifest: RuntimeManifest | None = None
    runtime_source: Path | None = None
    if not software_fixture:
        if runtime_manifest_path is None:
            raise RuntimeError(
                "scientific EXP3 runs require a hash-sealed runtime manifest; "
                "software_fixture=True is reserved for model-free tests")
        runtime_source = Path(runtime_manifest_path).expanduser().resolve()
        runtime_manifest = RuntimeManifest.load(runtime_source)
        expected_provider_sha = str(callback_contract.get(
            "provider_source_sha256", callback_contract["source_sha256"]))
        observed = {
            "checkpoint_sha256": runtime_manifest.checkpoint_sha256,
            "architecture_sha256": runtime_manifest.architecture_sha256,
            "exp2_code_sha256": runtime_manifest.exp2_code_sha256,
            "provider_source_sha256": runtime_manifest.provider_source_sha256,
        }
        expected = {
            "checkpoint_sha256": handoff.checkpoint_sha256,
            "architecture_sha256": handoff.architecture_sha256,
            "exp2_code_sha256": handoff.code_sha256,
            "provider_source_sha256": expected_provider_sha,
        }
        if observed != expected:
            raise RuntimeError(
                "runtime manifest does not match the EXP2 handoff or callback code")
        if runtime_manifest.provider_id != callback_contract["provider_id"]:
            raise RuntimeError(
                "runtime manifest provider_id does not match the bound callback/provider")
    config_source = None if config_path is None else Path(config_path).expanduser().resolve()
    config = Exp3Config() if config_source is None else Exp3Config.load(config_source)
    planned_router = build_exp3_research_router(run_id)
    analysis_plan: ConfirmatoryAnalysisPlan | None = None
    analysis_plan_source: Path | None = None
    if not software_fixture:
        if analysis_plan_path is None:
            raise RuntimeError(
                "scientific EXP3 runs require a confirmatory Phase 3--8 "
                "analysis plan sealed before locked data")
        analysis_plan_source = Path(analysis_plan_path).expanduser().resolve()
        analysis_plan = ConfirmatoryAnalysisPlan.load(analysis_plan_source)
        analysis_plan.validate(
            required_experiment_ids=_confirmatory_experiment_ids(planned_router),
            expected_config_sha256=config.config_sha256,
            expected_locked_split=config.locked_chromosome,
        )
    token_panel_refs = load_token_panel_refs(
        tuple(token_panel_paths),
        require_scientific_roles=not software_fixture,
    )
    if not software_fixture:
        required_panel_roles = {
            "discovery": config.discovery_chromosome,
            "development": config.development_chromosome,
            "locked": config.locked_chromosome,
        }
        by_role: dict[str, list[TokenPanelRef]] = {
            role: [] for role in required_panel_roles
        }
        for reference in token_panel_refs:
            if reference.source_role in required_panel_roles:
                expected_chromosome = required_panel_roles[reference.source_role]
                if not _split_mentions_chromosome(
                        reference.source_split, expected_chromosome):
                    raise RuntimeError(
                        f"{reference.source_role} token panel {reference.name!r} "
                        f"does not identify {expected_chromosome!r} in source_split")
                by_role[reference.source_role].append(reference)
            elif reference.source_role != "external":
                raise RuntimeError(
                    f"scientific token panel {reference.name!r} has unsupported "
                    f"source role {reference.source_role!r}")
        missing_roles = sorted(role for role, refs in by_role.items() if not refs)
        if missing_roles:
            raise RuntimeError(
                "scientific EXP3 runs require hash-sealed discovery, development, "
                f"and locked token panels; missing={missing_roles}")
        split_panels = [reference.verify() for role in required_panel_roles
                        for reference in by_role[role]]
        for left_index, left in enumerate(split_panels):
            for right in split_panels[left_index + 1:]:
                left.assert_split_disjoint(right)
    token_panel_artifacts = tuple(
        reference.artifact_ref() for reference in token_panel_refs)
    locked_inputs = sorted(
        name for name, value in handoff.inputs.items()
        if _split_mentions_chromosome(
            value.source_split, config.locked_chromosome)
    )
    if locked_inputs:
        raise RuntimeError(
            "EXP2 handoff selection inputs overlap the EXP3 locked chromosome "
            f"{config.locked_chromosome}: {locked_inputs}")

    run_dir = Path(output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_source = (
        None if resume_path is None else Path(resume_path).expanduser().resolve()
    )
    contract_path = _prepare_run_contract(
        run_dir=run_dir,
        resume_path=resume_source,
        expected=_contract_payload(
            run_id=run_id, handoff=handoff, config=config,
            run_mode="software_fixture" if software_fixture else "scientific",
            callback_contract=callback_contract,
            runtime_manifest=runtime_manifest,
            analysis_plan=analysis_plan,
            token_panel_refs=token_panel_refs,
            exp3_code_sha256=_exp3_tree_sha256(),
            graph_schema_sha256=_router_schema_sha256(planned_router)),
    )
    if resume_path is None:
        router = planned_router
    else:
        router = AdaptiveRouter.load(resume_source)
        if router.run_id != run_id:
            raise RuntimeError(
                f"resume run_id {router.run_id!r} does not match {run_id!r}"
            )

    def invalid_provenance(error: Exception) -> ExperimentOutcome:
        return ExperimentOutcome(
            decision=ScientificDecision(
                EvidenceStatus.UNRESOLVED,
                "provenance_invalid",
                "No scientific interpretation is permitted after sealed run-input mutation.",
            ),
            validity=ValidityReport(
                provenance_valid=False,
                details={
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
            ),
        )

    def invoke(execution: ExecutionContext) -> ExperimentResult:
        # The handoff was validated before the graph was constructed.  Check it
        # around every external callback as well so accidental mutation or an
        # artifact replacement becomes provenance invalidity, never evidence.
        try:
            handoff.validate(verify_files=True)
            for artifact in token_panel_artifacts:
                ArtifactStore.verify(artifact)
            required = NODE_REQUIRED_HANDOFF_INPUTS.get(
                execution.experiment_id, ())
            if required:
                handoff.require_inputs(*required)
        except Exception as error:
            return invalid_provenance(error)
        context = Exp3RunContext(
            experiment_id=execution.experiment_id,
            execution=execution,
            handoff=handoff,
            config=config,
            run_dir=run_dir,
            analysis_plan=analysis_plan,
            token_panel_refs=token_panel_refs,
        )
        try:
            result = callback(context)
        except Exception:
            # If the callback both damaged provenance and raised, provenance
            # takes precedence.  Otherwise AdaptiveRouter records a retryable
            # implementation failure from the original exception.
            try:
                handoff.validate(verify_files=True)
                for artifact in token_panel_artifacts:
                    ArtifactStore.verify(artifact)
            except Exception as error:
                return invalid_provenance(error)
            raise
        try:
            handoff.validate(verify_files=True)
            for artifact in token_panel_artifacts:
                ArtifactStore.verify(artifact)
        except Exception as error:
            return invalid_provenance(error)
        if isinstance(result, ScientificDecision):
            raise TypeError(
                "production EXP3 callbacks must return ExperimentOutcome with an "
                "explicit ValidityReport; a bare ScientificDecision is not auditable")
        if isinstance(result, ExperimentOutcome) and result.validity.valid:
            try:
                validate_decision_schema(execution.experiment_id, result.decision)
                _validate_config_bound_diagnostics(
                    execution.experiment_id, result, config,
                    software_fixture=software_fixture)
                _validate_runtime_provenance(
                    execution.experiment_id, result, handoff)
                _validate_confirmatory_analysis(
                    execution.experiment_id, result, analysis_plan)
            except Exception as error:
                # A result bound to the wrong config/checkpoint/code, or one
                # outside the preregistered decision schema, is invalid data
                # lineage.  It must hard-fail this node instead of entering
                # the retry channel used for transient CUDA/software errors.
                return ExperimentOutcome(
                    decision=result.decision,
                    validity=ValidityReport(
                        measurement_valid=False,
                        provenance_valid=False,
                        leakage_free=result.validity.leakage_free,
                        details={
                            "error_type": type(error).__name__,
                            "reason": str(error),
                            "stage": "decision_contract_validation",
                        },
                    ),
                )
        return result

    runners = {experiment_id: invoke for experiment_id in router.records}
    state_path = run_dir / "adaptive_state.json"
    # Persist the initial graph and then every node transition.  A scheduler
    # pre-emption during a later GPU callback can lose only that in-flight
    # node, never all earlier decisions from the invocation.
    router.save(state_path)
    executed_items: list[str] = []
    while max_nodes is None or len(executed_items) < max_nodes:
        batch = router.run_ready(runners, max_nodes=1)
        if not batch:
            break
        executed_items.extend(batch)
        router.save(state_path)
    executed = tuple(executed_items)
    report: dict[str, Any] = {
        "schema_version": 1,
        "program": "EXP3",
        "run_id": run_id,
        "status": _program_status(router),
        "run_mode": "software_fixture" if software_fixture else "scientific",
        "runtime_manifest": (
            None if runtime_manifest is None else {
                "path": str(runtime_source),
                "manifest_sha256": runtime_manifest.manifest_sha256,
                "provider_id": runtime_manifest.provider_id,
            }),
        "confirmatory_analysis_plan": (
            None if analysis_plan is None else {
                "path": str(analysis_plan_source),
                "plan_id": analysis_plan.plan_id,
                "plan_sha256": analysis_plan.plan_sha256,
                "sealed_at_utc": analysis_plan.sealed_at_utc,
                "registration_reference": analysis_plan.registration_reference,
            }),
        "token_panels": [reference.as_dict()
                         for reference in token_panel_refs],
        "exp2_was_executed": False,
        "exp2_handoff": {
            "path": str(handoff_source),
            "handoff_id": handoff.handoff_id,
            "exp2_run_id": handoff.exp2_run_id,
            "handoff_sha256": handoff.handoff_sha256,
            "accepted_foundations": dict(handoff.accepted_exp2_foundations),
        },
        "config": {
            "path": None if config_source is None else str(config_source),
            "config_sha256": config.config_sha256,
        },
        "executed_this_invocation": list(executed),
        "state_path": str(state_path),
        "run_contract_path": str(contract_path),
        "router": router.report(),
    }
    report_path = run_dir / "report.json"
    _atomic_json(report_path, report)
    return report_path, report, router


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exp3")
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="exp3")
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--analysis-plan")
    parser.add_argument(
        "--token-panel", action="append", default=[],
        help=("Path to a schema-v2 token-panel directory; repeat for "
              "discovery, development, locked, and optional external panels"),
    )
    parser.add_argument(
        "--callback",
        default="exp3.callbacks:configured_callback",
        help="EXP3 callback as package.module:function",
    )
    arguments = parser.parse_args(argv)
    report_path, report, _ = execute_program(
        handoff_path=arguments.handoff,
        output_dir=arguments.output_dir,
        callback=_load_callback(arguments.callback),
        config_path=arguments.config,
        run_id=arguments.run_id,
        resume_path=arguments.resume,
        max_nodes=arguments.max_nodes,
        runtime_manifest_path=arguments.runtime_manifest,
        analysis_plan_path=arguments.analysis_plan,
        token_panel_paths=arguments.token_panel,
    )
    print(
        f"[exp3] {report['status']}; executed "
        f"{len(report['executed_this_invocation'])} EXP3 nodes; wrote {report_path}"
    )
    return 0 if report["status"] in {"complete", "ready", "retry_required"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Exp3RunContext", "ExperimentCallback", "callback_contract",
    "execute_program", "main",
]

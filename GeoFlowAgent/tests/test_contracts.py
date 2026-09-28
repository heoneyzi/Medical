from __future__ import annotations

from pathlib import Path

import pytest

from geoflowagent.constants import INVALID_REGRET, STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine, condition_holds
from geoflowagent.evaluation.environment import SymbolicGenomicsEnvironment
from geoflowagent.utils.io import read_jsonl


def _engine(raw_dir: Path) -> ContractEngine:
    return ContractEngine(read_jsonl(raw_dir / "tools.jsonl"))


def test_exists_and_missing_distinguish_absence_from_present_none() -> None:
    state = {"present_none": None, "nested": {"value": 7}}

    assert condition_holds(state, {"field": "present_none", "op": "exists"})
    assert not condition_holds(state, {"field": "present_none", "op": "missing"})
    assert condition_holds(state, {"field": "absent", "op": "missing"})
    assert not condition_holds(state, {"field": "absent", "op": "exists"})
    assert condition_holds(state, {"field": "nested.absent", "op": "missing"})


def test_exact_genome_build_predicates_are_enforced(smoke_raw_dir: Path) -> None:
    engine = _engine(smoke_raw_dir)
    state = {
        "raw_variant": "SYNTH_TX.1:c.101A>G",
        "assembly": "GRCh37",
        "sequence_reference": "SYNTH_CONTIG.1",
        "coordinate_system": "one_based_closed",
    }

    assert engine.applicable(state, "parse_variant")
    assert not engine.applicable(state, "liftover_grch37_to_grch38")
    assert not engine.applicable(state, "verify_reference_allele")

    parsed = engine.execute(state, "parse_variant").state
    assert engine.applicable(parsed, "liftover_grch37_to_grch38")
    assert not engine.applicable(parsed, "verify_reference_allele")

    lifted = engine.execute(parsed, "liftover_grch37_to_grch38").state
    assert lifted["assembly"] == "GRCh38"
    assert lifted["ref_alt_verified"] is False
    assert engine.applicable(lifted, "verify_reference_allele")
    assert not engine.applicable(lifted, "liftover_grch37_to_grch38")


def test_shortest_path_analysis_and_terminal_stop(smoke_raw_dir: Path) -> None:
    tools = read_jsonl(smoke_raw_dir / "tools.jsonl")
    tasks = read_jsonl(smoke_raw_dir / "tasks.jsonl")
    engine = ContractEngine(tools)
    task = next(row for row in tasks if row["task_id"] == "synth-train-001")

    distance = engine.shortest_distance(
        task["initial_state"], task["goal"], task["available_tools"], max_depth=12
    )
    optimal, regrets = engine.action_analysis(
        task["initial_state"], task["goal"], task["available_tools"], max_depth=12
    )

    assert distance == 8
    # Variant parsing and phenotype matching are independent at the initial state,
    # so both orders begin a shortest valid trajectory.
    assert optimal == ["match_phenotype", "parse_variant"]
    assert regrets["parse_variant"] == 0.0
    assert regrets["match_phenotype"] == 0.0
    assert regrets[STOP_TOOL_ID] == INVALID_REGRET

    state = task["initial_state"]
    trajectory = next(
        row
        for row in read_jsonl(smoke_raw_dir / "trajectories.jsonl")
        if row["task_id"] == task["task_id"]
    )
    for tool_id in trajectory["tool_ids"]:
        state = engine.execute(state, tool_id).state

    assert engine.goal_satisfied(state, task["goal"])
    terminal_actions, terminal_regrets = engine.action_analysis(
        state, task["goal"], task["available_tools"]
    )
    assert terminal_actions == [STOP_TOOL_ID]
    assert terminal_regrets[STOP_TOOL_ID] == 0.0
    assert all(terminal_regrets[tool_id] == INVALID_REGRET for tool_id in task["available_tools"])


def test_argument_bindings_are_resolved_and_schema_checked() -> None:
    engine = ContractEngine(
        [
            {
                "tool_id": "lookup_position",
                "name": "Lookup position",
                "description": "Synthetic nested argument-binding test.",
                "input_schema": {
                    "type": "object",
                    "required": ["chromosome", "position"],
                    "properties": {
                        "chromosome": {"type": "string", "enum": ["1"]},
                        "position": {"type": "integer", "minimum": 1},
                    },
                },
                "argument_bindings": {
                    "chromosome": "selected_variant.chromosome",
                    "position": "selected_variant.position",
                },
                "preconditions": [{"field": "assembly", "op": "eq", "value": "GRCh38"}],
                "effects": [{"field": "looked_up", "op": "set", "value": True}],
            }
        ]
    )
    state = {
        "assembly": "GRCh38",
        "selected_variant": {"chromosome": "1", "position": 101},
    }

    assert engine.applicable(state, "lookup_position")
    assert engine.resolve_arguments(state, "lookup_position") == {
        "chromosome": "1",
        "position": 101,
    }
    transition = engine.execute(state, "lookup_position")
    assert transition.observation["arguments"]["position"] == 101
    assert not engine.applicable(
        {**state, "selected_variant": {"chromosome": "1", "position": "101"}},
        "lookup_position",
    )


def _snapshot_tool() -> dict:
    return {
        "tool_id": "snapshot_lookup",
        "name": "Snapshot lookup",
        "description": "Resolve a pinned offline result.",
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": ["case_id"],
            "properties": {"case_id": {"type": "string"}},
            "additionalProperties": False,
        },
        "argument_bindings": {"case_id": "case.id"},
        "output_schema": {
            "type": "object",
            "required": ["annotation"],
            "properties": {
                "annotation": {
                    "type": "object",
                    "required": ["gene"],
                    "properties": {"gene": {"type": "string"}},
                }
            },
        },
        # Fixed direction: snapshot output path -> typed-state destination path.
        "output_bindings": {"annotation.gene": "result.gene_symbol"},
        "preconditions": [{"field": "case.id", "op": "nonempty"}],
        "effects": [{"field": "lookup_complete", "op": "set", "value": True}],
    }


def test_snapshot_lookup_is_exact_validated_and_updates_typed_state() -> None:
    snapshot = {
        "snapshot_id": "snapshot-case-1",
        "tool_id": "snapshot_lookup",
        "arguments": {"case_id": "case-1"},
        "status": "success",
        "output": {"annotation": {"gene": "GENE1"}},
        "source_revision": "fixture-v1",
        "summary": "Pinned result found.",
    }
    engine = ContractEngine([_snapshot_tool()], [snapshot])
    state = {"case": {"id": "case-1"}}

    assert engine.applicable(state, "snapshot_lookup")
    assert engine.executable(state, "snapshot_lookup")
    transition = engine.execute(state, "snapshot_lookup")

    assert transition.state["result"]["gene_symbol"] == "GENE1"
    assert transition.state["lookup_complete"] is True
    assert transition.observation["status"] == "success"
    assert transition.observation["arguments"] == {"case_id": "case-1"}
    assert transition.observation["output"] == snapshot["output"]
    assert transition.observation["changed_fields"] == ["lookup_complete", "result.gene_symbol"]

    unseen = {"case": {"id": "case-2"}}
    assert engine.applicable(unseen, "snapshot_lookup")
    assert not engine.executable(unseen, "snapshot_lookup")
    with pytest.raises(LookupError, match="No exact snapshot"):
        engine.execute(unseen, "snapshot_lookup")


def test_duplicate_snapshot_key_and_invalid_success_output_fail_closed() -> None:
    base = {
        "snapshot_id": "snapshot-a",
        "tool_id": "snapshot_lookup",
        "arguments": {"case_id": "case-1"},
        "status": "success",
        "output": {"annotation": {"gene": "GENE1"}},
        "source_revision": "fixture-v1",
    }
    duplicate = {**base, "snapshot_id": "snapshot-b"}
    with pytest.raises(ValueError, match="Duplicate snapshot"):
        ContractEngine([_snapshot_tool()], [base, duplicate])

    invalid = {**base, "output": {"annotation": {}}}
    with pytest.raises(ValueError, match="output fails schema"):
        ContractEngine([_snapshot_tool()], [invalid])


def test_non_success_snapshot_preserves_state_and_records_typed_status() -> None:
    snapshot = {
        "snapshot_id": "snapshot-empty",
        "tool_id": "snapshot_lookup",
        "arguments": {"case_id": "empty-case"},
        "status": "empty",
        "output": {},
        "source_revision": "fixture-v1",
        "summary": "The pinned response contained no result.",
    }
    engine = ContractEngine([_snapshot_tool()], [snapshot])
    state = {"case": {"id": "empty-case"}}

    transition = engine.execute(state, "snapshot_lookup")

    assert transition.state == state
    assert transition.observation["ok"] is False
    assert transition.observation["status"] == "empty"
    assert transition.observation["output"] == {}

    task = {
        "task_id": "empty-case",
        "query": "Lookup the fixture.",
        "initial_state": state,
        "goal": {"lookup_complete": True},
        "available_tools": ["snapshot_lookup"],
    }
    environment = SymbolicGenomicsEnvironment(task, [_snapshot_tool()], snapshots=[snapshot])
    step = environment.step("snapshot_lookup")
    assert step.valid_action is True
    assert step.execution_success is False
    assert step.observation["status"] == "empty"
    assert step.observation["output"] == {}


def test_contract_invalid_attempt_does_not_consume_tool_result_perturbation() -> None:
    tool = {
        "tool_id": "requires_ready",
        "name": "Requires ready",
        "description": "A call whose exact contract requires ready=true.",
        "preconditions": [{"field": "ready", "op": "eq", "value": True}],
        "effects": [{"field": "done", "op": "set", "value": True}],
    }
    perturbation = {
        "name": "one-empty-result",
        "tool_id": "requires_ready",
        "kind": "empty_result",
        "once": True,
    }
    task = {
        "task_id": "invalid-does-not-trigger",
        "query": "Exercise a result perturbation only after a valid call.",
        "initial_state": {"ready": False},
        "goal": {"done": True},
        "available_tools": ["requires_ready"],
    }

    invalid_environment = SymbolicGenomicsEnvironment(task, [tool], perturbations=[perturbation])
    invalid = invalid_environment.step("requires_ready")

    assert invalid.valid_action is False
    assert invalid.perturbation_applied is False
    assert invalid.observation["status"] == "contract_invalid"
    assert invalid_environment.applied_perturbations == []

    valid_task = {**task, "initial_state": {"ready": True}}
    valid_environment = SymbolicGenomicsEnvironment(
        valid_task, [tool], perturbations=[perturbation]
    )
    valid = valid_environment.step("requires_ready")

    assert valid.valid_action is True
    assert valid.perturbation_applied is True
    assert valid.observation["status"] == "empty_result"
    assert valid_environment.applied_perturbations == ["one-empty-result"]

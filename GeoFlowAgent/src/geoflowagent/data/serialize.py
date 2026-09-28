from __future__ import annotations

import re
from typing import Any

from geoflowagent.utils.io import canonical_json, sha256_text

IUPAC_DNA = re.compile(r"[ACGTRYSWKMBDHVN]+", re.IGNORECASE)
ENTITY_FIELD_TOKENS = {
    "accession",
    "allele",
    "disease",
    "entity",
    "gene",
    "hgvs",
    "hpo",
    "identifier",
    "phenotype",
    "rsid",
    "transcript",
    "variant",
}


def serialize_goal(goal: dict[str, Any]) -> str:
    return "Goal predicates: " + canonical_json(goal)


def serialize_state(state: dict[str, Any]) -> str:
    exact_keys = [
        "assembly",
        "sequence_reference",
        "coordinate_system",
        "variant_normalization",
    ]
    exact = {key: state[key] for key in exact_keys if key in state}
    remainder = {key: value for key, value in state.items() if key not in exact_keys}
    return f"Exact genomic state: {canonical_json(exact)}. Other state: {canonical_json(remainder)}"


def serialize_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No tool has been executed yet."
    compact = [
        {
            "tool_id": step.get("tool_id"),
            "ok": step.get("ok"),
            "status": step.get("status"),
            "summary": step.get("summary"),
            "arguments": step.get("arguments", {}),
            "output": step.get("output", {}),
            "changed_fields": step.get("changed_fields", []),
        }
        for step in history
    ]
    return "Executed tool history: " + canonical_json(compact)


def serialize_tool_description(tool: dict[str, Any]) -> str:
    tags = ", ".join(tool.get("tags", []))
    return f"Tool: {tool['name']}. Description: {tool['description']}. Tags: {tags}."


def serialize_tool_contract(tool: dict[str, Any]) -> str:
    payload = {
        "input_schema": tool.get("input_schema", {}),
        "argument_bindings": tool.get("argument_bindings", {}),
        "output_schema": tool.get("output_schema", {}),
        "output_bindings": tool.get("output_bindings", {}),
        "execution_mode": tool.get("execution_mode", "symbolic"),
        "preconditions": tool.get("preconditions", []),
        "effects": tool.get("effects", []),
    }
    return "Executable contract: " + canonical_json(payload)


def serialize_background(card: dict[str, Any]) -> str:
    return (
        f"{card.get('title', '')}\n{card.get('text', '')}\nSource: {card.get('source', '')}".strip()
    )


def serialize_sequence(state: dict[str, Any]) -> str:
    """Return a nucleotide-only sequence window for sequence encoders.

    Empty sequence context is allowed and becomes a zero embedding in the pooling
    layer.  Non-empty metadata strings such as ``GRCh38`` are rejected instead of
    being silently passed to a DNA language model.
    """

    sequence = "".join(str(state.get("sequence_context", "")).split()).upper()
    if sequence and IUPAC_DNA.fullmatch(sequence) is None:
        raise ValueError(
            "state.sequence_context must contain only IUPAC DNA symbols; "
            f"received {sequence[:80]!r}"
        )
    return sequence


def serialize_entities(state: dict[str, Any]) -> str:
    """Serialize only explicitly entity-bearing state fields for entity encoders."""

    selected: dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                child = f"{path}.{key}" if path else str(key)
                visit(value[key], child)
            return
        leaf = path.rsplit(".", 1)[-1].lower()
        if any(token in leaf for token in ENTITY_FIELD_TOKENS):
            selected[path] = value

    visit(state, "")
    return "Biomedical entities: " + canonical_json(selected) if selected else ""


def runtime_context_key(
    task: dict[str, Any],
    state: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """Stable key for reusing a precomputed closed-loop prefix embedding."""

    visible = {
        "task_id": task["task_id"],
        "query": task["query"],
        "goal": task["goal"],
        "state": state,
        "history": history,
        "background_ids": task.get("background_ids", []),
    }
    return sha256_text(canonical_json(visible))

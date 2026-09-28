from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from geoflowagent.utils.io import read_json, read_jsonl


def test_documented_json_schemas_validate_smoke_and_processed_rows(
    repository_root: Path,
    processed_dir: Path,
) -> None:
    schema_dir = repository_root / "data" / "schemas"
    cases = [
        (schema_dir / "tool.schema.json", repository_root / "data/raw/smoke/tools.jsonl"),
        (schema_dir / "task.schema.json", repository_root / "data/raw/smoke/tasks.jsonl"),
        (
            schema_dir / "trajectory.schema.json",
            repository_root / "data/raw/smoke/trajectories.jsonl",
        ),
        (
            schema_dir / "background.schema.json",
            repository_root / "data/raw/smoke/background.jsonl",
        ),
        (
            schema_dir / "minimal_pair.schema.json",
            repository_root / "data/raw/smoke/minimal_pairs.jsonl",
        ),
        (schema_dir / "processed_example.schema.json", processed_dir / "examples.jsonl"),
    ]
    for schema_path, rows_path in cases:
        schema = read_json(schema_path)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for row_number, row in enumerate(read_jsonl(rows_path), start=1):
            errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
            assert not errors, f"{rows_path}:{row_number}: {errors[0].message}"


def test_private_verifier_and_snapshot_envelopes_have_machine_readable_schemas(
    repository_root: Path,
) -> None:
    schema_dir = repository_root / "data" / "schemas"
    rows = {
        "verifier.schema.json": {
            "task_id": "case-1",
            "private_verifier": {
                "conditions": [{"field": "result.code", "op": "eq", "value": "A1"}]
            },
        },
        "snapshot.schema.json": {
            "snapshot_id": "snapshot-1",
            "tool_id": "lookup",
            "arguments": {"case_id": "case-1"},
            "status": "success",
            "output": {"code": "A1"},
            "source_revision": "fixture-v1",
        },
    }
    for filename, row in rows.items():
        schema = read_json(schema_dir / filename)
        Draft202012Validator.check_schema(schema)
        errors = list(Draft202012Validator(schema).iter_errors(row))
        assert not errors, f"{filename}: {errors}"


def test_model_facing_task_schema_rejects_private_verifier(repository_root: Path) -> None:
    schema = read_json(repository_root / "data" / "schemas" / "task.schema.json")
    row = {
        "task_id": "leaky",
        "query": "Find an answer.",
        "initial_state": {},
        "goal": {"ready": True},
        "available_tools": ["lookup"],
        "private_verifier": {"answer": "SECRET"},
    }

    errors = list(Draft202012Validator(schema).iter_errors(row))

    assert errors

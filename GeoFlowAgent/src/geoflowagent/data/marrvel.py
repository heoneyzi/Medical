from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from geoflowagent.data.schema import validate_verifier
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
)

DATASET_ID = "hjeong84/marrvel-mcp-benchmark-data"


def _load_rows(
    input_path: str | None, dataset_id: str, revision: str | None
) -> list[dict[str, Any]]:
    if input_path:
        path = Path(input_path)
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        value = read_json(path)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("data", "rows", "test"):
                if isinstance(value.get(key), list):
                    return value[key]
        raise ValueError(f"Cannot find rows in {path}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Downloading MARRVEL requires `pip install -e '.[hf]'`, or pass --input."
        ) from exc
    dataset = load_dataset(dataset_id, revision=revision, split="test")
    return [dict(row) for row in dataset]


def import_marrvel(
    output_dir: str | Path,
    *,
    input_path: str | None = None,
    dataset_id: str = DATASET_ID,
    revision: str | None = None,
) -> dict[str, Any]:
    """Create a curation queue; never hallucinate trajectories from QA answers."""
    output_dir = Path(output_dir)
    rows = _load_rows(input_path, dataset_id, revision)
    required = {"index", "name", "category", "input", "expected"}
    queue = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"MARRVEL row missing fields {missing}: {row}")
        task_id = f"marrvel-{int(row['index']):03d}"
        queue.append(
            {
                "task_id": task_id,
                "name": row["name"],
                "category": row["category"],
                "query": row["input"],
                "expected_answer": row["expected"],
                "initial_state": {},
                "goal": {},
                "private_verifier": {},
                "available_tools": [],
                "gold_tool_ids": [],
                "background_ids": [],
                "split_group": None,
                "curation_status": "needs_typed_state_and_executable_trajectory",
                "provenance": {
                    "dataset_id": dataset_id,
                    "dataset_revision": revision or "UNPINNED_DO_NOT_PUBLISH",
                    "source_index": row["index"],
                },
            }
        )
    write_jsonl(output_dir / "marrvel_curation_queue.jsonl", queue)
    manifest = {
        "dataset_id": dataset_id,
        "revision": revision,
        "rows": len(queue),
        "content_sha256": sha256_text("\n".join(canonical_json(row) for row in queue)),
        "warning": (
            "The public benchmark is QA-only. initial_state, public goal, private_verifier, "
            "available_tools, and gold_tool_ids must be manually curated and "
            "execution-validated before training."
        ),
    }
    write_json(output_dir / "marrvel_import_manifest.json", manifest)
    return manifest


def finalize_curation(queue_path: str | Path, raw_dir: str | Path) -> dict[str, int]:
    raw_dir = Path(raw_dir)
    rows = read_jsonl(queue_path)
    tasks = []
    trajectories = []
    reference_answers = []
    verifiers = []
    errors = []
    for row in rows:
        if row.get("curation_status") != "validated":
            continue
        required = {
            "task_id": str,
            "query": str,
            "category": str,
            "expected_answer": str,
            "initial_state": dict,
            "goal": dict,
            "available_tools": list,
            "gold_tool_ids": list,
            "private_verifier": dict,
        }
        missing = [
            field
            for field, expected_type in required.items()
            if field not in row or not isinstance(row[field], expected_type)
        ]
        if missing:
            errors.append(f"{row.get('task_id')}: missing or incorrectly typed {missing}")
            continue
        if not row["goal"] or not row["private_verifier"] or not row["available_tools"]:
            errors.append(
                f"{row.get('task_id')}: goal, private_verifier, and available_tools must be non-empty"
            )
            continue
        try:
            validate_verifier(
                {"task_id": row["task_id"], "private_verifier": row["private_verifier"]}
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{row.get('task_id')}: invalid private_verifier: {exc}")
            continue
        task = {
            "task_id": row["task_id"],
            "query": row["query"],
            "category": row["category"],
            "initial_state": row["initial_state"],
            "goal": row["goal"],
            "available_tools": row["available_tools"],
            "background_ids": row.get("background_ids", []),
            "split": row.get("split"),
            "provenance": row.get("provenance", {}),
        }
        if row.get("split_group") is not None:
            if not isinstance(row["split_group"], str) or not row["split_group"].strip():
                errors.append(f"{row.get('task_id')}: split_group must be a non-empty string")
                continue
            task["split_group"] = row["split_group"]
        tasks.append(task)
        trajectories.append({"task_id": row["task_id"], "tool_ids": row["gold_tool_ids"]})
        verifiers.append(
            {
                "task_id": row["task_id"],
                "private_verifier": row["private_verifier"],
                "provenance": row.get("provenance", {}),
            }
        )
        reference_answers.append(
            {
                "task_id": row["task_id"],
                "expected_answer": row.get("expected_answer"),
                "visibility": "evaluation_only_not_model_input",
            }
        )
    if errors:
        raise ValueError("Invalid validated rows:\n" + "\n".join(errors))
    if not tasks:
        raise ValueError("No rows have curation_status='validated'")
    write_jsonl(raw_dir / "tasks.jsonl", tasks)
    write_jsonl(raw_dir / "trajectories.jsonl", trajectories)
    write_jsonl(raw_dir / "verifiers.private.jsonl", verifiers)
    write_jsonl(raw_dir / "reference_answers.private.jsonl", reference_answers)
    return {
        "tasks": len(tasks),
        "trajectories": len(trajectories),
        "private_verifiers": len(verifiers),
        "private_reference_answers": len(reference_answers),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import or finalize MARRVEL QA curation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import")
    importer.add_argument("--output-dir", required=True)
    importer.add_argument("--input")
    importer.add_argument("--dataset-id", default=DATASET_ID)
    importer.add_argument("--revision")
    finalizer = subparsers.add_parser("finalize")
    finalizer.add_argument("--queue", required=True)
    finalizer.add_argument("--raw-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "import":
        result = import_marrvel(
            args.output_dir,
            input_path=args.input,
            dataset_id=args.dataset_id,
            revision=args.revision,
        )
    else:
        result = finalize_curation(args.queue, args.raw_dir)
    print(result)


if __name__ == "__main__":
    main()

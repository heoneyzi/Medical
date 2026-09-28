from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from geoflowagent.constants import PAD_TOOL_ID, STOP_TOOL_ID
from geoflowagent.data.contracts import ContractEngine
from geoflowagent.data.preprocess import prepare_dataset
from geoflowagent.utils.io import canonical_json, read_json, read_jsonl, sha256_text, write_jsonl


def test_require_optimal_demo_fails_when_search_depth_cannot_prove_it(
    smoke_raw_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Could not verify an optimal demonstration"):
        prepare_dataset(
            smoke_raw_dir,
            tmp_path / "processed",
            max_search_depth=1,
            require_optimal_demo=True,
            counterfactual_policy="symbolic",
        )


def test_preprocessing_emits_one_explicit_terminal_row_per_task(processed_dir: Path) -> None:
    tools = read_jsonl(processed_dir / "tools.jsonl")
    tasks = read_jsonl(processed_dir / "tasks.jsonl")
    examples = read_jsonl(processed_dir / "examples.jsonl")
    manifest = read_json(processed_dir / "manifest.json")
    tool_ids = {row["tool_id"] for row in tools}

    assert STOP_TOOL_ID not in tool_ids
    assert PAD_TOOL_ID not in tool_ids
    assert manifest["counts"]["tasks"] == len(tasks)
    assert manifest["counts"]["terminal_examples"] == len(tasks)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in examples:
        by_task[row["task_id"]].append(row)
        assert STOP_TOOL_ID not in row["gold_suffix_tool_ids"]
        assert PAD_TOOL_ID not in row["gold_suffix_tool_ids"]

    assert set(by_task) == {row["task_id"] for row in tasks}
    for rows in by_task.values():
        rows.sort(key=lambda row: row["phase"])
        terminal = [row for row in rows if row["terminal"]]
        assert len(terminal) == 1
        assert terminal[0] is rows[-1]
        assert terminal[0]["gold_next_tool"] == STOP_TOOL_ID
        assert terminal[0]["gold_suffix_tool_ids"] == []
        assert terminal[0]["valid_next_tools"] == [STOP_TOOL_ID]
        assert terminal[0]["contract_valid_tools"] == []
        assert terminal[0]["state"] == terminal[0]["next_state"]
        assert [row["phase"] for row in rows] == list(range(1, len(rows) + 1))


def test_task_splits_are_disjoint_and_do_not_fragment_trajectories(processed_dir: Path) -> None:
    examples = read_jsonl(processed_dir / "examples.jsonl")
    manifest = read_json(processed_dir / "manifest.json")
    split_sets = {split: set(task_ids) for split, task_ids in manifest["task_ids_by_split"].items()}

    assert split_sets["train"].isdisjoint(split_sets["dev"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["dev"].isdisjoint(split_sets["test"])
    for task_id in set().union(*split_sets.values()):
        example_splits = {row["split"] for row in examples if row["task_id"] == task_id}
        assert len(example_splits) == 1
        assert task_id in split_sets[next(iter(example_splits))]


def test_declared_split_group_cannot_cross_research_splits(
    smoke_raw_dir: Path, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(smoke_raw_dir, raw)
    tasks = read_jsonl(raw / "tasks.jsonl")
    left = next(row for row in tasks if row["split"] == "train")
    right = next(row for row in tasks if row["split"] == "test")
    left["split_group"] = "same-biological-case"
    right["split_group"] = "same-biological-case"
    write_jsonl(raw / "tasks.jsonl", tasks)

    with pytest.raises(ValueError, match="Split group.*crosses splits"):
        prepare_dataset(raw, tmp_path / "processed")


def test_minimal_pair_must_match_its_source_task_split(smoke_raw_dir: Path, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(smoke_raw_dir, raw)
    tasks = read_jsonl(raw / "tasks.jsonl")
    train_task = next(row for row in tasks if row["split"] == "train")
    pairs = read_jsonl(raw / "minimal_pairs.jsonl")
    pairs[0]["source_task_id"] = train_task["task_id"]
    write_jsonl(raw / "minimal_pairs.jsonl", pairs)

    with pytest.raises(ValueError, match="crosses its source task split"):
        prepare_dataset(raw, tmp_path / "processed")


def test_prefix_examples_replay_without_future_state_or_observation_leakage(
    processed_dir: Path,
) -> None:
    tools = read_jsonl(processed_dir / "tools.jsonl")
    tasks = {row["task_id"]: row for row in read_jsonl(processed_dir / "tasks.jsonl")}
    examples = read_jsonl(processed_dir / "examples.jsonl")
    engine = ContractEngine(tools)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in examples:
        by_task[row["task_id"]].append(row)
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        assert row["record_sha256"] == sha256_text(canonical_json(unsigned))

    for task_id, rows in by_task.items():
        rows.sort(key=lambda row: row["phase"])
        state = tasks[task_id]["initial_state"]
        history: list[dict] = []
        prefix: list[str] = []
        for row in rows:
            assert row["state"] == state
            assert row["history"] == history
            assert row["prefix_tool_ids"] == prefix
            if row["terminal"]:
                continue
            assert row["gold_suffix_tool_ids"][0] == row["gold_next_tool"]
            transition = engine.execute(state, row["gold_next_tool"])
            assert row["next_state"] == transition.state
            state = transition.state
            history = [*history, transition.observation]
            prefix = [*prefix, row["gold_next_tool"]]

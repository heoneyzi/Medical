from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.search_paths import SearchPathStore


def _action(tool_id: str, successor_id: str, cost: float) -> dict:
    return {
        "tool_id": tool_id,
        "known": True,
        "label_mask": True,
        "transition_known": True,
        "executable": True,
        "reachable": True,
        "successor_id": successor_id,
        "cost": cost,
    }


def _row(
    task_id: str,
    state_id: str,
    *,
    value_star: float | None,
    actions: list[dict],
    paths: list[dict],
    terminal: bool = False,
    reachable: bool = True,
    training: bool = True,
) -> dict:
    return {
        "example_id": f"{task_id}:search:{state_id}",
        "task_id": task_id,
        "state_id": state_id,
        "split": "train",
        "terminal": terminal,
        "training_mask": training,
        "goal_reachable": reachable,
        "value_star": value_star,
        "action_supervision": actions,
        "top_k_paths": paths,
    }


def _store() -> SearchFeatureStore:
    rows = [
        _row(
            "task-a",
            "s0",
            value_star=2.0,
            actions=[_action("fast", "s1", 1.0), _action("detour", "s2", 2.0)],
            paths=[
                {
                    "tool_ids": ["fast", "finish"],
                    "state_ids": ["s0", "s1", "sg"],
                    "total_cost": 2.0,
                },
                {
                    "tool_ids": ["detour", "finish"],
                    "state_ids": ["s0", "s2", "sg"],
                    "total_cost": 3.0,
                },
            ],
        ),
        _row(
            "task-a",
            "s1",
            value_star=1.0,
            actions=[_action("finish", "sg", 1.0)],
            paths=[
                {
                    "tool_ids": ["finish"],
                    "state_ids": ["s1", "sg"],
                    "total_cost": 1.0,
                }
            ],
        ),
        _row(
            "task-a",
            "s2",
            value_star=1.0,
            actions=[_action("finish", "sg", 1.0)],
            paths=[
                {
                    "tool_ids": ["finish"],
                    "state_ids": ["s2", "sg"],
                    "total_cost": 1.0,
                }
            ],
        ),
        _row(
            "task-a",
            "sg",
            value_star=0.0,
            actions=[],
            paths=[{"tool_ids": [], "state_ids": ["sg"], "total_cost": 0.0}],
            terminal=True,
        ),
        _row(
            "task-a",
            "dead",
            value_star=None,
            actions=[],
            paths=[],
            reachable=False,
            training=False,
        ),
        _row(
            "task-b",
            "u0",
            value_star=1.0,
            actions=[_action("finish", "ug", 1.0)],
            paths=[
                {
                    "tool_ids": ["finish"],
                    "state_ids": ["u0", "ug"],
                    "total_cost": 1.0,
                }
            ],
        ),
        _row(
            "task-b",
            "ug",
            value_star=0.0,
            actions=[],
            paths=[{"tool_ids": [], "state_ids": ["ug"], "total_cost": 0.0}],
            terminal=True,
        ),
    ]
    tool_ids = ["detour", "fast", "finish"]
    value = SimpleNamespace(
        examples=rows,
        tool_ids=tool_ids,
        tool_index={tool_id: index for index, tool_id in enumerate(tool_ids)},
    )
    return cast(SearchFeatureStore, value)


def test_path_store_validates_and_reports_complete_references() -> None:
    store = _store()
    paths = SearchPathStore(store, beta=1.0)
    root = paths.references(0)

    assert [reference.tool_ids for reference in root] == [
        ("fast", "finish"),
        ("detour", "finish"),
    ]
    assert root[0].state_indices == (0, 1, 3)
    assert root[0].tool_indices == (1, 2)
    assert root[0].excess_cost == 0.0
    assert root[1].excess_cost == 1.0
    assert root[0].probability == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert sum(reference.probability for reference in root) == pytest.approx(1.0)

    report = paths.report("train")
    assert report["eligible_tasks"] == 2
    assert report["eligible_states"] == 4
    assert report["references"] == 5
    assert report["multi_path_states"] == 1
    assert report["cost_diverse_states"] == 1
    assert report["max_horizon"] == 2
    assert report["excluded_rows"] == {"terminal": 2, "unreachable_or_unknown": 1}


def test_path_store_rejects_successor_chain_mismatch() -> None:
    store = _store()
    store.examples[0]["top_k_paths"][0]["state_ids"] = ["s0", "s2", "sg"]

    with pytest.raises(ValueError, match="chain mismatch"):
        SearchPathStore(store)


def test_path_store_rejects_cost_mismatch() -> None:
    store = _store()
    store.examples[0]["top_k_paths"][0]["total_cost"] = 2.5

    with pytest.raises(ValueError, match="cost mismatch"):
        SearchPathStore(store)


def test_mult_path_sampling_is_weighted_reproducible_and_task_balanced() -> None:
    store = _store()
    paths = SearchPathStore(store, beta=1.0)

    first_rng = random.Random(17)
    second_rng = random.Random(17)
    first = [paths.sample_reference(0, first_rng).tool_ids for _ in range(100)]
    second = [paths.sample_reference(0, second_rng).tool_ids for _ in range(100)]
    assert first == second

    torch_a = torch.Generator(device="cpu").manual_seed(23)
    torch_b = torch.Generator(device="cpu").manual_seed(23)
    from_torch_a = [paths.sample_reference(0, torch_a).tool_ids for _ in range(100)]
    from_torch_b = [paths.sample_reference(0, torch_b).tool_ids for _ in range(100)]
    assert from_torch_a == from_torch_b

    weighted_rng = random.Random(29)
    draws = [paths.sample_reference(0, weighted_rng).tool_ids[0] for _ in range(5_000)]
    fast_fraction = draws.count("fast") / len(draws)
    assert fast_fraction == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), abs=0.025)

    # task-a has three eligible states and task-b has one. Hierarchical sampling
    # should nevertheless choose the two tasks equally often instead of 3:1.
    balanced_rng = random.Random(31)
    sampled_tasks = [
        store.examples[paths.sample_state_index("train", balanced_rng)]["task_id"]
        for _ in range(4_000)
    ]
    task_a_fraction = sampled_tasks.count("task-a") / len(sampled_tasks)
    assert task_a_fraction == pytest.approx(0.5, abs=0.04)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_PROCESSED = _REPOSITORY_ROOT / "artifacts/search_smoke/processed"
_SMOKE_CACHE = _REPOSITORY_ROOT / "artifacts/search_smoke/cache"


@pytest.mark.skipif(
    not (_SMOKE_PROCESSED / "manifest.json").exists()
    or not (_SMOKE_CACHE / "manifest.json").exists(),
    reason="optional generated search smoke artifacts are not present",
)
def test_generated_search_smoke_paths_are_flow_ready() -> None:
    store = SearchFeatureStore(_SMOKE_PROCESSED, _SMOKE_CACHE, structured_dim=32)
    paths = SearchPathStore(store, beta=1.0)
    report = paths.report()

    active_tasks = {
        row["task_id"]
        for row in store.examples
        if row.get("terminal") is not True
        and row.get("goal_reachable") is True
        and row.get("training_mask") is True
    }
    assert {row["split"] for row in store.examples} == {"train", "dev"}
    assert report["eligible_tasks"] == len(active_tasks) == 14
    assert report["eligible_states"] > 0
    assert report["references"] >= report["eligible_states"]
    assert report["multi_path_states"] > 0
    assert report["max_horizon"] >= 4


def test_store_construction_does_not_mutate_source_rows() -> None:
    store = _store()
    before = copy.deepcopy(store.examples)
    SearchPathStore(store, beta=0.5)
    assert store.examples == before


def test_allowed_splits_do_not_read_sealed_path_labels() -> None:
    store = _store()
    sealed = copy.deepcopy(store.examples[0])
    sealed["example_id"] = "sealed:search:s0"
    sealed["task_id"] = "sealed"
    sealed["split"] = "test"
    # This is deliberately malformed. A train/dev-only adapter must skip it
    # before looking at top-k paths, oracle values, or reachability labels.
    sealed["top_k_paths"] = "DO NOT READ"
    store.examples.append(sealed)

    paths = SearchPathStore(store, allowed_splits=("train", "dev"))

    assert paths.report()["excluded_rows"]["split_not_loaded"] == 1
    assert all(
        reference.split != "test"
        for index in paths.eligible_indices()
        for reference in paths.references(index)
    )

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from geoflowagent.constants import CACHE_VERSION, FEATURE_SPEC_VERSION, SCHEMA_VERSION
from geoflowagent.training import features as demonstration_features
from geoflowagent.training.features import FeatureStore
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.utils.io import sha256_file, write_json, write_jsonl


def _action(
    tool_id: str,
    *,
    q: float | None = None,
    cost: float | None = None,
    regret: float | None = None,
    q_known: bool = False,
    transition_known: bool = False,
    transition_labeled: bool = False,
    reachability: bool | None = None,
    reachability_labeled: bool = False,
    executable: bool | None = None,
    successor_id: str | None = None,
    successor: dict | None = None,
    optimal: bool | None = None,
) -> dict:
    return {
        "tool_id": tool_id,
        "known": q_known,
        "label_mask": q_known,
        "q_known_mask": q_known,
        "transition_known": transition_known,
        "transition_known_mask": transition_labeled,
        "reachability_known_mask": reachability_labeled,
        "contract_applicable": tool_id != "blocked",
        "executable": executable,
        "successor": successor,
        "successor_id": successor_id,
        "cost": cost,
        "q_star": q,
        "regret": regret,
        "is_optimal": optimal,
        "reachable": reachability,
    }


def _write_array(root: Path, relative: str, array: np.ndarray, files: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    files[relative] = {
        "rows": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "dtype": str(array.dtype),
        "sha256": sha256_file(path),
    }


@pytest.fixture()
def search_store(tmp_path: Path) -> SearchFeatureStore:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    processed.mkdir()
    cache.mkdir()
    tool_ids = ["advance", "uncertain", "blocked", "mystery"]
    tools = [
        {
            "tool_id": tool_id,
            "name": tool_id.title(),
            "description": f"Tool {tool_id}",
            "preconditions": [],
            "effects": [{"field": tool_id, "op": "set", "value": True}],
        }
        for tool_id in tool_ids
    ]
    tasks = [{"task_id": "task-1", "query": "case", "goal": {"done": True}}]
    root_actions = {
        "advance": _action(
            "advance",
            q=2.0,
            cost=1.0,
            regret=0.0,
            q_known=True,
            transition_known=True,
            transition_labeled=True,
            reachability=True,
            reachability_labeled=True,
            executable=True,
            successor_id="s1",
            successor={"phase": 1, "sequence_context": "CG"},
            optimal=True,
        ),
        # The executor knows this transition and edge cost, but search did not
        # certify its Q/regret.  It must remain out of the Q loss denominator.
        "uncertain": _action(
            "uncertain",
            cost=0.5,
            transition_known=True,
            transition_labeled=True,
            executable=True,
            successor_id="s1",
            successor={"phase": 1, "sequence_context": "CG"},
        ),
        "blocked": _action(
            "blocked",
            transition_known=False,
            transition_labeled=True,
            reachability=False,
            reachability_labeled=True,
            executable=False,
            optimal=False,
        ),
        "mystery": _action("mystery"),
    }
    null_map = {tool_id: None for tool_id in tool_ids}
    false_map = {tool_id: False for tool_id in tool_ids}
    examples = [
        {
            "example_id": "task-1:search:s0",
            "task_id": "task-1",
            "state_id": "s0",
            "split": "train",
            "terminal": False,
            "training_mask": True,
            "query": "case query",
            "goal": {"done": True},
            "state": {"phase": 0, "sequence_context": "AC"},
            "history": [],
            "candidate_tools": tool_ids,
            "contract_valid_tools": ["advance", "uncertain", "mystery"],
            "optimal_actions": ["advance"],
            "valid_next_tools": ["advance"],
            "action_supervision": root_actions,
            "value_star": 2.0,
            "value_known_mask": True,
            "goal_reachable": True,
            "goal_reachability_known_mask": True,
            "background_ids": ["card-1"],
        },
        {
            "example_id": "task-1:search:s1",
            "task_id": "task-1",
            "state_id": "s1",
            "split": "train",
            "terminal": False,
            "training_mask": True,
            "query": "case query",
            "goal": {"done": True},
            "state": {"phase": 1, "sequence_context": "CG"},
            "history": [{"tool_id": "advance", "ok": True}],
            "candidate_tools": tool_ids,
            "contract_valid_tools": ["advance"],
            "optimal_actions": ["advance"],
            "valid_next_tools": ["advance"],
            # Flat maps exercise compatibility with compiler-adjacent formats.
            "action_known_mask": {"advance": True, **{key: False for key in tool_ids[1:]}},
            "action_q": {"advance": 1.0, **{key: None for key in tool_ids[1:]}},
            "action_cost": {"advance": 1.0, **{key: None for key in tool_ids[1:]}},
            "action_regret": {"advance": 0.0, **{key: None for key in tool_ids[1:]}},
            "action_successor_ids": {"advance": "s2", **{key: None for key in tool_ids[1:]}},
            "value_star": 1.0,
            "value_known_mask": True,
            "goal_reachable": True,
            "goal_reachability_known_mask": True,
            "background_ids": [],
        },
        {
            "example_id": "task-1:search:s2",
            "task_id": "task-1",
            "state_id": "s2",
            "split": "train",
            "terminal": True,
            "training_mask": True,
            "query": "case query",
            "goal": {"done": True},
            "state": {"phase": 2, "done": True, "sequence_context": "GT"},
            "history": [{"tool_id": "advance", "ok": True}],
            "candidate_tools": tool_ids,
            "contract_valid_tools": [],
            "optimal_actions": [],
            "valid_next_tools": ["<STOP>"],
            "action_known_mask": false_map,
            "action_q": null_map,
            "action_cost": null_map,
            "action_regret": null_map,
            "action_successor_ids": null_map,
            "value_star": 0.0,
            "value_known_mask": True,
            "goal_reachable": True,
            "goal_reachability_known_mask": True,
            "background_ids": [],
        },
    ]
    background = [{"card_id": "card-1", "title": "context", "text": "background"}]
    write_jsonl(processed / "tools.jsonl", tools)
    write_jsonl(processed / "tasks.jsonl", tasks)
    write_jsonl(processed / "examples.jsonl", examples)
    write_jsonl(processed / "background.jsonl", background)
    processed_files = {
        name: sha256_file(processed / name)
        for name in ("tools.jsonl", "tasks.jsonl", "examples.jsonl", "background.jsonl")
    }
    write_json(
        processed / "manifest.json",
        {"schema_version": SCHEMA_VERSION, "processed_files": processed_files},
    )

    # Values are deliberately easy to distinguish in pooling assertions.
    query = np.array([[1.0, 0.0], [8.0, 0.0], [3.0, 0.0]])
    goal = np.array([[0.0, 7.0], [0.0, 8.0], [0.0, 9.0]])
    state = np.array([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    history = np.array([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]])
    tool = np.arange(8, dtype=np.float32).reshape(4, 2)
    sequence = np.array([[11.0, 1.0], [12.0, 2.0], [13.0, 3.0]])
    files: dict = {}
    for field, values in {
        "query": query,
        "goal": goal,
        "state": state,
        "history": history,
    }.items():
        _write_array(cache, f"general/examples.{field}.npy", values, files)
    _write_array(cache, "general/tools.description.npy", tool, files)
    _write_array(cache, "general/background.text.npy", np.array([[9.0, 0.0]]), files)
    _write_array(cache, "sequence_aux/examples.sequence.npy", sequence, files)
    _write_array(cache, "sequence_aux/examples.next_sequence.npy", sequence, files)
    write_json(
        cache / "manifest.json",
        {
            "cache_version": CACHE_VERSION,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "config_sha256": "fixture",
            "source_manifest_sha256": sha256_file(processed / "manifest.json"),
            "counts": {"examples": 3, "tools": 4, "background": 1},
            "indices": {
                "examples": [row["example_id"] for row in examples],
                "tools": tool_ids,
                "background": ["card-1"],
                "minimal_pairs": [],
            },
            "views": {
                "general": {
                    "kind": "prototype",
                    "example_fields": ["query", "goal", "state", "history"],
                    "tool_fields": ["description"],
                    "dim": 2,
                },
                "sequence_aux": {
                    "kind": "state_only",
                    "example_fields": ["sequence", "next_sequence"],
                    "tool_fields": [],
                    "snapshot_field": "sequence",
                    "dim": 2,
                },
            },
            "files": files,
            "rules": {"backbones_frozen": True},
        },
    )
    return SearchFeatureStore(processed, cache, structured_dim=8)


def test_separate_context_goal_and_zero_filled_genomic_views(
    search_store: SearchFeatureStore,
) -> None:
    batch = search_store.batch([0])

    assert search_store.views == ("general", "sequence_aux")
    assert search_store.view_roles["general"]["goal_zero_filled"] is False
    assert search_store.view_roles["sequence_aux"] == {
        "kind": "state_only",
        "state_semantics": "snapshot:sequence",
        "snapshot_field": "sequence",
        "goal_zero_filled": True,
        "tool_zero_filled": True,
    }
    # (query + state + history + mean(background)) / 4; goal is not included.
    torch.testing.assert_close(batch["state_views"]["general"], torch.tensor([[4.5, 0.0]]))
    torch.testing.assert_close(batch["goal_views"]["general"], torch.tensor([[0.0, 7.0]]))
    torch.testing.assert_close(
        batch["state_snapshot_views"]["general"], torch.tensor([[3.0, 0.0]])
    )
    torch.testing.assert_close(
        batch["state_views"]["sequence_aux"], torch.tensor([[11.0, 1.0]])
    )
    assert torch.count_nonzero(batch["goal_views"]["sequence_aux"]) == 0
    assert torch.count_nonzero(batch["tool_views"]["sequence_aux"]) == 0
    assert search_store.all_view_dims == {"general": 2, "sequence_aux": 2}


def test_q_transition_reachability_and_unknown_masks_are_distinct(
    search_store: SearchFeatureStore,
) -> None:
    batch = search_store.batch([0])
    advance, uncertain, blocked, mystery = range(4)

    assert batch["known_action_mask"][0].tolist() == [True, False, False, False]
    assert batch["transition_known_mask"][0].tolist() == [True, True, True, False]
    assert batch["transition_observed_target"][0, advance].item() == 1.0
    assert batch["transition_observed_target"][0, uncertain].item() == 1.0
    assert batch["transition_observed_target"][0, blocked].item() == 0.0
    assert torch.isnan(batch["transition_observed_target"][0, mystery])
    assert batch["action_reachability_mask"][0].tolist() == [True, False, True, False]
    assert batch["action_reachability_target"][0, blocked].item() == 0.0

    assert batch["q_star"][0, advance].item() == 2.0
    assert torch.isnan(batch["q_star"][0, uncertain])
    assert batch["edge_cost_mask"][0].tolist() == [True, True, False, False]
    assert batch["regret_mask"][0].tolist() == [True, False, False, False]
    assert batch["optimal_action_mask"][0].tolist() == [True, False, False, False]
    assert batch["policy_candidate_mask"][0].tolist() == [True, True, False, True]
    assert batch["known_candidate_mask"][0].tolist() == [True, False, False, False]


def test_successor_targets_and_stop_are_separate_from_real_tools(
    search_store: SearchFeatureStore,
) -> None:
    root = search_store.batch([0])
    terminal = search_store.batch([2])

    assert root["tool_views"]["general"].shape[0] == 4
    assert "<STOP>" not in root["tool_ids"]
    assert root["successor_index"][0, 0].item() == 1
    assert root["successor_feature_mask"][0, 0]
    # Per-action transition supervision uses the successor snapshot, not its
    # query/history contextual average.
    torch.testing.assert_close(
        root["successor_state_views"]["general"][0, 0], torch.tensor([4.0, 0.0])
    )
    assert not torch.equal(
        root["successor_context_views"]["general"][0, 0],
        root["successor_state_views"]["general"][0, 0],
    )
    assert torch.isnan(root["successor_state_views"]["general"][0, 3]).all()
    assert root["successor_value_target"][0, 0].item() == 1.0

    assert terminal["completion_target"].item() == 1.0
    assert terminal["stop_target"].item() == 1.0
    assert terminal["stop_mask"].item()
    assert not terminal["policy_candidate_mask"].any()


def test_split_and_batch_index_validation(search_store: SearchFeatureStore) -> None:
    assert search_store.indices("train") == [0, 1, 2]
    assert search_store.indices("train", include_terminal=False) == [0, 1]
    assert search_store.indices("dev") == []
    with pytest.raises(IndexError, match="outside"):
        search_store.batch([99])
    with pytest.raises(TypeError, match="integers"):
        search_store.batch([True])


def test_feature_ablation_is_capacity_matched_and_applies_to_runtime(
    search_store: SearchFeatureStore,
) -> None:
    original_structured = search_store.structured_state.clone()
    search_store.apply_feature_ablation(
        {"zero_views": ["general"], "zero_structured": True}
    )
    batch = search_store.batch([0])

    assert search_store.view_dims == {"general": 2, "sequence_aux": 2}
    assert search_store.feature_ablation == {
        "zero_views": ["general"],
        "zero_structured": True,
    }
    assert torch.count_nonzero(batch["state_views"]["general"]) == 0
    assert torch.count_nonzero(batch["tool_views"]["general"]) == 0
    assert torch.count_nonzero(batch["state_views"]["sequence_aux"]) > 0
    assert torch.count_nonzero(batch["structured_state"]) == 0
    assert torch.count_nonzero(original_structured) > 0

    state, goal, structured_state, structured_goal = search_store.ablate_runtime_inputs(
        {"general": torch.ones(1, 2), "sequence_aux": torch.ones(1, 2)},
        {"general": torch.ones(1, 2), "sequence_aux": torch.ones(1, 2)},
        torch.ones(1, 8),
        torch.ones(1, 8),
    )
    assert torch.count_nonzero(state["general"]) == 0
    assert torch.count_nonzero(goal["general"]) == 0
    assert torch.count_nonzero(state["sequence_aux"]) > 0
    assert torch.count_nonzero(structured_state) == 0
    assert torch.count_nonzero(structured_goal) == 0


def test_demonstration_feature_store_rejects_search_dataset_before_mislabeling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        demonstration_features,
        "verify_processed_dataset",
        lambda _path: {
            "dataset_kind": "weighted_search_supervision",
            "processed_files": {"examples.jsonl": "unused"},
        },
    )

    with pytest.raises(ValueError, match="cannot consume weighted search supervision"):
        FeatureStore(tmp_path / "processed", tmp_path / "cache")

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from geoflowagent.training.dagger import _dev_selection_score, rollout_search_graph
from geoflowagent.training.value import _task_balanced_epoch


class _GraphStore:
    def __init__(self) -> None:
        self.tool_ids = ["good", "bad"]
        self.examples = [
            {
                "example_id": "root",
                "task_id": "task-1",
                "depth": 0,
                "terminal": False,
                "record_sha256": "root-hash",
            },
            {
                "example_id": "goal",
                "task_id": "task-1",
                "depth": 1,
                "terminal": True,
                "record_sha256": "goal-hash",
            },
        ]

    def indices(self, split: str) -> list[int]:
        return [0, 1] if split == "train" else []

    def batch(self, indices: list[int], device: torch.device) -> dict:
        index = indices[0]
        terminal = index == 1
        candidate = torch.tensor([[not terminal, not terminal]], device=device)
        return {
            "rows": [self.examples[index]],
            "state_views": {},
            "goal_views": {},
            "tool_views": {},
            "structured_state": torch.tensor([[float(index)]], device=device),
            "structured_goal": torch.zeros((1, 1), device=device),
            "structured_tools": torch.zeros((2, 1), device=device),
            "policy_candidate_mask": candidate,
            "known_action_mask": torch.tensor([[True, True]], device=device),
            "optimal_action_mask": torch.tensor([[True, False]], device=device),
            "regret": torch.tensor([[0.0, 1.0]], device=device),
            "successor_index": torch.tensor([[1, -1]], device=device),
        }


class _FixedPolicy(torch.nn.Module):
    def __init__(self, selected: int) -> None:
        super().__init__()
        self.selected = selected

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        del args
        candidate = kwargs["candidate_mask"]
        device = candidate.device
        q = torch.tensor([[0.0, 1.0]], device=device)
        if self.selected == 1:
            q = q.flip(1)
        logits = -q
        logits = logits.masked_fill(~candidate, -torch.inf)
        return {
            "completion_logit": torch.tensor([-10.0], device=device),
            "policy_logits": logits,
            "q": q,
            "action_reachability_logit": torch.full((1, 2), 10.0, device=device),
        }


def test_graph_dagger_visits_learner_distribution_and_keeps_exact_labels() -> None:
    report = rollout_search_graph(
        _FixedPolicy(0),
        _GraphStore(),
        torch.device("cpu"),
        completion_threshold=0.5,
        reachability_threshold=0.5,
        max_steps=3,
    )

    assert report["success_rate"] == 1.0
    assert report["visited_example_ids"] == ["root", "goal"]
    assert report["hard_example_ids"] == []
    assert report["episodes"][0]["trace"][0]["record_sha256"] == "root-hash"


def test_graph_dagger_marks_suboptimal_unknown_successor_as_hard() -> None:
    report = rollout_search_graph(
        _FixedPolicy(1),
        _GraphStore(),
        torch.device("cpu"),
        completion_threshold=0.5,
        reachability_threshold=0.5,
        max_steps=3,
    )

    assert report["success_rate"] == 0.0
    assert report["hard_example_ids"] == ["root"]
    assert report["failure_counts"] == {"unknown_successor": 1}


def test_external_replay_is_additive_to_task_balanced_epoch() -> None:
    store = SimpleNamespace(examples=[{"task_id": "a"}, {"task_id": "b"}])
    selected = _task_balanced_epoch(
        store,
        [0, 1],
        random.Random(17),
        max_states_per_task=1,
        hard_pool=(),
        hard_replay_fraction=0.0,
        external_replay_pool=[1],
        external_replay_fraction=0.5,
    )

    assert len(selected) == 3
    assert selected.count(1) == 2


def test_dagger_dev_score_matches_registered_scalarization() -> None:
    score = _dev_selection_score(
        {
            "joint_stop_action_accuracy": 0.8,
            "gated_optimal_set_accuracy": 0.6,
            "completion": {"balanced_accuracy": 0.5},
            "action_reachability": {"balanced_accuracy": 0.4},
            "state_reachability": {"balanced_accuracy": 0.2},
        }
    )
    assert score == pytest.approx(0.54)

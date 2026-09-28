from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.models.value_geometry import GoalConditionedValueGeometry
from geoflowagent.training import value as value_training
from geoflowagent.training.value import (
    _balanced_bce,
    _capture_global_rng_state,
    _prepare_evaluation_batches,
    _restore_global_rng_state,
    _target_scale,
    compare_existing_value_run_groups,
    compare_value_capacities,
    compare_value_geometries,
    load_value_geometry_checkpoint,
)
from geoflowagent.utils.io import write_json


def test_value_resume_restores_python_numpy_and_torch_rngs() -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = _capture_global_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    _restore_global_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[:2] == pytest.approx(expected[:2])
    torch.testing.assert_close(actual[2], expected[2])


def test_prepared_evaluation_batches_drop_wide_successor_inputs() -> None:
    class Store:
        @staticmethod
        def indices(split: str) -> list[int]:
            assert split == "dev"
            return [0]

        @staticmethod
        def batch(indices: list[int], device: torch.device) -> dict:
            assert indices == [0]
            actions = 2
            return {
                "rows": [{"task_id": "task-0"}],
                "state_views": {"view": torch.zeros(1, 3, device=device)},
                "goal_views": {"view": torch.zeros(1, 3, device=device)},
                "tool_views": {"view": torch.zeros(actions, 3, device=device)},
                "structured_state": torch.zeros(1, 2, device=device),
                "structured_goal": torch.zeros(1, 2, device=device),
                "structured_tools": torch.zeros(actions, 2, device=device),
                "successor_context_views": {
                    "view": torch.zeros(1, actions, 4096, device=device)
                },
                "policy_candidate_mask": torch.ones(1, actions, dtype=torch.bool, device=device),
                "optimal_action_mask": torch.ones(1, actions, dtype=torch.bool, device=device),
                "completion_target": torch.zeros(1, device=device),
                "completion_mask": torch.ones(1, dtype=torch.bool, device=device),
                "regret": torch.zeros(1, actions, device=device),
                "regret_mask": torch.ones(1, actions, dtype=torch.bool, device=device),
                "q_star": torch.zeros(1, actions, device=device),
                "q_mask": torch.ones(1, actions, dtype=torch.bool, device=device),
                "value_target": torch.zeros(1, device=device),
                "value_mask": torch.ones(1, dtype=torch.bool, device=device),
                "action_reachability_mask": torch.ones(
                    1, actions, dtype=torch.bool, device=device
                ),
                "action_reachability_target": torch.ones(1, actions, device=device),
                "state_reachability_mask": torch.ones(1, dtype=torch.bool, device=device),
                "state_reachability_target": torch.ones(1, device=device),
            }

    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
            del args, kwargs
            return {
                "policy_logits": torch.zeros(1, 2),
                "q": torch.zeros(1, 2),
                "value": torch.zeros(1),
                "action_reachability_logit": torch.zeros(1, 2),
                "state_reachability_logit": torch.zeros(1),
                "completion_logit": torch.zeros(1),
                "state_latent": torch.zeros(1, 3),
                "goal_latent": torch.zeros(1, 3),
                "successor_latent": torch.zeros(1, 2, 3),
            }

    prepared = _prepare_evaluation_batches(Model(), Store(), "dev", torch.device("cpu"), 1)
    _, compact_batch, compact_output = prepared[0]

    assert "successor_context_views" not in compact_batch
    assert "state_views" not in compact_batch
    assert "successor_latent" not in compact_output
    assert compact_batch["rows"] == [{"task_id": "task-0"}]


def test_balanced_bce_equalizes_classes_and_ignores_masked_labels() -> None:
    logits = torch.tensor([-2.0, -1.0, 1.0, 50.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 0.0, float("nan")])
    mask = torch.tensor([True, True, True, False])

    loss = _balanced_bce(logits, targets, mask)
    positive_loss = F.binary_cross_entropy_with_logits(logits[:1], targets[:1])
    negative_loss = F.binary_cross_entropy_with_logits(logits[1:3], targets[1:3])

    torch.testing.assert_close(loss, (positive_loss + negative_loss) / 2)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[-1].item() == 0.0


def test_target_scale_uses_only_positive_finite_training_q_values() -> None:
    store = SimpleNamespace(
        _targets={
            "q_star": torch.tensor(
                [
                    [float("nan"), 0.0, 1.0],
                    [2.0, 100.0, float("nan")],
                    [10_000.0, 10_000.0, 10_000.0],
                ]
            )
        }
    )

    # The held-out third row cannot affect target normalization.
    assert _target_scale(store, [0, 1]) == pytest.approx(2.0)
    assert _target_scale(store, []) == pytest.approx(1.0)


def test_value_checkpoint_round_trip_reconstructs_exact_model(tmp_path) -> None:
    torch.manual_seed(7)
    original = GoalConditionedValueGeometry(
        {"general": 3, "biomedical": 2},
        structured_dim=4,
        shared_dim=5,
        hidden_dim=7,
        energy="directed_quasimetric",
        energy_rank=3,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "value_geometry.pt"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "kind": "search_value_geometry",
            "model_config": original.export_config(),
            "model_state": original.state_dict(),
        },
        checkpoint_path,
    )

    restored, checkpoint = load_value_geometry_checkpoint(checkpoint_path, "cpu")

    assert checkpoint["model_config"] == original.export_config()
    assert restored.training is False
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_geometry_tie_break_prefers_true_zero_regret(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 17\n", encoding="utf-8")

    def fake_train(*args, energy_override: str, **kwargs):
        del args, kwargs
        regret = 0.0 if energy_override == "zero-regret" else 1.0
        return {
            "parameter_count": 10,
            "checkpoint_sha256": f"hash-{energy_override}",
            "metrics": {
                "dev": {
                    "joint_stop_action_accuracy": 0.5,
                    "regret_at_1": regret,
                }
            },
        }

    monkeypatch.setattr(value_training, "train_value_geometry", fake_train)
    monkeypatch.setattr(value_training, "SearchFeatureStore", lambda *args, **kwargs: object())
    report = compare_value_geometries(
        tmp_path / "processed",
        tmp_path / "cache",
        tmp_path / "comparison",
        config_path,
        energies=["positive-regret", "zero-regret"],
        seeds=[17],
    )

    assert report["selected_energy"] == "zero-regret"
    assert report["aggregate"][0]["mean_dev_regret_at_1"] == 0.0


def test_geometry_comparison_reports_paired_task_effects(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 17\n", encoding="utf-8")

    def fake_train(*args, energy_override: str, seed_override: int, **kwargs):
        del args, kwargs
        gain = 0.2 if energy_override == "directed" else 0.0
        seed_shift = 0.01 if seed_override == 29 else 0.0
        return {
            "parameter_count": 10,
            "checkpoint_sha256": f"hash-{energy_override}-{seed_override}",
            "metrics": {
                "dev": {
                    "joint_stop_action_accuracy": 0.5 + gain,
                    "regret_at_1": 0.2 - gain,
                    "per_task_policy_accuracy": {
                        "task-a": 0.4 + gain + seed_shift,
                        "task-b": 0.6 + gain + seed_shift,
                    },
                    "per_task_joint_accuracy": {
                        "task-a": 0.3 + gain + seed_shift,
                        "task-b": 0.5 + gain + seed_shift,
                    },
                    "per_task_regret_at_1": {
                        "task-a": 0.4 - gain + seed_shift,
                        "task-b": 0.2 - gain + seed_shift,
                    },
                    "latent_geometry": {
                        "mean_relative_asymmetry": gain + seed_shift,
                        "energy_family": energy_override,
                    },
                }
            },
        }

    monkeypatch.setattr(value_training, "train_value_geometry", fake_train)
    monkeypatch.setattr(value_training, "SearchFeatureStore", lambda *args, **kwargs: object())
    report = compare_value_geometries(
        tmp_path / "processed",
        tmp_path / "cache",
        tmp_path / "comparison",
        config_path,
        energies=["euclidean", "directed"],
        seeds=[17, 29],
    )

    assert report["selected_energy"] == "directed"
    effect = report["paired_task_macro_policy"]["energy_minus_reference"]["directed"]
    assert effect["mean"] == pytest.approx(0.2)
    assert effect["low"] > 0.0
    assert effect["seeds"] == [17, 29]
    joint_effect = report["paired_task_macro"]["joint_stop_action_accuracy"][
        "energy_minus_reference"
    ]["directed"]
    assert joint_effect["mean"] == pytest.approx(0.2)
    regret_effect = report["paired_task_macro"]["regret_at_1"][
        "energy_minus_reference"
    ]["directed"]
    assert regret_effect["mean"] == pytest.approx(-0.2)
    directed = next(row for row in report["aggregate"] if row["energy"] == "directed")
    assert directed["mean_latent_geometry"]["mean_relative_asymmetry"] == pytest.approx(
        0.205
    )


def test_existing_run_groups_require_and_bootstrap_matched_dev_tasks(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for seed in (17, 29):
        shift = 0.01 if seed == 29 else 0.0
        for root, label_gain, parameters in ((left, 0.2, 100), (right, 0.0, 101)):
            write_json(
                root / f"seed-{seed}" / "value_metrics.json",
                {
                    "seed": seed,
                    "test_reported": False,
                    "parameter_count": parameters,
                    "checkpoint_sha256": f"checkpoint-{root.name}-{seed}",
                    "metrics": {
                        "dev": {
                            "split": "dev",
                            "policy_optimal_set_accuracy": 0.5 + label_gain + shift,
                            "joint_stop_action_accuracy": 0.4 + label_gain + shift,
                            "regret_at_1": 0.4 - label_gain + shift,
                            "per_task_policy_accuracy": {
                                "task-a": 0.4 + label_gain + shift,
                                "task-b": 0.6 + label_gain + shift,
                            },
                            "per_task_joint_accuracy": {
                                "task-a": 0.3 + label_gain + shift,
                                "task-b": 0.5 + label_gain + shift,
                            },
                            "per_task_regret_at_1": {
                                "task-a": 0.5 - label_gain + shift,
                                "task-b": 0.3 - label_gain + shift,
                            },
                        }
                    },
                },
            )

    report = compare_existing_value_run_groups(
        left,
        right,
        tmp_path / "comparison.json",
        left_label="frozen",
        right_label="hash",
        seeds=[29, 17, 17],
    )

    assert report["test_sealed"] is True
    assert report["seeds"] == [17, 29]
    assert report["parameter_match"]["relative_left_minus_right"] == pytest.approx(-1 / 101)
    assert report["paired_task_macro_effects"]["joint_stop_action_accuracy"][
        "mean"
    ] == pytest.approx(0.2)
    assert report["paired_task_macro_effects"]["regret_at_1"]["mean"] == pytest.approx(
        -0.2
    )

    mismatched = right / "seed-29" / "value_metrics.json"
    payload = value_training.read_json(mismatched)
    del payload["metrics"]["dev"]["per_task_policy_accuracy"]["task-b"]
    write_json(mismatched, payload)
    with pytest.raises(ValueError, match="Dev task IDs differ"):
        compare_existing_value_run_groups(
            left,
            right,
            tmp_path / "invalid.json",
            left_label="frozen",
            right_label="hash",
            seeds=[17, 29],
        )


def test_capacity_comparison_reports_half_configured_and_two_x_curve(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "value_training:\n  shared_dim: 8\n  hidden_dim: 16\n  energy_rank: 4\n",
        encoding="utf-8",
    )

    def fake_train(*args, model_capacity_override: dict, **kwargs):
        del args, kwargs
        hidden = model_capacity_override["hidden_dim"]
        score = {8: 0.4, 16: 0.6, 32: 0.55}[hidden]
        return {
            "parameter_count": hidden * 10,
            "checkpoint_sha256": f"checkpoint-{hidden}",
            "metrics": {
                "train": {"joint_stop_action_accuracy": score + 0.1},
                "dev": {"joint_stop_action_accuracy": score},
            },
        }

    monkeypatch.setattr(value_training, "train_value_geometry", fake_train)
    monkeypatch.setattr(value_training, "SearchFeatureStore", lambda *args, **kwargs: object())
    report = compare_value_capacities(
        tmp_path / "processed",
        tmp_path / "cache",
        tmp_path / "capacities",
        config_path,
        seeds=[17, 29],
    )

    assert [row["capacity"] for row in report["aggregate"]] == [
        "half_x",
        "configured",
        "two_x",
    ]
    assert report["half_x_minus_configured_dev_joint_accuracy"] == pytest.approx(-0.2)
    assert report["two_x_minus_configured_dev_joint_accuracy"] == pytest.approx(-0.05)

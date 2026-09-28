from __future__ import annotations

import math

import pytest
import torch

from geoflowagent.models.value_geometry import (
    AsymmetricBilinearGoalEnergy,
    DirectedQuasimetricGoalEnergy,
    GoalConditionedValueGeometry,
    ValueGeometryLossWeights,
    make_goal_energy,
    masked_action_distribution_loss,
    search_distillation_loss,
)


def _features(batch: int = 3, actions: int = 5) -> dict[str, object]:
    generator = torch.Generator().manual_seed(7)
    return {
        "state_views": {
            "general": torch.randn(batch, 8, generator=generator),
            "biomedical": torch.randn(batch, 6, generator=generator),
        },
        "goal_views": {
            "general": torch.randn(batch, 8, generator=generator),
            "biomedical": torch.randn(batch, 6, generator=generator),
        },
        "tool_views": {
            "general": torch.randn(actions, 8, generator=generator),
            "biomedical": torch.randn(actions, 6, generator=generator),
        },
        "structured_state": torch.randn(batch, 4, generator=generator),
        "structured_goal": torch.randn(batch, 4, generator=generator),
        "structured_tools": torch.randn(actions, 4, generator=generator),
    }


@pytest.mark.parametrize(
    "energy",
    [
        "cosine",
        "euclidean",
        "diagonal_mahalanobis",
        "lowrank_mahalanobis",
        "asymmetric_bilinear",
        "order_violation",
        "directed_quasimetric",
        "pair_mlp",
        "poincare",
    ],
)
def test_value_geometry_shapes_and_finite_outputs(energy: str) -> None:
    batch, actions = 3, 5
    model = GoalConditionedValueGeometry(
        {"general": 8, "biomedical": 6},
        structured_dim=4,
        shared_dim=12,
        hidden_dim=20,
        energy=energy,
        energy_rank=6,
    )
    features = _features(batch, actions)
    output = model(**features)

    assert output["energy"].shape == (batch, actions)
    assert output["q"].shape == (batch, actions)
    assert output["masked_q"].shape == (batch, actions)
    assert output["value"].shape == (batch,)
    assert output["state_reachability_logit"].shape == (batch,)
    assert output["action_reachability_logit"].shape == (batch, actions)
    assert output["completion_logit"].shape == (batch,)
    assert output["successor_latent"].shape == (batch, actions, 12)
    assert output["goal_latent"].shape == (batch, 12)
    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["q"]).all()
    assert (output["q"] >= 0).all()


def test_batched_tool_features_and_contract_mask() -> None:
    batch, actions = 2, 4
    features = _features(batch, actions)
    features["tool_views"] = {
        name: tensor.unsqueeze(0).expand(batch, -1, -1).clone()
        for name, tensor in features["tool_views"].items()
    }
    features["structured_tools"] = features["structured_tools"].unsqueeze(0).expand(
        batch, -1, -1
    )
    candidate_mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    model = GoalConditionedValueGeometry(
        {"general": 8, "biomedical": 6},
        structured_dim=4,
        shared_dim=10,
        hidden_dim=16,
    )
    output = model(**features, candidate_mask=candidate_mask)

    assert torch.isinf(output["masked_q"][~candidate_mask]).all()
    assert torch.isfinite(output["masked_q"][candidate_mask]).all()
    assert torch.isneginf(output["policy_logits"][~candidate_mask]).all()


def test_directed_quasimetric_is_nonnegative_asymmetric_and_triangular() -> None:
    energy = DirectedQuasimetricGoalEnergy(2, rank=2)
    with torch.no_grad():
        energy.projection.weight.copy_(torch.eye(2))
        energy.raw_direction.copy_(torch.tensor([2.0, 0.0]))
    x = torch.tensor([[0.0, 0.0]])
    y = torch.tensor([[1.0, 0.0]])
    z = torch.tensor([[1.0, 1.0]])

    d_xy = energy(x, y)
    d_yx = energy(y, x)
    d_xz = energy(x, z)
    d_yz = energy(y, z)

    assert d_xy.item() >= 0
    assert d_yx.item() >= 0
    assert energy(x, x).item() == pytest.approx(0.0, abs=1e-7)
    assert not torch.allclose(d_xy, d_yx)
    assert torch.all(d_xz <= d_xy + d_yz + 1e-6)


def test_directed_quasimetric_zero_direction_has_learning_signal() -> None:
    energy = DirectedQuasimetricGoalEnergy(2, rank=2)
    with torch.no_grad():
        energy.projection.weight.copy_(torch.eye(2))
        energy.raw_direction.zero_()
    source = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, -0.5]])

    energy(source, target).backward()

    assert energy.raw_direction.grad is not None
    assert energy.raw_direction.grad.norm().item() > 0.0


def test_asymmetric_bilinear_reversing_roles_changes_energy() -> None:
    energy = AsymmetricBilinearGoalEnergy(2)
    with torch.no_grad():
        energy.weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])

    assert energy(x, y).item() == pytest.approx(-1.0)
    assert energy(y, x).item() == pytest.approx(0.0)


def test_all_energy_factories_accept_broadcastable_pairs() -> None:
    source = torch.randn(2, 3, 8)
    target = torch.randn(2, 1, 8)
    for name in (
        "cosine",
        "euclidean",
        "diagonal_mahalanobis",
        "lowrank_mahalanobis",
        "bilinear",
        "order",
        "quasimetric",
        "mlp",
        "poincare",
    ):
        result = make_goal_energy(name, 8, rank=4, hidden_dim=12)(source, target)
        assert result.shape == (2, 3)
        assert torch.isfinite(result).all()


def test_unknown_actions_are_excluded_from_policy_denominator_and_gradient() -> None:
    # Unknown actions 2 and 3 look artificially attractive.  They must have no
    # effect on either the normalizer or gradients.
    q_value = torch.tensor([[1.0, 2.0, -100.0, -200.0]], requires_grad=True)
    known = torch.tensor([[True, True, False, False]])
    optimal = torch.tensor([[True, False, False, False]])
    loss = masked_action_distribution_loss(
        -q_value,
        known,
        optimal_action_mask=optimal,
    )
    expected = -math.log(math.exp(-1.0) / (math.exp(-1.0) + math.exp(-2.0)))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert q_value.grad is not None
    assert torch.equal(q_value.grad[0, 2:], torch.zeros(2))


def test_full_search_distillation_loss_is_finite_with_partial_labels() -> None:
    batch, actions = 3, 5
    model = GoalConditionedValueGeometry(
        {"general": 8, "biomedical": 6},
        structured_dim=4,
        shared_dim=12,
        hidden_dim=20,
        energy="directed_quasimetric",
    )
    output = model(**_features(batch, actions))
    known = torch.tensor(
        [
            [True, True, False, True, False],
            [True, False, True, True, False],
            [False, False, False, False, False],
        ]
    )
    candidate = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )
    optimal = torch.tensor(
        [
            [True, False, False, True, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    q_target = torch.tensor(
        [
            [2.0, 3.0, float("nan"), 2.0, float("nan")],
            [4.0, float("nan"), 1.0, 3.0, float("nan")],
            [float("nan")] * actions,
        ]
    )
    regret = q_target - torch.tensor([[2.0], [1.0], [0.0]])
    edge_cost = torch.where(torch.isfinite(q_target), torch.ones_like(q_target), q_target)
    successor_value = q_target - edge_cost
    action_reachable = torch.where(
        torch.isfinite(q_target), torch.ones_like(q_target), q_target
    )
    transition_target = output["successor_latent"].detach() + 0.1
    transition_target[~known] = float("nan")

    losses = search_distillation_loss(
        output,
        known,
        candidate_mask=candidate,
        optimal_action_mask=optimal,
        q_target=q_target,
        value_target=torch.tensor([2.0, 1.0, float("inf")]),
        regret_target=regret,
        edge_cost=edge_cost,
        successor_value_target=successor_value,
        transition_target=transition_target,
        action_reachability_target=action_reachable,
        state_reachability_target=torch.tensor([1.0, 1.0, 0.0]),
        completion_target=torch.tensor([0.0, 0.0, 1.0]),
    )

    expected = {
        "action",
        "q",
        "value",
        "regret",
        "bellman",
        "transition",
        "action_reachability",
        "state_reachability",
        "completion",
        "total",
    }
    assert set(losses) == expected
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_no_known_actions_returns_differentiable_zero_action_losses() -> None:
    batch, actions = 2, 3
    q_value = torch.randn(batch, actions, requires_grad=True)
    action_reachability = torch.randn(batch, actions, requires_grad=True)
    state_logit = torch.randn(batch, requires_grad=True)
    completion = torch.randn(batch, requires_grad=True)
    value = torch.rand(batch, requires_grad=True)
    output = {
        "q": q_value,
        "policy_logits": -q_value,
        "value": value,
        "action_reachability_logit": action_reachability,
        "state_reachability_logit": state_logit,
        "completion_logit": completion,
    }
    losses = search_distillation_loss(
        output,
        torch.zeros(batch, actions, dtype=torch.bool),
        optimal_action_mask=torch.ones(batch, actions, dtype=torch.bool),
        q_target=torch.ones(batch, actions),
        regret_target=torch.zeros(batch, actions),
        action_reachability_target=torch.ones(batch, actions),
        weights=ValueGeometryLossWeights(
            action=1.0,
            q=1.0,
            value=0.0,
            regret=1.0,
            bellman=0.0,
            transition=0.0,
            action_reachability=1.0,
            state_reachability=0.0,
            completion=0.0,
        ),
    )
    assert losses["total"].item() == pytest.approx(0.0)
    losses["total"].backward()
    assert torch.equal(q_value.grad, torch.zeros_like(q_value))
    assert torch.equal(action_reachability.grad, torch.zeros_like(action_reachability))


def test_hard_reachability_filter_can_abstain() -> None:
    model = GoalConditionedValueGeometry(
        {"general": 8}, structured_dim=0, shared_dim=8, hidden_dim=12
    )
    output = {
        "q": torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
        "action_reachability_logit": torch.tensor([[-10.0, -9.0], [10.0, -10.0]]),
    }
    selected = model.select_action(output, reachability_threshold=0.5)
    assert selected.tolist() == [-1, 0]

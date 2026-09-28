from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

from geoflowagent.models.flow import WholePlanFlow
from geoflowagent.training.flow import (
    _flow_losses,
    flow_selection_key,
    ode_turn_angle,
    polyline_geometry,
)


def _flow() -> WholePlanFlow:
    return WholePlanFlow(
        dim=8,
        context_dim=6,
        max_plan_length=4,
        hidden_dim=16,
        layers=1,
        heads=2,
        dropout=0.0,
    )


def test_flow_selection_breaks_action_ties_with_stop_then_edit() -> None:
    base = {
        "valid_first_action_accuracy": 0.6,
        "stop_balanced_accuracy": 0.5,
        "normalized_edit_distance": 0.8,
    }
    better_stop = {**base, "stop_balanced_accuracy": 0.7}
    better_edit = {**better_stop, "normalized_edit_distance": 0.6}
    worse_action = {**better_edit, "valid_first_action_accuracy": 0.5}

    assert flow_selection_key(better_stop) > flow_selection_key(base)
    assert flow_selection_key(better_edit) > flow_selection_key(better_stop)
    assert flow_selection_key(worse_action) < flow_selection_key(base)


def test_flow_targets_place_tools_then_stop_then_pad() -> None:
    torch.manual_seed(3)
    model = _flow()
    prototypes = F.normalize(torch.randn(3, 8), dim=-1)
    built = model.build_targets(prototypes, [[], [0, 2], [1]], pad_weight=0.125)
    stop, pad = model.special_anchors()

    assert built["target"].shape == (3, 4, 8)
    assert built["plan_mask"].tolist() == [
        [True, False, False, False],
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert built["stop_target"].tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    assert built["tool_target"].tolist() == [
        [-100, -100, -100, -100],
        [0, 2, -100, -100],
        [1, -100, -100, -100],
    ]
    torch.testing.assert_close(built["target"][0, 0], stop)
    torch.testing.assert_close(built["target"][1, :2], prototypes[torch.tensor([0, 2])])
    torch.testing.assert_close(built["target"][1, 2], stop)
    torch.testing.assert_close(built["target"][1, 3], pad)
    torch.testing.assert_close(
        built["weights"],
        torch.tensor(
            [
                [1.0, 0.125, 0.125, 0.125],
                [1.0, 1.0, 1.0, 0.125],
                [1.0, 1.0, 0.125, 0.125],
            ]
        ),
    )


def test_flow_solve_needs_no_gold_length_and_returns_transport() -> None:
    model = _flow().eval()
    parameters = inspect.signature(model.solve).parameters
    assert "length" not in parameters
    assert "mask" not in parameters
    assert "suffixes" not in parameters

    context = torch.randn(2, 6)
    tool_set_context = torch.randn(2, 8)
    noise = torch.randn(2, 4, 8)
    endpoint, transport = model.solve(
        context,
        tool_set_context,
        nfe=3,
        noise=noise,
        return_transport=True,
    )

    assert endpoint.shape == (2, 4, 8)
    assert transport is not None
    assert len(transport) == 4
    torch.testing.assert_close(transport[0], noise)
    torch.testing.assert_close(transport[-1], endpoint)
    assert torch.isfinite(endpoint).all()


def test_flow_decode_respects_candidate_mask_and_stop_head() -> None:
    torch.manual_seed(5)
    model = _flow().eval()
    plan = torch.randn(1, 4, 8)
    prototypes = F.normalize(torch.randn(3, 8), dim=-1)
    candidate_mask = torch.tensor([[False, True, False]])
    linear = model.stop_head[-1]

    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(-20.0)
    assert model.decode(plan, prototypes, candidate_mask, stop_threshold=0.5) == [[1, 1, 1, 1]]

    with torch.no_grad():
        linear.bias.fill_(20.0)
    assert model.decode(plan, prototypes, candidate_mask, stop_threshold=0.5) == [[]]


def test_flow_matching_step_has_joint_plan_tensor_shapes() -> None:
    model = _flow()
    context = torch.randn(2, 6)
    tool_set_context = torch.randn(2, 8)
    target = torch.randn(2, 4, 8)
    result = model.flow_matching_step(
        target,
        context,
        tool_set_context,
        time=torch.tensor([0.25, 0.75]),
        noise=torch.zeros_like(target),
    )

    for key in (
        "interpolated",
        "target_velocity",
        "predicted_velocity",
        "predicted_endpoint",
    ):
        assert result[key].shape == target.shape
        assert torch.isfinite(result[key]).all()


def test_plan_slot_and_ode_geometry_are_measured_on_separate_axes() -> None:
    straight = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    corner = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    straight_length, straight_angle = polyline_geometry(straight)
    corner_length, corner_angle = polyline_geometry(corner)

    assert straight_length > 0
    assert corner_length > 0
    assert straight_angle < corner_angle

    start = torch.zeros(1, 1, 2)
    transport_straight = [
        start,
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[2.0, 0.0]]]),
    ]
    transport_corner = [
        start,
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 1.0]]]),
    ]
    assert ode_turn_angle(transport_straight) == 0.0
    assert ode_turn_angle(transport_corner) > 1.5


def test_tool_codebook_loss_excludes_non_candidate_tools() -> None:
    model = _flow()
    prototypes = F.normalize(torch.randn(3, 8), dim=-1)
    targets = model.build_targets(prototypes, [[0]], pad_weight=0.1)
    endpoint = torch.randn(1, 4, 8)
    step = {
        "predicted_velocity": torch.zeros_like(endpoint),
        "target_velocity": torch.zeros_like(endpoint),
        "predicted_endpoint": endpoint,
    }
    candidates = torch.tensor([[True, False, False]])

    _, components = _flow_losses(
        model,
        step,
        targets,
        prototypes,
        candidates,
        stop_weight=0.0,
        tool_weight=1.0,
        separation_weight=0.0,
    )

    assert components["tool"] == 0.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from geoflowagent.models.search_state_flow import SearchStateFlow
from geoflowagent.training.search_paths import SearchPathReference
from geoflowagent.training.state_flow import (
    _capture_global_rng_state,
    _restore_global_rng_state,
    _state_flow_progress_backup_path,
    _transition_classification_loss,
)


def test_state_flow_resume_restores_global_rngs() -> None:
    import random

    import numpy as np

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = _capture_global_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(2))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    _restore_global_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(2))

    assert actual[:2] == pytest.approx(expected[:2])
    torch.testing.assert_close(actual[2], expected[2])


def test_state_flow_replication_progress_paths_are_run_specific(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "persistent"
    monkeypatch.setenv("GEOFLOW_STATE_FLOW_PROGRESS_BACKUP_DIR", str(root))

    first = _state_flow_progress_backup_path(
        tmp_path / "seed-17",
        seed=17,
        training_hash="a" * 64,
        configured=tmp_path / "configured.json",
    )
    second = _state_flow_progress_backup_path(
        tmp_path / "seed-29",
        seed=29,
        training_hash="b" * 64,
        configured=tmp_path / "configured.json",
    )

    assert first is not None and second is not None
    assert first.parent == root
    assert second.parent == root
    assert first != second
    assert "seed-17" in first.name
    assert "seed-29" in second.name


def _model() -> SearchStateFlow:
    return SearchStateFlow(
        snapshot_input_dim=7,
        context_input_dim=8,
        goal_input_dim=9,
        tool_input_dim=6,
        dim=12,
        max_plan_length=5,
        hidden_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
        projection_seed=13,
    )


def test_fixed_state_flow_targets_have_explicit_stop_and_pad_channels() -> None:
    model = _model()
    successor = torch.randn(2, 7)
    tools = torch.randn(2, 6)
    sequence = model.transition_anchor(successor, tools)
    targets = model.build_targets([sequence], pad_weight=0.2)

    assert targets["target"].shape == (1, 5, 12)
    torch.testing.assert_close(targets["target"][0, :2], sequence)
    torch.testing.assert_close(targets["target"][0, 2], model.stop_anchor)
    torch.testing.assert_close(targets["target"][0, 3], model.pad_anchor)
    assert torch.count_nonzero(sequence[:, -2:]) == 0
    assert targets["active_mask"].tolist() == [[True, True, True, False, False]]
    assert targets["transition_mask"].tolist() == [[True, True, False, False, False]]
    assert targets["weights"].tolist()[0] == pytest.approx([1.0, 1.0, 1.0, 0.2, 0.2])


def test_state_flow_solve_has_no_gold_length_input_and_is_reproducible() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    condition = model.encode_condition(torch.randn(2, 8), torch.randn(2, 9))
    noise = torch.randn(2, 5, 12)

    first, first_transport = model.solve(condition, nfe=3, noise=noise, return_transport=True)
    second, _ = model.solve(condition, nfe=3, noise=noise, return_transport=False)

    torch.testing.assert_close(first, second)
    assert first.shape == (2, 5, 12)
    assert first_transport is not None
    assert len(first_transport) == 4


def test_projection_is_fixed_and_checkpoint_config_reconstructs_model() -> None:
    original = _model()
    reconstructed = SearchStateFlow(**original.export_config())

    assert not any(name.endswith("projection") for name, _ in original.named_parameters())
    torch.testing.assert_close(original.snapshot_projection, reconstructed.snapshot_projection)
    torch.testing.assert_close(original.tool_projection, reconstructed.tool_projection)
    assert "snapshot_projection" in original.state_dict()
    assert "tool_projection" in original.state_dict()


def test_flow_matching_step_produces_trainable_velocity() -> None:
    model = _model()
    successor = torch.randn(2, 7)
    tools = torch.randn(2, 6)
    targets = model.build_targets(
        [
            model.transition_anchor(successor[:1], tools[:1]),
            model.transition_anchor(successor, tools),
        ],
        pad_weight=0.1,
    )
    condition = model.encode_condition(torch.randn(2, 8), torch.randn(2, 9))
    step = model.flow_matching_step(
        targets["target"],
        condition,
        time=torch.tensor([0.25, 0.75]),
        noise=torch.zeros_like(targets["target"]),
    )
    loss = (step["predicted_velocity"] - step["target_velocity"]).square().mean()
    loss.backward()

    assert model.output[-1].weight.grad is not None
    assert torch.isfinite(loss)


def test_vectorized_transition_loss_matches_per_transition_reference() -> None:
    torch.manual_seed(11)
    model = _model()
    snapshot = torch.randn(5, 7)
    tools = torch.randn(3, 6)
    candidate = torch.tensor(
        [[True, True, False], [False, True, True], [True, False, True]]
    )
    successor = torch.tensor([[1, 2, -1], [-1, 3, 4], [3, -1, 4]])
    targets = {
        "policy_candidate_mask": candidate,
        "transition_known_mask": candidate.clone(),
        "successor_feature_mask": candidate.clone(),
        "successor_index": successor,
    }
    features = SimpleNamespace(
        snapshot=snapshot,
        tool=tools,
        store=SimpleNamespace(_targets=targets),
    )
    references = [
        SearchPathReference(
            row_index=0,
            example_id="e0",
            task_id="t0",
            split="train",
            root_state_id="s0",
            rank=0,
            tool_ids=("b", "c"),
            tool_indices=(1, 2),
            state_ids=("s0", "s1", "s2"),
            state_indices=(0, 1, 2),
            total_cost=2.0,
            value_star=2.0,
            excess_cost=0.0,
            probability=1.0,
        ),
        SearchPathReference(
            row_index=2,
            example_id="e2",
            task_id="t1",
            split="train",
            root_state_id="s2",
            rank=0,
            tool_ids=("a",),
            tool_indices=(0,),
            state_ids=("s2", "s3"),
            state_indices=(2, 3),
            total_cost=1.0,
            value_star=1.0,
            excess_cost=0.0,
            probability=1.0,
        ),
    ]
    endpoints = torch.randn(2, model.max_plan_length, model.dim, requires_grad=True)
    temperature = 0.17

    expected = []
    for batch, reference in enumerate(references):
        for slot, (source_index, gold_tool) in enumerate(
            zip(reference.state_indices[:-1], reference.tool_indices, strict=True)
        ):
            tool_indices = torch.where(candidate[source_index])[0]
            anchors = model.transition_anchor(
                snapshot[successor[source_index, tool_indices]], tools[tool_indices]
            )
            logits = torch.nn.functional.cosine_similarity(
                endpoints[batch, slot].unsqueeze(0), anchors, dim=-1
            ) / temperature
            label = torch.where(tool_indices == gold_tool)[0]
            expected.append(torch.nn.functional.cross_entropy(logits[None], label[:1]))

    actual = _transition_classification_loss(
        model,
        endpoints,
        references,
        features,  # type: ignore[arg-type]
        temperature=temperature,
    )
    torch.testing.assert_close(actual, torch.stack(expected).mean())
    actual.backward()
    assert endpoints.grad is not None

from __future__ import annotations

import pytest
import torch

from geoflowagent.models.distances import make_energy
from geoflowagent.models.functional import FunctionalGeometryModel


@pytest.mark.parametrize(
    "name",
    [
        "euclidean",
        "cosine",
        "diagonal_mahalanobis",
        "lowrank_mahalanobis",
        "bilinear",
        "poincare",
    ],
)
def test_pairwise_energy_is_finite_and_has_batch_tool_shape(name: str) -> None:
    torch.manual_seed(7)
    query = torch.randn(3, 8, requires_grad=True)
    tools = torch.randn(5, 8, requires_grad=True)
    query.data[0].zero_()
    tools.data[0].zero_()
    energy = make_energy(name, 8, rank=4)(query, tools)

    assert energy.shape == (3, 5)
    assert torch.isfinite(energy).all()
    energy.mean().backward()
    assert query.grad is not None
    assert tools.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(tools.grad).all()


@pytest.mark.parametrize(
    "name",
    [
        "euclidean",
        "cosine",
        "diagonal_mahalanobis",
        "lowrank_mahalanobis",
        "bilinear",
        "poincare",
    ],
)
def test_functional_geometry_forward_supports_multiview_aux_and_masks(name: str) -> None:
    torch.manual_seed(11)
    model = FunctionalGeometryModel(
        {"biomedical": 12, "general": 10},
        aux_dims={"sequence_aux": 6},
        structured_dim=7,
        shared_dim=8,
        hidden_dim=16,
        distance=name,
        distance_rank=4,
    )
    candidate_mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    output = model(
        {"biomedical": torch.randn(2, 12), "general": torch.randn(2, 10)},
        {"biomedical": torch.randn(4, 12), "general": torch.randn(4, 10)},
        torch.randn(2, 7),
        torch.randn(4, 7),
        aux_views={"sequence_aux": torch.randn(2, 6)},
        candidate_mask=candidate_mask,
    )

    assert output["energy"].shape == (2, 4)
    assert output["view_energy"].shape == (2, 3, 4)
    assert output["view_weights"].shape == (2, 3)
    assert output["context"].shape == (2, 8)
    assert torch.isfinite(output["energy"]).all()
    assert torch.all(output["energy"][~candidate_mask] > 1e20)
    torch.testing.assert_close(output["view_weights"].sum(dim=1), torch.ones(2))

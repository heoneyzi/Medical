from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseEnergy(nn.Module, ABC):
    @abstractmethod
    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        """Return lower-is-better energy with shape [B, T]."""


class SquaredEuclideanEnergy(PairwiseEnergy):
    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        return (
            query.square().sum(-1, keepdim=True)
            + tools.square().sum(-1).unsqueeze(0)
            - 2 * query @ tools.T
        ).clamp_min(0)


class CosineEnergy(PairwiseEnergy):
    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.normalize(query, dim=-1) @ F.normalize(tools, dim=-1).T


class DiagonalMahalanobisEnergy(PairwiseEnergy):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return SquaredEuclideanEnergy()(query * scale, tools * scale)


class LowRankMahalanobisEnergy(PairwiseEnergy):
    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(4, dim // 2)
        self.projection = nn.Linear(dim, rank, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        return SquaredEuclideanEnergy()(self.projection(query), self.projection(tools))


class BilinearEnergy(PairwiseEnergy):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        return -(query @ self.weight) @ tools.T


class PoincareEnergy(PairwiseEnergy):
    def __init__(self, dim: int, curvature: float = 1.0) -> None:
        super().__init__()
        self.query_map = nn.Linear(dim, dim, bias=False)
        self.tool_map = nn.Linear(dim, dim, bias=False)
        self.curvature = float(curvature)
        nn.init.eye_(self.query_map.weight)
        nn.init.eye_(self.tool_map.weight)

    def _ball(self, value: torch.Tensor) -> torch.Tensor:
        norm = value.norm(dim=-1, keepdim=True)
        direction = value / norm.clamp_min(1e-8)
        radius = 0.95 / self.curvature**0.5
        return radius * torch.tanh(norm) * direction

    def forward(self, query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
        query = self._ball(self.query_map(query))
        tools = self._ball(self.tool_map(tools))
        difference = (query[:, None, :] - tools[None, :, :]).square().sum(-1)
        q_norm = query.square().sum(-1, keepdim=True)
        t_norm = tools.square().sum(-1).unsqueeze(0)
        denominator = (1 - self.curvature * q_norm).clamp_min(1e-6) * (
            1 - self.curvature * t_norm
        ).clamp_min(1e-6)
        argument = 1 + 2 * self.curvature * difference / denominator
        return torch.acosh(argument.clamp_min(1 + 1e-6)) / self.curvature**0.5


def make_energy(name: str, dim: int, *, rank: int | None = None) -> PairwiseEnergy:
    if name == "euclidean":
        return SquaredEuclideanEnergy()
    if name == "cosine":
        return CosineEnergy()
    if name == "diagonal_mahalanobis":
        return DiagonalMahalanobisEnergy(dim)
    if name == "lowrank_mahalanobis":
        return LowRankMahalanobisEnergy(dim, rank=rank)
    if name == "bilinear":
        return BilinearEnergy(dim)
    if name == "poincare":
        return PoincareEnergy(dim)
    raise ValueError(f"Unknown energy {name!r}")

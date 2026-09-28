from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from geoflowagent.models.distances import make_energy


class FunctionalGeometryModel(nn.Module):
    """State-conditioned multi-view energy over frozen encoder observations."""

    def __init__(
        self,
        view_dims: dict[str, int],
        *,
        aux_dims: dict[str, int] | None = None,
        structured_dim: int,
        shared_dim: int = 64,
        hidden_dim: int = 128,
        distance: str = "cosine",
        distance_rank: int | None = None,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if not view_dims:
            raise ValueError("view_dims cannot be empty")
        self.view_names = sorted(view_dims)
        self.aux_names = sorted(aux_dims or {})
        self.view_dims = dict(view_dims)
        self.aux_dims = dict(aux_dims or {})
        self.structured_dim = int(structured_dim)
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.distance_name = distance
        self.distance_rank = distance_rank
        self.temperature = float(temperature)
        self.state_projectors = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, shared_dim),
                )
                for view, dim in view_dims.items()
            }
        )
        self.tool_projectors = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, shared_dim),
                )
                for view, dim in view_dims.items()
            }
        )
        self.aux_projectors = nn.ModuleDict(
            {
                view: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, shared_dim))
                for view, dim in self.aux_dims.items()
            }
        )
        self.energies = nn.ModuleDict(
            {
                view: make_energy(distance, shared_dim, rank=distance_rank)
                for view in self.view_names
            }
        )
        self.structured_state = nn.Linear(structured_dim, shared_dim)
        self.structured_tool = nn.Linear(structured_dim, shared_dim)
        self.structured_energy = make_energy("bilinear", shared_dim)
        gate_input = shared_dim * (len(self.view_names) + len(self.aux_names) + 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input),
            nn.Linear(gate_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.view_names) + 1),
        )
        self.prototype_logits = nn.Parameter(torch.zeros(len(self.view_names)))
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(gate_input),
            nn.Linear(gate_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )
        snapshot_input = gate_input
        self.snapshot_fuse = nn.Sequential(
            nn.LayerNorm(snapshot_input),
            nn.Linear(snapshot_input, shared_dim),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )

    def project_state(self, state_views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {view: self.state_projectors[view](state_views[view]) for view in self.view_names}

    def project_tools(self, tool_views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {view: self.tool_projectors[view](tool_views[view]) for view in self.view_names}

    def project_aux(self, aux_views: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        aux_views = aux_views or {}
        return {view: self.aux_projectors[view](aux_views[view]) for view in self.aux_names}

    def encode_context(
        self,
        state_views: dict[str, torch.Tensor],
        structured_state: torch.Tensor,
        aux_views: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected = self.project_state(state_views)
        projected_aux = self.project_aux(aux_views)
        pieces = [projected[view] for view in self.view_names]
        pieces.extend(projected_aux[view] for view in self.aux_names)
        pieces.append(self.structured_state(structured_state))
        return self.context_fuse(torch.cat(pieces, dim=-1))

    def encode_snapshot(
        self,
        state_views: dict[str, torch.Tensor],
        structured_state: torch.Tensor,
        aux_views: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected = self.project_state(state_views)
        projected_aux = self.project_aux(aux_views)
        pieces = [projected[view] for view in self.view_names]
        pieces.extend(projected_aux[view] for view in self.aux_names)
        pieces.append(self.structured_state(structured_state))
        return self.snapshot_fuse(torch.cat(pieces, dim=-1))

    def tool_prototypes(self, tool_views: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = self.project_tools(tool_views)
        weights = self.prototype_logits.softmax(dim=0)
        prototype = sum(
            weights[index] * projected[view] for index, view in enumerate(self.view_names)
        )
        return F.normalize(prototype, dim=-1)

    def forward(
        self,
        state_views: dict[str, torch.Tensor],
        tool_views: dict[str, torch.Tensor],
        structured_state: torch.Tensor,
        structured_tools: torch.Tensor,
        *,
        aux_views: dict[str, torch.Tensor] | None = None,
        candidate_mask: torch.Tensor | None = None,
        contract_mask: torch.Tensor | None = None,
        apply_contract_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        projected_state = self.project_state(state_views)
        projected_tools = self.project_tools(tool_views)
        projected_aux = self.project_aux(aux_views)
        structured_state_code = self.structured_state(structured_state)
        structured_tool_code = self.structured_tool(structured_tools)
        gate_input = torch.cat(
            [
                *[projected_state[view] for view in self.view_names],
                *[projected_aux[view] for view in self.aux_names],
                structured_state_code,
            ],
            dim=-1,
        )
        weights = self.gate(gate_input).softmax(dim=-1)
        view_energies = [
            self.energies[view](projected_state[view], projected_tools[view]) / self.temperature
            for view in self.view_names
        ]
        view_energies.append(
            self.structured_energy(structured_state_code, structured_tool_code) / self.temperature
        )
        stacked = torch.stack(view_energies, dim=1)
        total = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        allowed = candidate_mask
        if apply_contract_mask and contract_mask is not None:
            allowed = contract_mask if allowed is None else allowed & contract_mask
        if allowed is not None:
            total = total.masked_fill(~allowed, torch.finfo(total.dtype).max / 100)
        return {
            "energy": total,
            "view_energy": stacked,
            "view_weights": weights,
            "context": self.context_fuse(gate_input),
        }

    def predict_next_snapshot(
        self, context: torch.Tensor, tool_prototype: torch.Tensor
    ) -> torch.Tensor:
        return self.transition_head(torch.cat([context, tool_prototype], dim=-1))

    def export_config(self) -> dict[str, Any]:
        return {
            "view_dims": self.view_dims,
            "aux_dims": self.aux_dims,
            "structured_dim": self.structured_dim,
            "shared_dim": self.shared_dim,
            "hidden_dim": self.hidden_dim,
            "distance": self.distance_name,
            "distance_rank": self.distance_rank,
            "temperature": self.temperature,
        }


def set_valued_nll(
    energy: torch.Tensor, valid_mask: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    score = -energy
    valid_score = score.masked_fill(~valid_mask, -torch.inf)
    candidate_score = score.masked_fill(~candidate_mask, -torch.inf)
    has_valid = valid_mask.any(dim=1)
    if not has_valid.any():
        return energy.sum() * 0
    numerator = torch.logsumexp(valid_score[has_valid], dim=1)
    denominator = torch.logsumexp(candidate_score[has_valid], dim=1)
    return -(numerator - denominator).mean()


def regret_ranking_loss(
    energy: torch.Tensor,
    regret: torch.Tensor,
    known_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    lower_regret = regret.unsqueeze(2) + 1e-6 < regret.unsqueeze(1)
    pair_mask = lower_regret & known_mask.unsqueeze(2) & known_mask.unsqueeze(1)
    pair_mask = pair_mask & candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
    # i has lower regret than j, so E_i + margin should be below E_j.
    violation = F.relu(margin + energy.unsqueeze(2) - energy.unsqueeze(1))
    if not pair_mask.any():
        return energy.sum() * 0
    return violation[pair_mask].mean()

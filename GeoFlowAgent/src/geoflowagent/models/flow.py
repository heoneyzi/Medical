from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(1, half - 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=time.device, dtype=time.dtype))
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return self.output(embedding)


class WholePlanFlow(nn.Module):
    """Joint rectified flow over [plan slot, shared embedding] tensors.

    ODE time and plan position are separate axes. Gold lengths are used only to
    construct weighted training targets; `solve` accepts no gold mask or length.
    """

    def __init__(
        self,
        dim: int,
        *,
        context_dim: int,
        max_plan_length: int = 10,
        hidden_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if noise_scale <= 0:
            raise ValueError("noise_scale must be positive")
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.max_plan_length = int(max_plan_length)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.noise_scale = float(noise_scale)
        self.input_projection = nn.Linear(dim, dim)
        self.position = nn.Parameter(torch.randn(max_plan_length, dim) * 0.02)
        self.time_embedding = SinusoidalTimeEmbedding(dim)
        self.condition = nn.Sequential(
            nn.Linear(context_dim + dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.stop_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        stop = torch.zeros(dim)
        pad = torch.zeros(dim)
        stop[0] = 1.0
        pad[min(1, dim - 1)] = 1.0
        self.stop_anchor = nn.Parameter(stop)
        self.pad_anchor = nn.Parameter(pad)

    def special_anchors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.stop_anchor, dim=0), F.normalize(self.pad_anchor, dim=0)

    def velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        tool_set_context: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition(torch.cat([context, tool_set_context], dim=-1))
        hidden = self.input_projection(state)
        hidden = hidden + self.position.unsqueeze(0)
        hidden = hidden + self.time_embedding(time).unsqueeze(1)
        hidden = hidden + condition.unsqueeze(1)
        return self.output(self.transformer(hidden))

    def build_targets(
        self,
        tool_prototypes: torch.Tensor,
        suffixes: list[list[int]],
        *,
        pad_weight: float,
    ) -> dict[str, torch.Tensor]:
        batch_size = len(suffixes)
        device = tool_prototypes.device
        stop_anchor, pad_anchor = self.special_anchors()
        target = pad_anchor.view(1, 1, -1).expand(batch_size, self.max_plan_length, -1).clone()
        plan_mask = torch.zeros((batch_size, self.max_plan_length), dtype=torch.bool, device=device)
        stop_target = torch.zeros((batch_size, self.max_plan_length), device=device)
        tool_target = torch.full(
            (batch_size, self.max_plan_length), -100, dtype=torch.long, device=device
        )
        weights = torch.full((batch_size, self.max_plan_length), float(pad_weight), device=device)
        for batch_index, suffix in enumerate(suffixes):
            if len(suffix) + 1 > self.max_plan_length:
                raise ValueError(
                    f"Suffix length {len(suffix)} needs {len(suffix) + 1} slots including STOP, "
                    f"but max_plan_length={self.max_plan_length}"
                )
            if suffix:
                indices = torch.as_tensor(suffix, dtype=torch.long, device=device)
                target[batch_index, : len(suffix)] = tool_prototypes[indices]
                tool_target[batch_index, : len(suffix)] = indices
            stop_position = len(suffix)
            target[batch_index, stop_position] = stop_anchor
            plan_mask[batch_index, : stop_position + 1] = True
            stop_target[batch_index, stop_position] = 1.0
            weights[batch_index, : stop_position + 1] = 1.0
        return {
            "target": target,
            "plan_mask": plan_mask,
            "stop_target": stop_target,
            "tool_target": tool_target,
            "weights": weights,
        }

    def flow_matching_step(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        tool_set_context: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = target.shape[0]
        if time is None:
            time = torch.rand(batch_size, device=target.device)
        if noise is None:
            noise = torch.randn_like(target) * self.noise_scale
        interpolated = (1 - time[:, None, None]) * noise + time[:, None, None] * target
        target_velocity = target - noise
        predicted_velocity = self.velocity(interpolated, time, context, tool_set_context)
        predicted_endpoint = interpolated + (1 - time[:, None, None]) * predicted_velocity
        return {
            "time": time,
            "noise": noise,
            "interpolated": interpolated,
            "target_velocity": target_velocity,
            "predicted_velocity": predicted_velocity,
            "predicted_endpoint": predicted_endpoint,
        }

    @torch.no_grad()
    def solve(
        self,
        context: torch.Tensor,
        tool_set_context: torch.Tensor,
        *,
        nfe: int = 16,
        noise: torch.Tensor | None = None,
        return_transport: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if nfe <= 0:
            raise ValueError("nfe must be positive")
        batch_size = context.shape[0]
        state = (
            torch.randn(batch_size, self.max_plan_length, self.dim, device=context.device)
            * self.noise_scale
            if noise is None
            else noise.clone()
        )
        if state.shape != (batch_size, self.max_plan_length, self.dim):
            raise ValueError(f"Unexpected noise shape {tuple(state.shape)}")
        trajectory = [state.clone()] if return_transport else None
        dt = 1.0 / nfe
        for step in range(nfe):
            time = torch.full((batch_size,), step / nfe, device=state.device)
            state = state + dt * self.velocity(state, time, context, tool_set_context)
            if trajectory is not None:
                trajectory.append(state.clone())
        return state, trajectory

    def stop_probability(self, plan: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.stop_head(plan).squeeze(-1))

    @torch.no_grad()
    def decode(
        self,
        plan: torch.Tensor,
        tool_prototypes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        stop_threshold: float = 0.5,
    ) -> list[list[int]]:
        normalized_plan = F.normalize(plan, dim=-1)
        normalized_tools = F.normalize(tool_prototypes, dim=-1)
        energy = 1.0 - torch.einsum("bld,td->blt", normalized_plan, normalized_tools)
        energy = energy.masked_fill(
            ~candidate_mask[:, None, :], torch.finfo(energy.dtype).max / 100
        )
        tool_choice = energy.argmin(dim=-1)
        stop = self.stop_probability(plan)
        output: list[list[int]] = []
        for batch_index in range(plan.shape[0]):
            sequence: list[int] = []
            for position in range(self.max_plan_length):
                if float(stop[batch_index, position].item()) >= stop_threshold:
                    break
                sequence.append(int(tool_choice[batch_index, position]))
            output.append(sequence)
        return output

    def special_anchor_separation_loss(
        self, tool_prototypes: torch.Tensor, margin: float = 0.5
    ) -> torch.Tensor:
        stop, pad = self.special_anchors()
        normalized_tools = F.normalize(tool_prototypes, dim=-1)
        stop_tool_similarity = normalized_tools @ stop
        pad_tool_similarity = normalized_tools @ pad
        stop_tool_loss = F.relu(stop_tool_similarity - (1.0 - margin)).mean()
        pad_tool_loss = F.relu(pad_tool_similarity - (1.0 - margin)).mean()
        stop_pad_similarity = torch.sum(stop * pad)
        stop_pad_loss = F.relu(stop_pad_similarity - (1.0 - margin))
        return stop_tool_loss + pad_tool_loss + stop_pad_loss

    def export_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "context_dim": self.context_dim,
            "max_plan_length": self.max_plan_length,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "noise_scale": self.noise_scale,
        }

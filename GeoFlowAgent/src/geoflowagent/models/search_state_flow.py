"""Conditional rectified flow over frozen search-state transition anchors.

This is intentionally separate from :class:`WholePlanFlow`, which models the
legacy single-demonstration tool suffix.  ``SearchStateFlow`` receives fixed
multi-view cache features, maps them through *non-trainable* random projections,
and generates a whole sequence of transition anchors followed by an explicit
STOP anchor.  Each transition anchor combines the successor-state embedding and
the selected tool embedding, so two tools that reach the same state remain
distinguishable during constrained decoding.

The final two latent coordinates are reserved for PAD and STOP.  State/tool
semantics can therefore never collapse onto either special token by construction.
Only the conditional velocity field and the STOP classifier are optimized.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from geoflowagent.models.flow import SinusoidalTimeEmbedding


def _fixed_projection(in_dim: int, out_dim: int, *, seed: int) -> torch.Tensor:
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("Projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weight = torch.randn(out_dim, in_dim, generator=generator)
    return weight / math.sqrt(float(in_dim))


class SearchStateFlow(nn.Module):
    """Generate padded successor-state/tool trajectories with rectified flow.

    The random feature projections are fixed model buffers.  They use no train,
    dev, or test statistics, while still reducing heterogeneous frozen encoder
    dimensions to one shared latent space.  A checkpoint stores their exact
    tensors, so reconstruction does not depend on a future RNG implementation.
    """

    def __init__(
        self,
        *,
        snapshot_input_dim: int,
        context_input_dim: int,
        goal_input_dim: int,
        tool_input_dim: int,
        dim: int = 32,
        max_plan_length: int = 12,
        hidden_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        projection_seed: int = 17,
        tool_mix: float = 0.35,
    ) -> None:
        super().__init__()
        if dim < 6:
            raise ValueError("dim must be at least 6 (including PAD/STOP coordinates)")
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if max_plan_length < 2:
            raise ValueError("max_plan_length must fit at least one transition and STOP")
        if hidden_dim <= 0 or layers <= 0 or heads <= 0:
            raise ValueError("hidden_dim, layers, and heads must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(tool_mix) or tool_mix < 0:
            raise ValueError("tool_mix must be finite and non-negative")

        self.snapshot_input_dim = int(snapshot_input_dim)
        self.context_input_dim = int(context_input_dim)
        self.goal_input_dim = int(goal_input_dim)
        self.tool_input_dim = int(tool_input_dim)
        self.dim = int(dim)
        self.semantic_dim = self.dim - 2
        self.max_plan_length = int(max_plan_length)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.projection_seed = int(projection_seed)
        self.tool_mix = float(tool_mix)

        # Fixed Johnson-Lindenstrauss style maps.  Offsets keep the modalities
        # independent while the base seed remains a compact reproducibility key.
        self.register_buffer(
            "snapshot_projection",
            _fixed_projection(
                self.snapshot_input_dim, self.semantic_dim, seed=self.projection_seed + 11
            ),
        )
        self.register_buffer(
            "context_projection",
            _fixed_projection(
                self.context_input_dim, self.semantic_dim, seed=self.projection_seed + 23
            ),
        )
        self.register_buffer(
            "goal_projection",
            _fixed_projection(
                self.goal_input_dim, self.semantic_dim, seed=self.projection_seed + 37
            ),
        )
        self.register_buffer(
            "tool_projection",
            _fixed_projection(
                self.tool_input_dim, self.semantic_dim, seed=self.projection_seed + 53
            ),
        )
        stop = torch.zeros(self.dim)
        pad = torch.zeros(self.dim)
        pad[-2] = 1.0
        stop[-1] = 1.0
        self.register_buffer("pad_anchor", pad)
        self.register_buffer("stop_anchor", stop)

        self.input_projection = nn.Linear(self.dim, self.dim)
        self.position = nn.Parameter(torch.randn(self.max_plan_length, self.dim) * 0.02)
        self.time_embedding = SinusoidalTimeEmbedding(self.dim)
        self.condition = nn.Sequential(
            nn.Linear(3 * self.semantic_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim))
        self.stop_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))

    @staticmethod
    def _project(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"Feature width {features.shape[-1]} does not match projection "
                f"width {weight.shape[-1]}"
            )
        return F.normalize(F.linear(features, weight), dim=-1)

    @staticmethod
    def _append_special_zeros(semantic: torch.Tensor) -> torch.Tensor:
        return F.pad(semantic, (0, 2))

    def encode_state(self, snapshot_features: torch.Tensor) -> torch.Tensor:
        return self._append_special_zeros(
            self._project(snapshot_features, self.snapshot_projection)
        )

    def encode_tool(self, tool_features: torch.Tensor) -> torch.Tensor:
        return self._append_special_zeros(self._project(tool_features, self.tool_projection))

    def encode_condition(
        self,
        context_features: torch.Tensor,
        goal_features: torch.Tensor,
    ) -> torch.Tensor:
        context = self._project(context_features, self.context_projection)
        goal = self._project(goal_features, self.goal_projection)
        return torch.cat([context, goal, goal - context], dim=-1)

    def transition_anchor(
        self,
        successor_features: torch.Tensor,
        tool_features: torch.Tensor,
    ) -> torch.Tensor:
        """Combine a successor state and action without trainable target drift."""

        state = self._project(successor_features, self.snapshot_projection)
        tool = self._project(tool_features, self.tool_projection)
        semantic = F.normalize(state + self.tool_mix * tool, dim=-1)
        return self._append_special_zeros(semantic)

    def velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (self.max_plan_length, self.dim):
            raise ValueError(
                "latent must have shape "
                f"[batch, {self.max_plan_length}, {self.dim}], got {tuple(latent.shape)}"
            )
        if condition.ndim != 2 or condition.shape[-1] != 3 * self.semantic_dim:
            raise ValueError("condition has an incompatible shape")
        hidden = self.input_projection(latent)
        hidden = hidden + self.position.unsqueeze(0)
        hidden = hidden + self.time_embedding(time).unsqueeze(1)
        hidden = hidden + self.condition(condition).unsqueeze(1)
        return self.output(self.transformer(hidden))

    def build_targets(
        self,
        transition_sequences: list[torch.Tensor],
        *,
        pad_weight: float,
    ) -> dict[str, torch.Tensor]:
        if not 0 <= pad_weight <= 1:
            raise ValueError("pad_weight must be in [0, 1]")
        if not transition_sequences:
            raise ValueError("At least one transition sequence is required")
        device = transition_sequences[0].device
        dtype = transition_sequences[0].dtype
        target = (
            self.pad_anchor.to(device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(len(transition_sequences), self.max_plan_length, -1)
            .clone()
        )
        active_mask = torch.zeros(
            (len(transition_sequences), self.max_plan_length), dtype=torch.bool, device=device
        )
        transition_mask = torch.zeros_like(active_mask)
        stop_target = torch.zeros(
            (len(transition_sequences), self.max_plan_length), dtype=dtype, device=device
        )
        weights = torch.full_like(stop_target, float(pad_weight))
        for row, sequence in enumerate(transition_sequences):
            if sequence.ndim != 2 or sequence.shape[-1] != self.dim:
                raise ValueError("Each transition sequence must have shape [horizon, dim]")
            horizon = int(sequence.shape[0])
            if horizon + 1 > self.max_plan_length:
                raise ValueError(
                    f"Path horizon {horizon} plus STOP exceeds max_plan_length="
                    f"{self.max_plan_length}"
                )
            if horizon:
                target[row, :horizon] = sequence
                transition_mask[row, :horizon] = True
            target[row, horizon] = self.stop_anchor.to(device=device, dtype=dtype)
            active_mask[row, : horizon + 1] = True
            stop_target[row, horizon] = 1.0
            weights[row, : horizon + 1] = 1.0
        return {
            "target": target,
            "active_mask": active_mask,
            "transition_mask": transition_mask,
            "stop_target": stop_target,
            "weights": weights,
        }

    def flow_matching_step(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = target.shape[0]
        if time is None:
            time = torch.rand(batch_size, device=target.device)
        if noise is None:
            noise = torch.randn_like(target)
        interpolated = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
        target_velocity = target - noise
        predicted_velocity = self.velocity(interpolated, time, condition)
        predicted_endpoint = interpolated + (1.0 - time[:, None, None]) * predicted_velocity
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
        condition: torch.Tensor,
        *,
        nfe: int = 12,
        noise: torch.Tensor | None = None,
        return_transport: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if nfe <= 0:
            raise ValueError("nfe must be positive")
        batch_size = condition.shape[0]
        latent = (
            torch.randn(batch_size, self.max_plan_length, self.dim, device=condition.device)
            if noise is None
            else noise.clone()
        )
        expected = (batch_size, self.max_plan_length, self.dim)
        if tuple(latent.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(latent.shape)}")
        transport = [latent.clone()] if return_transport else None
        dt = 1.0 / float(nfe)
        for step in range(nfe):
            time = torch.full(
                (batch_size,), step / float(nfe), device=latent.device, dtype=latent.dtype
            )
            latent = latent + dt * self.velocity(latent, time, condition)
            if transport is not None:
                transport.append(latent.clone())
        return latent, transport

    def stop_probability(self, endpoint: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.stop_head(endpoint).squeeze(-1))

    def export_config(self) -> dict[str, Any]:
        return {
            "snapshot_input_dim": self.snapshot_input_dim,
            "context_input_dim": self.context_input_dim,
            "goal_input_dim": self.goal_input_dim,
            "tool_input_dim": self.tool_input_dim,
            "dim": self.dim,
            "max_plan_length": self.max_plan_length,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "projection_seed": self.projection_seed,
            "tool_mix": self.tool_mix,
        }


__all__ = ["SearchStateFlow"]

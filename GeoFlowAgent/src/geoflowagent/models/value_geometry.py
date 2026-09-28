"""Goal-conditioned value geometry for search-distilled tool planning.

This module deliberately consumes *cached/frozen* features.  It does not know how
those features were produced and it never needs to execute a candidate action at
inference time.  Instead, a small transition model predicts the latent successor
for every candidate action and the selected geometry compares that prediction to
the encoded goal.

The public loss helpers distinguish three action states:

``known_action_mask=True``
    The executor/search oracle produced a trustworthy label for the action.
``candidate_mask=True``
    The action is currently selectable according to hard contracts.
otherwise
    The action is unknown or unavailable.  It is neither a negative example nor
    part of the policy normalizer.

This distinction is important for partially explored state graphs: treating an
unexecuted branch as a negative action gives confidently wrong supervision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _positive_scale(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + eps


class GoalPairEnergy(nn.Module):
    """Lower-is-better energy between broadcastable ``[..., D]`` tensors."""

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CosineGoalEnergy(GoalPairEnergy):
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - (F.normalize(source, dim=-1) * F.normalize(target, dim=-1)).sum(-1)


class SquaredEuclideanGoalEnergy(GoalPairEnergy):
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (source - target).square().sum(-1)


class DiagonalMahalanobisGoalEnergy(GoalPairEnergy):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = (source - target) * _positive_scale(self.raw_scale)
        return difference.square().sum(-1)


class LowRankMahalanobisGoalEnergy(GoalPairEnergy):
    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(4, dim // 2)
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.projection = nn.Linear(dim, rank, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.projection(source - target)).square().sum(-1)


class AsymmetricBilinearGoalEnergy(GoalPairEnergy):
    """Unconstrained directional compatibility energy (not a metric)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -torch.einsum("...i,ij,...j->...", source, self.weight, target)


class OrderViolationGoalEnergy(GoalPairEnergy):
    """Directed pseudo-distance for a coordinate-wise information order.

    The convention is that moving from ``source`` to ``target`` should not need
    to decrease a coordinate.  A learned non-negative projection permits the
    model to discover monotone evidence axes while retaining non-negativity.
    """

    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or dim
        self.raw_projection = nn.Parameter(torch.empty(rank, dim))
        nn.init.orthogonal_(self.raw_projection)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        projection = F.softplus(self.raw_projection)
        source_code = F.linear(source, projection)
        target_code = F.linear(target, projection)
        return F.relu(source_code - target_code).square().sum(-1)


class DirectedQuasimetricGoalEnergy(GoalPairEnergy):
    r"""A simple learned, non-negative directed quasimetric.

    For ``u=Lx``, ``v=Ly`` and a vector ``w`` constrained to ``||w|| < 1``:

    ``D(x,y) = ||v-u||_2 + w^T(v-u)``.

    Cauchy--Schwarz guarantees non-negativity.  The norm obeys the triangle
    inequality and the potential term telescopes, so their sum also obeys it.
    The construction is generally asymmetric.  When ``L`` is rank deficient it
    is technically a pseudo-quasimetric because distinct inputs may coincide.
    """

    def __init__(self, dim: int, rank: int | None = None, max_direction_norm: float = 0.95) -> None:
        super().__init__()
        rank = rank or dim
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= max_direction_norm < 1.0:
            raise ValueError("max_direction_norm must be in [0, 1)")
        self.projection = nn.Linear(dim, rank, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.raw_direction = nn.Parameter(torch.zeros(rank))
        self.max_direction_norm = float(max_direction_norm)

    def direction(self) -> torch.Tensor:
        # This smooth radial map is bounded by ``max_direction_norm`` and has an
        # identity-scaled derivative at the zero initialization. A normalized
        # ``raw * tanh(norm) / norm.clamp(...)`` map has a dead gradient at zero,
        # silently reducing every run to a symmetric projected Euclidean norm.
        denominator = torch.sqrt(1.0 + self.raw_direction.square().sum())
        return self.max_direction_norm * self.raw_direction / denominator

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = self.projection(target - source)
        distance = delta.norm(dim=-1) + (delta * self.direction()).sum(-1)
        return distance.clamp_min(0.0)


class PairMLPGoalEnergy(GoalPairEnergy):
    """Flexible capacity-matched control without metric assumptions."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source, target = torch.broadcast_tensors(source, target)
        pair = torch.cat([source, target, target - source, source * target], dim=-1)
        return self.network(pair).squeeze(-1)


class PoincareGoalEnergy(GoalPairEnergy):
    """Symmetric Poincare-ball distance, primarily for hierarchy ablations."""

    def __init__(self, dim: int, curvature: float = 1.0) -> None:
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        self.curvature = float(curvature)
        self.map = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.map.weight)

    def _ball(self, value: torch.Tensor) -> torch.Tensor:
        value = self.map(value)
        norm = value.norm(dim=-1, keepdim=True)
        radius = 0.95 / self.curvature**0.5
        return radius * torch.tanh(norm) * value / norm.clamp_min(1e-8)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = self._ball(source)
        target = self._ball(target)
        difference = (source - target).square().sum(-1)
        source_norm = source.square().sum(-1)
        target_norm = target.square().sum(-1)
        denominator = (1 - self.curvature * source_norm).clamp_min(1e-6) * (
            1 - self.curvature * target_norm
        ).clamp_min(1e-6)
        argument = 1 + 2 * self.curvature * difference / denominator
        return torch.acosh(argument.clamp_min(1 + 1e-6)) / self.curvature**0.5


def make_goal_energy(
    name: str,
    dim: int,
    *,
    rank: int | None = None,
    hidden_dim: int = 128,
) -> GoalPairEnergy:
    """Build one geometry family under a common pairwise API."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"cosine", "raw_cosine"}:
        return CosineGoalEnergy()
    if normalized in {"euclidean", "squared_euclidean"}:
        return SquaredEuclideanGoalEnergy()
    if normalized in {"diagonal_mahalanobis", "diag_mahalanobis"}:
        return DiagonalMahalanobisGoalEnergy(dim)
    if normalized in {"lowrank_mahalanobis", "low_rank_mahalanobis"}:
        return LowRankMahalanobisGoalEnergy(dim, rank=rank)
    if normalized in {"bilinear", "asymmetric_bilinear"}:
        return AsymmetricBilinearGoalEnergy(dim)
    if normalized in {"order", "order_violation"}:
        return OrderViolationGoalEnergy(dim, rank=rank)
    if normalized in {"quasimetric", "directed_quasimetric"}:
        return DirectedQuasimetricGoalEnergy(dim, rank=rank)
    if normalized in {"mlp", "pair_mlp"}:
        return PairMLPGoalEnergy(dim, hidden_dim)
    if normalized in {"poincare", "hyperbolic"}:
        return PoincareGoalEnergy(dim)
    raise ValueError(f"Unknown goal energy {name!r}")


class _StructuredEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim > 0:
            self.network: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, output_dim), nn.GELU()
            )
        else:
            self.network = None
        self.output_dim = output_dim

    def forward(self, value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        shape = (*reference.shape[:-1], self.output_dim)
        if self.network is None:
            return reference.new_zeros(shape)
        if value is None:
            raise ValueError("structured features are required when structured_dim > 0")
        if value.shape[:-1] != reference.shape[:-1] or value.shape[-1] != self.input_dim:
            raise ValueError(
                f"structured tensor must have shape {(*reference.shape[:-1], self.input_dim)}, "
                f"got {tuple(value.shape)}"
            )
        return self.network(value)


class GoalConditionedValueGeometry(nn.Module):
    """Predict search cost, reachability, completion, and latent successors.

    ``tool_views`` and ``structured_tools`` may be global ``[A, D]`` tensors or
    per-example ``[B, A, D]`` tensors.  State and goal views are ``[B, D]``.
    All upstream feature tensors can be precomputed; this module is intentionally
    small enough to train while the encoders remain frozen.
    """

    def __init__(
        self,
        view_dims: Mapping[str, int],
        *,
        structured_dim: int,
        shared_dim: int = 64,
        hidden_dim: int = 128,
        energy: str = "directed_quasimetric",
        energy_rank: int | None = None,
        dropout: float = 0.0,
        return_successor_by_default: bool = True,
    ) -> None:
        super().__init__()
        if not view_dims:
            raise ValueError("view_dims cannot be empty")
        if shared_dim <= 0 or hidden_dim <= 0:
            raise ValueError("shared_dim and hidden_dim must be positive")
        self.view_names = tuple(sorted(view_dims))
        self.view_dims = {name: int(dim) for name, dim in view_dims.items()}
        self.structured_dim = int(structured_dim)
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.energy_name = energy
        self.energy_rank = energy_rank
        self.return_successor_by_default = bool(return_successor_by_default)

        # State and goal share an encoder.  Consequently the selected metric is
        # applied in one coordinate system rather than between unrelated maps.
        self.entity_view_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, shared_dim), nn.GELU())
                for name, dim in self.view_dims.items()
            }
        )
        self.tool_view_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, shared_dim), nn.GELU())
                for name, dim in self.view_dims.items()
            }
        )
        self.entity_structured = _StructuredEncoder(self.structured_dim, shared_dim)
        self.tool_structured = _StructuredEncoder(self.structured_dim, shared_dim)
        fuse_dim = shared_dim * (len(self.view_names) + 1)
        self.entity_fuser = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.tool_fuser = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.successor_head = nn.Sequential(
            nn.LayerNorm(shared_dim * 4),
            nn.Linear(shared_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.energy_head = make_goal_energy(
            energy, shared_dim, rank=energy_rank, hidden_dim=hidden_dim
        )
        pair_dim = shared_dim * 4
        self.value_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.state_reachability_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.action_reachability_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.completion_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_q_scale = nn.Parameter(torch.tensor(0.0))
        self.q_offset = nn.Parameter(torch.tensor(0.0))

    def export_config(self) -> dict[str, object]:
        """Return the constructor arguments required for a portable checkpoint."""

        return {
            "view_dims": dict(self.view_dims),
            "structured_dim": self.structured_dim,
            "shared_dim": self.shared_dim,
            "hidden_dim": self.hidden_dim,
            "energy": self.energy_name,
            "energy_rank": self.energy_rank,
            # Dropout changes training behavior but has no parameters. Checkpoints
            # are loaded for evaluation, so retaining zero is sufficient only when
            # it was actually zero. Store the configured value explicitly.
            "dropout": next(
                (
                    float(module.p)
                    for module in self.modules()
                    if isinstance(module, nn.Dropout)
                ),
                0.0,
            ),
            "return_successor_by_default": self.return_successor_by_default,
        }

    @staticmethod
    def _pair_features(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat([source, target, target - source, source * target], dim=-1)

    def _validate_views(self, views: Mapping[str, torch.Tensor], *, label: str) -> None:
        missing = set(self.view_names) - set(views)
        extra = set(views) - set(self.view_names)
        if missing or extra:
            raise ValueError(f"{label} view mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        for name in self.view_names:
            if views[name].shape[-1] != self.view_dims[name]:
                raise ValueError(
                    f"{label}[{name!r}] last dimension must be {self.view_dims[name]}, "
                    f"got {views[name].shape[-1]}"
                )

    def encode_entity(
        self,
        views: Mapping[str, torch.Tensor],
        structured: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode either a state or goal using the shared entity encoder."""

        self._validate_views(views, label="entity")
        reference = views[self.view_names[0]]
        if reference.ndim != 2:
            raise ValueError("state/goal views must have shape [B, D]")
        pieces = [self.entity_view_projectors[name](views[name]) for name in self.view_names]
        pieces.append(self.entity_structured(structured, reference))
        return self.entity_fuser(torch.cat(pieces, dim=-1))

    def encode_tools(
        self,
        views: Mapping[str, torch.Tensor],
        structured: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode global ``[A,D]`` or batched ``[B,A,D]`` action descriptions."""

        self._validate_views(views, label="tool")
        reference = views[self.view_names[0]]
        if reference.ndim not in {2, 3}:
            raise ValueError("tool views must have shape [A, D] or [B, A, D]")
        if reference.ndim == 3 and reference.shape[0] != batch_size:
            raise ValueError("batched tool views must share the state batch dimension")
        pieces = [self.tool_view_projectors[name](views[name]) for name in self.view_names]
        pieces.append(self.tool_structured(structured, reference))
        encoded = self.tool_fuser(torch.cat(pieces, dim=-1))
        if encoded.ndim == 2:
            encoded = encoded.unsqueeze(0).expand(batch_size, -1, -1)
        return encoded

    @staticmethod
    def _masked_q(
        q_value: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        action_reachability_logit: torch.Tensor,
        *,
        mask_unreachable: bool,
        reachability_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = torch.ones_like(q_value, dtype=torch.bool)
        if candidate_mask is not None:
            if candidate_mask.shape != q_value.shape:
                raise ValueError("candidate_mask must have the same [B, A] shape as q")
            allowed &= candidate_mask.bool()
        if mask_unreachable:
            allowed &= action_reachability_logit.sigmoid() >= reachability_threshold
        return q_value.masked_fill(~allowed, torch.inf), allowed

    def forward(
        self,
        state_views: Mapping[str, torch.Tensor],
        goal_views: Mapping[str, torch.Tensor],
        tool_views: Mapping[str, torch.Tensor],
        structured_state: torch.Tensor | None,
        structured_goal: torch.Tensor | None,
        structured_tools: torch.Tensor | None,
        *,
        candidate_mask: torch.Tensor | None = None,
        mask_unreachable: bool = False,
        reachability_threshold: float = 0.5,
        return_successor: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_entity(state_views, structured_state)
        goal = self.encode_entity(goal_views, structured_goal)
        if state.shape != goal.shape:
            raise ValueError("state and goal batches must have the same shape")
        tools = self.encode_tools(tool_views, structured_tools, batch_size=state.shape[0])

        expanded_state = state[:, None, :].expand_as(tools)
        transition_features = self._pair_features(expanded_state, tools)
        successor = expanded_state + self.successor_head(transition_features)
        expanded_goal = goal[:, None, :].expand_as(successor)
        energy = self.energy_head(successor, expanded_goal)

        # A monotone calibration permits comparison to weighted search cost while
        # preserving the ranking induced by every geometry family.
        q_value = F.softplus(_positive_scale(self.raw_q_scale) * energy + self.q_offset)
        state_goal_features = self._pair_features(state, goal)
        value = F.softplus(self.value_head(state_goal_features).squeeze(-1))
        state_reachability_logit = self.state_reachability_head(state_goal_features).squeeze(-1)
        completion_logit = self.completion_head(state_goal_features).squeeze(-1)
        action_goal_features = self._pair_features(successor, expanded_goal)
        action_reachability_logit = self.action_reachability_head(action_goal_features).squeeze(-1)
        masked_q, allowed_mask = self._masked_q(
            q_value,
            candidate_mask,
            action_reachability_logit,
            mask_unreachable=mask_unreachable,
            reachability_threshold=reachability_threshold,
        )
        # Soft reachability is part of the policy score; the optional hard mask
        # is used at deployment to prevent a predicted-unreachable transition.
        policy_logits = -q_value + F.logsigmoid(action_reachability_logit)
        policy_logits = policy_logits.masked_fill(~allowed_mask, -torch.inf)

        output = {
            "energy": energy,
            "q": q_value,
            "masked_q": masked_q,
            "policy_logits": policy_logits,
            "value": value,
            "state_reachability_logit": state_reachability_logit,
            "reachability_logit": action_reachability_logit,
            "action_reachability_logit": action_reachability_logit,
            "completion_logit": completion_logit,
            "stop_logit": completion_logit,
            "state_latent": state,
            "goal_latent": goal,
            "tool_latent": tools,
            "allowed_action_mask": allowed_mask,
        }
        if return_successor if return_successor is not None else self.return_successor_by_default:
            output["successor_latent"] = successor
        return output

    @torch.no_grad()
    def select_action(
        self,
        output: Mapping[str, torch.Tensor],
        *,
        candidate_mask: torch.Tensor | None = None,
        reachability_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Select a reachable minimum-Q action, returning ``-1`` if none exists."""

        q_value = output["q"]
        reachability = output["action_reachability_logit"]
        masked_q, allowed = self._masked_q(
            q_value,
            candidate_mask,
            reachability,
            mask_unreachable=True,
            reachability_threshold=reachability_threshold,
        )
        selected = masked_q.argmin(dim=-1)
        return torch.where(allowed.any(dim=-1), selected, torch.full_like(selected, -1))


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _finite_action_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    known_action_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != known_action_mask.shape:
        raise ValueError("action prediction, target, and known_action_mask must share [B, A]")
    mask = known_action_mask.bool() & torch.isfinite(target)
    if candidate_mask is not None:
        if candidate_mask.shape != prediction.shape:
            raise ValueError("candidate_mask must share the action prediction shape")
        mask &= candidate_mask.bool()
    return mask


def masked_action_distribution_loss(
    logits: torch.Tensor,
    known_action_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    optimal_action_mask: torch.Tensor | None = None,
    soft_action_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Set-valued NLL or soft-label cross entropy over known candidate actions.

    Unknown actions are excluded from both numerator and denominator.  Rows with
    no known candidates (or no target mass) are safely skipped.
    """

    if logits.shape != known_action_mask.shape:
        raise ValueError("logits and known_action_mask must have the same shape")
    if optimal_action_mask is None and soft_action_target is None:
        return _differentiable_zero(logits)
    allowed = known_action_mask.bool()
    if candidate_mask is not None:
        if candidate_mask.shape != logits.shape:
            raise ValueError("candidate_mask must have the same shape as logits")
        allowed &= candidate_mask.bool()
    masked_logits = logits.masked_fill(~allowed, -torch.inf)
    has_denominator = allowed.any(dim=-1)

    if soft_action_target is not None:
        if soft_action_target.shape != logits.shape:
            raise ValueError("soft_action_target must have the same shape as logits")
        target = torch.where(
            allowed & torch.isfinite(soft_action_target),
            soft_action_target.clamp_min(0.0),
            torch.zeros_like(soft_action_target),
        )
        mass = target.sum(dim=-1)
        rows = has_denominator & (mass > 0)
        if not rows.any():
            return _differentiable_zero(logits)
        normalized_target = target[rows] / mass[rows, None]
        row_logits = masked_logits[rows]
        log_denominator = torch.logsumexp(row_logits, dim=-1, keepdim=True)
        log_probability = row_logits - log_denominator
        # Indexing only positive target entries avoids the undefined 0 * -inf.
        positive = normalized_target > 0
        return -(normalized_target[positive] * log_probability[positive]).sum() / rows.sum()

    assert optimal_action_mask is not None
    if optimal_action_mask.shape != logits.shape:
        raise ValueError("optimal_action_mask must have the same shape as logits")
    optimal = optimal_action_mask.bool() & allowed
    rows = has_denominator & optimal.any(dim=-1)
    if not rows.any():
        return _differentiable_zero(logits)
    numerator = torch.logsumexp(masked_logits[rows].masked_fill(~optimal[rows], -torch.inf), dim=-1)
    denominator = torch.logsumexp(masked_logits[rows], dim=-1)
    return -(numerator - denominator).mean()


def masked_regret_ranking_loss(
    q_value: torch.Tensor,
    regret: torch.Tensor,
    known_action_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    margin: float = 0.1,
    scale_margin_by_regret: bool = True,
) -> torch.Tensor:
    """Rank every known lower-regret action ahead of higher-regret actions."""

    mask = _finite_action_mask(q_value, regret, known_action_mask, candidate_mask)
    # Pair (i,j): i has strictly lower oracle regret than j.
    regret_gap = regret.unsqueeze(1) - regret.unsqueeze(2)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & (regret_gap > 1e-7)
    if not pair_mask.any():
        return _differentiable_zero(q_value)
    required_margin = margin * regret_gap.clamp_min(1.0) if scale_margin_by_regret else margin
    violation = F.relu(required_margin + q_value.unsqueeze(2) - q_value.unsqueeze(1))
    return violation[pair_mask].mean()


@dataclass(frozen=True)
class ValueGeometryLossWeights:
    action: float = 1.0
    q: float = 1.0
    value: float = 0.5
    regret: float = 0.5
    bellman: float = 0.5
    transition: float = 0.25
    action_reachability: float = 0.5
    state_reachability: float = 0.25
    completion: float = 0.5


def search_distillation_loss(
    output: Mapping[str, torch.Tensor],
    known_action_mask: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    optimal_action_mask: torch.Tensor | None = None,
    soft_action_target: torch.Tensor | None = None,
    q_target: torch.Tensor | None = None,
    value_target: torch.Tensor | None = None,
    value_known_mask: torch.Tensor | None = None,
    regret_target: torch.Tensor | None = None,
    edge_cost: torch.Tensor | None = None,
    successor_value_target: torch.Tensor | None = None,
    transition_target: torch.Tensor | None = None,
    action_reachability_target: torch.Tensor | None = None,
    state_reachability_target: torch.Tensor | None = None,
    state_reachability_known_mask: torch.Tensor | None = None,
    completion_target: torch.Tensor | None = None,
    completion_known_mask: torch.Tensor | None = None,
    weights: ValueGeometryLossWeights | None = None,
    action_temperature: float = 1.0,
    ranking_margin: float = 0.1,
    discount: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compose robust search-distillation losses.

    Every action-level term is intersected with ``known_action_mask`` and the
    optional hard ``candidate_mask``.  NaN/inf oracle placeholders are ignored.
    Targets used on the right-hand side of Bellman/transition regression are
    detached so this function cannot accidentally train a target network/cache.
    """

    if action_temperature <= 0:
        raise ValueError("action_temperature must be positive")
    weights = weights or ValueGeometryLossWeights()
    q_value = output["q"]
    if known_action_mask.shape != q_value.shape:
        raise ValueError("known_action_mask must have the same [B, A] shape as q")
    policy_logits = output.get("policy_logits", -q_value) / action_temperature
    losses: dict[str, torch.Tensor] = {}
    losses["action"] = masked_action_distribution_loss(
        policy_logits,
        known_action_mask,
        candidate_mask=candidate_mask,
        optimal_action_mask=optimal_action_mask,
        soft_action_target=soft_action_target,
    )

    if q_target is None:
        losses["q"] = _differentiable_zero(q_value)
    else:
        mask = _finite_action_mask(q_value, q_target, known_action_mask, candidate_mask)
        safe_target = torch.where(mask, q_target, q_value.detach())
        elementwise = F.smooth_l1_loss(q_value, safe_target, reduction="none")
        losses["q"] = elementwise[mask].mean() if mask.any() else _differentiable_zero(q_value)

    value = output["value"]
    if value_target is None:
        losses["value"] = _differentiable_zero(value)
    else:
        if value.shape != value_target.shape:
            raise ValueError("value and value_target must have shape [B]")
        value_mask = torch.isfinite(value_target)
        if value_known_mask is not None:
            value_mask &= value_known_mask.bool()
        safe_target = torch.where(value_mask, value_target, value.detach())
        elementwise = F.smooth_l1_loss(value, safe_target, reduction="none")
        losses["value"] = (
            elementwise[value_mask].mean() if value_mask.any() else _differentiable_zero(value)
        )

    if regret_target is None:
        losses["regret"] = _differentiable_zero(q_value)
    else:
        losses["regret"] = masked_regret_ranking_loss(
            q_value,
            regret_target,
            known_action_mask,
            candidate_mask=candidate_mask,
            margin=ranking_margin,
        )

    if edge_cost is None or successor_value_target is None:
        losses["bellman"] = _differentiable_zero(q_value)
    else:
        if edge_cost.shape != q_value.shape or successor_value_target.shape != q_value.shape:
            raise ValueError("edge_cost and successor_value_target must have shape [B, A]")
        bellman_target = edge_cost + discount * successor_value_target
        mask = _finite_action_mask(q_value, bellman_target, known_action_mask, candidate_mask)
        safe_target = torch.where(mask, bellman_target.detach(), q_value.detach())
        elementwise = F.smooth_l1_loss(q_value, safe_target, reduction="none")
        losses["bellman"] = (
            elementwise[mask].mean() if mask.any() else _differentiable_zero(q_value)
        )

    successor = output.get("successor_latent")
    if transition_target is None or successor is None:
        losses["transition"] = _differentiable_zero(q_value)
    else:
        if successor.shape != transition_target.shape:
            raise ValueError("transition_target must match successor_latent [B, A, D]")
        finite_target = torch.isfinite(transition_target).all(dim=-1)
        action_mask = known_action_mask.bool() & finite_target
        if candidate_mask is not None:
            action_mask &= candidate_mask.bool()
        safe_target = torch.where(
            action_mask.unsqueeze(-1), transition_target.detach(), successor.detach()
        )
        elementwise = F.smooth_l1_loss(successor, safe_target, reduction="none").mean(-1)
        losses["transition"] = (
            elementwise[action_mask].mean()
            if action_mask.any()
            else _differentiable_zero(successor)
        )

    action_reachability = output["action_reachability_logit"]
    if action_reachability_target is None:
        losses["action_reachability"] = _differentiable_zero(action_reachability)
    else:
        mask = _finite_action_mask(
            action_reachability,
            action_reachability_target,
            known_action_mask,
            candidate_mask,
        )
        safe_target = torch.where(
            mask, action_reachability_target, torch.zeros_like(action_reachability_target)
        ).to(action_reachability.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            action_reachability, safe_target, reduction="none"
        )
        losses["action_reachability"] = (
            elementwise[mask].mean()
            if mask.any()
            else _differentiable_zero(action_reachability)
        )

    state_reachability = output["state_reachability_logit"]
    if state_reachability_target is None:
        losses["state_reachability"] = _differentiable_zero(state_reachability)
    else:
        if state_reachability.shape != state_reachability_target.shape:
            raise ValueError("state reachability tensors must have shape [B]")
        mask = torch.isfinite(state_reachability_target)
        if state_reachability_known_mask is not None:
            mask &= state_reachability_known_mask.bool()
        safe_target = torch.where(
            mask, state_reachability_target, torch.zeros_like(state_reachability_target)
        ).to(state_reachability.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            state_reachability, safe_target, reduction="none"
        )
        losses["state_reachability"] = (
            elementwise[mask].mean()
            if mask.any()
            else _differentiable_zero(state_reachability)
        )

    completion = output["completion_logit"]
    if completion_target is None:
        losses["completion"] = _differentiable_zero(completion)
    else:
        if completion.shape != completion_target.shape:
            raise ValueError("completion tensors must have shape [B]")
        mask = torch.isfinite(completion_target)
        if completion_known_mask is not None:
            mask &= completion_known_mask.bool()
        safe_target = torch.where(
            mask, completion_target, torch.zeros_like(completion_target)
        ).to(completion.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            completion, safe_target, reduction="none"
        )
        losses["completion"] = (
            elementwise[mask].mean() if mask.any() else _differentiable_zero(completion)
        )

    total = (
        weights.action * losses["action"]
        + weights.q * losses["q"]
        + weights.value * losses["value"]
        + weights.regret * losses["regret"]
        + weights.bellman * losses["bellman"]
        + weights.transition * losses["transition"]
        + weights.action_reachability * losses["action_reachability"]
        + weights.state_reachability * losses["state_reachability"]
        + weights.completion * losses["completion"]
    )
    losses["total"] = total
    return losses


__all__ = [
    "AsymmetricBilinearGoalEnergy",
    "CosineGoalEnergy",
    "DiagonalMahalanobisGoalEnergy",
    "DirectedQuasimetricGoalEnergy",
    "GoalConditionedValueGeometry",
    "GoalPairEnergy",
    "LowRankMahalanobisGoalEnergy",
    "OrderViolationGoalEnergy",
    "PairMLPGoalEnergy",
    "PoincareGoalEnergy",
    "SquaredEuclideanGoalEnergy",
    "ValueGeometryLossWeights",
    "make_goal_energy",
    "masked_action_distribution_loss",
    "masked_regret_ranking_loss",
    "search_distillation_loss",
]

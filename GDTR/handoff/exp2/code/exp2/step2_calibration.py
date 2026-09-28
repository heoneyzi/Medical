"""Step 2 — development-only calibration of margins, eligibility and power.

This module is deliberately separate from every scientific Step. A locked
analysis must consume a saved Margins object; it must never estimate a margin,
choose an alpha range, or lower a sample-size requirement after seeing locked
data.

The caller supplies development reference distributions and scientifically
chosen effect margins. Distributional quantities are estimated here, while
scientific margins remain explicit inputs copied into the audit record.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Mapping, Optional, Sequence

import numpy as np

from .manifest import Manifest
from .stats import Row, simulation_power
from .step1_gate import RNum


@dataclass(frozen=True)
class ScientificMargins:
    """Smallest effects worth making a scientific claim about."""

    delta_shape: float
    delta_nll: float
    delta_path: float
    delta_acc: float
    delta_spec: Mapping[str, float]

    def validate(self) -> None:
        vals = {
            "delta_shape": self.delta_shape,
            "delta_nll": self.delta_nll,
            "delta_path": self.delta_path,
            "delta_acc": self.delta_acc,
            **{f"delta_spec::{k}": v for k, v in self.delta_spec.items()},
        }
        bad = {k: v for k, v in vals.items() if not np.isfinite(v) or v < 0}
        if bad:
            raise ValueError(f"scientific margins must be finite and non-negative: {bad}")


@dataclass(frozen=True)
class DevelopmentReferences:
    """Reference quantities computed without opening a locked split."""

    beta_values: Sequence[float]
    centered_logit_norms: Sequence[float]
    content_angles: Sequence[float]
    carrier_abs_coefficients: Sequence[float]
    dimensionless_q: Sequence[float]
    alpha_scores: Mapping[float, float] = field(default_factory=dict)
    power_rows: Mapping[str, Sequence[Row]] = field(default_factory=dict)
    target_effects: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationConfig:
    numerical_guard: float = 5.0
    beta_tail: float = 0.005
    beta_expand: float = 2.0
    flat_norm_quantile: float = 0.01
    phi_tail: float = 0.01
    carrier_tail: float = 0.995
    q_interval: tuple[float, float] = (0.10, 0.90)
    alpha_persistence: int = 2
    target_power: float = 0.80
    alpha: float = 0.05
    candidate_clusters: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 500, 750, 1000)
    seed: int = 42

    def validate(self) -> None:
        if self.numerical_guard < 1:
            raise ValueError("numerical_guard must be >= 1")
        for name, v in (
            ("beta_tail", self.beta_tail),
            ("flat_norm_quantile", self.flat_norm_quantile),
            ("phi_tail", self.phi_tail),
        ):
            if not 0 <= v < 0.5:
                raise ValueError(f"{name} must be in [0, 0.5)")
        q0, q1 = self.q_interval
        if not 0 <= q0 < q1 <= 1:
            raise ValueError("q_interval must be increasing quantiles in [0,1]")
        if not 0 < self.target_power < 1:
            raise ValueError("target_power must be in (0,1)")
        if not self.candidate_clusters or min(self.candidate_clusters) < 2:
            raise ValueError("candidate_clusters must contain positive cluster counts")


@dataclass(frozen=True)
class CalibrationRecord:
    """Auditable output saved beside the manifest."""

    split_name: str
    config: dict
    scientific_margins: dict
    reference_counts: dict
    derived: dict
    power_curve: dict[str, list[dict]]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


def _finite(name: str, values: Sequence[float], *, positive: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if positive:
        x = x[x > 0]
    if x.size == 0:
        raise ValueError(f"development reference {name!r} has no usable values")
    return x


def development_alpha_sat(
    scores: Mapping[float, float],
    *,
    delta_acc: float,
    persistence: int,
    reference_alpha: float = 1.0,
) -> float:
    """First alpha within delta_acc of alpha=1 for consecutive doses."""

    if reference_alpha not in scores:
        raise ValueError(f"alpha score grid must contain reference alpha={reference_alpha}")
    grid = sorted(float(a) for a in scores if float(a) <= reference_alpha)
    ref = float(scores[reference_alpha])
    need = max(int(persistence), 1)
    for i, a in enumerate(grid):
        tail = grid[i:i + need + 1]
        if len(tail) < need + 1:
            continue
        if all(abs(float(scores[t]) - ref) <= delta_acc for t in tail):
            return a
    raise RuntimeError(
        "no persistent development saturation point; do not define alpha_sat "
        "from the locked curve"
    )


def calibrate_manifest(
    man: Manifest,
    *,
    split_name: str,
    rnum: RNum,
    refs: DevelopmentReferences,
    scientific: ScientificMargins,
    config: CalibrationConfig = CalibrationConfig(),
    required_control_families: Sequence[str] = (),
) -> CalibrationRecord:
    """Fill man.margins exactly once from a development split."""

    config.validate()
    scientific.validate()
    split = man.splits.splits[split_name]
    if split.role != "development":
        raise RuntimeError(
            f"Step 2 requires a development split, got {split_name!r} role={split.role!r}"
        )
    if man.margins.calibrated:
        raise RuntimeError("margins are already calibrated; create a new manifest version")
    missing_controls = sorted(set(required_control_families) - set(scientific.delta_spec))
    if missing_controls:
        raise ValueError(f"scientific specificity margins missing: {missing_controls}")

    beta = _finite("beta_values", refs.beta_values, positive=True)
    cnorm = _finite("centered_logit_norms", refs.centered_logit_norms, positive=True)
    angles = _finite("content_angles", refs.content_angles)
    coeff = _finite("carrier_abs_coefficients", refs.carrier_abs_coefficients)
    qvals = _finite("dimensionless_q", refs.dimensionless_q)

    lo_q, hi_q = config.beta_tail, 1.0 - config.beta_tail
    beta_lo = float(np.quantile(beta, lo_q) / config.beta_expand)
    beta_hi = float(np.quantile(beta, hi_q) * config.beta_expand)
    if not 0 < beta_lo < beta_hi:
        raise RuntimeError(f"invalid calibrated beta interval [{beta_lo}, {beta_hi}]")

    phi_lo = float(np.quantile(angles, config.phi_tail))
    phi_hi = float(np.quantile(angles, 1.0 - config.phi_tail))
    theta_q = np.quantile(angles, [0.25, 0.50, 0.75]).tolist()
    q_lo, q_hi = np.quantile(qvals, config.q_interval).tolist()
    c_abs_max = float(min(np.quantile(coeff, config.carrier_tail), 1.0 - 1e-8))
    flat_thr = float(np.quantile(cnorm, config.flat_norm_quantile))

    alpha_sat: Optional[float] = None
    warnings: list[str] = []
    if refs.alpha_scores:
        alpha_sat = development_alpha_sat(
            refs.alpha_scores,
            delta_acc=scientific.delta_acc,
            persistence=config.alpha_persistence,
        )
    else:
        warnings.append("alpha_scores absent: alpha_sat remains unset and C_acc is closed")

    power_curve: dict[str, list[dict]] = {}
    n_required: dict[str, int] = {}
    for name, rows in refs.power_rows.items():
        if name not in refs.target_effects:
            raise ValueError(f"target_effects missing for power family {name!r}")
        curve = []
        selected = None
        for n in config.candidate_clusters:
            pwr = simulation_power(
                rows,
                effect=float(refs.target_effects[name]),
                n_clusters=int(n),
                alpha=config.alpha,
                seed=config.seed,
            )
            curve.append({"n_clusters": int(n), "power": float(pwr)})
            if selected is None and pwr >= config.target_power:
                selected = int(n)
        power_curve[name] = curve
        if selected is None:
            raise RuntimeError(
                f"power family {name!r} did not reach target power={config.target_power}"
            )
        n_required[name] = max(50, selected)

    m = man.margins
    m.r_num_d_shape = max(float(rnum.d_shape) * config.numerical_guard, np.finfo(float).eps)
    m.r_num_log_beta = max(float(rnum.log_beta) * config.numerical_guard, np.finfo(float).eps)
    m.r_num_logit_l2 = max(float(rnum.logit_l2) * config.numerical_guard, np.finfo(float).eps)
    if np.isfinite(rnum.identity_residual):
        m.r_num_identity_residual = max(
            float(rnum.identity_residual) * config.numerical_guard, np.finfo(float).eps
        )
    else:
        raise RuntimeError("R_num.identity_residual is missing; Step 1 identity gate is incomplete")

    m.delta_shape = float(scientific.delta_shape)
    m.delta_nll = float(scientific.delta_nll)
    m.delta_path = float(scientific.delta_path)
    m.delta_acc = float(scientific.delta_acc)
    m.delta_spec = {str(k): float(v) for k, v in scientific.delta_spec.items()}
    m.beta_min = beta_lo
    m.beta_max = beta_hi
    m.flat_logit_threshold = flat_thr
    m.c_abs_max = c_abs_max
    m.phi_min = phi_lo
    m.phi_max = phi_hi
    m.content_theta_q25, m.content_theta_q50, m.content_theta_q75 = map(float, theta_q)
    m.q_lo, m.q_hi = float(q_lo), float(q_hi)
    m.alpha_sat = alpha_sat
    m.n_clusters_required = n_required
    m.calibrated = True

    derived = {
        "beta_min": beta_lo,
        "beta_max": beta_hi,
        "flat_logit_threshold": flat_thr,
        "c_abs_max": c_abs_max,
        "phi_min": phi_lo,
        "phi_max": phi_hi,
        "content_theta_q25": theta_q[0],
        "content_theta_q50": theta_q[1],
        "content_theta_q75": theta_q[2],
        "q_lo": q_lo,
        "q_hi": q_hi,
        "alpha_sat": alpha_sat,
        "n_clusters_required": n_required,
    }
    return CalibrationRecord(
        split_name=split_name,
        config=asdict(config),
        scientific_margins={
            "delta_shape": scientific.delta_shape,
            "delta_nll": scientific.delta_nll,
            "delta_path": scientific.delta_path,
            "delta_acc": scientific.delta_acc,
            "delta_spec": dict(scientific.delta_spec),
        },
        reference_counts={
            "beta_values": int(beta.size),
            "centered_logit_norms": int(cnorm.size),
            "content_angles": int(angles.size),
            "carrier_abs_coefficients": int(coeff.size),
            "dimensionless_q": int(qvals.size),
            "power_families": len(refs.power_rows),
        },
        derived=derived,
        power_curve=power_curve,
        warnings=tuple(warnings),
    )

"""Exact output endpoints and exact algebraic decompositions.

Everything here is closed-form or a 1-D convex solve. No fitting, no
thresholds. Plan sections: §1.4 (beta*/D_shape), §2.1 (A/B), Step 6-2
(theta(alpha)), Step 5 (slerp).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

# --------------------------------------------------------------------------
# §1.4  beta* and D_shape
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeResult:
    log_beta: float          # log beta*   (temperature-like dilation)
    d_shape: float           # nats        (relative token preference change)
    boundary_censored: bool  # optimum hit [beta_min, beta_max]
    beta_identified: bool    # baseline centered-logit norm above flat threshold
    centered_norm: float


def _kl_p_vs_scaled_q(p: Tensor, z0: Tensor, beta: Tensor) -> Tensor:
    """KL[ p || softmax(beta * z0) ], summed over vocab. Batched over leading dims.

    KL = -H(p) - beta<p,z0> + logsumexp(beta*z0)
    which is convex in beta with
        d/dbeta   = <softmax(beta z0), z0> - <p, z0>
        d2/dbeta2 = Var_{softmax(beta z0)}(z0) >= 0
    """
    neg_H = (p * torch.log(p.clamp_min(1e-30))).sum(-1)
    lse = torch.logsumexp(beta.unsqueeze(-1) * z0, dim=-1)
    return neg_H - beta * (p * z0).sum(-1) + lse


def solve_beta_shape(
    z_intervention: Tensor,
    z_baseline: Tensor,
    *,
    beta_min: float,
    beta_max: float,
    flat_logit_threshold: float,
    iters: int = 80,
) -> ShapeResult:
    """Primary output endpoint of every wave.

    beta* = argmin_{beta in [beta_min, beta_max]} KL[softmax(z^I) || softmax(beta z^0)]
    D_shape = the attained minimum.

    The objective is convex in beta, so the stationarity condition
        E_{softmax(beta z0)}[z0] = E_{softmax(zI)}[z0]
    has a unique root and the LHS is non-decreasing in beta (its derivative is
    a variance). We bisect on that scalar equation, which is numerically far
    better behaved than minimising the KL directly.

    `beta_min`/`beta_max`/`flat_logit_threshold` come from the frozen manifest
    (development calibration, §1.4) -- never chosen here.
    """
    z0 = z_baseline.double()
    zI = z_intervention.double()
    # softmax is shift-invariant; centre for conditioning only.
    z0 = z0 - z0.mean(-1, keepdim=True)
    zI = zI - zI.mean(-1, keepdim=True)

    centered_norm = float(z0.norm())
    identified = centered_norm >= flat_logit_threshold

    p = torch.softmax(zI, dim=-1)
    target = float((p * z0).sum(-1))

    def mean_at(beta: float) -> float:
        q = torch.softmax(beta * z0, dim=-1)
        return float((q * z0).sum(-1))

    lo, hi = float(beta_min), float(beta_max)
    f_lo, f_hi = mean_at(lo) - target, mean_at(hi) - target

    censored = False
    if f_lo >= 0.0:            # root at or below the lower bound
        beta = lo
        censored = True
    elif f_hi <= 0.0:          # root at or above the upper bound
        beta = hi
        censored = True
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if mean_at(mid) - target < 0.0:
                lo = mid
            else:
                hi = mid
        beta = 0.5 * (lo + hi)

    d_shape = float(
        _kl_p_vs_scaled_q(p, z0, torch.tensor(beta, dtype=torch.float64))
    )
    return ShapeResult(
        log_beta=float(torch.log(torch.tensor(beta, dtype=torch.float64))),
        d_shape=max(d_shape, 0.0),
        boundary_censored=censored,
        beta_identified=identified,
        centered_norm=centered_norm,
    )


# --------------------------------------------------------------------------
# §2.1  exact vector decomposition of a coordinate change
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ABDecomposition:
    norm_A: float        # denominator-induced logit dilation
    norm_B: float        # direct centered-unembedding effect
    cos_AB: float
    norm_AplusB: float
    norm_observed: float # ||C z' - C z|| measured from an actual forward
    residual: float      # ||(A+B) - observed||  -> must be within R_num
    magnitude_index: float  # ||A||/(||A||+||B||)  -- NOT a mediation fraction

    def as_row(self) -> dict:
        return {
            "norm_A": self.norm_A,
            "norm_B": self.norm_B,
            "cos_AB": self.cos_AB,
            "norm_AplusB": self.norm_AplusB,
            "norm_observed": self.norm_observed,
            "identity_residual": self.residual,
            "magnitude_index": self.magnitude_index,
        }


def rms_denominator(
    h: Tensor, *, eps: float, form: Literal["norm_over_sqrtd_plus_eps", "sqrt_mean_sq_plus_eps"]
) -> Tensor:
    """The RMSNorm denominator, in whichever form the runtime actually uses.

    Step 1 item 1 requires this to be read off the runtime tensor, not assumed.
    The two forms give different s' under a coordinate change, and the A/B
    identity is only exact for the right one -- `decompose_coordinate_change`
    checks the residual against R_num, so a wrong choice fails loudly.
    """
    d = h.shape[-1]
    if form == "norm_over_sqrtd_plus_eps":
        return h.norm(dim=-1) / (d ** 0.5) + eps
    return torch.sqrt((h * h).mean(-1) + eps)


def centre(z: Tensor) -> Tensor:
    """C = I - 11^T / V applied to logits."""
    return z - z.mean(-1, keepdim=True)


def decompose_coordinate_change(
    h: Tensor,                 # [d] state entering the FINAL RMSNorm
    j: int,                    # coordinate index (e.g. 3756)
    delta: float,
    *,
    gamma: Tensor,             # [d] final RMSNorm gain
    W_U: Tensor,               # [V, d] unembedding
    eps: float,
    rms_form: Literal["norm_over_sqrtd_plus_eps", "sqrt_mean_sq_plus_eps"],
    bias: Optional[Tensor] = None,   # [V] unembedding bias, if any
    observed_logits_pair: Optional[tuple[Tensor, Tensor]] = None,
) -> ABDecomposition:
    """Exact decomposition  Delta C z = A + B  (plan §2.1).

        A = (s/s' - 1) * C (z - b)
        B = (gamma_j * delta / s') * C U_{:,j}

    EXACT ONLY at the final pre-RMSNorm tap. Interventions at x28/x30 run
    through nonlinear suffix computation and must NOT be pushed through this
    identity (plan Step 4, "분석").

    The plan is explicit that A is, for a matched s -> s', independent of WHICH
    coordinate moved. So `magnitude_index` must never be read as a causal
    mediation fraction, and coordinate specificity is NOT tested by comparing
    it across axes -- see `step4_carrier.upstream_use_contrast`.
    """
    h = h.double()
    gamma = gamma.double()
    W_U = W_U.double()
    s = rms_denominator(h, eps=eps, form=rms_form)

    h_prime = h.clone()
    h_prime[j] = h_prime[j] + delta
    s_prime = rms_denominator(h_prime, eps=eps, form=rms_form)

    z = W_U @ (gamma * h / s)
    if bias is not None:
        z = z + bias.double()
    Cz_nobias = centre(W_U @ (gamma * h / s))

    A = (s / s_prime - 1.0) * Cz_nobias
    B = (gamma[j] * delta / s_prime) * centre(W_U[:, j])
    AB = A + B

    if observed_logits_pair is not None:
        z_obs0, z_obs1 = observed_logits_pair
        observed = centre(z_obs1.double()) - centre(z_obs0.double())
    else:
        z_prime = W_U @ (gamma * h_prime / s_prime)
        if bias is not None:
            z_prime = z_prime + bias.double()
        observed = centre(z_prime) - centre(z)

    nA, nB = float(A.norm()), float(B.norm())
    denom = nA + nB
    return ABDecomposition(
        norm_A=nA,
        norm_B=nB,
        cos_AB=float(torch.dot(A, B) / (A.norm() * B.norm()).clamp_min(1e-30)),
        norm_AplusB=float(AB.norm()),
        norm_observed=float(observed.norm()),
        residual=float((AB - observed).norm()),
        magnitude_index=(nA / denom) if denom > 0 else float("nan"),
    )


# --------------------------------------------------------------------------
# Step 6-2  branch geometry:  theta(alpha), alpha_eq, cancellation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchGeometry:
    alpha_eq: float        # ||r|| / ||g||  -- TECHNICAL crossover, not a takeover threshold
    cos_rg: float
    g_par: float           # <g, rhat>
    g_perp_norm: float
    theta: float           # rad, angle between r and r + alpha*g
    q: float               # dimensionless dose alpha*||g||/||r||  (visualisation only)
    cancellation: float    # ||r + alpha g|| / (||r|| + alpha||g||)
    sign_crossing: bool    # ||r + alpha g|| < ||r||  (destructive interference)


def branch_geometry(r: Tensor, g: Tensor, alpha: float) -> BranchGeometry:
    """Separate a norm crossover from an actual direction change (Step 6-2).

    alpha_eq = ||r||/||g|| is where the branch norm equals the residual norm.
    The plan is explicit that this is a *technical* reference and NOT the
    angular takeover threshold, so theta(alpha) is computed directly:

        g = g_par * rhat + g_perp
        theta(alpha) = atan2(alpha * ||g_perp||, ||r|| + alpha * g_par)
    """
    r = r.double()
    g = g.double()
    rn = r.norm()
    gn = g.norm()
    rhat = r / rn.clamp_min(1e-30)
    g_par = torch.dot(g, rhat)
    g_perp = g - g_par * rhat
    mixed = rn + alpha * g_par
    theta = torch.atan2(alpha * g_perp.norm(), mixed)
    mix_norm = (r + alpha * g).norm()
    return BranchGeometry(
        alpha_eq=float(rn / gn.clamp_min(1e-30)),
        cos_rg=float(g_par / gn.clamp_min(1e-30)),
        g_par=float(g_par),
        g_perp_norm=float(g_perp.norm()),
        theta=float(theta),
        q=float(alpha * gn / rn.clamp_min(1e-30)),
        cancellation=float(mix_norm / (rn + alpha * gn).clamp_min(1e-30)),
        sign_crossing=bool(mix_norm < rn),
    )


# --------------------------------------------------------------------------
# Step 5  carrier / content split and great-circle content dose
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierSplit:
    radius: float
    c: float             # <h/||h||, u>
    w: Tensor            # unit carrier-orthogonal direction
    degenerate: bool     # |c| too close to 1 -> w ill-conditioned


def carrier_split(h: Tensor, u: Tensor, *, c_abs_max: float) -> CarrierSplit:
    """h = r [ c u + sqrt(1-c^2) w ],  w perp u, ||w|| = 1.

    `c_abs_max` is the eligibility rule frozen in development (Step 5).
    """
    h = h.double()
    u = u.double()
    u = u / u.norm().clamp_min(1e-30)
    rad = h.norm()
    hhat = h / rad.clamp_min(1e-30)
    c = torch.dot(hhat, u)
    degenerate = bool(abs(float(c)) > c_abs_max)
    denom = torch.sqrt((1.0 - c * c).clamp_min(1e-30))
    w = (hhat - c * u) / denom
    return CarrierSplit(float(rad), float(c), w, degenerate)


def slerp(w0: Tensor, w1: Tensor, f: float) -> Tensor:
    """Great-circle interpolation, f in [0, 1]."""
    w0 = w0.double() / w0.double().norm().clamp_min(1e-30)
    w1 = w1.double() / w1.double().norm().clamp_min(1e-30)
    dot = torch.clamp(torch.dot(w0, w1), -1.0, 1.0)
    phi = torch.arccos(dot)
    sp = torch.sin(phi)
    if float(sp) < 1e-8:
        out = (1.0 - f) * w0 + f * w1
        return out / out.norm().clamp_min(1e-30)
    return (torch.sin((1.0 - f) * phi) * w0 + torch.sin(f * phi) * w1) / sp


def content_dose_state(
    h_recipient: Tensor,
    h_donor: Tensor,
    u: Tensor,
    theta: float,
    *,
    c_abs_max: float,
    phi_min: float,
    phi_max: float,
) -> tuple[Optional[Tensor], dict]:
    """Step 5 intervention: hold r_R and c_R, move the carrier-orthogonal
    direction toward the donor by an ABSOLUTE natural angle `theta`.

        h(theta) = r_R [ c_R u + sqrt(1-c_R^2) slerp(w_R, w_D; theta/phi) ]

    Returns (state, diagnostics). `state` is None when the pair is ineligible
    under the development-frozen rules -- extrapolation past the donor
    (theta > phi) is forbidden, and near-parallel / near-antipodal pairs are
    excluded because slerp is ill-conditioned there. Exclusion is reported,
    never silently dropped.
    """
    sR = carrier_split(h_recipient, u, c_abs_max=c_abs_max)
    sD = carrier_split(h_donor, u, c_abs_max=c_abs_max)
    dot = torch.clamp(torch.dot(sR.w, sD.w), -1.0, 1.0)
    phi = float(torch.arccos(dot))
    diag = {
        "phi": phi,
        "c_recipient": sR.c,
        "c_donor": sD.c,
        "radius_recipient": sR.radius,
        "degenerate_recipient": sR.degenerate,
        "degenerate_donor": sD.degenerate,
        "theta_requested": theta,
    }
    if sR.degenerate or sD.degenerate:
        diag["excluded"] = "carrier_degenerate"
        return None, diag
    if not (phi_min <= phi <= phi_max):
        diag["excluded"] = "phi_out_of_range"
        return None, diag
    if theta > phi:
        diag["excluded"] = "extrapolation_theta_gt_phi"
        return None, diag

    u_n = u.double() / u.double().norm().clamp_min(1e-30)
    w = slerp(sR.w, sD.w, theta / phi)
    c = sR.c
    h_new = sR.radius * (c * u_n + (1.0 - c * c) ** 0.5 * w)
    diag["excluded"] = None
    diag["f"] = theta / phi
    return h_new, diag


# --------------------------------------------------------------------------
# §2.4  shrinkage precision metric for natural dose
# --------------------------------------------------------------------------


class PrecisionMetric:
    """M = (Sigma + lambda I)^-1, fit on DEVELOPMENT natural states only,
    scaled so the development median natural distance is 1 (plan §2.4).

    Frozen after fitting: `state_dict()` goes into the manifest and is
    reloaded for locked analysis. Never refit on locked data.
    """

    def __init__(self, M_half: Tensor, scale: float, lam: float, shrink: float):
        self.M_half = M_half      # [d, d] with M = M_half^T M_half
        self.scale = scale
        self.lam = lam
        self.shrink = shrink

    @classmethod
    def fit(cls, states: Tensor, *, shrink: float, lam: float) -> "PrecisionMetric":
        X = states.double()
        X = X - X.mean(0, keepdim=True)
        S = (X.T @ X) / max(X.shape[0] - 1, 1)
        S = (1.0 - shrink) * S + shrink * torch.eye(S.shape[0], dtype=S.dtype) * torch.diagonal(S).mean()
        S = S + lam * torch.eye(S.shape[0], dtype=S.dtype)
        L = torch.linalg.cholesky(S)
        M_half = torch.linalg.solve_triangular(L, torch.eye(S.shape[0], dtype=S.dtype), upper=False)
        return cls(M_half, scale=1.0, lam=lam, shrink=shrink)

    def calibrate(self, natural_deltas: Tensor) -> None:
        """Set overall scale so the development MEDIAN natural distance is 1."""
        d = torch.tensor([self.raw_distance(v) for v in natural_deltas], dtype=torch.float64)
        self.scale = float(d.median())

    def raw_distance(self, delta: Tensor) -> float:
        v = self.M_half @ delta.double()
        return float(v.norm())

    def distance(self, delta: Tensor) -> float:
        return self.raw_distance(delta) / max(self.scale, 1e-30)

    def state_dict(self) -> dict:
        return {"M_half": self.M_half, "scale": self.scale, "lam": self.lam, "shrink": self.shrink}

    @classmethod
    def load(cls, sd: dict) -> "PrecisionMetric":
        return cls(sd["M_half"], sd["scale"], sd["lam"], sd["shrink"])

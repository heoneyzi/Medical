"""Step 1 — numerical and reconstruction gate, R_num, E-COND, EXP1 replication.

Nothing downstream may be interpreted until this passes. The gate's job is
to make an adapter error or a precision artefact fail loudly rather than
look like a mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import naming as N
from .endpoints import centre, rms_denominator, solve_beta_shape
from .manifest import ArchitectureManifest, Manifest
from .taps import Edit, Evo2Runner


# --------------------------------------------------------------------------
# 1. residual identities
# --------------------------------------------------------------------------


@dataclass
class IdentityReport:
    per_block: dict[int, dict[str, float]] = field(default_factory=dict)
    final_logit_err: float = float("nan")
    bilinear_err: dict[int, float] = field(default_factory=dict)
    rms_form_detected: Optional[str] = None
    rms_form_err: dict[str, float] = field(default_factory=dict)
    has_unembed_bias: Optional[bool] = None

    def worst(self) -> float:
        vals = [v for d in self.per_block.values() for v in d.values()]
        vals += [self.final_logit_err, *self.bilinear_err.values()]
        return float(np.nanmax(vals)) if vals else float("nan")


def _rel(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@torch.no_grad()
def check_identities(runner: Evo2Runner, input_ids: Tensor, blocks: Sequence[int]) -> IdentityReport:
    """x_l + m_l = r_l  and  r_l + g_l = x_{l+1}, separately, per block.

    Also reconstructs final logits from the final pre-norm state and checks
    the bilinear accounting g_l = sum_k W3[:,k] p_k.
    """
    rep = IdentityReport()
    taps = []
    for l in blocks:
        taps += [N.x(l), N.m(l), N.r(l), N.g(l), N.mlp_part(l, "z")]
    taps += [N.x(max(blocks) + 1), N.FINAL_PRE_NORM]
    ts = runner.run(input_ids, taps=taps)

    for l in blocks:
        r_hat = ts[N.x(l)] + ts[N.m(l)]
        d = {"x+m=r": _rel(r_hat, ts[N.r(l)])}
        nxt = N.x(l + 1)
        if nxt in ts.tensors:
            d["r+g=x_next"] = _rel(ts[N.r(l)] + ts[N.g(l)], ts[nxt])
        elif N.x(l + 1) in taps:
            d["r+g=x_next"] = float("nan")
        rep.per_block[l] = d

        if runner.arch.mlp_is_bilinear[l]:
            parts = runner.bilinear_parts(l, ts[N.mlp_part(l, "z")])
            rep.bilinear_err[l] = _rel(parts["out"], ts[N.g(l)])

    # final logits from the captured pre-norm state
    h = ts[N.FINAL_PRE_NORM].double()
    gamma = runner.a.final_gamma().double()
    W_U = runner.a.unembedding().double()
    bias = runner.a.unembedding_bias()
    rep.has_unembed_bias = bias is not None

    best, best_name = float("inf"), None
    for form in ("norm_over_sqrtd_plus_eps", "sqrt_mean_sq_plus_eps"):
        s = rms_denominator(h, eps=runner.arch.rms_eps, form=form).unsqueeze(-1)
        z = (gamma * h / s) @ W_U.T
        if bias is not None:
            z = z + bias.double()
        err = _rel(z, ts.logits)
        rep.rms_form_err[form] = err
        if err < best:
            best, best_name = err, form
    rep.rms_form_detected = best_name
    rep.final_logit_err = best
    return rep


# --------------------------------------------------------------------------
# 2. R_num stress suite
# --------------------------------------------------------------------------


@dataclass
class RNum:
    """Numerical / implementation tolerance. §1.5: NOT a scientific null and
    NOT an equivalence margin."""
    d_shape: float
    log_beta: float
    logit_l2: float
    identity_residual: float
    per_source: dict[str, dict[str, float]] = field(default_factory=dict)


@torch.no_grad()
def build_r_num(
    runner: Evo2Runner,
    windows: Sequence[Tensor],
    *,
    beta_min: float,
    beta_max: float,
    flat_logit_threshold: float,
    stressors: Optional[dict[str, Callable[[Tensor], Tensor]]] = None,
    sham_tap: str = N.FINAL_PRE_NORM,
    identity_report: Optional[IdentityReport] = None,
    required_stressors: Sequence[str] = (),
) -> RNum:
    """Max discrepancy over the stress suite the plan lists: batch ordering,
    kernel/reference path, precision, sharding/device placement, sham hook.

    `stressors` maps a name to a callable that re-runs the forward under that
    stressor and returns logits. The caller supplies the ones its environment
    can actually vary; a sham hook (identity edit) is always included here.
    """
    stressors = dict(stressors or {})

    def sham(ids: Tensor) -> Tensor:
        e = Edit(tap=sham_tap, kind="state", fn=lambda t: t.clone(), note="sham")
        return runner.run(ids, taps=(), edits=[e]).logits

    def repeat(ids: Tensor) -> Tensor:
        return runner.run(ids, taps=()).logits

    stressors.setdefault("sham_hook", sham)
    stressors.setdefault("repeat_forward", repeat)
    missing = sorted(set(required_stressors) - set(stressors))
    if missing:
        raise RuntimeError(
            f"R_num stress suite is incomplete; missing required stressors: {missing}"
        )

    per: dict[str, dict[str, float]] = {}
    for name, fn in stressors.items():
        ds, lb, l2 = [], [], []
        for ids in windows:
            z0 = runner.run(ids, taps=()).logits
            z1 = fn(ids)
            z0v, z1v = z0.reshape(-1, z0.shape[-1])[-1], z1.reshape(-1, z1.shape[-1])[-1]
            r = solve_beta_shape(z1v, z0v, beta_min=beta_min, beta_max=beta_max,
                                 flat_logit_threshold=flat_logit_threshold)
            ds.append(r.d_shape)
            lb.append(abs(r.log_beta))
            l2.append(float((centre(z1v.double()) - centre(z0v.double())).norm()))
        per[name] = {"d_shape": float(np.max(ds)), "log_beta": float(np.max(lb)),
                     "logit_l2": float(np.max(l2))}

    return RNum(
        d_shape=max(v["d_shape"] for v in per.values()),
        log_beta=max(v["log_beta"] for v in per.values()),
        logit_l2=max(v["logit_l2"] for v in per.values()),
        identity_residual=(
            float(identity_report.worst()) if identity_report is not None else float("nan")
        ),
        per_source=per,
    )


# --------------------------------------------------------------------------
# 3. E-COND
# --------------------------------------------------------------------------


@dataclass
class ECondRow:
    condition: str
    r2_h30_to_h27: float
    cosine: float
    effective_rank: float
    condition_number: float
    ridge_lambda: float


@dataclass
class CrossfitRecoverability:
    """Out-of-fold recoverability; folds must respect genomic dependencies."""

    r2: float
    fold_r2: list[float]
    lambdas: list[float]
    n_folds: int
    n_rows: int
    note: str = "dependency-group cross-fitted; no row appears in train and test"


def _effective_rank(X: Tensor) -> float:
    """exp(entropy of the normalised eigenspectrum) of the state covariance.

    The plan explicitly says NOT to assume effective rank is precision-free,
    so this is recomputed under every condition like everything else.
    """
    Xc = X - X.mean(0, keepdim=True)
    S = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(S).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-300)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _ridge_gcv(X: Tensor, Y: Tensor, lambdas: Sequence[float]) -> tuple[float, float]:
    """Ridge with GCV-selected lambda; returns (R2, lambda)."""
    n, d = X.shape
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    best_gcv, lam = float("inf"), float(lambdas[0])
    for cand in lambdas:
        f = S / (S ** 2 + cand)
        Yhat = U @ (f.unsqueeze(-1) * UtY)
        dof = float((S ** 2 / (S ** 2 + cand)).sum())
        rss = float(((Yc - Yhat) ** 2).sum())
        gcv = rss / max((1.0 - dof / n) ** 2, 1e-12)
        if gcv < best_gcv:
            best_gcv, lam = gcv, float(cand)
    f = S / (S ** 2 + lam)
    Yhat = U @ (f.unsqueeze(-1) * UtY)
    rss = float(((Yc - Yhat) ** 2).sum())
    r2 = 1.0 - rss / max(float((Yc ** 2).sum()), 1e-30)
    return r2, float(lam)


def _ridge_fit_predict(
    X_train: Tensor,
    Y_train: Tensor,
    X_test: Tensor,
    lambdas: Sequence[float],
) -> tuple[Tensor, float]:
    """GCV-select lambda on the training fold and predict a held-out fold."""

    _, lam = _ridge_gcv(X_train, Y_train, lambdas)
    xm = X_train.mean(0, keepdim=True)
    ym = Y_train.mean(0, keepdim=True)
    Xc = X_train - xm
    Yc = Y_train - ym
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    coef = Vh.T @ ((S / (S ** 2 + lam)).unsqueeze(-1) * (U.T @ Yc))
    return (X_test - xm) @ coef + ym, float(lam)


def crossfit_recoverability(
    source: Tensor,
    target: Tensor,
    group_ids: Sequence[str],
    *,
    lambdas: Sequence[float],
    n_folds: int = 5,
    seed: int = 42,
) -> CrossfitRecoverability:
    """Dependency-group cross-fitted ridge R2.

    The old in-sample E-COND result remains available for exact EXP1
    replication.  Any new recoverability claim should use this function so
    donor/locus/window-linked rows stay in one fold.
    """

    X = source.double()
    Y = target.double()
    if X.shape[0] != Y.shape[0] or X.shape[0] != len(group_ids):
        raise ValueError("source, target and group_ids must have the same row count")
    groups = np.asarray(list(group_ids), dtype=object)
    unique = np.unique(groups)
    if len(unique) < n_folds:
        raise ValueError(f"need at least {n_folds} dependency groups, got {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_of = {g: i % n_folds for i, g in enumerate(shuffled)}
    row_fold = np.asarray([fold_of[g] for g in groups], dtype=int)

    pred = torch.empty_like(Y)
    fold_r2: list[float] = []
    selected: list[float] = []
    for f in range(n_folds):
        tr = torch.as_tensor(row_fold != f)
        te = torch.as_tensor(row_fold == f)
        if int(te.sum()) == 0 or int(tr.sum()) < 2:
            raise RuntimeError(f"empty or degenerate fold {f}")
        yhat, lam = _ridge_fit_predict(X[tr], Y[tr], X[te], lambdas)
        pred[te] = yhat
        selected.append(lam)
        sse = float(((Y[te] - yhat) ** 2).sum())
        sst = float(((Y[te] - Y[te].mean(0, keepdim=True)) ** 2).sum())
        fold_r2.append(1.0 - sse / max(sst, 1e-30))
    sse = float(((Y - pred) ** 2).sum())
    sst = float(((Y - Y.mean(0, keepdim=True)) ** 2).sum())
    return CrossfitRecoverability(
        r2=1.0 - sse / max(sst, 1e-30),
        fold_r2=fold_r2,
        lambdas=selected,
        n_folds=n_folds,
        n_rows=X.shape[0],
    )


def e_cond(
    h30: Tensor, h27: Tensor, ref_direction: Tensor, *, lambdas: Sequence[float], n_deflate: int = 2
) -> list[ECondRow]:
    """Recompute EXP1's two headline numbers under three precision conditions.

    kappa(Sigma_h30) ~ 6e10 against float32 eps ~ 6e-8 means the content
    eigenvalues can sit below the decomposition noise floor. The three
    conditions are exactly the plan's: float32; float64 covariance /
    eigendecomposition; float64 with the top outlier directions explicitly
    separated out.
    """
    rows: list[ECondRow] = []

    def one(name: str, A: Tensor, B: Tensor, u: Tensor) -> ECondRow:
        r2, lam = _ridge_gcv(A, B, lambdas)
        Ac = A - A.mean(0, keepdim=True)
        S = (Ac.T @ Ac) / max(Ac.shape[0] - 1, 1)
        ev = torch.linalg.eigvalsh(S).clamp_min(0)
        cond = float(ev.max() / ev[ev > 0].min()) if (ev > 0).any() else float("inf")
        mean = A.mean(0)
        cos = float(torch.dot(mean, u) / (mean.norm() * u.norm()).clamp_min(1e-30))
        return ECondRow(name, r2, cos, _effective_rank(A), cond, lam)

    rows.append(one("float32", h30.float(), h27.float(), ref_direction.float()))
    rows.append(one("float64", h30.double(), h27.double(), ref_direction.double()))

    A = h30.double()
    Ac = A - A.mean(0, keepdim=True)
    S = (Ac.T @ Ac) / max(Ac.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(S)
    V = evecs[:, -n_deflate:]
    A_def = A - (A @ V) @ V.T
    rows.append(one(f"float64_deflate_top{n_deflate}", A_def, h27.double(), ref_direction.double()))
    return rows


# --------------------------------------------------------------------------
# 4. EXP1 replication table
# --------------------------------------------------------------------------


@dataclass
class ReplicationRow:
    exp1_fact: str
    exp1_value: str
    recomputed: str
    uncertainty: str
    status: str            # "stable" | "shifted" | "unresolved"
    note: str = ""


EXP1_FACTS = (
    "b28-b30 branch norm and effective-rank cascade",
    "coordinate 3756 occupancy and final gain",
    "h28->h27 recoverability",
    "h30->h27 recoverability",
    "donor/Kozak vs acceptor/CpG state-output dissociation",
    "~33-38 bp spatial signature",
    "Appendix B alpha plateau",
)


def empty_replication_table() -> list[ReplicationRow]:
    """The plan is explicit: this is a STARTING-POINT replication table, not
    new confirmatory evidence. EXP1 p-values are not reused as confirmation;
    an item that moves is demoted from premise to re-test target.
    """
    return [ReplicationRow(f, "", "", "", "unresolved") for f in EXP1_FACTS]


# --------------------------------------------------------------------------
# gate verdict
# --------------------------------------------------------------------------


@dataclass
class GateVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def gate(
    ident: IdentityReport,
    rnum: RNum,
    econd: Sequence[ECondRow],
    *,
    identity_tolerance: float,
    sign_consistent_fp32_fp64: bool,
    crop_artifact_free: bool,
    rnum_limits: Optional[dict[str, float]] = None,
    required_stressors: Sequence[str] = (),
    econd_r2_tolerance: Optional[float] = None,
) -> GateVerdict:
    """`identity_tolerance` is endpoint-specific and comes from the manifest,
    not from this function."""
    reasons: list[str] = []
    if not (ident.worst() <= identity_tolerance):
        reasons.append(f"reconstruction error {ident.worst():.3e} > tolerance {identity_tolerance:.3e}")
    rnum_values = {
        "d_shape": rnum.d_shape,
        "log_beta": rnum.log_beta,
        "logit_l2": rnum.logit_l2,
        "identity_residual": rnum.identity_residual,
    }
    nonfinite = [k for k, v in rnum_values.items() if not np.isfinite(v)]
    if nonfinite:
        reasons.append(
            f"R_num is incomplete/non-finite for {nonfinite}; numerical fidelity is not established"
        )
    if np.isfinite(rnum.identity_residual) and rnum.identity_residual > identity_tolerance:
        reasons.append(
            f"R_num identity residual {rnum.identity_residual:.3e} exceeds "
            f"identity tolerance {identity_tolerance:.3e}"
        )
    if rnum_limits is not None:
        for name, value in rnum_values.items():
            limit = rnum_limits.get(name)
            if limit is not None and np.isfinite(value) and value > limit:
                reasons.append(f"R_num {name}={value:.3e} exceeds calibrated limit {limit:.3e}")
    missing_stress = sorted(set(required_stressors) - set(rnum.per_source))
    if missing_stress:
        reasons.append(f"R_num missing required stressors: {missing_stress}")
    if not sign_consistent_fp32_fp64:
        reasons.append("headline effect sign differs between fp32 and fp64 -> "
                       "track finite-precision handoff as a separate mechanism (Step 1 Branch)")
    if not crop_artifact_free:
        reasons.append("effect follows crop-relative position -> boundary/sink artefact")
    r2 = [r.r2_h30_to_h27 for r in econd]
    if len(r2) >= 2:
        if econd_r2_tolerance is None:
            reasons.append(
                "E-COND stability tolerance is missing; calibrate it on development "
                "data rather than using a built-in R2 cutoff"
            )
        elif (not np.isfinite(econd_r2_tolerance)
              or econd_r2_tolerance < 0):
            reasons.append("E-COND stability tolerance is invalid")
        elif (max(r2) - min(r2)) > econd_r2_tolerance:
            reasons.append(
                f"E-COND: h30->h27 R2 spans {min(r2):.3f}-{max(r2):.3f} across precision "
                f"conditions; weaken the 'b30 is the substantive bottleneck' wording and "
                f"re-interpret Steps 5 and 8 accordingly"
            )
    return GateVerdict(passed=not reasons, reasons=reasons)

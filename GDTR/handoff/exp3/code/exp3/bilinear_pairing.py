"""EXP3 test of whether learned W1--W2 channel pairing causes the writer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .exp2_api import (
    N, Edit, Evo2Runner, Manifest, Row, paired_contrast, solve_beta_shape,
)


@dataclass(frozen=True)
class PairingControl:
    native: Tensor
    w2_only_permuted: Tensor
    jointly_relabelled: Tensor
    permutation: Tensor
    joint_relative_error: float
    w1_singular_value_error: float
    w2_singular_value_error: float
    w3_singular_value_error: float


@torch.no_grad()
def bilinear_pairing_control(z: Tensor, W1: Tensor, W2: Tensor, W3: Tensor,
                             permutation: Tensor) -> PairingControl:
    """Break only factor pairing while preserving all marginal spectra.

    Jointly permuting W1/W2 rows and the corresponding W3 columns is an exact
    function-preserving relabelling and therefore the implementation control.
    """
    z, W1, W2, W3 = (x.detach().double() for x in (z, W1, W2, W3))
    if z.shape[-1] != W1.shape[1] or W1.shape != W2.shape \
            or W3.shape[1] != W1.shape[0]:
        raise ValueError("bilinear pairing tensors have incompatible shapes")
    p = permutation.detach().to(device=W1.device, dtype=torch.long).reshape(-1)
    width = W1.shape[0]
    if len(p) != width or set(p.cpu().tolist()) != set(range(width)):
        raise ValueError("permutation must contain every hidden channel exactly once")
    if torch.equal(p.cpu(), torch.arange(width)):
        raise ValueError("identity permutation is not a pairing control")

    def forward(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
        return ((z @ A.T) * (z @ B.T)) @ C.T

    native = forward(W1, W2, W3)
    broken = forward(W1, W2[p], W3)
    relabelled = forward(W1[p], W2[p], W3[:, p])

    def spectrum_error(A: Tensor, B: Tensor) -> float:
        sa, sb = torch.linalg.svdvals(A), torch.linalg.svdvals(B)
        return float((sa - sb).norm() / sa.norm().clamp_min(1e-30))

    return PairingControl(
        native=native, w2_only_permuted=broken,
        jointly_relabelled=relabelled, permutation=p,
        joint_relative_error=float(
            (relabelled - native).norm() / native.norm().clamp_min(1e-30)),
        w1_singular_value_error=spectrum_error(W1, W1[p]),
        w2_singular_value_error=spectrum_error(W2, W2[p]),
        w3_singular_value_error=spectrum_error(W3, W3[:, p]),
    )


@dataclass(frozen=True)
class PairingControlRow:
    locus: str
    condition: str
    d_shape: float
    log_beta: float
    branch_relative_change: float
    joint_relative_error: float
    spectrum_error_max: float
    permutation_sha256: str
    dependency_keys: tuple[str, ...]


@torch.no_grad()
def bilinear_pairing_panel(runner: Evo2Runner, loci: Sequence[tuple], man: Manifest,
                           *, block: int, permutation: Tensor,
                           dependency_keys: Optional[
                               Callable[[str], Sequence[str]]] = None,
                           identity_tolerance: float = 1e-8
                           ) -> list[PairingControlRow]:
    man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    p = permutation.detach().to(torch.long).cpu()
    digest = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    rows: list[PairingControlRow] = []
    for item in loci:
        if len(item) < 2:
            raise ValueError("locus entry must begin with (locus, input_ids)")
        locus, ids = str(item[0]), item[1]
        metadata = item[-1] if len(item) >= 3 and isinstance(item[-1], Mapping) else {}
        keys = tuple(dependency_keys(locus) if dependency_keys else
                     metadata.get("keys", (f"locus:{locus}",)))
        taps = {N.mlp_part(block, "z"), N.g(block)}
        native = runner.run(ids, taps=taps)
        W1, W2, W3 = runner.a.mlp_weights(block)
        control = bilinear_pairing_control(
            native[N.mlp_part(block, "z")], W1, W2, W3, p)
        if control.joint_relative_error > identity_tolerance:
            raise RuntimeError(
                f"joint relabelling failed at block {block}: "
                f"{control.joint_relative_error:.3e}")
        z0 = native.logits.reshape(-1, native.logits.shape[-1])[-1]
        variants = {
            "native": native[N.g(block)],
            "w2_only_permuted": control.w2_only_permuted.to(native[N.g(block)]),
            "jointly_relabelled": control.jointly_relabelled.to(native[N.g(block)]),
        }
        spectrum_max = max(control.w1_singular_value_error,
                           control.w2_singular_value_error,
                           control.w3_singular_value_error)
        for condition, replacement in variants.items():
            trace = native if condition == "native" else runner.run(
                ids, taps=taps, edits=[Edit(
                    N.g(block), "update",
                    lambda t, value=replacement: value.to(t.dtype).to(t.device),
                    f"EXP3 pairing:{condition}")])
            z = trace.logits.reshape(-1, trace.logits.shape[-1])[-1]
            shape = solve_beta_shape(
                z, z0, beta_min=man.margins.beta_min,
                beta_max=man.margins.beta_max,
                flat_logit_threshold=man.margins.flat_logit_threshold)
            rows.append(PairingControlRow(
                locus=locus, condition=condition, d_shape=shape.d_shape,
                log_beta=shape.log_beta,
                branch_relative_change=float(
                    (replacement.double() - native[N.g(block)].double()).norm()
                    / native[N.g(block)].double().norm().clamp_min(1e-30)),
                joint_relative_error=control.joint_relative_error,
                spectrum_error_max=spectrum_max, permutation_sha256=digest,
                dependency_keys=keys))
    return rows


def bilinear_pairing_verdict(rows: Sequence[PairingControlRow], *,
                             break_effect_margin: float,
                             identity_equivalence_margin: float,
                             min_clusters: int, n_boot: int = 10_000,
                             alpha: float = 0.05) -> dict:
    hashes = {r.permutation_sha256 for r in rows}
    if len(hashes) != 1:
        raise ValueError("pairing panel needs exactly one frozen permutation")
    by_condition: dict[str, dict[str, Row]] = {}
    for row in rows:
        cells = by_condition.setdefault(row.condition, {})
        if row.locus in cells:
            raise ValueError(f"duplicate pairing cell {row.locus}:{row.condition}")
        cells[row.locus] = Row(row.d_shape, keys=row.dependency_keys,
                               unit_id=row.locus)
    required = {"native", "w2_only_permuted", "jointly_relabelled"}
    if set(by_condition) != required:
        raise ValueError(f"pairing conditions must be {sorted(required)}")
    loci = sorted(set.intersection(*(set(v) for v in by_condition.values())))
    broken = paired_contrast(
        [by_condition["w2_only_permuted"][k] for k in loci],
        [by_condition["native"][k] for k in loci], min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    relabelled = paired_contrast(
        [by_condition["jointly_relabelled"][k] for k in loci],
        [by_condition["native"][k] for k in loci], min_clusters=min_clusters,
        n_boot=n_boot, alpha=alpha)
    checks = {
        "w2_only_breaks_writer_output": broken.lo > break_effect_margin,
        "joint_relabelling_output_equivalent": (
            relabelled.lo > -identity_equivalence_margin
            and relabelled.hi < identity_equivalence_margin),
        "joint_relabelling_algebra_exact": (
            max(r.joint_relative_error for r in rows) <= identity_equivalence_margin
            and max(r.spectrum_error_max for r in rows)
            <= identity_equivalence_margin),
    }
    return {
        "status": "supported" if all(checks.values()) else "unresolved",
        "claim": ("learned W1-W2 channel pairing is output-causal"
                  if all(checks.values()) else "pairing claim not established"),
        "checks": checks, "w2_only_effect": broken.as_row(),
        "joint_relabelling_effect": relabelled.as_row(),
        "permutation_sha256": next(iter(hashes)),
    }


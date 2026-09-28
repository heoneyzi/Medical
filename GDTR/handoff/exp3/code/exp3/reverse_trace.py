"""EXP3 Phase 3 -- reverse tracing from b30 content back to sequence.

This module deliberately starts from the sealed, accepted EXP2 stage result;
it does not re-run or re-adjudicate EXP2 Steps 1--10.
It does not discover a direction from task labels and it does not treat a
reconstruction as causal evidence.  The intended order is:

1. remove the frozen carrier axis from intervention-induced b30 responses;
2. build a label-free, discovery-only direction bank;
3. verify a selected direction with a held-out output intervention;
4. propagate that fixed direction back through b29 and the exact b28
   bilinear expression; and
5. attribute the resulting fixed path score to sequence edits.

All helpers are model-free.  A runtime callback can therefore evaluate the
actual Evo 2 path while this file owns the bookkeeping, algebra, and
fail-loud validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import prod
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor


def _finite_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _unit_axis(axis: Tensor, *, width: int, name: str) -> Tensor:
    u = _finite_tensor(name, axis).detach().double().reshape(-1)
    if u.numel() != width:
        raise ValueError(f"{name} has width {u.numel()}, expected {width}")
    norm = u.norm()
    if float(norm) <= 0:
        raise ValueError(f"{name} has zero norm")
    return u / norm


def carrier_removed(values: Tensor, carrier_axis: Tensor) -> Tensor:
    """Remove only the component parallel to a pre-specified carrier axis."""

    x = _finite_tensor("values", values)
    if x.ndim < 1:
        raise ValueError("values must have a feature dimension")
    u = _unit_axis(carrier_axis, width=x.shape[-1], name="carrier_axis")
    u = u.to(device=x.device, dtype=x.dtype)
    return x - (x @ u).unsqueeze(-1) * u


def _canonicalize_basis_rows(basis: Tensor) -> Tensor:
    """Give each unoriented SVD axis a deterministic sign."""

    out = basis.clone()
    for j in range(out.shape[0]):
        pivot = int(out[j].abs().argmax())
        if float(out[j, pivot]) < 0:
            out[j] = -out[j]
    return out


@dataclass(frozen=True)
class DirectionBank:
    """A label-free bank of carrier-orthogonal b30 response directions.

    ``basis`` contains orthonormal row vectors with shape ``[rank, d]``.
    ``sample_indices`` records which rows survived the pre-specified norm
    gate; silently replacing excluded rows by zeros would bias the SVD.
    """

    basis: Tensor
    carrier_axis: Tensor
    singular_values: Tensor
    explained_energy_fraction: float
    sample_indices: Tensor
    scores: Tensor
    assignments: Tensor
    assignment_signs: Tensor
    discovery_mean: Tensor
    min_content_norm: float
    centered: bool
    source_split: str
    source_role: str
    source_artifact_sha256: str
    source_response_sha256: str
    source_unit_ids: tuple[str, ...]
    bank_sha256: str

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    @property
    def width(self) -> int:
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class DirectionProjection:
    content: Tensor
    unit_content: Tensor
    scores: Tensor
    reconstruction: Tensor
    residual_norm_fraction: Tensor
    eligible: Tensor


def carrier_excluded_direction_bank(
    responses: Tensor,
    carrier_axis: Tensor,
    *,
    rank: int,
    min_content_norm: float,
    center: bool = False,
    source_split: str,
    source_artifact_sha256: str,
    source_unit_ids: Sequence[str],
    source_role: str = "discovery",
) -> DirectionBank:
    """Fit a discovery-only SVD bank to carrier-removed b30 *deltas*.

    Rows should be intervention-induced responses such as ``delta m30`` or a
    pre-registered internal HCL component, not absolute residual states.  A
    task label is intentionally absent from the API.  Rows are normalized
    before SVD so the bank captures direction rather than rediscovering the
    branch-norm explosion.
    """

    x = _finite_tensor("responses", responses).detach().double()
    if not source_split.strip() or source_role != "discovery":
        raise RuntimeError("direction banks must be fit on a named discovery split")
    if (len(source_artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_artifact_sha256)):
        raise ValueError("source_artifact_sha256 must be a lowercase full SHA-256")
    unit_ids = tuple(str(value) for value in source_unit_ids)
    if len(unit_ids) != len(x) or len(set(unit_ids)) != len(unit_ids) \
            or any(not value for value in unit_ids):
        raise ValueError("source_unit_ids must uniquely identify every response row")
    source_value = x.detach().cpu().contiguous()
    source_digest = hashlib.sha256(source_value.numpy().tobytes()).hexdigest()
    if x.ndim != 2:
        raise ValueError(f"responses must be [n,d], got {tuple(x.shape)}")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer frozen in discovery")
    if not torch.isfinite(torch.tensor(float(min_content_norm))) or min_content_norm <= 0:
        raise ValueError("min_content_norm must be finite and positive")

    u = _unit_axis(carrier_axis, width=x.shape[1], name="carrier_axis").to(x.device)
    content = x - (x @ u).unsqueeze(-1) * u
    norms = content.norm(dim=-1)
    eligible = norms >= float(min_content_norm)
    idx = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if idx.numel() < rank:
        raise RuntimeError(
            f"only {idx.numel()} carrier-excluded responses pass the norm gate; "
            f"cannot fit frozen rank={rank}"
        )
    directions = content[idx] / norms[idx, None]
    discovery_mean = torch.zeros(
        directions.shape[1], dtype=directions.dtype, device=directions.device
    )
    if center:
        discovery_mean = directions.mean(0)
        directions = directions - discovery_mean.unsqueeze(0)
        renorm = directions.norm(dim=-1)
        if bool((renorm < 1e-12).any()):
            raise RuntimeError("centering made at least one discovery direction degenerate")
        directions = directions / renorm[:, None]

    _, singular, vh = torch.linalg.svd(directions, full_matrices=False)
    tol = max(directions.shape) * torch.finfo(directions.dtype).eps * singular[0]
    numerical_rank = int((singular > tol).sum())
    if numerical_rank < rank:
        raise RuntimeError(
            f"requested direction rank={rank}, but discovery responses have "
            f"numerical rank={numerical_rank}"
        )
    basis = vh[:rank]
    # SVD should already make these carrier-orthogonal.  Projecting once more
    # and QR-normalising prevents tiny carrier leakage from later accumulating.
    basis = basis - (basis @ u).unsqueeze(-1) * u
    basis = torch.linalg.qr(basis.T, mode="reduced").Q.T
    basis = _canonicalize_basis_rows(basis)

    scores = directions @ basis.T
    assignments = scores.abs().argmax(dim=-1)
    signs = torch.sign(scores.gather(1, assignments[:, None]).squeeze(1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    explained = float((singular[:rank] ** 2).sum() / (singular ** 2).sum().clamp_min(1e-30))
    payload = {
        "source_split": source_split,
        "source_role": source_role,
        "min_content_norm": float(min_content_norm),
        "centered": bool(center),
        "rank": int(rank),
        "explained_energy_fraction": explained,
        "source_artifact_sha256": source_artifact_sha256,
        "source_response_sha256": source_digest,
        "source_unit_ids": unit_ids,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    for tensor in (
        basis, u, singular, idx, discovery_mean, scores, assignments, signs,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return DirectionBank(
        basis=basis,
        carrier_axis=u,
        singular_values=singular,
        explained_energy_fraction=explained,
        sample_indices=idx,
        scores=scores,
        assignments=assignments,
        assignment_signs=signs,
        discovery_mean=discovery_mean,
        min_content_norm=float(min_content_norm),
        centered=bool(center),
        source_split=source_split,
        source_role=source_role,
        source_artifact_sha256=source_artifact_sha256,
        source_response_sha256=source_digest,
        source_unit_ids=unit_ids,
        bank_sha256=digest.hexdigest(),
    )


def validate_direction_bank(bank: DirectionBank) -> None:
    """Verify discovery provenance, geometry, and byte-level integrity."""
    if bank.source_role != "discovery" or not bank.source_split.strip():
        raise RuntimeError("direction bank provenance is not discovery-only")
    if (bank.basis.ndim != 2 or bank.carrier_axis.numel() != bank.width
            or not torch.isfinite(bank.basis).all()
            or not torch.isfinite(bank.carrier_axis).all()):
        raise RuntimeError("direction bank tensors are malformed")
    gram = bank.basis.double() @ bank.basis.double().T
    eye = torch.eye(bank.rank, dtype=gram.dtype, device=gram.device)
    if not torch.allclose(gram, eye, rtol=1e-8, atol=1e-10):
        raise RuntimeError("direction bank basis is no longer orthonormal")
    if float((bank.basis.double() @ bank.carrier_axis.double()).norm()) > 1e-8:
        raise RuntimeError("direction bank leaked into the carrier axis")
    payload = {
        "source_split": bank.source_split,
        "source_role": bank.source_role,
        "min_content_norm": float(bank.min_content_norm),
        "centered": bool(bank.centered),
        "rank": bank.rank,
        "explained_energy_fraction": float(bank.explained_energy_fraction),
        "source_artifact_sha256": bank.source_artifact_sha256,
        "source_response_sha256": bank.source_response_sha256,
        "source_unit_ids": bank.source_unit_ids,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    for tensor in (
        bank.basis, bank.carrier_axis, bank.singular_values,
        bank.sample_indices, bank.discovery_mean, bank.scores,
        bank.assignments, bank.assignment_signs,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if digest.hexdigest() != bank.bank_sha256:
        raise RuntimeError("direction bank changed after discovery sealing")


def project_direction_bank(responses: Tensor, bank: DirectionBank) -> DirectionProjection:
    """Project held-out responses without refitting or changing the bank."""

    validate_direction_bank(bank)

    x = _finite_tensor("responses", responses).detach().double()
    if x.ndim != 2 or x.shape[1] != bank.width:
        raise ValueError(
            f"responses must be [n,{bank.width}], got {tuple(x.shape)}"
        )
    u = bank.carrier_axis.to(x.device)
    basis = bank.basis.to(x.device)
    content = x - (x @ u).unsqueeze(-1) * u
    norms = content.norm(dim=-1)
    eligible = norms >= bank.min_content_norm
    unit = torch.zeros_like(content)
    unit[eligible] = content[eligible] / norms[eligible, None]
    if bank.centered:
        unit[eligible] = unit[eligible] - bank.discovery_mean.to(x.device)
        centered_norm = unit.norm(dim=-1)
        eligible = eligible & (centered_norm >= 1e-12)
        unit[eligible] = unit[eligible] / centered_norm[eligible, None]
        unit[~eligible] = 0
    scores = unit @ basis.T
    reconstruction = scores @ basis
    residual = (unit - reconstruction).norm(dim=-1)
    residual = torch.where(eligible, residual, torch.full_like(residual, float("nan")))
    return DirectionProjection(content, unit, scores, reconstruction, residual, eligible)


@dataclass(frozen=True)
class BilinearContributionScore:
    """Exact projection accounting for ``g28 = W3[(W1 z)*(W2 z)]``."""

    factor_a: Tensor
    factor_b: Tensor
    product: Tensor
    channel_contributions: Tensor
    projected_output: Tensor
    reconstructed_projection: Tensor
    relative_residual: float
    channel_order: tuple[int, ...]
    target_direction: Tensor

    def topk(self, k: int) -> tuple[int, ...]:
        if not 0 < k <= len(self.channel_order):
            raise ValueError(f"k must be in [1,{len(self.channel_order)}]")
        return self.channel_order[:k]


def exact_b28_contribution_scores(
    z: Tensor,
    W1: Tensor,
    W2: Tensor,
    W3: Tensor,
    target_direction: Tensor,
    *,
    carrier_axis: Optional[Tensor] = None,
    normalize_direction: bool = True,
) -> BilinearContributionScore:
    """Compute the signed, exact per-channel contribution to a fixed axis.

    For channel ``j`` the score is exactly

    ``(q.T @ W3[:,j]) * (W1 z)[j] * (W2 z)[j]``.

    Ranking by this value is different from ranking by ``abs(p_j)`` or branch
    norm: it includes the projection direction the reverse trace is asking
    about.  The returned residual must be numerically zero before using the
    ranking for an intervention.
    """

    z0 = _finite_tensor("z", z).double()
    A = _finite_tensor("W1", W1).double()
    B = _finite_tensor("W2", W2).double()
    C = _finite_tensor("W3", W3).double()
    if z0.ndim < 1 or A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("z must end in d_in and W1/W2/W3 must be matrices")
    if A.shape != B.shape:
        raise ValueError(f"W1 and W2 shapes differ: {tuple(A.shape)} vs {tuple(B.shape)}")
    hidden, d_in = A.shape
    if z0.shape[-1] != d_in or C.shape[1] != hidden:
        raise ValueError(
            f"incompatible bilinear shapes: z={tuple(z0.shape)}, W1={tuple(A.shape)}, "
            f"W2={tuple(B.shape)}, W3={tuple(C.shape)}"
        )
    q = _finite_tensor("target_direction", target_direction).double().reshape(-1)
    if q.numel() != C.shape[0]:
        raise ValueError(f"target direction has width {q.numel()}, expected {C.shape[0]}")
    if carrier_axis is not None:
        u = _unit_axis(carrier_axis, width=q.numel(), name="carrier_axis").to(q.device)
        q = q - torch.dot(q, u) * u
    if normalize_direction:
        qn = q.norm()
        if float(qn) <= 0:
            raise ValueError("target direction is zero after carrier removal")
        q = q / qn

    A = A.to(z0.device)
    B = B.to(z0.device)
    C = C.to(z0.device)
    q = q.to(z0.device)
    a = z0 @ A.T
    b = z0 @ B.T
    p = a * b
    readout = C.T @ q
    channel = p * readout
    reconstructed = channel.sum(dim=-1)
    output = p @ C.T
    projected = output @ q
    residual = (reconstructed - projected).norm()
    denominator = projected.norm().clamp_min(1e-30)
    mean_abs = channel.reshape(-1, hidden).abs().mean(0)
    order = tuple(sorted(range(hidden), key=lambda j: (-float(mean_abs[j]), j)))
    return BilinearContributionScore(
        factor_a=a,
        factor_b=b,
        product=p,
        channel_contributions=channel,
        projected_output=projected,
        reconstructed_projection=reconstructed,
        relative_residual=float(residual / denominator),
        channel_order=order,
        target_direction=q,
    )


@dataclass(frozen=True)
class CandidateHit:
    sequence_id: str
    start: int
    end: int
    method: str
    score: float
    strand: str = "."
    sequence: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sequence_id:
            raise ValueError("candidate sequence_id is empty")
        if not (0 <= self.start < self.end):
            raise ValueError(
                f"invalid half-open candidate interval {self.sequence_id}:{self.start}-{self.end}"
            )
        if not self.method:
            raise ValueError("candidate method is empty")
        if not torch.isfinite(torch.tensor(float(self.score))):
            raise ValueError("candidate score must be finite")


@dataclass(frozen=True)
class ConsensusCandidate:
    sequence_id: str
    start: int
    end: int
    strand: str
    methods: tuple[str, ...]
    method_scores: Mapping[str, float]
    combined_score: float
    members: tuple[CandidateHit, ...]


def candidate_method_consensus(
    candidates: Mapping[str, Sequence[CandidateHit]],
    *,
    min_methods: int = 2,
    max_gap: int = 0,
    require_same_strand: bool = False,
) -> list[ConsensusCandidate]:
    """Require independent candidate generators to agree on a locus.

    Overlapping (or at most ``max_gap`` apart) half-open intervals are joined.
    Several hits from one method still count as one vote, preventing a dense
    ISM scan from outvoting a PWM or natural-library method by row count.
    """

    if min_methods < 2:
        raise ValueError("motif consensus requires at least two independent methods")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    hits: list[CandidateHit] = []
    for method, rows in candidates.items():
        if not method:
            raise ValueError("candidate method name is empty")
        for hit in rows:
            hit.validate()
            if hit.method != method:
                raise ValueError(
                    f"candidate declares method={hit.method!r} inside mapping key {method!r}"
                )
            hits.append(hit)
    if len(candidates) < min_methods:
        raise ValueError(
            f"only {len(candidates)} candidate methods supplied, fewer than min_methods={min_methods}"
        )

    n = len(hits)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    order = sorted(range(n), key=lambda i: (hits[i].sequence_id, hits[i].start, hits[i].end))
    for oi, i in enumerate(order):
        hi = hits[i]
        for j in order[oi + 1:]:
            hj = hits[j]
            if hj.sequence_id != hi.sequence_id:
                if hj.sequence_id > hi.sequence_id:
                    break
                continue
            if hj.start > hi.end + max_gap:
                break
            if require_same_strand and hi.strand != hj.strand:
                continue
            # Strict overlap when max_gap=0; with a positive gap allowance,
            # nearby intervals may also vote for the same candidate region.
            overlap = min(hi.end, hj.end) - max(hi.start, hj.start)
            gap = max(0, max(hi.start, hj.start) - min(hi.end, hj.end))
            if overlap > 0 or (max_gap > 0 and gap <= max_gap):
                union(i, j)

    groups: dict[int, list[CandidateHit]] = {}
    for i, hit in enumerate(hits):
        groups.setdefault(find(i), []).append(hit)

    result: list[ConsensusCandidate] = []
    for members in groups.values():
        methods = sorted({h.method for h in members})
        if len(methods) < min_methods:
            continue
        per_method = {
            method: max(float(h.score) for h in members if h.method == method)
            for method in methods
        }
        strand_set = {h.strand for h in members}
        strand = next(iter(strand_set)) if len(strand_set) == 1 else "."
        result.append(ConsensusCandidate(
            sequence_id=members[0].sequence_id,
            start=min(h.start for h in members),
            end=max(h.end for h in members),
            strand=strand,
            methods=tuple(methods),
            method_scores=per_method,
            combined_score=sum(per_method.values()) / len(per_method),
            members=tuple(sorted(members, key=lambda h: (h.start, h.end, h.method))),
        ))
    return sorted(result, key=lambda h: (-len(h.methods), -h.combined_score,
                                         h.sequence_id, h.start, h.end))


@dataclass(frozen=True)
class KmerEnrichment:
    kmer: str
    positive_count: int
    matched_count: int
    log_odds_ratio: float


def matched_kmer_enrichment(positive_sequences: Sequence[str],
                            matched_sequences: Sequence[str], *, k: int,
                            pseudocount: float = 0.5,
                            min_total_count: int = 3) -> list[KmerEnrichment]:
    """Label-free natural-library generator for direction-bank clusters.

    ``positive_sequences`` are selected only by a frozen internal direction;
    matched controls must already satisfy the sequence confound contract in
    EXP3 Phase 4.  The result is a candidate generator, never causal evidence.
    """
    if k <= 0 or pseudocount <= 0 or min_total_count < 1:
        raise ValueError("invalid k-mer enrichment settings")
    alphabet = set("ACGT")

    def counts(seqs):
        out = {}
        total = 0
        for seq in seqs:
            seq = str(seq).upper()
            for i in range(max(0, len(seq) - k + 1)):
                word = seq[i:i + k]
                if set(word) <= alphabet:
                    out[word] = out.get(word, 0) + 1
                    total += 1
        return out, total

    pos, npos = counts(positive_sequences)
    neg, nneg = counts(matched_sequences)
    if npos == 0 or nneg == 0:
        raise ValueError("both sequence libraries need at least one valid k-mer")
    vocab = sorted(set(pos) | set(neg))
    size = 4 ** k
    rows = []
    for word in vocab:
        a, b = pos.get(word, 0), neg.get(word, 0)
        if a + b < min_total_count:
            continue
        pa = (a + pseudocount) / (npos + pseudocount * size)
        pb = (b + pseudocount) / (nneg + pseudocount * size)
        rows.append(KmerEnrichment(word, a, b, float(torch.log(torch.tensor(pa / pb)))))
    return sorted(rows, key=lambda r: (-r.log_odds_ratio, -r.positive_count, r.kmer))


def seqlet_candidates(sequence_id: str, sequence: str, base_scores: Sequence[float], *,
                      lengths: Sequence[int], top_k: int,
                      method: str = "path_seqlet") -> list[CandidateHit]:
    """Convert fixed path attribution scores into multi-width candidate seqlets."""
    scores = torch.as_tensor(base_scores, dtype=torch.float64)
    if scores.ndim != 1 or len(scores) != len(sequence) or not torch.isfinite(scores).all():
        raise ValueError("base_scores must be one finite value per base")
    widths = sorted({int(x) for x in lengths})
    if not widths or widths[0] <= 0 or widths[-1] > len(sequence) or top_k <= 0:
        raise ValueError("invalid seqlet widths/top_k")
    hits = []
    for width in widths:
        kernel = torch.ones(1, 1, width, dtype=scores.dtype)
        window = torch.nn.functional.conv1d(scores.reshape(1, 1, -1), kernel).reshape(-1)
        order = sorted(range(len(window)), key=lambda i: (-float(window[i]), i))[:top_k]
        hits.extend(CandidateHit(sequence_id, i, i + width, method,
                                 float(window[i]), sequence=sequence[i:i + width],
                                 metadata={"width": width}) for i in order)
    return sorted(hits, key=lambda h: (-h.score, h.start, h.end))


def _scalar_path_score(value: float | Tensor, *, label: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must return a scalar, got shape {tuple(value.shape)}")
        out = float(value.detach().double().cpu())
    else:
        out = float(value)
    if not torch.isfinite(torch.tensor(out)):
        raise ValueError(f"{label} returned a non-finite score")
    return out


@dataclass(frozen=True)
class PathISMResult:
    sequence: str
    alphabet: tuple[str, ...]
    wildtype_score: float
    alternative_scores: Tensor
    effects: Tensor
    evaluated: Tensor

    def top_edits(self, k: int, *, absolute: bool = True) -> list[tuple[int, str, float]]:
        entries: list[tuple[int, str, float]] = []
        for i in range(len(self.sequence)):
            for j, base in enumerate(self.alphabet):
                if bool(self.evaluated[i, j]):
                    entries.append((i, base, float(self.effects[i, j])))
        if not 0 < k <= len(entries):
            raise ValueError(f"k must be in [1,{len(entries)}]")
        key = (lambda x: (-abs(x[2]), x[0], x[1])) if absolute else (
            lambda x: (-x[2], x[0], x[1]))
        return sorted(entries, key=key)[:k]


def path_restricted_ism(
    sequence: str,
    path_score: Callable[[str], float | Tensor],
    *,
    alphabet: Sequence[str] = ("A", "C", "G", "T"),
    positions: Optional[Sequence[int]] = None,
) -> PathISMResult:
    """Exact single-base mutagenesis of a *fixed internal path score*.

    ``path_score`` may measure a frozen g28 direction, a b30 bank direction,
    or a path-specific output contrast.  It should not refit the direction for
    each mutation.  The wild-type alternative is marked evaluated with zero
    effect without wasting another model forward.
    """

    if not sequence:
        raise ValueError("sequence is empty")
    alpha = tuple(str(a).upper() for a in alphabet)
    if len(alpha) < 2 or len(set(alpha)) != len(alpha) or any(len(a) != 1 for a in alpha):
        raise ValueError("alphabet must contain unique single-character symbols")
    seq = sequence.upper()
    bad = sorted(set(seq) - set(alpha))
    if bad:
        raise ValueError(f"sequence contains symbols outside the alphabet: {bad}")
    pos = list(range(len(seq))) if positions is None else [int(i) for i in positions]
    if len(set(pos)) != len(pos) or any(i < 0 or i >= len(seq) for i in pos):
        raise ValueError("positions must be unique valid sequence indices")

    wild = _scalar_path_score(path_score(seq), label="path_score(wildtype)")
    scores = torch.full((len(seq), len(alpha)), float("nan"), dtype=torch.float64)
    effects = torch.full_like(scores, float("nan"))
    evaluated = torch.zeros_like(scores, dtype=torch.bool)
    for i in pos:
        for j, base in enumerate(alpha):
            if base == seq[i]:
                score = wild
            else:
                mutant = seq[:i] + base + seq[i + 1:]
                score = _scalar_path_score(path_score(mutant), label=f"path_score({i}:{base})")
            scores[i, j] = score
            effects[i, j] = score - wild
            evaluated[i, j] = True
    return PathISMResult(seq, alpha, wild, scores, effects, evaluated)


@dataclass(frozen=True)
class WindowAttribution:
    start: int
    end: int
    score: float
    effect: float
    depth: int


def hierarchical_path_scan(
    sequence: str,
    path_score: Callable[[str], float | Tensor],
    perturb_window: Callable[[str, int, int], str],
    *,
    min_window: int = 1,
    prune_below: Optional[float] = None,
) -> list[WindowAttribution]:
    """Gradient-free coarse-to-fine attribution of a fixed path score.

    With ``prune_below=None`` every node of the binary interval tree is
    evaluated.  A pruning threshold is allowed only when fixed before locked
    analysis; otherwise a weak parent could hide antagonistic child effects.
    """

    if not sequence:
        raise ValueError("sequence is empty")
    if min_window < 1:
        raise ValueError("min_window must be positive")
    if prune_below is not None and prune_below < 0:
        raise ValueError("prune_below must be non-negative")
    baseline = _scalar_path_score(path_score(sequence), label="path_score(wildtype)")
    queue: list[tuple[int, int, int]] = [(0, len(sequence), 0)]
    out: list[WindowAttribution] = []
    while queue:
        lo, hi, depth = queue.pop(0)
        changed = perturb_window(sequence, lo, hi)
        if not isinstance(changed, str) or len(changed) != len(sequence):
            raise ValueError("perturb_window must return a same-length string")
        score = _scalar_path_score(path_score(changed), label=f"path_score({lo}:{hi})")
        effect = score - baseline
        out.append(WindowAttribution(lo, hi, score, effect, depth))
        width = hi - lo
        should_split = width > min_window
        if prune_below is not None:
            should_split = should_split and abs(effect) >= prune_below
        if should_split:
            mid = lo + width // 2
            if mid > lo:
                queue.append((lo, mid, depth + 1))
            if hi > mid:
                queue.append((mid, hi, depth + 1))
    return sorted(out, key=lambda r: (r.depth, r.start, r.end))


REVERSE_CHAIN_LINKS = (
    "output_to_b30",
    "b30_to_b29",
    "b29_to_b28",
    "b28_to_sequence",
)


@dataclass(frozen=True)
class ReverseChainScore:
    score: float
    bottleneck: float
    geometric_mean: float
    off_path_leakage: float
    limiting_link: str
    links: Mapping[str, float]
    natural_support: bool
    note: str = (
        "descriptive conjunction of pre-normalised held-out link strengths; "
        "not a p-value and not a substitute for block/rescue evidence"
    )


def reverse_chain_score(
    links: Mapping[str, float],
    *,
    off_path_leakage: float,
    natural_support: bool,
) -> ReverseChainScore:
    """Summarise a completed reverse trace without letting strong links hide one weak link.

    Every link strength and leakage must be normalised to ``[0,1]`` by a
    rule frozen before confirmatory evaluation.  The primary score is the
    weakest link, penalised by measured off-path leakage.  A geometric mean
    is returned only as a secondary smooth summary.
    """

    missing = sorted(set(REVERSE_CHAIN_LINKS) - set(links))
    extra = sorted(set(links) - set(REVERSE_CHAIN_LINKS))
    if missing or extra:
        raise ValueError(f"reverse-chain links incomplete; missing={missing}, extra={extra}")
    values = {name: float(links[name]) for name in REVERSE_CHAIN_LINKS}
    bad = {k: v for k, v in values.items() if not (0.0 <= v <= 1.0)}
    if bad or not (0.0 <= float(off_path_leakage) <= 1.0):
        raise ValueError(f"link strengths and leakage must lie in [0,1]; bad={bad}")
    limiting = min(REVERSE_CHAIN_LINKS, key=lambda name: values[name])
    bottleneck = values[limiting]
    geometric = prod(values.values()) ** (1.0 / len(values))
    score = bottleneck * (1.0 - float(off_path_leakage))
    if not natural_support:
        score = 0.0
    return ReverseChainScore(
        score=score,
        bottleneck=bottleneck,
        geometric_mean=geometric,
        off_path_leakage=float(off_path_leakage),
        limiting_link=limiting,
        links=values,
        natural_support=bool(natural_support),
    )


def reverse_linear_adjoint(output_direction: Tensor, forward_operators: Sequence[Tensor]) -> Tensor:
    """Apply exact linear adjoints in reverse order for audited local maps.

    ``forward_operators`` are Jacobian/reduced-map matrices in forward order,
    each shaped ``[d_out, d_in]``.  This helper is appropriate for a frozen
    local linearization; it does not pretend that a nonlinear suffix is
    globally linear.
    """

    q = _finite_tensor("output_direction", output_direction).double().reshape(-1)
    for i, op in reversed(list(enumerate(forward_operators))):
        J = _finite_tensor(f"forward_operators[{i}]", op).double()
        if J.ndim != 2 or J.shape[0] != q.numel():
            raise ValueError(
                f"operator {i} has shape {tuple(J.shape)} but current adjoint width is {q.numel()}"
            )
        q = J.T @ q.to(J.device)
    return q

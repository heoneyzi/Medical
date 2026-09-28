"""EXP3 Phase 7 — chromosome, task, checkpoint and training-time generalization.

This module deliberately evaluates *frozen* features.  It contains no layer or
direction search.  chr22 may be used for discovery/development; chr17 (or a
new chromosome if chr17 has already been inspected) is supplied as the locked
test mask and is never used for scaling, rank selection or readout tuning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .artifacts import ArtifactRef, ArtifactStore
from .exp2_api import Row, cluster_bootstrap


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    # Pairwise definition handles ties exactly and avoids a scipy dependency.
    d = score[pos, None] - score[None, neg]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def _regression_metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    sse = float(np.square(y - pred).sum())
    sst = float(np.square(y - y.mean()).sum())
    return {"r2": float(1 - sse / sst) if sst > 0 else float("nan"),
            "mae": float(np.abs(y - pred).mean())}


def _fit_ridge(X: np.ndarray, y: np.ndarray, T: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs, Ts = (X - mu) / sd, (T - mu) / sd
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    Ts = np.column_stack([np.ones(len(Ts)), Ts])
    P = np.eye(Xs.shape[1]) * ridge
    P[0, 0] = 0.0
    system, rhs = Xs.T @ Xs + P, Xs.T @ y
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return Ts @ coef


@dataclass(frozen=True)
class FrozenFeatureTable:
    unit_ids: tuple[str, ...]
    chromosomes: tuple[str, ...]
    groups: tuple[str, ...]
    labels: tuple[float, ...]
    matrices: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        n = len(self.unit_ids)
        if not (len(self.chromosomes) == len(self.groups) == len(self.labels) == n):
            raise ValueError("feature-table row metadata lengths differ")
        if len(set(self.unit_ids)) != n:
            raise ValueError("unit_ids are not unique")
        if n == 0 or any(not x for x in self.chromosomes) or any(not x for x in self.groups):
            raise ValueError("feature table is empty or has blank chromosome/group IDs")
        if not np.isfinite(np.asarray(self.labels, dtype=float)).all():
            raise ValueError("feature-table labels are non-finite")
        for method, X in self.matrices.items():
            X = np.asarray(X)
            if X.ndim != 2 or len(X) != n or not np.isfinite(X).all():
                raise ValueError(f"bad feature matrix {method!r}: {X.shape}")
            if method not in self.feature_names or len(self.feature_names[method]) != X.shape[1]:
                raise ValueError(f"feature-name mismatch for {method!r}")


def label_efficiency_benchmark(table: FrozenFeatureTable, *,
                               train_chromosomes: Sequence[str],
                               test_chromosome: str,
                               sizes: Sequence[int | str],
                               task: str, ridge: float,
                               zero_shot_weights: Mapping[str, Sequence[float]] | None = None,
                               repeats: int = 20, seed: int = 42) -> dict:
    """Same group samples and untouched chromosome for every feature baseline."""
    table.validate()
    if task not in {"binary", "regression"}:
        raise ValueError("task must be 'binary' or 'regression'")
    chrom = np.asarray(table.chromosomes)
    train_mask = np.isin(chrom, list(train_chromosomes))
    test_mask = chrom == test_chromosome
    if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
        raise ValueError("train/test chromosome masks are empty or overlap")
    groups = np.asarray(table.groups)
    y = np.asarray(table.labels, dtype=float)
    if task == "binary" and not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("binary labels must be encoded as 0/1")
    train_groups = np.unique(groups[train_mask])
    leaked = set(groups[train_mask]) & set(groups[test_mask])
    if leaked:
        raise RuntimeError(
            "dependency groups cross the chromosome split: "
            f"{sorted(map(str, leaked))[:5]}"
        )
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, list[float]]] = {}

    for requested in sizes:
        key = str(requested)
        output[key] = {m: [] for m in table.matrices}
        n_repeat = 1 if requested in (0, "full") else repeats
        for _ in range(n_repeat):
            if requested == 0:
                selected = np.zeros(len(y), dtype=bool)
            elif requested == "full":
                selected = train_mask.copy()
            else:
                k = int(requested)
                if k <= 0 or k > len(train_groups):
                    raise ValueError(f"invalid label budget {requested}")
                chosen = rng.choice(train_groups, size=k, replace=False)
                selected = train_mask & np.isin(groups, chosen)
            for method, matrix in table.matrices.items():
                X = np.asarray(matrix, dtype=float)
                if requested == 0:
                    if zero_shot_weights is None or method not in zero_shot_weights:
                        continue
                    w = np.asarray(zero_shot_weights[method], dtype=float)
                    if len(w) != X.shape[1]:
                        raise ValueError(f"zero-shot weight width mismatch for {method}")
                    pred = X[test_mask] @ w
                else:
                    if selected.sum() <= X.shape[1]:
                        continue
                    if task == "binary" and len(np.unique(y[selected])) < 2:
                        continue
                    pred = _fit_ridge(X[selected], y[selected], X[test_mask], ridge)
                metric = (_auc(y[test_mask], pred) if task == "binary" else
                          _regression_metric(y[test_mask], pred)["r2"])
                output[key][method].append(float(metric))
    summary = {}
    for size, methods in output.items():
        summary[size] = {}
        for method, values in methods.items():
            a = np.asarray(values, dtype=float)
            summary[size][method] = {
                "mean": float(np.nanmean(a)) if len(a) else None,
                "sd": float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0 if len(a) else None,
                "n_repeats": len(a),
            }
    return {"task": task, "train_chromosomes": list(train_chromosomes),
            "test_chromosome": test_chromosome, "sizes": summary,
            "unit_of_sampling": "dependency_group"}


@dataclass(frozen=True)
class Alignment:
    matrix: Tensor
    source_dim: int
    target_dim: int
    fit_anchor_cosine_descriptive: float
    validation_anchor_cosine: float
    validation_ci_lo: float
    validation_ci_hi: float
    n_validation_clusters: int
    fit_anchor_ids: tuple[str, ...]
    validation_anchor_ids: tuple[str, ...]
    fit_anchor_sha256: str
    validation_anchor_sha256: str
    matrix_sha256: str
    validation_min_clusters: int
    validation_n_boot: int
    validation_alpha: float
    validation_seed: int
    sha256: str


    @property
    def anchor_cosine(self) -> float:
        """Backward-readable name; always the disjoint validation estimate."""
        return self.validation_anchor_cosine

    def verify(self) -> None:
        matrix = self.matrix.detach().double()
        if (matrix.ndim != 2 or matrix.shape !=
                (self.source_dim, self.target_dim) or
                not torch.isfinite(matrix).all()):
            raise ValueError("alignment matrix/dimensions are invalid")
        matrix_sha = _tensor_sha256(matrix)
        if matrix_sha != self.matrix_sha256:
            raise RuntimeError("checkpoint alignment matrix changed after sealing")
        if set(self.fit_anchor_ids) & set(self.validation_anchor_ids):
            raise RuntimeError("fit and validation anchors overlap")
        if (len(set(self.fit_anchor_ids)) != len(self.fit_anchor_ids) or
                len(set(self.validation_anchor_ids)) != len(self.validation_anchor_ids) or
                not self.fit_anchor_ids or not self.validation_anchor_ids or
                not self.validation_ci_lo <= self.validation_anchor_cosine <=
                self.validation_ci_hi or self.n_validation_clusters <
                self.validation_min_clusters or
                not 0 < self.validation_alpha < 1 or
                self.validation_n_boot < 100):
            raise ValueError("alignment validation provenance/interval is invalid")
        payload = _alignment_payload(
            matrix_sha256=self.matrix_sha256,
            source_dim=self.source_dim, target_dim=self.target_dim,
            fit_anchor_sha256=self.fit_anchor_sha256,
            validation_anchor_sha256=self.validation_anchor_sha256,
            fit_anchor_ids=self.fit_anchor_ids,
            validation_anchor_ids=self.validation_anchor_ids,
            fit_anchor_cosine_descriptive=self.fit_anchor_cosine_descriptive,
            validation_anchor_cosine=self.validation_anchor_cosine,
            validation_ci_lo=self.validation_ci_lo,
            validation_ci_hi=self.validation_ci_hi,
            n_validation_clusters=self.n_validation_clusters,
            validation_min_clusters=self.validation_min_clusters,
            validation_n_boot=self.validation_n_boot,
            validation_alpha=self.validation_alpha,
            validation_seed=self.validation_seed,
        )
        if _json_sha256(payload) != self.sha256:
            raise RuntimeError("checkpoint alignment provenance changed after sealing")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + tensor.numpy().tobytes()).hexdigest()


def _json_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: str) -> bool:
    return (len(value) == 64 and
            all(char in "0123456789abcdef" for char in value.lower()))


def seal_alignment_anchor_pair(anchor_ids: Sequence[str], source: Tensor,
                               target: Tensor) -> str:
    """Bind ordered anchor IDs to both checkpoint tensors before fitting/use."""
    ids = tuple(str(value) for value in anchor_ids)
    if not ids or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("anchor IDs must be unique and non-empty")
    A, B = source.detach().double(), target.detach().double()
    if A.ndim != 2 or B.ndim != 2 or len(A) != len(B) or len(A) != len(ids):
        raise ValueError("paired anchor tensors/IDs are not row-aligned")
    if not torch.isfinite(A).all() or not torch.isfinite(B).all():
        raise ValueError("checkpoint anchors are non-finite")
    return _json_sha256({
        "anchor_ids": list(ids),
        "source_sha256": _tensor_sha256(A),
        "target_sha256": _tensor_sha256(B),
    })


def _alignment_payload(**kwargs) -> dict:
    payload = dict(kwargs)
    payload["fit_anchor_ids"] = list(payload["fit_anchor_ids"])
    payload["validation_anchor_ids"] = list(payload["validation_anchor_ids"])
    return payload


def orthogonal_checkpoint_alignment(
    source_anchors: Tensor, target_anchors: Tensor, *,
    fit_anchor_ids: Sequence[str], expected_fit_anchor_sha256: str,
    validation_source_anchors: Tensor, validation_target_anchors: Tensor,
    validation_anchor_ids: Sequence[str],
    validation_dependency_keys: Sequence[tuple[str, ...]],
    expected_validation_anchor_sha256: str,
    min_validation_clusters: int, n_boot: int = 5000,
    alpha: float = 0.05, seed: int = 42,
) -> Alignment:
    """Fit on discovery anchors and score on hash-bound disjoint anchors.

    The fit-set cosine is descriptive only.  Downstream transfer gates use the
    lower dependency-cluster bootstrap bound from the validation anchors.
    """
    A, B = source_anchors.detach().double(), target_anchors.detach().double()
    if A.ndim != 2 or B.ndim != 2 or len(A) != len(B) or len(A) < 2:
        raise ValueError("paired anchor matrices must be [n,d] with the same n")
    if A.device != B.device:
        raise ValueError("source and target anchors are on different devices")
    if not torch.isfinite(A).all() or not torch.isfinite(B).all():
        raise ValueError("checkpoint anchors are non-finite")
    fit_ids = tuple(str(value) for value in fit_anchor_ids)
    validation_ids = tuple(str(value) for value in validation_anchor_ids)
    if len(fit_ids) != len(A):
        raise ValueError("fit anchor IDs are not row-aligned")
    if set(fit_ids) & set(validation_ids):
        raise RuntimeError("alignment fit and validation anchor IDs overlap")
    fit_sha = seal_alignment_anchor_pair(fit_ids, A, B)
    if fit_sha != expected_fit_anchor_sha256:
        raise RuntimeError("fit anchors differ from the sealed input")

    VA = validation_source_anchors.detach().double()
    VB = validation_target_anchors.detach().double()
    if VA.device != VB.device or A.device != VA.device:
        raise ValueError("fit/validation anchors are on different devices")
    if VA.shape[1:] != A.shape[1:] or VB.shape[1:] != B.shape[1:]:
        raise ValueError("validation anchor widths differ from fit widths")
    if len(validation_ids) != len(VA) or len(validation_dependency_keys) != len(VA):
        raise ValueError("validation IDs/dependency keys are not row-aligned")
    validation_sha = seal_alignment_anchor_pair(validation_ids, VA, VB)
    if validation_sha != expected_validation_anchor_sha256:
        raise RuntimeError("validation anchors differ from the sealed input")
    if any(not keys for keys in validation_dependency_keys):
        raise ValueError("validation anchors need dependency keys")
    if min_validation_clusters < 2 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError("invalid alignment validation bootstrap configuration")

    Ac = A - A.mean(0, keepdim=True)
    Bc = B - B.mean(0, keepdim=True)
    if Ac.norm() <= 1e-12 or Bc.norm() <= 1e-12:
        raise ValueError("checkpoint anchors contain no centred variation")
    U, _, Vh = torch.linalg.svd(Ac.T @ Bc, full_matrices=False)
    W = U @ Vh
    fit_cosines = torch.nn.functional.cosine_similarity(Ac @ W, Bc, dim=-1)
    if not torch.isfinite(fit_cosines).all():
        raise ValueError("checkpoint alignment cosine is undefined")

    VAc = VA - VA.mean(0, keepdim=True)
    VBc = VB - VB.mean(0, keepdim=True)
    validation_cosines = torch.nn.functional.cosine_similarity(VAc @ W, VBc, dim=-1)
    if not torch.isfinite(validation_cosines).all():
        raise ValueError("validation alignment cosine is undefined")
    estimate = cluster_bootstrap(
        [Row(float(value), keys=tuple(validation_dependency_keys[index]),
             unit_id=validation_ids[index])
         for index, value in enumerate(validation_cosines)],
        n_boot=n_boot, alpha=alpha, seed=seed,
        min_clusters=min_validation_clusters,
    )
    matrix_sha = _tensor_sha256(W)
    fields = dict(
        matrix_sha256=matrix_sha, source_dim=A.shape[1], target_dim=B.shape[1],
        fit_anchor_sha256=fit_sha, validation_anchor_sha256=validation_sha,
        fit_anchor_ids=fit_ids, validation_anchor_ids=validation_ids,
        fit_anchor_cosine_descriptive=float(fit_cosines.mean()),
        validation_anchor_cosine=estimate.point,
        validation_ci_lo=estimate.lo, validation_ci_hi=estimate.hi,
        n_validation_clusters=estimate.n_clusters,
        validation_min_clusters=min_validation_clusters,
        validation_n_boot=n_boot, validation_alpha=alpha,
        validation_seed=seed,
    )
    return Alignment(matrix=W, sha256=_json_sha256(_alignment_payload(**fields)),
                     **fields)


@dataclass(frozen=True)
class DirectionTransfer:
    direction: Tensor
    retained_norm_fraction: float
    retained_energy_fraction: float
    projection_loss_fraction: float
    minimum_retained_norm_fraction: float
    alignment_sha256: str


def transfer_direction(direction: Tensor, alignment: Alignment, *,
                       minimum_retained_norm_fraction: float) -> DirectionTransfer:
    alignment.verify()
    d = direction.detach().double().reshape(-1)
    if len(d) != alignment.source_dim:
        raise ValueError("direction width differs from alignment source")
    if not torch.isfinite(d).all() or d.norm() <= 1e-12:
        raise ValueError("direction must be finite and non-zero")
    if not 0 < minimum_retained_norm_fraction <= 1:
        raise ValueError("minimum_retained_norm_fraction must lie in (0,1]")
    out = d @ alignment.matrix.to(d.device)
    retained = float(out.norm() / d.norm())
    if not np.isfinite(retained) or retained < minimum_retained_norm_fraction:
        raise RuntimeError(
            "aligned direction retains too little norm; rectangular projection "
            f"retained={retained:.6g}, required={minimum_retained_norm_fraction:.6g}")
    return DirectionTransfer(
        direction=out / out.norm(), retained_norm_fraction=retained,
        retained_energy_fraction=float(min(1.0, retained * retained)),
        projection_loss_fraction=float(
            np.sqrt(max(0.0, 1.0 - retained * retained))),
        minimum_retained_norm_fraction=float(minimum_retained_norm_fraction),
        alignment_sha256=alignment.sha256,
    )


@dataclass(frozen=True)
class CrossCheckpointRow:
    pair_id: str
    transferred_effect: float
    wrong_layer_effect: float
    index_matched_random_effect: float
    b30_block_remaining: float
    predicted_delta_rescue: float
    prediction_sha256: str
    predicted_delta_sha256: str
    regenerated_delta_sha256: str
    dependency_keys: tuple[str, ...]


@dataclass(frozen=True)
class CrossCheckpointLinearModel:
    """Auditable train-only ridge model used for locked checkpoint replay.

    The coefficient tensor is a hash-addressed artifact.  Discovery and
    development tensors are committed separately in this record; neither can
    overlap the units supplied later to :class:`CrossCheckpointPrediction`.
    The development panel is evaluated, but is never refit or inspected while
    predicting the locked panel.
    """

    artifact: ArtifactRef
    method: str
    ridge: float
    input_dim: int
    output_dim: int
    train_split: str
    development_split: str
    train_unit_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    train_inputs_sha256: str
    train_targets_sha256: str
    development_inputs_sha256: str
    development_targets_sha256: str
    development_mse: float
    sha256: str

    def _payload(self) -> dict:
        return {
            "artifact": self.artifact.as_dict(),
            "method": self.method,
            "ridge": self.ridge,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "train_split": self.train_split,
            "development_split": self.development_split,
            "train_unit_ids": list(self.train_unit_ids),
            "development_unit_ids": list(self.development_unit_ids),
            "train_inputs_sha256": self.train_inputs_sha256,
            "train_targets_sha256": self.train_targets_sha256,
            "development_inputs_sha256": self.development_inputs_sha256,
            "development_targets_sha256": self.development_targets_sha256,
            "development_mse": self.development_mse,
        }

    def verify(self) -> None:
        ArtifactStore.verify(self.artifact)
        groups = (self.train_unit_ids, self.development_unit_ids)
        if (self.method != "ridge-linear-v1" or not np.isfinite(self.ridge) or
                self.ridge < 0 or self.input_dim < 1 or self.output_dim < 1 or
                not self.train_split or not self.development_split or
                self.train_split == self.development_split or
                any(not ids or len(set(ids)) != len(ids) or any(not value for value in ids)
                    for ids in groups) or
                set(self.train_unit_ids) & set(self.development_unit_ids) or
                any(not _is_sha256(value) for value in (
                    self.train_inputs_sha256, self.train_targets_sha256,
                    self.development_inputs_sha256,
                    self.development_targets_sha256)) or
                not np.isfinite(self.development_mse) or self.development_mse < 0):
            raise ValueError("invalid cross-checkpoint fitted-model provenance")
        expected_metadata = {
            "semantic_role": "cross_checkpoint_delta_model",
            "method": self.method,
            "ridge": self.ridge,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "train_split": self.train_split,
            "development_split": self.development_split,
            "train_inputs_sha256": self.train_inputs_sha256,
            "train_targets_sha256": self.train_targets_sha256,
            "development_inputs_sha256": self.development_inputs_sha256,
            "development_targets_sha256": self.development_targets_sha256,
        }
        if (self.artifact.kind != "tensor" or
                self.artifact.source_split != self.development_split or
                self.artifact.source_role != "development" or
                self.artifact.created_by != "fit_cross_checkpoint_linear_model" or
                any(self.artifact.metadata.get(key) != value
                    for key, value in expected_metadata.items())):
            raise RuntimeError("fitted-model artifact metadata is inconsistent")
        coefficients = ArtifactStore.load_tensor(self.artifact).detach().double()
        if (coefficients.shape != (self.input_dim + 1, self.output_dim) or
                not torch.isfinite(coefficients).all()):
            raise RuntimeError("fitted-model coefficient artifact is invalid")
        if _json_sha256(self._payload()) != self.sha256:
            raise RuntimeError("cross-checkpoint fitted model changed after sealing")


def _as_finite_matrix(value: Tensor, *, name: str) -> Tensor:
    matrix = value.detach().cpu().double()
    if matrix.ndim != 2 or len(matrix) == 0 or not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return matrix.contiguous()


def fit_cross_checkpoint_linear_model(
    train_inputs: Tensor, train_targets: Tensor,
    development_inputs: Tensor, development_targets: Tensor, *,
    train_unit_ids: Sequence[str], development_unit_ids: Sequence[str],
    train_split: str, development_split: str,
    artifact_store: ArtifactStore, artifact_name: str,
    ridge: float,
) -> CrossCheckpointLinearModel:
    """Fit and seal a fixed-ridge linear delta predictor on discovery only."""
    X = _as_finite_matrix(train_inputs, name="train_inputs")
    Y = _as_finite_matrix(train_targets, name="train_targets")
    D = _as_finite_matrix(development_inputs, name="development_inputs")
    V = _as_finite_matrix(development_targets, name="development_targets")
    train_ids = tuple(str(value) for value in train_unit_ids)
    development_ids = tuple(str(value) for value in development_unit_ids)
    if (len(X) != len(Y) or len(X) != len(train_ids) or
            len(D) != len(V) or len(D) != len(development_ids) or
            X.shape[1] != D.shape[1] or Y.shape[1] != V.shape[1]):
        raise ValueError("cross-checkpoint fit matrices/IDs are not row-aligned")
    if (not train_split or not development_split or
            train_split == development_split or
            not train_ids or not development_ids or
            len(set(train_ids)) != len(train_ids) or
            len(set(development_ids)) != len(development_ids) or
            set(train_ids) & set(development_ids) or
            any(not value for value in (*train_ids, *development_ids))):
        raise ValueError("cross-checkpoint train/development provenance overlaps")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    augmented = torch.cat((torch.ones((len(X), 1), dtype=X.dtype), X), dim=1)
    penalty = torch.eye(augmented.shape[1], dtype=X.dtype) * float(ridge)
    penalty[0, 0] = 0.0
    system = augmented.T @ augmented + penalty
    rhs = augmented.T @ Y
    try:
        coefficients = torch.linalg.solve(system, rhs)
    except RuntimeError:
        coefficients = torch.linalg.lstsq(system, rhs).solution
    development_prediction = torch.cat(
        (torch.ones((len(D), 1), dtype=D.dtype), D), dim=1) @ coefficients
    development_mse = float(torch.square(development_prediction - V).mean())
    hashes = {
        "train_inputs_sha256": _tensor_sha256(X),
        "train_targets_sha256": _tensor_sha256(Y),
        "development_inputs_sha256": _tensor_sha256(D),
        "development_targets_sha256": _tensor_sha256(V),
    }
    metadata = {
        "semantic_role": "cross_checkpoint_delta_model",
        "method": "ridge-linear-v1",
        "ridge": float(ridge),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "train_split": str(train_split),
        "development_split": str(development_split),
        **hashes,
    }
    artifact = artifact_store.put_tensor(
        artifact_name, coefficients, source_split=str(development_split),
        source_role="development",
        created_by="fit_cross_checkpoint_linear_model", metadata=metadata)
    values = dict(
        artifact=artifact, method="ridge-linear-v1", ridge=float(ridge),
        input_dim=int(X.shape[1]), output_dim=int(Y.shape[1]),
        train_split=str(train_split), development_split=str(development_split),
        train_unit_ids=train_ids, development_unit_ids=development_ids,
        development_mse=development_mse, **hashes,
    )
    provisional = CrossCheckpointLinearModel(sha256="0" * 64, **values)
    result = CrossCheckpointLinearModel(
        sha256=_json_sha256(provisional._payload()), **values)
    result.verify()
    return result


def replay_cross_checkpoint_model(model: CrossCheckpointLinearModel,
                                  locked_inputs: Tensor) -> Tensor:
    """Regenerate deltas from the sealed artifact; no supplied outputs enter."""
    model.verify()
    X = _as_finite_matrix(locked_inputs, name="locked_inputs")
    if X.shape[1] != model.input_dim:
        raise ValueError("locked input width differs from fitted model")
    coefficients = ArtifactStore.load_tensor(model.artifact).detach().double()
    augmented = torch.cat((torch.ones((len(X), 1), dtype=X.dtype), X), dim=1)
    return (augmented @ coefficients).contiguous()


@dataclass(frozen=True)
class CrossCheckpointPrediction:
    model: CrossCheckpointLinearModel
    method: str
    model_artifact_sha256: str
    train_split: str
    development_split: str
    locked_split: str
    train_unit_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    locked_unit_ids: tuple[str, ...]
    locked_inputs_sha256: str
    locked_prediction_hashes: tuple[tuple[str, str], ...]
    prediction_table_sha256: str
    held_out: bool
    target_derived: bool
    sha256: str

    def _payload(self) -> dict:
        return {
            "model_sha256": self.model.sha256,
            "method": self.method,
            "model_artifact_sha256": self.model_artifact_sha256,
            "train_split": self.train_split,
            "development_split": self.development_split,
            "locked_split": self.locked_split,
            "train_unit_ids": list(self.train_unit_ids),
            "development_unit_ids": list(self.development_unit_ids),
            "locked_unit_ids": list(self.locked_unit_ids),
            "locked_inputs_sha256": self.locked_inputs_sha256,
            "locked_prediction_hashes": [list(value)
                                           for value in self.locked_prediction_hashes],
            "prediction_table_sha256": self.prediction_table_sha256,
            "held_out": self.held_out,
            "target_derived": self.target_derived,
        }

    def verify(self) -> None:
        self.model.verify()
        groups = (self.train_unit_ids, self.development_unit_ids,
                  self.locked_unit_ids)
        if (not self.method or not _is_sha256(self.model_artifact_sha256) or
                not _is_sha256(self.locked_inputs_sha256) or
                not _is_sha256(self.prediction_table_sha256) or
                any(not split for split in
                    (self.train_split, self.development_split, self.locked_split)) or
                any(not ids or len(set(ids)) != len(ids) or any(not x for x in ids)
                    for ids in groups)):
            raise ValueError("invalid cross-checkpoint prediction provenance")
        if (self.method != self.model.method or
                self.model_artifact_sha256 != self.model.artifact.sha256 or
                self.train_split != self.model.train_split or
                self.development_split != self.model.development_split or
                self.train_unit_ids != self.model.train_unit_ids or
                self.development_unit_ids != self.model.development_unit_ids):
            raise RuntimeError("prediction provenance differs from its fitted model")
        if (set(self.train_unit_ids) & set(self.development_unit_ids) or
                set(self.train_unit_ids) & set(self.locked_unit_ids) or
                set(self.development_unit_ids) & set(self.locked_unit_ids)):
            raise RuntimeError("cross-checkpoint prediction splits overlap")
        if len({self.train_split, self.development_split, self.locked_split}) != 3:
            raise RuntimeError("cross-checkpoint prediction split labels overlap")
        if not self.held_out or self.target_derived:
            raise RuntimeError("cross-checkpoint prediction is not held-out/target-free")
        hash_items = tuple((str(key), str(value))
                           for key, value in self.locked_prediction_hashes)
        if (tuple(key for key, _ in hash_items) != self.locked_unit_ids or
                any(not _is_sha256(value) for _, value in hash_items) or
                _json_sha256([list(value) for value in hash_items]) !=
                self.prediction_table_sha256):
            raise RuntimeError("locked prediction commitment table is invalid")
        if _json_sha256(self._payload()) != self.sha256:
            raise RuntimeError("cross-checkpoint prediction changed after sealing")


def seal_cross_checkpoint_prediction(
    *, model: CrossCheckpointLinearModel, locked_inputs: Tensor,
    locked_unit_ids: Sequence[str], locked_split: str,
    held_out: bool, target_derived: bool,
) -> CrossCheckpointPrediction:
    """Seal target-free predictions generated inside the fitted model replay."""
    model.verify()
    X = _as_finite_matrix(locked_inputs, name="locked_inputs")
    locked_ids = tuple(str(value) for value in locked_unit_ids)
    if (len(X) != len(locked_ids) or not locked_ids or
            len(set(locked_ids)) != len(locked_ids) or
            any(not value for value in locked_ids)):
        raise ValueError("locked inputs and ordered unit IDs are not row-aligned")
    if (not locked_split or locked_split in
            {model.train_split, model.development_split} or
            set(locked_ids) & set(model.train_unit_ids) or
            set(locked_ids) & set(model.development_unit_ids)):
        raise RuntimeError("locked prediction panel overlaps train/development")
    predicted = replay_cross_checkpoint_model(model, X)
    prediction_hashes = tuple(
        (unit_id, _tensor_sha256(predicted[index]))
        for index, unit_id in enumerate(locked_ids))
    table_sha = _json_sha256([list(value) for value in prediction_hashes])
    input_sha = _json_sha256({
        "unit_ids": list(locked_ids),
        "tensor_sha256": _tensor_sha256(X),
    })
    values = dict(
        model=model, method=model.method,
        model_artifact_sha256=model.artifact.sha256,
        train_split=model.train_split, development_split=model.development_split,
        locked_split=str(locked_split),
        train_unit_ids=model.train_unit_ids,
        development_unit_ids=model.development_unit_ids,
        locked_unit_ids=locked_ids, locked_inputs_sha256=input_sha,
        locked_prediction_hashes=prediction_hashes,
        prediction_table_sha256=table_sha,
        held_out=bool(held_out), target_derived=bool(target_derived),
    )
    provisional = CrossCheckpointPrediction(sha256="0" * 64, **values)
    result = CrossCheckpointPrediction(
        sha256=_json_sha256(provisional._payload()), **values)
    result.verify()
    return result


def cross_checkpoint_verdict(rows: Sequence[CrossCheckpointRow], *,
                             transfer_min: float, control_max: float,
                             rescue_min: float, minimum_anchor_cosine: float,
                             alignment: Alignment,
                             prediction: CrossCheckpointPrediction,
                             locked_model_inputs: Tensor,
                             min_clusters: int,
                             n_boot: int = 5000, alpha: float = 0.05,
                             seed: int = 42) -> dict:
    """Test transfer using dependency-cluster CIs and explicit equivalence.

    Positive transfer/rescue use lower confidence bounds. Wrong-layer,
    random-direction and b30-block effects pass only when their entire
    confidence intervals lie inside the registered equivalence region.
    """
    if not rows:
        raise ValueError("cross-checkpoint table is empty")
    prediction.verify()
    if len({r.pair_id for r in rows}) != len(rows):
        raise ValueError("cross-checkpoint pair_id must be unique")
    if min_clusters < 2 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError("invalid cross-checkpoint bootstrap configuration")
    if transfer_min < 0 or rescue_min < 0 or control_max < 0:
        raise ValueError("cross-checkpoint margins must be non-negative")
    numeric = [(r.transferred_effect, r.wrong_layer_effect,
                r.index_matched_random_effect, r.b30_block_remaining,
                r.predicted_delta_rescue) for r in rows]
    if not np.isfinite(numeric).all():
        raise ValueError("cross-checkpoint table contains non-finite values")
    if any(not r.pair_id or not r.dependency_keys for r in rows):
        raise ValueError("every cross-checkpoint row needs an ID and dependency keys")
    if {row.pair_id for row in rows} != set(prediction.locked_unit_ids):
        raise RuntimeError("cross-checkpoint result set differs from locked prediction IDs")
    if tuple(row.pair_id for row in rows) != prediction.locked_unit_ids:
        raise RuntimeError("cross-checkpoint result order differs from locked model inputs")
    locked_inputs = _as_finite_matrix(
        locked_model_inputs, name="locked_model_inputs")
    input_sha = _json_sha256({
        "unit_ids": list(prediction.locked_unit_ids),
        "tensor_sha256": _tensor_sha256(locked_inputs),
    })
    if input_sha != prediction.locked_inputs_sha256:
        raise RuntimeError("locked model inputs changed after prediction sealing")
    internally_regenerated = replay_cross_checkpoint_model(
        prediction.model, locked_inputs)
    internal_hashes = tuple(
        (unit_id, _tensor_sha256(internally_regenerated[index]))
        for index, unit_id in enumerate(prediction.locked_unit_ids))
    if (internal_hashes != prediction.locked_prediction_hashes or
            _json_sha256([list(value) for value in internal_hashes]) !=
            prediction.prediction_table_sha256):
        raise RuntimeError("sealed model replay differs from prediction commitment")
    delta_table = dict(internal_hashes)
    for row in rows:
        if row.prediction_sha256 != prediction.sha256:
            raise RuntimeError("cross-checkpoint row uses a different predictor")
        internal_sha = delta_table[row.pair_id]
        if (row.predicted_delta_sha256 != internal_sha or
                row.regenerated_delta_sha256 != internal_sha):
            raise RuntimeError(
                "caller-supplied hashes do not match internal fitted-model replay")
    if not -1 <= minimum_anchor_cosine <= 1:
        raise ValueError("minimum_anchor_cosine must lie in [-1,1]")
    alignment.verify()
    if prediction.model.output_dim != alignment.target_dim:
        raise RuntimeError("delta predictor output width differs from aligned checkpoint")
    if alignment.validation_ci_lo < minimum_anchor_cosine:
        return {"claim": "unidentified: alignment fidelity failed",
                "validation_anchor_cosine": alignment.validation_anchor_cosine,
                "validation_ci_lo": alignment.validation_ci_lo,
                "validation_ci_hi": alignment.validation_ci_hi,
                "validation_anchor_sha256": alignment.validation_anchor_sha256}

    fields = (
        "transferred_effect", "wrong_layer_effect",
        "index_matched_random_effect", "b30_block_remaining",
        "predicted_delta_rescue",
    )
    estimates = {}
    for offset, field in enumerate(fields):
        estimates[field] = cluster_bootstrap(
            [Row(float(getattr(row, field)), keys=row.dependency_keys,
                 unit_id=row.pair_id) for row in rows],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )
    checks = {
        "transfer": estimates["transferred_effect"].lo >= transfer_min,
        "wrong_layer_equivalent": (
            estimates["wrong_layer_effect"].lo >= -control_max and
            estimates["wrong_layer_effect"].hi <= control_max),
        "random_equivalent": (
            estimates["index_matched_random_effect"].lo >= -control_max and
            estimates["index_matched_random_effect"].hi <= control_max),
        "reencoder_required": (
            estimates["b30_block_remaining"].lo >= -control_max and
            estimates["b30_block_remaining"].hi <= control_max),
        "predicted_delta_rescues": (
            estimates["predicted_delta_rescue"].lo >= rescue_min),
    }
    return {"claim": "cross-checkpoint mechanism transfer" if all(checks.values())
            else "transfer claim not established", "checks": checks,
            "estimates": {name: estimate.as_row()
                          for name, estimate in estimates.items()},
            "alignment_sha256": alignment.sha256,
            "prediction_sha256": prediction.sha256,
            "prediction_table_sha256": prediction.prediction_table_sha256,
            "validation_alignment": {
                "point": alignment.validation_anchor_cosine,
                "ci_lo": alignment.validation_ci_lo,
                "ci_hi": alignment.validation_ci_hi,
                "n_clusters": alignment.n_validation_clusters,
                "anchor_sha256": alignment.validation_anchor_sha256,
            },
            "margins": {"transfer_min": transfer_min,
                        "control_equivalence": control_max,
                        "rescue_min": rescue_min},
            "bootstrap": {"n_boot": n_boot, "alpha": alpha,
                          "min_clusters": min_clusters, "seed": seed}}


@dataclass(frozen=True)
class BlindRolePrediction:
    checkpoint: str
    prediction_sha256: str
    writer_block: int
    reencoder_block: int
    carrier_coordinate: int | None
    expects_direction_plateau: bool
    source_split: str
    source_artifact_sha256: str
    source_unit_ids: tuple[str, ...]
    prediction_config_sha256: str

    @staticmethod
    def create(checkpoint: str, writer_block: int, reencoder_block: int,
               carrier_coordinate: int | None,
               expects_direction_plateau: bool, *, source_split: str,
               source_artifact_sha256: str, source_unit_ids: Sequence[str],
               prediction_config_sha256: str) -> "BlindRolePrediction":
        body = dict(checkpoint=checkpoint, writer_block=writer_block,
                    reencoder_block=reencoder_block,
                    carrier_coordinate=carrier_coordinate,
                    expects_direction_plateau=expects_direction_plateau,
                    source_split=str(source_split),
                    source_artifact_sha256=str(source_artifact_sha256),
                    source_unit_ids=tuple(str(value) for value in source_unit_ids),
                    prediction_config_sha256=str(prediction_config_sha256))
        payload = {**body, "source_unit_ids": list(body["source_unit_ids"])}
        digest = _json_sha256(payload)
        return BlindRolePrediction(prediction_sha256=digest, **body)

    def verify(self) -> None:
        if (not self.checkpoint or not self.source_split or
                not _is_sha256(self.source_artifact_sha256) or
                not _is_sha256(self.prediction_config_sha256) or
                not self.source_unit_ids or
                len(set(self.source_unit_ids)) != len(self.source_unit_ids) or
                any(not value for value in self.source_unit_ids)):
            raise ValueError("invalid blind-role prediction provenance")
        again = BlindRolePrediction.create(
            self.checkpoint, self.writer_block, self.reencoder_block,
            self.carrier_coordinate, self.expects_direction_plateau,
            source_split=self.source_split,
            source_artifact_sha256=self.source_artifact_sha256,
            source_unit_ids=self.source_unit_ids,
            prediction_config_sha256=self.prediction_config_sha256)
        if again.prediction_sha256 != self.prediction_sha256:
            raise RuntimeError("blind checkpoint prediction changed after sealing")


@dataclass(frozen=True)
class BlockRoleObservation:
    unit_id: str
    checkpoint: str
    source_split: str
    source_artifact_sha256: str
    prediction_sha256: str
    held_out: bool
    block: int
    writer_effect: float
    reencoder_effect: float
    observed_carrier_coordinate: int | None
    observed_direction_plateau: bool
    dependency_keys: tuple[str, ...]


def score_blind_role_prediction(prediction: BlindRolePrediction,
                                observations: Sequence[BlockRoleObservation], *,
                                expected_observation_split: str,
                                expected_observation_source_sha256: str,
                                tolerance_blocks: int = 0,
                                minimum_role_gap: float = 0.0,
                                hit_rate_min: float = 0.8,
                                min_clusters: int,
                                n_boot: int = 5000,
                                alpha: float = 0.05,
                                seed: int = 42) -> dict:
    prediction.verify()
    if not observations:
        raise ValueError("no prospective checkpoint observations")
    if (not expected_observation_split or
            not _is_sha256(expected_observation_source_sha256)):
        raise ValueError("invalid blind-role observation source")
    if (expected_observation_split == prediction.source_split or
            expected_observation_source_sha256 == prediction.source_artifact_sha256):
        raise RuntimeError("blind-role observations reuse prediction source data")
    if (tolerance_blocks < 0 or minimum_role_gap < 0 or
            not 0 <= hit_rate_min <= 1 or min_clusters < 2 or
            n_boot < 100 or not 0 < alpha < 1):
        raise ValueError("invalid blind-role thresholds/bootstrap")
    keys = [(row.unit_id, row.block) for row in observations]
    if len(set(keys)) != len(keys):
        raise ValueError("prospective unit/block observations contain duplicates")
    if not np.isfinite([(row.writer_effect, row.reencoder_effect)
                        for row in observations]).all():
        raise ValueError("prospective block observations are non-finite")
    if any(
        row.checkpoint != prediction.checkpoint or
        row.source_split != expected_observation_split or
        row.source_artifact_sha256 != expected_observation_source_sha256 or
        row.prediction_sha256 != prediction.prediction_sha256 or
        not row.held_out or not row.unit_id or not row.dependency_keys
        for row in observations
    ):
        raise RuntimeError("blind-role row is not from the sealed held-out panel")
    units = sorted({row.unit_id for row in observations})
    if set(units) & set(prediction.source_unit_ids):
        raise RuntimeError("blind-role observations reuse prediction unit IDs")
    blocks_by_unit = {unit: {row.block for row in observations
                             if row.unit_id == unit} for unit in units}
    grids = {tuple(sorted(blocks)) for blocks in blocks_by_unit.values()}
    if len(grids) != 1 or len(next(iter(grids))) < 2:
        raise RuntimeError("blind-role panel is not an exact unit x block grid")

    summaries = []
    for unit in units:
        rows = [row for row in observations if row.unit_id == unit]
        carrier_values = {row.observed_carrier_coordinate for row in rows}
        plateau_values = {row.observed_direction_plateau for row in rows}
        dependency_values = {row.dependency_keys for row in rows}
        if (len(carrier_values) != 1 or len(plateau_values) != 1 or
                len(dependency_values) != 1):
            raise RuntimeError(
                f"unit {unit!r} changes carrier/plateau/dependency across blocks")
        actual_w = max(rows, key=lambda row: row.writer_effect).block
        actual_r = max(rows, key=lambda row: row.reencoder_effect).block
        writer_order = sorted((row.writer_effect for row in rows), reverse=True)
        reencoder_order = sorted((row.reencoder_effect for row in rows), reverse=True)
        summaries.append({
            "unit_id": unit, "keys": rows[0].dependency_keys,
            "writer_hit": float(
                abs(prediction.writer_block - actual_w) <= tolerance_blocks),
            "reencoder_hit": float(
                abs(prediction.reencoder_block - actual_r) <= tolerance_blocks),
            "carrier_hit": float(
                prediction.carrier_coordinate == next(iter(carrier_values))),
            "plateau_hit": float(
                prediction.expects_direction_plateau == next(iter(plateau_values))),
            "writer_gap": writer_order[0] - writer_order[1],
            "reencoder_gap": reencoder_order[0] - reencoder_order[1],
        })

    def boot(field: str, offset: int):
        return cluster_bootstrap(
            [Row(float(item[field]), keys=item["keys"], unit_id=item["unit_id"])
             for item in summaries],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )

    estimates = {field: boot(field, offset) for offset, field in enumerate((
        "writer_hit", "reencoder_hit", "carrier_hit", "plateau_hit",
        "writer_gap", "reencoder_gap"))}
    checks = {
        "writer_hit": estimates["writer_hit"].lo >= hit_rate_min,
        "reencoder_hit": estimates["reencoder_hit"].lo >= hit_rate_min,
        "carrier_coordinate_hit": estimates["carrier_hit"].lo >= hit_rate_min,
        "direction_plateau_hit": estimates["plateau_hit"].lo >= hit_rate_min,
        "writer_identified": estimates["writer_gap"].lo > minimum_role_gap,
        "reencoder_identified": estimates["reencoder_gap"].lo > minimum_role_gap,
    }
    return {"checkpoint": prediction.checkpoint,
            "writer_predicted": prediction.writer_block,
            "reencoder_predicted": prediction.reencoder_block,
            "carrier_predicted": prediction.carrier_coordinate,
            "plateau_predicted": prediction.expects_direction_plateau,
            **checks, "all_hit": all(checks.values()),
            "estimates": {name: estimate.as_row()
                          for name, estimate in estimates.items()},
            "prediction_sha256": prediction.prediction_sha256,
            "observation_source_sha256": expected_observation_source_sha256}


@dataclass(frozen=True)
class TrainingSnapshot:
    training_step: int
    writer_effect: float
    b29_specialization: float
    reencoder_effect: float
    carrier_concentration: float
    benchmark_score: float
    unit_id: str
    seed_id: str
    dependency_keys: tuple[str, ...]


def _sustained_crossing(rows: Sequence[TrainingSnapshot], attr: str,
                        threshold: float,
                        sustain_checkpoints: int) -> int | None:
    ordered = sorted(rows, key=lambda value: value.training_step)
    for index, row in enumerate(ordered):
        remainder = ordered[index:]
        if (len(remainder) >= sustain_checkpoints and
                all(float(getattr(value, attr)) >= threshold
                    for value in remainder)):
            return row.training_step
    return None


def training_dynamics_verdict(rows: Sequence[TrainingSnapshot], *,
                              thresholds: Mapping[str, float],
                              sustain_checkpoints: int,
                              identified_fraction_min: float,
                              ordered_fraction_min: float,
                              crossing_equivalence_steps: int,
                              min_clusters: int,
                              n_boot: int = 5000,
                              alpha: float = 0.05,
                              seed: int = 42) -> dict:
    """Test ordered acquisition on an exact repeated unit x checkpoint panel.

    A crossing is the first checkpoint after which the signal stays above its
    frozen threshold through the end of the trajectory (with at least
    ``sustain_checkpoints`` observations remaining).  Thus a noisy one-step
    spike cannot establish acquisition.  The headline requires both a lower
    confidence bound on the fraction of fully identified/ordered runs and
    upper confidence bounds on every adjacent reversal gap.
    """
    attrs = ("writer_effect", "b29_specialization", "reencoder_effect",
             "carrier_concentration", "benchmark_score")
    if set(thresholds) != set(attrs):
        raise ValueError(
            "thresholds must contain exactly the five preregistered signals")
    if (not rows or sustain_checkpoints < 2 or
            not 0 <= identified_fraction_min <= 1 or
            not 0 <= ordered_fraction_min <= 1 or
            crossing_equivalence_steps < 0 or min_clusters < 2 or
            n_boot < 100 or not 0 < alpha < 1 or
            not np.isfinite([float(thresholds[attr]) for attr in attrs]).all()):
        raise ValueError("invalid training-dynamics design or margins")
    numeric = [(row.training_step, *(float(getattr(row, attr)) for attr in attrs))
               for row in rows]
    if not np.isfinite(numeric).all():
        raise ValueError("training-dynamics table contains non-finite values")
    if any(row.training_step < 0 or not row.unit_id or not row.seed_id or
           not row.dependency_keys for row in rows):
        raise ValueError(
            "every training snapshot needs a non-negative step and provenance")
    keys = [(row.unit_id, row.training_step) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("training-dynamics unit/checkpoint cells are duplicated")

    unit_ids = tuple(sorted({row.unit_id for row in rows}))
    by_unit = {unit_id: [row for row in rows if row.unit_id == unit_id]
               for unit_id in unit_ids}
    grids = {tuple(sorted(row.training_step for row in unit_rows))
             for unit_rows in by_unit.values()}
    if len(grids) != 1:
        raise RuntimeError("training dynamics is not an exact unit x checkpoint grid")
    checkpoint_grid = next(iter(grids))
    if (len(checkpoint_grid) < sustain_checkpoints or
            any(a >= b for a, b in zip(checkpoint_grid, checkpoint_grid[1:]))):
        raise ValueError("training checkpoint grid is too short or unordered")

    seed_by_unit: dict[str, str] = {}
    dependency_by_unit: dict[str, tuple[str, ...]] = {}
    for unit_id, unit_rows in by_unit.items():
        seeds = {row.seed_id for row in unit_rows}
        dependencies = {row.dependency_keys for row in unit_rows}
        if len(seeds) != 1 or len(dependencies) != 1:
            raise RuntimeError(
                f"unit {unit_id!r} changes seed/dependency across checkpoints")
        seed_by_unit[unit_id] = next(iter(seeds))
        dependency_by_unit[unit_id] = next(iter(dependencies))
    if len(set(seed_by_unit.values())) != len(seed_by_unit):
        raise RuntimeError("independent training units reuse a random seed")

    penalty = float(checkpoint_grid[-1] - checkpoint_grid[0] +
                    crossing_equivalence_steps + 1)
    summaries = []
    crossings_by_unit: dict[str, dict[str, int | None]] = {}
    for unit_id, unit_rows in by_unit.items():
        crossings = {
            attr: _sustained_crossing(
                unit_rows, attr, float(thresholds[attr]), sustain_checkpoints)
            for attr in attrs
        }
        crossings_by_unit[unit_id] = crossings
        identified = all(value is not None for value in crossings.values())
        strict_ordered = identified and all(
            int(crossings[a]) <= int(crossings[b])
            for a, b in zip(attrs, attrs[1:]))
        equivalence_ordered = identified and all(
            int(crossings[a]) <= int(crossings[b]) + crossing_equivalence_steps
            for a, b in zip(attrs, attrs[1:]))
        reversal_gaps = {}
        for a, b in zip(attrs, attrs[1:]):
            name = f"{a}_before_{b}"
            reversal_gaps[name] = (
                float(max(0, int(crossings[a]) - int(crossings[b])))
                if identified else penalty)
        summaries.append({
            "unit_id": unit_id,
            "keys": dependency_by_unit[unit_id],
            "identified": float(identified),
            "strict_ordered": float(strict_ordered),
            "equivalence_ordered": float(equivalence_ordered),
            **reversal_gaps,
        })

    def boot(field: str, offset: int):
        return cluster_bootstrap(
            [Row(float(item[field]), keys=item["keys"], unit_id=item["unit_id"])
             for item in summaries],
            n_boot=n_boot, alpha=alpha, seed=seed + offset,
            min_clusters=min_clusters,
        )

    estimates = {
        "identified_fraction": boot("identified", 0),
        "strict_ordered_fraction": boot("strict_ordered", 1),
        "equivalence_ordered_fraction": boot("equivalence_ordered", 2),
    }
    reversal_names = [f"{a}_before_{b}" for a, b in zip(attrs, attrs[1:])]
    reversal_estimates = {
        name: boot(name, 3 + index)
        for index, name in enumerate(reversal_names)
    }
    checks = {
        "identified_fraction": (
            estimates["identified_fraction"].lo >= identified_fraction_min),
        "ordered_fraction": (
            estimates["equivalence_ordered_fraction"].lo >= ordered_fraction_min),
        "adjacent_reversals_equivalent": all(
            estimate.hi <= crossing_equivalence_steps
            for estimate in reversal_estimates.values()),
    }
    supported = all(checks.values())
    if supported:
        claim = "ordered mechanism acquisition"
        bounded_alternative = None
    elif checks["identified_fraction"]:
        claim = "training-order claim not established"
        bounded_alternative = (
            "mechanism signals are acquired, but their preregistered order is "
            "heterogeneous or reversed")
    else:
        claim = "training-order claim not established"
        bounded_alternative = (
            "only partial or non-sustained mechanism acquisition is supported")
    return {
        "claim": claim,
        "bounded_alternative": bounded_alternative,
        "checks": checks,
        "crossings_by_unit": crossings_by_unit,
        "expected_order": list(attrs),
        "checkpoint_grid": list(checkpoint_grid),
        "estimates": {
            **{name: estimate.as_row() for name, estimate in estimates.items()},
            "adjacent_reversal_gaps": {
                name: estimate.as_row()
                for name, estimate in reversal_estimates.items()},
        },
        "margins": {
            "sustain_checkpoints": sustain_checkpoints,
            "identified_fraction_min": identified_fraction_min,
            "ordered_fraction_min": ordered_fraction_min,
            "crossing_equivalence_steps": crossing_equivalence_steps,
        },
        "bootstrap": {"n_boot": n_boot, "alpha": alpha,
                      "min_clusters": min_clusters, "seed": seed},
    }


@dataclass(frozen=True)
class ReplicationPlan:
    criteria: tuple[tuple[str, float, int], ...]
    required_domains: tuple[str, ...]
    source_sha256_by_domain: tuple[tuple[str, str], ...]
    analysis_config_sha256: str
    sha256: str

    def _payload(self) -> dict:
        return {
            "criteria": [list(item) for item in self.criteria],
            "required_domains": list(self.required_domains),
            "source_sha256_by_domain": [list(item)
                                         for item in self.source_sha256_by_domain],
            "analysis_config_sha256": self.analysis_config_sha256,
        }

    def verify(self) -> None:
        mechanisms = [name for name, _, _ in self.criteria]
        domains = list(self.required_domains)
        sources = dict(self.source_sha256_by_domain)
        if (not mechanisms or len(set(mechanisms)) != len(mechanisms) or
                any(not name or not np.isfinite(margin) or margin < 0 or sign not in (-1, 1)
                    for name, margin, sign in self.criteria) or
                not domains or len(set(domains)) != len(domains) or
                any(not domain for domain in domains) or set(sources) != set(domains) or
                any(not _is_sha256(value) for value in sources.values()) or
                not _is_sha256(self.analysis_config_sha256)):
            raise ValueError("invalid replication plan")
        if _json_sha256(self._payload()) != self.sha256:
            raise RuntimeError("replication plan changed after sealing")


def seal_replication_plan(*, minimum_effect: Mapping[str, float],
                          expected_sign: Mapping[str, int],
                          required_domains: Sequence[str],
                          source_sha256_by_domain: Mapping[str, str],
                          analysis_config_sha256: str) -> ReplicationPlan:
    if set(minimum_effect) != set(expected_sign):
        raise ValueError("minimum-effect and expected-sign mechanisms differ")
    criteria = tuple(sorted((str(name), float(minimum_effect[name]),
                             int(expected_sign[name]))
                            for name in minimum_effect))
    domains = tuple(str(value) for value in required_domains)
    sources = tuple(sorted((str(key), str(value))
                           for key, value in source_sha256_by_domain.items()))
    provisional = ReplicationPlan(
        criteria=criteria, required_domains=domains,
        source_sha256_by_domain=sources,
        analysis_config_sha256=str(analysis_config_sha256), sha256="0" * 64)
    result = ReplicationPlan(
        criteria=criteria, required_domains=domains,
        source_sha256_by_domain=sources,
        analysis_config_sha256=str(analysis_config_sha256),
        sha256=_json_sha256(provisional._payload()))
    result.verify()
    return result


@dataclass(frozen=True)
class ReplicationCell:
    mechanism: str
    domain: str
    unit_id: str
    effect: float
    source_artifact_sha256: str
    plan_sha256: str
    dependency_keys: tuple[str, ...]


def robustness_conjunction(rows: Sequence[ReplicationCell], *,
                           plan: ReplicationPlan, min_clusters: int,
                           n_boot: int = 5000, alpha: float = 0.05,
                           seed: int = 42) -> dict:
    """Compute every preregistered mechanism x domain CI from raw clusters."""
    plan.verify()
    if min_clusters < 2 or n_boot < 100 or not 0 < alpha < 1:
        raise ValueError("invalid replication bootstrap configuration")
    criteria = {name: (margin, sign) for name, margin, sign in plan.criteria}
    mechanisms = tuple(sorted(criteria))
    domains = plan.required_domains
    sources = dict(plan.source_sha256_by_domain)
    expected_grid = {(mechanism, domain)
                     for mechanism in mechanisms for domain in domains}
    seen_units: set[tuple[str, str, str]] = set()
    observed_grid = set()
    for row in rows:
        key = (row.mechanism, row.domain)
        unit_key = (*key, row.unit_id)
        if unit_key in seen_units:
            raise ValueError(f"duplicate replication observation {unit_key!r}")
        seen_units.add(unit_key)
        observed_grid.add(key)
        if (not row.unit_id or not row.dependency_keys or
                not np.isfinite(row.effect) or row.plan_sha256 != plan.sha256 or
                row.source_artifact_sha256 != sources.get(row.domain)):
            raise RuntimeError(f"replication row violates sealed plan/source {unit_key!r}")
    missing = sorted(expected_grid - observed_grid)
    extra = sorted(observed_grid - expected_grid)
    if missing or extra:
        raise ValueError(
            f"replication grid differs from preregistration; missing={missing}, extra={extra}")

    for domain in domains:
        panels = [{row.unit_id for row in rows
                   if row.domain == domain and row.mechanism == mechanism}
                  for mechanism in mechanisms]
        if any(panel != panels[0] for panel in panels[1:]):
            raise RuntimeError(
                f"replication mechanisms use different unit panels in {domain!r}")

    out = {}
    for mechanism_index, mechanism in enumerate(mechanisms):
        margin, sign = criteria[mechanism]
        checks = {}
        estimates = {}
        for domain_index, domain in enumerate(domains):
            cell_rows = [row for row in rows
                         if row.mechanism == mechanism and row.domain == domain]
            estimate = cluster_bootstrap(
                [Row(row.effect, keys=row.dependency_keys, unit_id=row.unit_id)
                 for row in cell_rows],
                n_boot=n_boot, alpha=alpha,
                seed=seed + mechanism_index * len(domains) + domain_index,
                min_clusters=min_clusters)
            estimates[domain] = estimate.as_row()
            checks[domain] = (estimate.lo >= margin if sign > 0 else
                              estimate.hi <= -margin)
        out[mechanism] = {"passed": all(checks.values()), "domains": checks,
                          "estimates": estimates, "margin": margin,
                          "expected_sign": sign}
    return {"mechanisms": out, "all_passed": bool(out) and
            all(v["passed"] for v in out.values()),
            "plan_sha256": plan.sha256,
            "bootstrap": {"min_clusters": min_clusters, "n_boot": n_boot,
                          "alpha": alpha, "seed": seed}}

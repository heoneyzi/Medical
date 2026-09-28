"""EXP3 Phase 8 — optional SAE/crosscoder annotation, never first-line evidence.

Feature models are opened only after the raw intervention chain and a learned
delta transport model have passed.  They may provide a sparse vocabulary for
the already-established computation.  Reconstruction quality alone is never a
mechanistic result; held-out intervention prediction, feature ablation and a
decoded-component rescue are required.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .exp2_api import Row, cluster_bootstrap


Mode = Literal["state_sae", "delta_autoencoder", "delta_transcoder"]


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


@dataclass(frozen=True)
class FeatureModelGate:
    raw_identity_passed: bool
    b29_role_resolved: bool
    direction_over_scale_passed: bool
    b30_mediation_passed: bool
    predicted_delta_rescue_passed: bool
    locked_chromosome_available: bool

    def assert_open(self) -> None:
        failed = [name for name, value in self.__dict__.items() if not value]
        if failed:
            raise RuntimeError(
                "SAE/crosscoder wave remains closed; upstream gates failed: "
                + ", ".join(failed)
            )


class SparseFeatureModel(nn.Module):
    """Bias-free sparse encoder/decoder with an explicit scientific mode."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, *, mode: Mode):
        super().__init__()
        if mode not in {"state_sae", "delta_autoencoder", "delta_transcoder"}:
            raise ValueError(mode)
        if min(d_in, d_hidden, d_out) <= 0:
            raise ValueError("feature dimensions must be positive")
        if mode in {"state_sae", "delta_autoencoder"} and d_in != d_out:
            raise ValueError(f"{mode} reconstructs its input and requires d_in=d_out")
        self.mode = mode
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError("feature model input must be [n,d]")
        code = torch.relu(self.encoder(x))
        return self.decoder(code), code

    @torch.no_grad()
    def decoded_feature(self, feature: int, activation: Tensor) -> Tensor:
        if not 0 <= feature < self.encoder.out_features:
            raise IndexError(feature)
        if activation.shape[-1] != 1:
            activation = activation.reshape(-1, 1)
        return activation * self.decoder.weight[:, feature].reshape(1, -1)


@dataclass(frozen=True)
class FittedFeatureModel:
    model: SparseFeatureModel
    candidate_name: str
    mode: Mode
    dev_mse: float
    dev_r2: float
    active_fraction: float
    dev_active_features: int
    model_state_sha256: str
    artifact_sha256: str
    fit_provenance_sha256: str
    feature_spec_sha256: str
    feature_dictionary_sha256: str
    feature_ids: tuple[str, ...]
    train_split: str
    dev_split: str
    train_unit_ids: tuple[str, ...]
    dev_unit_ids: tuple[str, ...]
    train_source_sha256: str
    dev_source_sha256: str

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.validate_artifact()
        self.model.eval()
        a = np.asarray(x, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != self.model.encoder.in_features \
                or not np.isfinite(a).all():
            raise ValueError("prediction input must be a finite [n,d_in] matrix")
        parameter = next(self.model.parameters())
        y, code = self.model(torch.as_tensor(a, dtype=parameter.dtype,
                                             device=parameter.device))
        return y.cpu().numpy().astype(np.float64), code.cpu().numpy().astype(np.float64)

    def validate_artifact(self) -> None:
        state_sha = _hash_model(self.model)
        if state_sha != self.model_state_sha256:
            raise RuntimeError("feature-model parameters changed after artifact sealing")
        dictionary_sha = _digest({
            "model_state_sha256": state_sha,
            "feature_spec_sha256": self.feature_spec_sha256,
            "feature_ids": list(self.feature_ids),
        })
        if dictionary_sha != self.feature_dictionary_sha256:
            raise RuntimeError("feature dictionary no longer matches the sealed model")
        artifact_sha = _digest({
            "candidate_name": self.candidate_name,
            "mode": self.mode,
            "model_state_sha256": state_sha,
            "fit_provenance_sha256": self.fit_provenance_sha256,
            "feature_dictionary_sha256": dictionary_sha,
        })
        if artifact_sha != self.artifact_sha256:
            raise RuntimeError("feature-model artifact provenance mismatch")


def _hash_model(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(_digest({"shape": list(array.shape)}).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def fit_sparse_feature_model(train_x: np.ndarray, train_target: np.ndarray,
                             dev_x: np.ndarray, dev_target: np.ndarray, *,
                             mode: Mode, hidden: int, l1: float,
                             lr: float, epochs: int, seed: int,
                             train_split: str, dev_split: str,
                             train_unit_ids: Sequence[str],
                             dev_unit_ids: Sequence[str],
                             train_source_sha256: str,
                             dev_source_sha256: str,
                             feature_spec: str,
                             candidate_name: str,
                             gate: FeatureModelGate) -> FittedFeatureModel:
    """Fit after the gate; hyperparameters must already be development choices."""
    gate.assert_open()
    arrays = [np.asarray(a, dtype=np.float32)
              for a in (train_x, train_target, dev_x, dev_target)]
    tx, ty, dx, dy = arrays
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("feature arrays must be finite matrices")
    if len(tx) == 0 or len(dx) == 0:
        raise ValueError("feature fitting needs non-empty train and development rows")
    if len(tx) != len(ty) or len(dx) != len(dy) or tx.shape[1] != dx.shape[1] \
            or ty.shape[1] != dy.shape[1]:
        raise ValueError("train/dev feature or target shapes do not align")
    if mode in {"state_sae", "delta_autoencoder"} and (
            not np.array_equal(tx, ty) or not np.array_equal(dx, dy)):
        raise ValueError(
            f"{mode} is an autoencoder and requires target arrays identical to inputs; "
            "use delta_transcoder for a cross-stage target"
        )
    if train_split == dev_split:
        raise ValueError("train and development splits must differ")
    train_ids = tuple(map(str, train_unit_ids))
    dev_ids = tuple(map(str, dev_unit_ids))
    if len(train_ids) != len(tx) or len(dev_ids) != len(dx):
        raise ValueError("train/dev unit ids must align exactly with feature rows")
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("train/dev unit ids must be unique")
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and development unit ids must be disjoint")
    if not _valid_sha256(train_source_sha256) or not _valid_sha256(dev_source_sha256):
        raise ValueError("train/dev feature inputs require source SHA-256 commitments")
    if not feature_spec.strip() or not candidate_name.strip():
        raise ValueError("feature specification and candidate name must be non-empty")
    if hidden <= 0 or l1 < 0 or lr <= 0 or epochs <= 0:
        raise ValueError("invalid sparse feature hyperparameters")
    feature_spec_sha = _digest({"feature_spec": feature_spec})
    fit_provenance_sha = _digest({
        "candidate_name": candidate_name, "mode": mode,
        "train_split": train_split, "dev_split": dev_split,
        "train_unit_ids": list(train_ids), "dev_unit_ids": list(dev_ids),
        "train_source_sha256": train_source_sha256,
        "dev_source_sha256": dev_source_sha256,
        "feature_spec_sha256": feature_spec_sha,
        "hyperparameters": {
            "hidden": hidden, "l1": l1, "lr": lr,
            "epochs": epochs, "seed": seed,
        },
    })
    # A feature-model side experiment must not perturb the parent experiment's
    # global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SparseFeatureModel(tx.shape[1], hidden, ty.shape[1], mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X, Y = torch.from_numpy(tx), torch.from_numpy(ty)
        for _ in range(epochs):
            pred, code = model(X)
            loss = torch.mean((pred - Y) ** 2) + l1 * torch.mean(torch.abs(code))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred, code = model(torch.from_numpy(dx))
    p, c = pred.numpy(), code.numpy()
    mse = float(np.mean((p - dy) ** 2))
    den = float(np.sum((dy - dy.mean(0)) ** 2))
    r2 = float(1 - np.sum((p - dy) ** 2) / max(den, 1e-30))
    state_sha = _hash_model(model)
    feature_ids = tuple(f"{candidate_name}:f{i:05d}" for i in range(hidden))
    dictionary_sha = _digest({
        "model_state_sha256": state_sha,
        "feature_spec_sha256": feature_spec_sha,
        "feature_ids": list(feature_ids),
    })
    artifact_sha = _digest({
        "candidate_name": candidate_name, "mode": mode,
        "model_state_sha256": state_sha,
        "fit_provenance_sha256": fit_provenance_sha,
        "feature_dictionary_sha256": dictionary_sha,
    })
    fitted = FittedFeatureModel(
        model=model, candidate_name=candidate_name, mode=mode,
        dev_mse=mse, dev_r2=r2, active_fraction=float(np.mean(c > 0)),
        dev_active_features=int(np.sum(np.any(c > 0, axis=0))),
        model_state_sha256=state_sha, artifact_sha256=artifact_sha,
        fit_provenance_sha256=fit_provenance_sha,
        feature_spec_sha256=feature_spec_sha,
        feature_dictionary_sha256=dictionary_sha,
        feature_ids=feature_ids, train_split=train_split, dev_split=dev_split,
        train_unit_ids=train_ids, dev_unit_ids=dev_ids,
        train_source_sha256=train_source_sha256,
        dev_source_sha256=dev_source_sha256,
    )
    fitted.validate_artifact()
    return fitted


@dataclass(frozen=True)
class ComplexityCandidate:
    name: str
    dev_error_mean: float
    dev_error_se: float
    active_features: int
    parameters: int
    artifact_sha256: str


@dataclass(frozen=True)
class ComplexitySelection:
    selected_name: str
    candidates: tuple[ComplexityCandidate, ...]
    candidate_names: tuple[str, ...]
    candidates_sha256: str
    selection_sha256: str

    def validate(self) -> None:
        if not self.candidates:
            raise ValueError("sealed complexity selection has no candidates")
        if any(not _valid_sha256(c.artifact_sha256) for c in self.candidates):
            raise ValueError("every complexity candidate requires an artifact SHA-256")
        if tuple(c.name for c in self.candidates) != self.candidate_names:
            raise ValueError("complexity candidate names/order changed after sealing")
        snapshot = [c.__dict__ for c in self.candidates]
        if _digest(snapshot) != self.candidates_sha256:
            raise ValueError("complexity candidate table commitment mismatch")
        best = min(self.candidates, key=lambda c: c.dev_error_mean)
        eligible = [c for c in self.candidates
                    if c.dev_error_mean <= best.dev_error_mean + best.dev_error_se]
        expected_name = min(
            eligible, key=lambda c: (c.active_features, c.parameters,
                                     c.dev_error_mean, c.name)).name
        if self.selected_name != expected_name:
            raise ValueError("selected model is not the one-SE least-complex choice")
        expected_sha = _digest({
            "rule": "one_standard_error_least_complex",
            "candidates_sha256": self.candidates_sha256,
            "selected_name": self.selected_name,
        })
        if self.selection_sha256 != expected_sha:
            raise ValueError("one-SE selection commitment hash mismatch")


def one_standard_error_selection(
        candidates: Sequence[ComplexityCandidate]) -> ComplexitySelection:
    """Seal the least-complex model within one SE of the best dev error."""
    if not candidates:
        raise ValueError("no candidates")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("complexity candidate names must be unique")
    if any(not np.isfinite((c.dev_error_mean, c.dev_error_se)).all()
           or c.dev_error_mean < 0 or c.dev_error_se < 0
           or c.active_features <= 0 or c.parameters <= 0 for c in candidates):
        raise ValueError("invalid complexity candidate")
    best = min(candidates, key=lambda c: c.dev_error_mean)
    eligible = [c for c in candidates
                if c.dev_error_mean <= best.dev_error_mean + best.dev_error_se]
    selected = min(eligible, key=lambda c: (c.active_features, c.parameters,
                                            c.dev_error_mean, c.name)).name
    ordered = tuple(sorted(candidates, key=lambda item: item.name))
    snapshot = [c.__dict__ for c in ordered]
    candidates_sha = _digest(snapshot)
    result = ComplexitySelection(
        selected_name=selected,
        candidates=ordered,
        candidate_names=tuple(item["name"] for item in snapshot),
        candidates_sha256=candidates_sha,
        selection_sha256=_digest({
            "rule": "one_standard_error_least_complex",
            "candidates_sha256": candidates_sha,
            "selected_name": selected,
        }),
    )
    result.validate()
    return result


def one_standard_error_choice(candidates: Sequence[ComplexityCandidate]) -> str:
    """Compatibility wrapper returning the sealed one-SE choice's name."""
    return one_standard_error_selection(candidates).selected_name


@dataclass(frozen=True)
class FrozenFeatureSet:
    model_name: str
    model_artifact_sha256: str
    feature_ids: tuple[str, ...]
    selected_on_split: str
    selected_on_source_sha256: str
    selection_method: str
    selection_evidence_sha256: str
    feature_set_sha256: str

    def validate(self, fitted: FittedFeatureModel) -> None:
        fitted.validate_artifact()
        if self.model_name != fitted.candidate_name \
                or self.model_artifact_sha256 != fitted.artifact_sha256:
            raise ValueError("frozen feature set belongs to a different fitted model")
        if self.selected_on_split != fitted.dev_split \
                or self.selected_on_source_sha256 != fitted.dev_source_sha256:
            raise ValueError("feature identities must be frozen on the development split")
        if not self.feature_ids or len(set(self.feature_ids)) != len(self.feature_ids) \
                or not set(self.feature_ids).issubset(fitted.feature_ids):
            raise ValueError("frozen feature ids are invalid for the fitted dictionary")
        if self.selection_method not in {
            "development_ablation", "development_transport", "development_stability",
        }:
            raise ValueError("feature selection method was not pre-registered")
        if not _valid_sha256(self.selection_evidence_sha256):
            raise ValueError("feature selection requires a sealed development evidence table")
        expected = _digest({
            "model_name": self.model_name,
            "model_artifact_sha256": self.model_artifact_sha256,
            "feature_ids": list(self.feature_ids),
            "selected_on_split": self.selected_on_split,
            "selected_on_source_sha256": self.selected_on_source_sha256,
            "selection_method": self.selection_method,
            "selection_evidence_sha256": self.selection_evidence_sha256,
        })
        if expected != self.feature_set_sha256:
            raise ValueError("frozen feature-set commitment hash mismatch")


def freeze_development_feature_set(
        fitted: FittedFeatureModel, feature_ids: Sequence[str], *,
        selection_method: str,
        selection_evidence_sha256: str) -> FrozenFeatureSet:
    ids = tuple(map(str, feature_ids))
    payload = {
        "model_name": fitted.candidate_name,
        "model_artifact_sha256": fitted.artifact_sha256,
        "feature_ids": list(ids),
        "selected_on_split": fitted.dev_split,
        "selected_on_source_sha256": fitted.dev_source_sha256,
        "selection_method": selection_method,
        "selection_evidence_sha256": selection_evidence_sha256,
    }
    frozen = FrozenFeatureSet(
        model_name=fitted.candidate_name,
        model_artifact_sha256=fitted.artifact_sha256,
        feature_ids=ids,
        selected_on_split=fitted.dev_split,
        selected_on_source_sha256=fitted.dev_source_sha256,
        selection_method=selection_method,
        selection_evidence_sha256=selection_evidence_sha256,
        feature_set_sha256=_digest(payload),
    )
    frozen.validate(fitted)
    return frozen


@dataclass(frozen=True)
class FeatureCausalRow:
    unit_id: str
    model_name: str
    heldout_delta_relative_energy: float
    feature_ablation_effect: float
    decoded_rescue_fraction: float
    random_feature_effect: float
    wrong_layer_effect: float
    cross_chromosome_cosine: float
    reconstruction_r2: float
    model_artifact_sha256: str
    fit_provenance_sha256: str
    feature_set_sha256: str
    complexity_selection_sha256: str
    heldout_split: str
    heldout_source_sha256: str
    dependency_keys: tuple[str, ...] = ()


def feature_model_verdict(rows: Sequence[FeatureCausalRow], *,
                          fitted_models: Mapping[str, FittedFeatureModel],
                          complexity_selections: Mapping[str, ComplexitySelection],
                          frozen_feature_sets: Mapping[str, FrozenFeatureSet],
                          locked_unit_ids: Sequence[str],
                          locked_split: str,
                          locked_source_sha256: str,
                          delta_relative_energy_min: float, ablation_min: float,
                          rescue_min: float, control_max: float,
                          stability_min: float, min_clusters: int = 2,
                          n_boot: int = 5_000, seed: int = 42,
                          alpha: float = 0.05) -> dict:
    """Judge sparse features with cluster CIs; reconstruction is descriptive.

    Positive requirements use the lower confidence bound.  Random-feature and
    wrong-layer controls use an intersection-union equivalence rule: their
    *entire* intervals must lie inside ``[-control_max, control_max]``.  Rows
    without explicit dependency keys fall back to one cluster per ``unit_id``
    so callers cannot accidentally treat repeated positions from the same
    genomic window as independent once proper keys are supplied.
    """
    if not rows:
        raise ValueError("feature causal table is empty")
    if min_clusters < 2 or n_boot < 1 or not (0 < alpha < 1):
        raise ValueError("invalid cluster-inference configuration")
    if min(delta_relative_energy_min, ablation_min, rescue_min,
           control_max, stability_min) < 0:
        raise ValueError("feature-model margins must be non-negative")
    locked_ids = tuple(map(str, locked_unit_ids))
    if not locked_ids or len(set(locked_ids)) != len(locked_ids):
        raise ValueError("locked unit ids must be non-empty, unique and ordered")
    if not locked_split or not _valid_sha256(locked_source_sha256):
        raise ValueError("locked feature evaluation requires split and source SHA-256")
    by_model: dict[str, list[FeatureCausalRow]] = {}
    for r in rows:
        numeric = (r.heldout_delta_relative_energy, r.feature_ablation_effect,
                   r.decoded_rescue_fraction, r.random_feature_effect,
                   r.wrong_layer_effect, r.cross_chromosome_cosine,
                   r.reconstruction_r2)
        if not r.unit_id or not r.model_name or not np.isfinite(numeric).all():
            raise ValueError("invalid feature causal row")
        by_model.setdefault(r.model_name, []).append(r)
    model_names = set(by_model)
    for label, mapping in (
            ("fitted model", fitted_models),
            ("complexity selection", complexity_selections),
            ("frozen feature set", frozen_feature_sets)):
        if set(mapping) != model_names:
            raise ValueError(
                f"{label} keys must exactly match causal-table models; "
                f"expected={sorted(model_names)}, observed={sorted(mapping)}"
            )
    results = {}
    causal_fields = (
        "heldout_delta_relative_energy", "feature_ablation_effect", "decoded_rescue_fraction",
        "random_feature_effect", "wrong_layer_effect", "cross_chromosome_cosine",
    )
    for model_index, (name, rs) in enumerate(sorted(by_model.items())):
        observed_ids = tuple(r.unit_id for r in rs)
        if observed_ids != locked_ids:
            raise ValueError(
                f"feature model {name!r} rows do not exactly match locked-unit order"
            )
        fitted = fitted_models[name]
        fitted.validate_artifact()
        if fitted.candidate_name != name:
            raise ValueError("fitted candidate name does not match causal table")
        if locked_split in {fitted.train_split, fitted.dev_split} \
                or set(locked_ids) & (set(fitted.train_unit_ids) | set(fitted.dev_unit_ids)):
            raise ValueError("locked feature evaluation overlaps model train/development data")
        selection = complexity_selections[name]
        selection.validate()
        if selection.selected_name != name or name not in selection.candidate_names:
            raise ValueError("causal feature model was not selected by the sealed one-SE rule")
        selected_candidate = next(
            candidate for candidate in selection.candidates
            if candidate.name == selection.selected_name
        )
        expected_parameters = sum(parameter.numel()
                                  for parameter in fitted.model.parameters())
        if selected_candidate.artifact_sha256 != fitted.artifact_sha256 \
                or selected_candidate.parameters != expected_parameters \
                or selected_candidate.active_features != fitted.dev_active_features:
            raise ValueError(
                "selected complexity row is not bound to the fitted model artifact"
            )
        feature_set = frozen_feature_sets[name]
        feature_set.validate(fitted)
        for r in rs:
            if r.model_artifact_sha256 != fitted.artifact_sha256 \
                    or r.fit_provenance_sha256 != fitted.fit_provenance_sha256:
                raise ValueError("causal row is not bound to the fitted model artifact")
            if r.feature_set_sha256 != feature_set.feature_set_sha256:
                raise ValueError("causal row uses a feature set not frozen on development")
            if r.complexity_selection_sha256 != selection.selection_sha256:
                raise ValueError("causal row uses an unselected model complexity")
            if r.heldout_split != locked_split \
                    or r.heldout_source_sha256 != locked_source_sha256:
                raise ValueError("causal row held-out provenance does not match locked input")
        means = {f: float(np.mean([getattr(r, f) for r in rs])) for f in (
            "heldout_delta_relative_energy", "feature_ablation_effect", "decoded_rescue_fraction",
            "random_feature_effect", "wrong_layer_effect", "cross_chromosome_cosine",
            "reconstruction_r2")}
        estimates = {}
        for field_index, field in enumerate(causal_fields):
            estimate = cluster_bootstrap(
                [Row(float(getattr(r, field)),
                     keys=(r.dependency_keys or (f"unit:{r.unit_id}",))) for r in rs],
                min_clusters=min_clusters,
                n_boot=n_boot,
                seed=seed + 100 * model_index + field_index,
                alpha=alpha,
            )
            estimates[field] = estimate
        checks = {
            "predicts_heldout_interventions":
                estimates["heldout_delta_relative_energy"].lo
                >= delta_relative_energy_min,
            "feature_is_necessary":
                estimates["feature_ablation_effect"].lo >= ablation_min,
            "decoded_component_rescues":
                estimates["decoded_rescue_fraction"].lo >= rescue_min,
            "random_feature_specificity":
                estimates["random_feature_effect"].lo >= -control_max
                and estimates["random_feature_effect"].hi <= control_max,
            "wrong_layer_specificity":
                estimates["wrong_layer_effect"].lo >= -control_max
                and estimates["wrong_layer_effect"].hi <= control_max,
            "cross_chromosome_stability":
                estimates["cross_chromosome_cosine"].lo >= stability_min,
        }
        results[name] = {"causal_feature_model": all(checks.values()),
                         "checks": checks, "means": means,
                         "provenance": {
                             "model_artifact_sha256": fitted.artifact_sha256,
                             "fit_provenance_sha256": fitted.fit_provenance_sha256,
                             "feature_set_sha256": feature_set.feature_set_sha256,
                             "complexity_selection_sha256": selection.selection_sha256,
                             "locked_split": locked_split,
                             "locked_source_sha256": locked_source_sha256,
                             "ordered_unit_ids": list(locked_ids),
                         },
                         "cluster_estimates": {
                             field: estimate.as_row()
                             for field, estimate in estimates.items()
                         },
                         "note": "reconstruction_r2 is not a causal check"}
    return {"models": results,
            "passing": [m for m, r in results.items() if r["causal_feature_model"]]}

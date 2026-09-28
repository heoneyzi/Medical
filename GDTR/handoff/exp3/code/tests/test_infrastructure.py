"""Infrastructure tests for concrete data/config/callback and optional models."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np
import torch
from torch import nn

from exp3.artifacts import ArtifactStore
from exp3.config import Exp3Config
from exp3.data import TokenPair, TokenPanel, TokenUnit, load_token_panel_refs
from exp3.optional_features import (
    ComplexityCandidate, FeatureCausalRow, FeatureModelGate,
    feature_model_verdict, fit_sparse_feature_model,
    freeze_development_feature_set, one_standard_error_choice,
    one_standard_error_selection,
)


OK = []
def check(name, value, detail=""):
    OK.append(bool(value))
    print(("PASS " if value else "FAIL ") + name + (f"   {detail}" if detail else ""))


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


cfg = Exp3Config()
check("extended config contains mandatory dense 1e-5..1 scan",
      cfg.alpha_doses[0] <= 1e-5 and cfg.alpha_doses[-1] == 1 and len(cfg.alpha_doses) >= 8)
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "config.json"
    cfg.save(path)
    again = Exp3Config.load(path)
    check("extended config is content-hash sealed", again.config_sha256 == cfg.config_sha256)

    tok = lambda s: [ord(c) for c in s]
    units = [
        (TokenUnit("a", "chr22", 0, 4, dependency_keys=("locus:a",)), "ACGT"),
        (TokenUnit("b", "chr22", 10, 14, dependency_keys=("locus:b",)), "AGGT"),
        (TokenUnit("c", "chr22", 20, 24, dependency_keys=("locus:c",)), "TTTT"),
    ]
    panel = TokenPanel.from_sequences(
        units, tokenizer=tok, pairs=[TokenPair("p", "a", "b", "c")])
    panel_dir = Path(td) / "panel"
    panel.save(panel_dir, panel_name="locked-motif-panel",
               source_split="chr22-locked", source_role="locked",
               created_by="test_infrastructure")
    loaded = TokenPanel.load(panel_dir,
                             expected_tokenizer_fingerprint=panel.tokenizer_fingerprint)
    check("token panel preserves exact ids, metadata and pairs",
          torch.equal(loaded.ids["a"], panel.ids["a"])
          and loaded.pairs[0].wrong_pair_unit == "c")
    check("token panel emits existing experiment tuple contracts",
          len(loaded.loci()[0]) == 4 and len(loaded.matched_pairs()[0]) == 4)
    other = TokenPanel.from_sequences([
        (TokenUnit("d", "chr17", 100, 104, dependency_keys=("locus:a",)), "ACGT")
    ], tokenizer=tok)
    try:
        loaded.assert_split_disjoint(other)
        check("non-overlapping chromosomes cannot share a dependency cluster", False)
    except RuntimeError:
        check("non-overlapping chromosomes cannot share a dependency cluster", True)
    try:
        TokenPanel({"bad": TokenUnit("bad", "chr1", 0, 2)},
                   {"bad": torch.tensor([1.2, 2.0])})
        check("floating token ids are not silently truncated", False)
    except TypeError:
        check("floating token ids are not silently truncated", True)

    panel_ref = loaded.reference()
    bound_refs = load_token_panel_refs((panel_dir,))
    check("run-contract helper returns the exact verified panel reference",
          bound_refs == (panel_ref,))
    artifact_ref = panel_ref.artifact_ref()
    ArtifactStore.verify(artifact_ref)
    check("token panel exports a full composite ArtifactRef",
          artifact_ref.kind == "token_panel"
          and artifact_ref.metadata["panel_metadata_sha256"]
          == loaded.panel_metadata_sha256)

    index_path = panel_dir / "panel.json"
    original_index = index_path.read_bytes()
    payload = json.loads(original_index)
    payload["units"]["a"]["family"] = "tampered-family"
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        TokenPanel.load(panel_dir)
        check("panel metadata mutation is rejected by its internal seal", False)
    except RuntimeError:
        check("panel metadata mutation is rejected by its internal seal", True)
    index_path.write_bytes(original_index)

    # Even an attacker who recomputes the self-seal cannot replace a panel
    # already pinned by the external TokenPanelRef/run contract.
    payload = json.loads(original_index)
    payload["pairs"][0]["family"] = "relabelled"
    payload.pop("panel_metadata_sha256")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()
    payload["panel_metadata_sha256"] = hashlib.sha256(canonical).hexdigest()
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        panel_ref.verify()
        check("external panel reference rejects a resealed substitution", False)
    except RuntimeError:
        check("external panel reference rejects a resealed substitution", True)
    index_path.write_bytes(original_index)

    token_path = panel_dir / "tokens.pt"
    original_tokens = token_path.read_bytes()
    token_path.write_bytes(original_tokens + b"changed")
    try:
        ArtifactStore.verify(artifact_ref)
        check("composite ArtifactRef detects bound token mutation", False)
    except RuntimeError:
        check("composite ArtifactRef detects bound token mutation", True)
    token_path.write_bytes(original_tokens)

    replacement = TokenPanel.from_sequences(
        [(unit, "TTTT" if unit.unit_id == "a" else sequence)
         for unit, sequence in units],
        tokenizer=tok, pairs=[TokenPair("p", "a", "b", "c")])
    try:
        replacement.save(
            panel_dir, panel_name="locked-motif-panel",
            source_split="chr22-locked", source_role="locked",
            created_by="test_infrastructure")
        check("sealed token-panel files cannot be overwritten", False)
    except FileExistsError:
        check("sealed token-panel files cannot be overwritten", True)
    token_path.write_bytes(original_tokens)


gate = FeatureModelGate(True, True, True, True, True, True)
rng = np.random.default_rng(4)
x = rng.normal(size=(100, 5)).astype(np.float32)
W = rng.normal(size=(5, 3)).astype(np.float32)
y = x @ W
train_feature_ids = tuple(f"tr-feature-{i}" for i in range(70))
dev_feature_ids = tuple(f"dv-feature-{i}" for i in range(30))
feature_fit_kwargs = dict(
    train_split="chr22-train", dev_split="chr21-development",
    train_unit_ids=train_feature_ids, dev_unit_ids=dev_feature_ids,
    train_source_sha256=digest("feature-train-source"),
    dev_source_sha256=digest("feature-dev-source"),
    feature_spec="b28-to-m30 intervention delta sparse vocabulary v1",
    candidate_name="delta", gate=gate,
)
fit = fit_sparse_feature_model(
    x[:70], y[:70], x[70:], y[70:], mode="delta_transcoder",
    hidden=16, l1=1e-5, lr=1e-2, epochs=300, seed=1,
    **feature_fit_kwargs)
check("conditional sparse delta feature model actually trains", fit.dev_r2 > .95,
      f"R2={fit.dev_r2:.4f}")

try:
    fit_sparse_feature_model(
        x[:70], y[:70], x[70:], y[70:], mode="state_sae",
        hidden=8, l1=1e-5, lr=1e-2, epochs=2, seed=1,
        **feature_fit_kwargs)
    check("state SAE cannot be trained against a different target", False)
except ValueError:
    check("state SAE cannot be trained against a different target", True)
try:
    fit_sparse_feature_model(
        x[:70], x[:70] * 2, x[70:], x[70:] * 2, mode="state_sae",
        hidden=8, l1=1e-5, lr=1e-2, epochs=1, seed=1,
        **feature_fit_kwargs)
    check("state SAE cannot be mislabeled as a cross-stage predictor", False)
except ValueError:
    check("state SAE cannot be mislabeled as a cross-stage predictor", True)

try:
    fit_sparse_feature_model(
        x[:70], y[:70], x[70:], y[70:], mode="delta_transcoder",
        hidden=8, l1=1e-5, lr=1e-2, epochs=1, seed=1,
        **{**feature_fit_kwargs,
           "dev_unit_ids": train_feature_ids[:30]})
    check("feature fitting rejects train/development unit overlap", False)
except ValueError:
    check("feature fitting rejects train/development unit overlap", True)

choice = one_standard_error_choice([
    ComplexityCandidate("big", .10, .02, 20, 1000, digest("big-model")),
    ComplexityCandidate("small", .115, .02, 5, 200, digest("small-model")),
    ComplexityCandidate("bad", .3, .01, 2, 100, digest("bad-model")),
])
check("one-SE rule selects simplest near-best model", choice == "small")

selection = one_standard_error_selection([
    ComplexityCandidate("delta", .10, .02, fit.dev_active_features, 128,
                        fit.artifact_sha256),
    ComplexityCandidate("too-large", .095, .02, 32, 512,
                        digest("too-large-model")),
])
frozen_features = freeze_development_feature_set(
    fit, fit.feature_ids[:4], selection_method="development_ablation",
    selection_evidence_sha256=digest("development-feature-ablation-table"))
locked_feature_ids = tuple(f"lk-feature-{i}" for i in range(10))
locked_feature_source = digest("locked-feature-source")


def feature_row(unit_id, model_name="delta", dependency_keys=()):
    return FeatureCausalRow(
        unit_id, model_name, .9, .8, .9, .01, .01, .95, .99,
        model_artifact_sha256=fit.artifact_sha256,
        fit_provenance_sha256=fit.fit_provenance_sha256,
        feature_set_sha256=frozen_features.feature_set_sha256,
        complexity_selection_sha256=selection.selection_sha256,
        heldout_split="chr17-locked",
        heldout_source_sha256=locked_feature_source,
        dependency_keys=dependency_keys,
    )


feature_rows = [feature_row(unit_id) for unit_id in locked_feature_ids]
verdict_provenance = dict(
    fitted_models={"delta": fit},
    complexity_selections={"delta": selection},
    frozen_feature_sets={"delta": frozen_features},
    locked_unit_ids=locked_feature_ids, locked_split="chr17-locked",
    locked_source_sha256=locked_feature_source,
)
fv = feature_model_verdict(feature_rows, **verdict_provenance,
                           delta_relative_energy_min=.8, ablation_min=.5,
                           rescue_min=.8, control_max=.05, stability_min=.8)
check("optional feature claim requires intervention and rescue, not reconstruction",
      fv["passing"] == ["delta"])

clustered = [feature_row(unit_id, dependency_keys=("same-window",))
             for unit_id in locked_feature_ids]
try:
    feature_model_verdict(clustered, delta_relative_energy_min=.8, ablation_min=.5,
                          rescue_min=.8, control_max=.05, stability_min=.8,
                          min_clusters=2, **verdict_provenance)
    check("optional feature inference refuses pseudoreplicated positions", False)
except RuntimeError:
    check("optional feature inference refuses pseudoreplicated positions", True)

try:
    feature_model_verdict(list(reversed(feature_rows)), **verdict_provenance,
                          delta_relative_energy_min=.8, ablation_min=.5,
                          rescue_min=.8, control_max=.05, stability_min=.8)
    check("feature verdict rejects a permuted locked-unit order", False)
except ValueError:
    check("feature verdict rejects a permuted locked-unit order", True)

forged_rows = list(feature_rows)
forged = forged_rows[0]
forged_rows[0] = FeatureCausalRow(
    forged.unit_id, forged.model_name, forged.heldout_delta_relative_energy,
    forged.feature_ablation_effect, forged.decoded_rescue_fraction,
    forged.random_feature_effect, forged.wrong_layer_effect,
    forged.cross_chromosome_cosine, forged.reconstruction_r2,
    model_artifact_sha256=digest("forged-model"),
    fit_provenance_sha256=forged.fit_provenance_sha256,
    feature_set_sha256=forged.feature_set_sha256,
    complexity_selection_sha256=forged.complexity_selection_sha256,
    heldout_split=forged.heldout_split,
    heldout_source_sha256=forged.heldout_source_sha256,
)
try:
    feature_model_verdict(forged_rows, **verdict_provenance,
                          delta_relative_energy_min=.8, ablation_min=.5,
                          rescue_min=.8, control_max=.05, stability_min=.8)
    check("feature verdict rejects a substituted model artifact", False)
except ValueError:
    check("feature verdict rejects a substituted model artifact", True)

saved_state = {name: value.detach().clone()
               for name, value in fit.model.state_dict().items()}
with torch.no_grad():
    next(fit.model.parameters()).add_(0.01)
try:
    fit.validate_artifact()
    check("sealed feature artifact detects post-fit parameter mutation", False)
except RuntimeError:
    check("sealed feature artifact detects post-fit parameter mutation", True)
fit.model.load_state_dict(saved_state)
fit.validate_artifact()


print()
print(f"{sum(OK)}/{len(OK)} passed")
if not all(OK):
    raise AssertionError("one or more infrastructure checks failed")
if __name__ == "__main__":
    sys.exit(0 if all(OK) else 1)

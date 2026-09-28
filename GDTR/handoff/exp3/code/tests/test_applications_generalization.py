"""Model-free regression tests for protocol, applications and generalisation."""
import sys
import tempfile
import hashlib
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np
import torch

from exp3.artifacts import ArtifactStore
from exp3.splits import GenomicUnit, assert_no_genomic_overlap
from exp3.applications import (
    CarrierDoseRow, CompressionRow, DiagnosticSignals, MechanismFingerprint,
    ExternalAssayRow, PathUseRow, RepairOutcome, SteeringScore, SteeringValidation,
    VariantCausalRow, beam_search_sequence, carrier_calibration_verdict,
    build_mechanism_compression_plan, canonicalize_update_norm,
    compress_weight_preserve_output_subspace, conditional_path_output,
    compression_comparison, debugger_verdict, diagnose_stage,
    external_assay_verdict, external_causal_triangulation,
    leave_family_out_fingerprint, long_context_path_verdict,
    seal_diagnosis_table, seal_mechanism_fingerprints,
    seal_external_predictions,
    seal_steering_selection, seal_variant_cohort_ids,
    reduce_bilinear_channels,
    select_carrier_dose_development, steering_verdict, variant_causal_verdict,
    variant_geometry,
)
from exp3.generalization import (
    BlindRolePrediction, BlockRoleObservation, CrossCheckpointRow,
    FrozenFeatureTable, ReplicationCell, TrainingSnapshot,
    cross_checkpoint_verdict, label_efficiency_benchmark,
    fit_cross_checkpoint_linear_model, replay_cross_checkpoint_model,
    orthogonal_checkpoint_alignment, robustness_conjunction,
    seal_alignment_anchor_pair, seal_cross_checkpoint_prediction,
    seal_replication_plan, transfer_direction,
    score_blind_role_prediction, training_dynamics_verdict,
)


OK = []
def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name + (f"   {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as td:
    store = ArtifactStore(Path(td) / "artifacts")
    ref = store.put_tensor("u28", torch.eye(4)[:, :2], source_split="chr22-discovery",
                           source_role="discovery", created_by="phase1_u28_freeze")
    ArtifactStore.verify(ref)
    loaded = ArtifactStore.load_tensor(ref)
    check("EXP3 artifact is atomic, hash-verified, and round-trips",
          torch.equal(loaded, torch.eye(4)[:, :2]))
    Path(ref.path).write_bytes(b"tampered")
    try:
        ArtifactStore.verify(ref)
        check("changed artifact cannot enter adaptive interpretation", False)
    except RuntimeError:
        check("changed artifact cannot enter adaptive interpretation", True)


a = [GenomicUnit("a", "chr22", 10, 20, ("a",))]
b = [GenomicUnit("b", "chr17", 10, 20, ("b",))]
assert_no_genomic_overlap(a, b, buffer_bp=10)
try:
    assert_no_genomic_overlap(a, [GenomicUnit("c", "chr22", 19, 30, ("c",))])
    check("genomic overlap leakage is rejected", False)
except RuntimeError:
    check("genomic overlap leakage is rejected", True)


# Label-free variant geometry separates direction, transport and carrier.
g0, g1 = torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.])
m0, m1, c = torch.tensor([1., 0., 2.]), torch.tensor([0., 1., 3.]), torch.tensor([0., 0., 1.])
vg = variant_geometry("v", g28_ref=g0, g28_alt=g1, m30_ref=m0, m30_alt=m1,
                      carrier_axis=c)
check("variant geometry records angular transport and carrier separately",
      vg.angular_g28 > 1.5 and vg.angular_m30_carrier_removed > 1.5
      and abs(vg.delta_carrier - 1) < 1e-12)

sealed_predictions = {f"v{i}": (-1. if i % 2 else 1.) * (1 + i / 20)
                      for i in range(12)}
cohort_by_variant = {f"v{i}": "cohort-A" if i < 6 else "cohort-B"
                     for i in range(12)}
variant_prediction_hash = seal_external_predictions(sealed_predictions)
variant_set_hash, variant_cohort_hash = seal_variant_cohort_ids(cohort_by_variant)
vr = [VariantCausalRow(
    f"v{i}", cohort_by_variant[f"v{i}"],
    .95, .94, .01, .01, .93, .01, .01, .001,
    "reduced-rank-transport", variant_prediction_hash, "chr17-locked",
    True, False, (f"locus:{i}",))
    for i in range(12)]
vv = variant_causal_verdict(vr, rescue_min=.8, null_max=.05,
                            off_target_max=.01, min_clusters=5,
                            expected_prediction_sha256=variant_prediction_hash,
                            expected_prediction_source_split="chr17-locked",
                            expected_variant_set_sha256=variant_set_hash,
                            expected_variant_cohort_sha256=variant_cohort_hash,
                            n_boot=200)
check("variant claim requires transfer, block, predicted rescue and controls",
      vv["claim"] == "direction-mediated causal variant transport"
      and len(vv["causal_evidence_sha256"]) == 64)
try:
    variant_causal_verdict(
        [replace(vr[0], prediction_target_derived=True), *vr[1:]],
        rescue_min=.8, null_max=.05, off_target_max=.01, min_clusters=5,
        expected_prediction_sha256=variant_prediction_hash,
        expected_prediction_source_split="chr17-locked",
        expected_variant_set_sha256=variant_set_hash,
        expected_variant_cohort_sha256=variant_cohort_hash, n_boot=100)
    check("target-derived variant rescue cannot support an application claim", False)
except RuntimeError:
    check("target-derived variant rescue cannot support an application claim", True)


# External labels are opened only after the same signed prediction table is sealed.
prediction_hash = variant_prediction_hash
assay_rows = [ExternalAssayRow(
    variant_id=vid, cohort_id=cohort_by_variant[vid],
    assay_name="mpra-A" if i < 6 else "mpra-B",
    predicted_effect=pred, observed_effect=pred * .98,
    observed_se=.05, baseline_prediction=0., matched_control_effect=.01,
    prediction_sha256=prediction_hash, dependency_keys=(f"locus:{i}",),
) for i, (vid, pred) in enumerate(sealed_predictions.items())]
av = external_assay_verdict(
    assay_rows, expected_prediction_sha256=prediction_hash,
    expected_variant_set_sha256=variant_set_hash,
    expected_variant_cohort_sha256=variant_cohort_hash, labels_opened=True,
    resolvable_z=1.96, sign_concordance_min=.8, error_gain_min=.1,
    specificity_gap_min=.2, min_clusters=5, n_boot=200,
)
tri = external_causal_triangulation(
    av, vv, expected_causal_evidence_sha256=vv["causal_evidence_sha256"],
    expected_prediction_sha256=prediction_hash,
    expected_variant_set_sha256=variant_set_hash,
    expected_variant_cohort_sha256=variant_cohort_hash)
check("sealed blind predictions validate on external assays without replacing causality",
      av["claim"] == "blinded external assay prediction supported"
      and tri["claim"] == "external biological effect and model causal path triangulated")
try:
    external_assay_verdict(
        assay_rows, expected_prediction_sha256="0" * 64,
        expected_variant_set_sha256=variant_set_hash,
        expected_variant_cohort_sha256=variant_cohort_hash, labels_opened=True,
        resolvable_z=1.96, sign_concordance_min=.8, error_gain_min=.1,
        specificity_gap_min=.2, min_clusters=5, n_boot=20,
    )
    check("external assay cannot be joined to a different prediction commitment", False)
except RuntimeError:
    check("external assay cannot be joined to a different prediction commitment", True)
tampered_assay = list(assay_rows)
tampered_assay[0] = ExternalAssayRow(
    **{**tampered_assay[0].__dict__,
       "predicted_effect": tampered_assay[0].predicted_effect + .25})
try:
    external_assay_verdict(
        tampered_assay, expected_prediction_sha256=prediction_hash,
        expected_variant_set_sha256=variant_set_hash,
        expected_variant_cohort_sha256=variant_cohort_hash, labels_opened=True,
        resolvable_z=1.96, sign_concordance_min=.8, error_gain_min=.1,
        specificity_gap_min=.2, min_clusters=5, n_boot=20,
    )
    check("sealed external prediction values cannot be changed after labels open", False)
except RuntimeError:
    check("sealed external prediction values cannot be changed after labels open", True)
try:
    external_causal_triangulation(
        av, {**vv, "variant_cohort_sha256": "0" * 64},
        expected_causal_evidence_sha256=vv["causal_evidence_sha256"],
        expected_prediction_sha256=prediction_hash,
        expected_variant_set_sha256=variant_set_hash,
        expected_variant_cohort_sha256=variant_cohort_hash)
    check("triangulation rejects a different variant/cohort commitment", False)
except RuntimeError:
    check("triangulation rejects a different variant/cohort commitment", True)


# Mechanism fingerprint predicts a held-out family with the exact fixed axes.
fps = []
for fam_i, fam in enumerate(("motif", "variant", "long", "calibration")):
    for j in range(5):
        x = np.array([
            fam_i + .1*j, j*.2, fam_i*.3 + j*.01, j*.1,
            .2*fam_i,
            .1 + .05*j, .7 + .01*fam_i, .8 + .01*j,
            .3*fam_i + .02*j, .01*(fam_i + j),
        ])
        y = float(2.0 * x[0] - .5 * x[1])
        fps.append(MechanismFingerprint.create(
            task_id=f"{fam}-{j}", task_family=fam,
            **dict(zip((
                "b28_content_write_magnitude", "b28_direction_stability",
                "b29_parallel_perp_ratio", "b29_gain", "scale_q",
                "scale_plateau_position", "b30_mediation_fraction",
                "b30_predicted_rescue_fraction", "x31_carrier_sensitivity",
                "offpath_leakage"), x)),
            target=y, source_split="chr22-development",
            source_role="development", source_artifact_sha256="b" * 64,
            features_frozen_before_labels=True))
fingerprint_hash = seal_mechanism_fingerprints(fps)
fp = leave_family_out_fingerprint(
    fps, ridge=1e-6, labels_opened=True,
    expected_feature_sha256=fingerprint_hash)
check("task fingerprint uses genuine leave-family-out prediction", fp["r2"] > .99)
try:
    leave_family_out_fingerprint(
        [replace(fps[0], b29_gain=fps[0].b29_gain + 1.), *fps[1:]],
        ridge=1e-6, labels_opened=True,
        expected_feature_sha256=fingerprint_hash)
    check("post-label fingerprint feature changes are rejected", False)
except RuntimeError:
    check("post-label fingerprint feature changes are rejected", True)


sig = DiagnosticSignals("case", .1, .2, .9, .0)
check("W/A/R/K diagnosis is selected before repair", diagnose_stage(
      sig, minimum_deficit=.3, uniqueness_gap=.2) == "R")
diagnosis = seal_diagnosis_table(
    {f"case{i}": "R" for i in range(8)}, source_split="chr17-pre-repair",
    source_sha256="3" * 64,
    diagnosis_config_sha256="c" * 64)
repairs = [RepairOutcome(
    f"case{i}", "R", stage, .95 if stage == "R" else .1, .001,
    diagnosis.sha256, "chr17-repair", True, (f"case:{i}",))
    for i in range(8) for stage in ("W", "A", "R", "K")]
check("blind crossover accepts only stage-specific repair",
      debugger_verdict(
          repairs, diagnosis=diagnosis, expected_validation_split="chr17-repair",
          recovery_min=.8, specificity_gap=.5, off_target_max=.01,
          min_clusters=5, n_boot=200)["all_passed"])
try:
    debugger_verdict(
        [replace(repairs[0], predicted_stage="W"), *repairs[1:]],
        diagnosis=diagnosis, expected_validation_split="chr17-repair",
        recovery_min=.8, specificity_gap=.5, off_target_max=.01,
        min_clusters=5, n_boot=100)
    check("post-outcome diagnosis changes are rejected", False)
except RuntimeError:
    check("post-outcome diagnosis changes are rejected", True)


dev_dose_rows = []
for i in range(6):
    for d in (-1., 0., 1.):
        dev_dose_rows.append(CarrierDoseRow(
            f"dev{i}", "chr22-development", "d" * 64, d, d, .001, False,
            (d-1.)**2, abs(d-1.) / 2, (f"dev:{i}",)))
selection = select_carrier_dose_development(
    dev_dose_rows, d_shape_max=.01, development_split="chr22-development",
    expected_source_sha256="d" * 64)
locked_dose_rows = []
for i in range(8):
    for d in (-1., 0., 1.):
        locked_dose_rows.append(CarrierDoseRow(
            f"locked{i}", "chr17-locked", "e" * 64, d, d, .001, False,
            (d-1.)**2, abs(d-1.) / 2, (f"locked:{i}",)))
cal = carrier_calibration_verdict(
    locked_dose_rows, selection=selection,
    expected_locked_split="chr17-locked",
    expected_locked_source_sha256="e" * 64,
    minimum_beta_span=1.5, monotonic_fraction_min=.95,
    rank_change_equivalence=.01, min_clusters=5, n_boot=200)
check("carrier dose is selected on development and validated as content preserving",
      selection.selected_dose == 1 and all(cal["checks"].values()))
try:
    carrier_calibration_verdict(
        [replace(row, source_sha256="d" * 64) for row in locked_dose_rows],
        selection=selection, expected_locked_split="chr17-locked",
        expected_locked_source_sha256="d" * 64,
        minimum_beta_span=1.5, monotonic_fraction_min=.95,
        rank_change_equivalence=.01, min_clusters=5, n_boot=100)
    check("carrier locked validation cannot reuse development source", False)
except RuntimeError:
    check("carrier locked validation cannot reuse development source", True)


paths = [PathUseRow(f"l{i}", 100, "local", .1, .1, .2,
                    (f"local:{i}",)) for i in range(6)]
paths += [PathUseRow(f"d{i}", 20000, "long", 1.0, .2, 1.2,
                     (f"distant:{i}",)) for i in range(6)]
pv = long_context_path_verdict(paths, local_max_bp=500, long_min_bp=10000,
                               local_equivalence=.01, long_superiority=.5,
                               joint_noninferiority=.01,
                               min_clusters=5, n_boot=200)
check("long HCL path use is distance-specific", all(pv["checks"].values()))


comp = []
for method, loss in (("mechanism", .01), ("random", .3), ("magnitude", .2)):
    comp += [CompressionRow(str(i), method, .2, loss, loss, loss,
                            (f"compression:{i}",)) for i in range(6)]
cv = compression_comparison(comp, min_clusters=5, n_boot=200)
check("compression comparison enforces matched compute baselines",
      cv["mechanism_best_task_retention"] and cv["mechanism_best_shape_retention"])
try:
    compression_comparison(comp[:-1], min_clusters=5, n_boot=100)
    check("compression baselines must use the identical unit panel", False)
except RuntimeError:
    check("compression baselines must use the identical unit panel", True)

compression_evidence = {
    "b28_noncausal_channels_equivalent": True,
    "b29_noncausal_channels_equivalent": True,
    "b30_causal_subspace_sufficient": True,
    "norm_canonicalization_equivalent": True,
    "hcl_router_generalizes": True,
    "drop:g30": True,
    "drop:b31": True,
}
plan = build_mechanism_compression_plan(
    evidence=compression_evidence, dropped_paths=("g30", "b31"),
    kept_b28_channels=(0, 2), kept_b29_channels=(1, 3),
    b30_residual_rank=1, canonical_branch_norm=2., hcl_gate_threshold=.7,
)
check("compression sequence is only built from passed causal gates",
      plan.dropped_paths == ("g30", "b31") and len(plan.evidence_sha256) == 64)
try:
    build_mechanism_compression_plan(
        evidence={**compression_evidence, "drop:b31": False},
        dropped_paths=("g30", "b31"), kept_b28_channels=(0,),
        kept_b29_channels=(0,), b30_residual_rank=0,
        canonical_branch_norm=1., hcl_gate_threshold=.5)
    check("an unverified path cannot enter a compression plan", False)
except RuntimeError:
    check("an unverified path cannot enter a compression plan", True)

torch.manual_seed(4)
z = torch.randn(5, 3)
W1, W2, W3 = torch.randn(4, 3), torch.randn(4, 3), torch.randn(3, 4)
reduced = reduce_bilinear_channels(W1, W2, W3, (0, 2))
expected = ((z @ W1[[0, 2]].T) * (z @ W2[[0, 2]].T)) @ W3[:, [0, 2]].T
actual = ((z @ reduced.W1.T) * (z @ reduced.W2.T)) @ reduced.W3.T
check("bilinear pruning exactly retains the selected native channels",
      torch.allclose(actual, expected))

weight = torch.randn(5, 4, dtype=torch.float64)
basis, _ = torch.linalg.qr(torch.randn(5, 2, dtype=torch.float64))
lowrank = compress_weight_preserve_output_subspace(weight, basis, residual_rank=1)
check("b30 low-rank compression preserves the frozen causal output subspace",
      torch.allclose(basis.T @ lowrank.weight, basis.T @ weight,
                     atol=1e-10, rtol=1e-10))
u = torch.randn(2, 3)
canon = canonicalize_update_norm(u, 2.)
gate = torch.tensor([True, False])
gated = conditional_path_output(canon, gate)
check("norm canonicalization preserves direction and frozen HCL gate is exact",
      torch.allclose(canon.norm(dim=-1), torch.full((2,), 2.))
      and torch.equal(gated[0], canon[0]) and torch.count_nonzero(gated[1]) == 0)


def score_batch(seqs):
    return [SteeringScore(s, s.count("G"), s.count("G") * .2, 0.) for s in seqs]
best = beam_search_sequence("AAAA", score_batch=score_batch,
                            editable_positions=range(4), max_mutations=2, beam_width=4)[0]
steering_selection = seal_steering_selection(
    sequence=best.sequence, development_split="chr22-development",
    development_unit_ids=tuple(f"search{i}" for i in range(8)),
    development_source_sha256="1" * 64,
    search_config_sha256="f" * 64)
steering_rows = [SteeringValidation(
    f"steer{i}", best.sequence, "chr17-locked", "2" * 64,
    steering_selection.sha256, True,
    1., 1., 0., .9, 0., .01, .01, .001, (f"steer:{i}",))
    for i in range(10)]
sv = steering_verdict(
    steering_rows, effect_min=.5, null_max=.05, rescue_min=.8,
    off_target_max=.01, selection=steering_selection,
    expected_validation_split="chr17-locked",
    expected_validation_source_sha256="2" * 64,
    min_clusters=5, n_boot=200)
check("mechanism-guided sequence search has a causal validation gate",
      best.sequence.count("G") == 2 and sv["claim"] == "mechanism-guided sequence")
try:
    steering_verdict(
        [replace(steering_rows[0], held_out=False), *steering_rows[1:]],
        effect_min=.5, null_max=.05, rescue_min=.8, off_target_max=.01,
        selection=steering_selection,
        expected_validation_split="chr17-locked",
        expected_validation_source_sha256="2" * 64,
        min_clusters=5, n_boot=100)
    check("steering winner cannot validate on its search panel", False)
except RuntimeError:
    check("steering winner cannot validate on its search panel", True)
try:
    steering_verdict(
        [replace(row, sequence="AAAA") for row in steering_rows],
        effect_min=.5, null_max=.05, rescue_min=.8, off_target_max=.01,
        selection=steering_selection,
        expected_validation_split="chr17-locked",
        expected_validation_source_sha256="2" * 64,
        min_clusters=5, n_boot=100)
    check("locked steering rows cannot swap the sealed winning sequence", False)
except RuntimeError:
    check("locked steering rows cannot swap the sealed winning sequence", True)


# Same dependency-group sample is used for all label-efficiency baselines.
rng = np.random.default_rng(1)
n = 120
chrom = np.array(["chr22"] * 80 + ["chr17"] * 40)
latent = rng.normal(size=n)
labels = (latent > 0).astype(float)
mech = np.column_stack([latent, rng.normal(size=(n, 3))])
raw = rng.normal(size=(n, 4))
table = FrozenFeatureTable(
    tuple(f"u{i}" for i in range(n)), tuple(chrom), tuple(f"g{i}" for i in range(n)),
    tuple(labels), {"mechanism": mech, "raw": raw},
    {"mechanism": ("g28", "b29", "m30", "carrier"),
     "raw": ("r0", "r1", "r2", "r3")})
curve = label_efficiency_benchmark(
    table, train_chromosomes=["chr22"], test_chromosome="chr17",
    sizes=(0, 20, "full"), task="binary", ridge=.1,
    zero_shot_weights={"mechanism": [1, 0, 0, 0], "raw": [1, 0, 0, 0]},
    repeats=3)
check("few-shot curve includes zero-shot, few-shot and full-supervised points",
      curve["sizes"]["0"]["mechanism"]["mean"] > .99
      and curve["sizes"]["full"]["mechanism"]["mean"] > .95)

leaked_groups = list(table.groups)
leaked_groups[-1] = leaked_groups[0]
leaked_table = FrozenFeatureTable(
    table.unit_ids, table.chromosomes, tuple(leaked_groups), table.labels,
    table.matrices, table.feature_names)
try:
    label_efficiency_benchmark(
        leaked_table, train_chromosomes=["chr22"], test_chromosome="chr17",
        sizes=("full",), task="binary", ridge=.1)
    check("dependency groups cannot cross the chromosome benchmark split", False)
except RuntimeError:
    check("dependency groups cannot cross the chromosome benchmark split", True)


A = torch.randn(30, 5, dtype=torch.float64)
VA = torch.randn(20, 5, dtype=torch.float64)
Q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
fit_ids = tuple(f"fit-anchor:{i}" for i in range(len(A)))
validation_ids = tuple(f"validation-anchor:{i}" for i in range(len(VA)))
fit_anchor_sha = seal_alignment_anchor_pair(fit_ids, A, A @ Q)
validation_anchor_sha = seal_alignment_anchor_pair(validation_ids, VA, VA @ Q)
align = orthogonal_checkpoint_alignment(
    A, A @ Q, fit_anchor_ids=fit_ids,
    expected_fit_anchor_sha256=fit_anchor_sha,
    validation_source_anchors=VA, validation_target_anchors=VA @ Q,
    validation_anchor_ids=validation_ids,
    validation_dependency_keys=tuple((f"validation-anchor:{i}",)
                                     for i in range(len(VA))),
    expected_validation_anchor_sha256=validation_anchor_sha,
    min_validation_clusters=5, n_boot=200)
cross_tmp = tempfile.TemporaryDirectory()
cross_store = ArtifactStore(Path(cross_tmp.name) / "cross-checkpoint")
generator = torch.Generator().manual_seed(771)
train_inputs = torch.randn(30, 3, dtype=torch.float64, generator=generator)
development_inputs = torch.randn(12, 3, dtype=torch.float64, generator=generator)
locked_inputs = torch.randn(10, 3, dtype=torch.float64, generator=generator)
true_coefficients = torch.randn(4, 5, dtype=torch.float64, generator=generator)

def make_delta(inputs):
    return torch.cat((torch.ones((len(inputs), 1), dtype=torch.float64), inputs),
                     dim=1) @ true_coefficients

cross_model = fit_cross_checkpoint_linear_model(
    train_inputs, make_delta(train_inputs),
    development_inputs, make_delta(development_inputs),
    train_unit_ids=tuple(f"train:{i}" for i in range(len(train_inputs))),
    development_unit_ids=tuple(
        f"development:{i}" for i in range(len(development_inputs))),
    train_split="chr22-train", development_split="chr21-development",
    artifact_store=cross_store, artifact_name="delta_predictor", ridge=1e-8)
cross_prediction = seal_cross_checkpoint_prediction(
    model=cross_model, locked_inputs=locked_inputs,
    locked_unit_ids=tuple(str(i) for i in range(len(locked_inputs))),
    locked_split="chr17-locked",
    held_out=True, target_derived=False)
delta_hashes = dict(cross_prediction.locked_prediction_hashes)
cross = [CrossCheckpointRow(
    str(i), .9, .01, .01, .01, .9, cross_prediction.sha256,
    delta_hashes[str(i)], delta_hashes[str(i)], (f"pair:{i}",))
    for i in range(10)]
xv = cross_checkpoint_verdict(cross, transfer_min=.8, control_max=.05,
                              rescue_min=.8, minimum_anchor_cosine=.99,
                              alignment=align, prediction=cross_prediction,
                              locked_model_inputs=locked_inputs,
                              min_clusters=5, n_boot=200)
check("cross-checkpoint direction needs alignment, specificity, block and rescue",
      xv["claim"] == "cross-checkpoint mechanism transfer"
      and xv["validation_alignment"]["ci_lo"] > .99)
wide_null = [CrossCheckpointRow(
    str(i), .9, .04 if i < 5 else .06, .01, .01, .9,
    cross_prediction.sha256, delta_hashes[str(i)], delta_hashes[str(i)],
    (f"wide-pair:{i}",)) for i in range(10)]
wide_xv = cross_checkpoint_verdict(
    wide_null, transfer_min=.8, control_max=.05, rescue_min=.8,
    minimum_anchor_cosine=.99, alignment=align, prediction=cross_prediction,
    locked_model_inputs=locked_inputs,
    min_clusters=5, n_boot=500)
check("a near-zero mean is insufficient when the control CI is not equivalent",
      not wide_xv["checks"]["wrong_layer_equivalent"])
mutated_locked_inputs = locked_inputs.clone()
mutated_locked_inputs[0, 0] += 1.
try:
    cross_checkpoint_verdict(
        cross, transfer_min=.8, control_max=.05, rescue_min=.8,
        minimum_anchor_cosine=.99, alignment=align,
        prediction=cross_prediction,
        locked_model_inputs=mutated_locked_inputs,
        min_clusters=5, n_boot=200)
    check("locked model-input mutation is rejected before transfer scoring", False)
except RuntimeError:
    check("locked model-input mutation is rejected before transfer scoring", True)
forged_hash_rows = [replace(
    row, predicted_delta_sha256="f" * 64,
    regenerated_delta_sha256="f" * 64) for row in cross]
try:
    cross_checkpoint_verdict(
        forged_hash_rows, transfer_min=.8, control_max=.05, rescue_min=.8,
        minimum_anchor_cosine=.99, alignment=align,
        prediction=cross_prediction, locked_model_inputs=locked_inputs,
        min_clusters=5, n_boot=200)
    check("matching caller hashes cannot substitute for internal replay", False)
except RuntimeError:
    check("matching caller hashes cannot substitute for internal replay", True)
check("sealed fitted model exactly regenerates the locked prediction width",
      replay_cross_checkpoint_model(cross_model, locked_inputs).shape == (10, 5))

mapped = transfer_direction(
    align.matrix[:, 0], align, minimum_retained_norm_fraction=.9)
check("direction transfer reports retained norm and projection loss",
      mapped.retained_norm_fraction > .99 and mapped.projection_loss_fraction < .01)

rect_target, _ = torch.linalg.qr(torch.randn(5, 3, dtype=torch.float64))
rect_fit_target = A @ rect_target
rect_validation_target = VA @ rect_target
rect_fit_sha = seal_alignment_anchor_pair(fit_ids, A, rect_fit_target)
rect_validation_sha = seal_alignment_anchor_pair(
    validation_ids, VA, rect_validation_target)
rect = orthogonal_checkpoint_alignment(
    A, rect_fit_target, fit_anchor_ids=fit_ids,
    expected_fit_anchor_sha256=rect_fit_sha,
    validation_source_anchors=VA,
    validation_target_anchors=rect_validation_target,
    validation_anchor_ids=validation_ids,
    validation_dependency_keys=tuple((f"validation-anchor:{i}",)
                                     for i in range(len(VA))),
    expected_validation_anchor_sha256=rect_validation_sha,
    min_validation_clusters=5, n_boot=200)
_, _, vh = torch.linalg.svd(rect.matrix.T, full_matrices=True)
null_direction = vh[-1]
try:
    transfer_direction(null_direction, rect, minimum_retained_norm_fraction=.1)
    check("rectangular transfer rejects a near-null projected direction", False)
except RuntimeError:
    check("rectangular transfer rejects a near-null projected direction", True)


pred = BlindRolePrediction.create(
    "20b", 28, 30, 7, True, source_split="20b-prediction",
    source_artifact_sha256="5" * 64,
    source_unit_ids=tuple(f"role-train:{i}" for i in range(10)),
    prediction_config_sha256="6" * 64)
obs = [BlockRoleObservation(
    unit_id=f"role-test:{unit}", checkpoint="20b",
    source_split="20b-prospective", source_artifact_sha256="7" * 64,
    prediction_sha256=pred.prediction_sha256, held_out=True, block=block,
    writer_effect=1. if block == 28 else 0.,
    reencoder_effect=1. if block == 30 else 0.,
    observed_carrier_coordinate=7, observed_direction_plateau=True,
    dependency_keys=(f"role-test:{unit}",))
       for unit in range(8) for block in range(25, 33)]
bv = score_blind_role_prediction(
    pred, obs, expected_observation_split="20b-prospective",
    expected_observation_source_sha256="7" * 64,
    minimum_role_gap=.5, hit_rate_min=.8, min_clusters=5, n_boot=200)
check("prospective checkpoint role prediction is hash sealed",
      bv["all_hit"])
wrong_carrier = [replace(row, observed_carrier_coordinate=8) for row in obs]
check("blind role score includes the sealed carrier and plateau fields",
      not score_blind_role_prediction(
          pred, wrong_carrier, expected_observation_split="20b-prospective",
          expected_observation_source_sha256="7" * 64,
          minimum_role_gap=.5, hit_rate_min=.8,
          min_clusters=5, n_boot=200)["all_hit"])


snaps = [TrainingSnapshot(
    i, float(i >= 1), float(i >= 2), float(i >= 3),
    float(i >= 4), float(i >= 5), unit_id=f"trajectory:{unit}",
    seed_id=f"seed:{unit}", dependency_keys=(f"trajectory:{unit}",))
    for unit in range(10) for i in range(8)]
tv = training_dynamics_verdict(snaps, thresholds={
    "writer_effect": .5, "b29_specialization": .5, "reencoder_effect": .5,
    "carrier_concentration": .5, "benchmark_score": .5},
    sustain_checkpoints=3, identified_fraction_min=.8,
    ordered_fraction_min=.8, crossing_equivalence_steps=0,
    min_clusters=5, n_boot=300)
check("training dynamics tests the predeclared acquisition order",
      tv["claim"] == "ordered mechanism acquisition"
      and tv["checks"]["adjacent_reversals_equivalent"])
transient = [replace(
    row, writer_effect=float(row.training_step == 1)) for row in snaps]
transient_verdict = training_dynamics_verdict(transient, thresholds={
    "writer_effect": .5, "b29_specialization": .5, "reencoder_effect": .5,
    "carrier_concentration": .5, "benchmark_score": .5},
    sustain_checkpoints=3, identified_fraction_min=.8,
    ordered_fraction_min=.8, crossing_equivalence_steps=0,
    min_clusters=5, n_boot=300)
check("a noisy transient threshold crossing cannot establish acquisition",
      transient_verdict["claim"] != "ordered mechanism acquisition"
      and all(crossings["writer_effect"] is None
              for crossings in transient_verdict["crossings_by_unit"].values()))
try:
    training_dynamics_verdict(snaps[:-1], thresholds={
        "writer_effect": .5, "b29_specialization": .5,
        "reencoder_effect": .5, "carrier_concentration": .5,
        "benchmark_score": .5}, sustain_checkpoints=3,
        identified_fraction_min=.8, ordered_fraction_min=.8,
        crossing_equivalence_steps=0, min_clusters=5, n_boot=200)
    check("training dynamics rejects an incomplete unit x checkpoint grid", False)
except RuntimeError:
    check("training dynamics rejects an incomplete unit x checkpoint grid", True)


replication_plan = seal_replication_plan(
    minimum_effect={"writer": .5, "reencoder": .5},
    expected_sign={"writer": 1, "reencoder": 1},
    required_domains=("chr22", "chr17"),
    source_sha256_by_domain={"chr22": "8" * 64, "chr17": "9" * 64},
    analysis_config_sha256="a" * 64)
cells = [ReplicationCell(
    m, d, f"{d}-unit{i}", 1., "8" * 64 if d == "chr22" else "9" * 64,
    replication_plan.sha256, (f"{d}-unit{i}",))
         for m in ("writer", "reencoder") for d in ("chr22", "chr17")
         for i in range(8)]
rv = robustness_conjunction(
    cells, plan=replication_plan, min_clusters=5, n_boot=200)
check("robustness is a conjunction across predeclared domains", rv["all_passed"])
try:
    robustness_conjunction(
        [*cells, cells[0]], plan=replication_plan,
        min_clusters=5, n_boot=100)
    check("robustness grid rejects duplicate cells", False)
except ValueError:
    check("robustness grid rejects duplicate cells", True)
try:
    robustness_conjunction(
        [row for row in cells
         if not (row.mechanism == "reencoder" and row.domain == "chr17")],
        plan=replication_plan, min_clusters=5, n_boot=100)
    check("robustness grid rejects a missing mechanism-domain cell", False)
except ValueError:
    check("robustness grid rejects a missing mechanism-domain cell", True)


print()
print(f"{sum(OK)}/{len(OK)} passed")
if not all(OK):
    raise AssertionError("one or more application/generalization checks failed")
if __name__ == "__main__":
    sys.exit(0 if all(OK) else 1)

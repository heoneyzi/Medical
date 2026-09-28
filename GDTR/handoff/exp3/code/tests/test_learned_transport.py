"""Model-free tests for Step 15 learned delta transport."""
from __future__ import annotations

import hashlib
import sys

import numpy as np

sys.path.insert(0, "..")

from exp3.learned_transport import (
    InterventionBatch,
    LagBatch,
    NativeFactorBatch,
    PredictedDelta,
    RescueControlEndpoint,
    RescueMargins,
    RescueRuntimeContract,
    SplitRecord,
    TransportProvenance,
    delta_r2,
    evaluate_locked_hcl,
    evaluate_locked_transport,
    fit_hcl_impulse_kernel,
    fit_native_factor_ladder,
    fit_randomized_transport,
    fit_sparse_delta_transcoder,
    intervention_source_sha256,
    intervention_target_sha256,
    lag_source_sha256,
    lag_target_sha256,
    native_factor_source_sha256,
    native_factor_target_sha256,
    native_factor_verdict,
    predict_locked_delta,
    predicted_delta_rescue_verdict,
    radial_tangential,
    randomized_delta_design,
    seal_rescue_record,
    spherical_log_map,
)


OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name + (f"   {detail}" if detail else ""))


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def mm(a, b):
    return np.einsum("...i,ij->...j", a, b, optimize=True)


rng = np.random.default_rng(123)
train_ids = tuple(f"tr{i}" for i in range(96))
dev_ids = tuple(f"dv{i}" for i in range(48))
lock_ids = tuple(f"lk{i}" for i in range(48))

try:
    TransportProvenance(
        SplitRecord("train", "train", ("same",), digest("a"), digest("ta")),
        SplitRecord("dev", "dev", ("same",), digest("b"), digest("tb")),
        SplitRecord("locked", "locked", ("c",), digest("c"), digest("tc")),
        "delta", "delta")
    check("split overlap is rejected", False)
except ValueError:
    check("split overlap is rejected", True)

try:
    TransportProvenance(
        SplitRecord("train", "train", ("a",), digest("a"), digest("ta")),
        SplitRecord("dev", "dev", ("b",), digest("b"), digest("tb")),
        SplitRecord("locked", "locked", ("c",), digest("c"), digest("tc")),
        "absolute_state reconstruction", "delta")
    check("absolute-state target/features are rejected", False)
except ValueError:
    check("absolute-state target/features are rejected", True)

def intervention(role, ids, A):
    n, d = len(ids), A.shape[0]
    base = rng.normal(size=(n, d))
    delta = randomized_delta_design(base, radial_sd=0.35, tangential_sd=0.45,
                                    seed={"train": 1, "dev": 2, "locked": 3}[role])
    out = mm(delta, A)
    return InterventionBatch(ids, role, base, delta, out,
                             randomization_seed={"train": 1, "dev": 2, "locked": 3}[role])


d, dout, rank = 6, 5, 2
A = mm(rng.normal(size=(d, rank)), rng.normal(size=(rank, dout)))
train = intervention("train", train_ids, A)
dev = intervention("dev", dev_ids, A)
locked = intervention("locked", lock_ids, A)


def intervention_record(name, batch):
    return SplitRecord(
        name, batch.role, batch.unit_ids,
        intervention_source_sha256(batch.unit_ids, batch.base, batch.delta_in),
        intervention_target_sha256(batch.delta_out, batch.output_base),
    )


prov = TransportProvenance(
    intervention_record("train", train),
    intervention_record("dev", dev),
    intervention_record("locked", locked),
    feature_spec="randomized radial/tangential intervention deltas",
    target_spec="held-out downstream intervention delta",
)

try:
    prov.assert_units("locked", tuple(reversed(lock_ids)))
    check("provenance rejects a row permutation with the same unit set", False)
except ValueError:
    check("provenance rejects a row permutation with the same unit set", True)

rt = radial_tangential(train.base, train.delta_in)
identity_err = np.max(np.abs(rt.radial + rt.tangential - train.delta_in))
orth_err = np.max(np.abs(np.sum(rt.tangential * train.base, axis=1)))
check("radial+tangential exactly reconstruct perturbation",
      identity_err < 1e-10 and orth_err < 1e-10,
      f"identity={identity_err:.2e}, orth={orth_err:.2e}")

transition = spherical_log_map(train.base[:10], train.base[:10] + train.delta_in[:10])
unit = train.base[:10] / np.linalg.norm(train.base[:10], axis=1, keepdims=True)
sphere_orth = np.nanmax(np.abs(np.sum(transition.tangent_log * unit, axis=1)))
angle_err = np.nanmax(np.abs(np.linalg.norm(transition.tangent_log, axis=1)
                             - transition.angle))
check("spherical log-map is tangent and has geodesic-angle norm",
      sphere_orth < 1e-10 and angle_err < 1e-10,
      f"orth={sphere_orth:.2e}, angle={angle_err:.2e}")

system = fit_randomized_transport(train, dev, prov, ranks=(1, 2, 3),
                                  ridges=(1e-8, 1e-4))
held = evaluate_locked_transport(system, locked, prov)
check("reduced-rank randomized system identifies locked delta transport",
      held.r2 > 0.999 and held.cosine_mean > 0.999,
      f"R2={held.r2:.6f}, cos={held.cosine_mean:.6f}")


def aligned_ray_batch(role, ids, seed):
    local = np.random.default_rng(seed)
    base = np.zeros((len(ids), d)); base[:, 0] = 1.0
    delta = local.normal(size=(len(ids), d))
    return InterventionBatch(ids, role, base, delta, mm(delta, A), seed)


aligned_train = aligned_ray_batch("train", train_ids, 81)
aligned_dev = aligned_ray_batch("dev", dev_ids, 82)
aligned_prov = TransportProvenance(
    intervention_record("train-aligned", aligned_train),
    intervention_record("dev-aligned", aligned_dev),
    intervention_record("locked", locked), "radial/tangential delta", "delta")
mixed_dev_prov = TransportProvenance(
    intervention_record("train", train),
    intervention_record("dev-aligned", aligned_dev),
    intervention_record("locked", locked), "radial/tangential delta", "delta")
check("rank-deficiency fixture has full-rank raw perturbations",
      np.linalg.matrix_rank(aligned_train.delta_in) == d)
try:
    fit_randomized_transport(aligned_train, aligned_dev, aligned_prov,
                             ranks=(1, 2), ridges=(1e-6,))
    check("radial/tangential rank deficiency is rejected despite full raw rank", False)
except ValueError:
    check("radial/tangential rank deficiency is rejected despite full raw rank", True)
try:
    fit_randomized_transport(train, aligned_dev, mixed_dev_prov,
                             ranks=(1, 2), ridges=(1e-6,))
    check("component identification requires independent dev support", False)
except ValueError:
    check("component identification requires independent dev support", True)


def native_batch(role, ids):
    n = len(ids)
    inc = rng.normal(size=(n, 4))
    mix = rng.normal(size=(n, 3))
    fa = rng.normal(size=(n, 2)); fb = rng.normal(size=(n, 2))
    product = rng.normal(size=(n, 3))
    y = mm(inc, W_inc) + mm(mix, W_mix) + mm(product, W_prod)
    return NativeFactorBatch(ids, role, inc, mix, fa, fb, product, y)


W_inc = rng.normal(size=(4, 4)) * 0.3
W_mix = rng.normal(size=(3, 4)) * 0.8
W_prod = rng.normal(size=(3, 4)) * 1.0
ntrain = native_batch("train", train_ids)
ndev = native_batch("dev", dev_ids)
nlocked = native_batch("locked", lock_ids)


def native_record(name, batch):
    return SplitRecord(
        name, batch.role, batch.unit_ids,
        native_factor_source_sha256(
            batch.unit_ids, batch.incoming_delta, batch.mixer_delta,
            batch.factor_a_delta, batch.factor_b_delta, batch.product_delta),
        native_factor_target_sha256(batch.outgoing_delta),
    )


native_prov = TransportProvenance(
    native_record("native-train", ntrain), native_record("native-dev", ndev),
    native_record("native-locked", nlocked),
    "native factor delta ladder", "outgoing native delta")
ladder = fit_native_factor_ladder(ntrain, ndev, native_prov, ranks=(1, 2, 3, 4),
                                  ridges=(1e-8, 1e-4))
ls = ladder.locked_scores(nlocked, native_prov)
lv = native_factor_verdict(
    ladder, nlocked, native_prov, amplifier_margin=0.05,
    cowriter_margin=0.05, m4_min_r2=0.95)
check("M0-M4 separates b29 amplifier and product co-writer gains",
      ls["amplifier_increment_M2_minus_M1"] > 0.05
      and ls["product_cowriter_increment_M4_minus_M3"] > 0.05
      and ls["r2"]["M4"] > 0.999 and all(lv["checks"].values()),
      str(ls))


# A factor-only construction is the regression test for the M4-M2 bug: M4-M2
# is large because it includes factor main effects, while the registered
# product-specific increment M4-M3 is null.
W_factor_a = rng.normal(size=(2, 4)) * 1.2
W_factor_b = rng.normal(size=(2, 4)) * 1.2


def factor_only_batch(role, ids, seed):
    local = np.random.default_rng(seed)
    n = len(ids)
    inc = local.normal(size=(n, 4))
    mix = local.normal(size=(n, 3))
    fa = local.normal(size=(n, 2)); fb = local.normal(size=(n, 2))
    product = local.normal(size=(n, 3))
    y = mm(inc, W_inc) + mm(mix, W_mix) \
        + mm(fa, W_factor_a) + mm(fb, W_factor_b)
    return NativeFactorBatch(ids, role, inc, mix, fa, fb, product, y)


factor_train = factor_only_batch("train", train_ids, 31)
factor_dev = factor_only_batch("dev", dev_ids, 32)
factor_locked = factor_only_batch("locked", lock_ids, 33)
factor_prov = TransportProvenance(
    native_record("factor-train", factor_train),
    native_record("factor-dev", factor_dev),
    native_record("factor-locked", factor_locked),
    "factor-only native ladder", "factor-only outgoing delta")
factor_ladder = fit_native_factor_ladder(
    factor_train, factor_dev, factor_prov,
    ranks=(1, 2, 3, 4), ridges=(1e-8, 1e-4))
factor_scores = factor_ladder.locked_scores(factor_locked, factor_prov)
factor_verdict = native_factor_verdict(
    factor_ladder, factor_locked, factor_prov, amplifier_margin=.05,
    cowriter_margin=.05, m4_min_r2=.95)
check("factor main effects cannot masquerade as a product co-writer",
      factor_scores["expanded_increment_M4_minus_M2"] > .05
      and factor_scores["product_cowriter_increment_M4_minus_M3"] < .05
      and not factor_verdict["checks"]["b29_product_cowriter_increment"],
      str(factor_scores))


def lag_batch(role, ids):
    n, length, din, dout_ = len(ids), 14, 2, 3
    x = rng.normal(size=(n, length, din))
    y = mm(x, K0)
    y[:, 2:] += mm(x[:, :-2], K2)
    return LagBatch(ids, role, x, y)


K0 = rng.normal(size=(2, 3))
K2 = rng.normal(size=(2, 3)) * 0.8
ltrain, ldev, llocked = (lag_batch("train", train_ids),
                          lag_batch("dev", dev_ids),
                          lag_batch("locked", lock_ids))


def lag_record(name, batch):
    return SplitRecord(
        name, batch.role, batch.unit_ids,
        lag_source_sha256(batch.unit_ids, batch.delta_in),
        lag_target_sha256(batch.delta_out))


lag_prov = TransportProvenance(
    lag_record("lag-train", ltrain), lag_record("lag-dev", ldev),
    lag_record("lag-locked", llocked), "causal lag delta", "lag output delta")
kernel = fit_hcl_impulse_kernel(ltrain, ldev, lag_prov, max_lag=3,
                                ranks=(1, 2, 3), ridges=(1e-8, 1e-4))
hk = evaluate_locked_hcl(kernel, llocked, lag_prov)
energies = np.asarray(hk["lag_energy"])
check("one-sided HCL impulse kernel recovers causal lag 0 and lag 2",
      hk["delta_r2"] > 0.999 and energies[0] > 10 * energies[1]
      and energies[2] > 10 * energies[3],
      f"R2={hk['delta_r2']:.6f}, energies={energies.tolist()}")


# Sparse nonlinear delta map; no absolute states are provided to the fitter.
latent = 8
W1 = rng.normal(size=(d, latent))
W2 = rng.normal(size=(latent, dout))


def sparse_batch(role, ids, seed):
    local = np.random.default_rng(seed)
    n = len(ids)
    delta = local.normal(size=(n, d))
    base = local.normal(size=(n, d))
    out = mm(np.maximum(mm(delta, W1), 0), W2)
    return InterventionBatch(ids, role, base, delta, out, seed)


strain = sparse_batch("train", train_ids, 21)
sdev = sparse_batch("dev", dev_ids, 22)
slock = sparse_batch("locked", lock_ids, 23)
sparse_prov = TransportProvenance(
    intervention_record("sparse-train", strain),
    intervention_record("sparse-dev", sdev),
    intervention_record("sparse-locked", slock),
    "sparse intervention delta", "sparse downstream delta")
tampered_base = slock.base.copy(); tampered_base[0, 0] += .01
try:
    InterventionBatch(
        lock_ids, "locked", tampered_base, slock.delta_in, slock.delta_out,
        slock.randomization_seed).validate(sparse_prov)
    check("locked source commitment rejects changed base/input tensors", False)
except ValueError:
    check("locked source commitment rejects changed base/input tensors", True)
tampered_target = slock.delta_out.copy(); tampered_target[0, 0] += .01
try:
    InterventionBatch(
        lock_ids, "locked", slock.base, slock.delta_in, tampered_target,
        slock.randomization_seed).validate(sparse_prov)
    check("locked target commitment rejects changed true deltas", False)
except ValueError:
    check("locked target commitment rejects changed true deltas", True)
transcoder = fit_sparse_delta_transcoder(
    strain, sdev, sparse_prov, hidden=latent, l1=1e-5, lr=1e-2, epochs=900, seed=7)
spred = predict_locked_delta(
    transcoder, slock.delta_in, lock_ids, sparse_prov,
    source="cross_fitted_sparse_delta_transcoder", base=slock.base)
swapped_locked_input = slock.delta_in.copy()
swapped_locked_input[[0, 1]] = swapped_locked_input[[1, 0]]
try:
    predict_locked_delta(
        transcoder, swapped_locked_input, lock_ids, sparse_prov,
        source="cross_fitted_sparse_delta_transcoder", base=slock.base)
    check("prediction refuses locked input swapped under sealed split labels", False)
except ValueError:
    check("prediction refuses locked input swapped under sealed split labels", True)
locked_sparse_r2 = delta_r2(slock.delta_out, spred.values)
check("sparse delta transcoder generalises to locked interventions",
      transcoder.dev_r2 > 0.90 and locked_sparse_r2 > 0.85,
      f"dev={transcoder.dev_r2:.4f}, locked={locked_sparse_r2:.4f}")


# Model-free endpoint patch: in a linear suffix, endpoint = patched delta.
native_endpoint = slock.delta_out
ablated_endpoint = np.zeros_like(native_endpoint)
predicted_endpoint = spred.values
random_control = rng.normal(size=predicted_endpoint.shape)
random_control *= (np.linalg.norm(predicted_endpoint, axis=1, keepdims=True)
                   / np.linalg.norm(random_control, axis=1, keepdims=True).clip(1e-12))
reverse_permutation = tuple(reversed(range(len(lock_ids))))
wrong_pair_permutation = tuple(np.roll(np.arange(len(lock_ids)), 1).tolist())


def per_unit_dose_match(values):
    return values * (
        np.linalg.norm(predicted_endpoint, axis=1, keepdims=True)
        / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12))


shuffled = per_unit_dose_match(spred.values[list(reverse_permutation)].copy())
wrong_pair = per_unit_dose_match(
    spred.values[list(wrong_pair_permutation)].copy())
wrong_layer = per_unit_dose_match(random_control[::-1].copy())
controls = {
    "dose_rank_matched_random": RescueControlEndpoint(
        "dose_rank_matched_random", random_control, lock_ids,
        digest("random-control"), True, True,
        injected_delta=random_control),
    "shuffled_prediction": RescueControlEndpoint(
        "shuffled_prediction", shuffled, lock_ids,
        digest("shuffle-control"), True, True,
        injected_delta=shuffled,
        prediction_sha256=spred.prediction_sha256,
        permutation=reverse_permutation),
    "wrong_layer": RescueControlEndpoint(
        "wrong_layer", wrong_layer, lock_ids, digest("wrong-layer-control"),
        True, True, injected_delta=wrong_layer,
        source_layer="b28", target_layer="m30"),
    "wrong_pair": RescueControlEndpoint(
        "wrong_pair", wrong_pair, lock_ids, digest("wrong-pair-control"),
        True, True, injected_delta=wrong_pair,
        permutation=wrong_pair_permutation),
}
try:
    RescueControlEndpoint(
        "dose_rank_matched_random", random_control, lock_ids,
        digest("bad-dose-control"), True, True,
        injected_delta=2.0 * random_control).validate(spred, sparse_prov)
    check("dose-matched control label is checked against injected delta", False)
except ValueError:
    check("dose-matched control label is checked against injected delta", True)
dependency_keys = tuple((f"locus:{unit_id}",) for unit_id in lock_ids)
runtime_contract = RescueRuntimeContract(
    checkpoint_sha256=digest("checkpoint"),
    architecture_sha256=digest("architecture"), code_sha256=digest("code"),
    handoff_sha256=digest("handoff"), config_sha256=digest("config"),
    tap_spec_sha256=digest("tap-spec"), edit_spec_sha256=digest("edit-spec"),
    intervention_sha256=digest("intervention"),
)
rescue_record = seal_rescue_record(
    spred, sparse_prov, dependency_keys=dependency_keys,
    true_delta=slock.delta_out, native_endpoint=native_endpoint,
    ablated_endpoint=ablated_endpoint,
    predicted_rescue_endpoint=predicted_endpoint,
    control_rescue_endpoints=controls, runtime_contract=runtime_contract,
)
verdict = predicted_delta_rescue_verdict(
    spred, sparse_prov, fitted_system=transcoder, locked_model_input=slock.delta_in,
    locked_model_base=slock.base,
    true_delta=slock.delta_out,
    native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
    predicted_rescue_endpoint=predicted_endpoint,
    rescue_record=rescue_record,
    control_rescue_endpoints=controls,
    expected_runtime_contract=runtime_contract,
    margins=RescueMargins(prediction_relative_energy=0.80, recovery_fraction=0.25,
                          control_superiority=0.10, max_relative_error=0.60),
    seed=17, n_boot=1000)
check("held-out predicted delta passes rescue and every specificity control",
      verdict["predicted_delta_rescue"], str(verdict["checks"]))

wrong_runtime_contract = RescueRuntimeContract(
    checkpoint_sha256=digest("different-checkpoint"),
    architecture_sha256=runtime_contract.architecture_sha256,
    code_sha256=runtime_contract.code_sha256,
    handoff_sha256=runtime_contract.handoff_sha256,
    config_sha256=runtime_contract.config_sha256,
    tap_spec_sha256=runtime_contract.tap_spec_sha256,
    edit_spec_sha256=runtime_contract.edit_spec_sha256,
    intervention_sha256=runtime_contract.intervention_sha256,
)
try:
    predicted_delta_rescue_verdict(
        spred, sparse_prov, fitted_system=transcoder,
        locked_model_input=slock.delta_in, locked_model_base=slock.base,
        true_delta=slock.delta_out, native_endpoint=native_endpoint,
        ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_endpoint,
        rescue_record=rescue_record, control_rescue_endpoints=controls,
        expected_runtime_contract=wrong_runtime_contract,
        margins=RescueMargins(.8, .25, .1, .6), n_boot=100)
    check("rescue receipt from another checkpoint/run is rejected", False)
except ValueError:
    check("rescue receipt from another checkpoint/run is rejected", True)

pseudoreplicated_record = seal_rescue_record(
    spred, sparse_prov, dependency_keys=tuple(("same-locus",) for _ in lock_ids),
    true_delta=slock.delta_out, native_endpoint=native_endpoint,
    ablated_endpoint=ablated_endpoint,
    predicted_rescue_endpoint=predicted_endpoint,
    control_rescue_endpoints=controls, runtime_contract=runtime_contract,
)
try:
    predicted_delta_rescue_verdict(
        spred, sparse_prov, fitted_system=transcoder,
        locked_model_input=slock.delta_in, locked_model_base=slock.base,
        true_delta=slock.delta_out,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_endpoint,
        rescue_record=pseudoreplicated_record,
        control_rescue_endpoints=controls,
        expected_runtime_contract=runtime_contract,
        margins=RescueMargins(.8, .25, .1, .6), n_boot=100,
        min_clusters=2)
    check("rescue inference refuses pseudoreplicated locked rows", False)
except RuntimeError:
    check("rescue inference refuses pseudoreplicated locked rows", True)

try:
    incomplete_controls = {
        name: control for name, control in controls.items()
        if name != "wrong_layer"
    }
    incomplete_record = seal_rescue_record(
        spred, sparse_prov, dependency_keys=dependency_keys,
        true_delta=slock.delta_out, native_endpoint=native_endpoint,
        ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_endpoint,
        control_rescue_endpoints=incomplete_controls,
        runtime_contract=runtime_contract,
    )
    predicted_delta_rescue_verdict(
        spred, sparse_prov, fitted_system=transcoder,
        locked_model_input=slock.delta_in, locked_model_base=slock.base,
        true_delta=slock.delta_out,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_endpoint,
        rescue_record=incomplete_record,
        control_rescue_endpoints=incomplete_controls,
        expected_runtime_contract=runtime_contract,
        margins=RescueMargins(.8, .25, .1, .6), n_boot=100)
    check("incomplete predicted-delta control panel is rejected", False)
except ValueError:
    check("incomplete predicted-delta control panel is rejected", True)

try:
    substituted_endpoint = predicted_endpoint.copy()
    substituted_endpoint[0] += 1.0
    predicted_delta_rescue_verdict(
        spred, sparse_prov, fitted_system=transcoder,
        locked_model_input=slock.delta_in, locked_model_base=slock.base,
        true_delta=slock.delta_out,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=substituted_endpoint,
        rescue_record=rescue_record,
        control_rescue_endpoints=controls,
        expected_runtime_contract=runtime_contract,
        margins=RescueMargins(.8, .25, .1, .6), n_boot=100)
    check("sealed rescue rejects a substituted endpoint", False)
except ValueError:
    check("sealed rescue rejects a substituted endpoint", True)

forged_true_delta = PredictedDelta(
    slock.delta_out, lock_ids, "cross_fitted_sparse_delta_transcoder",
    transcoder.artifact_sha256, sparse_prov.provenance_sha256)
try:
    forged_true_delta.validate(sparse_prov)
    check("self-declared true delta cannot pose as a fitted prediction", False)
except ValueError:
    check("self-declared true delta cannot pose as a fitted prediction", True)

forged_bound = PredictedDelta(
    slock.delta_out, lock_ids, spred.source, spred.model_sha256,
    sparse_prov.provenance_sha256, model_artifact=spred.model_artifact,
    prediction_input_sha256=spred.prediction_input_sha256)
forged_bound.validate(sparse_prov)
try:
    predicted_delta_rescue_verdict(
        forged_bound, sparse_prov, fitted_system=transcoder,
        locked_model_input=slock.delta_in, locked_model_base=slock.base,
        true_delta=slock.delta_out,
        native_endpoint=native_endpoint, ablated_endpoint=ablated_endpoint,
        predicted_rescue_endpoint=predicted_endpoint,
        rescue_record=rescue_record, control_rescue_endpoints=controls,
        expected_runtime_contract=runtime_contract,
        margins=RescueMargins(.8, .25, .1, .6), n_boot=100)
    check("true delta forged under a real artifact fails internal regeneration", False)
except ValueError:
    check("true delta forged under a real artifact fails internal regeneration", True)

bad = PredictedDelta(slock.delta_out, lock_ids, "cached_actual_delta",
                     "0" * 64, sparse_prov.provenance_sha256)
try:
    bad.validate(sparse_prov)
    check("cached actual delta is forbidden as primary rescue evidence", False)
except ValueError:
    check("cached actual delta is forbidden as primary rescue evidence", True)


print()
print(f"{sum(OK)}/{len(OK)} passed")
if not all(OK):
    raise AssertionError("one or more learned-transport checks failed")
if __name__ == "__main__":
    raise SystemExit(0 if all(OK) else 1)

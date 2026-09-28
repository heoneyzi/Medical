"""Model-free regression tests for the Step 11--12 causal contracts."""
import sys
sys.path.insert(0, "..")

from pathlib import Path
import tempfile
import numpy as np
import torch

from exp3.artifacts import ArtifactStore

from exp3.mechanism import (
    B29RoleRow, DeltaPrediction, PreconditioningRow, ScaleScanRow,
    b29_parallel_plateau_gate, b29_role_verdict, freeze_subspace,
    m28_preconditioning_verdict, normalize_scale_plateau, parallel_perpendicular,
    scale_curve_collapse_gate, seal_independent_b29_event,
    seal_prediction_lineage,
    DEFAULT_LOG_SCALE_GRID,
)
from exp3.causal_use import (
    CausalCubeRow, DirectionTransferRow, b30_block_rescue_conditions,
    causal_factorization_synthesis, cube_b30_mediation,
    cube_factorial_effects, cube_factorization_verdict,
    direction_transfer_verdict, validate_b30_condition,
)
from exp3.bilinear_pairing import (
    PairingControlRow, bilinear_pairing_control, bilinear_pairing_verdict,
)


OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name
          + (f"   {detail}" if detail else ""))


# ---- Learned W1/W2 pairing rather than marginal norms -------------------
torch.manual_seed(2)
z_pair = torch.randn(6, 5, dtype=torch.float64)
w1_pair = torch.randn(9, 5, dtype=torch.float64)
w2_pair = torch.randn(9, 5, dtype=torch.float64)
w3_pair = torch.randn(5, 9, dtype=torch.float64)
permutation = torch.randperm(9)
pairing = bilinear_pairing_control(
    z_pair, w1_pair, w2_pair, w3_pair, permutation)
check("joint W1/W2/W3 relabelling is function preserving",
      pairing.joint_relative_error < 1e-12)
check("W2-only permutation preserves spectra but breaks learned pairing",
      pairing.w2_singular_value_error < 1e-12
      and float((pairing.w2_only_permuted - pairing.native).norm()) > 1e-6)

pair_rows = []
for i in range(20):
    for condition, effect in (
        ("native", 0.0),
        ("w2_only_permuted", 1.0),
        ("jointly_relabelled", 0.001),
    ):
        pair_rows.append(PairingControlRow(
            locus=f"P{i}", condition=condition, d_shape=effect,
            log_beta=0.0, branch_relative_change=float(condition != "native"),
            joint_relative_error=pairing.joint_relative_error,
            spectrum_error_max=max(
                pairing.w1_singular_value_error,
                pairing.w2_singular_value_error,
                pairing.w3_singular_value_error),
            permutation_sha256="a" * 64,
            dependency_keys=(f"locus:P{i}",),
        ))
pair_verdict = bilinear_pairing_verdict(
    pair_rows, break_effect_margin=0.5,
    identity_equivalence_margin=0.01, min_clusters=5, n_boot=200)
check("pairing claim needs broken-pair damage plus exact relabelling control",
      pair_verdict["status"] == "supported"
      and all(pair_verdict["checks"].values()), str(pair_verdict["checks"]))


# ---- Frozen U28 / compact projections ------------------------------------
torch.manual_seed(4)
B, _ = torch.linalg.qr(torch.randn(8, 2, dtype=torch.float64))
space = freeze_subspace(B, source_split="chr22-discovery")
x = torch.randn(3, 8, dtype=torch.float64)
par, perp = parallel_perpendicular(x, space.basis)
check("U28 parallel/perpendicular decomposition is exact",
      float((x - par - perp).norm()) < 1e-12)
check("U28 perpendicular component is orthogonal",
      float((perp @ space.basis).norm()) < 1e-12)


# ---- m28 preconditioner verdict ------------------------------------------
pre = []
for i in range(20):
    common = dict(
        locus=f"L{i}", log_beta=0.0, nll=None,
        g28_norm_ratio=1.0, g28_cos_native=1.0,
        g28_relative_delta=0.0, factor_a_relative_delta=0.0,
        factor_b_relative_delta=0.0, product_relative_delta=0.0,
        dependency_keys=(f"locus:L{i}",))
    pre.append(PreconditioningRow(condition="native", d_shape=0.0, **common))
    pre.append(PreconditioningRow(
        condition="m28_zero_recompute", d_shape=0.8,
        **{**common, "g28_relative_delta": 0.5,
           "product_relative_delta": 0.6}))
    pre.append(PreconditioningRow(
        condition="m28_zero_g28_clamp", d_shape=0.01,
        rescue_method="exact_clamp", **common))
    pre.append(PreconditioningRow(
        condition="m28_zero_g28_predicted_rescue", d_shape=0.02,
        rescue_method="heldout_rrr", rescue_held_out=True,
        rescue_target_derived=False, **common))
pv = m28_preconditioning_verdict(
    pre, damage_margin=0.1, rescue_margin=0.1,
    equivalence_margin=0.05, product_change_margin=0.1,
    min_clusters=5, n_boot=200)
check("m28 label needs damage + clamp + held-out rescue",
      pv["claim"] == "m28 preconditions the g28 write"
      and all(pv["checks"].values()), str(pv["checks"]))


# ---- b29 U28 parallel plateau and amplifier role -------------------------
b29 = []
grid = DEFAULT_LOG_SCALE_GRID
for i in range(20):
    loc = f"L{i}"
    base = dict(
        locus=loc, d_shape=0.0, log_beta=0.0,
        parallel_response_norm=2.0, perpendicular_response_norm=0.01,
        perpendicular_fraction=0.005, realized_parallel_gain=4.0,
        realized_perpendicular_gain=0.01, subspace_hash=space.sha256,
        dependency_keys=(f"locus:{loc}",))
    vals = {
        "upstream": 1.0,
        "upstream_block_parallel": 0.05,
        "upstream_block_perpendicular": 0.99,
        "upstream_block_all": 0.02,
        "upstream_parallel_only": 0.99,
        "native_plus_predicted_parallel": 0.99,
        "native_plus_predicted_perpendicular": 0.01,
        "native_plus_predicted_full": 0.995,
        "upstream_block_all_predicted_parallel_rescue": 0.99,
        "upstream_block_all_predicted_perpendicular_rescue": 0.02,
        "upstream_block_all_predicted_full_rescue": 0.995,
    }
    for cond, value in vals.items():
        predicted = "predicted" in cond
        b29.append(B29RoleRow(
            condition=cond, signed_upstream_following=value,
            prediction_method="heldout_rrr" if predicted else "",
            prediction_held_out=predicted,
            prediction_target_derived=False, **base))
    for cond, value in (
        ("independent_event", 1.0),
        ("independent_event_block_perpendicular", 0.99),
    ):
        b29.append(B29RoleRow(
            condition=cond, signed_upstream_following=value,
            independent_event=True, independent_event_id=f"E{i}",
            independent_source_split="chr22-development",
            independent_source_role="development",
            independent_selection_method="predeclared_motif_edit",
            independent_selection_sha256=f"sealed-{i}",
            independent_held_out=True,
            independent_target_derived=False, **base))
    for dose in grid:
        # Saturates well before the final two doses.
        value = float(1.0 - np.exp(-100.0 * dose))
        b29.append(B29RoleRow(
            condition="parallel_dose", signed_upstream_following=value,
            parallel_dose=dose, parallel_alpha_star=0.01,
            parallel_q=dose / 0.01, **base))

pg = b29_parallel_plateau_gate(
    b29, effect_margin=0.1, increment_equivalence_margin=0.05,
    min_clusters=5, n_boot=200)
check("b29 parallel dose has explicit alpha-star plateau gate",
      pg["passed"] and all(pg["checks"].values()), str(pg["checks"]))
bv = b29_role_verdict(
    b29, effect_margin=0.1, equivalence_margin=0.05,
    perpendicular_fraction_margin=0.05, independent_effect_margin=0.1,
    parallel_plateau_gate=pg,
    min_clusters=5, n_boot=200)
check("b29 amplifier requires parallel remove/only/rescue and plateau",
      bv["role"] == "b29 amplifier of the frozen b28 mode",
      bv["role"])

sealed_event = seal_independent_b29_event(
    torch.tensor([1, 2, 4], dtype=torch.long), event_id="motif-edit-1",
    source_split="chr22-development", source_role="development",
    selection_method="predeclared_motif_edit",
    dependency_keys=("motif-family:1",), held_out=True)
sealed_event.validate(torch.tensor([1, 2, 3]), locked=True)
try:
    object.__setattr__(sealed_event, "selection_method", "posthoc")
    sealed_event.validate(torch.tensor([1, 2, 3]), locked=True)
    check("independent b29 event mutation is rejected", False)
except RuntimeError:
    check("independent b29 event mutation is rejected", True)


# ---- Normalised scale geometry -------------------------------------------
r = torch.tensor([1.0, 0.0], dtype=torch.float64)
g = torch.tensor([0.0, 2.0], dtype=torch.float64)
points = normalize_scale_plateau(r, g, [1e-5, 1e-3, 1.0])
check("scale normalization uses q=alpha||g||/||host||",
      abs(points[-1].q - 2.0) < 1e-12)
check("scale geometry approaches the update direction monotonically",
      points[0].theta_fraction < points[1].theta_fraction < points[2].theta_fraction)


# ---- 7B/1B normalised curve-collapse hard gate ---------------------------
scan = []
sites = {"evo2_7b": ("b28_g", "b29_g", "b30_m"),
         "evo2_1b": ("late_g",)}
for model, model_sites in sites.items():
    for site in model_sites:
        for i in range(20):
            for a in grid:
                q = a / 0.001
                radial = q / (1.0 + q)
                common = dict(
                    model=model, locus=f"L{i}", site=site, block=1,
                    update_tap="g1", alpha=a, alpha_star=0.001, q=q,
                    d_shape_from_natural=0.0, log_beta_from_natural=0.0,
                    theta=0.0, theta_fraction=radial,
                    update_norm=1.0, host_norm=0.001, control_angle=1.0,
                    dependency_keys=(f"locus:L{i}",))
                scan.append(ScaleScanRow(
                    control="radial", path_fraction=radial, **common))
                scan.append(ScaleScanRow(
                    control="equal_norm_angular",
                    path_fraction=radial - 0.5, **common))
sg = scale_curve_collapse_gate(
    scan, required_models=("evo2_7b", "evo2_1b"), required_sites=sites,
    plateau_increment_margin=0.01, angular_specificity_margin=0.1,
    collapse_rmse_margin=1e-8, min_clusters=5, n_boot=200)
check("0-2 gate checks b28/b29/b30, 1B, alpha-star and curve collapse",
      sg["passed"] and all(sg["checks"].values()), str(sg["checks"]))


# ---- b30 projected vs predicted rescue contract --------------------------
reference = torch.randn(2, 3, 8, dtype=torch.float64)
pred_delta = ((reference @ space.basis) @ space.basis.T)
locked_prediction_input = torch.tensor([1, 2, 3], dtype=torch.long)
_model_directory = tempfile.TemporaryDirectory()
_model_store = ArtifactStore(Path(_model_directory.name))
_model_ref = _model_store.put_json(
    "cross_fitted_model", {"weights": "fixture"},
    source_split="chr22-train+chr21-development",
    source_role="development", created_by="fit_cross_fitted_fixture",
    metadata={
        "fit_role": "train_dev_cross_fitted",
        "train_source_sha256": "1" * 64,
        "dev_source_sha256": "2" * 64,
        "train_unit_order_sha256": "3" * 64,
        "dev_unit_order_sha256": "4" * 64,
    })
_prediction_lineage = seal_prediction_lineage(
    pred_delta, model_artifact=_model_ref, method="heldout_native_factor",
    locked_input=locked_prediction_input, locked_unit_id="test-unit")
pred = DeltaPrediction(
    pred_delta, "heldout_native_factor", "chr22-development",
    held_out=True, target_derived=False, lineage=_prediction_lineage)
conditions = b30_block_rescue_conditions(
    reference, baseline_update=torch.zeros_like(reference),
    space=space, prediction=pred, locked=True,
    locked_input=locked_prediction_input, locked_unit_id="test-unit")
for c in conditions.values():
    validate_b30_condition(c, locked=True)
check("projected rescue is marked diagnostic",
      not conditions["projected_rescue"].strong_sufficiency
      and conditions["projected_rescue"].target_derived)
check("held-out predicted b30 delta is strong sufficiency",
      conditions["predicted_rescue"].strong_sufficiency)
try:
    bad = DeltaPrediction(
        pred_delta, "leaky", "chr17", held_out=True, target_derived=True)
    b30_block_rescue_conditions(
        reference, baseline_update=torch.zeros_like(reference),
        space=space, prediction=bad, locked=True)
    check("target-derived predicted rescue is refused", False)
except RuntimeError:
    check("target-derived predicted rescue is refused", True)


# ---- Exact 2^3 cube contrasts --------------------------------------------
cube = []
for i in range(20):
    for s, sname in enumerate(("low", "high")):
        for c, cname in enumerate(("self", "donor")):
            for k, kname in enumerate(("self", "donor")):
                y = s + 2 * c + 3 * k + 7 * s * c * k
                cube.append(CausalCubeRow(
                    pair_id=f"P{i}", scale_level=sname, scale=0.5 + 0.5*s,
                    content_level=cname, carrier_level=kname,
                    b30_condition="free", d_shape=abs(y), log_beta=0.0,
                    signed_donor_following=float(y), branch_dose=float(s),
                    carrier_shift=float(k), b30_strong_sufficiency=False,
                    b30_rescue_held_out=False, b30_target_derived=False,
                    dependency_keys=(f"pair:P{i}",)))
cf = cube_factorial_effects(
    cube, low_scale="low", high_scale="high", min_clusters=5, n_boot=200)
check("cube recovers the scale-content-carrier three-way interaction",
      abs(cf["estimates"]["scale:content:carrier"]["point"] - 7.0) < 1e-12)
cv = cube_factorization_verdict(
    {"content_endpoint": cf, "calibration_endpoint": cf},
    axis_sources={
        "content": ("content_endpoint", "content"),
        "scale": ("calibration_endpoint", "scale"),
        "carrier": ("calibration_endpoint", "carrier"),
    },
    effect_margins={"scale": .1, "content": .1, "carrier": .1},
    equivalence_margin=.05,
)
check("cube verdict reports a strong three-way interaction as conditional",
      cv["conclusion_code"] == "axes_conditionally_separable")


# ---- Reciprocal direction transfer + b30 necessity/rescue ----------------
transfer = []
for i in range(20):
    for direction in ("A<-B", "B<-A"):
        did = f"P{i}:{direction}"
        common = dict(
            pair_id=f"P{i}", direction_id=did, d_shape=0.0, log_beta=0.0,
            branch_dose=1.0, angular_dose=0.5,
            dependency_keys=(f"pair:P{i}",))
        for condition, value in (
            ("direction_transfer", 1.0), ("norm_only", 0.0),
            ("wrong_layer", 0.0), ("wrong_pair", 0.0),
            ("random_matched", 0.0)):
            transfer.append(DirectionTransferRow(
                condition=condition, b30_condition="free",
                signed_donor_following=value,
                b30_strong_sufficiency=False, b30_rescue_held_out=False,
                b30_target_derived=False, **common))
        for bname, value, strong, held in (
            ("block", 0.01, False, False),
            ("projected_rescue", 0.85, False, False),
            ("predicted_rescue", 0.99, True, True),
        ):
            transfer.append(DirectionTransferRow(
                condition="direction_transfer", b30_condition=bname,
                signed_donor_following=value,
                b30_strong_sufficiency=strong, b30_rescue_held_out=held,
                b30_target_derived=False, **common))
tv = direction_transfer_verdict(
    transfer, effect_margin=0.1,
    specificity_margins={k: 0.1 for k in
                         ("norm_only", "wrong_layer", "wrong_pair", "random_matched")},
    equivalence_margin=0.05, rescue_margin=0.1,
    min_clusters=5, n_boot=200)
check("direction transfer needs reciprocity, controls, block and predicted rescue",
      tv["claim"] == "specific reciprocal g28-direction transfer through b30"
      and all(tv["checks"].values()), str(tv["checks"]))


# ---- Cube content effect is blocked and predicted-rescued at b30 ----------
med = []
for i in range(20):
    for bname, donor_value, strong, held in (
        ("free", 1.0, False, False),
        ("block", 0.01, False, False),
        ("projected_rescue", 0.85, False, False),
        ("predicted_rescue", 0.99, True, True),
    ):
        for content, value in (("self", 0.0), ("donor", donor_value)):
            med.append(CausalCubeRow(
                pair_id=f"P{i}", scale_level="high", scale=1.0,
                content_level=content, carrier_level="self",
                b30_condition=bname, d_shape=abs(value), log_beta=0.0,
                signed_donor_following=value, branch_dose=1.0,
                carrier_shift=0.0, b30_strong_sufficiency=strong,
                b30_rescue_held_out=held, b30_target_derived=False,
                dependency_keys=(f"pair:P{i}",)))
mv = cube_b30_mediation(
    med, scale_level="high", effect_margin=0.1, rescue_margin=0.1,
    min_clusters=5, n_boot=200)
check("cube content path closes with b30 block/predicted rescue",
      all(mv["checks"].values()), str(mv["checks"]))
syn = causal_factorization_synthesis(cv, mv, tv)
check("factorization synthesis preserves the cube's conditional result",
      syn["conclusion_code"]
      == "conditional_scale_carrier_content_factorization")


print()
print(f"{sum(OK)}/{len(OK)} passed")
if not all(OK):
    raise AssertionError("one or more mechanism/causal-use checks failed")
if __name__ == "__main__":
    sys.exit(0 if all(OK) else 1)

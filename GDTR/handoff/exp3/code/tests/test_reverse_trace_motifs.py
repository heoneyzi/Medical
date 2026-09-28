"""Model-free regression tests for Steps 13--14.

The tests use exact synthetic tensors so a failure identifies bookkeeping or
algebra, not an Evo 2 checkpoint or a stochastic biological sample.
"""
import sys
sys.path.insert(0, "..")

from dataclasses import replace
import torch

from exp3.reverse_trace import (
    CandidateHit,
    carrier_excluded_direction_bank,
    carrier_removed,
    candidate_method_consensus,
    exact_b28_contribution_scores,
    hierarchical_path_scan,
    path_restricted_ism,
    project_direction_bank,
    reverse_chain_score,
    reverse_linear_adjoint, matched_kmer_enrichment, seqlet_candidates,
)
from exp3.motifs import (
    MOTIF_EQUIVALENCE_CRITERIA,
    MOTIF_POSITIVE_CRITERIA,
    VARIANT_EQUIVALENCE_CRITERIA,
    VARIANT_POSITIVE_CRITERIA,
    BilinearCellFactors,
    ConfoundRecord,
    LockedContrast,
    context_window_saturation,
    exact_bilinear_epistasis,
    motif_causal_verdict,
    ordered_block_rescue_bypass_verdict,
    ordered_serial_path_verdict,
    validate_confound_matches,
    variant_causal_verdict,
)


OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name
          + (f"   {detail}" if detail else ""))


# ---- carrier-excluded, label-free direction bank ----------------------------
u = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
x = torch.tensor([
    [20.0, 1.0, 0.0, 0.0],
    [-9.0, -1.0, 0.0, 0.0],
    [5.0, 0.0, 1.0, 0.0],
    [7.0, 0.0, -1.0, 0.0],
], dtype=torch.float64)
removed = carrier_removed(x, u)
check("carrier removal exactly eliminates the frozen axis",
      float((removed @ u).abs().max()) < 1e-12)

bank = carrier_excluded_direction_bank(
    x, u, rank=2, min_content_norm=0.5,
    source_artifact_sha256="a" * 64,
    source_unit_ids=("u0", "u1", "u2", "u3"),
    source_split="chr22-discovery", source_role="discovery")
P = bank.basis.T @ bank.basis
target_subspace = torch.diag(torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64))
check("direction bank recovers content subspace, not carrier magnitude",
      float((P - target_subspace).norm()) < 1e-10
      and float((bank.basis @ u).abs().max()) < 1e-12)

held = torch.tensor([[1000.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
held_no_carrier = held.clone(); held_no_carrier[:, 0] = 0
p1 = project_direction_bank(held, bank)
p2 = project_direction_bank(held_no_carrier, bank)
check("held-out bank score is invariant to carrier-only changes",
      torch.allclose(p1.scores, p2.scores, atol=1e-12))

saved_basis = bank.basis.clone()
bank.basis[0, 0] += 0.25
try:
    project_direction_bank(held, bank)
    check("direction-bank mutation is detected before held-out use", False)
except RuntimeError:
    check("direction-bank mutation is detected before held-out use", True)
bank.basis.copy_(saved_basis)

try:
    carrier_excluded_direction_bank(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), u,
        rank=1, min_content_norm=0.1,
        source_artifact_sha256="b" * 64, source_unit_ids=("carrier-only",),
        source_split="chr22-discovery", source_role="discovery")
    check("direction bank refuses carrier-only discovery rows", False)
except RuntimeError:
    check("direction bank refuses carrier-only discovery rows", True)


# ---- exact b28 contribution accounting --------------------------------------
gen = torch.Generator().manual_seed(4)
z = torch.randn(3, 4, generator=gen, dtype=torch.float64)
W1 = torch.randn(5, 4, generator=gen, dtype=torch.float64)
W2 = torch.randn(5, 4, generator=gen, dtype=torch.float64)
W3 = torch.randn(4, 5, generator=gen, dtype=torch.float64)
q = torch.randn(4, generator=gen, dtype=torch.float64)
score = exact_b28_contribution_scores(z, W1, W2, W3, q)
check("exact b28 channel scores sum to the requested g28 projection",
      score.relative_residual < 1e-12
      and torch.allclose(score.projected_output, score.reconstructed_projection,
                         atol=1e-12))
check("exact b28 ranking is deterministic and complete",
      sorted(score.channel_order) == list(range(5)) and len(score.topk(2)) == 2)


# ---- candidate consensus and path-restricted sequence attribution -----------
hits = {
    "ism": [CandidateHit("chr22", 100, 108, "ism", 3.0)],
    "seqlet": [CandidateHit("chr22", 104, 111, "seqlet", 2.0)],
    "natural": [CandidateHit("chr22", 300, 307, "natural", 9.0)],
}
consensus = candidate_method_consensus(hits, min_methods=2)
check("candidate requires agreement of independent methods",
      len(consensus) == 1 and consensus[0].methods == ("ism", "seqlet")
      and (consensus[0].start, consensus[0].end) == (100, 111))

enriched = matched_kmer_enrichment(
    ["AAGTAAGT", "CCGTACGT"], ["AAAAAAAA", "CCCCCCCC"], k=2,
    min_total_count=2)
check("natural direction-bank sequences yield matched k-mer candidates",
      enriched[0].log_odds_ratio > 0 and enriched[0].kmer in {"GT", "TA"})
seqlets = seqlet_candidates("chr22", "ACGT", [0., 3., 2., 0.],
                            lengths=(1, 2), top_k=1)
check("path scores yield multi-width seqlets",
      {(h.start, h.end) for h in seqlets} == {(1, 2), (1, 3)})

weights = {"A": 0.0, "C": 1.0, "G": 3.0, "T": -1.0}
ism = path_restricted_ism("AC", lambda s: weights[s[0]] + 2 * weights[s[1]])
check("path-restricted ISM evaluates exact base substitutions",
      abs(float(ism.effects[0, 2]) - 3.0) < 1e-12
      and abs(float(ism.effects[1, 2]) - 4.0) < 1e-12)

scan = hierarchical_path_scan(
    "AAAA", lambda s: float(s.count("C")),
    lambda s, lo, hi: s[:lo] + "C" * (hi - lo) + s[hi:], min_window=1)
check("hierarchical scan reaches every single-base leaf",
      {(r.start, r.end) for r in scan if r.end - r.start == 1}
      == {(0, 1), (1, 2), (2, 3), (3, 4)})


# ---- reverse-link summary and adjoint ----------------------------------------
chain = reverse_chain_score({
    "output_to_b30": 0.9,
    "b30_to_b29": 0.8,
    "b29_to_b28": 0.6,
    "b28_to_sequence": 0.7,
}, off_path_leakage=0.1, natural_support=True)
check("reverse score is bottleneck-limited and leakage-penalised",
      chain.limiting_link == "b29_to_b28" and abs(chain.score - 0.54) < 1e-12)

J1 = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
J2 = torch.tensor([[3.0, 0.0], [1.0, 4.0]], dtype=torch.float64)
qo = torch.tensor([1.0, -1.0], dtype=torch.float64)
adj = reverse_linear_adjoint(qo, [J1, J2])
check("reverse adjoint applies transposed local maps in reverse order",
      torch.allclose(adj, (J2 @ J1).T @ qo))


# ---- locked causal criteria --------------------------------------------------
serial_positive = {
    name: LockedContrast(.8, .7, .9, .5, "greater")
    for name in (
        "b28_block_loss", "b29_rescue_after_b28_block", "b29_block_loss",
        "m30_rescue_after_b29_block", "b30_block_loss")
}
serial_equivalence = {
    name: LockedContrast(0., -.02, .02, .05, "equivalent")
    for name in ("g28_rescue_under_b29_block", "b29_rescue_under_b30_block")
}
serial = ordered_serial_path_verdict(
    positive=serial_positive, equivalence=serial_equivalence,
    direction_frozen_before_test=True,
    b29_rescue_held_out=True, b29_rescue_target_derived=False,
    m30_rescue_held_out=True, m30_rescue_target_derived=False,
    evidence_artifact_sha256="e" * 64,
    direction_artifact_sha256="d" * 64,
    source_split="chr17-locked", source_role="locked",
    source_unit_ids=("serial-1", "serial-2"),
    dependency_keys=("locus:serial-1", "locus:serial-2"))
serial.verify_seal()
check("full serial claim explicitly intervenes on b29 with sealed CI evidence",
      serial.passed and serial.claim == "ordered g28->b29->m30 serial path")

positive = {name: LockedContrast(0.5, 0.3, 0.7, 0.1, "greater")
            for name in MOTIF_POSITIVE_CRITERIA}
equiv = {name: LockedContrast(0.0, -0.02, 0.02, 0.05, "equivalent")
         for name in MOTIF_EQUIVALENCE_CRITERIA}
specificity = {name: LockedContrast(0.4, 0.2, 0.6, 0.1, "greater")
               for name in ("wrong_motif", "shifted_position", "energy_matched")}
mv = motif_causal_verdict(
    positive=positive, equivalence=equiv, specificity=specificity,
    candidate_consensus=True, direction_frozen_before_test=True,
    natural_support=True, predicted_rescue_held_out=True,
    predicted_rescue_target_derived=False,
    edit_modes_tested=("insertion", "deletion"), ordered_serial_path=serial)
check("motif claim requires and passes the full serial causal conjunction",
      mv.passed and mv.claim == "serial-circuit causal motif"
      and mv.upstream_evidence_sha256["ordered_g28_b29_m30"]
      == serial.evidence_sha256)

weak_positive = dict(positive)
weak_positive["m30_propagation"] = LockedContrast(0.05, -0.01, 0.2, 0.1, "greater")
weak = motif_causal_verdict(
    positive=weak_positive, equivalence=equiv, specificity=specificity,
    candidate_consensus=True, direction_frozen_before_test=True,
    natural_support=True, predicted_rescue_held_out=True,
    predicted_rescue_target_derived=False,
    edit_modes_tested=("insertion", "deletion"), ordered_serial_path=serial)
check("one failed serial link prevents a motif mechanism claim",
      not weak.passed and "m30_propagation" in weak.failures)

vpositive = {name: LockedContrast(0.5, 0.3, 0.7, 0.1, "greater")
             for name in VARIANT_POSITIVE_CRITERIA}
vequiv = {name: LockedContrast(0.0, -0.02, 0.02, 0.05, "equivalent")
          for name in VARIANT_EQUIVALENCE_CRITERIA}
vv = variant_causal_verdict(
    positive=vpositive, equivalence=vequiv, specificity=specificity,
    direction_frozen_before_test=True, held_out_variant=True, natural_support=True,
    predicted_rescue_held_out=True, predicted_rescue_target_derived=False,
    ordered_serial_path=serial,
    expected_ordered_serial_path_sha256=serial.evidence_sha256)
check("held-out variant needs prediction, rescue, and off-target equivalence", vv.passed)

try:
    motif_causal_verdict(
        positive=positive, equivalence=equiv, specificity=specificity,
        candidate_consensus=True, direction_frozen_before_test=True,
        natural_support=True, predicted_rescue_held_out=True,
        predicted_rescue_target_derived=False,
        edit_modes_tested=("insertion", "deletion"))
    check("motif claim cannot omit the ordered serial-path verdict", False)
except ValueError:
    check("motif claim cannot omit the ordered serial-path verdict", True)

try:
    variant_causal_verdict(
        positive=vpositive, equivalence=vequiv, specificity=specificity,
        direction_frozen_before_test=True, held_out_variant=True,
        natural_support=True, predicted_rescue_held_out=True,
        predicted_rescue_target_derived=False,
        ordered_serial_path=replace(serial, claim="substituted verdict"))
    check("serial-path verdict mutation is rejected downstream", False)
except RuntimeError:
    check("serial-path verdict mutation is rejected downstream", True)

failed_serial_positive = dict(serial_positive)
failed_serial_positive["b29_rescue_after_b28_block"] = LockedContrast(
    .1, -.05, .2, .5, "greater")
failed_serial = ordered_serial_path_verdict(
    positive=failed_serial_positive, equivalence=serial_equivalence,
    direction_frozen_before_test=True,
    b29_rescue_held_out=True, b29_rescue_target_derived=False,
    m30_rescue_held_out=True, m30_rescue_target_derived=False,
    evidence_artifact_sha256="e" * 64,
    direction_artifact_sha256="d" * 64,
    source_split="chr17-locked", source_role="locked",
    source_unit_ids=("serial-1", "serial-2"),
    dependency_keys=("locus:serial-1", "locus:serial-2"))
negative_motif = motif_causal_verdict(
    positive=positive, equivalence=equiv, specificity=specificity,
    candidate_consensus=True, direction_frozen_before_test=True,
    natural_support=True, predicted_rescue_held_out=True,
    predicted_rescue_target_derived=False,
    edit_modes_tested=("insertion", "deletion"),
    ordered_serial_path=failed_serial)
check("valid serial-path non-support cleanly prevents the motif claim",
      not negative_motif.passed
      and "ordered_immediate_serial_path" in negative_motif.failures)


# ---- exact bilinear motif grammar -------------------------------------------
cells = {
    "WT": BilinearCellFactors(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])),
    "A": BilinearCellFactors(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0])),
    "B": BilinearCellFactors(torch.tensor([1.0, 1.0]), torch.tensor([3.0, 1.0])),
    "AB": BilinearCellFactors(torch.tensor([2.0, 1.0]), torch.tensor([3.0, 1.0])),
}
epi = exact_bilinear_epistasis(
    cells, torch.eye(2, dtype=torch.float64),
    torch.tensor([1.0, 0.0], dtype=torch.float64))
check("bilinear epistasis recovers exact multiplicative interaction",
      abs(float(epi.total_interaction) - 2.0) < 1e-12
      and epi.relative_residual < 1e-12)


# ---- ordered bypass and context range ---------------------------------------
bypass = ordered_block_rescue_bypass_verdict(
    native_effect=1.0, b28_block_effect=0.1,
    b28_block_plus_m30_delta_rescue=0.9,
    b30_block_effect=0.05, b30_block_plus_g28_rescue=0.08,
    max_block_fraction=0.2, min_rescue_fraction=0.8,
    max_wrong_order_fraction=0.2)
check("scalar bypass is only an honest g28-before-m30 diagnostic",
      bypass.passed and bypass.claim == "g28-before-m30 bypass asymmetry diagnostic")

sat = context_window_saturation(
    {1: 0.4, 3: 0.92, 5: 0.98, 9: 1.0},
    target_fraction=0.9, plateau_tolerance_fraction=0.1)
check("context saturation selects the first persistent full-effect window",
      sat.saturated and sat.minimal_width == 3)


# ---- next-base and genomic confound matching --------------------------------
records = [
    ConfoundRecord("M1", "target", "G", -2.0, 0.50, "LINE", 4.0, "+", 1),
    ConfoundRecord("M1", "control", "G", -2.05, 0.52, "LINE", 4.2, "+", 1),
    ConfoundRecord("M2", "target", "A", -1.0, 0.40, "none", 8.0, "-", 0),
    ConfoundRecord("M2", "control", "A", -0.98, 0.39, "none", 8.1, "-", 0),
]
tolerances = {"kmer_log_likelihood": 0.1, "gc_fraction": 0.05,
              "relative_position": 0.5}
report = validate_confound_matches(records, continuous_tolerances=tolerances)
check("matched controls lock token, likelihood, GC, repeat, position, strand, and phase",
      report.passed and report.n_matches == 2 and report.n_controls == 2)

bad_records = list(records)
bad_records[1] = ConfoundRecord(
    "M1", "control", "T", -2.05, 0.52, "LINE", 4.2, "+", 1)
bad_report = validate_confound_matches(
    bad_records, continuous_tolerances=tolerances, strict=False)
check("next-token mismatch is reported instead of becoming a motif",
      not bad_report.passed and any("target_token" in x for x in bad_report.failures))


print()
print(f"{sum(OK)}/{len(OK)} passed")
if not all(OK):
    raise AssertionError("one or more reverse-trace/motif checks failed")
if __name__ == "__main__":
    sys.exit(0 if all(OK) else 1)

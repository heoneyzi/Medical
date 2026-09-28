"""The analysis frame: constants that cannot be arbitrary, verdicts that cannot be forced."""

from __future__ import annotations

import numpy as np
import pytest

from geoflowagent.geoacmg.cited import (
    Provenance,
    assert_decision_safe,
    convention,
    guideline,
    measured,
)
from geoflowagent.geoacmg.claims import (
    Claim,
    Evidence,
    Finding,
    Outcome,
    Preregistration,
    Role,
    adjudicate,
)
from geoflowagent.geoacmg.estimators import cluster_slope, permutation_null
from geoflowagent.geoacmg.inference import adjust_exploratory, moderator_slope, paired_contrast


class TestCitedConstants:
    def test_a_constant_cannot_exist_without_a_source(self):
        with pytest.raises(ValueError):
            guideline(0.05, "   ")

    def test_a_measured_constant_must_name_its_sample(self):
        with pytest.raises(ValueError):
            measured(1.2, "median latency", sample="")
        assert measured(1.2, "median latency", sample="20-gene pilot").provenance is Provenance.MEASURED

    def test_a_convention_may_not_decide_an_outcome(self):
        with pytest.raises(ValueError, match="convention decide"):
            assert_decision_safe(convention(0.5, "seemed fine"), "gate")
        assert_decision_safe(guideline(0.05, "pre-registered level"), "gate")


CLAIM = Claim("C3", "plan generation beats greedy", "curve optimised at k=1", +1)
PREREG = Preregistration("t", (CLAIM,), ("P3_flow_minus_greedy_at_matched_cost",))


def _finding(low, high, name="P3_flow_minus_greedy_at_matched_cost", role=Role.PRIMARY):
    return Finding(
        claim_id="C3", name=name, evidence=Evidence.PAIRED_INTERVAL,
        estimate=(low + high) / 2, ci_low=low, ci_high=high,
        unit="gene", n_units=40, role=role,
    )


class TestAdjudication:
    def test_interval_excluding_zero_in_the_expected_direction_supports(self):
        assert adjudicate([CLAIM], [_finding(0.02, 0.14)])[0].outcome is Outcome.SUPPORTED

    def test_interval_excluding_zero_the_other_way_refutes(self):
        assert adjudicate([CLAIM], [_finding(-0.14, -0.02)])[0].outcome is Outcome.REFUTED

    def test_interval_straddling_zero_is_unresolved_not_a_trend(self):
        verdict = adjudicate([CLAIM], [_finding(-0.03, 0.09)])[0]
        assert verdict.outcome is Outcome.UNRESOLVED
        assert "includes zero" in verdict.reason

    def test_a_claim_with_no_primary_finding_is_not_tested(self):
        exploratory = _finding(0.5, 0.9, name="side_sweep", role=Role.EXPLORATORY)
        assert adjudicate([CLAIM], [exploratory])[0].outcome is Outcome.NOT_TESTED

    def test_one_refutation_outweighs_other_support(self):
        prereg = Preregistration("t", (CLAIM,), ("a", "b"))
        findings = prereg.enforce_roles([_finding(0.1, 0.3, "a"), _finding(-0.3, -0.1, "b")])
        assert adjudicate([CLAIM], findings)[0].outcome is Outcome.REFUTED

    def test_an_unregistered_finding_cannot_claim_primary_role(self):
        with pytest.raises(ValueError, match="not pre-registered"):
            PREREG.enforce_roles([_finding(0.9, 1.0, name="found_later")])

    def test_a_finding_for_an_unknown_claim_is_rejected(self):
        stray = Finding("C9", "x", Evidence.PAIRED_INTERVAL, 1.0, "gene", 5, ci_low=0.5, ci_high=1.5)
        with pytest.raises(KeyError):
            adjudicate([CLAIM], [stray])

    def test_a_paired_finding_without_an_interval_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="needs an interval"):
            Finding("C3", "x", Evidence.PAIRED_INTERVAL, 1.0, "gene", 5)

    def test_preregistration_must_name_its_primaries(self):
        with pytest.raises(ValueError, match="primary findings"):
            Preregistration("t", (CLAIM,), ())


class TestEstimators:
    def test_permutation_null_is_calibrated_when_the_null_is_true(self):
        """Under no association the p-value should not be small."""

        rng = np.random.default_rng(0)
        values = list(rng.normal(size=200))
        labels = list(rng.integers(0, 2, size=200))

        def statistic(pairs):
            ones = [v for v, tag in pairs if tag == 1]
            zeros = [v for v, tag in pairs if tag == 0]
            return float(np.mean(ones) - np.mean(zeros))

        result = permutation_null(statistic, values, labels, draws=200, seed=3)
        assert result["p_value"] > 0.05
        assert result["draws"] == 200
        assert result["p_value"] >= 1.0 / 201, "add-one correction keeps p away from zero"

    def test_cluster_slope_recovers_a_known_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = 0.7 * x + rng.normal(scale=0.5, size=400)
        interval = cluster_slope(x, y, [i % 40 for i in range(400)], resamples=400)
        assert interval.low < 0.7 < interval.high
        assert interval.units == 40

    def test_paired_contrast_requires_a_cluster_label_per_observation(self):
        with pytest.raises(ValueError, match="cluster label"):
            paired_contrast(claim_id="C3", name="x", left=[1, 2], right=[0, 1], clusters=["g"])

    def test_moderator_slope_reports_the_moderator_it_used(self):
        rng = np.random.default_rng(1)
        moderator = rng.uniform(size=200)
        benefit = 0.4 * moderator + rng.normal(scale=0.05, size=200)
        finding = moderator_slope(
            claim_id="C3", name="geometry_by_irreversibility", moderator=moderator,
            benefit=benefit, clusters=[i % 25 for i in range(200)],
            moderator_name="irreversibility", resamples=300,
        )
        assert finding.detail["moderator"] == "irreversibility"
        assert finding.ci_low > 0

    def test_exploratory_findings_are_adjusted_and_primaries_are_not(self):
        primary = Finding("C3", "P3_flow_minus_greedy_at_matched_cost", Evidence.NULL_POSITION,
                          1.0, "gene", 30, Role.PRIMARY, p_value=0.01, null_draws=200)
        others = [
            Finding("C3", f"sweep_{i}", Evidence.NULL_POSITION, 1.0, "gene", 30,
                    Role.EXPLORATORY, p_value=p, null_draws=200)
            for i, p in enumerate([0.01, 0.02, 0.04])
        ]
        adjusted = adjust_exploratory([primary, *others])
        assert primary.name not in adjusted
        assert all(adjusted[f.name] >= f.p_value for f in others)

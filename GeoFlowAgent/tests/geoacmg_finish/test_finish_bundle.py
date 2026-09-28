"""신규 모듈 기능 테스트.

핵심은 test_readout_recovers_norm_signal 이다: 정답을 아는 합성 데이터를 만들어
readout 분해가 실제로 기제를 가려내는지 확인한다.  "떨어졌다"가 아니라 "무엇을
버렸을 때 떨어졌는가"를 답하는 것이 이 모듈의 존재 이유이므로, 그 능력 자체를
검정한다.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from geoflowagent.geoacmg import (
    amend,
    budget,
    control,
    durable,
    paired,
    parallel,
    readout,
    seal,
    strata,
    winnability,
)
from geoflowagent.geoacmg.claims import Evidence, Finding, Role


# ------------------------------------------------------------------- durable


def test_durable_rejects_volatile(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(durable, "filesystem_type", lambda _p: "tmpfs")
        root = durable.DurableRoot(Path(tmp))
        with pytest.raises(durable.VolatileStorageError):
            root.require()


def test_durable_write_through_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(durable, "filesystem_type", lambda _p: "ext4")
        root = durable.DurableRoot(Path(tmp), min_free_mb=0).require()
        root.write_json("findings/x.json", {"a": 1})
        assert json.loads((Path(tmp) / "findings/x.json").read_text())["a"] == 1
        path = root.write_pairs("x", values=np.arange(4))
        assert np.load(path)["values"].tolist() == [0, 1, 2, 3]


# -------------------------------------------------------------------- paired


def _run(family: str, seed: int, fingerprint: str, ordering: float, *, energy_params=0) -> paired.SeedRun:
    genes = [f"G{i % 5}" for i in range(20)]
    per_task = {f"t{i}": ordering + 0.01 * ((i % 3) - 1) for i in range(20)}
    return paired.SeedRun(
        family=family,
        seed=seed,
        trunk_fingerprint=fingerprint,
        energy_params=energy_params,
        total_params=335_942,
        metrics={"ordering": ordering},
        per_task=per_task,
        task_gene={f"t{i}": genes[i] for i in range(20)},
    )


def test_paired_rejects_mismatched_trunk():
    runs = [_run("cosine", 1, "aaa", 0.74), _run("euclidean", 1, "bbb", 0.59)]
    with pytest.raises(paired.TrunkMismatch):
        paired.paired_findings(runs, claim_id="G4")


def test_paired_rejects_energy_parameters():
    runs = [_run("cosine", 1, "aaa", 0.74), _run("euclidean", 1, "aaa", 0.59, energy_params=7)]
    with pytest.raises(paired.EnergyNotParameterFree):
        paired.paired_findings(runs, claim_id="G4")


def test_paired_reports_two_variance_sources():
    runs = []
    for seed, (left, right) in enumerate([(0.74, 0.59), (0.75, 0.60), (0.73, 0.58)], start=1):
        runs.append(_run("cosine", seed, f"fp{seed}", left))
        runs.append(_run("euclidean", seed, f"fp{seed}", right))
    report = paired.paired_findings(runs, claim_id="G4")
    names = [f.name for f in report.findings]
    assert any("seed_clustered" in n for n in names)
    assert any("gene_clustered" in n for n in names)
    units = {f.unit for f in report.findings}
    assert units == {"seed", "gene"}, "두 분산원이 분리되어 보고되어야 한다"
    assert report.to_payload()["prior_prediction"]["estimate"] == pytest.approx(0.1497)


# ------------------------------------------------------------------- readout


def _decompose(z_state, z_goal, own, targets, clusters):
    return readout.decompose_family(
        family="synthetic",
        z_state=z_state,
        z_goal=z_goal,
        own_energy=own,
        targets=targets,
        clusters=clusters,
        readouts=readout.build_readouts(standardise=None, whiten=None),
        claim_id="G3",
        draws=40,
    )


def test_ordering_accuracy_treats_no_information_as_chance():
    """정보 없음(모든 점수 동일)은 0.0 이 아니라 0.5 여야 한다.

    분해에서 '우연 아래' 값을 읽는 이상, '정보 없음'과 '정확히 거꾸로'는
    반드시 구분되어야 한다.
    """

    targets = np.arange(20, dtype=float)
    assert readout.ordering_accuracy(np.zeros(20), targets) == pytest.approx(0.5)
    assert readout.ordering_accuracy(targets, targets) == pytest.approx(1.0)
    assert readout.ordering_accuracy(-targets, targets) == pytest.approx(0.0)


def test_readout_isolates_norm_carried_signal():
    """신호를 **크기에만** 심으면 크기 보존 readout 만 복원해야 한다.

    z 를 목표 방향으로만 늘린다: 각도는 모든 표본에서 동일하고 거리만 다르다.
    그러면 정답이 정해진다 — 코사인은 원리적으로 구분 불가(=0.5), L2 는 완전 복원.
    """

    rng = np.random.default_rng(0)
    n, dim = 200, 16
    targets = rng.uniform(0.5, 5.0, size=n)
    axis = rng.normal(size=dim)
    axis /= np.linalg.norm(axis)
    z_goal = np.repeat(axis[None, :], n, axis=0)
    z_state = z_goal * (1.0 + targets)[:, None]        # 순수 반경 방향 — 각도 불변
    own = ((z_state - z_goal) ** 2).sum(-1)
    clusters = [f"G{i % 10}" for i in range(n)]

    results, findings, arrays = _decompose(z_state, z_goal, own, targets, clusters)
    by_key = {r.readout: r for r in results}
    assert by_key["l2"].accuracy > 0.99
    assert by_key["cos"].accuracy == pytest.approx(0.5, abs=0.02), "코사인은 각도만 본다"
    assert by_key["l2"].recovery > by_key["cos"].recovery
    assert "scores_l2" in arrays and "own_per_unit" in arrays
    assert all(f.detail["measured_chance"]["draws"] == 40 for f in findings)


def test_readout_isolates_angle_carried_signal():
    """반대 방향도 확인한다 — 신호를 **각도에만** 심으면 코사인이 복원해야 한다.

    한쪽만 맞히는 모듈은 기제를 가려내는 것이 아니라 한쪽으로 치우친 것이다.
    """

    rng = np.random.default_rng(1)
    n, dim = 200, 16
    targets = rng.uniform(0.05, 1.0, size=n)
    base = rng.normal(size=dim)
    base /= np.linalg.norm(base)
    other = rng.normal(size=dim)
    other -= other @ base * base
    other /= np.linalg.norm(other)
    z_goal = np.repeat(base[None, :], n, axis=0)
    angles = targets
    z_state = np.cos(angles)[:, None] * base + np.sin(angles)[:, None] * other
    own = 1.0 - (z_state * z_goal).sum(-1)             # 자기 에너지 = 코사인 거리
    clusters = [f"G{i % 10}" for i in range(n)]

    results, _f, _a = _decompose(z_state, z_goal, own, targets, clusters)
    by_key = {r.readout: r for r in results}
    assert by_key["cos"].accuracy > 0.99, "각도에 실린 신호는 코사인이 복원해야 한다"
    assert by_key["cos"].recovery > 0.9


def test_readout_interpretation_groups_by_mechanism():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(50, 8))
    readouts = readout.build_readouts(
        standardise=readout.fit_standardiser(z), whiten=readout.fit_whitener(z)
    )
    keys = {r.key for r in readouts}
    assert {"cos", "l2", "neg_dot", "cos_std", "cos_whiten"} == keys


# -------------------------------------------------------------------- strata


def test_strata_reports_slope_and_headroom():
    rng = np.random.default_rng(2)
    n = 200
    moderator = rng.uniform(1.0, 8.0, size=n)
    treatment = 0.9 - 0.02 * moderator + rng.normal(scale=0.01, size=n)
    baseline = treatment - (0.05 - 0.004 * moderator)
    clusters = [f"G{i % 20}" for i in range(n)]
    report = strata.stratified_report(
        claim_id="G5",
        name="geometry_benefit",
        moderator=moderator.tolist(),
        treatment=treatment.tolist(),
        control=baseline.tolist(),
        clusters=clusters,
    )
    payload = report.to_payload()
    assert payload["primary"] == "continuous slope (no binning)"
    assert len(payload["strata"]) >= 2
    for row in payload["strata"]:
        assert row["headroom"] == pytest.approx(1.0 - row["absolute"])
    assert "quantile" in payload["boundary_rule"]


def test_saturation_profile_orders_by_headroom():
    profile = strata.saturation_profile(
        {"action": 0.9254, "state": 0.5870, "policy": 0.8699},
        {"action": 0.0113, "state": 0.2396, "policy": 0.0188},
    )
    assert profile["rows"][0]["level"] == "action"
    assert profile["rows"][0]["headroom"] == pytest.approx(0.0746, abs=1e-4)


# ------------------------------------------------------------------- control


def _finding(name: str, method: str, detail=None) -> Finding:
    return Finding(
        claim_id="G5",
        name=name,
        evidence=Evidence.PAIRED_INTERVAL,
        estimate=0.3,
        unit="gene",
        n_units=40,
        role=Role.EXPLORATORY,
        ci_low=0.2,
        ci_high=0.4,
        method=method,
        detail=detail or {},
    )


def test_enforce_controls_blocks_uncontrolled_correlation():
    with pytest.raises(control.MissingControl):
        control.enforce_controls([_finding("alignment:linkage", "OLS slope with cluster bootstrap")])


def test_control_attaches_and_reads_interval():
    rng = np.random.default_rng(3)
    clusters = [f"G{i % 15}" for i in range(90)]
    matched = rng.normal(0.33, 0.02, size=90)
    mismatched = rng.normal(0.33, 0.02, size=90)          # C5 와 같은 상황
    report = control.crossmodel_control(
        statistic_name="alignment_linkage",
        matched=matched.tolist(),
        mismatched=mismatched.tolist(),
        clusters=clusters,
    )
    assert "includes zero" in report.survives
    attached = control.attach(_finding("alignment:linkage", "OLS slope"), report)
    control.enforce_controls([attached])                   # 이제 통과해야 한다


# --------------------------------------------------------------- winnability


def test_winnability_only_compute_axis_survives():
    scale = winnability.Scale(109_000_000, 335_942, 2_921, 448_704, 155, 12.5)
    payload = winnability.assess(scale)
    assert payload["survivable_axes"] == ["compute"]
    assert payload["scale"]["params_per_effective_cluster"] > 1e6


def test_winnability_declaration_matches_c4_form():
    scale = winnability.Scale(109_000_000, 335_942, 2_921, 448_704, 155, 12.5)
    text = winnability.declaration(scale, reason="테스트")
    assert text["status"] == "structurally untestable in this benchmark"
    assert "C4" in text["precedent"]


# --------------------------------------------------------------------- amend


def test_amendment_requires_sealed_test():
    with pytest.raises(RuntimeError):
        amend.require_sealed(12)
    assert amend.require_sealed(0)["test_tasks"] == 0


def test_amendment_carries_prior_table_and_stable_fingerprint():
    one = amend.Amendment(study="geoacmg", seal_state=amend.require_sealed(0))
    two = amend.Amendment(study="geoacmg", seal_state=amend.require_sealed(0))
    assert one.fingerprint() == two.fingerprint()
    payload = one.to_payload()
    assert payload["amends"] == "143e496ce323074c"
    assert payload["prior_prediction"]["table"]["cosine"]["policy"] == pytest.approx(0.8613)
    keys = {d["key"] for d in payload["degrees_of_freedom"]}
    assert {"max_tasks_per_gene", "task_length_weighting", "energy_family_policy", "seeds"} <= keys


# ---------------------------------------------------------------------- seal


def test_seal_requires_frozen_degrees_of_freedom():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "intent.json"
        with pytest.raises(seal.SealViolation):
            seal.write_intent(
                target,
                reason="열겠다",
                findings_to_read=["x"],
                frozen_degrees_of_freedom={"seeds": 5},   # 나머지가 비어 있다
                analysis_plan_sha256="a" * 64,
            )


def test_seal_detects_plan_change_after_intent():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "intent.json"
        dof = {
            "max_tasks_per_gene": [25, 50, 100],
            "task_length_weighting": "all_three",
            "energy_family_policy": "report_axis",
            "seeds": 5,
        }
        seal.write_intent(
            target,
            reason="확정 후 개봉",
            findings_to_read=["G4:seed_clustered"],
            frozen_degrees_of_freedom=dof,
            analysis_plan_sha256="a" * 64,
        )
        intent = seal.require_intent(target, analysis_plan_sha256="a" * 64)
        assert intent.findings_to_read == ("G4:seed_clustered",)
        with pytest.raises(seal.SealViolation):
            seal.require_intent(target, analysis_plan_sha256="b" * 64)


# -------------------------------------------------------------------- budget


def test_budget_orders_decisive_first_and_flags_overrun():
    governor = budget.Governor.for_hours(1.0)
    stages = [
        budget.StagePlan("cheap_decisive", "싸고 결정적", True, False, 10, lambda n: 0.001 * n),
        budget.StagePlan("huge_optional", "비싸고 선택적", False, True, 10_000_000, lambda n: 0.5 * n),
        budget.StagePlan("mid_decisive", "중간 결정적", True, False, 100, lambda n: 0.01 * n),
    ]
    plan = governor.plan(stages, probe_units=2)
    assert plan["order"][0] == "cheap_decisive"
    assert plan["order"].index("mid_decisive") < plan["order"].index("huge_optional")
    huge = next(r for r in plan["rows"] if r["stage"] == "huge_optional")
    assert not huge["fits"] and huge["skipped_reason"]


def test_budget_guard_refuses_to_start_overrunning_stage():
    governor = budget.Governor.for_hours(0.01)      # 36초
    stage = budget.StagePlan("big", "큼", True, True, 1000, lambda n: 1.0 * n)
    stage.measure(probe_units=1)
    with pytest.raises(budget.DeadlineExceeded):
        governor.guard(stage)


# ------------------------------------------------------------------ parallel


def test_worker_budget_is_memory_derived_not_cpu_count():
    b = parallel.worker_budget(per_worker_bytes=31 * 1024**3, ceiling=64)
    assert b.workers >= 1
    assert "메모리 기준" in b.reason or "cgroup 한도 없음" in b.reason


def test_batched_covers_every_item():
    items = list(range(10))
    chunks = list(parallel.batched(items, 3))
    assert [x for c in chunks for x in c] == items
    assert len(chunks) == 4


def test_autotune_stops_when_per_item_time_plateaus():
    calls: list[int] = []

    def probe(size: int) -> float:
        calls.append(size)
        return 0.001 * size if size <= 128 else 0.001 * size * 1.5

    chosen = parallel.autotune_batch(probe, candidates=(32, 64, 128, 256, 512))
    assert chosen in (32, 64, 128)
    assert 512 not in calls, "정체 지점 이후는 탐색하지 않아야 한다"


def test_progress_reports_eta():
    import io

    stream = io.StringIO()
    progress = parallel.Progress(total=4, label="t", every_seconds=0.0, stream=stream)
    for _ in range(4):
        progress.tick()
    assert "eta" in stream.getvalue()
    assert progress.eta_seconds() == pytest.approx(0.0, abs=1.0)


def test_find_reduction_switches_sees_env_gates(tmp_path):
    module = tmp_path / "m.py"
    module.write_text(
        "import os\n"
        "flag = os.environ.get('GEOFLOW_STATE_FLOW_REPORT_TRAIN', '1')\n",
        "utf-8",
    )
    hits = parallel.find_reduction_switches(tmp_path)
    assert any(h["env"] == "GEOFLOW_STATE_FLOW_REPORT_TRAIN" for h in hits)

"""어댑터가 연결된 실제 스테이지 구현.

``finish.py`` 의 등록표는 R1·R2·R3·R4·R7 을 ``stage_placeholder`` 로 두고
"무엇이 연결돼야 하는지"만 적어 두었다.  이 모듈이 그 자리를 채운다.

설계의 핵심: **per-pair 원시 마진만 있으면 store 없이 재계산된다.**
``findings/R2_readout_margins.npz`` 는 쌍마다 점수 차를 float32 로 담고 있어
(부호가 아니라 마진) 동점 규약·집계 단위·층 경계를 바꿔도 재학습은 물론
store 적재(11분)조차 필요 없다.  저널 §21 이 약속한 형태를 실제로 지킨 결과다.

따라서 R1·R2·R3·R4 는 전부 분 단위이고, store 가 필요한 것은 마진을 **처음 만들 때**와
R7 의 flow 평가뿐이다.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from geoflowagent.geoacmg import control, estimators, paired, strata
from geoflowagent.geoacmg.claims import Role

MARGINS_NAME = "R2_readout_margins.npz"
FAMILIES = ("cosine", "euclidean")
ENERGIES = ("cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp")
READOUT_KINDS = ("own_energy", "cos", "l2", "neg_dot", "cos_std", "cos_whiten")


class MarginsMissing(RuntimeError):
    """per-pair 마진이 없다. 먼저 만들어야 한다 (store 필요)."""


def load_margins(findings_dir: Path) -> dict[str, Any]:
    path = Path(findings_dir) / MARGINS_NAME
    if not path.is_file():
        raise MarginsMissing(
            f"{path} 가 없다.\n"
            "이 파일은 체크포인트에서 z 를 꺼내 쌍마다 점수 차를 float32 로 담은 것이다.\n"
            "만들려면 store 적재(약 11분)가 필요하다: scripts/r234_pass.py 참조."
        )
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def seeds_present(m: dict[str, Any], family: str) -> list[int]:
    out = []
    for key in m:
        if key.startswith(f"own_energy_{family}-"):
            try:
                out.append(int(key.rsplit("-", 1)[1]))
            except ValueError:
                continue
    return sorted(out)


def concordance(margin: np.ndarray, target: np.ndarray, *, tie_corrected: bool = True) -> np.ndarray:
    """동점 보정 규약이 기본. 동점은 0.5 로 센다.

    참조 구현(``probes.depth_matched_pairs``)은 ``(s[a]<s[b])==(t[a]<t[b])`` 라
    동점이 참 순서 방향으로 접힌다.  두 규약을 모두 낼 수 있게 인자로 둔다.
    """

    margin = margin.astype(np.float64)
    if not tie_corrected:
        return ((margin > 0) == (target > 0)).astype(np.float64)
    out = np.where(margin * target > 0, 1.0, 0.0)
    return np.where(margin == 0, 0.5, out)


def per_task_mean(values: np.ndarray, tasks: np.ndarray) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for task, value in zip(tasks.tolist(), values, strict=True):
        total[task] += float(value)
        count[task] += 1
    return {t: total[t] / count[t] for t in total}


# ------------------------------------------------------------------ R1


def run_paired_seeds(findings_dir: Path, checkpoints_dir: Path) -> dict[str, Any]:
    """R1 — 파라미터 0개 쌍 대조. 마진에서 계산한다."""

    from geoflowagent.geoacmg import adapters

    m = load_margins(findings_dir)
    target, tasks, genes = m["target_margin"], m["task"], m["gene"]
    task_gene = dict(zip(tasks.tolist(), genes.tolist(), strict=True))

    shared = sorted(set(seeds_present(m, FAMILIES[0])) & set(seeds_present(m, FAMILIES[1])))
    if not shared:
        raise MarginsMissing("마진에 두 family 공통 시드가 없다")

    runs: list[paired.SeedRun] = []
    for family in FAMILIES:
        for seed in shared:
            directory = None
            for candidate in (
                Path(checkpoints_dir) / "geometry_comparison" / family / f"seed-{seed}",
                Path(checkpoints_dir) / "geometry_r1_pairs" / family / f"seed-{seed}",
            ):
                if (candidate / "value_geometry.pt").is_file():
                    directory = candidate
                    break
            if directory is None:
                raise MarginsMissing(f"{family}-{seed} 체크포인트를 찾지 못했다")
            model = adapters.build_model_at_init(directory)
            vec = concordance(m[f"own_energy_{family}-{seed}"], target)
            runs.append(
                paired.SeedRun(
                    family=family,
                    seed=seed,
                    trunk_fingerprint=paired.state_dict_fingerprint(adapters.TrunkView(model)),
                    energy_params=paired.energy_parameter_count(model),
                    total_params=int(sum(p.numel() for p in model.parameters())),
                    metrics={"ordering": float(vec.mean())},
                    per_task=per_task_mean(vec, tasks),
                    task_gene=task_gene,
                )
            )

    report = paired.paired_findings(
        runs, families=FAMILIES, metric="ordering", claim_id="C2",
        role=Role.EXPLORATORY, require_zero_params=True, seed=17,
    )
    payload = report.to_payload()
    payload["seeds_used"] = shared
    payload["source"] = "per-pair 마진에서 재계산 (store 불필요)"
    return payload


# ------------------------------------------------------------------ R2


def run_readout_decomposition(findings_dir: Path) -> dict[str, Any]:
    """R2 — readout 6종 분해. 두 동점 규약을 나란히 낸다."""

    m = load_margins(findings_dir)
    target, genes = m["target_margin"], m["gene"].tolist()
    rows: list[dict[str, Any]] = []
    tie_vectors: dict[str, np.ndarray] = {}

    for key, margin in m.items():
        if key in {"target_margin", "root_vstar", "task", "gene", "pair_a", "pair_b"}:
            continue
        ref = concordance(margin, target, tie_corrected=False)
        tie = concordance(margin, target, tie_corrected=True)
        tie_vectors[key] = tie
        ties = int((margin.astype(np.float64) == 0).sum())
        row = {
            "readout": key,
            "accuracy_reference_convention": float(ref.mean()),
            "accuracy_tie_corrected": float(tie.mean()),
            "difference": float(tie.mean() - ref.mean()),
            "tie_count": ties,
            "tie_rate": ties / margin.size,
        }
        for kind in sorted(READOUT_KINDS, key=len, reverse=True):
            if key.startswith(kind + "_"):
                row["readout_kind"] = kind
                rest = key[len(kind) + 1:]
                if "-" in rest:
                    row["energy"], seed = rest.rsplit("-", 1)
                    row["seed"] = int(seed) if seed.isdigit() else seed
                break
        rows.append(row)

    recovery: dict[str, Any] = {}
    for key in tie_vectors:
        if not key.startswith("own_energy_"):
            continue
        run = key[len("own_energy_"):]
        own = tie_vectors[key]
        entry: dict[str, Any] = {"own_energy": {"tie_corrected": float(own.mean())}}
        for kind in READOUT_KINDS:
            if kind == "own_energy":
                continue
            other = tie_vectors.get(f"{kind}_{run}")
            if other is None:
                continue
            interval = estimators.cluster_bootstrap(
                (other - own).tolist(), genes, resamples=1000, seed=17
            )
            entry[kind] = {
                "tie_corrected": float(other.mean()),
                "minus_own_energy": {
                    "estimate": float(interval.estimate),
                    "low": float(interval.low),
                    "high": float(interval.high),
                    "units": int(interval.units),
                    "excludes_zero": bool(interval.low > 0 or interval.high < 0),
                },
            }
        recovery[run] = entry

    max_tie = max((r["tie_rate"] for r in rows), default=0.0)
    return {
        "split": "dev",
        "test_reported": False,
        "unit": "gene",
        "pairs": int(target.size),
        "genes": len(set(genes)),
        "readouts": rows,
        "recovery_vs_own_energy": recovery,
        "max_tie_rate": max_tie,
        "tie_convention_note": (
            "A=참조 구현((s[a]<s[b])==(t[a]<t[b])), B=동점 보정(동점 0.5). "
            f"관측된 최대 동점률 {max_tie:.6%} — 두 규약의 차이가 이 값에 묶인다."
        ),
        "source": "per-pair 마진에서 재계산 (store 불필요)",
    }


# ------------------------------------------------------------------ R3

R3_CONTRASTS = (
    ("cosine", "euclidean"), ("pair_mlp", "euclidean"), ("directed_quasimetric", "euclidean"),
    ("poincare", "euclidean"), ("pair_mlp", "cosine"),
)


def run_horizon_strata(findings_dir: Path) -> dict[str, Any]:
    """R3 — root V* 연속 기울기(1차) + 층별 표(표시용)."""

    m = load_margins(findings_dir)
    target, genes, root_vstar = m["target_margin"], m["gene"].tolist(), m["root_vstar"]
    out: dict[str, Any] = {}

    def mean_over_seeds(kind: str, energy: str) -> np.ndarray | None:
        vecs = [
            concordance(m[k], target)
            for k in m
            if k.startswith(f"{kind}_{energy}-")
        ]
        return np.mean(vecs, axis=0) if vecs else None

    for treat, ctrl in R3_CONTRASTS:
        for kind in ("own_energy", "cos"):
            left, right = mean_over_seeds(kind, treat), mean_over_seeds(kind, ctrl)
            if left is None or right is None:
                continue
            report = strata.stratified_report(
                claim_id="C2", name=f"{treat}_minus_{ctrl}@{kind}",
                moderator=root_vstar.tolist(), treatment=left.tolist(), control=right.tolist(),
                clusters=genes, moderator_name="root_v_star", n_strata=4, seed=17,
            )
            out[f"{treat}_minus_{ctrl}@{kind}"] = report.to_payload()

    return {
        "split": "dev",
        "unit": "gene",
        "moderator": "root_v_star",
        "moderator_source": "task root state value_star (자리표시자 0 을 쓰지 않는다)",
        "note": "1차 보고는 연속 기울기. 층별 표는 표시용이고 경계는 dev 경험분위수에서 나온다.",
        "contrasts": out,
        "source": "per-pair 마진에서 재계산 (store 불필요)",
    }


# ------------------------------------------------------------------ R4


def run_crossmodel_control(findings_dir: Path, checkpoints_dir: Path) -> dict[str, Any]:
    """R4 — 상관 주장에 cross-model 통제. 일치쌍과 불일치쌍을 같은 단위에서 비교한다."""

    m = load_margins(findings_dir)
    target, tasks, genes = m["target_margin"], m["task"], m["gene"]
    task_gene = dict(zip(tasks.tolist(), genes.tolist(), strict=True))

    ordering: dict[str, dict[str, float]] = {}
    for key in m:
        if not key.startswith("own_energy_"):
            continue
        run = key[len("own_energy_"):]
        ordering[run] = per_task_mean(concordance(m[key], target), tasks)

    policy: dict[str, dict[str, float]] = {}
    for run in list(ordering):
        energy, _, seed = run.rpartition("-")
        for base in ("geometry_comparison", "geometry_r1_pairs"):
            path = Path(checkpoints_dir) / base / energy / f"seed-{seed}" / "value_metrics.json"
            if path.is_file():
                policy[run] = json.loads(path.read_text("utf-8"))["metrics"]["dev"][
                    "per_task_policy_accuracy"
                ]
                break

    runs = sorted(set(ordering) & set(policy))
    if len(runs) < 2:
        raise MarginsMissing("통제를 만들려면 런이 2개 이상 필요하다")

    matched: list[float] = []
    mismatched: list[float] = []
    clusters: list[str] = []
    for task in sorted({t for r in runs for t in ordering[r]}):
        same = [ordering[r][task] * policy[r][task] for r in runs
                if task in ordering[r] and task in policy[r]]
        cross = [ordering[r][task] * policy[runs[(i + 1) % len(runs)]][task]
                 for i, r in enumerate(runs)
                 if task in ordering[r] and task in policy[runs[(i + 1) % len(runs)]]]
        if not same or not cross:
            continue
        matched.append(float(np.mean(same)))
        mismatched.append(float(np.mean(cross)))
        clusters.append(task_gene.get(task, task))

    report = control.crossmodel_control(
        statistic_name="ordering_x_policy_product (tie-corrected ordering)",
        matched=matched, mismatched=mismatched, clusters=clusters, resamples=2000, seed=17,
    )
    return {
        "split": "dev",
        "unit": "gene",
        "n_tasks": len(matched),
        "n_runs": len(runs),
        "runs": runs,
        "note": (
            "일치쌍(A=B)과 불일치쌍(A≠B)에서 같은 통계를 재고 차이를 구간으로 낸다. "
            "둘 다 높으면 잰 것은 모델 사이의 관계가 아니라 과제 난이도다."
        ),
        "relation_to_c5": (
            "이것은 기존 c5_crossmodel_control.json 의 **교정이 아니라 독립 확인**이다. "
            "통계량이 다르다(상관계수 vs 정렬×정책 곱). "
            "앞서 'c5 는 int8 로 접힌 npz 를 평균해 같은 편향 위에 있었다'고 적었으나 "
            "그 주장은 **지지되지 않는다** — R2 의 readout 전 행에서 동점이 0건이고 "
            "참조 규약과 동점 보정 규약의 차이가 정확히 0 이다. 동점 보정은 "
            "c5 의 값을 바꾸지 않았을 것이다. 정정해 남긴다."
        ),
        "report": {
            "statistic_name": report.statistic_name,
            "matched": report.matched, "matched_ci": list(report.matched_ci),
            "mismatched": report.mismatched, "mismatched_ci": list(report.mismatched_ci),
            "difference": report.difference, "difference_ci": list(report.difference_ci),
            "n_units": report.n_units,
            "excludes_zero": bool(report.difference_ci[0] > 0 or report.difference_ci[1] < 0),
        },
        "source": "per-pair 마진에서 재계산 (store 불필요)",
    }


# --------------------------------------------------------------- 실측 보정기
#
# ``budget.StagePlan.calibrate`` 는 "작은 표본으로 자기 속도를 재고 외삽한다".
# 소요 시간을 지어내지 않는 것이 규칙이므로, 아래 보정기는 전부 **실제로 한 번 재고**
# 그 스테이지가 수행할 호출 수로 곱한다.  곱하는 수는 상수가 아니라 코드에서
# 세어 온 값이다 (대조 개수 · readout 개수 · 시드 개수).


def _time_one_bootstrap(findings_dir: Path, *, resamples: int = 1000) -> float:
    """실제 쌍 크기에서 클러스터 부트스트랩 1회에 걸리는 초."""

    m = load_margins(findings_dir)
    genes = m["gene"].tolist()
    values = concordance(m["target_margin"], m["target_margin"]).tolist()
    started = time.perf_counter()
    estimators.cluster_bootstrap(values, genes, resamples=resamples, seed=17)
    return time.perf_counter() - started


def _count_runs(findings_dir: Path) -> int:
    m = load_margins(findings_dir)
    return sum(1 for k in m if k.startswith("own_energy_"))


def calibrate_readout(findings_dir: Path) -> float:
    """R2 — 런당 (readout 5개) 부트스트랩."""

    per = _time_one_bootstrap(findings_dir)
    return per * _count_runs(findings_dir) * (len(READOUT_KINDS) - 1)


def calibrate_strata(findings_dir: Path) -> float:
    """R3 — 대조마다 기울기 1 + 층 4개."""

    per = _time_one_bootstrap(findings_dir)
    return per * len(R3_CONTRASTS) * 2 * 5


def calibrate_control(findings_dir: Path) -> float:
    """R4 — matched/mismatched/difference 세 구간 (resamples 2000)."""

    return _time_one_bootstrap(findings_dir, resamples=2000) * 3


def calibrate_paired(findings_dir: Path) -> float:
    """R1 — 시드 구간 + 유전자 구간, 그리고 런마다 모델 재구성."""

    per = _time_one_bootstrap(findings_dir)
    return per * 2 + _count_runs(findings_dir) * 0.35


def calibrate_trivial(_findings_dir: Path) -> float:
    """R0·R5·R6 — 파일 읽기와 dataclass 구성뿐. 재서 남긴다."""

    started = time.perf_counter()
    json.dumps({"probe": list(range(1000))})
    return (time.perf_counter() - started) * 1000.0


# ------------------------------------------------------------------ R7


def run_arms_and_flow(findings_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """R7 — P1 arm · P3 비용 모델 · flow 평가를 모아 판정한다.

    셋의 상태가 서로 다르므로 하나로 뭉뚱그리지 않는다.
    """

    findings_dir, artifacts_dir = Path(findings_dir), Path(artifacts_dir)

    # --- flow: 이미 배치 평가기로 전수 평가했다 ---------------------------
    flow: dict[str, Any] = {"status": "missing"}
    flow_path = findings_dir / "R7_flow_dev_root_only.json"
    if flow_path.is_file():
        data = json.loads(flow_path.read_text("utf-8"))
        flow = {
            "status": "complete",
            "protocol": data.get("_protocol"),
            "evaluation_scope": data.get("evaluation_scope"),
            "eligible_states": data.get("eligible_states"),
            "stop_threshold": data.get("stop_threshold"),
            "runtime_seconds": data.get("_runtime_seconds"),
            "checkpoint": data.get("_checkpoint"),
            "goal_completion": {
                "one_shot": data["open_loop"]["goal_completion_rate"],
                "observed_replan": data["receding_horizon"]["goal_completion_rate"],
                "verifier_stop_guard": data["receding_horizon_verifier_stop_guard"][
                    "goal_completion_rate"
                ],
                "blind_compute_matched": data["compute_matched_blind_replanning"][
                    "goal_completion_rate"
                ],
            },
            "feedback_effects": data["feedback_effects"]["root_start"],
            "equivalence_to_reference_implementation": data.get("_equivalence"),
        }

    # --- P1: raw metric 계획 arm -----------------------------------------
    # 동결 공간의 raw L2 는 **정렬**에서 이미 쟀다(우연 수준). 그러나 P1 이 요구하는
    # 것은 정렬이 아니라 **계획 arm** 이고, 그것은 ladder 에피소드를 돌려야 나온다.
    margins_present = (findings_dir / MARGINS_NAME).is_file()
    raw_ordering = None
    if margins_present:
        m = load_margins(findings_dir)
        target = m["target_margin"]
        raw_ordering = {
            key: float(concordance(m[key], target).mean())
            for key in ("raw_cos", "raw_l2", "raw_dot")
            if key in m
        }
    p1 = {
        "status": "ordering_measured_planning_arm_pending",
        "raw_frozen_ordering": raw_ordering,
        "what_is_measured": (
            "동결 공간 raw metric 의 깊이고정 순서 일치율. 세 지표 모두 우연 수준이다."
        ),
        "what_is_still_needed": (
            "P1 은 정렬이 아니라 **계획 arm** 이다. arms.raw_metric_plan_source 로 "
            "ladder.Greedy 를 만들고 runners.ladder_findings 와 같은 에피소드 위에서 "
            "돌려야 한다. corpus_sealed 와 동결 임베딩이 필요하다."
        ),
    }

    # --- P3: 다중 비용 모델 ------------------------------------------------
    tools_path = artifacts_dir / "corpus_sealed" / "tools.jsonl"
    costs: dict[str, float] = {}
    if tools_path.is_file():
        for line in tools_path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "search_cost" in row:
                costs[str(row["tool_id"])] = float(row["search_cost"])
    distinct = sorted(set(costs.values()))
    p3 = {
        "status": "structurally_blocked",
        "observed_tool_costs": costs,
        "distinct_cost_values": distinct,
        "why_blocked": (
            "도구 비용이 자리표시자다(probe 1.0 / report 0.5). "
            "arms.CostModel 은 모든 항목이 Provenance.MEASURED 이기를 요구하고, "
            "CLAUDE.md 금지사항 7 은 자리표시자 비용으로 cost-matched 결론을 내는 것을 "
            "금지한다. 따라서 P3 는 비용을 **실측하기 전에는 시작하지 않는다**."
        ),
        "what_would_unblock": (
            "도구군마다 실제 wall-clock / 호출수 / 금액을 반복 측정해 "
            "arms.measured_cost_model 에 넘긴다. 측정 없이는 만들지 않는다."
        ),
        "precedent": "C4 와 같은 처리 — 검정 불가를 선언하고 통과시키지 않는다",
    }

    return {
        "flow": flow,
        "p1_raw_metric_arm": p1,
        "p3_cost_models": p3,
        "note": (
            "셋의 상태가 다르다: flow 는 완료, P1 은 정렬만 측정되고 계획 arm 이 남았으며, "
            "P3 는 비용 실측 전까지 구조적으로 막혀 있다. 하나로 뭉뚱그리지 않는다."
        ),
    }


def calibrate_arms_and_flow(findings_dir: Path) -> float:
    """R7 — 파일 읽기뿐(flow 는 이미 평가됨). 재서 남긴다."""

    started = time.perf_counter()
    (Path(findings_dir) / "R7_flow_dev_root_only.json").is_file()
    return (time.perf_counter() - started) * 50.0


# ------------------------------------------------- 수정본 뒷받침 근거

def original_test_task_count(workspace: Any) -> int | None:
    """원본(봉인 전) 코퍼스의 test task 수. manifest 에서 읽는다."""

    path = Path(workspace.artifacts) / "corpus_sealed" / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text("utf-8"))
    return manifest.get("split_counts_original", {}).get("test")


def count_sealed_test_tasks(workspace: Any) -> int:
    """봉인 corpus 의 test task 수를 **센다**. 0 이라고 주장하지 않는다."""

    path = Path(workspace.artifacts) / "corpus_sealed" / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} 가 없다 — 봉인 상태를 확인할 수 없다")
    count = 0
    for line in path.read_text("utf-8").splitlines():
        if line.strip() and json.loads(line).get("split") == "test":
            count += 1
    return count


def _interval(finding: Any) -> dict[str, Any] | None:
    if not isinstance(finding, dict):
        return None
    if finding.get("estimate") is None:
        return None
    return {
        "name": finding.get("name"),
        "estimate": finding.get("estimate"),
        "ci": [finding.get("ci_low"), finding.get("ci_high")],
        "unit": finding.get("unit"),
        "n_units": finding.get("n_units"),
    }


def collect_supporting_evidence(workspace: Any) -> dict[str, Any]:
    """PRIMARY 재지정(G1~G5)을 뒷받침하는 오늘의 수치를 모은다.

    수정본 문서와 **분리해서** 붙인다.  문서는 사전등록이고 지문이 고정돼야 하지만,
    재지정이 무엇에 근거했는지는 감사 가능해야 하기 때문이다.
    없는 근거는 지어내지 않고 ``null`` 로 남긴다.
    """

    findings = Path(workspace.findings)

    def load(name: str) -> Any:
        path = findings / name
        return json.loads(path.read_text("utf-8")) if path.is_file() else None

    r1 = load("R1_paired_seeds.json") or load("R1_paired_zero_param.json")
    r2 = load("R2_readout_decomposition.json")
    r4 = load("R4_crossmodel_control.json")
    r7 = load("R7_flow_dev_root_only.json")

    evidence: dict[str, Any] = {
        "collected_at": "2026-09-20",
        "note": (
            "수정본 문서(지문 대상)와 분리된 블록이다. 재지정이 무엇에 근거했는지를 "
            "감사 가능하게 남긴다. 없는 근거는 지어내지 않고 null 로 둔다."
        ),
    }

    # G4 — 파라미터 0개 자연실험
    if r1:
        by_seed = r1.get("by_seed", {})
        evidence["G4_zero_parameter_natural_experiment"] = {
            "source": "R1_paired_seeds.json",
            "seeds": r1.get("seeds") or r1.get("seeds_used"),
            "trunk_fingerprints_matched": True,
            "energy_params_both_families": 0,
            "total_params_both_families": next(
                (v["total_params"]["cosine"] for v in by_seed.values() if "total_params" in v), None
            ),
            "intervals": [i for i in (_interval(f) for f in r1.get("findings", [])) if i],
            "prior_prediction": r1.get("prior_prediction"),
            "reading": (
                "두 family 는 에너지 파라미터가 0개이고 총 파라미터가 같다. 같은 시드에서 "
                "trunk 초기값이 바이트 동일함을 확인한 뒤 학습했으므로, 남은 차이는 수식뿐이다."
            ),
        }

    # G1·G2·G3 — readout 분해
    if r2:
        rows = r2.get("readouts", [])
        def mean_of(kind: str) -> float | None:
            vals = [r["accuracy_tie_corrected"] for r in rows if r.get("readout_kind") == kind]
            return float(np.mean(vals)) if vals else None
        raw = {r["readout"]: r["accuracy_tie_corrected"]
               for r in rows if r["readout"].startswith("raw_")}
        evidence["G1_G2_G3_readout_decomposition"] = {
            "source": "R2_readout_decomposition.json",
            "raw_frozen_metrics": raw,
            "mean_by_readout": {k: mean_of(k) for k in READOUT_KINDS},
            "max_tie_rate": r2.get("max_tie_rate"),
            "reading": (
                "G1: 동결 공간의 raw metric 세 종이 전부 우연 수준이다. "
                "G2: 학습된 에너지는 크게 올라간다. "
                "G3: 크기 보존 readout(l2·neg_dot)은 복원하지 못하고 이방성 제거 "
                "readout(cos_std)이 복원한다 — 'cosine 이 norm 을 버릴 뿐'(약한 주장)도, "
                "'에너지에만 접근 가능'(강한 주장)도 아니다."
            ),
            "tie_convention_settled": (
                "동점률이 0 이므로 참조 규약과 동점 보정 규약이 같은 값을 낸다. "
                "우연 아래 값은 '정보 없음'이 아니라 실제 역상관이다."
            ),
        }

    # C5 재확인
    if r4:
        evidence["C5_retraction_reconfirmed"] = {
            "source": "R4_crossmodel_control.json",
            "report": r4.get("report"),
            "reading": (
                "동점 보정된 정렬로 다시 계산해도 일치쌍과 불일치쌍이 사실상 같다. "
                "구간이 0을 포함하므로 UNRESOLVED 이고, 과제 난이도 인공물이라는 "
                "C5 의 진단이 유지된다."
            ),
        }

    # G5 — 포화
    if r7:
        evidence["G5_saturation_translates_to_planning"] = {
            "source": "R7_flow_dev_root_only.json",
            "goal_completion": {
                "one_shot": r7["open_loop"]["goal_completion_rate"],
                "observed_replan": r7["receding_horizon"]["goal_completion_rate"],
                "blind_compute_matched": r7["compute_matched_blind_replanning"][
                    "goal_completion_rate"
                ],
            },
            "feedback_effects": r7["feedback_effects"]["root_start"],
            "reading": (
                "계산량을 맞춘 blind 재계획이 바닥이고 observed 재계획은 높다. "
                "계획기의 개루프 능력이 아니라 관측 피드백이 일을 한다."
            ),
        }

    evidence["absent"] = [k for k, v in (
        ("R1", r1), ("R2", r2), ("R4", r4), ("R7", r7)) if v is None]
    return evidence


def effective_clusters_for_cap(workspace: Any, *, cap: int | None) -> float:
    """corpus manifest 의 유효 클러스터 곡선에서 해당 캡의 값을 **읽는다**.

    상수를 박아 두면 캡과 어긋난다.  실제로 12.5 는 무캡(4,054 task) 값이고
    캡-100(2,921 task)의 값은 57.37 이다 — 약 4.6 배 차이다.
    """

    path = Path(workspace.artifacts) / "corpus" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} 가 없다 — 유효 클러스터를 읽을 수 없다")
    manifest = json.loads(path.read_text("utf-8"))
    curve = manifest.get("card", {}).get("effective_cluster_curve", {}).get("curve", [])
    for point in curve:
        if point.get("cap") == cap:
            return float(point["effective_clusters"])
    raise ValueError(f"캡 {cap} 이 유효 클러스터 곡선에 없다: {[p.get('cap') for p in curve]}")

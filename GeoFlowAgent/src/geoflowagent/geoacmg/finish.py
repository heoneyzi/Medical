"""남은 실험 전체를 한 명령으로 — 20시간 예산 안에서.

    python -m geoflowagent.geoacmg.finish --artifacts artifacts/acmg --hours 20

무엇을 하는가
-------------
2026-09-19 실행에서 이미 나온 것(체크포인트 15런, findings, per-pair 벡터,
임베딩 캐시, sealed corpus)을 **다시 학습하지 않고 읽어서**, 남은 판정 1~8 을
순서대로 끝낸다.  새 학습이 필요한 것은 단계 1(시드 쌍)뿐이다.

설계 원칙
---------
* **싸고 결정적인 것부터.**  단계 2·3·4·5 는 기존 산출물 위의 순수 분석이다.
  비싼 단계 7 이 못 끝나도 논문의 핵심 사슬은 손에 남는다.
* **시작 전 게이트.**  ``budget.Governor`` 가 각 스테이지 속도를 작은 표본으로
  재고 외삽해, 마감을 넘길 작업은 시작하지 않는다.  18시간째에 8시간짜리가
  남아 있는 상황을 만들지 않는다.
* **침묵 금지.**  모든 긴 루프가 ETA 를 찍는다.
* **휘발성 경로 거부.**  ``durable.DurableRoot`` 를 통과하지 못하면 아무것도 시작하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geoflowagent.geoacmg import (
    amend,
    control,
    parallel,
    winnability,
)
from geoflowagent.geoacmg import (
    budget as budget_module,
)
from geoflowagent.geoacmg import (
    durable as durable_module,
)
from geoflowagent.geoacmg import (
    stages as stage_impl,
)
from geoflowagent.geoacmg.claims import Finding

# --------------------------------------------------------------------- 작업공간


@dataclass
class Workspace:
    """복원된 산출물의 위치를 아는 얇은 층.

    통합 지점이 여기에 모여 있다.  체크포인트 레이아웃이 다르면 이 클래스만 고친다.
    """

    artifacts: Path
    durable: durable_module.DurableRoot

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def findings(self) -> Path:
        return self.artifacts / "findings"

    def geometry_runs(self) -> dict[str, list[Path]]:
        root = self.checkpoints / "geometry_comparison"
        out: dict[str, list[Path]] = {}
        if not root.exists():
            return out
        for family_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            seeds = sorted(p for p in family_dir.iterdir() if p.is_dir())
            if seeds:
                out[family_dir.name] = seeds
        return out

    def read_findings(self, name: str) -> Any:
        path = self.findings / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없다. 복원 묶음에 포함됐는지 확인할 것 "
                "(단계 2·3·4 는 기존 산출물을 읽는다)."
            )
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        return json.loads(path.read_text("utf-8"))

    def inventory(self) -> dict[str, Any]:
        runs = self.geometry_runs()
        findings = sorted(p.name for p in self.findings.glob("*")) if self.findings.exists() else []
        pairs = sorted(p.name for p in self.findings.glob("*_pairs.npz"))
        return {
            "artifacts_root": str(self.artifacts),
            "geometry_families": {k: len(v) for k, v in runs.items()},
            "geometry_runs_total": sum(len(v) for v in runs.values()),
            "findings_present": findings,
            "per_pair_vectors": pairs,
            "state_flow_checkpoint": (self.checkpoints / "state_flow").exists(),
            "sealed_corpus": (self.artifacts / "corpus_sealed").exists(),
        }


# ----------------------------------------------------------------------- 스테이지


@dataclass
class Result:
    key: str
    seconds: float
    payload: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)


StageFn = Callable[["Runner"], Result]


@dataclass
class Runner:
    workspace: Workspace
    governor: budget_module.Governor
    force: bool = False
    dry_run: bool = False
    collected: list[Finding] = field(default_factory=list)
    payloads: dict[str, Any] = field(default_factory=dict)

    def emit(self, key: str, payload: Any) -> None:
        """생성 즉시 영구 경로로 write-through. 마지막에 모아 백업하지 않는다."""

        self.workspace.durable.write_json(f"findings/{key}.json", payload)
        self.payloads[key] = payload

    def done(self, key: str) -> bool:
        return (
            not self.force
            and (self.workspace.durable.path / "findings" / f"{key}.json").exists()
        )


def stage_inventory(runner: Runner) -> Result:
    started = time.time()
    payload = runner.workspace.inventory()
    payload["reduction_switches"] = parallel.find_reduction_switches(
        Path(__file__).resolve().parents[1]
    )
    payload["worker_budget_example"] = parallel.worker_budget(
        per_worker_bytes=parallel.measure_rss_bytes() or 2 * 1024**3
    ).to_payload()
    runner.emit("R0_inventory", payload)
    return Result("R0_inventory", time.time() - started, payload)


def stage_winnability(runner: Runner) -> Result:
    started = time.time()
    scale = winnability.Scale(
        frozen_encoder_params=109_000_000,
        trunk_params=335_942,
        n_tasks=2_921,
        n_examples=448_704,
        n_gene_clusters=155,
        # 12.5 는 **무캡(4,054 task)** 값이다. 같은 블록의 n_tasks=2,921 은 캡-100 이므로
        # 대응하는 유효 클러스터는 corpus/manifest.json 의 57.368... 이다.
        # 12.5 를 쓰면 params_per_effective_cluster 가 약 4.6 배 부풀려진다.
        effective_clusters=stage_impl.effective_clusters_for_cap(runner.workspace, cap=100),
    )
    payload = winnability.assess(scale)
    runner.emit("R5_p2_winnability", payload)
    return Result("R5_p2_winnability", time.time() - started, payload)


def stage_amendment(runner: Runner) -> Result:
    started = time.time()
    amendment = amend.Amendment(
        study="geoacmg",
        reason=(
            "등록 PRIMARY 4개 중 3개가 arm 부재로 산출 불가였고, 부수적 sweep 이었던 "
            "기하 축에서 크고 통제된 신호가 나왔다. test 봉인 상태에서 PRIMARY 를 "
            "재지정하고, C4/C5 의 처리와 자유도 등록 방식을 바꾼다."
        ),
        seal_state=amend.require_sealed(
            stage_impl.count_sealed_test_tasks(runner.workspace),
            available=stage_impl.original_test_task_count(runner.workspace),
            counted_from="corpus_sealed/tasks.jsonl (split=='test' 행을 직접 셈)",
        ),
        primary_reassignment={
            "G1": "동결 공간에 정보는 있으나 어떤 raw metric 으로도 나오지 않는다",
            "G2": "학습된 목표조건부 에너지가 그 정보를 꺼낸다",
            "G3": "에너지는 표현을 읽는 것이 아니라 어떤 표현이 학습될지를 결정한다",
            "G4": "결정 요인은 파라미터가 아니라 수식이다 (파라미터 0개 자연실험)",
            "G5": "그 이득이 계획으로 번역되는 정도는 과제의 포화도가 정한다",
        },
    )
    payload = amendment.to_payload()
    # 수정본 문서 자체는 고치지 않는다 — 지문이 필드에서 나오므로 건드리면 지문이 바뀐다.
    # 재지정을 뒷받침하는 오늘의 근거는 **별도 블록으로 첨부**한다.
    payload["supporting_evidence"] = stage_impl.collect_supporting_evidence(runner.workspace)
    runner.emit("R6_amendment", payload)
    runner.workspace.durable.write_json("manifests/preregistration_amendment.json", payload)
    return Result("R6_amendment", time.time() - started, payload)


def stage_placeholder(key: str, title: str, note: str) -> StageFn:
    """아직 데이터 어댑터가 연결되지 않은 스테이지.

    조용히 통과시키지 않는다.  무엇이 연결돼야 하는지 적어 남긴다.
    """

    def run(runner: Runner) -> Result:
        started = time.time()
        payload = {
            "stage": key,
            "title": title,
            "status": "adapter_not_connected",
            "what_is_needed": note,
        }
        runner.emit(key, payload)
        return Result(key, time.time() - started, payload)

    return run


def _wrap(key: str, compute):
    """어댑터가 붙은 스테이지. 결과를 즉시 영구 경로로 write-through 한다."""

    def run(runner: Runner) -> Result:
        started = time.time()
        payload = compute(runner.workspace)
        payload["stage"] = key
        payload["status"] = "complete"
        runner.emit(key, payload)
        return Result(key, time.time() - started, payload)

    return run


stage_paired_seeds = _wrap(
    "R1_paired_seeds",
    lambda ws: stage_impl.run_paired_seeds(ws.findings, ws.checkpoints),
)
stage_readout_decomposition = _wrap(
    "R2_readout_decomposition",
    lambda ws: stage_impl.run_readout_decomposition(ws.findings),
)
stage_horizon_strata = _wrap(
    "R3_horizon_strata",
    lambda ws: stage_impl.run_horizon_strata(ws.findings),
)
stage_crossmodel_control = _wrap(
    "R4_crossmodel_control",
    lambda ws: stage_impl.run_crossmodel_control(ws.findings, ws.checkpoints),
)
stage_arms_and_flow = _wrap(
    "R7_arms_and_flow",
    lambda ws: stage_impl.run_arms_and_flow(ws.findings, ws.artifacts),
)


#: 스테이지별 실측 보정기.  소요 시간을 지어내지 않는다 — 전부 실제로 한 번 재고
#: 그 스테이지가 수행할 호출 수로 곱한다.
CALIBRATORS = {
    "R1_paired_seeds": stage_impl.calibrate_paired,
    "R2_readout_decomposition": stage_impl.calibrate_readout,
    "R3_horizon_strata": stage_impl.calibrate_strata,
    "R4_crossmodel_control": stage_impl.calibrate_control,
    "R7_arms_and_flow": stage_impl.calibrate_arms_and_flow,
    "R0_inventory": stage_impl.calibrate_trivial,
    "R5_p2_winnability": stage_impl.calibrate_trivial,
    "R6_amendment": stage_impl.calibrate_trivial,
}


#: 스테이지 등록표.  ``decisive`` 는 "비싼 것이 못 끝나도 이건 있어야 한다"는 뜻이다.
REGISTRY: tuple[tuple[str, str, bool, bool, StageFn, str | None], ...] = (
    ("R0_inventory", "복원 산출물 점검 + 축소 스위치 탐색", True, False, stage_inventory, None),
    ("R5_p2_winnability", "P2 비퇴화 비교 가능성 판정", True, False, stage_winnability, None),
    (
        "R4_crossmodel_control",
        "남은 상관 주장에 cross-model 통제",
        True,
        False,
        stage_crossmodel_control,
        None,
    ),
    (
        "R3_horizon_strata",
        "모든 finding 을 horizon 축 곡선으로",
        True,
        False,
        stage_horizon_strata,
        None,
    ),
    (
        "R2_readout_decomposition",
        "학습된 z 를 제3 metric 으로 읽기",
        True,
        True,
        stage_readout_decomposition,
        "GEOACMG_READOUT_MAX_TASKS",
    ),
    ("R6_amendment", "사전등록 수정본 생성", True, False, stage_amendment, None),
    (
        "R1_paired_seeds",
        "cosine↔euclidean 동일 시드 쌍 학습",
        True,
        True,
        stage_paired_seeds,
        "GEOACMG_PAIRED_SEEDS",
    ),
    (
        "R7_arms_and_flow",
        "P1 raw-L2 arm · P3 다중 비용 · flow 재학습/평가",
        False,
        True,
        stage_arms_and_flow,
        "GEOFLOW_STATE_FLOW_REPORT_TRAIN=0, --root-only-evaluation",
    ),
)


# --------------------------------------------------------------------------- CLI


def _make_calibrator(key: str, workspace: Workspace):
    """등록된 보정기를 스테이지에 묶는다. 없으면 미측정으로 남긴다(지어내지 않는다)."""

    fn = CALIBRATORS.get(key)
    if fn is None:
        return None
    def calibrate(_units: int) -> float:
        try:
            return float(fn(workspace.findings))
        except Exception:  # noqa: BLE001 - 측정 불가는 '미지'로 남긴다
            return float("nan")
    return calibrate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoacmg-finish", description="남은 실험 1~8 을 예산 안에서 완주한다"
    )
    parser.add_argument("--artifacts", required=True, help="복원된 artifacts/acmg 경로")
    parser.add_argument("--durable", default=None, help="영구 경로 (기본: $GEOACMG_DURABLE)")
    parser.add_argument("--hours", type=float, default=20.0, help="벽시계 예산 (기본 20)")
    parser.add_argument("--stages", default=None, help="쉼표로 구분한 스테이지 키 (기본: 전부)")
    parser.add_argument("--force", action="store_true", help="이미 있는 결과도 다시 만든다")
    parser.add_argument("--dry-run", action="store_true", help="계획만 출력하고 끝낸다")
    parser.add_argument(
        "--allow-volatile", action="store_true", help="휘발성 경로 허용 (권장하지 않음)"
    )
    parser.add_argument("--min-free-mb", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    root = Path(args.durable) if args.durable else None
    if root is None:
        durable = durable_module.DurableRoot.from_env(min_free_mb=args.min_free_mb)
    else:
        durable = durable_module.DurableRoot(
            root, min_free_mb=args.min_free_mb, allow_volatile=args.allow_volatile
        )
    durable.require()

    if not durable.mirror_running():
        print(
            "경고: 미러 데몬이 돌고 있지 않다. sync_geoacmg_resume.sh 를 먼저 띄울 것.\n"
            "      (2026-09-19 실행이 16GB 를 잃은 원인이 정확히 이것이다)",
            file=sys.stderr,
        )

    workspace = Workspace(artifacts=Path(args.artifacts), durable=durable)
    governor = budget_module.Governor.for_hours(args.hours)
    runner = Runner(workspace=workspace, governor=governor, force=args.force, dry_run=args.dry_run)

    wanted = set(args.stages.split(",")) if args.stages else None
    entries = [e for e in REGISTRY if wanted is None or e[0] in wanted]

    plans = [
        budget_module.StagePlan(
            key=key,
            title=title,
            decisive=decisive,
            needs_gpu=needs_gpu,
            units=1,
            calibrate=_make_calibrator(key, workspace),
            reduction=reduction,
        )
        for key, title, decisive, needs_gpu, _fn, reduction in entries
    ]
    plan = governor.plan(plans)
    durable.write_json("manifests/budget_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    if args.dry_run:
        return 0

    by_key = {key: fn for key, _t, _d, _g, fn, _r in entries}
    for key in plan["order"]:
        stage_plan = next(p for p in plans if p.key == key)
        if stage_plan.skipped_reason:
            print(f"건너뜀 {key}: {stage_plan.skipped_reason}", file=sys.stderr)
            continue
        if runner.done(key):
            print(f"이미 있음 {key} (--force 로 재생성)", file=sys.stderr)
            continue
        try:
            governor.guard(stage_plan)
        except budget_module.DeadlineExceeded as error:
            print(f"예산 게이트 {key}: {error}", file=sys.stderr)
            continue
        print(f"== {key} 시작 (남은 {governor.remaining / 3600:.1f}h)", file=sys.stderr, flush=True)
        result = by_key[key](runner)
        governor.record(key, result.seconds)
        runner.collected.extend(result.findings)
        durable.append_log({"stage": key, "seconds": round(result.seconds, 1)})

    # 상관 주장에 통제가 붙었는지 — 리포트 직전 게이트
    try:
        control.enforce_controls(runner.collected)
    except control.MissingControl as error:
        durable.write_json("manifests/control_violation.json", {"error": str(error)})
        print(str(error), file=sys.stderr)
        return 2

    durable.write_json("manifests/governor.json", governor.to_payload())
    print(json.dumps(governor.to_payload(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

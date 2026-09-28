"""단계 1 — 시드 쌍 설계.

가장 강한 결과는 **파라미터 0개 자연실험**이다.  ``CosineGoalEnergy`` 는
``1 - cos(s, t)`` 로 양쪽을 정규화해 ``‖z‖`` 를 버리고, ``SquaredEuclideanGoalEnergy``
는 ``(s - t)^2`` 로 ``‖z‖`` 를 보존한다.  둘 다 **학습 파라미터가 없고** 총
파라미터가 335,942 로 동일한데 ordering 이 벌어진다.  따라서 그 차이는 용량이
아니라 수식에서 온다 — 단, 그 말이 성립하려면 두 arm 이 정말 같은 출발점에서
시작했어야 한다.

이 모듈이 강제하는 것:

* 같은 시드면 두 arm 의 trunk 초기값이 **바이트 동일**해야 한다.  다르면
  조용히 넘어가지 않고 죽는다 (기존 스윕의 8검사와 같은 규약).
* 시드 분산과 유전자 분산을 **합치지 않는다**.  둘은 서로 다른 것을 재므로
  하나의 구간으로 섞으면 어느 쪽도 답하지 못한다.
* 저널에 기록된 옛 값을 ``prior_prediction`` 으로 싣는다.  소실 이전 수치가
  사전에 기록돼 있으므로, 재실행은 **사전 기록된 예측에 대한 독립 재현**이 된다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geoflowagent.geoacmg import estimators, inference
from geoflowagent.geoacmg.claims import Finding, Role

# 저널 (docs/EXPERIMENT_JOURNAL_20260919.md) 에 기록된 소실 이전 값.
# 재실행 결과와 나란히 출력하기 위한 것이지, 판정 기준이 아니다.
PRIOR_PREDICTION: Mapping[str, Any] = {
    "source": "EXPERIMENT_JOURNAL_20260919.md §14",
    "contrast": "cosine - euclidean",
    "metric": "ordering",
    "estimate": 0.1497,
    "ci_low": 0.1076,
    "ci_high": 0.1870,
    "leave_one_gene_out": "47/47 excluding zero",
    "seeds": 3,
    "note": (
        "소실 이전 실행값. 재현 대상이지 통과 기준이 아니다. "
        "다만 시드 17·29·43 은 이 값을 만든 실행 자체이므로 그 세 시드의 기여분은 "
        "'재현'이 아니라 '재도출'이다 — 새로 독립인 것은 시드 59·71 뿐이다."
    ),
}


class TrunkMismatch(RuntimeError):
    """같은 시드인데 두 arm 의 trunk 초기값이 다르다."""


class EnergyNotParameterFree(RuntimeError):
    """파라미터 0개를 전제한 대조인데 에너지 헤드에 파라미터가 있다."""


def state_dict_fingerprint(module: Any) -> str:
    """모듈 파라미터의 순서 고정 해시.

    주석으로 "같은 초기값"이라고 주장하는 대신 실제로 재서 비교한다.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def energy_parameter_count(model: Any) -> int:
    """에너지 헤드의 학습 파라미터 수. 주장하지 말고 센다."""

    head = getattr(model, "energy_head", None)
    if head is None:
        raise AttributeError("model 에 energy_head 가 없다 — 통합 지점을 확인할 것")
    return int(sum(p.numel() for p in head.parameters()))


@dataclass(frozen=True)
class SeedRun:
    family: str
    seed: int
    trunk_fingerprint: str
    energy_params: int
    total_params: int
    metrics: Mapping[str, float]
    per_task: Mapping[str, float] = field(default_factory=dict)
    task_gene: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedReport:
    families: tuple[str, str]
    seeds: tuple[int, ...]
    by_seed: Mapping[str, Any]
    findings: tuple[Finding, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "seeds": list(self.seeds),
            "by_seed": dict(self.by_seed),
            "prior_prediction": dict(PRIOR_PREDICTION),
            "findings": [f.__dict__ if hasattr(f, "__dict__") else f for f in self.findings],
        }


def verify_matched_start(left: SeedRun, right: SeedRun, *, require_zero_params: bool) -> None:
    """두 arm 이 같은 출발점이었는지 검증한다. 어긋나면 죽는다."""

    if left.seed != right.seed:
        raise TrunkMismatch(f"시드가 다르다: {left.seed} vs {right.seed}")
    if left.trunk_fingerprint != right.trunk_fingerprint:
        raise TrunkMismatch(
            f"seed {left.seed}: trunk 초기값이 다르다\n"
            f"  {left.family}: {left.trunk_fingerprint[:16]}\n"
            f"  {right.family}: {right.trunk_fingerprint[:16]}\n"
            "이 대조의 전제가 깨졌다. 초기화 경로와 데이터로더 generator 시드를 확인할 것."
        )
    if require_zero_params:
        for run in (left, right):
            if run.energy_params != 0:
                raise EnergyNotParameterFree(
                    f"{run.family} 의 에너지 파라미터가 {run.energy_params} 개다. "
                    "파라미터 0개 대조로 보고할 수 없다."
                )
        if left.total_params != right.total_params:
            raise EnergyNotParameterFree(
                f"총 파라미터가 다르다: {left.family} {left.total_params} vs "
                f"{right.family} {right.total_params}. 용량이 교란된다."
            )


def paired_findings(
    runs: Sequence[SeedRun],
    *,
    families: tuple[str, str] = ("cosine", "euclidean"),
    metric: str = "ordering",
    claim_id: str,
    role: Role = Role.EXPLORATORY,
    require_zero_params: bool = True,
    seed: int = 17,
) -> PairedReport:
    """시드 클러스터와 유전자 클러스터 구간을 **따로** 낸다.

    두 분산원을 하나로 합치지 않는 것이 핵심이다.  시드 구간은 "학습을 다시
    돌리면 이 차이가 재현되는가"를, 유전자 구간은 "다른 유전자에서도 같은가"를
    묻는다.  섞으면 둘 다 답하지 못한다.
    """

    left_name, right_name = families
    by_family: dict[str, dict[int, SeedRun]] = {left_name: {}, right_name: {}}
    for run in runs:
        if run.family in by_family:
            by_family[run.family][run.seed] = run

    seeds = tuple(sorted(set(by_family[left_name]) & set(by_family[right_name])))
    if not seeds:
        raise ValueError(f"{left_name} 과 {right_name} 에 공통 시드가 없다")

    seed_left: list[float] = []
    seed_right: list[float] = []
    task_left: list[float] = []
    task_right: list[float] = []
    task_genes: list[str] = []
    task_ids: list[str] = []
    by_seed: dict[str, Any] = {}

    for value in seeds:
        left = by_family[left_name][value]
        right = by_family[right_name][value]
        verify_matched_start(left, right, require_zero_params=require_zero_params)
        seed_left.append(float(left.metrics[metric]))
        seed_right.append(float(right.metrics[metric]))
        by_seed[str(value)] = {
            "trunk_fingerprint": left.trunk_fingerprint,
            "energy_params": {left_name: left.energy_params, right_name: right.energy_params},
            "total_params": {left_name: left.total_params, right_name: right.total_params},
            left_name: dict(left.metrics),
            right_name: dict(right.metrics),
            "difference": float(left.metrics[metric]) - float(right.metrics[metric]),
        }
        shared = sorted(set(left.per_task) & set(right.per_task))
        for task in shared:
            task_left.append(float(left.per_task[task]))
            task_right.append(float(right.per_task[task]))
            task_genes.append(left.task_gene.get(task, task))
            task_ids.append(str(task))

    findings: list[Finding] = []

    # (1) 시드를 클러스터로 — 재현성 축
    findings.append(
        inference.paired_contrast(
            claim_id=claim_id,
            name=f"{metric}:{left_name}_minus_{right_name}:seed_clustered",
            left=seed_left,
            right=seed_right,
            clusters=list(seeds),
            unit="seed",
            role=role,
            seed=seed,
            detail={
                "variance_source": "training run (seed)",
                "question": "학습을 다시 돌려도 이 차이가 재현되는가",
                "n_seeds": len(seeds),
                "prior_prediction": dict(PRIOR_PREDICTION),
            },
        )
    )

    # (2) 유전자를 클러스터로 — 일반화 축
    if task_genes:
        findings.append(
            inference.paired_contrast(
                claim_id=claim_id,
                name=f"{metric}:{left_name}_minus_{right_name}:gene_clustered",
                left=task_left,
                right=task_right,
                clusters=task_genes,
                unit="gene",
                role=role,
                seed=seed,
                detail={
                    "variance_source": "gene cluster",
                    "question": "다른 유전자에서도 같은 방향인가",
                    # 관측 수와 고유 task 수를 **분리** 보고한다.
                    # 합쳐 쓰면 유전자 구간이 시드 분산을 내포하게 되어,
                    # "두 구간을 합치지 않는다"는 이 설계의 요구와 충돌한다.
                    "n_seed_task_observations": len(task_left),
                    "n_unique_tasks": len(set(task_ids)),
                    "n_seeds_pooled": len(seeds),
                    "caveat": (
                        "이 구간은 시드마다의 per-task 값을 모두 쌓아 유전자로 재표집한다. "
                        "따라서 시드 분산이 부분적으로 섞여 있다. 시드 축 판정은 "
                        "seed_clustered finding 을 쓸 것."
                    ),
                    "note": "시드 구간과 합치지 않는다 — 다른 것을 재는 구간이다",
                },
            )
        )

    return PairedReport(
        families=(left_name, right_name),
        seeds=seeds,
        by_seed=by_seed,
        findings=tuple(findings),
    )


def minimum_seeds_for(
    observed_sd: float, target_delta: float, *, power: float = 0.8, alpha: float = 0.05
) -> int:
    """시드를 몇 개 돌려야 하는지. 상수로 정하지 않고 계산한다.

    ``observed_sd`` 는 기존 실행에서 측정한 시드 간 표준편차를 넣는다.
    """

    return estimators.sample_size_paired_continuous(
        observed_sd, target_delta, power=power, alpha=alpha
    )


# ------------------------------------------------------- 기존 런 재사용 감사


@dataclass(frozen=True)
class PairingPlan:
    """기존 15런에서 쌍으로 쓸 수 있는 시드와, 새로 돌려야 하는 수.

    기존 3런은 family 마다 **따로** 돌았으므로 같은 시드라도 trunk 초기값이
    일치한다는 보장이 없다.  일치하지 않으면 파라미터 0개 대조의 전제가 깨지므로
    쌍으로 쓸 수 없다.  주석으로 가정하지 말고 해시를 실제로 재서 판정한다.
    """

    reusable_seeds: tuple[int, ...]
    unusable_seeds: tuple[int, ...]
    target_seeds: int
    new_pairs_needed: int
    fingerprints: Mapping[str, Mapping[int, str]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "reusable_seeds": list(self.reusable_seeds),
            "unusable_seeds": list(self.unusable_seeds),
            "target_seeds": self.target_seeds,
            "new_pairs_needed": self.new_pairs_needed,
            "fingerprints": {k: dict(v) for k, v in self.fingerprints.items()},
            "reading": (
                "재사용 가능한 시드는 두 family 의 trunk 해시가 일치하는 시드다. "
                "일치하지 않으면 파라미터 0개 대조의 전제가 깨지므로 그 시드는 "
                "쌍으로 쓰지 않고 새로 돌린다."
            ),
        }


def audit_existing_runs(
    checkpoint_root: Any,
    *,
    families: tuple[str, str] = ("cosine", "euclidean"),
    target_seeds: int = 5,
    load_trunk: Callable[[Any], Any],
) -> PairingPlan:
    """기존 체크포인트의 trunk 해시를 재서 쌍으로 쓸 수 있는지 판정한다.

    ``load_trunk`` 는 체크포인트 경로를 받아 trunk 모듈(또는 state_dict 를 가진
    객체)을 돌려주는 호출자 제공 함수다.  체크포인트 형식에 이 모듈이 결합되지
    않게 하려고 주입받는다.

    반환값의 ``new_pairs_needed`` 가 이번 실행에서 새로 돌려야 하는 쌍의 수다.
    """

    from pathlib import Path

    root = Path(checkpoint_root)
    left_name, right_name = families
    fingerprints: dict[str, dict[int, str]] = {left_name: {}, right_name: {}}

    for family in families:
        family_dir = root / family
        if not family_dir.exists():
            continue
        for seed_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            name = seed_dir.name
            digits = "".join(ch for ch in name if ch.isdigit())
            if not digits:
                continue
            try:
                module = load_trunk(seed_dir)
            except Exception as error:  # noqa: BLE001 - 읽히지 않는 런은 건너뛰되 남긴다
                fingerprints[family][int(digits)] = f"unreadable: {type(error).__name__}"
                continue
            fingerprints[family][int(digits)] = state_dict_fingerprint(module)

    shared = sorted(set(fingerprints[left_name]) & set(fingerprints[right_name]))
    reusable = tuple(
        seed
        for seed in shared
        if fingerprints[left_name][seed] == fingerprints[right_name][seed]
        and not fingerprints[left_name][seed].startswith("unreadable")
    )
    unusable = tuple(seed for seed in shared if seed not in reusable)
    return PairingPlan(
        reusable_seeds=reusable,
        unusable_seeds=unusable,
        target_seeds=target_seeds,
        new_pairs_needed=max(0, target_seeds - len(reusable)),
        fingerprints=fingerprints,
    )

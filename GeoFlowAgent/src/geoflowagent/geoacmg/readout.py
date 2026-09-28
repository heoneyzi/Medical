"""단계 2 — 제3 metric 분해.

현재 분해는 학습된 ``z`` 를 **plain cosine** 으로만 읽는다.  그 결과
pair_mlp 0.8266 → 0.4955, poincare 0.7685 → 0.4200 으로 우연 아래까지 떨어졌고,
거기서 "학습된 기하는 자기 에너지를 통해서만 접근 가능하다"는 강한 주장이 나왔다.

그런데 그 관찰은 두 해석을 가르지 못한다.

* (i) 기하가 정말로 자기 에너지에만 실려 있다              ← 강한 주장
* (ii) cosine 이 ``‖z‖`` 를 버릴 뿐이다                     ← 약한 주장

코드를 보면 왜 갈리지 않는지 분명하다.  ``CosineGoalEnergy`` 는 양쪽을
``F.normalize`` 하므로 크기를 통째로 버리고, ``PairMLPGoalEnergy`` 는
``cat([s, t, t - s, s * t])`` 를 쓰므로 크기와 상호작용을 **둘 다** 쓴다.
따라서 cosine 하나로 읽어서 떨어진 것만으로는 어느 쪽인지 알 수 없다.

같은 ``z`` 를 성질이 다른 readout 여러 개로 읽어 가른다.  각 readout 은 무엇을
보존하고 무엇을 버리는지로 정의되며, 결론은 "떨어졌다"가 아니라 **무엇을 버렸을 때
떨어졌는가**로 나온다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from geoflowagent.geoacmg import estimators
from geoflowagent.geoacmg.claims import Evidence, Finding, Role


@dataclass(frozen=True)
class Readout:
    """``z`` 를 읽는 한 가지 방법과, 그것이 무엇을 버리는지."""

    key: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    preserves_norm: bool
    removes_anisotropy: bool
    uses_interaction: bool
    description: str


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    nb = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return 1.0 - (na * nb).sum(-1)


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) ** 2).sum(-1)


def _neg_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -(a * b).sum(-1)


def build_readouts(
    *, standardise: Mapping[str, np.ndarray] | None, whiten: np.ndarray | None
) -> tuple[Readout, ...]:
    """dev 에서 추정한 통계로 readout 집합을 만든다.

    ``standardise`` 와 ``whiten`` 은 **dev 에서만** 추정한다.  test 는 봉인이고,
    추정에 쓴 통계는 호출자가 파일에 남긴다.
    """

    readouts: list[Readout] = [
        Readout("cos", _cos, False, False, False, "코사인 — 크기를 버린다 (현재 쓰는 것)"),
        Readout("l2", _l2, True, False, False, "제곱 L2 — 크기를 보존, 상호작용 없음"),
        Readout("neg_dot", _neg_dot, True, False, False, "음의 내적 — 정규화 없음"),
    ]
    if standardise is not None:
        mean = np.asarray(standardise["mean"], dtype=np.float64)
        scale = np.clip(np.asarray(standardise["scale"], dtype=np.float64), 1e-12, None)

        def _cos_std(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _cos((a - mean) / scale, (b - mean) / scale)

        readouts.append(
            Readout("cos_std", _cos_std, False, True, False, "차원 표준화 후 코사인 — 이방성 제거")
        )
    if whiten is not None:
        matrix = np.asarray(whiten, dtype=np.float64)

        def _cos_whiten(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _cos(a @ matrix.T, b @ matrix.T)

        readouts.append(
            Readout("cos_whiten", _cos_whiten, False, True, False, "백색화 후 코사인 — 상관까지 제거")
        )
    return tuple(readouts)


def fit_standardiser(z: np.ndarray) -> dict[str, np.ndarray]:
    return {"mean": z.mean(0), "scale": z.std(0)}


def fit_whitener(z: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    centred = z - z.mean(0)
    covariance = np.cov(centred, rowvar=False) + ridge * np.eye(z.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T
    return inverse_root


# --------------------------------------------------------------------- ordering


def ordering_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    """짝지은 순서 일치율. ``scores`` 는 낮을수록 목표에 가깝다는 뜻(에너지).

    **동점 처리**: 점수가 같은 쌍은 0.5 로 센다.  이것이 표준적인 동점 보정이며,
    여기서는 단순한 관례가 아니라 해석의 문제다.  동점을 불일치로 세면 아무 정보도
    없는 readout(모든 점수가 같은 경우)이 0.5 가 아니라 0.0 으로 보고되어,
    "정보 없음"과 "정확히 거꾸로"가 구분되지 않는다.  분해에서 우연 아래 값을
    읽어야 하는 이상, 이 둘은 반드시 구분되어야 한다.

    대상(target)이 동점인 쌍은 분모에서 제외한다 — 순서가 정의되지 않으므로.
    """

    if scores.size != targets.size:
        raise ValueError("scores 와 targets 길이가 다르다")
    n = int(scores.size)
    if n < 2:
        return math.nan
    agree = 0.0
    total = 0
    for i in range(n - 1):
        ds = scores[i + 1 :] - scores[i]
        dt = targets[i + 1 :] - targets[i]
        comparable = dt != 0
        if not comparable.any():
            continue
        ds = ds[comparable]
        dt = dt[comparable]
        tied = ds == 0
        agree += float(((ds * dt) > 0).sum()) + 0.5 * float(tied.sum())
        total += int(comparable.sum())
    return agree / total if total else math.nan


def _measured_chance(
    scores: np.ndarray, targets: np.ndarray, *, draws: int = 200, seed: int = 17
) -> dict[str, float]:
    """우연 수준을 가정하지 않고 라벨 순열로 **측정**한다."""

    rng = np.random.default_rng(seed)
    observed = ordering_accuracy(scores, targets)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        null[index] = ordering_accuracy(scores, rng.permutation(targets))
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if draws > 1 else 0.0,
        "draws": int(draws),
    }


def _recovery_ratio(readout: float, own: float, chance: float) -> float:
    """``own_energy`` 가 확보한 여유 중 이 readout 이 되찾은 비율."""

    denominator = own - chance
    if abs(denominator) < 1e-12:
        return math.nan
    return (readout - chance) / denominator


def _ratio_cluster_bootstrap(
    per_unit_readout: Sequence[float],
    per_unit_own: Sequence[float],
    chance: float,
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> estimators.Interval:
    """복원 비율을 클러스터 부트스트랩한다.

    입력은 **단위(유전자 클러스터)별로 이미 집계된** 값이다.  한 단위가 한
    클러스터이므로 단위를 재표집하는 것이 곧 클러스터 부트스트랩이다.

    비율은 차이가 아니므로 ``paired_contrast`` 로 대신할 수 없다.  분자와 분모를
    같은 재표집 안에서 **함께** 다시 계산해야 구간이 정직해진다.  분모를 고정한 채
    분자만 흔들면 구간이 좁아진다.
    """

    readout_values = np.asarray(per_unit_readout, dtype=np.float64)
    own_values = np.asarray(per_unit_own, dtype=np.float64)
    if readout_values.size != own_values.size:
        raise ValueError("단위별 배열 길이가 다르다")

    usable = np.isfinite(readout_values) & np.isfinite(own_values)
    readout_values = readout_values[usable]
    own_values = own_values[usable]
    n_units = int(readout_values.size)
    if n_units == 0:
        return estimators.Interval(math.nan, math.nan, math.nan, 0, "no usable units")

    point = _recovery_ratio(
        float(readout_values.mean()), float(own_values.mean()), chance
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = rng.integers(0, n_units, size=n_units)
        draws[index] = _recovery_ratio(
            float(readout_values[picked].mean()), float(own_values[picked].mean()), chance
        )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return estimators.Interval(point, math.nan, math.nan, n_units, "percentile (degenerate)")
    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return estimators.Interval(
        point, float(low), float(high), n_units, "cluster percentile"
    )


@dataclass(frozen=True)
class ReadoutResult:
    family: str
    readout: str
    accuracy: float
    recovery: float
    ci_low: float
    ci_high: float
    mechanism: Mapping[str, bool]


def decompose_family(
    *,
    family: str,
    z_state: np.ndarray,
    z_goal: np.ndarray,
    own_energy: np.ndarray,
    targets: np.ndarray,
    clusters: Sequence[Any],
    readouts: Sequence[Readout],
    claim_id: str,
    role: Role = Role.EXPLORATORY,
    draws: int = 200,
    seed: int = 17,
) -> tuple[list[ReadoutResult], list[Finding], dict[str, np.ndarray]]:
    """한 family 의 학습된 ``z`` 를 readout 격자로 읽는다.

    반환: 결과표, Finding 목록, per-pair 배열(``durable.write_pairs`` 로 저장).
    """

    chance = _measured_chance(own_energy, targets, draws=draws, seed=seed)
    own_accuracy = chance["observed"]

    labels = np.asarray(list(clusters))
    unique = np.unique(labels)

    def per_unit(scores: np.ndarray) -> np.ndarray:
        out = np.empty(unique.size, dtype=np.float64)
        for position, name in enumerate(unique):
            rows = np.flatnonzero(labels == name)
            out[position] = ordering_accuracy(scores[rows], targets[rows])
        return out

    own_per_unit = per_unit(own_energy)
    results: list[ReadoutResult] = []
    findings: list[Finding] = []
    arrays: dict[str, np.ndarray] = {
        "targets": targets,
        "own_energy": own_energy,
        "clusters": labels.astype("U64"),
        "own_per_unit": own_per_unit,
    }

    for readout in readouts:
        scores = readout.fn(z_state, z_goal)
        accuracy = ordering_accuracy(scores, targets)
        unit_scores = per_unit(scores)
        interval = _ratio_cluster_bootstrap(
            unit_scores, own_per_unit, chance["null_mean"], seed=seed
        )
        mechanism = {
            "preserves_norm": readout.preserves_norm,
            "removes_anisotropy": readout.removes_anisotropy,
            "uses_interaction": readout.uses_interaction,
        }
        results.append(
            ReadoutResult(
                family=family,
                readout=readout.key,
                accuracy=float(accuracy),
                recovery=float(interval.estimate),
                ci_low=float(interval.low),
                ci_high=float(interval.high),
                mechanism=mechanism,
            )
        )
        findings.append(
            Finding(
                claim_id=claim_id,
                name=f"readout:{family}:{readout.key}",
                evidence=Evidence.PAIRED_INTERVAL,
                estimate=float(interval.estimate),
                ci_low=float(interval.low),
                ci_high=float(interval.high),
                unit="gene",
                n_units=int(interval.units),
                role=role,
                method="recovery ratio, cluster percentile bootstrap, measured chance",
                detail={
                    "readout": readout.key,
                    "readout_description": readout.description,
                    "accuracy": float(accuracy),
                    "own_energy_accuracy": float(own_accuracy),
                    "measured_chance": chance,
                    **mechanism,
                },
            )
        )
        arrays[f"scores_{readout.key}"] = scores
        arrays[f"per_unit_{readout.key}"] = unit_scores

    return results, findings, arrays


def interpret(results: Sequence[ReadoutResult]) -> dict[str, Any]:
    """무엇을 버렸을 때 떨어졌는지로 해석을 만든다.

    임계값으로 자르지 않는다.  각 기제군의 복원 비율을 구간과 함께 그대로 싣고,
    어느 기제가 설명하는지는 구간이 서로 겹치는지로 독자가 읽게 둔다.
    """

    def group(predicate: Callable[[ReadoutResult], bool]) -> list[dict[str, Any]]:
        return [
            {
                "readout": r.readout,
                "recovery": r.recovery,
                "ci": [r.ci_low, r.ci_high],
                "accuracy": r.accuracy,
            }
            for r in results
            if predicate(r)
        ]

    return {
        "question": "학습된 기하는 자기 에너지에만 실려 있는가, 아니면 cosine 이 크기를 버릴 뿐인가",
        "norm_preserving": group(lambda r: r.mechanism["preserves_norm"]),
        "anisotropy_removing": group(lambda r: r.mechanism["removes_anisotropy"]),
        "norm_discarding_only": group(
            lambda r: not r.mechanism["preserves_norm"] and not r.mechanism["removes_anisotropy"]
        ),
        "reading": (
            "크기 보존 readout 이 복원하면 주장은 '코사인이 크기를 버릴 뿐'으로 축소된다. "
            "이방성 제거 readout 이 복원하면 이방성 이야기다. "
            "어느 쪽도 복원하지 못하면 상호작용항(t-s, s*t)이 싣고 있다는 강한 형태가 남는다."
        ),
    }

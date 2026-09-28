#!/usr/bin/env python3
"""geometry 스윕 15런의 dev 지표를 거리 x 시드 표로 만든다.

읽기 전용. value_metrics.json 만 읽고 아무것도 쓰지 않는다.
주의: 여기 나오는 숫자는 dev 이고 test 는 봉인 상태다. 판정이 아니라 관측이다.
"""
import json, os, statistics as st

ROOT = "artifacts/acmg/checkpoints/geometry_comparison"
ENER = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
KEYS = [
    ("policy_optimal_set_accuracy", "policy"),
    ("joint_stop_action_accuracy", "joint"),
    ("regret_at_1", "regret@1"),
]

def dev_of(e, s):
    p = f"{ROOT}/{e}/seed-{s}/value_metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    d = m.get("metrics", m).get("dev") or m.get("dev") or {}
    return d if d else None

rows = {e: {s: dev_of(e, s) for s in SEEDS} for e in ENER}
n_done = sum(1 for e in ENER for s in SEEDS if rows[e][s])
print(f"완료 {n_done}/15    (dev 지표. test 는 봉인 — 판정이 아니라 관측이다)\n")

for key, label in KEYS:
    hi_is_good = key != "regret_at_1"
    print(f"== {label} ({'높을수록 좋음' if hi_is_good else '낮을수록 좋음'}) ==")
    print(f"  {'거리':<22s}" + "".join(f"{('seed'+str(s)):>10s}" for s in SEEDS) + f"{'평균':>10s}{'표준편차':>10s}")
    means = {}
    for e in ENER:
        vals = []
        line = f"  {e:<22s}"
        for s in SEEDS:
            d = rows[e][s]
            if d and key in d:
                vals.append(float(d[key])); line += f"{d[key]:>10.4f}"
            else:
                line += f"{'-':>10s}"
        if vals:
            m = sum(vals)/len(vals); means[e] = m
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            line += f"{m:>10.4f}{sd:>10.4f}"
        else:
            line += f"{'-':>10s}{'-':>10s}"
        print(line)
    if len(means) >= 2:
        best = (max if hi_is_good else min)(means, key=means.get)
        full = [e for e in ENER if all(rows[e][s] for s in SEEDS)]
        note = "" if len(full) == len(ENER) else f"  (3시드 완주한 거리: {', '.join(full) or '없음'})"
        print(f"  -> 현재 선두: {best} ({means[best]:.4f}){note}")
    print()

# euclidean 은 코드가 paired contrast 의 reference 로 쓴다 (value.py:1649)
if all(rows["euclidean"][s] for s in SEEDS):
    print("== euclidean 대비 차이 (평균, 3시드 완주한 거리만) ==")
    base = {k: sum(float(rows['euclidean'][s][k]) for s in SEEDS)/3 for k, _ in KEYS}
    for e in ENER:
        if e == "euclidean" or not all(rows[e][s] for s in SEEDS):
            continue
        parts = []
        for key, label in KEYS:
            m = sum(float(rows[e][s][key]) for s in SEEDS)/3
            parts.append(f"{label} {m-base[key]:+.4f}")
        print(f"  {e:<22s} " + "   ".join(parts))
    print("\n  주의: 이 차이는 dev 평균의 단순 차이다. 짝지은 신뢰구간이 아니므로 판정 근거가 아니다.")

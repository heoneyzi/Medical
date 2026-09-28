#!/usr/bin/env python3
"""geometry 스윕 15런 요약. 읽기 전용 — value_metrics.json 만 읽는다.

주의: 여기 숫자는 전부 dev 다. test 는 봉인돼 있고 이 표는 판정이 아니라 관측이다.
판정은 집계가 만드는 짝지은 구간과 사전등록된 finding 으로만 한다.
"""
import json, os, statistics as st

ROOT = "artifacts/acmg/checkpoints/geometry_comparison"
ENER = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
REF = "euclidean"   # value.py:1649 가 paired contrast 의 reference 로 쓰는 arm

def load(e, s):
    p = f"{ROOT}/{e}/seed-{s}/value_metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return m.get("metrics", m).get("dev") or m.get("dev") or None

def prog(e, s):
    p = f"{ROOT}/{e}/seed-{s}/training_progress.json"
    return json.load(open(p)) if os.path.exists(p) else {}

D = {e: {s: load(e, s) for s in SEEDS} for e in ENER}
full = [e for e in ENER if all(D[e][s] for s in SEEDS)]
n = sum(1 for e in ENER for s in SEEDS if D[e][s])

print(f"geometry 스윕 — {n}/15 완료, 3시드 완주 {len(full)}/5")
print("dev 지표. test 봉인. 판정 아님 — 관측이다.\n")

# 1) 봉인·조기종료 무결성
print("== 무결성 ==")
bad = []
for e in ENER:
    for s in SEEDS:
        m = D[e][s]
        if not m: continue
        pj = prog(e, s)
        raw = json.load(open(f"{ROOT}/{e}/seed-{s}/value_metrics.json"))
        if raw.get("test_reported") is not False: bad.append(f"{e}-{s} test_reported={raw.get('test_reported')}")
        if pj.get("status") != "complete": bad.append(f"{e}-{s} status={pj.get('status')}")
print("  test_reported=False 전부 확인, status=complete 전부 확인" if not bad else "  !! " + "; ".join(bad))
eps = {f"{e}-{s}": (prog(e,s).get("completed_epoch"), prog(e,s).get("best_epoch"))
       for e in ENER for s in SEEDS if D[e][s]}
print(f"  종료 epoch 범위 {min(v[0] for v in eps.values())}~{max(v[0] for v in eps.values())}, "
      f"best_epoch 범위 {min(v[1] for v in eps.values())}~{max(v[1] for v in eps.values())}\n")

# 2) 지표별 표
for key, label, hi in [("policy_optimal_set_accuracy","policy",True),
                       ("joint_stop_action_accuracy","joint",True),
                       ("regret_at_1","regret@1",False)]:
    print(f"== {label} ({'높을수록 좋음' if hi else '낮을수록 좋음'}) ==")
    print(f"  {'거리':<22s}" + "".join(f"{'seed'+str(s):>10s}" for s in SEEDS) + f"{'평균':>10s}{'표준편차':>10s}")
    for e in ENER:
        vals = [float(D[e][s][key]) for s in SEEDS if D[e][s] and key in D[e][s]]
        line = f"  {e:<22s}" + "".join(
            f"{D[e][s][key]:>10.4f}" if D[e][s] and key in D[e][s] else f"{'-':>10s}" for s in SEEDS)
        if vals:
            line += f"{sum(vals)/len(vals):>10.4f}{(st.pstdev(vals) if len(vals)>1 else 0.0):>10.4f}"
        print(line)
    print()

# 3) reference 대비 차이 — 차이가 시드 변동보다 큰지 함께 본다
if REF in full:
    print(f"== {REF} 대비 (3시드 완주 거리만) ==")
    print(f"  {'거리':<22s}{'지표':>10s}{'차이':>10s}{'시드sd':>10s}  판단")
    for key, label, hi in [("policy_optimal_set_accuracy","policy",True),
                           ("joint_stop_action_accuracy","joint",True),
                           ("regret_at_1","regret@1",False)]:
        base = sum(float(D[REF][s][key]) for s in SEEDS)/3
        for e in full:
            if e == REF: continue
            vals = [float(D[e][s][key]) for s in SEEDS]
            diff = sum(vals)/3 - base
            sd = st.pstdev(vals)
            verdict = "시드변동에 묻힘" if abs(diff) < sd else "시드변동보다 큼"
            print(f"  {e:<22s}{label:>10s}{diff:>+10.4f}{sd:>10.4f}  {verdict}")
    print("\n  이 '판단'은 어림이지 검정이 아니다. 짝지은 신뢰구간이 0을 배제하는지로만 판정한다.")

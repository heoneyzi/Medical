#!/usr/bin/env python3
"""C5 상관이 '과제 난이도' 인공물인지 가르는 통제.

문제. task 를 단위로 (정렬, 계획)의 상관을 재면, 둘 다 같은 모델이 같은 task 에서 낸 값이므로
"쉬운 task 는 둘 다 잘 된다"는 공통 요인 하나로 설명될 수 있다.

통제. 모델 A 의 정렬과 **다른 모델 B** 의 계획을 짝지어 같은 상관을 잰다.
  - 상관이 과제 난이도 때문이라면 A=B 든 A!=B 든 비슷하게 나온다.
  - 표현-계획의 모델 고유 연결이라면 A=B 가 A!=B 보다 뚜렷이 높아야 한다.

저장된 산출물만 쓴다. src/ 수정 없음.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")

z = np.load("artifacts/acmg/findings/decomposition_pairs.npz", allow_pickle=True)
ROOT = Path("artifacts/acmg/checkpoints/geometry_comparison")
ENERGIES = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
task = z["task"].astype(str); gene = z["gene"].astype(str)
task_gene = dict(zip(task, gene))

def per_task_ordering(vec):
    s = defaultdict(float); n = defaultdict(int)
    for t, c in zip(task, vec): s[t] += float(c); n[t] += 1
    return {t: s[t]/n[t] for t in s}
def per_task_policy(e, s):
    return json.load(open(ROOT/e/f"seed-{s}"/"value_metrics.json"))["metrics"]["dev"]["per_task_policy_accuracy"]
def pearson(x, y):
    x=np.asarray(x); y=np.asarray(y)
    if len(x)<3 or x.std()==0 or y.std()==0: return float("nan")
    return float(np.corrcoef(x,y)[0,1])

runs = [(e,s) for e in ENERGIES for s in SEEDS if f"energy_{e}-{s}" in z]
ordc = {f"{e}-{s}": per_task_ordering(z[f"energy_{e}-{s}"].astype(float)) for e,s in runs}
polc = {f"{e}-{s}": per_task_policy(e,s) for e,s in runs}
print(f"  런 {len(runs)}개 · task {len(task_gene)}")

within, cross_same_energy, cross_diff_energy = [], [], []
detail = []
for ea,sa in runs:
    A=f"{ea}-{sa}"
    for eb,sb in runs:
        B=f"{eb}-{sb}"
        common = sorted(set(ordc[A]) & set(polc[B]))
        r = pearson([ordc[A][t] for t in common], [float(polc[B][t]) for t in common])
        if r!=r: continue
        detail.append({"ordering_from":A,"policy_from":B,"r":r})
        if A==B: within.append(r)
        elif ea==eb: cross_same_energy.append(r)
        else: cross_diff_energy.append(r)

def summ(v,l):
    v=np.asarray(v); print(f"  {l:<34s} n={len(v):>3d}  평균 r={v.mean():+.3f}  중앙 {np.median(v):+.3f}  범위 [{v.min():+.3f}, {v.max():+.3f}]")
print()
summ(within,            "A=B  (같은 모델)")
summ(cross_same_energy, "A!=B, 같은 에너지 다른 시드")
summ(cross_diff_energy, "A!=B, 다른 에너지")

# 유클리드를 뺀 '좋은 기하'만으로도 확인
good=[(e,s) for e,s in runs if e!="euclidean"]
gw=[d["r"] for d in detail if d["ordering_from"]==d["policy_from"] and d["ordering_from"].rsplit("-",1)[0]!="euclidean"]
gc=[d["r"] for d in detail if d["ordering_from"]!=d["policy_from"]
    and d["ordering_from"].rsplit("-",1)[0]!="euclidean" and d["policy_from"].rsplit("-",1)[0]!="euclidean"]
print()
print("  유클리드 제외 (좋은 기하 12런만)")
summ(gw,"  A=B"); summ(gc,"  A!=B")
print()
d=np.mean(gw)-np.mean(gc)
print(f"  => 같은 모델 − 다른 모델 = {d:+.3f}")
print("     이 값이 0 근처면 상관은 과제 난이도이고, 뚜렷이 양수면 모델 고유 연결이다.")
json.dump({"within":within,"cross_same_energy":cross_same_energy,"cross_diff_energy":cross_diff_energy,
           "good_within":gw,"good_cross":gc,"detail":detail},
          open("artifacts/acmg/findings/c5_crossmodel_control.json","w"), indent=2)
print("\n  저장: artifacts/acmg/findings/c5_crossmodel_control.json")

#!/usr/bin/env python3
"""기하 선택이 모델의 어떤 출력을 바꾸는가.

동기. 지금까지 policy·joint·regret 셋만 봤고, 거기서 "기하는 계획 성능의 바닥을 정하지만
천장을 정하지 않는다"(정렬 0.09 변동 대비 policy 0.01)를 얻었다. 그렇다면 정렬 능력이
실제로 값을 바꾸는 출력은 어디인가. dev 지표 전체를 훑어 효과의 소재를 찾는다.

방법. euclidean 을 기준으로 각 에너지의 짝지은 차이를 낸다.
per-task 값이 있는 지표는 집계 코드와 같은 방식(시드 매칭 평균 후 dev task 재표집)을 쓰고,
스칼라만 있는 지표는 시드 3개의 평균과 표준편차를 함께 보고한다(구간 없음을 명시).

저장된 산출물만 쓴다. src/ 수정 없음. dev 한정, test 봉인.
"""
import json, sys, statistics as st
from pathlib import Path
import numpy as np
sys.path.insert(0,"src")
from geoflowagent.training.value import _bootstrap_task_mean

ROOT=Path("artifacts/acmg/checkpoints/geometry_comparison")
ENERGIES=["cosine","directed_quasimetric","poincare","pair_mlp"]
REF="euclidean"; SEEDS=[17,29,43]
OUT=Path("artifacts/acmg/findings/metric_profile.json")

def dev(e,s): return json.load(open(ROOT/e/f"seed-{s}"/"value_metrics.json"))["metrics"]["dev"]

# 스칼라 지표 목록 (높을수록 좋은지 표시)
SCALARS=[("policy_optimal_set_accuracy",True),("joint_stop_action_accuracy",True),
         ("gated_optimal_set_accuracy",True),("q_only_optimal_set_accuracy",True),
         ("pairwise_regret_order_accuracy",True),("regret_at_1",False),
         ("value_mae",False),("q_mae",False),("selected_known_unreachable_rate",False),
         ("regret_at_1_label_coverage",True)]
NESTED=[("completion","balanced_accuracy",True),("completion","brier",False),
        ("action_reachability","balanced_accuracy",True),("action_reachability","brier",False),
        ("state_reachability","balanced_accuracy",True),("state_reachability","brier",False)]

def get(d,key,sub=None):
    v=d[key]
    return float(v[sub]) if sub else float(v)

res={"reference":REF,"test_reported":False,"scalar":{},"paired_task":{}}
print(f"  기준 = {REF}.  아래는 '각 에너지 − {REF}' 다.\n")
hdr=f"  {'metric':<40s}" + "".join(f"{e[:12]:>14s}" for e in ENERGIES) + f"{'ref값':>10s}{'ref sd':>9s}"
print(hdr); print("  "+"-"*len(hdr))
for key,sub,*rest in [(k,None,h) for k,h in SCALARS]+[(k,s,h) for k,s,h in NESTED]:
    higher=rest[0]
    ref=[get(dev(REF,s),key,sub) for s in SEEDS]
    row=f"  {(key if not sub else key+'.'+sub):<40s}"
    entry={}
    for e in ENERGIES:
        vals=[get(dev(e,s),key,sub) for s in SEEDS]
        d=st.mean(vals)-st.mean(ref)
        entry[e]={"mean_diff":d,"mean":st.mean(vals),"sd":st.pstdev(vals)}
        mark="" if (d>0)==higher else "!"   # ! 는 기준보다 나쁨
        row+=f"{d:>+13.4f}{mark:<1s}"
    row+=f"{st.mean(ref):>10.4f}{st.pstdev(ref):>9.4f}"
    print(row)
    res["scalar"][key if not sub else f"{key}.{sub}"]={"higher_is_better":higher,
        "reference_mean":st.mean(ref),"reference_sd":st.pstdev(ref),"arms":entry}

# per-task 가 있는 세 지표는 짝지은 구간까지
print("\n  === per-task 값이 있는 지표: 짝지은 구간 (dev task 584, 시드 매칭 평균 후 재표집) ===")
PT=[("per_task_policy_accuracy","policy",True),("per_task_joint_accuracy","joint",True),
    ("per_task_regret_at_1","regret@1",False)]
for key,lbl,higher in PT:
    for e in ENERGIES:
        diffs={}
        for s in SEEDS:
            a=dev(e,s)[key]; b=dev(REF,s)[key]
            for t in set(a)&set(b): diffs.setdefault(t,[]).append(float(a[t])-float(b[t]))
        iv=_bootstrap_task_mean(diffs,reps=2000,seed=840_000)
        ex="0 배제" if (iv["low"]>0 or iv["high"]<0) else "UNRESOLVED"
        print(f"    {lbl:<10s} {e:<22s} {iv['mean']:+.4f} [{iv['low']:+.4f}, {iv['high']:+.4f}]  {ex}")
        res["paired_task"].setdefault(lbl,{})[e]=iv
OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2,sort_keys=True))
print(f"\n  저장: {OUT}")

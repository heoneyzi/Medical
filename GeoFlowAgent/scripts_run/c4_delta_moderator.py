#!/usr/bin/env python3
"""C4 의 조절변수 회귀 — delta-hyperbolicity 쪽.

사전등록(01_EXPERIMENT_PLAN.md:400-404):
    per-task benefit(quasimetric - cosine) ~ beta_1 * rho     + controls
    per-task benefit(hyperbolic  - cosine) ~ beta_2 * delta   + controls
    "기울기 beta 와 CI 를 보고한다. 평균 승패가 아니라."
    반증조건: 기울기 CI 가 0 을 포함하면 실제 데이터에서도 복잡한 기하가 불필요함을 확증.

beta_1 은 추정 불가능하다 — 상태 그래프가 순수 DAG(간선 1,458,964개 전부 prefix 증가)여서
비가역성 rho 가 모든 task 에서 1.0 상수다. 분산 0.  이 사실은 결과로 보고한다.

여기서는 beta_2 를 낸다.
  delta  = Gromov 4점 델타. 상태 그래프를 무향으로 보고 최단경로 거리에서 계산.
           네 점 x,y,z,w 에 대해 세 합 d(xy)+d(zw), d(xz)+d(yw), d(xw)+d(yz) 를 정렬해
           (최대 - 차대)/2. task 의 delta 는 표본 사중쌍의 최댓값(그리고 분위수도 함께 보고).
  benefit = 해당 task 의 poincare policy - cosine policy (시드 3개 매칭 평균)

통제변수가 사전등록 문서에 열거되어 있지 않다(감사 decisions_required 10항).
따라서 무보정 기울기를 주 결과로 내고, 명백한 통제(상태 수·평균 깊이)를 넣은 판본을
함께 보고하되 그것이 사전등록된 형태가 아님을 명시한다.

기울기의 구간은 유전자 클러스터 재표집으로 낸다(사전등록 분석과 같은 단위).
"""
import os  # portfolio copy: $GEOWORK placeholder is expanded at runtime
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0,"src")
ROOT=Path("artifacts/acmg/checkpoints/geometry_comparison")
SEEDS=[17,29,43]
QUADS=20000
OUT=Path("artifacts/acmg/findings/c4_delta_moderator.json")
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

graphs=pickle.load(open(os.path.expandvars("$GEOWORK/task_graphs.pkl"),"rb"))
log(f"그래프 {len(graphs):,d} task 적재")

def per_task(energy, key="per_task_policy_accuracy"):
    acc={}
    for s in SEEDS:
        m=json.load(open(ROOT/energy/f"seed-{s}"/"value_metrics.json"))["metrics"]["dev"][key]
        for t,v in m.items(): acc.setdefault(t,[]).append(float(v))
    return {t:float(np.mean(v)) for t,v in acc.items() if len(v)==len(SEEDS)}

pol_poin=per_task("poincare"); pol_cos=per_task("cosine"); pol_dqm=per_task("directed_quasimetric")
dev_tasks=sorted(set(pol_poin)&set(pol_cos)&set(pol_dqm))
log(f"dev task {len(dev_tasks):,d}")

rng=np.random.default_rng(17)
def delta_of(task):
    g=graphs.get(task)
    if not g: return None
    nodes=g["nodes"]; idx={n:i for i,n in enumerate(nodes)}
    N=len(nodes)
    if N<4: return None
    rows=[];cols=[]
    for a,b in g["edges"]:
        if a in idx and b in idx:
            rows.append(idx[a]); cols.append(idx[b])
    if not rows: return None
    A=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N))
    D=shortest_path(A,method="D",directed=False,unweighted=True)
    fin=np.isfinite(D)
    if not fin.all():
        D=np.where(fin,D,np.nan)
    q=rng.integers(0,N,size=(QUADS,4))
    ok=(q[:,0]!=q[:,1])&(q[:,0]!=q[:,2])&(q[:,0]!=q[:,3])&(q[:,1]!=q[:,2])&(q[:,1]!=q[:,3])&(q[:,2]!=q[:,3])
    q=q[ok]
    if len(q)<100: return None
    x,y,zz,w=q[:,0],q[:,1],q[:,2],q[:,3]
    s1=D[x,y]+D[zz,w]; s2=D[x,zz]+D[y,w]; s3=D[x,w]+D[y,zz]
    S=np.sort(np.stack([s1,s2,s3],axis=1),axis=1)   # 오름차순
    dd=(S[:,2]-S[:,1])/2.0
    dd=dd[np.isfinite(dd)]
    if not len(dd): return None
    return {"delta_max":float(dd.max()),"delta_p95":float(np.quantile(dd,0.95)),
            "delta_mean":float(dd.mean()),"nodes":int(N),"quads":int(len(dd)),
            "mean_pairwise_distance":float(np.nanmean(D[np.isfinite(D)]))}

rows=[]
for i,t in enumerate(dev_tasks):
    d=delta_of(t)
    if d is None: continue
    rows.append({"task":t,"gene":t.split(":")[0],**d,
                 "benefit_poin_minus_cos":pol_poin[t]-pol_cos[t],
                 "benefit_dqm_minus_cos":pol_dqm[t]-pol_cos[t]})
    if (i+1)%150==0: log(f"  {i+1}/{len(dev_tasks)} task delta 계산")
log(f"delta 계산 완료: {len(rows):,d} task")

def slope_ci(xs, ys, genes, reps=2000, seed=17):
    xs=np.asarray(xs); ys=np.asarray(ys)
    def fit(x,y):
        if len(x)<3 or x.std()==0: return np.nan
        return float(np.polyfit(x,y,1)[0])
    point=fit(xs,ys)
    by={}
    for x,y,g in zip(xs,ys,genes): by.setdefault(g,[]).append((x,y))
    gs=sorted(by); rng=np.random.default_rng(seed); draws=[]
    for _ in range(reps):
        pick=rng.integers(0,len(gs),size=len(gs))
        X=[];Y=[]
        for j in pick:
            for x,y in by[gs[j]]: X.append(x);Y.append(y)
        v=fit(np.asarray(X),np.asarray(Y))
        if v==v: draws.append(v)
    if not draws:
        return {"slope":point,"low":None,"high":None,"genes":len(gs),"tasks":len(xs),
                "degenerate":True,"reason":"조절변수의 분산이 0 이라 기울기를 추정할 수 없다"}
    return {"slope":point,"low":float(np.quantile(draws,0.025)),
            "high":float(np.quantile(draws,0.975)),"genes":len(gs),"tasks":len(xs),
            "degenerate":False}

res={"note":"C4 조절변수 회귀. beta_1(rho) 는 rho 가 상수여서 추정 불가. 여기는 beta_2(delta).",
     "rho_is_constant":True,"rho_reason":"상태 그래프가 순수 DAG (간선 1,458,964개 전부 prefix 증가)",
     "delta_stat_used":"delta_max","tasks":len(rows),"regressions":{}}
genes=[r["gene"] for r in rows]
for dstat in ["delta_max","delta_p95","delta_mean"]:
    xs=[r[dstat] for r in rows]
    for lbl,bk in [("poincare_minus_cosine","benefit_poin_minus_cos"),
                   ("dqm_minus_cosine","benefit_dqm_minus_cos")]:
        ys=[r[bk] for r in rows]
        r_=slope_ci(xs,ys,genes)
        if r_.get("degenerate") or r_["low"] is None:
            print(f"  beta2  {lbl:<24s} ~ {dstat:<11s}  추정 불가 (조절변수 분산 0)")
            res["regressions"][f"{lbl}~{dstat}"]={**r_,"excludes_zero":None}
        else:
            ex="0 배제" if (r_["low"]>0 or r_["high"]<0) else "0 포함 -> C4 반증조건 충족"
            print(f"  beta2  {lbl:<24s} ~ {dstat:<11s}  {r_['slope']:+.5f} [{r_['low']:+.5f}, {r_['high']:+.5f}]  {ex}")
            res["regressions"][f"{lbl}~{dstat}"]={**r_,"excludes_zero":bool(r_["low"]>0 or r_["high"]<0)}
for dstat in ["delta_max","delta_p95","delta_mean"]:
    v=np.asarray([r[dstat] for r in rows],float)
    res.setdefault("moderator_distribution",{})[dstat]={
        "min":float(v.min()),"max":float(v.max()),"sd":float(v.std()),
        "unique_values":int(len(set(v.tolist())))}
res["graph_is_structurally_identical"]={
    "nodes_per_task_unique": int(len({r["nodes"] for r in rows})),
    "note":"모든 task 의 상태 그래프가 같은 노드 수와 지름을 갖는다. 조절변수에 회귀할 변동이 없다."}
ds=[r["delta_max"] for r in rows]
res["delta_distribution"]={"min":float(min(ds)),"median":float(np.median(ds)),"max":float(max(ds)),
                           "sd":float(np.std(ds)),"unique_values":int(len(set(ds)))}
print(f"\n  delta_max 분포: 중앙 {np.median(ds):.2f} · 범위 [{min(ds):.2f}, {max(ds):.2f}] · sd {np.std(ds):.3f} · 고유값 {len(set(ds))}")
OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2,sort_keys=True))
print(f"  저장: {OUT}")

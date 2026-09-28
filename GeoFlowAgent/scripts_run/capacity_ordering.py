#!/usr/bin/env python3
"""용량 대조 arm 의 정렬 능력 — 표현 수준에서도 용량 독립인가.

용량 대조 실험(A/B/C)은 계획 지표(policy·joint·regret)에서 "이득은 용량이 아니라 형태"를 보였다.
그러나 그 arm 들의 **정렬**(깊이 고정 상태 순서)은 재지 않았다. 표현 수준에서도 같은지 확인한다.

  pair_mlp @h111  (335,530 파라미터, euclidean@h128 의 335,942 보다 412개 적음)
  euclidean @h147 (368,983, pair_mlp@h128 의 369,479 보다 496개 적음)

기준선은 기존 15런과 동일한 쌍·행 선별·통계를 쓴다.
store 1회 적재. 진행 출력. src/ 무수정. dev 한정, test 무접근.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,"src")
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import load_value_geometry_checkpoint
from geoflowagent.geoacmg import probes
from geoflowagent.geoacmg.runners import _state_depth
from geoflowagent.geoacmg.inference import paired_contrast
from geoflowagent.geoacmg.claims import Role
from geoflowagent.utils.io import read_yaml

CONFIG, PROCESSED, CACHE = "configs/geoacmg.yaml", "artifacts/acmg/processed", "artifacts/acmg/cache"
GEO=Path("artifacts/acmg/checkpoints/geometry_comparison")
CAP=Path("artifacts/acmg/checkpoints/capacity_matched")
SEEDS=[17,29,43]; CHUNK=8192
OUT=Path("artifacts/acmg/findings/capacity_ordering.json")
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

cfg=read_yaml(CONFIG); tc=cfg.get("value_training",{})
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
store=SearchFeatureStore(PROCESSED,CACHE,structured_dim=int(tc.get("structured_dim",64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

sel,tasks,depths,vstar,genes=[],[],[],[],[]
for i in sorted(store.indices("dev")):
    r=store.examples[i]
    if not r.get("value_known_mask"): continue
    if not (r.get("optimal_actions") or []): continue
    gi=(r.get("group_ids") or {}).get("entity")
    g=gi[0] if isinstance(gi,list) and gi else (gi if gi else r["task_id"])
    sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state",{})))
    vstar.append(float(r["value_star"])); genes.append(str(g))
log(f"dev 사용 {len(sel):,d}")

@torch.no_grad()
def energies(p):
    m,_=load_value_geometry_checkpoint(p,device); m.eval()
    out=np.empty(len(sel))
    for s in range(0,len(sel),CHUNK):
        b=store.batch(sel[s:s+CHUNK],device=device)
        a=m.encode_entity(b["state_views"],b["structured_state"])
        g=m.encode_entity(b["goal_views"],b["structured_goal"])
        e=m.energy_head(a,g)
        out[s:s+e.shape[0]]=e.to(torch.float64).cpu().numpy()
    return out, sum(q.numel() for q in m.parameters() if q.requires_grad)

ARMS=[("pair_mlp@h111", CAP/"pair_mlp-h111"), ("euclidean@h147", CAP/"euclidean-h147"),
      ("pair_mlp@h128", GEO/"pair_mlp"),      ("euclidean@h128", GEO/"euclidean")]
vecs={}; params={}
for lbl,d in ARMS:
    cs=[]
    for s in SEEDS:
        p=d/f"seed-{s}"/"value_geometry.pt"
        if not p.exists(): log(f"  {lbl} seed-{s} 체크포인트 없음"); continue
        en,np_=energies(p); params[lbl]=np_
        c,chance,clusters=probes.depth_matched_pairs(list(en),vstar,depths,tasks)
        cs.append(np.asarray(c,dtype=np.int8))
        log(f"  {lbl:<18s} s{s}: 정렬 {np.mean(c):.4f}  (파라미터 {np_:,d})")
    if cs: vecs[lbl]=np.mean(cs,axis=0)
gene_of=dict(zip(tasks,genes)); cl=[gene_of[t] for t in clusters]
def contrast(a,b,name):
    f=paired_contrast(claim_id="C1",name=name,left=list(vecs[a]),right=list(vecs[b]),
                      clusters=cl,unit="gene",role=Role.EXPLORATORY)
    ex="0 배제" if (f.ci_low>0 or f.ci_high<0) else "UNRESOLVED"
    print(f"  {name:<44s} {f.estimate:+.4f} [{f.ci_low:+.4f}, {f.ci_high:+.4f}]  {ex}", flush=True)
    return {"estimate":f.estimate,"ci_low":f.ci_low,"ci_high":f.ci_high,"n_units":f.n_units,"excludes_zero":bool(f.ci_low>0 or f.ci_high<0)}
print()
res={"test_reported":False,"parameters":params,
     "ordering":{k:float(v.mean()) for k,v in vecs.items()},"contrasts":{}}
print("  용량 대조 (정렬)")
if "pair_mlp@h111" in vecs and "euclidean@h128" in vecs:
    res["contrasts"]["A: pair_mlp@h111(335,530) - euclidean@h128(335,942)"]=contrast("pair_mlp@h111","euclidean@h128","A 치료군이 파라미터 적음")
if "pair_mlp@h128" in vecs and "euclidean@h147" in vecs:
    res["contrasts"]["B: pair_mlp@h128(369,479) - euclidean@h147(368,983)"]=contrast("pair_mlp@h128","euclidean@h147","B 대조군에 용량 부여")
if "pair_mlp@h111" in vecs and "pair_mlp@h128" in vecs:
    res["contrasts"]["C: pair_mlp@h111 - pair_mlp@h128"]=contrast("pair_mlp@h111","pair_mlp@h128","C 같은 에너지 용량만 -9.2%")
if "euclidean@h147" in vecs and "euclidean@h128" in vecs:
    res["contrasts"]["D: euclidean@h147 - euclidean@h128"]=contrast("euclidean@h147","euclidean@h128","D euclidean 용량 +9.8%")
OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2,sort_keys=True))
log(f"저장: {OUT}")

#!/usr/bin/env python3
"""두 가지 타당성 점검을 store 1회 적재로 처리한다.

A. 일반화 — 모든 결과가 dev 한정이고 체크포인트는 dev 로 선택됐다(best_dev_selection_score).
   정렬 능력이 선택 인공물인지 보려면 train 에서도 재야 한다.
   train >> dev 면 과적합, 비슷하면 일반화한다는 뜻이다.

B. 층화 — 쌍은 같은 (task, depth) 안에서 V* 가 다른 상태끼리 만든다. 그 V* 차이의 크기와
   깊이에 따라 효과가 어떻게 달라지는지 본다. 거의 동률인 쌍이 대부분이면 해석이 달라진다.
   이를 위해 쌍 열거를 직접 하되, probes.depth_matched_pairs 와 결과가 같은지 검증한다.

원칙: store 1회 적재 · 진행 출력 · src/ 무수정 · test 무접근.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import load_value_geometry_checkpoint
from geoflowagent.geoacmg import probes
from geoflowagent.geoacmg.runners import _state_depth
from geoflowagent.geoacmg.inference import paired_contrast
from geoflowagent.geoacmg.claims import Role
from geoflowagent.utils.io import read_yaml

CONFIG, PROCESSED, CACHE = "configs/geoacmg.yaml", "artifacts/acmg/processed", "artifacts/acmg/cache"
ROOT = Path("artifacts/acmg/checkpoints/geometry_comparison")
ENERGIES = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
OUT = Path("artifacts/acmg/findings/generalization_and_strata.json")
CHUNK = 8192
MAX_TRAIN_PAIRS = 3_000_000   # 안전장치: 넘으면 task 를 부분표집하고 그 사실을 보고한다

t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg=read_yaml(CONFIG); tc=cfg.get("value_training",{})
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
store=SearchFeatureStore(PROCESSED,CACHE,structured_dim=int(tc.get("structured_dim",64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

def select(split):
    sel,tasks,depths,vstar,genes=[],[],[],[],[]
    tot=nv=no=0
    for i in sorted(store.indices(split)):
        r=store.examples[i]; tot+=1
        if not r.get("value_known_mask"): nv+=1; continue
        if not (r.get("optimal_actions") or []): no+=1; continue
        gi=(r.get("group_ids") or {}).get("entity")
        g=gi[0] if isinstance(gi,list) and gi else (gi if gi else r["task_id"])
        sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state",{})))
        vstar.append(float(r["value_star"])); genes.append(str(g))
    return sel,tasks,depths,vstar,genes,tot,nv,no

def enumerate_pairs(tasks,depths,vstar):
    """probes.depth_matched_pairs 와 같은 순서로 열거하되 메타데이터도 남긴다."""
    buckets=defaultdict(list)
    for i,(t,d) in enumerate(zip(tasks,depths)): buckets[(t,d)].append(i)
    li=[];ri=[];dep=[];dv=[]
    for (t,d),mem in buckets.items():
        for a in range(len(mem)):
            for b in range(a+1,len(mem)):
                x,y=mem[a],mem[b]
                if vstar[x]==vstar[y]: continue
                li.append(x); ri.append(y); dep.append(d); dv.append(abs(vstar[x]-vstar[y]))
    return np.array(li),np.array(ri),np.array(dep,np.int16),np.array(dv)

@torch.no_grad()
def energies(path, sel):
    m,_=load_value_geometry_checkpoint(path,device); m.eval()
    out=np.empty(len(sel))
    for s in range(0,len(sel),CHUNK):
        b=store.batch(sel[s:s+CHUNK],device=device)
        a=m.encode_entity(b["state_views"],b["structured_state"])
        g=m.encode_entity(b["goal_views"],b["structured_goal"])
        e=m.energy_head(a,g)
        out[s:s+e.shape[0]]=e.to(torch.float64).cpu().numpy()
    return out

res={"test_reported":False,"splits":{}}

for split in ("dev","train"):
    sel,tasks,depths,vstar,genes,tot,nv,no=select(split)
    log(f"[{split}] 상태 {tot:,d} -> 사용 {len(sel):,d} (제외 value_known {nv:,d} / optimal_actions {no:,d})")
    li,ri,dep,dv=enumerate_pairs(tasks,depths,vstar)
    log(f"[{split}] 쌍 {len(li):,d}")
    sub=None
    if len(li)>MAX_TRAIN_PAIRS:
        rng=np.random.default_rng(17); sub=rng.choice(len(li),MAX_TRAIN_PAIRS,replace=False); sub.sort()
        li,ri,dep,dv=li[sub],ri[sub],dep[sub],dv[sub]
        log(f"[{split}] 쌍이 상한을 넘어 {MAX_TRAIN_PAIRS:,d}개로 부분표집 (이 사실을 결과에 남긴다)")
    va=np.array(vstar)
    true_order=(va[li]<va[ri])
    gene_of=dict(zip(tasks,genes)); cl=[gene_of[tasks[i]] for i in li]
    blk={"states_total":tot,"states_used":len(sel),"dropped_value_known":nv,"dropped_no_optimal":no,
         "pairs":int(len(li)),"subsampled":bool(sub is not None),"genes":len(set(cl)),
         "tasks":len(set(tasks)),"runs":{}}
    # 검증: dev 에서 probes 구현과 일치하는지 (첫 체크포인트로 한 번만)
    verify = (split=="dev")
    for e in ENERGIES:
        for s in SEEDS:
            p=ROOT/e/f"seed-{s}"/"value_geometry.pt"
            if not p.exists(): continue
            en=energies(p,sel)
            corr=((en[li]<en[ri])==true_order).astype(np.int8)
            acc=float(corr.mean())
            if verify:
                ref,_,_=probes.depth_matched_pairs(list(en),vstar,depths,tasks)
                ok = (len(ref)==len(corr)) and float(np.mean(ref))==acc
                log(f"   참조 구현 일치 검증({e}-{s}): {'통과' if ok else '불일치!'} (ref {np.mean(ref):.6f} vs {acc:.6f})")
                verify=False
            f=paired_contrast(claim_id="C1",name=f"GEN_{split}_{e}_{s}",left=list(corr.astype(float)),
                              right=[0.5]*len(corr),clusters=cl,unit="gene",role=Role.EXPLORATORY)
            strat={}
            for lo,hi,lbl in [(0,2,"dv<2"),(2,5,"2<=dv<5"),(5,1e9,"dv>=5")]:
                m_=(dv>=lo)&(dv<hi)
                if m_.sum()>100: strat[lbl]={"pairs":int(m_.sum()),"acc":float(corr[m_].mean())}
            dstrat={}
            for dlo,dhi,lbl in [(0,3,"depth<3"),(3,6,"3<=depth<6"),(6,99,"depth>=6")]:
                m_=(dep>=dlo)&(dep<dhi)
                if m_.sum()>100: dstrat[lbl]={"pairs":int(m_.sum()),"acc":float(corr[m_].mean())}
            blk["runs"][f"{e}-{s}"]={"ordering":acc,"vs_chance":{"estimate":f.estimate,"ci_low":f.ci_low,
                "ci_high":f.ci_high,"n_units":f.n_units},"by_delta_vstar":strat,"by_depth":dstrat}
            log(f"   {split} {e:<22s} s{s}: {acc:.4f}")
    res["splits"][split]=blk

OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2,sort_keys=True))
log(f"저장: {OUT}")

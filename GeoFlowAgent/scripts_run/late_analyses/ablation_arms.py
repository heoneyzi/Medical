#!/usr/bin/env python3
"""결정적 대조 — 정보가 MedCPT 에 있는가, 기호 해시에 있는가.

R8 이 보인 것:
  MedCPT 768d 단독      : cos 0.5055 · cos_std 0.5011 · PCA-64+cos 0.4775  (전부 우연)
  MedCPT + structured   : cos 0.6469 · PCA-64+cos 0.7040                   (0 배제)

`structured_state` 는 언어모델 임베딩이 아니다. `data/structured.py` 의
``StructuredHasher`` 가 상태 dict 의 key=value 를 blake2b 로 64 차원에 해싱한
**학습 없는 기호적 특징**이다.

따라서 빠진 팔은 하나다: **기호 해시 단독.** 그것만으로 0.70 이 나오면
동결 LM 임베딩의 기여는 0 이고, "동결 공간이 실행 구조를 담는다"는 서사 전체가
기호 특징의 공으로 넘어간다. 그 대조 없이는 어느 쪽도 말할 수 없다.

통계는 전부 train 에서 추정한다. test 미개봉.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.geoacmg.runners import _state_depth
from geoflowagent.geoacmg import readout as RO, estimators, probes
from geoflowagent.utils.io import read_yaml

CHUNK, TRAIN_FIT_MAX = 8192, 40000
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg = read_yaml("configs/geoacmg.yaml"); tc = cfg.get("value_training", {})
store = SearchFeatureStore("artifacts/acmg/processed", "artifacts/acmg/cache",
                           structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

def select(split):
    sel, tasks, depths, vstar, genes = [], [], [], [], []
    for i in sorted(store.indices(split)):
        r = store.examples[i]
        if not r.get("value_known_mask"): continue
        if not (r.get("optimal_actions") or []): continue
        gi = (r.get("group_ids") or {}).get("entity")
        gene = gi[0] if isinstance(gi, list) and gi else (gi if gi else r["task_id"])
        sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state", {})))
        vstar.append(float(r["value_star"])); genes.append(str(gene))
    return sel, tasks, depths, vstar, genes

dsel, dtasks, ddepth, dvstar, dgenes = select("dev")
tsel, *_ = select("train")
rng = np.random.default_rng(17)
if len(tsel) > TRAIN_FIT_MAX:
    tsel = [tsel[i] for i in sorted(rng.choice(len(tsel), TRAIN_FIT_MAX, replace=False))]

buckets = defaultdict(list)
for idx, (task, depth) in enumerate(zip(dtasks, ddepth)):
    buckets[(task, depth)].append(idx)
A, B, PG = [], [], []
for (task, _), members in buckets.items():
    for li in range(len(members)):
        for ri in range(li + 1, len(members)):
            a, b = members[li], members[ri]
            if dvstar[a] == dvstar[b]: continue
            A.append(a); B.append(b); PG.append(dgenes[a])
A = np.asarray(A); B = np.asarray(B); genes_list = PG
vs = np.asarray(dvstar); DT = vs[B] - vs[A]
log(f"쌍 {len(A):,d} · 유전자 {len(set(genes_list))}")

def gather(indices, arm):
    n = len(indices); S = G = None
    for s in range(0, n, CHUNK):
        b = store.batch(indices[s:s+CHUNK], device="cpu")
        view = sorted(b["state_views"])[0]
        mv_s = b["state_views"][view].to(torch.float64).numpy()
        mv_g = b["goal_views"][view].to(torch.float64).numpy()
        sh_s = b["structured_state"].to(torch.float64).numpy()
        sh_g = b["structured_goal"].to(torch.float64).numpy()
        if arm == "medcpt_only":   st, gl = mv_s, mv_g
        elif arm == "structured_only": st, gl = sh_s, sh_g
        else:                      st, gl = np.concatenate([mv_s, sh_s],1), np.concatenate([mv_g, sh_g],1)
        if S is None: S = np.empty((n, st.shape[1])); G = np.empty((n, gl.shape[1]))
        S[s:s+len(st)] = st; G[s:s+len(gl)] = gl
    return S, G

def conc(scores):
    margin = scores[B] - scores[A]
    return np.where(margin == 0, 0.5, np.where(margin * DT > 0, 1.0, 0.0))

def report(name, scores, bag):
    v = conc(scores)
    iv = estimators.cluster_bootstrap((v - 0.5).tolist(), genes_list, resamples=1000, seed=17)
    ez = iv.low > 0 or iv.high < 0
    bag[name] = {"accuracy": float(v.mean()), "minus_chance": float(iv.estimate),
                 "ci": [float(iv.low), float(iv.high)], "excludes_zero": bool(ez)}
    log(f"  {name:26s} {v.mean():.4f}  우연대비 {iv.estimate:+.4f} [{iv.low:+.4f},{iv.high:+.4f}] "
        f"{'0배제' if ez else 'UNRESOLVED'}")
    return v

results = {}
for arm in ("medcpt_only", "structured_only", "medcpt_plus_structured"):
    log(f"=== {arm} ===")
    dS, dG = gather(dsel, arm); tS, tG = gather(tsel, arm)
    bag = {}
    geo = probes.hubness(dS[:1500])
    bag["_geometry"] = {k: float(geo[k]) for k in ("dimension","effective_rank","mean_cosine_similarity")}
    log(f"  기하 {bag['_geometry']}")
    report("cos", RO._cos(dS, dG), bag)
    report("l2", RO._l2(dS, dG), bag)
    both = np.concatenate([tS, tG], 0); mu = both.mean(0)
    _, _, Vt = np.linalg.svd(both - mu, full_matrices=False)
    for k in (16, 32, 64):
        if k > Vt.shape[0]: continue
        P = Vt[:k]
        report(f"PCA-{k} + cos", RO._cos((dS-mu) @ P.T, (dG-mu) @ P.T), bag)
    results[arm] = bag

Path("artifacts/acmg/findings/R9_information_source_ablation.json").write_text(json.dumps({
 "title": "정보의 출처 — 동결 LM 임베딩인가, 학습 없는 기호 해시인가",
 "question": ("R8 에서 MedCPT 단독은 어떤 선형 readout 으로도 우연 수준이었고, "
              "기호 해시를 더하자 PCA-64+cos 가 0.7040 이 됐다. "
              "기호 해시 단독으로도 같은 값이 나오면 동결 LM 임베딩의 기여는 0 이다."),
 "structured_is_not_an_embedding": ("data/structured.py 의 StructuredHasher — 상태 dict 의 "
              "key=value 를 blake2b 로 64 차원에 해싱한다. 학습도 언어모델도 없다."),
 "split":"dev","test_reported":False,"unit":"gene","pairs":int(len(A)),
 "fitting":"PCA 는 train 에서 적합. test 미개봉.",
 "reference_learned_trunk":{"cosine own_energy (5seeds)":0.7327,"euclidean own_energy (5seeds)":0.5895,
                            "cosine cos_std train-fit":0.7515,"euclidean cos_std train-fit":0.7345},
 "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
log("기록: R9_information_source_ablation.json")
log("완료 %.1f 분" % ((time.time()-t0)/60))

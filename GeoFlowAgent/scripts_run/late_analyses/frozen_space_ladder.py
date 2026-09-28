#!/usr/bin/env python3
"""근본 질문 — 동결 임베딩 공간에 정보가 있는가, 아니면 읽는 법이 없었을 뿐인가.

동기. 지금까지 "동결 공간은 우연 수준(raw cos 0.5055 / l2 0.5078 / dot 0.4966)이고
학습된 에너지가 정보를 꺼낸다"고 읽었다. 그런데 **동결 공간에는 cos·l2·dot 세 개만
적용했다.** 학습된 z 에서는 차원 표준화(cos_std)만으로 0.38 -> 0.72 로 뛰는 것을 봤다.

그리고 동결 공간의 기하가 극단적이다:
    768 차원 · 유효 랭크 10.28 · **평균 코사인 유사도 0.969**
모든 임베딩이 거의 같은 방향을 가리킨다(cone effect). 평범한 코사인이 못 읽는 것이
당연할 수 있다. 그렇다면 trunk 335k 가 하는 일은 "정보를 만드는 것"이 아니라
"이방성을 펴는 것"일 수 있다.

이 스크립트가 가른다. 같은 쌍 141,517 개 위에서 동결 공간에:
  (a) cos · l2 · neg_dot            — 기존 세 지표
  (b) cos_std · cos_whiten          — 이방성/상관 제거 (train 에서 추정, 홀드아웃)
  (c) PCA-k + cos                   — 차원 축소만으로 되는가
  (d) 무작위 사영 768->64 + cos/cos_std  — 아무 사영이나 되는가 (대조)
  (e) 구조 특징 결합                 — trunk 가 보는 입력과 맞춤

통계는 전부 **train 에서 추정**한다. test 는 건드리지 않는다.
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

CHUNK = 8192
TRAIN_FIT_MAX = 40000
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
log(f"dev {len(dsel):,d} · train(추정용) {len(tsel):,d}")

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

def gather(indices, with_structured=False):
    n = len(indices)
    S = G = None
    for s in range(0, n, CHUNK):
        b = store.batch(indices[s:s+CHUNK], device="cpu")
        view = sorted(b["state_views"])[0]
        st = b["state_views"][view].to(torch.float64).numpy()
        gl = b["goal_views"][view].to(torch.float64).numpy()
        if with_structured:
            st = np.concatenate([st, b["structured_state"].to(torch.float64).numpy()], 1)
            gl = np.concatenate([gl, b["structured_goal"].to(torch.float64).numpy()], 1)
        if S is None:
            S = np.empty((n, st.shape[1])); G = np.empty((n, gl.shape[1]))
        S[s:s+len(st)] = st; G[s:s+len(gl)] = gl
    return S, G

def conc(scores):
    margin = scores[B] - scores[A]
    out = np.where(margin * DT > 0, 1.0, 0.0)
    return np.where(margin == 0, 0.5, out)

def report(name, scores, bag):
    v = conc(scores)
    iv = estimators.cluster_bootstrap((v - 0.5).tolist(), genes_list, resamples=1000, seed=17)
    ez = iv.low > 0 or iv.high < 0
    bag[name] = {"accuracy": float(v.mean()),
                 "minus_chance": {"estimate": float(iv.estimate),
                                  "ci": [float(iv.low), float(iv.high)],
                                  "excludes_zero": bool(ez), "gene_clusters": int(iv.units)}}
    log(f"  {name:34s} {v.mean():.4f}   우연대비 {iv.estimate:+.4f} [{iv.low:+.4f},{iv.high:+.4f}] "
        f"{'0배제' if ez else 'UNRESOLVED'}")
    return v

results = {}
for with_struct in (False, True):
    tag = "views+structured" if with_struct else "views_only"
    log(f"=== 동결 공간 ({tag}) ===")
    dS, dG = gather(dsel, with_struct)
    tS, tG = gather(tsel, with_struct)
    bag = {}
    geo = probes.hubness(dS[:1500])
    bag["_geometry"] = {k: geo[k] for k in ("dimension", "effective_rank", "mean_cosine_similarity")}
    log(f"  기하: {bag['_geometry']}")

    report("cos", RO._cos(dS, dG), bag)
    report("l2", RO._l2(dS, dG), bag)
    report("neg_dot", RO._neg_dot(dS, dG), bag)

    both_tr = np.concatenate([tS, tG], 0)
    std = RO.fit_standardiser(both_tr)
    mean, scale = std["mean"], np.clip(std["scale"], 1e-12, None)
    report("cos_std (train-fit)", RO._cos((dS - mean) / scale, (dG - mean) / scale), bag)
    report("l2_std (train-fit)", RO._l2((dS - mean) / scale, (dG - mean) / scale), bag)

    W = RO.fit_whitener(both_tr)
    report("cos_whiten (train-fit)", RO._cos(dS @ W.T, dG @ W.T), bag)

    # PCA-k (train 에서 적합)
    centred = both_tr - both_tr.mean(0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    for k in (8, 16, 32, 64, 128, 256):
        if k > Vt.shape[0]: continue
        P = Vt[:k]
        report(f"PCA-{k} + cos", RO._cos((dS - both_tr.mean(0)) @ P.T, (dG - both_tr.mean(0)) @ P.T), bag)

    # 무작위 사영 대조
    r = np.random.default_rng(17)
    R = r.normal(size=(64, dS.shape[1])) / np.sqrt(64)
    report("random-64 + cos", RO._cos(dS @ R.T, dG @ R.T), bag)
    rs = RO.fit_standardiser(np.concatenate([tS @ R.T, tG @ R.T], 0))
    rm, rsc = rs["mean"], np.clip(rs["scale"], 1e-12, None)
    report("random-64 + cos_std", RO._cos((dS @ R.T - rm) / rsc, (dG @ R.T - rm) / rsc), bag)
    results[tag] = bag

Path("artifacts/acmg/findings/R8_frozen_space_ladder.json").write_text(json.dumps({
    "title": "동결 임베딩 공간에 정보가 있는가 — readout 사다리",
    "question": ("지금까지 동결 공간을 cos·l2·dot 세 지표로만 읽고 '우연 수준'이라 판정했다. "
                 "학습된 z 에서는 차원 표준화만으로 0.38->0.72 로 뛰었다. "
                 "동결 공간의 평균 코사인 유사도가 0.969 (cone effect) 이므로 "
                 "평범한 코사인이 못 읽는 것이 공간의 정보 부재가 아니라 이방성 때문일 수 있다."),
    "split": "dev", "test_reported": False, "unit": "gene", "pairs": int(len(A)),
    "fitting": "표준화·백색화·PCA 는 전부 **train 에서 추정**했다 (홀드아웃). test 미개봉.",
    "reference_learned": {
        "trunk+own_energy(cosine, 5seeds)": 0.7327,
        "trunk+own_energy(euclidean, 5seeds)": 0.5895,
        "trunk+cos_std(cosine, train-fit)": 0.7515,
        "trunk+cos_std(euclidean, train-fit)": 0.7345,
    },
    "results": results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
log("기록: R8_frozen_space_ladder.json")
log("완료 %.1f 분" % ((time.time()-t0)/60))

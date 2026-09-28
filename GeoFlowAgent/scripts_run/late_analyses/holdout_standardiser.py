#!/usr/bin/env python3
"""R2b 의 최대 한계 제거 — 표준화 통계를 홀드아웃에서 추정한다.

문제: `cos_std` 의 mean/scale 을 **평가에 쓰는 바로 그 dev z** 에서 추정했다.
그러면 그 readout 에 구조적으로 유리하고, "자기 에너지 격차의 80%가 읽기 채널
차이"라는 결론이 in-sample fitting 의 산물일 수 있다.

세 가지로 가른다.
  A) dev 에서 추정 (현재 방식, 비교 기준)
  B) **train z 에서 추정** — 깨끗한 홀드아웃. test 는 건드리지 않는다.
  C) **유전자 leave-one-out** — 각 유전자의 쌍은 그 유전자를 뺀 dev z 로 추정한 통계로 읽는다.

B 나 C 에서도 격차가 크게 줄면 결론이 선다. 되돌아오면 결론을 철회한다.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import load_value_geometry_checkpoint
from geoflowagent.geoacmg.runners import _state_depth
from geoflowagent.geoacmg import readout as RO, estimators
from geoflowagent.utils.io import read_yaml

FAMS = ("cosine", "euclidean")
SEEDS = [17, 29, 43, 59, 71]
ROOTS = {17: "geometry_comparison", 29: "geometry_comparison", 43: "geometry_comparison",
         59: "geometry_r1_pairs", 71: "geometry_r1_pairs"}
CHUNK = 8192
TRAIN_FIT_MAX = 40000     # 표준화 통계 추정용 train 표본 상한 (평균/표준편차에 충분)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg = read_yaml("configs/geoacmg.yaml"); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
log(f"dev {len(dsel):,d} · train(표준화 추정용) {len(tsel):,d}")

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
A = np.asarray(A); B = np.asarray(B); PG = np.asarray(PG)
vs = np.asarray(dvstar); DT = vs[B] - vs[A]
log(f"쌍 {len(A):,d} · 유전자 {len(set(PG.tolist()))}")

@torch.no_grad()
def encode(path, indices):
    model, _ = load_value_geometry_checkpoint(path, device); model.eval()
    n = len(indices)
    zs = np.empty((n, model.shared_dim), dtype=np.float32)
    zg = np.empty((n, model.shared_dim), dtype=np.float32)
    for s in range(0, n, CHUNK):
        b = store.batch(indices[s:s+CHUNK], device=device)
        a_ = model.encode_entity(b["state_views"], b["structured_state"])
        g_ = model.encode_entity(b["goal_views"], b["structured_goal"])
        zs[s:s+a_.shape[0]] = a_.float().cpu().numpy()
        zg[s:s+g_.shape[0]] = g_.float().cpu().numpy()
    return zs.astype(np.float64), zg.astype(np.float64)

def conc(margin):
    out = np.where(margin * DT > 0, 1.0, 0.0)
    return np.where(margin == 0, 0.5, out)

def cos_std_scores(zs, zg, std):
    mean = std["mean"]; scale = np.clip(std["scale"], 1e-12, None)
    return RO._cos((zs - mean) / scale, (zg - mean) / scale)

vec = {}
for fam in FAMS:
    for s in SEEDS:
        p = Path(f"artifacts/acmg/checkpoints/{ROOTS[s]}/{fam}/seed-{s}/value_geometry.pt")
        dzs, dzg = encode(p, dsel)
        tzs, tzg = encode(p, tsel)
        # A) dev 에서 추정
        std_dev = RO.fit_standardiser(np.concatenate([dzs, dzg], 0))
        sa = cos_std_scores(dzs, dzg, std_dev)
        # B) train 에서 추정 (홀드아웃)
        std_tr = RO.fit_standardiser(np.concatenate([tzs, tzg], 0))
        sb = cos_std_scores(dzs, dzg, std_tr)
        # C) 유전자 leave-one-out (dev 안에서)
        sc = np.empty(len(dsel))
        garr = np.asarray(dgenes)
        for g in sorted(set(dgenes)):
            keep = garr != g
            stdg = RO.fit_standardiser(np.concatenate([dzs[keep], dzg[keep]], 0))
            idx = np.where(~keep)[0]
            sc[idx] = cos_std_scores(dzs[idx], dzg[idx], stdg)
        for tag, sc_ in (("dev_fit", sa), ("train_fit", sb), ("logo_fit", sc)):
            vec[(tag, fam, s)] = conc(sc_[B] - sc_[A])
        log(f"  {fam:10s} s{s}: dev_fit {vec[('dev_fit',fam,s)].mean():.4f} | "
            f"train_fit {vec[('train_fit',fam,s)].mean():.4f} | logo {vec[('logo_fit',fam,s)].mean():.4f}")

genes_list = PG.tolist()
res = {}
print()
log("=== cosine − euclidean, cos_std 의 통계를 어디서 추정했는가 ===")
for tag in ("dev_fit", "train_fit", "logo_fit"):
    a = np.mean([vec[(tag, "cosine", s)] for s in SEEDS], axis=0)
    b = np.mean([vec[(tag, "euclidean", s)] for s in SEEDS], axis=0)
    iv = estimators.cluster_bootstrap((a - b).tolist(), genes_list, resamples=2000, seed=17)
    ez = iv.low > 0 or iv.high < 0
    res[tag] = {"cosine": float(a.mean()), "euclidean": float(b.mean()),
                "difference": float(iv.estimate), "ci": [float(iv.low), float(iv.high)],
                "gene_clusters": int(iv.units), "excludes_zero": bool(ez)}
    log(f"  {tag:10s} cos {a.mean():.4f} / euc {b.mean():.4f} -> {iv.estimate:+.4f} "
        f"[{iv.low:+.4f}, {iv.high:+.4f}] {'0배제' if ez else 'UNRESOLVED'}")

Path("artifacts/acmg/findings/R2d_holdout_standardiser.json").write_text(json.dumps({
    "title": "cos_std 의 표준화 통계를 홀드아웃에서 추정해도 격차 감소가 유지되는가",
    "split": "dev", "test_reported": False, "unit": "gene", "seeds": SEEDS,
    "pairs": int(len(A)),
    "fits": {"dev_fit": "평가에 쓰는 dev z 에서 추정 (in-sample, 기존 방식)",
             "train_fit": "train z 에서 추정 (깨끗한 홀드아웃, test 미개봉)",
             "logo_fit": "유전자 leave-one-out — 각 유전자 쌍은 그 유전자를 뺀 dev z 통계로"},
    "results": res,
    "reference_own_energy": "자기 에너지로 읽으면 +0.1432 [+0.1079, +0.1788] (5시드, 유전자)",
}, ensure_ascii=False, indent=2), encoding="utf-8")
log("기록: R2d_holdout_standardiser.json")
log("완료 %.1f 분" % ((time.time()-t0)/60))

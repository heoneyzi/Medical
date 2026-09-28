#!/usr/bin/env python3
"""신규 시드 59·71 에 제3 metric 6종을 계산해 마진 파일에 병합한다.

R2b 의 핵심 분해(자기 에너지 격차의 80%가 읽기 채널 차이)가 cos_std·l2 열에서
시드 3개(17·29·43)만으로 계산됐다. 신규 시드에서도 유지되는지 확인해야 확정된다.
기존 15런 마진은 재사용하고 신규 4런만 같은 쌍 위에서 계산한다.
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
from geoflowagent.geoacmg import readout as RO
from geoflowagent.utils.io import read_yaml

NEW = Path("artifacts/acmg/checkpoints/geometry_r1_pairs")
FAMILIES = ("cosine", "euclidean")
NEW_SEEDS = (59, 71)
CHUNK = 8192
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg = read_yaml("configs/geoacmg.yaml"); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = SearchFeatureStore("artifacts/acmg/processed", "artifacts/acmg/cache",
                           structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

# 참조 구현과 동일한 행 선별 · 쌍 구성
sel, tasks, depths, vstar, genes = [], [], [], [], []
for i in sorted(store.indices("dev")):
    r = store.examples[i]
    if not r.get("value_known_mask"): continue
    if not (r.get("optimal_actions") or []): continue
    gi = (r.get("group_ids") or {}).get("entity")
    gene = gi[0] if isinstance(gi, list) and gi else (gi if gi else r["task_id"])
    sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state", {})))
    vstar.append(float(r["value_star"])); genes.append(str(gene))
n = len(sel)
buckets = defaultdict(list)
for idx, (task, depth) in enumerate(zip(tasks, depths)):
    buckets[(task, depth)].append(idx)
A, B = [], []
for (task, _), members in buckets.items():
    for li in range(len(members)):
        for ri in range(li + 1, len(members)):
            a, b = members[li], members[ri]
            if vstar[a] == vstar[b]: continue
            A.append(a); B.append(b)
A = np.asarray(A); B = np.asarray(B)
log(f"쌍 {len(A):,d} (기존 마진과 같은 구성이어야 한다)")

old = np.load("artifacts/acmg/findings/R2_readout_margins.npz", allow_pickle=True)
assert old["target_margin"].size == len(A), "쌍 개수가 기존 마진과 다르다 — 중단"
log("쌍 개수 일치 확인")

@torch.no_grad()
def coords(path):
    model, _ = load_value_geometry_checkpoint(path, device); model.eval()
    zs = np.empty((n, model.shared_dim), dtype=np.float32)
    zg = np.empty((n, model.shared_dim), dtype=np.float32)
    en = np.empty(n)
    for s in range(0, n, CHUNK):
        b = store.batch(sel[s:s+CHUNK], device=device)
        a_ = model.encode_entity(b["state_views"], b["structured_state"])
        g_ = model.encode_entity(b["goal_views"], b["structured_goal"])
        e_ = model.energy_head(a_, g_)
        zs[s:s+a_.shape[0]] = a_.float().cpu().numpy()
        zg[s:s+g_.shape[0]] = g_.float().cpu().numpy()
        en[s:s+e_.shape[0]] = e_.to(torch.float64).cpu().numpy()
    return zs, zg, en

new_margins, stats = {}, {}
for fam in FAMILIES:
    for s in NEW_SEEDS:
        key = f"{fam}-{s}"
        zs, zg, en = coords(NEW / fam / f"seed-{s}" / "value_geometry.pt")
        zs64, zg64 = zs.astype(np.float64), zg.astype(np.float64)
        both = np.concatenate([zs64, zg64], axis=0)
        std = RO.fit_standardiser(both); whi = RO.fit_whitener(both)
        stats[key] = {"standardiser_scale_min": float(std["scale"].min()),
                      "standardiser_scale_max": float(std["scale"].max()),
                      "whitener_frobenius": float(np.linalg.norm(whi)),
                      "fitted_on": "dev z (state+goal concatenated)"}
        scores = {"own_energy": en}
        for r in RO.build_readouts(standardise=std, whiten=whi):
            scores[r.key] = r.fn(zs64, zg64)
        for rk, v in scores.items():
            new_margins[f"{rk}_{key}"] = (v[B] - v[A]).astype(np.float32)
        log(f"  {fam:10s} s{s}: " + " ".join(f"{rk}={ (scores[rk][B]-scores[rk][A]) .std():.3g}" for rk in ("own_energy","cos_std")))

merged = {k: old[k] for k in old.files}
before = len(merged)
merged.update(new_margins)
out = Path("artifacts/acmg/findings/R2_readout_margins.npz")
tmp = out.with_suffix(".npz.part")
with tmp.open("wb") as fh: np.savez_compressed(fh, **merged)
tmp.replace(out)
log(f"마진 병합: 벡터 {before} -> {len(merged)}개 ({out.stat().st_size/1e6:.1f} MB)")
Path("artifacts/acmg/findings/R2c_newseed_readout_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
log("완료 %.1f 분" % ((time.time()-t0)/60))

#!/usr/bin/env python3
"""정렬 능력을 trunk 기여와 에너지 헤드 기여로 분해한다.

동기. 앞선 실험은 "학습된 에너지가 raw 거리보다 훨씬 잘 정렬한다"를 보였다(+0.32).
그러나 에너지는 학습된 좌표 z = encode_entity(x) 위에서 계산된다. 따라서
"에너지 공식이 일한다"와 "공유 trunk 가 만든 좌표가 일한다"가 분리되지 않았다.
trunk 가 이미 정렬 가능한 공간을 만들어 놓았다면 에너지 공식은 장식일 수 있다.

분해. 체크포인트마다 같은 쌍 위에서 세 가지로 정렬한다.
  (a) model.energy_head(z_s, z_g)      <- 모델 자신의 에너지
  (b) 1 - cos(z_s, z_g)                <- z 위의 평범한 코사인 (에너지 헤드 제거)
  (c) ||z_s - z_g||^2                  <- z 위의 평범한 제곱 L2
(b) 자체가 trunk 의 기여이고, (a)-(b) 가 에너지 헤드의 순수 기여다.

추가로 P1 이 이름에서 지목한 raw L2 기준선을 동결 공간에서 함께 잰다
(지금까지 raw 기준선은 코사인뿐이었다).

원칙:
  - store 를 한 번만 적재한다 (오늘 최대 낭비였던 반복 적재를 피한다).
  - 진행 상황을 계속 출력한다 (train_flow 가 3.5시간 침묵한 실수를 반복하지 않는다).
  - 쌍 구성·행 선별·통계는 참조 구현을 그대로 쓴다.
  - src/ 는 수정하지 않는다. dev 에서만 평가하고 test 는 건드리지 않는다.
"""
import json, sys, time
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
OUT = Path("artifacts/acmg/findings/mechanism_decomposition.json")
VEC = Path("artifacts/acmg/findings/decomposition_pairs.npz")
CHUNK = 8192

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg = read_yaml(CONFIG); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device={device}")
store = SearchFeatureStore(PROCESSED, CACHE, structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

# --- 참조 구현과 동일한 행 선별 --------------------------------------------
sel, tasks, depths, vstar, genes = [], [], [], [], []
n_dev = n_no_value = n_no_opt = 0
for i in sorted(store.indices("dev")):
    r = store.examples[i]; n_dev += 1
    if not r.get("value_known_mask"): n_no_value += 1; continue
    if not (r.get("optimal_actions") or []): n_no_opt += 1; continue
    gi = (r.get("group_ids") or {}).get("entity")
    gene = gi[0] if isinstance(gi, list) and gi else (gi if gi else r["task_id"])
    sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state", {})))
    vstar.append(float(r["value_star"])); genes.append(str(gene))
log(f"dev {n_dev:,d} -> 사용 {len(sel):,d} (제외: value_known_mask 거짓 {n_no_value:,d}, optimal_actions 없음 {n_no_opt:,d})")

def order(vals):
    return probes.depth_matched_pairs(list(vals), vstar, depths, tasks)

# --- 동결 공간 raw 기준선 3종 ----------------------------------------------
log("동결 공간 raw 기준선 계산 (cosine / L2 / dot)")
n = len(sel)
raw_cos = np.empty(n); raw_l2 = np.empty(n); raw_dot = np.empty(n)
for s in range(0, n, CHUNK):
    b = store.batch(sel[s:s+CHUNK], device="cpu")
    view = sorted(b["state_views"])[0]
    st = b["state_views"][view].to(torch.float64).numpy()
    gl = b["goal_views"][view].to(torch.float64).numpy()
    us = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-12)
    ug = gl / (np.linalg.norm(gl, axis=1, keepdims=True) + 1e-12)
    raw_cos[s:s+len(st)] = 1.0 - np.sum(us*ug, axis=1)
    raw_l2[s:s+len(st)]  = np.square(st-gl).sum(axis=1)
    raw_dot[s:s+len(st)] = -np.sum(st*gl, axis=1)   # 작을수록 가깝게 부호 맞춤
log("  raw 기준선 완료")

c_cos, chance, clusters = order(raw_cos)
gene_of = dict(zip(tasks, genes)); cl_gene = [gene_of[t] for t in clusters]
c_l2,  _, _ = order(raw_l2)
c_dot, _, _ = order(raw_dot)
log(f"  쌍 {len(c_cos):,d} · 유전자 {len(set(cl_gene))}개")
log(f"  raw cosine {np.mean(c_cos):.4f} · raw L2 {np.mean(c_l2):.4f} · raw dot {np.mean(c_dot):.4f}  (우연 0.5)")

def pack(f): return {"estimate": f.estimate, "ci_low": f.ci_low, "ci_high": f.ci_high, "n_units": f.n_units}
def contrast(left, right, name):
    return pack(paired_contrast(claim_id="C1", name=name, left=list(left), right=list(right),
                                clusters=cl_gene, unit="gene", role=Role.EXPLORATORY))

res = {"split": "dev", "test_reported": False, "unit": "gene",
       "pairs": len(c_cos), "gene_clusters": len(set(cl_gene)),
       "dev_states_total": n_dev, "dev_states_used": len(sel),
       "dropped_value_known_mask_false": n_no_value, "dropped_no_optimal_actions": n_no_opt,
       "raw_frozen": {
           "cosine": float(np.mean(c_cos)), "squared_l2": float(np.mean(c_l2)), "neg_dot": float(np.mean(c_dot)),
           "cosine_vs_chance": contrast(c_cos, chance, "raw_cos_vs_chance"),
           "squared_l2_vs_chance": contrast(c_l2, chance, "raw_l2_vs_chance"),
           "neg_dot_vs_chance": contrast(c_dot, chance, "raw_dot_vs_chance"),
       },
       "frozen_geometry": None,   # 아래에서 채운다
       "checkpoints": {}}

# 동결 공간의 기하 진단 (probe 가 보고한 effective_rank 18.07 과 대조용)
_b = store.batch(sel[:1500], device="cpu")
_view = sorted(_b["state_views"])[0]
res["frozen_geometry"] = probes.hubness(_b["state_views"][_view].to(torch.float64).numpy())
log(f"  동결 공간: effective_rank {res['frozen_geometry']['effective_rank']:.2f} / {res['frozen_geometry']['dimension']}, "
    f"mean_cos {res['frozen_geometry']['mean_cosine_similarity']:.4f}")
vecs = {"raw_cos": np.asarray(c_cos, np.int8), "raw_l2": np.asarray(c_l2, np.int8),
        "raw_dot": np.asarray(c_dot, np.int8), "gene": np.asarray(cl_gene), "task": np.asarray(clusters),
        "depth_of_pair": np.asarray([0]*len(c_cos), np.int16)}

@torch.no_grad()
def coords(path):
    model, _ = load_value_geometry_checkpoint(path, device); model.eval()
    zs = np.empty((n, model.shared_dim), dtype=np.float32)
    zg = np.empty((n, model.shared_dim), dtype=np.float32)
    en = np.empty(n)
    for s in range(0, n, CHUNK):
        b = store.batch(sel[s:s+CHUNK], device=device)
        a = model.encode_entity(b["state_views"], b["structured_state"])
        g = model.encode_entity(b["goal_views"], b["structured_goal"])
        e = model.energy_head(a, g)
        zs[s:s+a.shape[0]] = a.float().cpu().numpy()
        zg[s:s+g.shape[0]] = g.float().cpu().numpy()
        en[s:s+e.shape[0]] = e.to(torch.float64).cpu().numpy()
    return zs, zg, en

for energy in ENERGIES:
    for seed in SEEDS:
        p = ROOT / energy / f"seed-{seed}" / "value_geometry.pt"
        if not p.exists(): log(f"  {energy}-{seed} 체크포인트 없음 — 건너뜀"); continue
        zs, zg, en = coords(p)
        zsn = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-12)
        zgn = zg / (np.linalg.norm(zg, axis=1, keepdims=True) + 1e-12)
        z_cos = 1.0 - np.sum(zsn*zgn, axis=1)
        z_l2  = np.square(zs.astype(np.float64)-zg.astype(np.float64)).sum(axis=1)
        c_en, _, _ = order(en); c_zc, _, _ = order(z_cos); c_zl, _, _ = order(z_l2)
        key = f"{energy}-{seed}"
        vecs[f"energy_{key}"] = np.asarray(c_en, np.int8)
        vecs[f"zcos_{key}"]   = np.asarray(c_zc, np.int8)
        vecs[f"zl2_{key}"]    = np.asarray(c_zl, np.int8)
        res["checkpoints"][key] = {
            "energy": energy, "seed": seed,
            "ordering_energy_head": float(np.mean(c_en)),
            "ordering_cosine_on_z": float(np.mean(c_zc)),
            "ordering_l2_on_z": float(np.mean(c_zl)),
            "energy_head_minus_cosine_on_z": contrast(c_en, c_zc, f"EH_{key}"),
            "cosine_on_z_minus_raw_cosine": contrast(c_zc, c_cos, f"TRUNK_{key}"),
            "learned_geometry": probes.hubness(zs[:1500].astype(np.float64)),
        }
        log(f"  {energy:22s} s{seed}: energy {np.mean(c_en):.4f} | cos(z) {np.mean(c_zc):.4f} | L2(z) {np.mean(c_zl):.4f} "
            f"| eff.rank {res['checkpoints'][key]['learned_geometry']['effective_rank']:.1f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
np.savez_compressed(VEC, **vecs)
log(f"저장 완료: {OUT}")

#!/usr/bin/env python3
"""메커니즘 실험: 학습된 에너지가 raw 거리가 실패한 '깊이 고정 정렬'을 고치는가.

probe 가 남긴 두 사실이 긴장한다:
  (1) 동결 MedCPT 상태 좌표에 선형 읽기를 붙이면 V* 가 복원되고(R^2 0.3355)
      최적 행동을 55.2% 맞춘다(측정된 우연 29.0%).
  (2) 같은 공간에서 목표까지의 raw 코사인으로 깊이를 고정해 정렬하면 우연 근처다(E2).
"정보는 있는데 metric 에는 없다." 학습된 거리 헤드가 채울 자리가 여기다. 그걸 잰다.

참조 구현을 그대로 따른다 — 이게 비교 가능성의 전부다:
  행 선별   : runners.load_probe_table 과 동일 (value_known_mask 참 AND optimal_actions 비어있지 않음)
  유전자    : row["group_ids"]["entity"][0]           (runners.py:53)
  깊이      : runners._state_depth(row["state"])
  쌍 구성   : probes.depth_matched_pairs (같은 task·같은 depth 안의 쌍만)
  raw 정의  : 1 - cos(state_emb, goal_emb)             (runners.py alignment 호출부)
  통계      : inference.paired_contrast, 유전자 클러스터 부트스트랩

바뀌는 것은 정렬 기준 하나뿐: raw 코사인 -> energy_head(encode_entity(state), encode_entity(goal)).
같은 행·같은 버킷·같은 열거 순서이므로 correct_raw 와 correct_learned 가 원소별로 짝지어진다.

dev 에서만 평가한다(모델이 train 으로 학습했으므로). test 는 건드리지 않는다.
per-pair 정확도 벡터를 저장해 두어, 나중에 클러스터 단위를 바꿔도 재계산이 필요 없게 한다.
src/ 는 수정하지 않는다.
"""
import json, sys
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
OUT = Path("artifacts/acmg/findings/ordering_mechanism.json")
VEC = Path("artifacts/acmg/findings/ordering_pairs.npz")
CHUNK = 4096

cfg = read_yaml(CONFIG); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = SearchFeatureStore(PROCESSED, CACHE, structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))

# --- 참조 구현과 동일한 행 선별 (store.examples 를 재사용해 두 번 읽지 않는다) ---
dev_set = set(store.indices("dev"))
sel, tasks, depths, vstar, genes = [], [], [], [], []
n_dev = n_no_value = n_no_optimal = 0
for i in sorted(dev_set):
    r = store.examples[i]
    n_dev += 1
    if not r.get("value_known_mask"): n_no_value += 1; continue
    if not (r.get("optimal_actions") or []): n_no_optimal += 1; continue
    gi = (r.get("group_ids") or {}).get("entity")
    gene = gi[0] if isinstance(gi, list) and gi else (gi if gi else r["task_id"])
    sel.append(i); tasks.append(str(r["task_id"])); depths.append(_state_depth(r.get("state", {})))
    vstar.append(float(r["value_star"])); genes.append(str(gene))

print(f"dev 상태 {n_dev:,d} -> 사용 {len(sel):,d}", flush=True)
print(f"  제외: value_known_mask 거짓 {n_no_value:,d} · optimal_actions 없음 {n_no_optimal:,d}  (fail-loud: 전부 보고)", flush=True)
print(f"  고유 task {len(set(tasks)):,d} · 고유 유전자 {len(set(genes)):,d} · 깊이 {min(depths)}~{max(depths)}", flush=True)

def raw_cosine():
    v = np.empty(len(sel))
    for s in range(0, len(sel), CHUNK):
        b = store.batch(sel[s:s+CHUNK], device="cpu")
        view = sorted(b["state_views"])[0]
        st = b["state_views"][view].to(torch.float64).numpy()
        gl = b["goal_views"][view].to(torch.float64).numpy()
        st /= (np.linalg.norm(st, axis=1, keepdims=True) + 1e-12)
        gl /= (np.linalg.norm(gl, axis=1, keepdims=True) + 1e-12)
        v[s:s+len(st)] = 1.0 - np.sum(st*gl, axis=1)
    return v

@torch.no_grad()
def learned(path):
    model, _ = load_value_geometry_checkpoint(path, device); model.eval()
    v = np.empty(len(sel))
    for s in range(0, len(sel), CHUNK):
        b = store.batch(sel[s:s+CHUNK], device=device)
        zs = model.encode_entity(b["state_views"], b["structured_state"])
        zg = model.encode_entity(b["goal_views"], b["structured_goal"])
        e = model.energy_head(zs, zg)
        v[s:s+e.shape[0]] = e.detach().to(torch.float64).cpu().numpy()
    return v

print("raw 코사인...", flush=True)
c_raw, chance, clusters = probes.depth_matched_pairs(list(raw_cosine()), vstar, depths, tasks)
gene_of = dict(zip(tasks, genes))
cl_gene = [gene_of[t] for t in clusters]
print(f"  쌍 {len(c_raw):,d} · 유전자 클러스터 {len(set(cl_gene))}개 · raw 정렬 {np.mean(c_raw):.4f} (우연 0.5)", flush=True)

def pack(f): return {"estimate": f.estimate, "ci_low": f.ci_low, "ci_high": f.ci_high, "n_units": f.n_units}
res = {"unit": "gene", "split": "dev", "test_reported": False,
       "pairs": len(c_raw), "gene_clusters": len(set(cl_gene)), "task_clusters": len(set(clusters)),
       "dev_states_total": n_dev, "dev_states_used": len(sel),
       "dropped_value_known_mask_false": n_no_value, "dropped_no_optimal_actions": n_no_optimal,
       "raw_cosine_ordering_accuracy": float(np.mean(c_raw)), "energies": {}}
vecs = {"raw": np.asarray(c_raw, dtype=np.int8), "gene": np.asarray(cl_gene), "task": np.asarray(clusters)}

for energy in ENERGIES:
    per = {}
    for seed in SEEDS:
        p = ROOT / energy / f"seed-{seed}" / "value_geometry.pt"
        if not p.exists(): print(f"  {energy}-{seed}: 체크포인트 없음 — 건너뜀", flush=True); continue
        c_l, _, cl2 = probes.depth_matched_pairs(list(learned(p)), vstar, depths, tasks)
        assert cl2 == clusters, "쌍 열거가 어긋났다 — 짝짓기 불가"
        vecs[f"{energy}-{seed}"] = np.asarray(c_l, dtype=np.int8)
        fc = paired_contrast(claim_id="C1", name=f"ORD_{energy}_s{seed}_vs_chance", left=c_l, right=chance,
                             clusters=cl_gene, unit="gene", role=Role.EXPLORATORY,
                             detail={"energy": energy, "seed": seed, "pairs": len(c_l)})
        fr = paired_contrast(claim_id="C1", name=f"ORD_{energy}_s{seed}_minus_raw", left=c_l, right=c_raw,
                             clusters=cl_gene, unit="gene", role=Role.EXPLORATORY,
                             detail={"energy": energy, "seed": seed, "pairs": len(c_l)})
        per[seed] = {"ordering_accuracy": float(np.mean(c_l)), "vs_chance": pack(fc), "minus_raw_cosine": pack(fr)}
        print(f"  {energy:22s} s{seed}: 정렬 {np.mean(c_l):.4f}  raw 대비 {fr.estimate:+.4f} "
              f"[{fr.ci_low:+.4f}, {fr.ci_high:+.4f}] n={fr.n_units}", flush=True)
    if per:
        a = [v["ordering_accuracy"] for v in per.values()]
        res["energies"][energy] = {"per_seed": per, "mean_ordering_accuracy": float(np.mean(a)),
                                   "std_ordering_accuracy": float(np.std(a))}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
np.savez_compressed(VEC, **vecs)
print(f"\n저장: {OUT}\n      {VEC} (per-pair 벡터 — 클러스터 단위 바꿔도 재계산 불필요)", flush=True)

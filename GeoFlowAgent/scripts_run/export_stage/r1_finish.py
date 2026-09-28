#!/usr/bin/env python3
"""R1 마무리 — 새 시드 2쌍의 정렬을 계산하고 시드 5개짜리 쌍 대조를 낸다.

기존 마진 파일(R2_readout_margins.npz)에는 시드 17·29·43 만 있다.
새로 학습한 59·71 을 같은 쌍 위에서 계산해 합치고, `paired.paired_findings` 로
**시드 클러스터와 유전자 클러스터 구간을 따로** 낸다.
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
from geoflowagent.geoacmg import paired, adapters
from geoflowagent.geoacmg.claims import Role
from geoflowagent.utils.io import read_yaml

OLD = Path("artifacts/acmg/checkpoints/geometry_comparison")
NEW = Path("artifacts/acmg/checkpoints/geometry_r1_pairs")
FAMILIES = ("cosine", "euclidean")
OLD_SEEDS, NEW_SEEDS = [17, 29, 43], [59, 71]
CHUNK = 8192
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---- 새 런의 trunk 지문 확인 (구조적 성질 재확인) --------------------------
log("=== 새 시드 trunk 지문 ===")
newfp = {}
for f in FAMILIES:
    for s in NEW_SEEDS:
        d = NEW / f / f"seed-{s}"
        newfp[(f, s)] = paired.state_dict_fingerprint(adapters.load_trunk(d))
for s in NEW_SEEDS:
    a, b = newfp[("cosine", s)], newfp[("euclidean", s)]
    log(f"  seed {s}: {'MATCH  ' if a == b else 'DIFFER '} {a[:16]} / {b[:16]}")

cfg = read_yaml("configs/geoacmg.yaml"); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = SearchFeatureStore("artifacts/acmg/processed", "artifacts/acmg/cache",
                           structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

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
A, B, PT, PG = [], [], [], []
for (task, _), members in buckets.items():
    for li in range(len(members)):
        for ri in range(li + 1, len(members)):
            a, b = members[li], members[ri]
            if vstar[a] == vstar[b]: continue
            A.append(a); B.append(b); PT.append(task); PG.append(genes[a])
A = np.asarray(A); B = np.asarray(B); PT = np.asarray(PT); PG = np.asarray(PG)
vs = np.asarray(vstar, dtype=np.float64); DT = vs[B] - vs[A]
log(f"쌍 {len(A):,d} · task {len(set(PT)):,d} · 유전자 {len(set(PG))}")

@torch.no_grad()
def own_energy(path):
    model, _ = load_value_geometry_checkpoint(path, device); model.eval()
    en = np.empty(n)
    for s in range(0, n, CHUNK):
        b = store.batch(sel[s:s+CHUNK], device=device)
        a_ = model.encode_entity(b["state_views"], b["structured_state"])
        g_ = model.encode_entity(b["goal_views"], b["structured_goal"])
        e_ = model.energy_head(a_, g_)
        en[s:s+e_.shape[0]] = e_.to(torch.float64).cpu().numpy()
    return en

def concord(margin):
    """동점 보정 규약. (마진 0 은 0.5) — 지금까지 동점은 0건이었다."""
    out = np.where(margin * DT > 0, 1.0, 0.0)
    return np.where(margin == 0, 0.5, out)

# ---- 기존 마진 재사용 + 새 런 계산 -----------------------------------------
old_npz = np.load("artifacts/acmg/findings/R2_readout_margins.npz", allow_pickle=True)
per_pair = {}
for f in FAMILIES:
    for s in OLD_SEEDS:
        per_pair[(f, s)] = concord(old_npz[f"own_energy_{f}-{s}"].astype(np.float64))
        log(f"  기존 {f}-{s}: ordering {per_pair[(f,s)].mean():.4f}")
new_margins = {}
for f in FAMILIES:
    for s in NEW_SEEDS:
        en = own_energy(NEW / f / f"seed-{s}" / "value_geometry.pt")
        m = en[B] - en[A]
        new_margins[f"own_energy_{f}-{s}"] = m.astype(np.float32)
        per_pair[(f, s)] = concord(m)
        ties = int((m == 0).sum())
        log(f"  신규 {f}-{s}: ordering {per_pair[(f,s)].mean():.4f}  (동점 {ties}건)")

# 마진 병합 저장
merged = {k: old_npz[k] for k in old_npz.files}
merged.update(new_margins)
out = Path("artifacts/acmg/findings/R2_readout_margins.npz")
tmp = out.with_suffix(".npz.part")
with tmp.open("wb") as fh: np.savez_compressed(fh, **merged)
tmp.replace(out)
log(f"마진 병합 저장: {out} ({out.stat().st_size/1e6:.1f} MB, 벡터 {len(merged)}개)")

# ---- SeedRun 구성 ----------------------------------------------------------
task_gene = {t: g for t, g in zip(PT.tolist(), PG.tolist())}
def per_task(vec):
    s = defaultdict(float); c = defaultdict(int)
    for t, v in zip(PT.tolist(), vec):
        s[t] += float(v); c[t] += 1
    return {t: s[t]/c[t] for t in s}

runs = []
for f in FAMILIES:
    for s in OLD_SEEDS + NEW_SEEDS:
        d = (OLD / f / f"seed-{s}") if s in OLD_SEEDS else (NEW / f / f"seed-{s}")
        model = adapters.build_model_at_init(d)
        fp = paired.state_dict_fingerprint(adapters.TrunkView(model))
        runs.append(paired.SeedRun(
            family=f, seed=s, trunk_fingerprint=fp,
            energy_params=paired.energy_parameter_count(model),
            total_params=int(sum(p.numel() for p in model.parameters())),
            metrics={"ordering": float(per_pair[(f, s)].mean())},
            per_task=per_task(per_pair[(f, s)]), task_gene=task_gene))
log(f"SeedRun {len(runs)}개 구성 (시드 {sorted(set(r.seed for r in runs))})")

report = paired.paired_findings(runs, families=FAMILIES, metric="ordering",
                                claim_id="C2", role=Role.EXPLORATORY,
                                require_zero_params=True, seed=17)
payload = report.to_payload()
payload["new_seeds_trained"] = NEW_SEEDS
payload["reused_seeds"] = OLD_SEEDS
Path("artifacts/acmg/findings/R1_paired_zero_param.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log("R1 결과 저장")
for f_ in report.findings:
    nm = getattr(f_, "name", "?"); est = getattr(f_, "estimate", None)
    lo, hi = getattr(f_, "ci_low", None), getattr(f_, "ci_high", None)
    unit = getattr(f_, "unit", "?"); nu = getattr(f_, "n_units", "?")
    log(f"  {nm}: {est:+.4f} [{lo:+.4f}, {hi:+.4f}]  unit={unit} n={nu}" if est is not None else f"  {nm}: {f_}")
log("=== 완료 %.1f 분 ===" % ((time.time()-t0)/60))

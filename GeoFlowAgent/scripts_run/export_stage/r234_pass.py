#!/usr/bin/env python3
"""R2·R3·R4 — 체크포인트에서 z 를 다시 꺼내 한 패스로 끝낸다.

왜 한 패스인가
--------------
store 적재 11분 절약은 부차적이다.  진짜 이유는 **R3 와 R4 가 R2 와 같은 버그를
물려받는다**는 것이다.  기존 `decomposition_pairs.npz` 는 쌍마다 **부호만** int8 로
남겼고(`np.asarray(c, np.int8)`), 동점이 어느 쪽으로 접혔는지 복원할 수 없다.
게다가 그 파일의 `depth_of_pair` 는 **전부 0 인 자리표시자**다
(`mechanism_decomposition.py`: `np.asarray([0]*len(c_cos), np.int16)`).
`c5_within_model.py` 는 그 npz 를 그대로 평균내므로 R4 도 같은 편향 위에 서 있다.

그래서 npz 재집계를 쓰지 않는다.  z 를 다시 꺼내 **원시 마진(float32)** 을 남기고,
R2·R3·R4 를 전부 그 위에서 계산한다.

무엇을 저장하는가
-----------------
쌍마다 점수 차 ``s[b] - s[a]`` 를 float32 로 남긴다.  부호가 아니라 마진이므로
앞으로 어떤 동점 규약이든 집계 단위든 재계산 없이 바꿀 수 있다 (저널 §21 의 약속을
실제로 지키는 형태).

동점 규약 두 가지를 나란히 낸다
-------------------------------
A) 참조 구현(`probes.depth_matched_pairs`): ``(s[a]<s[b]) == (t[a]<t[b])``
   동점이면 왼쪽이 False 가 되어 **참 순서 방향에 따라** 0/1 로 접힌다.
B) 동점 보정: 일치 1, 불일치 0, **동점 0.5**.
두 값과 **동점 발생률**을 함께 보고한다.  동점률은 readout 마다 다르므로 차이값에서
상쇄되지 않는다.
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
from geoflowagent.geoacmg import estimators, strata, control
from geoflowagent.geoacmg.durable import DurableRoot
from geoflowagent.utils.io import read_yaml

CONFIG, PROCESSED, CACHE = "configs/geoacmg.yaml", "artifacts/acmg/processed", "artifacts/acmg/cache"
ROOT = Path("artifacts/acmg/checkpoints/geometry_comparison")
ENERGIES = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
CHUNK = 8192

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

cfg = read_yaml(CONFIG); tc = cfg.get("value_training", {})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device={device}")
store = SearchFeatureStore(PROCESSED, CACHE, structured_dim=int(tc.get("structured_dim", 64)))
store.apply_feature_ablation(tc.get("feature_ablation"))
log("store 적재 완료")

# ---- 참조 구현과 동일한 행 선별 -------------------------------------------
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
n = len(sel)
log(f"dev {n_dev:,d} -> 사용 {n:,d} (제외: value_known_mask {n_no_value:,d}, optimal_actions 없음 {n_no_opt:,d})")

# ---- 쌍 인덱스: 참조와 같은 순서로, 단 마진을 낼 수 있게 (a,b) 를 보관 ------
buckets = defaultdict(list)
for idx, (task, depth) in enumerate(zip(tasks, depths)):
    buckets[(task, depth)].append(idx)
A, B, PT, PG = [], [], [], []
for (task, _), members in buckets.items():
    for li in range(len(members)):
        for ri in range(li + 1, len(members)):
            a, b = members[li], members[ri]
            if vstar[a] == vstar[b]:
                continue
            A.append(a); B.append(b); PT.append(task); PG.append(genes[a])
A = np.asarray(A); B = np.asarray(B)
PT = np.asarray(PT); PG = np.asarray(PG)
vs = np.asarray(vstar, dtype=np.float64)
DT = vs[B] - vs[A]                        # 참 순서 마진 (0 아님이 보장됨)
log(f"쌍 {len(A):,d} · task {len(set(PT)):,d} · 유전자 {len(set(PG))}")

# ---- task 별 root V* (R3 의 horizon 축). 자리표시자를 쓰지 않는다 ----------
root_vstar = {}
for i, r in enumerate(store.examples):
    if not r.get("prefix_tool_ids"):
        root_vstar[str(r["task_id"])] = float(r["value_star"])
missing = sorted({t for t in set(PT) if t not in root_vstar})
if missing:
    log(f"경고: root 상태를 못 찾은 task {len(missing)}개 — 그 task 의 최대 value_star 로 대체")
    for t in missing:
        root_vstar[t] = max(vs[i] for i in range(n) if tasks[i] == t)
PRV = np.asarray([root_vstar[t] for t in PT], dtype=np.float64)
log(f"root V* 범위 {PRV.min():.1f} ~ {PRV.max():.1f} (고유값 {len(np.unique(PRV))}개)")

# ---- 두 동점 규약 ----------------------------------------------------------
def conv_reference(ds, dt):
    """참조 구현: (s[a]<s[b]) == (t[a]<t[b]).  ds=s[b]-s[a], dt=t[b]-t[a]."""
    return ((ds > 0) == (dt > 0)).astype(np.float64)

def conv_tie_half(ds, dt):
    """동점 보정: 일치 1, 불일치 0, 동점 0.5."""
    out = np.where(ds * dt > 0, 1.0, 0.0)
    return np.where(ds == 0, 0.5, out)

def summarise(ds, dt, label):
    ds = ds.astype(np.float64)
    ties = int((ds == 0).sum())
    a = conv_reference(ds, dt); b = conv_tie_half(ds, dt)
    return {
        "readout": label,
        "accuracy_reference_convention": float(a.mean()),
        "accuracy_tie_corrected": float(b.mean()),
        "difference": float(b.mean() - a.mean()),
        "tie_count": ties,
        "tie_rate": ties / len(ds),
    }, a, b

margins = {}          # 이름 -> float32 마진 (s[b]-s[a])
rows_r2 = []
per_unit_ref = {}     # 이름 -> 쌍별 일치(참조 규약)
per_unit_tie = {}

# ---- 동결 공간 raw 기준선 --------------------------------------------------
log("동결 공간 raw 기준선")
raw = {"raw_cos": np.empty(n), "raw_l2": np.empty(n), "raw_dot": np.empty(n)}
for s in range(0, n, CHUNK):
    b = store.batch(sel[s:s+CHUNK], device="cpu")
    view = sorted(b["state_views"])[0]
    st = b["state_views"][view].to(torch.float64).numpy()
    gl = b["goal_views"][view].to(torch.float64).numpy()
    raw["raw_cos"][s:s+len(st)] = RO._cos(st, gl)
    raw["raw_l2"][s:s+len(st)]  = RO._l2(st, gl)
    raw["raw_dot"][s:s+len(st)] = RO._neg_dot(st, gl)
for k, v in raw.items():
    d = (v[B] - v[A])
    margins[k] = d.astype(np.float32)
    row, ra, rb = summarise(d, DT, k)
    rows_r2.append(row); per_unit_ref[k] = ra; per_unit_tie[k] = rb
    log(f"  {k:10s} ref {row['accuracy_reference_convention']:.4f} | tie-corr {row['accuracy_tie_corrected']:.4f} "
        f"| 동점 {row['tie_rate']*100:.3f}%")

# ---- 체크포인트마다 6 readout ----------------------------------------------
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

stats_recorded = {}
for energy in ENERGIES:
    for seed in SEEDS:
        p = ROOT / energy / f"seed-{seed}" / "value_geometry.pt"
        if not p.exists():
            log(f"  {energy}-{seed} 체크포인트 없음 — 건너뜀"); continue
        key = f"{energy}-{seed}"
        zs, zg, en = coords(p)
        zs64, zg64 = zs.astype(np.float64), zg.astype(np.float64)
        # dev 에서만 추정한다. 추정에 쓴 통계를 남긴다.
        both = np.concatenate([zs64, zg64], axis=0)
        std = RO.fit_standardiser(both)
        whi = RO.fit_whitener(both)
        stats_recorded[key] = {
            "standardiser_mean_norm": float(np.linalg.norm(std["mean"])),
            "standardiser_scale_min": float(std["scale"].min()),
            "standardiser_scale_max": float(std["scale"].max()),
            "whitener_frobenius": float(np.linalg.norm(whi)),
            "fitted_on": "dev z (state+goal concatenated)",
        }
        scores = {"own_energy": en}
        for r in RO.build_readouts(standardise=std, whiten=whi):
            scores[r.key] = r.fn(zs64, zg64)
        for rk, v in scores.items():
            name = f"{rk}_{key}"
            d = v[B] - v[A]
            margins[name] = d.astype(np.float32)
            row, ra, rb = summarise(d, DT, name)
            row.update({"energy": energy, "seed": seed, "readout_kind": rk})
            rows_r2.append(row); per_unit_ref[name] = ra; per_unit_tie[name] = rb
        log(f"  {energy:22s} s{seed}: " + " | ".join(
            f"{rk} {summarise(scores[rk][B]-scores[rk][A], DT, rk)[0]['accuracy_tie_corrected']:.4f}"
            for rk in ("own_energy", "cos", "l2", "neg_dot", "cos_std", "cos_whiten")))

log(f"readout 계산 완료: 마진 벡터 {len(margins)}개")

# ---- 저장: 원시 마진 (float32) ---------------------------------------------
# 작업 경로(/tmp)에 **먼저** 쓴다. 영구 경로(/root)는 프로젝트 쿼터가 작아
# ENOSPC 가 날 수 있는데, 그때 11분짜리 store 적재 결과를 통째로 잃으면 안 된다.
# 영구 복사는 best-effort 로 하고 실패를 결과에 남긴다 (조용히 넘기지 않는다).
d = DurableRoot.from_env().require()
pairs_payload = {k: v for k, v in margins.items()}
pairs_payload.update({
    "target_margin": DT.astype(np.float32),
    "root_vstar": PRV.astype(np.float32),
    "task": PT, "gene": PG,
    "pair_a": A.astype(np.int32), "pair_b": B.astype(np.int32),
})
work = Path("artifacts/acmg/findings/R2_readout_margins.npz")
work.parent.mkdir(parents=True, exist_ok=True)
tmp_w = work.with_suffix(".npz.part")
with tmp_w.open("wb") as fh:
    np.savez_compressed(fh, **pairs_payload)
tmp_w.replace(work)
size_mb = work.stat().st_size / 1e6
log(f"원시 마진 저장(작업 경로): {work} ({size_mb:.1f} MB, 벡터 {len(pairs_payload)}개)")

durable_pairs = {"path": None, "error": None, "size_mb": round(size_mb, 1)}
try:
    dp = d.write_pairs("R2_readout_margins", **pairs_payload)
    durable_pairs["path"] = str(dp)
    log(f"원시 마진 영구 복사: {dp}")
except OSError as err:
    durable_pairs["error"] = f"{type(err).__name__}: {err}"
    log(f"경고: 영구 경로에 마진을 쓰지 못했다 ({err.__class__.__name__}). "
        f"작업 경로에는 남아 있다: {work}")
    log("      /root 프로젝트 쿼터(약 1.3GB)가 원인이다. 결과 JSON 에 사실을 남긴다.")


def save_json(relative: str, payload) -> dict:
    """작업 경로에 먼저 쓰고 영구 경로로 best-effort 복사한다."""
    work = Path("artifacts/acmg") / relative
    work.parent.mkdir(parents=True, exist_ok=True)
    work.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out = {"work": str(work), "durable": None, "error": None}
    try:
        out["durable"] = str(d.write_json(relative, payload))
    except OSError as err:
        out["error"] = f"{type(err).__name__}: {err}"
        log(f"경고: 영구 경로 쓰기 실패 ({relative}) — 작업 경로에는 있다")
    return out

# =========================== R2 — readout 분해 ==============================
log("=== R2: readout 분해 ===")
gene_of_pair = PG.tolist()

def contrast_pairs(left, right, name):
    """유전자 클러스터 부트스트랩으로 짝지은 차이의 구간."""
    diff = (np.asarray(left) - np.asarray(right)).tolist()
    iv = estimators.cluster_bootstrap(diff, gene_of_pair, resamples=1000, seed=17)
    return {"name": name, "estimate": float(iv.estimate),
            "low": float(iv.low), "high": float(iv.high),
            "units": int(iv.units), "excludes_zero": bool(iv.low > 0 or iv.high < 0)}

r2 = {"split": "dev", "test_reported": False, "unit": "gene",
      "pairs": int(len(A)), "tasks": int(len(set(PT))), "genes": int(len(set(PG))),
      "dev_states_total": n_dev, "dev_states_used": n,
      "tie_convention_note": (
          "A=참조 구현((s[a]<s[b])==(t[a]<t[b]), 동점이 참 순서 방향으로 접힘), "
          "B=동점 보정(동점 0.5). 동점률이 readout 마다 다르므로 차이값에서 상쇄되지 않는다."),
      "standardiser_whitener_stats": stats_recorded,
      "per_pair_margins": durable_pairs,
      "readouts": rows_r2}

# readout 이 own_energy 의 순서를 얼마나 복원하는가 — 두 규약 모두로
recovery = {}
for energy in ENERGIES:
    for seed in SEEDS:
        key = f"{energy}-{seed}"
        if f"own_energy_{key}" not in per_unit_tie: continue
        own_t = per_unit_tie[f"own_energy_{key}"]; own_r = per_unit_ref[f"own_energy_{key}"]
        entry = {}
        for rk in ("cos", "l2", "neg_dot", "cos_std", "cos_whiten"):
            nm = f"{rk}_{key}"
            entry[rk] = {
                "tie_corrected": float(per_unit_tie[nm].mean()),
                "reference_convention": float(per_unit_ref[nm].mean()),
                "minus_own_energy_tie_corrected": contrast_pairs(per_unit_tie[nm], own_t, f"{rk}-own@{key}"),
            }
        entry["own_energy"] = {"tie_corrected": float(own_t.mean()),
                               "reference_convention": float(own_r.mean())}
        recovery[key] = entry
r2["recovery_vs_own_energy"] = recovery
save_json("findings/R2_readout_decomposition.json", r2)
log("R2 저장 완료")
for row in rows_r2:
    if row.get("readout_kind") in (None, "cos", "own_energy"):
        log(f"    {row['readout']:34s} ref {row['accuracy_reference_convention']:.4f} "
            f"tie {row['accuracy_tie_corrected']:.4f} diff {row['difference']:+.4f} 동점 {row['tie_rate']*100:.3f}%")

# =========================== R3 — horizon 층화 ==============================
log("=== R3: horizon(root V*) 층화 ===")
r3 = {"split": "dev", "unit": "gene", "moderator": "root_v_star",
      "moderator_source": "task root state value_star (자리표시자 0 을 쓰지 않는다)",
      "note": "1차 보고는 연속 기울기, 층별 표는 표시용. 경계는 dev 경험분위수.",
      "contrasts": {}}
R3_PAIRS = [("cosine", "euclidean"), ("pair_mlp", "euclidean"), ("directed_quasimetric", "euclidean"),
            ("poincare", "euclidean"), ("pair_mlp", "cosine")]
for treat_e, ctrl_e in R3_PAIRS:
    for rk in ("own_energy", "cos"):
        tv = np.mean([per_unit_tie[f"{rk}_{treat_e}-{s}"] for s in SEEDS if f"{rk}_{treat_e}-{s}" in per_unit_tie], axis=0)
        cv = np.mean([per_unit_tie[f"{rk}_{ctrl_e}-{s}"] for s in SEEDS if f"{rk}_{ctrl_e}-{s}" in per_unit_tie], axis=0)
        rep = strata.stratified_report(
            claim_id="C2", name=f"{treat_e}_minus_{ctrl_e}@{rk}",
            moderator=PRV.tolist(), treatment=tv.tolist(), control=cv.tolist(),
            clusters=gene_of_pair, moderator_name="root_v_star", n_strata=4, seed=17)
        r3["contrasts"][f"{treat_e}_minus_{ctrl_e}@{rk}"] = rep.to_payload()
        sl = rep.slope
        est = getattr(sl, "estimate", None)
        lo, hi = getattr(sl, "ci_low", None), getattr(sl, "ci_high", None)
        log(f"  {treat_e:22s} - {ctrl_e:10s} @{rk:11s} slope "
            + (f"{est:+.5f} [{lo:+.5f}, {hi:+.5f}]" if est is not None else str(sl)[:80]))
save_json("findings/R3_horizon_strata.json", r3)
log("R3 저장 완료")

# =========================== R4 — cross-model 통제 ==========================
log("=== R4: cross-model 통제 (동점 보정된 정렬로 재계산) ===")
task_arr = PT
uniq_tasks = sorted(set(task_arr.tolist()))
task_gene = {t: g for t, g in zip(task_arr.tolist(), PG.tolist())}

def per_task_mean(vec):
    s = defaultdict(float); c = defaultdict(int)
    for t, v in zip(task_arr.tolist(), vec):
        s[t] += float(v); c[t] += 1
    return {t: s[t]/c[t] for t in s}

def per_task_policy(energy, seed):
    p = ROOT / energy / f"seed-{seed}" / "value_metrics.json"
    return json.load(open(p))["metrics"]["dev"]["per_task_policy_accuracy"]

ordering_by_run = {f"{e}-{s}": per_task_mean(per_unit_tie[f"own_energy_{e}-{s}"])
                   for e in ENERGIES for s in SEEDS if f"own_energy_{e}-{s}" in per_unit_tie}
policy_by_run = {}
for e in ENERGIES:
    for s in SEEDS:
        try: policy_by_run[f"{e}-{s}"] = per_task_policy(e, s)
        except Exception as err: log(f"  policy 없음 {e}-{s}: {type(err).__name__}")

runs = sorted(set(ordering_by_run) & set(policy_by_run))
log(f"  런 {len(runs)}개로 {len(runs)**2} 조합")

matched, mismatched, clusters_r4 = [], [], []
for t in uniq_tasks:
    g = task_gene[t]
    mv = [ordering_by_run[r].get(t) for r in runs]
    if any(v is None for v in mv): continue
    # 일치쌍 A=B : 같은 런의 정렬 x 그 런의 정책
    same = [ordering_by_run[r][t] * policy_by_run[r].get(t, np.nan) for r in runs]
    # 불일치쌍 A!=B : 런 r 의 정렬 x 다음 런의 정책 (순환 이동)
    diff = [ordering_by_run[r][t] * policy_by_run[runs[(i+1) % len(runs)]].get(t, np.nan)
            for i, r in enumerate(runs)]
    same = [v for v in same if v == v]; diff = [v for v in diff if v == v]
    if not same or not diff: continue
    matched.append(float(np.mean(same))); mismatched.append(float(np.mean(diff)))
    clusters_r4.append(g)

rep = control.crossmodel_control(
    statistic_name="ordering_x_policy_product (tie-corrected ordering)",
    matched=matched, mismatched=mismatched, clusters=clusters_r4, resamples=2000, seed=17)
r4 = {"split": "dev", "unit": "gene", "n_tasks": len(matched),
      "note": ("R4 입력이 R2 와 같은 측도(깊이고정 순서 일치율)임을 확인하고 "
               "동점 보정된 정렬로 재계산했다. 기존 c5_within_model.json 은 "
               "int8 로 접힌 npz 를 평균해 같은 편향 위에 있었다."),
      "report": rep.__dict__ if hasattr(rep, "__dict__") else str(rep)}
save_json("findings/R4_crossmodel_control.json", r4)
log(f"R4 저장 완료: matched {rep.matched:.4f} {rep.matched_ci}  mismatched {rep.mismatched:.4f} {rep.mismatched_ci}")
log(f"          차이 {rep.difference:.4f} {rep.difference_ci}  n_units={rep.n_units}")
log("=== 전체 완료 %.1f 분 ===" % ((time.time()-t0)/60))

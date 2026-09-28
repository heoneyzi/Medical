#!/usr/bin/env python3
"""C5 의 모델 내부 검정: 표현이 잘 정렬하는 task 가 계획도 잘 되는가.

C5 는 "계획기 타당성과 표현 타당성은 같은 주장"이다. 앞선 분석은 에너지 5개를 단위로
상관을 봤는데 n=5 는 근거가 되지 못한다. 같은 질문을 **모델 안에서 task 를 단위로** 물으면
n=584 가 된다.

  task t 마다:  x_t = 그 task 의 쌍에서의 깊이고정 정렬 정확도
                y_t = 그 task 의 dev policy 정확도 (value_metrics.json 의 per_task_policy_accuracy)

표현 수준(x)과 계획 수준(y)이 같은 축이라면 task 를 가로질러 함께 움직여야 한다.
상관의 구간은 유전자 클러스터 부트스트랩으로 낸다(사전등록 분석과 같은 단위).

저장된 산출물만 쓴다 — store 적재 없음. src/ 수정 없음. dev 한정, test 봉인.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")

NPZ = "artifacts/acmg/findings/decomposition_pairs.npz"
ROOT = Path("artifacts/acmg/checkpoints/geometry_comparison")
ENERGIES = ["cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp"]
SEEDS = [17, 29, 43]
OUT = Path("artifacts/acmg/findings/c5_within_model.json")

z = np.load(NPZ, allow_pickle=True)
task = z["task"].astype(str); gene = z["gene"].astype(str)
task_gene = {}
for t, g in zip(task, gene): task_gene[t] = g
print(f"  쌍 {len(task):,d} · task {len(set(task)):,d} · 유전자 {len(set(gene))}")

def per_task_ordering(vec):
    s = defaultdict(float); n = defaultdict(int)
    for t, c in zip(task, vec):
        s[t] += float(c); n[t] += 1
    return {t: s[t]/n[t] for t in s}, dict(n)

def per_task_policy(energy, seed):
    p = ROOT / energy / f"seed-{seed}" / "value_metrics.json"
    m = json.load(open(p))
    return m["metrics"]["dev"]["per_task_policy_accuracy"]

def pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def gene_bootstrap_corr(pairs, genes, reps=2000, seed=17):
    """유전자를 재표집해 상관의 구간을 낸다."""
    by = defaultdict(list)
    for (x, y), g in zip(pairs, genes): by[g].append((x, y))
    gs = sorted(by)
    point = pearson([p[0] for p in pairs], [p[1] for p in pairs])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(gs), size=len(gs))
        xs, ys = [], []
        for i in pick:
            for x, y in by[gs[i]]: xs.append(x); ys.append(y)
        r = pearson(xs, ys)
        if r == r: draws.append(r)
    if not draws: return {"r": point, "low": None, "high": None, "genes": len(gs)}
    return {"r": point, "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)), "genes": len(gs)}

res = {"split": "dev", "test_reported": False, "unit_for_interval": "gene",
       "note": ("task 를 단위로 표현 수준 정렬과 계획 성능의 상관을 본다. "
                "상관의 구간은 유전자 클러스터 재표집으로 낸다. 탐색적 분석이다."),
       "per_run": {}}
rows = []
for e in ENERGIES:
    for s in SEEDS:
        k = f"{e}-{s}"
        key = f"energy_{k}"
        if key not in z: print(f"  {k}: 벡터 없음 — 건너뜀"); continue
        ordt, npairs = per_task_ordering(z[key].astype(float))
        pol = per_task_policy(e, s)
        common = sorted(set(ordt) & set(pol))
        pts = [(ordt[t], float(pol[t])) for t in common]
        gs = [task_gene[t] for t in common]
        bs = gene_bootstrap_corr(pts, gs)
        res["per_run"][k] = {"energy": e, "seed": s, "tasks": len(common), **bs}
        rows.append((e, s, bs["r"], bs["low"], bs["high"], len(common)))
        print(f"  {e:<22s} s{s}: r={bs['r']:+.3f} [{bs['low']:+.3f}, {bs['high']:+.3f}]  task {len(common)}  유전자 {bs['genes']}")

# 동결 raw 기준선으로도 같은 검정 (통제)
print()
for base in ["raw_cos", "raw_l2"]:
    ordt, _ = per_task_ordering(z[base].astype(float))
    pol = per_task_policy("cosine", 17)   # 계획 성능은 임의 고정 — raw 정렬과 무관해야 한다
    common = sorted(set(ordt) & set(pol))
    pts = [(ordt[t], float(pol[t])) for t in common]
    gs = [task_gene[t] for t in common]
    bs = gene_bootstrap_corr(pts, gs)
    res.setdefault("frozen_baseline", {})[base] = {"tasks": len(common), **bs}
    print(f"  통제: {base:<8s} 정렬 vs cosine-17 policy: r={bs['r']:+.3f} [{bs['low']:+.3f}, {bs['high']:+.3f}]")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
print(f"\n  저장: {OUT}")

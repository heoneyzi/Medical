#!/usr/bin/env python3
"""용량 대조의 짝지은 구간.

집계 코드(value.py paired_task_effect)와 같은 순서로 계산한다:
  시드별로 같은 dev task 의 per-task 값을 짝지어 차이를 내고,
  매칭된 시드에 대해 평균한 뒤, 독립 dev task 를 재표집한다.

비교:
  A. pair_mlp@h111 (335,530) − euclidean@h128 (335,942)   <- 치료군이 파라미터가 적다
  B. pair_mlp@h128 (369,479) − euclidean@h147 (368,983)   <- 대조군에 용량을 줬다
A 에서 이기고 B 에서도 euclidean 이 못 따라잡으면 용량 설명은 죽는다.

읽기 전용. src/ 를 수정하지 않는다.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from geoflowagent.training.value import _bootstrap_task_mean

GEO = Path("artifacts/acmg/checkpoints/geometry_comparison")
CAP = Path("artifacts/acmg/checkpoints/capacity_matched")
SEEDS = [17, 29, 43]
KEYS = [("per_task_policy_accuracy", "policy", True),
        ("per_task_joint_accuracy", "joint", True),
        ("per_task_regret_at_1", "regret@1", False)]

def load(run_dir, seed):
    p = Path(run_dir) / f"seed-{seed}" / "value_metrics.json"
    if not p.exists(): return None
    m = json.load(open(p))
    return m["metrics"]["dev"], m["parameter_count"]

def contrast(left_dir, right_dir, label):
    print(f"\n{'='*88}\n  {label}\n{'='*88}")
    pl = pr = None
    ok = True
    for seed in SEEDS:
        if load(left_dir, seed) is None or load(right_dir, seed) is None:
            print(f"  seed {seed}: 아직 없음 — 건너뜀"); ok = False
    for key, name, higher in KEYS:
        diffs_by_task = {}
        used = 0
        for seed in SEEDS:
            L = load(left_dir, seed); R = load(right_dir, seed)
            if L is None or R is None: continue
            used += 1
            dl, pl = L; dr, pr = R
            a, b = dl.get(key), dr.get(key)
            if a is None or b is None:
                print(f"  {name}: 키 {key} 없음"); break
            for t in set(a) & set(b):
                diffs_by_task.setdefault(t, []).append(float(a[t]) - float(b[t]))
        if not diffs_by_task: continue
        # 참조 구현과 동일: {task_id: [시드별 차이]} 를 그대로 넘긴다.
        # 함수가 task 안에서 먼저 평균하고, task 단위로 재표집한다.
        tasks = sorted(diffs_by_task)
        iv = _bootstrap_task_mean(diffs_by_task, reps=2000, seed=840_000)
        if iv is None:
            print(f"  {name}: 재표집 불가"); continue
        m, lo, hi = iv["mean"], iv["low"], iv["high"]
        good = (m > 0) if higher else (m < 0)
        excl = (lo > 0 or hi < 0)
        print(f"  {name:<10s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  task {len(tasks)}  시드 {used}  "
              f"{'0 배제' if excl else '0 포함 -> UNRESOLVED'}  {'(치료군 유리)' if good else '(대조군 유리)'}")
    if pl and pr:
        print(f"  파라미터: 좌 {pl:,d} · 우 {pr:,d}  (차 {pl-pr:+,d})")
    return ok

contrast(CAP/"pair_mlp-h111", GEO/"euclidean", "A. pair_mlp@h111(335,530) − euclidean@h128(335,942)  [치료군이 파라미터 적음]")
contrast(GEO/"pair_mlp", CAP/"euclidean-h147", "B. pair_mlp@h128(369,479) − euclidean@h147(368,983)  [대조군에 용량 부여]")
contrast(CAP/"pair_mlp-h111", GEO/"pair_mlp", "C. pair_mlp@h111(335,530) − pair_mlp@h128(369,479)  [같은 에너지, 용량만 다름]")

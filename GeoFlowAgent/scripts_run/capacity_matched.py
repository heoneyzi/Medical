#!/usr/bin/env python3
"""용량 혼입을 깨는 대조 실험.

문제: 스윕 1위 pair_mlp 는 369,479 파라미터로 euclidean(335,942)보다 +33,537(+10.0%) 많다.
그 차이는 정확히 PairMLPGoalEnergy 헤드의 크기다
(LayerNorm(256)=512 + Linear(256,128)=32,896 + Linear(128,1)=129 = 33,537).
따라서 "거리 공리를 버려서 이겼다"와 "파라미터가 많아서 이겼다"가 분리되지 않는다.

설계 — 양방향으로 막는다:
  A. 치료군 불리:  pair_mlp  @ hidden_dim=111 -> 335,530  (euclidean 보다 412개 적다)
  B. 대조군 유리:  euclidean @ hidden_dim=147 -> 368,983  (pair_mlp 보다 496개 적다)
A 에서도 pair_mlp 가 이기고 B 에서도 euclidean 이 못 따라잡으면 용량 설명은 죽는다.

src/ 를 건드리지 않는다. train_value_geometry 를 그대로 호출하므로
source_tree_sha256 가 유지되고 기존 15런과 과학적으로 비교 가능하다.
SearchFeatureStore 를 한 번만 만들어 6런에 공유한다(메모리 1벌, 로딩 1회).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "src")
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training.value import train_value_geometry
from geoflowagent.utils.io import read_yaml

CONFIG = "configs/geoacmg.yaml"
PROCESSED = "artifacts/acmg/processed"
CACHE = "artifacts/acmg/cache"
OUT = Path("artifacts/acmg/checkpoints/capacity_matched")
SEEDS = [17, 29, 43]
ARMS = [
    ("pair_mlp", 111),    # 335,530  <= euclidean 335,942
    ("euclidean", 147),   # 368,983  <= pair_mlp 369,479
]

cfg = read_yaml(CONFIG)
tc = cfg.get("value_training", {})
store = SearchFeatureStore(PROCESSED, CACHE, structured_dim=int(tc.get("structured_dim", 64)))
print(f"store 적재 완료: train={len(store.indices('train'))} dev={len(store.indices('dev'))}", flush=True)

for energy, hidden in ARMS:
    for seed in SEEDS:
        run_dir = OUT / f"{energy}-h{hidden}" / f"seed-{seed}"
        if (run_dir / "value_metrics.json").exists():
            print(f"skip {energy}-h{hidden} seed-{seed} (완료됨)", flush=True)
            continue
        print(f"=== {energy} hidden_dim={hidden} seed={seed} 시작", flush=True)
        r = train_value_geometry(
            PROCESSED, CACHE, run_dir, CONFIG,
            energy_override=energy,
            seed_override=seed,
            include_test=False,                       # test 봉인 유지
            model_capacity_override={"hidden_dim": hidden},
            _store_override=store,
        )
        dev = r["metrics"]["dev"]
        print(f"    params={r['parameter_count']:,d} "
              f"policy={dev['policy_optimal_set_accuracy']:.4f} "
              f"joint={dev['joint_stop_action_accuracy']:.4f} "
              f"regret={dev['regret_at_1']:.4f}", flush=True)
print("ALL DONE", flush=True)

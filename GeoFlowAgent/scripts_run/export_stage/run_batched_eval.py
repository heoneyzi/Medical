"""배치 평가기를 원본과 대조하고, 같으면 dev 584 루트 전수 평가를 돌린다.

store 적재가 11분이므로 한 프로세스에서 대조와 본실행을 모두 한다.
"""
import json, os, time
from pathlib import Path
import torch

T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

from geoflowagent.utils.io import read_yaml
from geoflowagent.utils.reproducibility import choose_device, seed_everything
from geoflowagent.training.search_features import SearchFeatureStore
from geoflowagent.training import state_flow as SF
from geoflowagent.geoacmg import flowfast_eval

cfg = read_yaml("configs/geoacmg.yaml")
tc = dict(cfg["state_flow_training"])
seed = int(cfg.get("seed", 17))
seed_everything(seed)
device = choose_device(str(cfg.get("device", "auto")))

store = SearchFeatureStore(Path("artifacts/acmg/processed"), Path("artifacts/acmg/cache"),
                           structured_dim=int(tc["structured_dim"]))
log("store loaded: %d examples" % len(store.examples))
path_store = SF.SearchPathStore(store, beta=float(tc["path_beta"]),
                                max_path_length=int(tc["max_plan_length"]) - 1,
                                allowed_splits=("train", "dev"))
features = SF.StateFlowFeatureSpace(store)
model = SF.SearchStateFlow(**features.dimensions, dim=int(tc["dim"]),
    max_plan_length=int(tc["max_plan_length"]), hidden_dim=int(tc["hidden_dim"]),
    layers=int(tc["layers"]), heads=int(tc["heads"]), dropout=float(tc["dropout"]),
    tool_mix=float(tc["tool_mix"]), projection_seed=int(tc["projection_seed"])).to(device)
ck = torch.load("artifacts/acmg/checkpoints/state_flow/training_resume.pt",
                map_location=device, weights_only=False)
model.load_state_dict(ck["best_state"]); model.eval()
log("model loaded (best_epoch=%s, dev loss %.4f)" % (ck["best_epoch"], ck["best_loss"]))

dev_refs = SF._all_references(path_store, "dev")
t = time.time()
threshold, cal = SF._calibrate_stop_threshold(
    model, dev_refs, features, device, grid=tc["stop_threshold_grid"],
    batch_size=int(tc["batch_size"]), pad_weight=float(tc["pad_weight"]), seed=seed + 200_000)
log("STOP threshold calibrated = %s  (%.1fs)" % (threshold, time.time()-t))

nfe = int(tc["nfe"]); samples = int(tc["evaluation_samples_per_state"])
dev_idx = path_store.eligible_indices("dev")
dev_roots = [i for i in dev_idx if not store.examples[i].get("prefix_tool_ids")]
log("dev roots = %d" % len(dev_roots))

orig_eligible = path_store.eligible_indices
def restrict(subset):
    path_store.eligible_indices = lambda split=None, _s=tuple(subset): _s

def numeric_leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out

# ---------------- 대조 ----------------
K = int(os.environ.get("K", "24"))
subset = dev_roots[:K]
log("=== EQUIVALENCE on %d dev roots ===" % K)

restrict(subset); t = time.time()
ref = SF.evaluate_search_state_flow(model, store, path_store, "dev", device,
        samples_per_state=samples, nfe=nfe, stop_threshold=threshold, seed=seed, root_only=True)
t_ref = time.time() - t
log("ORIGINAL evaluate_search_state_flow : %.1fs  (%.2f s/root)" % (t_ref, t_ref/K))

restrict(subset); t = time.time()
cand = flowfast_eval.evaluate_batched(model, store, path_store, "dev", device,
        samples_per_state=samples, nfe=nfe, stop_threshold=threshold, seed=seed, root_only=True)
t_bat = time.time() - t
log("BATCHED  evaluate_batched          : %.1fs  (%.2f s/root)" % (t_bat, t_bat/K))
log("SPEEDUP on full evaluation: %.1fx" % (t_ref / max(t_bat, 1e-9)))

a, b = numeric_leaves(ref), numeric_leaves(cand)
keys = sorted(set(a) & set(b))
diffs = [(k, a[k], b[k]) for k in keys if abs(a[k] - b[k]) > 1e-9]
log("metrics compared: %d   mismatched: %d" % (len(keys), len(diffs)))
for k, x, y in diffs[:25]:
    log("   DIFF %-62s orig=%.6f batched=%.6f" % (k, x, y))
only = sorted(set(a) ^ set(b))
if only:
    log("keys only on one side: %s" % only[:10])

equivalent = not diffs and not only
log("EQUIVALENT: %s" % equivalent)

out = {
    "measured_at": "2026-09-20",
    "stop_threshold": threshold,
    "dev_roots_total": len(dev_roots),
    "equivalence_subset": K,
    "original_seconds": round(t_ref, 2),
    "batched_seconds": round(t_bat, 2),
    "speedup": round(t_ref / max(t_bat, 1e-9), 2),
    "metrics_compared": len(keys),
    "metrics_mismatched": len(diffs),
    "mismatches": [{"key": k, "original": x, "batched": y} for k, x, y in diffs[:50]],
    "equivalent": equivalent,
}
Path(os.path.expandvars("$GEOWORK/flowfast/batched_equivalence.json")).write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
log("wrote batched_equivalence.json")

# ---------------- 전수 ----------------
if equivalent:
    log("=== FULL dev root-only evaluation (%d roots, batched) ===" % len(dev_roots))
    path_store.eligible_indices = orig_eligible
    t = time.time()
    full = flowfast_eval.evaluate_batched(model, store, path_store, "dev", device,
            samples_per_state=samples, nfe=nfe, stop_threshold=threshold, seed=seed, root_only=True)
    t_full = time.time() - t
    log("FULL evaluation done in %.1fs (%.1f min)" % (t_full, t_full/60))
    full["_runtime_seconds"] = round(t_full, 1)
    full["_stop_threshold_calibration"] = cal
    full["_checkpoint"] = {"best_epoch": int(ck["best_epoch"]), "best_loss": float(ck["best_loss"])}
    full["_protocol"] = ("root-only (task 당 독립 루트 1개), train 롤아웃 미보고. "
                         "전수 평가와 다른 프로토콜이다.")
    Path(os.path.expandvars("$GEOWORK/flowfast/dev_root_only_full.json")).write_text(
        json.dumps(full, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("wrote dev_root_only_full.json")
    log("goal_completion open_loop=%.4f  replan=%.4f  guarded=%.4f  blind=%.4f" % (
        full["open_loop"]["goal_completion_rate"],
        full["receding_horizon"]["goal_completion_rate"],
        full["receding_horizon_verifier_stop_guard"]["goal_completion_rate"],
        full["compute_matched_blind_replanning"]["goal_completion_rate"]))
else:
    log("NOT EQUIVALENT — 전수 평가를 돌리지 않는다. 불일치를 먼저 고친다.")
log("DONE total %.1f min" % ((time.time()-T0)/60))

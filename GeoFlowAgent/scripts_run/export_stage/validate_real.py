"""flowfast 를 실제 store·체크포인트 위에서 검증하고 실측 배속을 잰다.

합성 테스트가 아니라 진짜 dev 루트로 원본과 배치본을 같은 시드에서 비교한다.
결과 판정이 아니라 엔지니어링 검증이므로 학습은 건드리지 않는다.
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
from geoflowagent.geoacmg import flowfast

cfg = read_yaml("configs/geoacmg.yaml")
tc = dict(cfg["state_flow_training"])
seed = int(cfg.get("seed", 17))
seed_everything(seed)
device = choose_device(str(cfg.get("device", "auto")))
log("device", device)

store = SearchFeatureStore(Path("artifacts/acmg/processed"), Path("artifacts/acmg/cache"),
                           structured_dim=int(tc["structured_dim"]))
log("store loaded: examples=%d tools=%d" % (len(store.examples), len(store.tool_ids)))

path_store = SF.SearchPathStore(store, beta=float(tc["path_beta"]),
                                max_path_length=int(tc["max_plan_length"]) - 1,
                                allowed_splits=("train", "dev"))
dev_idx = path_store.eligible_indices("dev")
dev_roots = [i for i in dev_idx if not store.examples[i].get("prefix_tool_ids")]
log("dev eligible=%d  dev ROOTS=%d" % (len(dev_idx), len(dev_roots)))

features = SF.StateFlowFeatureSpace(store)
model = SF.SearchStateFlow(**features.dimensions, dim=int(tc["dim"]),
    max_plan_length=int(tc["max_plan_length"]), hidden_dim=int(tc["hidden_dim"]),
    layers=int(tc["layers"]), heads=int(tc["heads"]), dropout=float(tc["dropout"]),
    tool_mix=float(tc["tool_mix"]), projection_seed=int(tc["projection_seed"])).to(device)
ck = torch.load("artifacts/acmg/checkpoints/state_flow/training_resume.pt",
                map_location=device, weights_only=False)
model.load_state_dict(ck["best_state"]); model.eval()
log("model loaded: best_epoch=%s best_loss=%.4f" % (ck["best_epoch"], ck["best_loss"]))

nfe = int(tc["nfe"]); samples = int(tc["evaluation_samples_per_state"])
K = int(os.environ.get("K", "64"))
rows = list(dev_roots[:K])
log("validating on %d dev roots, samples=%d nfe=%d" % (len(rows), samples, nfe))

def gens():   # state_flow.py:894 과 동일한 행별 시드
    return {r: torch.Generator(device="cpu").manual_seed(seed + 10_000 * int(r)) for r in rows}

# ---- 원본 경로 ----
g = gens(); t = time.time()
ref = {r: SF._sample_plans(model, features, r, device, samples=samples, nfe=nfe, generator=g[r])
       for r in rows}
t_ref = time.time() - t
log("ORIGINAL _sample_plans : %.2fs  (%.1f ms/root)" % (t_ref, 1e3*t_ref/len(rows)))

# ---- 배치 경로 ----
g = gens(); t = time.time()
cand = flowfast.batch_sample_plans(model, features, rows, device,
                                   samples=samples, nfe=nfe, generators=g)
t_bat = time.time() - t
log("BATCHED batch_sample_plans: %.2fs  (%.1f ms/root)" % (t_bat, 1e3*t_bat/len(rows)))

rep = flowfast.compare_plans(ref, cand)
log("EQUIVALENCE: decisions_identical=%s  %d/%d  max_abs_diff=%.3e"
    % (rep.decisions_identical, rep.decision_matches, rep.checked, rep.max_absolute_difference))

speed = t_ref / max(t_bat, 1e-9)
log("MEASURED SPEEDUP (plan sampling): %.1fx" % speed)

proj = flowfast.projected_speedup(len(dev_roots), samples=samples, nfe=nfe,
                                  rollout_kinds=3, max_steps=int(tc["max_plan_length"]) * 2)
log("PROJECTED over %d dev roots: %s" % (len(dev_roots), json.dumps(proj, ensure_ascii=False)))

out = {
    "measured_at": "2026-09-20",
    "device": str(device),
    "dev_roots_total": len(dev_roots),
    "validated_on_roots": len(rows),
    "samples_per_state": samples, "nfe": nfe,
    "plan_sampling": {
        "original_seconds": round(t_ref, 3),
        "batched_seconds": round(t_bat, 3),
        "measured_speedup": round(speed, 2),
    },
    "equivalence": rep.to_payload(),
    "projected_call_reduction": proj,
    "checkpoint": {"best_epoch": int(ck["best_epoch"]), "best_loss": float(ck["best_loss"])},
}
Path(os.path.expandvars("$GEOWORK/flowfast/real_validation.json")).write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
log("wrote $GEOWORK/flowfast/real_validation.json")

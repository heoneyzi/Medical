#!/usr/bin/env python3
"""Verify later seven-seed/readout/ablation artifacts without Torch or test access.

Usage: python verify_geoacmg_latest.py RESULTS/geoacmg_working_findings --output result.json
Requires numpy and sibling verify_geoacmg_metrics.py. Checkpoint defaults to sibling
RESULTS/geoacmg_latest_checkpoints. Large raw inputs and checkpoints are never loaded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from verify_geoacmg_metrics import bootstrap_mean, concordance, grouped, load_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "results/geoacmg_working_findings")
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.root
    cp = a.checkpoints or root.parent / "geoacmg_latest_checkpoints"
    checks, notes = [], []
    result = {"schema": "geoflowagent.latest-metrics-verification.v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Saved dev artifacts only; no training, model inference or test access.", "numpy_version": np.__version__}
    def check(name, actual, stored, tolerance=1e-9):
        delta = abs(float(actual) - float(stored))
        checks.append({"name": name, "actual": float(actual), "stored": float(stored), "tolerance": tolerance, "abs_error": delta, "pass": bool(np.isfinite(delta) and delta <= tolerance)})
    def interval(name, value, stored):
        check(name + ".estimate", value["estimate"], stored.get("est", stored.get("estimate")))
        for i in (0, 1):
            check(name + ".ci" + str(i), value["ci"][i], stored["ci"][i], 1e-7)
    with np.load(root / "R2_readout_margins.npz", allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    metadata = {"target_margin", "root_vstar", "task", "gene", "pair_a", "pair_b"}
    hit = {k: concordance(v, m["target_margin"]) for k, v in m.items() if k not in metadata}
    result["sample"] = {"pairs": len(m["task"]), "tasks": len(set(m["task"])), "genes": len(set(m["gene"])), "npz_arrays": len(m), "readouts": len(hit), "readouts_with_any_tie": [k for k in hit if np.any(m[k] == 0)]}
    seeds = [17, 29, 43, 59, 71, 83, 97]
    r2 = load_json(root / "R2f_readout_7seeds.json")
    out = {}
    for kind, row in r2["by_readout"].items():
        left = np.array([hit[f"{kind}_cosine-{s}"] for s in seeds])
        right = np.array([hit[f"{kind}_euclidean-{s}"] for s in seeds])
        sd = left.mean(axis=1) - right.mean(axis=1)
        seed_iv = bootstrap_mean(sd, range(len(seeds)))
        gene_iv = bootstrap_mean((left-right).mean(axis=0), m["gene"])
        check("R2f." + kind + ".cosine", left.mean(), row["cosine"])
        check("R2f." + kind + ".euclidean", right.mean(), row["euclidean"])
        interval("R2f." + kind + ".seed", seed_iv, row["seed"])
        interval("R2f." + kind + ".gene", gene_iv, row["gene"])
        out[kind] = {"cosine": float(left.mean()), "euclidean": float(right.mean()), "seed": seed_iv, "gene": gene_iv}
    result["R2f_seven_seeds"] = out

    metrics = {}
    for base in ("geometry_comparison", "geometry_r1_pairs", "geometry_r1_pairs2"):
        for path in (cp / base).glob("*/seed-*/value_metrics.json"):
            data = load_json(path)
            metrics[(data["energy"], data["seed"])] = data["metrics"]["dev"]
            check("test_flag." + base + "." + data["energy"] + "." + str(data["seed"]), data["test_reported"], False)
    task_keys = sorted(set(map(str, m["task"])))
    task_gene = dict(zip(map(str, m["task"]), map(str, m["gene"])))
    task_labels = [task_gene[t] for t in task_keys]
    cols = {"policy": "per_task_policy_accuracy", "joint": "per_task_joint_accuracy", "regret@1": "per_task_regret_at_1"}
    r3 = load_json(root / "R3c_planning_axis_7seeds.json")
    out = {}
    for name, col in cols.items():
        left = np.array([[metrics[("cosine", s)][col][t] for t in task_keys] for s in seeds])
        right = np.array([[metrics[("euclidean", s)][col][t] for t in task_keys] for s in seeds])
        # Recorded seed statistic is difference of task-macro run means.
        sd = left.mean(axis=1) - right.mean(axis=1)
        seed_iv = bootstrap_mean(sd, range(len(seeds)))
        gene_iv = bootstrap_mean((left-right).mean(axis=0), task_labels)
        interval("R3c." + name + ".seed", seed_iv, r3["axes"][name]["seed"])
        interval("R3c." + name + ".gene", gene_iv, r3["axes"][name]["gene"])
        out[name] = {"seed": seed_iv, "gene": gene_iv, "per_seed": dict(zip(map(str,seeds),sd.tolist()))}
    result["R3c_seven_seed_planning"] = out

    r10 = load_json(root / "R10_all_families_neutral_readout.json")
    for family, row in r10["family_means"].items():
        for kind in ("own_energy", "cos_std", "l2", "cos_whiten"):
            value = float(np.mean([hit[f"{kind}_{family}-{s}"].mean() for s in row["seeds"]]))
            check(f"R10.{family}.{kind}", value, row[kind])
    r10_out = {}
    for contrast, row in r10["pairwise"].items():
        families, kind = contrast.split("@")
        left, right = families.split("_minus_")
        shared = sorted(set(r10["family_means"][left]["seeds"]) & set(r10["family_means"][right]["seeds"]))
        diff = np.mean([hit[f"{kind}_{left}-{s}"] - hit[f"{kind}_{right}-{s}"] for s in shared], axis=0)
        iv = bootstrap_mean(diff, m["gene"])
        interval("R10." + contrast, iv, row)
        r10_out[contrast] = {**iv, "matched_seeds": shared}
    result["R10_matched_seed_family_contrasts"] = r10_out

    # R8/R9 have summaries but not the feature-level readout outputs required to
    # independently recreate fitted PCA/whitening. Verify arithmetic and cross-file
    # agreement only, and keep this distinct from the NPZ-based checks above.
    r8, r9 = load_json(root / "R8_frozen_space_ladder.json"), load_json(root / "R9_information_source_ablation.json")
    for source, payload in (("R8", r8), ("R9", r9)):
        for arm, readouts in payload["results"].items():
            for kind, row in readouts.items():
                if kind.startswith("_"):
                    continue
                chance = row["minus_chance"]
                if isinstance(chance, dict):
                    chance = chance["estimate"]
                check(f"{source}.{arm}.{kind}.chance_arithmetic", row["accuracy"] - 0.5, chance)
    for kind, margin in (("cos","raw_cos"),("l2","raw_l2"),("neg_dot","raw_dot")):
        check("R8.raw."+kind, hit[margin].mean(), r8["results"]["views_only"][kind]["accuracy"])
    result["R8_R9_saved_summary_audit"] = {"verification": "Arithmetic and available raw three readouts only; fitted feature readouts and their CIs are stored evidence, not independently regenerated.", "R8_MedCPT_only": {k:v for k,v in r8["results"]["views_only"].items() if k in ("l2_std (train-fit)","cos_whiten (train-fit)","random-64 + cos")}, "R9_PCA64_accuracy": {k: v["PCA-64 + cos"]["accuracy"] for k,v in r9["results"].items()}}
    notes.append("R9/R11 narrative says any linear frozen readout is chance and MedCPT contribution is zero. R8's standardized L2/whitening/random-projection results contradict that blanket wording; similar point estimates do not prove equivalence.")

    r11 = load_json(root / "R11_medcpt_ablation.json")
    ablations = {}
    for base in ("geometry_ablate_medcpt", "geometry_ablate_medcpt2"):
        for path in (cp / base).glob("*/seed-*/value_metrics.json"):
            data = load_json(path)
            ablations[(data["energy"], data["seed"])] = data["metrics"]["dev"]
        # A saved aggregate can retain the exact dev vectors when a leaf JSON is absent.
        agg = cp / base / "comparison.json"
        if agg.is_file():
            data = load_json(agg)
            for run in data.get("runs", []):
                ablations.setdefault((run["energy"],run["seed"]),run["dev"])
    original = [(family, seed) for family in ("cosine", "euclidean") for seed in (17,29)]
    missing = [f"{f}-{s}" for f,s in original if (f,s) not in ablations]
    if missing:
        notes.append("R11 original four-run aggregate cannot be fully recomputed: missing ablation dev vectors for " + ", ".join(missing))
    r11_out = {"original_four_run_missing_vectors": missing, "original_stored_axes": r11["axes"]}
    if not missing:
        verified = {}
        for name,col in cols.items():
            left = np.array([[ablations[run][col][t] for t in task_keys] for run in original])
            right = np.array([[metrics[run][col][t] for t in task_keys] for run in original])
            diff = left-right
            run_iv = bootstrap_mean(left.mean(1)-right.mean(1), range(len(original)))
            gene_iv = bootstrap_mean(diff.mean(0),task_labels)
            interval("R11."+name+".run",run_iv,r11["axes"][name]["run_clustered"])
            interval("R11."+name+".gene",gene_iv,r11["axes"][name]["gene_clustered"])
            verified[name] = {"run":run_iv,"gene":gene_iv}
        r11_out["recomputed_original_four_run"] = verified
    extended = [run for run in sorted(ablations) if run in metrics]
    if len(extended)>4:
        derived = {}
        for name,col in cols.items():
            difference = np.array([[ablations[run][col][t]-metrics[run][col][t] for t in task_keys] for run in extended])
            derived[name] = {"run":bootstrap_mean(difference.mean(1),range(len(extended))),"gene":bootstrap_mean(difference.mean(0),task_labels)}
        r11_out["curation_derived_available_run_extension"] = {"label":"New archive-curation calculation, not a historical reported R11 finding; available runs only.","runs":[f"{f}-{s}" for f,s in extended],"axes":derived}
    result["R11"] = r11_out
    notes.append("Working R3/R4 JSON files retain an earlier subset analysis, despite other working R2/R3c/R8-R11 files being later. Chronology must be resolved per artifact; the export's recomputed 19-run R4 remains distinct.")
    result["notes"] = notes
    result["checks"] = checks
    failed = [c for c in checks if not c["pass"]]
    result["status"] = "PASS" if not failed else "FAIL"
    result["summary"] = {"checks":len(checks),"passed":len(checks)-len(failed),"failed":len(failed)}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],**result["summary"],"failures":failed,"notes":notes},ensure_ascii=True,indent=2))
    return int(bool(failed))

if __name__ == "__main__":
    raise SystemExit(main())

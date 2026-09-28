#!/usr/bin/env python3
"""Recompute GeoACMG saved-result statistics without training or opening test.

Requires Python 3.10+ and NumPy; no Torch, SciPy, network or original machine paths.
ROOT contains findings/, pairs/, and checkpoints_small/ (JSON metrics only suffice).
Use --checkpoints to supply geometry_comparison/ and geometry_r1_pairs/ elsewhere.
Checks estimates and saved intervals, and exits nonzero if numerical evidence differs.
The original README is historical: known stale prose is reported, never treated as truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grouped(values, labels, *, sorted_keys=True):
    keys = sorted(set(map(str, labels))) if sorted_keys else list(dict.fromkeys(map(str, labels)))
    index = {k: i for i, k in enumerate(keys)}
    inverse = np.fromiter((index[str(k)] for k in labels), dtype=np.int64)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    return keys, inverse, sums, counts


def bootstrap_mean(values, labels, *, resamples=2000, seed=17, bca=True):
    """Same cluster resampling and BCa specification; sufficient sums avoid row expansion."""
    values = np.asarray(values, dtype=np.float64)
    _, _, sums, counts = grouped(values, labels)
    n = len(sums)
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n, size=(resamples, n))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    q = [0.025, 0.975]
    if bca and n >= 3:
        # Floating sums can perturb ties in tiny discrete seed samples. Recreate
        # original indexed means for <=10 observations to preserve bias correction.
        if values.size <= 10:
            keys = sorted(set(map(str, labels)))
            groups = [np.flatnonzero(np.asarray(list(map(str, labels))) == k) for k in keys]
            draws = np.array([values[np.concatenate([groups[i] for i in row])].mean() for row in picks])
        norm = NormalDist()
        bias = norm.inv_cdf(float(np.clip(np.mean(draws < point), 1e-6, 1 - 1e-6)))
        jack = (sums.sum() - sums) / (counts.sum() - counts)
        centered = jack.mean() - jack
        denominator = 6 * float(np.sum(centered ** 2)) ** 1.5
        acceleration = float(np.sum(centered ** 3)) / denominator if denominator else 0.0
        adjusted = []
        for quantile in q:
            zq = norm.inv_cdf(quantile)
            z = bias + (bias + zq) / max(1e-12, 1 - acceleration * (bias + zq))
            adjusted.append(float(np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)))
        q = adjusted
    lo, hi = np.quantile(draws, q)
    return {"estimate": point, "ci": [float(lo), float(hi)], "clusters": n}


def bootstrap_slope(x, y, labels):
    """OLS slope; gene clusters retain first-appearance ordering used by source."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    _, inverse, sy, count = grouped(y, labels, sorted_keys=False)
    n = len(count)
    sx = np.bincount(inverse, weights=x, minlength=n)
    sxx = np.bincount(inverse, weights=x * x, minlength=n)
    sxy = np.bincount(inverse, weights=x * y, minlength=n)
    def slope(c, ax, ay, axx, axy):
        return (axy - ax * ay / c) / (axx - ax * ax / c)
    point = slope(count.sum(), sx.sum(), sy.sum(), sxx.sum(), sxy.sum())
    picks = np.random.default_rng(17).integers(0, n, size=(2000, n))
    draws = slope(*(v[picks].sum(axis=1) for v in (count, sx, sy, sxx, sxy)))
    return {"estimate": float(point), "ci": np.quantile(draws, [0.025, 0.975]).tolist(), "clusters": n}


def concordance(margin, target):
    margin = np.asarray(margin, dtype=np.float64)
    return np.where(margin == 0, 0.5, (margin * target > 0).astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "findings").is_dir() and (root / "results/geoacmg_final").is_dir():
        root = root / "results/geoacmg_final"
    cp = args.checkpoints or root / "checkpoints_small"
    checks = []
    report = {
        "schema": "geoflowagent.curated-verification.v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved dev statistics only. No training, model inference, test opening, or clinical validation.",
        "numpy_version": np.__version__,
        "limitations": [
            "R7 arithmetic is checked against saved aggregates; per-task rollout records are not in this export, so R7 confidence intervals are not independently recomputed.",
            "Initial trunk equality and model parameter counts are historical recorded evidence; this script does not instantiate Torch models.",
            "The stored test flags are audited; absence of historical test access cannot be proven by flags alone.",
        ],
        "known_stale_prose": [
            "00_README.md and R6_amendment.json narrative cite older R3/R4 values; use recalculated R3/R4 below.",
            "00_README.md says 103 readout rows. There are 103 NPZ arrays but 6 are metadata, so there are 97 readout vectors.",
            "D restore journal says flow evaluation incomplete; final C export has completed root-only dev R7 evaluation.",
        ],
    }
    def check(name, actual, expected, tolerance=1e-9):
        if isinstance(actual, (bool, str)) or isinstance(expected, (bool, str)):
            ok = actual == expected
            delta = None
        else:
            actual, expected = float(actual), float(expected)
            delta = abs(actual - expected)
            ok = np.isfinite(actual) and np.isfinite(expected) and delta <= tolerance
        checks.append({"name": name, "actual": actual, "expected": expected, "abs_error": delta,
                       "tolerance": tolerance, "pass": bool(ok)})

    def check_interval(name, actual, expected):
        check(name + ".estimate", actual["estimate"], expected["estimate"])
        for i, label in enumerate(("low", "high")):
            check(name + "." + label, actual["ci"][i], expected["ci"][i], 1e-7)

    with np.load(root / "pairs/R2_readout_margins.npz", allow_pickle=False) as archive:
        margins = {k: archive[k] for k in archive.files}
    metadata = {"target_margin", "root_vstar", "task", "gene", "pair_a", "pair_b"}
    target, tasks, genes = margins["target_margin"], margins["task"], margins["gene"]
    hits = {k: concordance(v, target) for k, v in margins.items() if k not in metadata}
    task_keys, task_inverse, _, task_counts = grouped(np.ones(len(tasks)), tasks)
    task_genes = {str(t): str(g) for t, g in zip(tasks, genes)}
    task_labels = [task_genes[t] for t in task_keys]
    per_task = {k: np.bincount(task_inverse, weights=v) / task_counts for k, v in hits.items()}
    report["sample"] = {"pairs": len(target), "tasks": len(task_keys), "genes": len(set(genes)),
                        "npz_arrays": len(margins), "readout_vectors": len(hits)}

    r2 = load_json(root / "findings/R2_readout_decomposition.json")
    check("R2.readout_count", len(hits), len(r2["readouts"]))
    check("R2.pairs", len(target), r2["pairs"])
    readout_means = {}
    for row in r2["readouts"]:
        key = row["readout"]
        actual = float(hits[key].mean())
        readout_means[key] = actual
        check("R2." + key + ".accuracy", actual, row["accuracy_tie_corrected"])
        check("R2." + key + ".ties", int(np.count_nonzero(margins[key] == 0)), row["tie_count"])
    kinds = ("own_energy", "cos", "l2", "neg_dot", "cos_std", "cos_whiten")
    original_means = {}
    family_means = {}
    for kind in kinds:
        chosen = [v for k, v in readout_means.items() if k.startswith(kind + "_") and int(k.rsplit("-", 1)[1]) in (17, 29, 43)]
        original_means[kind] = float(np.mean(chosen))
        family_means[kind] = {family: float(np.mean([readout_means[f"{kind}_{family}-{s}"] for s in (17, 29, 43)]))
                              for family in ("cosine", "euclidean", "directed_quasimetric", "poincare", "pair_mlp")}
    report["R2"] = {"all_ties_zero": all(row["tie_count"] == 0 for row in r2["readouts"]),
                    "raw": {k: readout_means[k] for k in ("raw_cos", "raw_l2", "raw_dot")},
                    "original_15_run_readout_means": original_means, "by_family": family_means}

    r1 = load_json(root / "findings/R1_paired_seeds.json")
    seed_diff, task_diff = [], []
    for seed in r1["seeds"]:
        left, right = hits[f"own_energy_cosine-{seed}"], hits[f"own_energy_euclidean-{seed}"]
        # Match the recorded statistic: difference of each run's pair means.
        # Averaging pair differences is algebraically equivalent but tiny binary
        # rounding changes BCa's strict '< observed' ties with only five seeds.
        diff = float(left.mean()) - float(right.mean())
        seed_diff.append(diff)
        task_diff.extend((per_task[f"own_energy_cosine-{seed}"] - per_task[f"own_energy_euclidean-{seed}"]).tolist())
        check(f"R1.seed{seed}.difference", diff, r1["by_seed"][str(seed)]["difference"])
    seed_interval = bootstrap_mean(seed_diff, r1["seeds"])
    gene_interval = bootstrap_mean(task_diff, task_labels * len(r1["seeds"]))
    for calculated, stored in zip((seed_interval, gene_interval), r1["findings"]):
        check_interval("R1." + stored["unit"], calculated,
                       {"estimate": stored["estimate"], "ci": [stored["ci_low"], stored["ci_high"]]})
    report["R1"] = {"seeds": r1["seeds"], "by_seed_difference": dict(zip(map(str, r1["seeds"]), seed_diff)),
                    "seed_clustered": seed_interval, "gene_clustered": gene_interval}

    r3 = load_json(root / "findings/R3_horizon_strata.json")
    r3_out = {}
    for key, data in r3["contrasts"].items():
        contrast, kind = key.split("@")
        treat, ctrl = contrast.split("_minus_")
        left_keys = [k for k in hits if k.startswith(f"{kind}_{treat}-")]
        right_keys = [k for k in hits if k.startswith(f"{kind}_{ctrl}-")]
        difference = np.mean([hits[k] for k in left_keys], axis=0) - np.mean([hits[k] for k in right_keys], axis=0)
        actual = bootstrap_slope(margins["root_vstar"], difference, genes)
        check_interval("R3." + key, actual, data["slope"])
        actual["treatment_seeds"] = sorted(int(k.rsplit("-", 1)[1]) for k in left_keys)
        actual["control_seeds"] = sorted(int(k.rsplit("-", 1)[1]) for k in right_keys)
        r3_out[key] = actual
    report["R3"] = r3_out

    metrics = {}
    for base in ("geometry_comparison", "geometry_r1_pairs"):
        for path in (cp / base).glob("*/seed-*/value_metrics.json"):
            item = load_json(path)
            metrics[f"{item['energy']}-{item['seed']}"] = item
            check(f"checkpoint.{item['energy']}-{item['seed']}.test_reported", item["test_reported"], False)
    r4 = load_json(root / "findings/R4_crossmodel_control.json")
    runs = r4["runs"]
    missing = sorted(set(runs) - set(metrics))
    if missing:
        raise FileNotFoundError(f"Missing value_metrics.json for {missing}; use --checkpoints")
    ordering = np.array([per_task["own_energy_" + run] for run in runs])
    policies = np.array([[metrics[run]["metrics"]["dev"]["per_task_policy_accuracy"][t] for t in task_keys] for run in runs])
    matched = np.mean(ordering * policies, axis=0)
    # Source stages.py uses one cyclic next-model mismatch per run, not all cross pairs.
    mismatched = np.mean(ordering * np.roll(policies, -1, axis=0), axis=0)
    r4_out = {}
    for label, values in (("matched", matched), ("mismatched", mismatched), ("difference", matched - mismatched)):
        actual = bootstrap_mean(values, task_labels)
        check_interval("R4." + label, actual, {"estimate": r4["report"][label], "ci": r4["report"][label + "_ci"]})
        r4_out[label] = actual
    r4_out["n_runs"] = len(runs)
    r4_out["mismatch_rule"] = "Lexicographically sorted runs, cyclic next-model policy; 19 mismatches per task, not all 342 ordered cross-model pairs."
    report["R4"] = r4_out

    r1b = load_json(root / "findings/R1b_new_seed_planning_axis.json")
    metric_keys = {"policy_accuracy": "per_task_policy_accuracy", "joint_stop_action_accuracy": "per_task_joint_accuracy", "regret_at_1": "per_task_regret_at_1"}
    r1b_out = {}
    for metric, per_task_key in metric_keys.items():
        differences = []
        for seed in (59, 71):
            left = metrics[f"cosine-{seed}"]["metrics"]["dev"][per_task_key]
            right = metrics[f"euclidean-{seed}"]["metrics"]["dev"][per_task_key]
            shared = sorted(set(left) & set(right))
            differences.append([left[t] - right[t] for t in shared])
        value = float(np.asarray(differences).mean())
        check("R1b." + metric, value, r1b["new_seeds"][metric]["estimate"])
        r1b_out[metric] = {"recomputed_task_macro_estimate": value, "stored_task_bootstrap_ci": r1b["new_seeds"][metric]["ci"]}
    r1b_out["limitation"] = "Two new seeds only. Stored intervals resample tasks after averaging seeds and do not quantify seed uncertainty."
    report["R1b"] = r1b_out

    r7 = load_json(root / "findings/R7_flow_dev_root_only.json")
    check("R7.test_unsealed", r7["test_unsealed"], False)
    check("R7.sample_count", r7["eligible_states"] * r7["samples_per_state"], r7["sample_count"])
    rates = {"one_shot": r7["open_loop"]["goal_completion_rate"], "observed_replan": r7["receding_horizon"]["goal_completion_rate"],
             "verifier_stop_guard": r7["receding_horizon_verifier_stop_guard"]["goal_completion_rate"], "blind": r7["compute_matched_blind_replanning"]["goal_completion_rate"]}
    for mode in ("receding_horizon", "receding_horizon_verifier_stop_guard", "compute_matched_blind_replanning"):
        data = r7[mode]
        successes = data["rollouts"] - sum(data["failure_counts"].values())
        check("R7." + mode + ".success_rate", successes / data["rollouts"], data["goal_completion_rate"])
    for label, other in (("replanning_minus_one_shot", "one_shot"), ("replanning_minus_blind", "blind")):
        check("R7." + label, rates["observed_replan"] - rates[other], r7["feedback_effects"]["root_start"][label]["estimate"])
    report["R7"] = {"rates": rates, "dev_tasks": r7["eligible_tasks"], "samples": r7["sample_count"],
                    "saved_root_feedback_effects": r7["feedback_effects"]["root_start"],
                    "verification": "Arithmetic and failure-count reconciliation only; rollout CIs are stored evidence."}

    b0 = json.loads((root / "findings/b0.jsonl").read_text("utf-8").splitlines()[0])
    check("B0.single_family_gap", b0["detail"]["left_mean"] - b0["detail"]["right_mean"], b0["estimate"])
    report["B0"] = {"estimate": b0["estimate"], "stored_ci": b0["ci"], "role": b0["role"], "gene_clusters": b0["n_units"]}
    report["checks"] = checks
    failed = [v for v in checks if not v["pass"]]
    report["status"] = "PASS" if not failed else "FAIL"
    report["summary"] = {"checks": len(checks), "passed": len(checks) - len(failed), "failed": len(failed)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["summary"], "failures": failed}, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

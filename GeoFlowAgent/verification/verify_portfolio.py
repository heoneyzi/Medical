#!/usr/bin/env python3
"""Portfolio verification — recomputes the headline numbers of GeoFlowAgent from this repository alone.

Python 3.10+ with NumPy; reads the saved result files and per-run metrics in results/.

What it does
  1. hard-v2: calls verify_hard_v2.verify() on results/hard_v2 (3-seed means, paired task deltas, flow means).
  2. GeoACMG, recomputed from the per-run value_metrics.json in results/geoacmg_latest_checkpoints/:
       - R3c  seven-seed cosine−Euclidean planning contrasts: estimates + seed- and gene-clustered BCa CIs
       - R11  MedCPT ablation (4 runs): estimates + run- and gene-clustered BCa CIs
       - R11+ eight-run ablation aggregate over all saved ablation runs
       - R1b  new-seed (59/71) planning estimates;  15-run geometry family dev means
     Gene clusters are read from the task-id prefix ("GENE:uuid"), which reproduces the margin-array gene labels.
  3. Arithmetic checks on stored aggregates (R7 flow, B0 benchmark gap).
  4. Margin-array numbers (R1–R4, R2f readouts) are cross-checked against the saved outputs of the two
     margin verifiers (verification/geoacmg_verified_metrics.json, geoacmg_latest_verified_metrics.json).
  5. Number traceability, both directions: each listed value must appear (formatted as written) on the pages
     that quote it, and every decimal printed in README.md, experiments/**/README.md, docs/README.md and
     docs/research/*.md must match a value computed or loaded here, or a quote that exists verbatim in its
     source report. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np

sys.dont_write_bytecode = True  # keep the repository free of __pycache__

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE))
from verify_geoacmg_metrics import bootstrap_mean  # noqa: E402  (same BCa code as the archive)
from verify_hard_v2 import verify as verify_hard_v2  # noqa: E402

CHECKS: list[dict] = []


def load(rel):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def check(name, actual, expected, tol=1e-9):
    ok = abs(float(actual) - float(expected)) <= tol
    CHECKS.append({"name": name, "actual": float(actual), "expected": float(expected), "pass": ok})
    return ok


def check_interval(name, iv, est, ci):
    check(name + ".estimate", iv["estimate"], est)
    check(name + ".ci_low", iv["ci"][0], ci[0], 1e-7)
    check(name + ".ci_high", iv["ci"][1], ci[1], 1e-7)


def per_run_metrics(*bases):
    out = {}
    for base in bases:
        for p in glob.glob(str(RES / "geoacmg_latest_checkpoints" / base / "*" / "seed-*" / "value_metrics.json")):
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            assert d["test_reported"] is False, p
            out[(d["energy"], d["seed"])] = d["metrics"]["dev"]
    return out


def _find_key(obj, key):
    """First value stored under `key` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if key in obj and not isinstance(obj[key], (dict, list)):
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


COLS = {"policy": "per_task_policy_accuracy", "joint": "per_task_joint_accuracy", "regret@1": "per_task_regret_at_1"}
V: dict[str, float] = {}  # values quoted in READMEs (filled below)


def main() -> int:
    # 1. hard-v2 -------------------------------------------------------------
    hv = verify_hard_v2(RES / "hard_v2")
    for k, v in hv["state_metric_seed_means"].items():
        V[f"hv2.{k}"] = v
    for k, v in hv["reference_state_metrics"].items():
        V[f"hv2.ref.{k}"] = v
    for k, v in hv["paired_task_macro_differences"].items():
        V[f"hv2.delta.{k}"] = v
    for k, v in hv["flow_seed_means"].items():
        V[f"hv2.flow.{k}"] = v
    CHECKS.append({"name": "hard_v2.verify_hard_v2", "pass": hv["status"] == "PASS"})
    ag = {c["condition"] + "/" + c["scenario"]: c["success"]["mean"] for c in
          load("hard_v2/final_reports/search_agent/medcpt_two_x_seed17/test_summary.json")["conditions"]}
    V["hv2.cl.learned.clean"], V["hv2.cl.learned.perturbed"] = ag["learned_policy/clean"], ag["learned_policy/perturbed"]
    V["hv2.cl.random.clean"], V["hv2.cl.random.perturbed"] = ag["random/clean"], ag["random/perturbed"]
    dq = load("hard_v2/final_reports/data_quality.json")["counts"]
    V["hv2.tasks"], V["hv2.tools"], V["hv2.snapshots"] = dq["tasks"], dq["tools"], dq["snapshots"]

    # 2. GeoACMG per-run recomputation --------------------------------------
    inc = per_run_metrics("geometry_comparison", "geometry_r1_pairs", "geometry_r1_pairs2")
    tasks = sorted(inc[("cosine", 17)]["per_task_policy_accuracy"])
    genes = [t.split(":", 1)[0] for t in tasks]
    check("geoacmg.dev_tasks", len(tasks), 584)
    check("geoacmg.dev_genes", len(set(genes)), 47)

    seeds = [17, 29, 43, 59, 71, 83, 97]
    r3c = load("geoacmg_working_findings/R3c_planning_axis_7seeds.json")["axes"]
    for name, col in COLS.items():
        left = np.array([[inc[("cosine", s)][col][t] for t in tasks] for s in seeds])
        right = np.array([[inc[("euclidean", s)][col][t] for t in tasks] for s in seeds])
        siv = bootstrap_mean(left.mean(1) - right.mean(1), range(len(seeds)))
        giv = bootstrap_mean((left - right).mean(0), genes)
        check_interval(f"R3c.{name}.seed", siv, r3c[name]["seed"]["est"], r3c[name]["seed"]["ci"])
        check_interval(f"R3c.{name}.gene", giv, r3c[name]["gene"]["est"], r3c[name]["gene"]["ci"])
        V[f"r3c.{name}"] = siv["estimate"]
        V[f"r3c.{name}.seed_lo"], V[f"r3c.{name}.seed_hi"] = siv["ci"]
        V[f"r3c.{name}.gene_lo"], V[f"r3c.{name}.gene_hi"] = giv["ci"]

    abl = per_run_metrics("geometry_ablate_medcpt", "geometry_ablate_medcpt2")
    for base in ("geometry_ablate_medcpt", "geometry_ablate_medcpt2"):
        for run in load(f"geoacmg_latest_checkpoints/{base}/comparison.json")["runs"]:
            abl.setdefault((run["energy"], run["seed"]), run["dev"])  # cosine/seed-29 leaf JSON is absent
    r11 = load("geoacmg_working_findings/R11_medcpt_ablation.json")["axes"]
    ext = json.loads((HERE / "geoacmg_latest_verified_metrics.json").read_text(encoding="utf-8"))["R11"]["curation_derived_available_run_extension"]["axes"]
    for label, runs in (("R11", [(f, s) for f in ("cosine", "euclidean") for s in (17, 29)]),
                        ("R11ext", [(f, s) for f in ("cosine", "euclidean") for s in (17, 29, 43, 59)])):
        for name, col in COLS.items():
            left = np.array([[abl[r][col][t] for t in tasks] for r in runs])
            right = np.array([[inc[r][col][t] for t in tasks] for r in runs])
            # Same floating-point formulation as the stored statistic (BCa ties are sensitive to it):
            # R11 used difference-of-run-means; the curation extension used mean-of-differences.
            run_values = left.mean(1) - right.mean(1) if label == "R11" else (left - right).mean(1)
            riv = bootstrap_mean(run_values, range(len(runs)))
            giv = bootstrap_mean((left - right).mean(0), genes)
            if label == "R11":
                check_interval(f"R11.{name}.run", riv, r11[name]["run_clustered"]["est"], r11[name]["run_clustered"]["ci"])
                check_interval(f"R11.{name}.gene", giv, r11[name]["gene_clustered"]["est"], r11[name]["gene_clustered"]["ci"])
            else:
                check_interval(f"R11ext.{name}.run", riv, ext[name]["run"]["estimate"], ext[name]["run"]["ci"])
                check_interval(f"R11ext.{name}.gene", giv, ext[name]["gene"]["estimate"], ext[name]["gene"]["ci"])
            key = label.lower()
            V[f"{key}.{name}"] = riv["estimate"]
            V[f"{key}.{name}.run_lo"], V[f"{key}.{name}.run_hi"] = riv["ci"]
            V[f"{key}.{name}.gene_lo"], V[f"{key}.{name}.gene_hi"] = giv["ci"]

    r1b = load("geoacmg_final/findings/R1b_new_seed_planning_axis.json")["new_seeds"]
    for name, col in (("policy_accuracy", COLS["policy"]), ("joint_stop_action_accuracy", COLS["joint"]),
                      ("regret_at_1", COLS["regret@1"])):
        est = float(np.mean([[inc[("cosine", s)][col][t] - inc[("euclidean", s)][col][t] for t in tasks] for s in (59, 71)]))
        check(f"R1b.{name}", est, r1b[name]["estimate"])
        V[f"r1b.{name}"] = est
        V[f"r1b.{name}.lo"], V[f"r1b.{name}.hi"] = r1b[name]["ci"]

    for fam in ("pair_mlp", "directed_quasimetric", "cosine", "poincare", "euclidean"):
        V[f"geo15.{fam}.policy"] = mean(inc[(fam, s)]["policy_optimal_set_accuracy"] for s in (17, 29, 43))

    # 3. arithmetic on stored aggregates --------------------------------------
    r7 = load("geoacmg_final/findings/R7_flow_dev_root_only.json")
    rates = {k: r7[k]["goal_completion_rate"] for k in ("open_loop", "receding_horizon", "compute_matched_blind_replanning",
                                                          "receding_horizon_verifier_stop_guard")}
    for k in ("receding_horizon", "compute_matched_blind_replanning", "receding_horizon_verifier_stop_guard"):
        check(f"R7.{k}.from_failures", (r7[k]["rollouts"] - sum(r7[k]["failure_counts"].values())) / r7[k]["rollouts"], rates[k])
    eff = r7["feedback_effects"]["root_start"]
    check("R7.replan_minus_one_shot", rates["receding_horizon"] - rates["open_loop"], eff["replanning_minus_one_shot"]["estimate"])
    check("R7.replan_minus_blind", rates["receding_horizon"] - rates["compute_matched_blind_replanning"], eff["replanning_minus_blind"]["estimate"])
    check("R7.test_unsealed_flag", float(r7["test_unsealed"]), 0.0)
    V.update({f"r7.{k}": v for k, v in rates.items()})
    V["r7.d_one_shot"], V["r7.d_one_shot.lo"], V["r7.d_one_shot.hi"] = (eff["replanning_minus_one_shot"][k] for k in ("estimate", "low", "high"))
    V["r7.d_blind"], V["r7.d_blind.lo"], V["r7.d_blind.hi"] = (eff["replanning_minus_blind"][k] for k in ("estimate", "low", "high"))
    b0 = json.loads((RES / "geoacmg_final/findings/b0.jsonl").read_text(encoding="utf-8").splitlines()[0])
    check("B0.gap", b0["detail"]["left_mean"] - b0["detail"]["right_mean"], b0["estimate"])
    V["b0"], (V["b0.lo"], V["b0.hi"]) = b0["estimate"], b0["ci"]
    V["b0.best_single"] = b0["detail"]["right_mean"]
    rq1 = {json.loads(line)["name"]: json.loads(line) for line in (RES / "geoacmg_final/findings/rq1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    e1 = rq1["E1_remaining_cost_r2_medcpt_minus_random_projection"]["detail"]
    V["probe.r2"], V["probe.r2_random"] = e1["observed_r2"], e1["random_projection_r2"]
    e1a = rq1["E1_optimal_action_accuracy_medcpt"]
    V["probe.action"], V["probe.action_null"] = e1a["estimate"], e1a["null_mean"]

    # 4. stored-only numbers cross-checked against the archive's NPZ-based verification output -------------
    latest = json.loads((HERE / "geoacmg_latest_verified_metrics.json").read_text(encoding="utf-8"))
    export = json.loads((HERE / "geoacmg_verified_metrics.json").read_text(encoding="utf-8"))
    CHECKS.append({"name": "archive.latest_verification_status", "pass": latest["status"] == "PASS" and latest["summary"]["failed"] == 0})
    CHECKS.append({"name": "archive.export_verification_status", "pass": export["status"] == "PASS" and export["summary"]["failed"] == 0})
    r2f = load("geoacmg_working_findings/R2f_readout_7seeds.json")["by_readout"]
    for kind in ("own_energy", "cos_std", "l2", "cos_whiten"):
        rec = latest["R2f_seven_seeds"][kind]
        check(f"R2f.{kind}.stored_vs_npz_check", r2f[kind]["seed"]["est"], rec["seed"]["estimate"])
        check(f"R2f.{kind}.seed_ci_low.stored_vs_npz_check", r2f[kind]["seed"]["ci"][0], rec["seed"]["ci"][0], 1e-7)
        check(f"R2f.{kind}.gene_ci_high.stored_vs_npz_check", r2f[kind]["gene"]["ci"][1], rec["gene"]["ci"][1], 1e-7)
        V[f"r2f.{kind}"] = r2f[kind]["seed"]["est"]
        V[f"r2f.{kind}.seed_lo"], V[f"r2f.{kind}.seed_hi"] = r2f[kind]["seed"]["ci"]
        V[f"r2f.{kind}.gene_lo"], V[f"r2f.{kind}.gene_hi"] = r2f[kind]["gene"]["ci"]
    r1 = export["R1"]["seed_clustered"]
    V["r1.seed"], (V["r1.seed.lo"], V["r1.seed.hi"]) = r1["estimate"], r1["ci"]
    raw = export["R2"]["raw"]
    V["raw.cos"], V["raw.l2"], V["raw.dot"] = raw["raw_cos"], raw["raw_l2"], raw["raw_dot"]
    r8 = load("geoacmg_working_findings/R8_frozen_space_ladder.json")["results"]["views_only"]
    V["r8.l2_std"], V["r8.cos_whiten"], V["r8.random64"] = (r8[k]["accuracy"] for k in ("l2_std (train-fit)", "cos_whiten (train-fit)", "random-64 + cos"))
    r9 = load("geoacmg_working_findings/R9_information_source_ablation.json")["results"]
    V["r9.structured"], V["r9.combined"] = r9["structured_only"]["PCA-64 + cos"]["accuracy"], r9["medcpt_plus_structured"]["PCA-64 + cos"]["accuracy"]
    r10 = load("geoacmg_working_findings/R10_readout_dependent_ranking.json")
    V["r10.orderings"], V["r10.first"] = r10["distinct_orderings"], len(r10["energies_that_rank_first"])

    V["r2.own_energy"], V["r2.cos_std"], V["r2.cos"] = (export["R2"]["original_15_run_readout_means"][k] for k in ("own_energy", "cos_std", "cos"))
    V["r4.diff"], (V["r4.diff.lo"], V["r4.diff.hi"]) = export["R4"]["difference"]["estimate"], export["R4"]["difference"]["ci"]
    r2d = load("geoacmg_working_findings/R2d_holdout_standardiser.json")["results"]["train_fit"]
    V["r2d"], (V["r2d.lo"], V["r2d.hi"]) = r2d["difference"], r2d["ci"]
    V["r8.mean_cos_sim"] = r8["_geometry"]["mean_cosine_similarity"]
    for k in ("l2", "neg_dot", "cos_whiten"):
        V[f"r2.{k}"] = export["R2"]["original_15_run_readout_means"][k]
    r3x = export["R3"]["cosine_minus_euclidean@own_energy"]
    V["r3.slope"], (V["r3.slope.lo"], V["r3.slope.hi"]) = r3x["estimate"], r3x["ci"]
    V["r4.matched"], V["r4.mismatched"] = export["R4"]["matched"]["estimate"], export["R4"]["mismatched"]["estimate"]
    r6 = load("geoacmg_final/findings/R6_amendment.json")["supporting_evidence"]
    V["r6.old_r4_difference"] = r6["C5_retraction_reconfirmed"]["report"]["difference"]
    V["r6.old_cos_15run"] = _find_key(r6, "cos")
    dev = {}
    for name in ("medcpt_only_vs_full", "medcpt_only_vs_hash_same_cosine", "frozen_vs_hash_same_cosine", "medcpt_cosine_vs_poincare"):
        e = load(f"hard_v2/dev_reports/decisive_dev_comparisons/{name}.json")["paired_task_macro_effects"]["policy_accuracy"]
        V[f"dev.{name}"], V[f"dev.{name}.lo"], V[f"dev.{name}.hi"] = e["mean"], e["low"], e["high"]
    ref = {c["condition"] + "/" + c["scenario"]: c["success"]["mean"] for c in
           load("hard_v2/final_reports/search_agent/original_full_dagger_round2/test_summary.json")["conditions"]}
    V["hv2.cl.ref.clean"], V["hv2.cl.ref.perturbed"] = ref["learned_policy/clean"], ref["learned_policy/perturbed"]
    for label, pat in (("15", "frozen_smoke_qwen15_multiview/seeds/seed*/agent_test/agent_summary.json"),
                       ("7", "frozen_qwen7b_multiview/seeds/seed*/agent_bilinear_test/agent_summary.json")):
        runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in sorted(glob.glob(str(RES / "synthetic_v1" / pat)))]
        check(f"synthetic_v1.{label}.seeds", len(runs), 3)
        cond = lambda name: mean(c["task_success"] for r in runs for c in r["conditions"] if c["condition"] == name)  # noqa: E731
        V[f"syn{label}.random"] = cond("random_contract_oracle_stop")
        V[f"syn{label}.commit1"] = cond("flow_closed_loop_commit1")
        V[f"syn{label}.blind"] = cond("flow_compute_matched_blind_replan")
    r2acc = {r["readout"]: r["accuracy_tie_corrected"] for r in load("geoacmg_final/findings/R2_readout_decomposition.json")["readouts"]}
    for fam in ("euclidean", "cosine", "poincare", "directed_quasimetric", "pair_mlp"):
        V[f"r2fam.{fam}"] = mean(r2acc[f"own_energy_{fam}-{s}"] for s in (17, 29, 43))

    # 5. README number lookup ----------------------------------------------------------------------------
    def read(rel):
        return (ROOT / rel).read_text(encoding="utf-8").replace("\u2212", "-")
    missing = []
    for key, fmt, where in QUOTED:
        s = format(V[key], fmt)
        for rel in where:
            if s not in read(rel):
                missing.append({"name": f"readme.quote.{key}", "pass": False, "expected": s, "file": rel})
    for quote, source, where in DOC_QUOTES:  # numbers that exist only in historical Markdown reports
        if quote not in read(source):
            missing.append({"name": "doc.source_quote", "pass": False, "expected": quote, "file": source})
        for rel in where:
            if quote not in read(rel):
                missing.append({"name": "doc.readme_quote", "pass": False, "expected": quote, "file": rel})
    # reverse check: every decimal (>= 3 places) printed in a README must be a known value or a documented quote
    allowed = set(ALLOW)
    for v in V.values():
        for nd in (3, 4, 6):
            allowed.add(format(v, f"+.{nd}f"))
            if v >= 0:
                allowed.add(format(v, f".{nd}f"))
    for quote, *_ in DOC_QUOTES:
        allowed.update({quote, quote.lstrip("+-")})
    token = re.compile(r"(?<![\w.])[+-]?\d+\.\d{3,}")
    for rel in README_FILES:
        text = re.sub(r"```.*?```", "", read(rel), flags=re.S)
        for m in token.finditer(text):
            s = m.group(0)
            if s not in allowed and ("+" + s) not in allowed:
                missing.append({"name": "readme.untraced_number", "pass": False, "expected": s, "file": rel})
    CHECKS.extend(missing)
    n_quotes = sum(len(w) for *_, w in QUOTED) + sum(len(w) + 1 for *_, w in DOC_QUOTES)
    CHECKS.append({"name": "readme.quotes_found", "pass": not missing, "checked": n_quotes})

    failed = [c for c in CHECKS if not c["pass"]]
    print(json.dumps({"status": "PASS" if not failed else "FAIL", "checks": len(CHECKS), "failed": failed,
                      "readme_numbers_checked": n_quotes}, indent=2, ensure_ascii=False))
    return 1 if failed else 0


M, EX = "README.md", "experiments/README.md"
E1, E2, E3 = "experiments/01_synthetic_v1/README.md", "experiments/02_search_pilot/README.md", "experiments/03_hard_v2/README.md"
E4, E5 = "experiments/04_geoacmg_dev/README.md", "experiments/05_geoacmg_followups/README.md"
G = "docs/README.md"
D1, D2, D3 = "docs/research/01_RESEARCH_STORY.md", "docs/research/02_EXPERIMENT_MAP.md", "docs/research/03_RESULTS_AND_VERIFICATION.md"
D4, D5 = "docs/research/04_CLAIMS_AND_EVIDENCE.md", "docs/research/05_REPRODUCIBILITY.md"

README_FILES = [M, EX, E1, E2, E3, E4, E5, G, D1, D2, D3, D4, D5]
ALLOW = {"2605.07339"}  # arXiv identifier in the links

# (value key, format, READMEs that must contain it verbatim after '\u2212' -> '-')
QUOTED = [
    # hard-v2
    ("hv2.policy_optimal_set_accuracy", ".4f", [M, E3, EX, G, D1, D3]), ("hv2.joint_stop_action_accuracy", ".4f", [E3]),
    ("hv2.regret_at_1", ".4f", [E3]), ("hv2.ref.policy_optimal_set_accuracy", ".4f", [M, E3, EX]),
    ("hv2.ref.joint_stop_action_accuracy", ".4f", [E3]), ("hv2.ref.regret_at_1", ".4f", [E3]),
    ("hv2.delta.per_task_policy_accuracy", "+.4f", [M, E3]), ("hv2.delta.per_task_joint_accuracy", "+.4f", [E3]),
    ("hv2.delta.per_task_regret_at_1", "+.4f", [E3]),
    ("hv2.flow.open_loop", ".4f", [M, E3]), ("hv2.flow.receding_horizon", ".4f", [M, E3, EX, G, D1, D3, D4]),
    ("hv2.flow.compute_matched_blind_replanning", ".4f", [M, E3]), ("hv2.flow.receding_horizon_verifier_stop_guard", ".4f", [E3]),
    ("hv2.cl.learned.clean", ".4f", [M, E3]), ("hv2.cl.learned.perturbed", ".4f", [E3]),
    ("hv2.cl.random.clean", ".4f", [M, E3]), ("hv2.cl.random.perturbed", ".4f", [M, E3]),
    ("hv2.cl.ref.clean", ".4f", [E3]), ("hv2.cl.ref.perturbed", ".4f", [E3]),
    ("hv2.tasks", ",d", [M, E3]), ("hv2.tools", "d", [M, E3]), ("hv2.snapshots", ",d", [M, E3]),
    ("dev.medcpt_only_vs_full", "+.4f", [E3]), ("dev.medcpt_only_vs_full.lo", "+.4f", [E3]), ("dev.medcpt_only_vs_full.hi", "+.4f", [E3]),
    ("dev.medcpt_only_vs_hash_same_cosine", "+.4f", [E3]), ("dev.medcpt_only_vs_hash_same_cosine.lo", "+.4f", [E3]),
    ("dev.medcpt_only_vs_hash_same_cosine.hi", "+.4f", [E3]), ("dev.frozen_vs_hash_same_cosine", "+.4f", [E3]),
    ("dev.frozen_vs_hash_same_cosine.lo", "+.4f", [E3]), ("dev.frozen_vs_hash_same_cosine.hi", "+.4f", [E3]),
    ("dev.medcpt_cosine_vs_poincare", "+.4f", [E3]), ("dev.medcpt_cosine_vs_poincare.lo", "+.4f", [E3]),
    ("dev.medcpt_cosine_vs_poincare.hi", "+.4f", [E3]),
    # synthetic v1 closed loop (agent summaries)
    ("syn15.random", ".4f", [E1]), ("syn7.random", ".4f", [E1]), ("syn15.commit1", ".4f", [E1]), ("syn7.commit1", ".4f", [E1]),
    ("syn15.blind", ".4f", [E1]),
    # GeoACMG export
    ("r7.open_loop", ".4f", [M, E4, EX, G, D1, D3, D4]), ("r7.receding_horizon", ".4f", [M, E4, EX, G, D1, D3, D4]), ("r7.compute_matched_blind_replanning", ".4f", [M, E4]),
    ("r7.receding_horizon_verifier_stop_guard", ".4f", [E4]), ("r7.d_one_shot", "+.4f", [M, E4]),
    ("r7.d_one_shot.lo", ".4f", [M, E4]), ("r7.d_one_shot.hi", ".4f", [M, E4]),
    ("b0", ".4f", [M, E4, G, D1, D3, D4]), ("b0.lo", ".4f", [M, E4]), ("b0.hi", ".4f", [M, E4]), ("b0.best_single", ".4f", [E4]),
    ("probe.r2", ".4f", [M, E4, G, D1, D3, D4]), ("probe.r2_random", "+.4f", [M, E4]), ("probe.action", ".4f", [M, E4]), ("probe.action_null", ".4f", [M, E4]),
    ("raw.cos", ".4f", [M, E4]), ("raw.l2", ".4f", [M, E4]), ("raw.dot", ".4f", [M, E4]),
    ("geo15.pair_mlp.policy", ".4f", [E4]), ("geo15.euclidean.policy", ".4f", [E4]),
    ("r1.seed", "+.4f", [E4]), ("r1.seed.lo", ".4f", [E4]), ("r1.seed.hi", ".4f", [E4]),
    ("r1b.policy_accuracy", "+.4f", [E4, D1, D2, D3, D4]), ("r1b.policy_accuracy.lo", "+.4f", [E4]), ("r1b.policy_accuracy.hi", "+.4f", [E4]),
    ("r2.own_energy", ".4f", [E4]), ("r2.cos_std", ".4f", [E4]), ("r2.cos", ".4f", [E4]),
    ("r4.diff", "+.6f", [E4, D1, D2, D3, D4]), ("r4.diff.lo", ".6f", [E4]), ("r4.diff.hi", ".6f", [E4]),
    # GeoACMG follow-ups
    ("r2f.own_energy", "+.4f", [M, E5, G, D1, D3, D4]), ("r2f.own_energy.seed_lo", ".4f", [E5]), ("r2f.own_energy.seed_hi", ".4f", [E5]),
    ("r2f.own_energy.gene_lo", ".4f", [E5]), ("r2f.own_energy.gene_hi", ".4f", [E5]),
    ("r2f.cos_std", "+.4f", [M, E5]), ("r2f.cos_std.seed_lo", "+.4f", [M, E5]), ("r2f.cos_std.seed_hi", ".4f", [M, E5]),
    ("r2f.cos_std.gene_lo", "+.4f", [E5]), ("r2f.cos_std.gene_hi", ".4f", [E5]),
    ("r2f.l2", "+.4f", [E5]), ("r2f.l2.seed_lo", "+.4f", [E5]), ("r2f.l2.seed_hi", ".4f", [E5]),
    ("r2f.l2.gene_lo", "+.4f", [E5]), ("r2f.l2.gene_hi", ".4f", [E5]),
    ("r2f.cos_whiten", "+.4f", [M, E5]), ("r2f.cos_whiten.seed_lo", "+.4f", [E5]), ("r2f.cos_whiten.seed_hi", "+.4f", [E5]),
    ("r2f.cos_whiten.gene_lo", "+.4f", [E5]), ("r2f.cos_whiten.gene_hi", "+.4f", [E5]),
    ("r3c.policy", "+.4f", [M, E5]), ("r3c.policy.seed_lo", "+.4f", [M, E5]), ("r3c.policy.seed_hi", ".4f", [M, E5]),
    ("r3c.policy.gene_lo", ".4f", [M, E5]), ("r3c.policy.gene_hi", ".4f", [M, E5]),
    ("r3c.joint", "+.4f", [E5]), ("r3c.joint.seed_lo", "+.4f", [E5]), ("r3c.joint.seed_hi", ".4f", [E5]),
    ("r3c.joint.gene_lo", "+.4f", [E5]), ("r3c.joint.gene_hi", ".4f", [E5]),
    ("r3c.regret@1", "+.4f", [E5]), ("r3c.regret@1.seed_lo", "+.4f", [E5]), ("r3c.regret@1.seed_hi", ".4f", [E5]),
    ("r3c.regret@1.gene_lo", "+.4f", [E5]), ("r3c.regret@1.gene_hi", "+.4f", [E5]),
    ("r2d", "+.4f", [E5]), ("r2d.lo", ".4f", [E5]), ("r2d.hi", ".4f", [E5]), ("r8.mean_cos_sim", ".3f", [E5]),
    ("r8.l2_std", ".4f", [E5]), ("r8.cos_whiten", ".4f", [E5]), ("r8.random64", ".4f", [E5]),
    ("r9.structured", ".4f", [M, E5]), ("r9.combined", ".4f", [M, E5]), ("r10.orderings", "d", [E5]), ("r10.first", "d", [E5]),
    ("r11.regret@1", "+.4f", [M, E5, EX, G, D1, D3, D4]), ("r11.regret@1.run_lo", ".4f", [M, E5]), ("r11.regret@1.run_hi", ".4f", [M, E5]),
    ("r11.regret@1.gene_lo", ".4f", [M, E5]), ("r11.regret@1.gene_hi", ".4f", [M, E5]),
    ("r11.policy", "+.4f", [E5, D3]), ("r11.policy.run_lo", "+.4f", [E5]), ("r11.policy.run_hi", "+.4f", [E5]),
    ("r11.policy.gene_lo", "+.4f", [E5]), ("r11.policy.gene_hi", ".4f", [E5]),
    ("r11.joint", "+.4f", [E5]), ("r11.joint.run_lo", "+.4f", [E5]), ("r11.joint.run_hi", "+.4f", [E5]),
    ("r11.joint.gene_lo", "+.4f", [E5]), ("r11.joint.gene_hi", ".4f", [E5]),
    ("r11ext.regret@1", "+.4f", [E5]), ("r11ext.regret@1.run_lo", ".4f", [E5]), ("r11ext.regret@1.run_hi", ".4f", [E5]),
    ("r11ext.regret@1.gene_lo", ".4f", [E5]), ("r11ext.regret@1.gene_hi", ".4f", [E5]),
    ("r11ext.policy", "+.4f", [E5]), ("r11ext.policy.gene_lo", "+.4f", [E5]), ("r11ext.policy.gene_hi", ".4f", [E5]),
    ("r11ext.joint", "+.4f", [E5]), ("r11ext.joint.gene_lo", "+.4f", [E5]), ("r11ext.joint.gene_hi", "+.4f", [E5]),
    # Korean research notes
    ("r2.l2", ".4f", [D3, D4]), ("r2.neg_dot", ".4f", [D3, D4]), ("r2.cos_whiten", ".4f", [D3, D4]),
    ("r3.slope", "+.4f", [D2, D3, D4]), ("r3.slope.lo", "+.4f", [D3, D4]), ("r3.slope.hi", "+.4f", [D3, D4]),
    ("r4.matched", ".4f", [D3, D4]), ("r4.mismatched", ".4f", [D3, D4]),
    ("r6.old_r4_difference", ".4f", [D4]), ("r6.old_cos_15run", ".4f", [D4]),
]

# Numbers whose only source is a historical Markdown report: (exact string, source file, READMEs that quote it)
RUN_STATE, INTERIM = "results/synthetic_v1/GeoFlowAgent_RUN_STATE.md", "results/hard_v2/GeoFlowAgent_A_EXPERIMENT_REPORT_CURRENT.md"
CONCL = "results/hard_v2/GeoFlowAgent_A_FINAL_EXPERIMENT_CONCLUSION.md"
DOC_QUOTES = [
    ("0.8551", RUN_STATE, [M, E1]), ("0.7681", RUN_STATE, [E1]), ("0.4359", RUN_STATE, [E1]), ("0.4744", RUN_STATE, [E1]),
    ("0.8630", INTERIM, [M, E2, EX]), ("0.5731", INTERIM, [M, E2, EX]), ("0.0859", INTERIM, [M, E2]),
    ("+0.2102", INTERIM, [E2]), ("+0.1604", INTERIM, [E2]), ("+0.2550", INTERIM, [E2]),
    ("0.9617", INTERIM, [E2]), ("0.9578", INTERIM, [E2]), ("0.0042", INTERIM, [E2]), ("0.9548", INTERIM, [E2]), ("0.9434", INTERIM, [E2]),
    ("0.8476", INTERIM, [E2]), ("0.4192", INTERIM, [E2]), ("+0.0146", INTERIM, [E2]), ("+0.0078", INTERIM, [E2]),
    ("+0.0219", INTERIM, [E2]), ("0.7525", INTERIM, [E2]), ("0.7398", INTERIM, [E2]), ("0.5320", INTERIM, [E2]),
    ("4,856", INTERIM, [E2]), ("41/41", INTERIM, [M, E2]),
    ("-0.0316", INTERIM, [E3]), ("-0.0843", INTERIM, [E3]), ("+0.0218", INTERIM, [E3]),
    ("+0.0708", CONCL, [M, E3]), ("+0.1682", CONCL, [M, E3]), ("+0.0153", CONCL, [E3]), ("+0.1231", CONCL, [E3]),
    ("-0.3263", CONCL, [E3]), ("-0.1754", CONCL, [E3]),
    ("+0.3282", "results/geoacmg_final/findings/R6_amendment.json", [E4]),
    ("+0.3289", "results/geoacmg_final/findings/R6_amendment.json", [E4]),
]

if __name__ == "__main__":
    raise SystemExit(main())

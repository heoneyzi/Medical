"""Command line for the corpus stages.

    python -m geoflowagent.geoacmg ingest    --corpus clingen.tsv --output-dir artifacts/acmg
    python -m geoflowagent.geoacmg build     --corpus clingen.tsv --output-dir artifacts/acmg
    python -m geoflowagent.geoacmg diagnose  --package artifacts/acmg
    python -m geoflowagent.geoacmg prereg    --output artifacts/acmg/preregistration.json

Each stage writes a report next to its artefacts and prints it, so a run leaves a
record of what was dropped and why without anyone having to remember to look.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from geoflowagent.geoacmg import splits
from geoflowagent.geoacmg.claims import Claim, Preregistration
from geoflowagent.geoacmg.clingen import read_corpus
from geoflowagent.geoacmg.diagnostics import corpus_card, single_family_gap_finding
from geoflowagent.geoacmg.evidence import Classification
from geoflowagent.geoacmg.tasks import (
    Task,
    build_tasks,
    cap_per_gene,
    fingerprint,
    initial_state,
    private_verifier,
    probe_tool,
    public_goal,
    report_tool,
    snapshots_for,
)
from geoflowagent.geoacmg.toolmap import coverage
from geoflowagent.utils.io import write_json, write_jsonl

PROBE_COST_NOTE = (
    "placeholder cost of 1.0 per query. Replace with the measured per-family latency "
    "from the capture stage before any budget-matched comparison is run; a comparison "
    "at matched cost is meaningless while the cost model is a placeholder."
)


def _claims() -> tuple[Claim, ...]:
    return (
        Claim("B0", "The benchmark contains a planning problem.",
              "the full registry does not beat the best single tool family", +1),
        Claim("C1", "Frozen encoder embeddings carry remaining-cost and next-action information.",
              "a metric- and parameter-matched random-feature control is not beaten", +1),
        Claim("C2", "That information is biological rather than surface-procedural.",
              "counterfactual entity swaps leave the model's action distribution unchanged", +1),
        Claim("C3", "Generating a whole plan beats step-wise selection at matched tool cost.",
              "the commit-length curve is optimised at k=1", +1),
        Claim("C4", "Non-Euclidean geometry helps in proportion to measured task structure.",
              "the moderator slope interval includes zero", +1),
        Claim("C5", "Planner validity and representation validity are the same claim.",
              "ablating the learned geometry leaves the planner's advantage intact", +1),
    )


def _primary_findings() -> tuple[str, ...]:
    return (
        # B0 is about the instrument, but it is still a claim the study can be wrong
        # about, so its finding is allowed to decide it.
        "benchmark_needs_more_than_one_tool_family",
        "P1_learned_geometry_minus_raw_l2",
        "P2_frozen_minus_trained_from_scratch",
        "P3_flow_minus_greedy_at_matched_cost",
    )


def command_prereg(args: argparse.Namespace) -> dict[str, Any]:
    prereg = Preregistration(
        study="GeoFlowAgent phase 2: execution geometry of frozen biomedical representations",
        claims=_claims(),
        primary_findings=_primary_findings(),
        amends=args.amends,
        note=args.note,
    )
    payload = prereg.to_dict() | {"fingerprint": prereg.fingerprint()}
    write_json(Path(args.output), payload)
    return {"output": str(args.output), "fingerprint": prereg.fingerprint()}


def command_ingest(args: argparse.Namespace) -> dict[str, Any]:
    records, report = read_corpus(args.corpus)
    payload = report.to_dict() | {"tool_coverage": coverage(records)}
    root = Path(args.output_dir)
    write_json(root / "ingest_report.json", payload)
    return {
        "kept": report.kept,
        "skipped": dict(report.skipped),
        "point_scale_fidelity": report.reconstruction_rate,
        "records_fully_decidable": payload["tool_coverage"]["records_fully_decidable"],
    }


def _write_corpus(
    tasks: list[Task], root: Path, source_release: str, *, split_by: str = "gene"
) -> dict[str, Any]:
    # One registry for the whole corpus: every task can call every probe, so the
    # action space never tells the agent which evidence lines are the relevant ones.
    families = sorted({f for task in tasks for f in task.families}, key=lambda f: f.value)
    tools = [probe_tool(family, cost=1.0) for family in families]
    tools += [report_tool(label, cost=0.5) for label in Classification]
    task_rows = []
    snapshot_rows = []
    verifier_rows = []
    for task in tasks:
        task_rows.append(
            {
                "task_id": task.task_id,
                "query": (
                    f"Classify {task.record.primary_hgvs} in {task.record.gene} for "
                    f"{task.record.disease or 'the associated condition'}, citing the "
                    "evidence you retrieve."
                ),
                "initial_state": initial_state(task),
                "goal": public_goal(),
                "available_tools": [tool["tool_id"] for tool in tools],
                "split": task.split,
                "split_group": (
                    task.record.expert_panel or "unassigned_panel"
                    if split_by == "panel"
                    else task.record.gene
                ),
                "category": task.record.expert_panel or "unassigned_panel",
                "background_ids": [],
                # Only groups we actually split on may be registered: the auditor
                # requires every registered group to sit wholly inside one split.
                # A panel curates many genes, so registering the panel while splitting
                # by gene guarantees a straddle -- and the panel IS a real leakage
                # channel, because its criteria specification is shared across its
                # genes. Splitting by panel is the conservative choice and nests the
                # gene inside it; splitting by gene is the powerful one and drops the
                # panel claim. The choice is explicit, never silent.
                "group_ids": (
                    {
                        "entity": task.record.gene,
                        "template": task.record.expert_panel or "unassigned_panel",
                    }
                    if split_by == "panel"
                    else {"entity": task.record.gene}
                ),
                "provenance": {
                    "generator": "geoacmg-v1",
                    "uuid": task.record.uuid,
                    "approval_date": (
                        task.record.approval_date.isoformat()
                        if task.record.approval_date
                        else None
                    ),
                    "required_families": sorted(f.value for f in task.required_families),
                    "optimal_orderings": task.optimal_orderings(),
                },
            }
        )
        snapshot_rows.extend(snapshots_for(task, source_release, registry=families))
        verifier_rows.append(
            {"task_id": task.task_id, "private_verifier": private_verifier(task)}
        )
    write_jsonl(root / "tasks.jsonl", task_rows)
    write_jsonl(root / "tools.jsonl", tools)
    write_jsonl(root / "snapshots.jsonl", snapshot_rows)
    write_jsonl(root / "verifiers.private.jsonl", verifier_rows)
    return {
        "tasks": len(task_rows),
        "tools": len(tools),
        "snapshots": len(snapshot_rows),
        "families": [f.value for f in families],
    }


def command_build(args: argparse.Namespace) -> dict[str, Any]:
    records, ingest = read_corpus(args.corpus)
    tasks, report = build_tasks(records, require_reconstruction=not args.keep_disagreements)
    if args.max_tasks_per_gene:
        tasks = cap_per_gene(tasks, args.max_tasks_per_gene)
    gene_plan = splits.by_gene(tasks, seed=args.seed, by=args.split_by)
    variant_plan = splits.by_variant(tasks, seed=args.seed)
    temporal_plan = splits.temporal(tasks, cutoff=date.fromisoformat(args.temporal_cutoff))
    for task in tasks:
        object.__setattr__(task, "split", gene_plan.assignment[task.task_id])
    root = Path(args.output_dir)
    counts = _write_corpus(tasks, root, args.source_release, split_by=args.split_by)
    manifest = {
        "generator": "geoacmg-v1",
        "source_release": args.source_release,
        "fingerprint": fingerprint(tasks),
        "ingest": ingest.to_dict(),
        "build": report.to_dict(),
        "counts": counts,
        "cost_model": {"probe": 1.0, "report": 0.5, "status": PROBE_COST_NOTE},
        "splits": {
            plan.axis: plan.report | {"counts": plan.counts()}
            for plan in (gene_plan, variant_plan, temporal_plan)
        },
        "card": corpus_card(tasks),
    }
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "splits.json",
        {
            "gene": gene_plan.assignment,
            "variant": variant_plan.assignment,
            "temporal": temporal_plan.assignment,
        },
    )
    return {
        "tasks": counts["tasks"],
        "genes": manifest["card"]["genes"],
        "dropped": report.to_dict()["dropped"],
        "single_family_gap": manifest["card"]["single_family_ceiling"]["gap"],
    }


def command_diagnose(args: argparse.Namespace) -> dict[str, Any]:
    from geoflowagent.geoacmg import report as report_module
    from geoflowagent.geoacmg.claims import Role

    records, _ = read_corpus(args.corpus)
    tasks, _ = build_tasks(records)
    # The uncapped corpus is what the cap curve is for: it is how a reader picks
    # the cap in the first place, so it stays in the card whatever the study runs.
    card = corpus_card(tasks)
    finding = single_family_gap_finding(tasks, resamples=args.resamples, seed=args.seed)
    payload: dict[str, Any] = {"card": card, "findings": [finding.to_dict()]}

    # The interval that adjudicates B0 has to come from the corpus the study
    # actually runs on. Reporting the uncapped gap and then running everything
    # else on the capped corpus would put the claim and the evidence on
    # different datasets.
    study_finding = finding
    if args.max_tasks_per_gene:
        study_tasks = cap_per_gene(tasks, args.max_tasks_per_gene)
        study_finding = single_family_gap_finding(
            study_tasks, resamples=args.resamples, seed=args.seed, role=Role.PRIMARY
        )
        payload["study_corpus"] = {
            "max_tasks_per_gene": args.max_tasks_per_gene,
            "card": corpus_card(study_tasks),
            "findings": [study_finding.to_dict()],
        }
    else:
        study_finding = single_family_gap_finding(
            tasks, resamples=args.resamples, seed=args.seed, role=Role.PRIMARY
        )

    if args.output:
        write_json(Path(args.output), payload)
    if args.findings_out:
        # diagnostics.json is not read by the report, which merges findings-dir
        # *.jsonl. Without this the benchmark-qualification claim never reaches
        # adjudication and B0 renders as untested no matter what was measured.
        report_module.write_findings(args.findings_out, [study_finding])
    return {
        "tasks": card["tasks"],
        "genes": card["genes"],
        "effective_clusters": card["tasks_per_gene"]["effective_number_of_clusters"],
        "majority_class_rate": card["target_balance"]["majority_class_rate"],
        "single_family_gap": finding.to_dict(),
        "study_single_family_gap": study_finding.to_dict(),
        "findings_out": args.findings_out,
    }


def command_probe(args: argparse.Namespace) -> dict[str, Any]:
    from geoflowagent.geoacmg import report as report_module
    from geoflowagent.geoacmg.runners import probe_findings, write_json

    cache = args.cache if args.cache and Path(args.cache).exists() else None
    findings, detail = probe_findings(args.processed, cache, permutations=args.permutations)
    written = report_module.write_findings(args.output, findings)
    write_json(Path(args.output).with_suffix(".report.json"), detail)
    return {"findings": written, "detail": detail.get("counts"), "views": list(detail.get("views", {}))}


def command_ladder(args: argparse.Namespace) -> dict[str, Any]:
    from geoflowagent.geoacmg import report as report_module
    from geoflowagent.geoacmg.runners import ladder_findings, write_json

    findings, detail = ladder_findings(
        args.package, args.processed, max_tasks=args.max_tasks, seed=args.seed
    )
    written = report_module.write_findings(args.output, findings)
    write_json(Path(args.output).with_suffix(".report.json"), detail)
    return {"findings": written, "rungs": detail["rungs"], "commit_curve": detail["commit_curve"]}


def command_report(args: argparse.Namespace) -> dict[str, Any]:
    from geoflowagent.geoacmg import report as report_module

    findings_dir = Path(args.findings_dir)
    merged = findings_dir / "all_findings.jsonl"
    rows: list[str] = []
    for path in sorted(findings_dir.glob("*.jsonl")):
        if path.name == merged.name:
            continue
        rows.extend(path.read_text(encoding="utf-8").splitlines())
    merged.write_text("\n".join(row for row in rows if row.strip()) + "\n", encoding="utf-8")
    card = None
    if args.card and Path(args.card).exists():
        card = json.loads(Path(args.card).read_text(encoding="utf-8")).get("card")
    return report_module.build(args.prereg, merged, args.output, corpus_card=card)


def command_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    from geoflowagent.geoacmg import pipeline

    ctx = pipeline.Context(
        root=Path(args.root),
        corpus=Path(args.corpus),
        config=Path(args.config),
        max_tasks_per_gene=args.max_tasks_per_gene,
        split_by=args.split_by,
        include_test=args.include_test,
        force=args.force,
    )
    ctx.root.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        return {"environment": pipeline.check_environment(ctx), "plan": pipeline.plan_table()}
    return pipeline.run(ctx, only=args.stage or None, skip=args.skip or ())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoacmg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prereg = sub.add_parser("prereg", help="write the frozen analysis plan")
    prereg.add_argument("--output", required=True)
    prereg.add_argument("--amends", default=None)
    prereg.add_argument("--note", default="")

    ingest = sub.add_parser("ingest", help="parse the ClinGen corpus and report coverage")
    ingest.add_argument("--corpus", required=True)
    ingest.add_argument("--output-dir", required=True)

    build = sub.add_parser("build", help="build the task corpus and the splits")
    build.add_argument("--corpus", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--source-release", default="clingen-erepo+gnomad_r4+ensembl_116")
    build.add_argument("--temporal-cutoff", default="2024-09-15")
    build.add_argument("--seed", type=int, default=17)
    build.add_argument(
        "--split-by",
        choices=("gene", "panel"),
        default="gene",
        help=(
            "gene: more clusters, but a panel's criteria specification is shared across "
            "splits. panel: conservative, nests genes inside panels, ~45 clusters"
        ),
    )
    build.add_argument(
        "--max-tasks-per-gene",
        type=int,
        default=None,
        help=(
            "cap the tasks kept per gene. One gene holds a quarter of the ClinGen "
            "corpus, which collapses the effective cluster count; see "
            "card.effective_cluster_curve in the manifest and pre-register a value"
        ),
    )
    build.add_argument(
        "--keep-disagreements",
        action="store_true",
        help="keep records where the point scale and the panel disagree (not well posed)",
    )

    diagnose = sub.add_parser("diagnose", help="is there a planning problem here")
    diagnose.add_argument("--corpus", required=True)
    diagnose.add_argument("--output")
    diagnose.add_argument("--resamples", type=int, default=2000)
    diagnose.add_argument("--seed", type=int, default=17)
    diagnose.add_argument(
        "--max-tasks-per-gene",
        type=int,
        help="the study's cap, so the adjudicating interval comes from the study's corpus",
    )
    diagnose.add_argument(
        "--findings-out", help="write the B0 finding where the report can read it"
    )

    probe = sub.add_parser("probe", help="RQ1: is the execution structure in the representation")
    probe.add_argument("--package", required=True)
    probe.add_argument("--processed", required=True)
    probe.add_argument("--cache")
    probe.add_argument("--output", required=True)
    probe.add_argument("--permutations", type=int, default=200)

    rungs = sub.add_parser("ladder", help="RQ2: whole-plan generation against step-wise")
    rungs.add_argument("--package", required=True)
    rungs.add_argument("--processed", required=True)
    rungs.add_argument("--output", required=True)
    rungs.add_argument("--max-tasks", type=int)
    rungs.add_argument("--seed", type=int, default=17)

    rep = sub.add_parser("report", help="claims -> verdicts -> readable report")
    rep.add_argument("--prereg", required=True)
    rep.add_argument("--findings-dir", required=True)
    rep.add_argument("--output", required=True)
    rep.add_argument("--card")

    pipe = sub.add_parser("pipeline", help="run every stage in order, resumably")
    pipe.add_argument("--root", required=True)
    pipe.add_argument("--corpus", required=True)
    pipe.add_argument("--config", required=True)
    pipe.add_argument("--max-tasks-per-gene", type=int, default=100)
    pipe.add_argument("--split-by", choices=("gene", "panel"), default="gene")
    pipe.add_argument("--stage", action="append", help="run only these stages")
    pipe.add_argument("--skip", action="append", default=[])
    pipe.add_argument("--include-test", action="store_true")
    pipe.add_argument("--force", action="store_true")
    pipe.add_argument("--check-only", action="store_true", help="print the plan and stop")
    return parser


COMMANDS = {
    "probe": command_probe,
    "ladder": command_ladder,
    "report": command_report,
    "pipeline": command_pipeline,
    "prereg": command_prereg,
    "ingest": command_ingest,
    "build": command_build,
    "diagnose": command_diagnose,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = COMMANDS[args.command](args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

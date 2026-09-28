from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from geoflowagent.data.schema import (
    validate_snapshot,
    validate_task,
    validate_tool,
    validate_verifier,
)
from geoflowagent.utils.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

WORKFLOW_FAMILIES: dict[str, dict[str, Any]] = {
    "annotation": {
        "finish_tool": "finish_annotation_report",
        "required_tools": [],
        "private_fields": ["gene_symbol"],
    },
    "population": {
        "finish_tool": "finish_population_report",
        "required_tools": ["lookup_population"],
        "private_fields": ["gene_symbol", "allele_frequency"],
    },
    "gene_disease": {
        "finish_tool": "finish_gene_disease_report",
        "required_tools": ["lookup_disease"],
        "private_fields": ["gene_symbol", "disease_id"],
    },
    "phenotype": {
        "finish_tool": "finish_phenotype_report",
        "required_tools": ["match_phenotype"],
        "private_fields": ["gene_symbol", "phenotype_gene"],
    },
    "ortholog": {
        "finish_tool": "finish_ortholog_report",
        "required_tools": ["lookup_ortholog"],
        "private_fields": ["gene_symbol", "mouse_ortholog"],
    },
    "literature": {
        "finish_tool": "finish_literature_report",
        "required_tools": ["search_literature"],
        "private_fields": ["gene_symbol", "pmid"],
    },
    "comprehensive": {
        "finish_tool": "finish_comprehensive_report",
        "required_tools": [
            "match_phenotype",
            "lookup_disease",
            "lookup_population",
            "lookup_ortholog",
            "search_literature",
        ],
        "private_fields": [
            "gene_symbol",
            "phenotype_gene",
            "disease_id",
            "allele_frequency",
            "mouse_ortholog",
            "pmid",
        ],
    },
}


def _condition(field: str, op: str, value: Any | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"field": field, "op": op}
    if value is not None:
        row["value"] = value
    return row


def _symbolic_tool(
    tool_id: str,
    name: str,
    description: str,
    *,
    preconditions: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    tags: list[str],
    cost: float = 1.0,
) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "name": name,
        "description": description,
        "execution_mode": "symbolic",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "preconditions": preconditions,
        "effects": effects,
        "tags": tags,
        "search_cost": cost,
        "provenance": {"generator": "geoflow-search-fixture-v1"},
    }


def _snapshot_tool(
    tool_id: str,
    name: str,
    description: str,
    *,
    required_arguments: list[str],
    argument_bindings: dict[str, str],
    output_properties: dict[str, dict[str, Any]],
    output_bindings: dict[str, str],
    preconditions: list[dict[str, Any]],
    tags: list[str],
    cost: float = 1.0,
) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "name": name,
        "description": description,
        "execution_mode": "snapshot",
        "input_schema": {
            "type": "object",
            "required": required_arguments,
            "properties": {name: {} for name in required_arguments},
            "additionalProperties": False,
        },
        "argument_bindings": argument_bindings,
        "output_schema": {
            "type": "object",
            "required": sorted(output_properties),
            "properties": output_properties,
            "additionalProperties": False,
        },
        "output_bindings": output_bindings,
        "preconditions": preconditions,
        "effects": [],
        "tags": tags,
        "search_cost": cost,
        "provenance": {"generator": "geoflow-search-fixture-v1"},
    }


def build_search_fixture_tools() -> list[dict[str, Any]]:
    """Return a deterministic tool catalogue with real branching and alternate paths.

    This catalogue is strictly a CI/research-method fixture.  Unlike the original
    smoke graph, it deliberately contains contract-valid but suboptimal actions,
    alternate sources, and verifier-detectable dead ends so a value learner
    receives more than a restatement of the contract mask. Empty/failure outcomes
    belong in the separately declared perturbation evaluation: a repeatable empty
    self-loop would make a bounded search label correctly *unknown* rather than Q*.
    """

    tools = [
        _symbolic_tool(
            "parse_variant",
            "Parse versioned variant",
            "Parse the input notation while preserving transcript and assembly provenance.",
            preconditions=[_condition("raw_variant", "nonempty"), _condition("parsed", "neq", True)],
            effects=[
                {"field": "parsed", "op": "set", "value": True},
                {"field": "normalized_variant", "op": "copy", "from": "raw_variant"},
            ],
            tags=["variant", "parse", "provenance"],
        ),
        _symbolic_tool(
            "liftover_grch37_to_grch38",
            "Lift variant to GRCh38",
            "Convert a parsed GRCh37 variant to GRCh38 and invalidate prior allele checks.",
            preconditions=[_condition("parsed", "eq", True), _condition("assembly", "eq", "GRCh37")],
            effects=[
                {"field": "assembly", "op": "set", "value": "GRCh38"},
                {"field": "lifted", "op": "set", "value": True},
                {"field": "reference_verified", "op": "set", "value": False},
            ],
            tags=["variant", "assembly", "liftover"],
        ),
        _symbolic_tool(
            "verify_reference",
            "Verify reference allele",
            "Verify reference and alternate alleles against the pinned GRCh38 reference.",
            preconditions=[
                _condition("parsed", "eq", True),
                _condition("assembly", "eq", "GRCh38"),
                _condition("reference_verified", "neq", True),
            ],
            effects=[{"field": "reference_verified", "op": "set", "value": True}],
            tags=["variant", "reference", "validation"],
        ),
        _symbolic_tool(
            "normalize_variant",
            "Normalize variant",
            "Normalize a reference-verified GRCh38 variant without discarding its source version.",
            preconditions=[
                _condition("reference_verified", "eq", True),
                _condition("normalized", "neq", True),
            ],
            effects=[{"field": "normalized", "op": "set", "value": True}],
            tags=["variant", "normalization"],
        ),
        _snapshot_tool(
            "annotate_primary",
            "Primary variant annotation",
            "Retrieve the preferred pinned gene annotation for a normalized variant.",
            required_arguments=["record_id", "database_release"],
            argument_bindings={"record_id": "record_id", "database_release": "database_release"},
            output_properties={
                "gene_symbol": {"type": "string"},
                "annotated": {"type": "boolean"},
            },
            output_bindings={"gene_symbol": "gene_symbol", "annotated": "annotated"},
            preconditions=[_condition("normalized", "eq", True), _condition("annotated", "neq", True)],
            tags=["variant", "gene", "annotation", "primary-source"],
            cost=1.0,
        ),
        _snapshot_tool(
            "annotate_alternate",
            "Alternate variant annotation",
            "Retrieve an independent pinned annotation when the preferred source is unsuitable.",
            required_arguments=["record_id", "database_release"],
            argument_bindings={"record_id": "record_id", "database_release": "database_release"},
            output_properties={
                "gene_symbol": {"type": "string"},
                "annotated": {"type": "boolean"},
            },
            output_bindings={"gene_symbol": "gene_symbol", "annotated": "annotated"},
            preconditions=[_condition("normalized", "eq", True), _condition("annotated", "neq", True)],
            tags=["variant", "gene", "annotation", "alternate-source"],
            cost=1.4,
        ),
        _snapshot_tool(
            "match_phenotype",
            "Match phenotype",
            "Match HPO terms against a pinned phenotype-to-gene source.",
            required_arguments=["record_id", "database_release"],
            argument_bindings={"record_id": "record_id", "database_release": "database_release"},
            output_properties={
                "phenotype_gene": {"type": "string"},
                "phenotype_matched": {"type": "boolean"},
            },
            output_bindings={
                "phenotype_gene": "phenotype_gene",
                "phenotype_matched": "phenotype_matched",
            },
            preconditions=[
                _condition("hpo_terms", "nonempty"),
                _condition("phenotype_matched", "neq", True),
            ],
            tags=["phenotype", "hpo", "gene"],
        ),
        _snapshot_tool(
            "lookup_disease",
            "Retrieve gene-disease evidence",
            "Retrieve pinned disease evidence after resolving a gene.",
            required_arguments=["gene_symbol", "database_release"],
            argument_bindings={"gene_symbol": "gene_symbol", "database_release": "database_release"},
            output_properties={
                "disease_id": {"type": "string"},
                "disease_evidence": {"type": "boolean"},
            },
            output_bindings={
                "disease_id": "disease_id",
                "disease_evidence": "disease_evidence",
            },
            preconditions=[
                _condition("gene_symbol", "nonempty"),
                _condition("disease_evidence", "neq", True),
            ],
            tags=["gene", "disease", "evidence"],
        ),
        _snapshot_tool(
            "lookup_population",
            "Retrieve population evidence",
            "Retrieve a pinned population allele frequency for the normalized variant.",
            required_arguments=["record_id", "database_release"],
            argument_bindings={"record_id": "record_id", "database_release": "database_release"},
            output_properties={
                "allele_frequency": {"type": "number"},
                "population_evidence": {"type": "boolean"},
            },
            output_bindings={
                "allele_frequency": "allele_frequency",
                "population_evidence": "population_evidence",
            },
            preconditions=[
                _condition("normalized", "eq", True),
                _condition("population_evidence", "neq", True),
            ],
            tags=["variant", "population", "frequency"],
        ),
        _snapshot_tool(
            "lookup_ortholog",
            "Retrieve ortholog evidence",
            "Retrieve a pinned mouse ortholog after resolving a gene.",
            required_arguments=["gene_symbol", "database_release"],
            argument_bindings={"gene_symbol": "gene_symbol", "database_release": "database_release"},
            output_properties={
                "mouse_ortholog": {"type": "string"},
                "ortholog_evidence": {"type": "boolean"},
            },
            output_bindings={
                "mouse_ortholog": "mouse_ortholog",
                "ortholog_evidence": "ortholog_evidence",
            },
            preconditions=[
                _condition("gene_symbol", "nonempty"),
                _condition("ortholog_evidence", "neq", True),
            ],
            tags=["gene", "ortholog", "model-organism"],
        ),
        _snapshot_tool(
            "search_literature",
            "Retrieve literature evidence",
            "Resolve a pinned publication identifier for the selected gene.",
            required_arguments=["gene_symbol", "database_release"],
            argument_bindings={"gene_symbol": "gene_symbol", "database_release": "database_release"},
            output_properties={
                "pmid": {"type": "string"},
                "literature_evidence": {"type": "boolean"},
            },
            output_bindings={"pmid": "pmid", "literature_evidence": "literature_evidence"},
            preconditions=[
                _condition("gene_symbol", "nonempty"),
                _condition("literature_evidence", "neq", True),
            ],
            tags=["gene", "literature", "evidence"],
        ),
        _snapshot_tool(
            "collect_secondary_context",
            "Collect optional secondary context",
            "Retrieve valid but non-required contextual evidence; useful as a controlled detour.",
            required_arguments=["record_id", "database_release"],
            argument_bindings={"record_id": "record_id", "database_release": "database_release"},
            output_properties={"secondary_context": {"type": "boolean"}},
            output_bindings={"secondary_context": "secondary_context"},
            preconditions=[_condition("secondary_context", "neq", True)],
            tags=["optional", "context", "controlled-detour"],
            cost=1.2,
        ),
    ]

    finish_requirements = {
        "finish_annotation_report": ["annotated"],
        "finish_population_report": ["annotated", "population_evidence"],
        "finish_gene_disease_report": ["annotated", "disease_evidence"],
        "finish_phenotype_report": ["annotated", "phenotype_matched"],
        "finish_ortholog_report": ["annotated", "ortholog_evidence"],
        "finish_literature_report": ["annotated", "literature_evidence"],
        "finish_comprehensive_report": [
            "annotated",
            "phenotype_matched",
            "disease_evidence",
            "population_evidence",
            "ortholog_evidence",
            "literature_evidence",
        ],
    }
    for tool_id, required in finish_requirements.items():
        report_type = tool_id.removeprefix("finish_").removesuffix("_report")
        tools.append(
            _symbolic_tool(
                tool_id,
                f"Finish {report_type.replace('_', ' ')} report",
                "Assemble only the evidence required by this workflow into a typed final report.",
                preconditions=[
                    *[_condition(field, "eq", True) for field in required],
                    _condition("report_ready", "neq", True),
                ],
                effects=[
                    {"field": "report_ready", "op": "set", "value": True},
                    {"field": "report_type", "op": "set", "value": report_type},
                ],
                tags=["report", "integration", report_type],
            )
        )
    for tool in tools:
        validate_tool(tool)
    return tools


def _snapshot(
    task_id: str,
    split: str,
    tool_id: str,
    arguments: dict[str, Any],
    output: dict[str, Any],
    source_revision: str,
    *,
    status: str = "success",
) -> dict[str, Any]:
    argument_suffix = sha256_text(canonical_json(arguments))[:12]
    row = {
        "snapshot_id": f"{task_id}:{tool_id}:{argument_suffix}",
        "tool_id": tool_id,
        "arguments": arguments,
        "status": status,
        "output": output,
        "summary": f"Deterministic {status} fixture for {tool_id}.",
        "source_revision": source_revision,
        "provenance": {
            "synthetic": True,
            "research_use": "pipeline_validation_only",
            "generator": "geoflow-search-fixture-v1",
            "task_id": task_id,
            "split": split,
        },
    }
    validate_snapshot(row)
    return row


def _query(family: str, index: int, raw_variant: str, assembly: str) -> str:
    variants = {
        "annotation": [
            "Resolve the gene annotation for {variant} reported on {assembly}.",
            "Normalize {variant} ({assembly}) and identify its gene.",
            "Prepare a source-pinned annotation for {variant} from {assembly}.",
        ],
        "population": [
            "Find the normalized population evidence for {variant} on {assembly}.",
            "Determine the population frequency of {variant}, preserving its build.",
            "Build a population-evidence report for {variant} ({assembly}).",
        ],
        "gene_disease": [
            "Connect {variant} on {assembly} to pinned gene-disease evidence.",
            "Resolve the gene and disease evidence for {variant}.",
            "Prepare a gene-disease report from {variant} ({assembly}).",
        ],
        "phenotype": [
            "Combine {variant} on {assembly} with the provided HPO profile.",
            "Resolve the variant and phenotype-supported gene for {variant}.",
            "Prepare a phenotype-matching report for {variant} ({assembly}).",
        ],
        "ortholog": [
            "Find a pinned mouse ortholog for the gene affected by {variant}.",
            "Resolve {variant} ({assembly}) and retrieve ortholog evidence.",
            "Prepare an ortholog report for {variant}.",
        ],
        "literature": [
            "Retrieve pinned literature evidence for the gene affected by {variant}.",
            "Resolve {variant} and identify its supporting publication.",
            "Prepare a literature-backed report for {variant} ({assembly}).",
        ],
        "comprehensive": [
            "Build a multi-source research dossier for {variant} on {assembly}.",
            "Integrate variant, phenotype, disease, population, ortholog, and literature evidence for {variant}.",
            "Prepare a comprehensive source-pinned report for {variant} ({assembly}).",
        ],
    }
    return variants[family][index % len(variants[family])].format(
        variant=raw_variant, assembly=assembly
    )


def generate_search_fixture(
    output_dir: str | Path,
    *,
    task_count: int = 42,
    seed: int = 17,
) -> dict[str, Any]:
    """Generate a sizeable exact synthetic fixture for end-to-end CI.

    The output is deliberately marked synthetic and must never be presented as
    medical evidence.  Its purpose is to prove that large search supervision,
    branching, masking, training, and sealed evaluation all execute correctly.
    Real experiments must replace these rows with reviewed source snapshots.
    """

    if task_count < 21:
        raise ValueError("task_count must be at least 21 so every split/family is represented")
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    tools = build_search_fixture_tools()
    tool_ids = {row["tool_id"] for row in tools}
    families = list(WORKFLOW_FAMILIES)
    tasks: list[dict[str, Any]] = []
    verifiers: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    # The first 21 rows guarantee every workflow family occurs in every split.
    # Remaining rows preserve a roughly 7:2:2 train/dev/test ratio.  This makes a
    # 21-task fixture structurally useful, while larger fixtures remain train-heavy.
    assignments = [(family, split) for family in families for split in ("train", "dev", "test")]
    split_cycle = ["train"] * 7 + ["dev"] * 2 + ["test"] * 2
    for offset in range(task_count - len(assignments)):
        assignments.append((families[offset % len(families)], split_cycle[offset % len(split_cycle)]))
    for index, (family, split) in enumerate(assignments):
        source_revision = f"synthetic-genomics-snapshot-2026-09-{split}-v1"
        task_id = f"search-{split}-{index:05d}"
        entity_group = f"synthetic-entity-{index:05d}"
        # Templates and releases are intentionally disjoint across split. Workflow
        # topology remains shared, allowing a controlled in-family generalization
        # test without leaking exact language/source versions.
        template_group = f"{family}-{split}-template-{index % 3}"
        topology_group = family
        assembly = "GRCh37" if index % 3 == 0 else "GRCh38"
        raw_variant = f"SYNTH_TX{index:05d}.1:c.{100 + index}A>G"
        gene_symbol = f"SGENE{index:05d}"
        disease_id = f"MONDO:SYNTH{index:05d}"
        phenotype_gene = gene_symbol
        mouse_ortholog = f"Sgene{index:05d}"
        pmid = f"9{index:07d}"
        allele_frequency = round(((index % 97) + 1) / 100_000, 6)
        primary_degraded = index % 5 == 0
        initial_state = {
            "is_synthetic": True,
            "record_id": task_id,
            "raw_variant": raw_variant,
            "assembly": assembly,
            "sequence_reference": "SYNTH_GRCh37" if assembly == "GRCh37" else "SYNTH_GRCh38",
            "coordinate_system": "one_based_closed",
            "database_release": source_revision,
            "hpo_terms": [f"HP:SYNTH{index % 13:03d}"],
            "source_status": {
                "primary_annotation": "degraded" if primary_degraded else "available"
            },
            "sequence_context": "ACGT" * 8,
        }
        family_config = WORKFLOW_FAMILIES[family]
        finish_tool = str(family_config["finish_tool"])
        available_tools = [
            "parse_variant",
            "liftover_grch37_to_grch38",
            "verify_reference",
            "normalize_variant",
            "annotate_primary",
            "annotate_alternate",
            *family_config["required_tools"],
            "collect_secondary_context",
            finish_tool,
        ]
        assert set(available_tools) <= tool_ids
        task = {
            "task_id": task_id,
            "query": _query(family, index, raw_variant, assembly),
            "category": family,
            "initial_state": initial_state,
            "goal": {
                "conditions": [
                    _condition("report_ready", "eq", True),
                    _condition("report_type", "eq", finish_tool.removeprefix("finish_").removesuffix("_report")),
                ]
            },
            "available_tools": available_tools,
            "background_ids": ["fixture-safety", "fixture-source-policy"],
            "split": split,
            "split_group": entity_group,
            "provenance": {
                "synthetic": True,
                "research_use": "pipeline_validation_only",
                "source": f"procedural-search-fixture-{split}",
                "source_revision": source_revision,
                "wrapper_revision": "geoflow-search-fixture-v1",
                "entity_group": entity_group,
                "template_group": template_group,
                "workflow_family": family,
                "topology_group": topology_group,
                "primary_annotation_degraded": primary_degraded,
            },
        }
        validate_task(task)
        tasks.append(task)

        target_values = {
            "gene_symbol": gene_symbol,
            "disease_id": disease_id,
            "phenotype_gene": phenotype_gene,
            "mouse_ortholog": mouse_ortholog,
            "pmid": pmid,
            "allele_frequency": allele_frequency,
        }
        private_conditions = [
            _condition(field, "eq", target_values[field])
            for field in family_config["private_fields"]
        ]
        verifier = {
            "task_id": task_id,
            "private_verifier": {"conditions": private_conditions},
            "provenance": {
                "synthetic": True,
                "review_status": "deterministically_generated_and_schema_checked",
                "source_revision": source_revision,
            },
        }
        validate_verifier(verifier)
        verifiers.append(verifier)

        common_record_args = {"record_id": task_id, "database_release": source_revision}
        gene_args = {"gene_symbol": gene_symbol, "database_release": source_revision}
        # A degraded primary source returns a schema-valid but verifier-wrong
        # annotation. This is a finite, fully-known dead end. Runtime empty/timeout
        # behavior is tested as a perturbation and must remain unknown unless an
        # explicit retry-state model is supplied.
        primary_gene = f"LOWCONF{index:05d}" if primary_degraded else gene_symbol
        snapshots.extend(
            [
                _snapshot(
                    task_id,
                    split,
                    "annotate_primary",
                    common_record_args,
                    {"gene_symbol": primary_gene, "annotated": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "annotate_alternate",
                    common_record_args,
                    {"gene_symbol": gene_symbol, "annotated": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "match_phenotype",
                    common_record_args,
                    {"phenotype_gene": phenotype_gene, "phenotype_matched": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "lookup_disease",
                    gene_args,
                    {"disease_id": disease_id, "disease_evidence": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "lookup_population",
                    common_record_args,
                    {"allele_frequency": allele_frequency, "population_evidence": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "lookup_ortholog",
                    gene_args,
                    {"mouse_ortholog": mouse_ortholog, "ortholog_evidence": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "search_literature",
                    gene_args,
                    {"pmid": pmid, "literature_evidence": True},
                    source_revision,
                ),
                _snapshot(
                    task_id,
                    split,
                    "collect_secondary_context",
                    common_record_args,
                    {"secondary_context": True},
                    source_revision,
                ),
            ]
        )
        if primary_degraded:
            low_confidence_args = {
                "gene_symbol": primary_gene,
                "database_release": source_revision,
            }
            if "lookup_disease" in family_config["required_tools"]:
                snapshots.append(
                    _snapshot(
                        task_id,
                        split,
                        "lookup_disease",
                        low_confidence_args,
                        {"disease_id": f"MONDO:LOWCONF{index:05d}", "disease_evidence": True},
                        source_revision,
                    )
                )
            if "lookup_ortholog" in family_config["required_tools"]:
                snapshots.append(
                    _snapshot(
                        task_id,
                        split,
                        "lookup_ortholog",
                        low_confidence_args,
                        {"mouse_ortholog": f"Lowconf{index:05d}", "ortholog_evidence": True},
                        source_revision,
                    )
                )
            if "search_literature" in family_config["required_tools"]:
                snapshots.append(
                    _snapshot(
                        task_id,
                        split,
                        "search_literature",
                        low_confidence_args,
                        {"pmid": f"8{index:07d}", "literature_evidence": True},
                        source_revision,
                    )
                )

    rng.shuffle(tasks)
    # The split field is immutable; shuffling only prevents accidental reliance on row order.
    task_order = {task["task_id"]: position for position, task in enumerate(tasks)}
    verifiers.sort(key=lambda row: task_order[row["task_id"]])
    snapshots.sort(key=lambda row: (task_order[row["snapshot_id"].split(":", 1)[0]], row["tool_id"]))
    background = [
        {
            "card_id": "fixture-safety",
            "title": "Synthetic fixture warning",
            "text": "All identifiers and evidence in this corpus are synthetic and only validate the pipeline.",
            "source": "GeoFlowAgent",
            "redistribution_status": "allowed",
        },
        {
            "card_id": "fixture-source-policy",
            "title": "Pinned source policy",
            "text": "Prefer the primary annotation source unless its typed status is degraded; preserve build and release provenance.",
            "source": "GeoFlowAgent",
            "redistribution_status": "allowed",
        },
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tools.jsonl": output_dir / "tools.jsonl",
        "tasks.jsonl": output_dir / "tasks.jsonl",
        "verifiers.private.jsonl": output_dir / "verifiers.private.jsonl",
        "snapshots.jsonl": output_dir / "snapshots.jsonl",
        "background.jsonl": output_dir / "background.jsonl",
    }
    write_jsonl(paths["tools.jsonl"], tools)
    write_jsonl(paths["tasks.jsonl"], tasks)
    write_jsonl(paths["verifiers.private.jsonl"], verifiers)
    write_jsonl(paths["snapshots.jsonl"], snapshots)
    write_jsonl(paths["background.jsonl"], background)
    counts_by_split = {
        split: sum(task["split"] == split for task in tasks) for split in ("train", "dev", "test")
    }
    manifest = {
        "kind": "synthetic_search_pipeline_fixture",
        "research_evidence": False,
        "seed": seed,
        "task_count": task_count,
        "counts": {
            "tasks": len(tasks),
            "tools": len(tools),
            "snapshots": len(snapshots),
            "verifiers": len(verifiers),
            "workflow_families": len({task["category"] for task in tasks}),
            "entities": len({task["provenance"]["entity_group"] for task in tasks}),
            "templates": len({task["provenance"]["template_group"] for task in tasks}),
            "split_tasks": counts_by_split,
        },
        "content_sha256": sha256_text(
            canonical_json(
                {
                    "tools": tools,
                    "tasks": tasks,
                    "verifiers": verifiers,
                    "snapshots": snapshots,
                    "background": background,
                }
            )
        ),
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "invariants": {
            "synthetic_identifiers_only": True,
            "private_verifier_separate": True,
            "source_revision_pinned": True,
            "task_level_splits": True,
            "contains_contract_valid_detours": True,
            "contains_alternate_sources": True,
            "contains_verifier_detectable_dead_ends": True,
            "runtime_empty_results_reserved_for_perturbation_eval": True,
        },
    }
    write_json(output_dir / "fixture_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the search-distillation CI fixture")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-count", type=int, default=42)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_search_fixture(args.output_dir, task_count=args.task_count, seed=args.seed)
    print(canonical_json(result["counts"]))


if __name__ == "__main__":
    main()

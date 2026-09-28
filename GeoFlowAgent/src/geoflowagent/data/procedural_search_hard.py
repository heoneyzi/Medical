"""Compositional stress fixture for the search-distillation research pipeline.

This remains synthetic pipeline evidence, never biomedical evidence.  Unlike the
small mechanics fixture, development and test contain held-out evidence
compositions and held-out finishing tools.  Each evidence type has a cheap
primary and a costlier alternate source; a task-visible status declares whether
the primary is degraded, while the private verifier determines whether its
typed output is actually correct.  Choosing a degraded source is irreversible.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import random
from pathlib import Path
from typing import Any

from geoflowagent.data.procedural_search import (
    _condition,
    _snapshot,
    _symbolic_tool,
    build_search_fixture_tools,
)
from geoflowagent.data.schema import validate_task, validate_tool, validate_verifier
from geoflowagent.utils.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

EVIDENCE: dict[str, dict[str, str]] = {
    "phenotype": {
        "tool": "match_phenotype",
        "flag": "phenotype_matched",
        "value": "phenotype_gene",
    },
    "disease": {
        "tool": "lookup_disease",
        "flag": "disease_evidence",
        "value": "disease_id",
    },
    "population": {
        "tool": "lookup_population",
        "flag": "population_evidence",
        "value": "allele_frequency",
    },
    "ortholog": {
        "tool": "lookup_ortholog",
        "flag": "ortholog_evidence",
        "value": "mouse_ortholog",
    },
    "literature": {
        "tool": "search_literature",
        "flag": "literature_evidence",
        "value": "pmid",
    },
}


def _signature(kinds: tuple[str, ...]) -> str:
    return "_".join(kinds)


def _composition_pools() -> dict[str, tuple[tuple[str, ...], ...]]:
    names = tuple(EVIDENCE)
    by_size = {
        size: tuple(itertools.combinations(names, size)) for size in range(1, len(names) + 1)
    }
    triples = by_size[3]
    quadruples = by_size[4]
    # Training sees every primitive, all pairwise interactions, and the full
    # composition. Dev/test hold out disjoint medium-order combinations.
    return {
        "train": (*by_size[1], *by_size[2], *by_size[5]),
        "dev": (*triples[::2], *quadruples[::2]),
        "test": (*triples[1::2], *quadruples[1::2]),
    }


def build_hard_search_fixture_tools() -> list[dict[str, Any]]:
    standard = {row["tool_id"]: row for row in build_search_fixture_tools()}
    tools = [copy.deepcopy(standard[name]) for name in (
        "parse_variant",
        "liftover_grch37_to_grch38",
        "verify_reference",
        "normalize_variant",
    )]
    tools.append(
        _symbolic_tool(
            "validate_release",
            "Validate source release",
            "Validate that the normalized variant and all candidate sources use the pinned release.",
            preconditions=[
                _condition("normalized", "eq", True),
                _condition("release_validated", "neq", True),
            ],
            effects=[{"field": "release_validated", "op": "set", "value": True}],
            tags=["release", "validation", "provenance"],
        )
    )
    for annotation_id, cost, label in (
        ("annotate_primary", 1.0, "primary"),
        ("annotate_alternate", 1.35, "alternate"),
    ):
        row = copy.deepcopy(standard[annotation_id])
        row["search_cost"] = cost
        row["name"] = f"{label.title()} annotation source"
        row["description"] = (
            f"Retrieve the {label} pinned annotation. Check annotation_primary status; "
            "a typed result can still be verifier-wrong."
        )
        row["preconditions"].append(_condition("release_validated", "eq", True))
        row["provenance"] = {"generator": "geoflow-search-hard-v2"}
        tools.append(row)

    for kind, spec in EVIDENCE.items():
        primary = copy.deepcopy(standard[spec["tool"]])
        primary["name"] = f"Primary {kind} evidence"
        primary["description"] = (
            f"Retrieve {kind} evidence from the cheap primary source. Check "
            f"{kind}_primary status; schema validity does not guarantee private correctness."
        )
        primary["search_cost"] = 1.0
        primary["provenance"] = {"generator": "geoflow-search-hard-v2"}
        tools.append(primary)

        alternate = copy.deepcopy(primary)
        alternate["tool_id"] = f"{spec['tool']}_alternate"
        alternate["name"] = f"Alternate {kind} evidence"
        alternate["description"] = (
            f"Retrieve {kind} evidence from the more costly alternate source when "
            f"{kind}_primary is degraded."
        )
        alternate["search_cost"] = 1.35
        alternate["tags"] = [*primary.get("tags", []), "alternate-source"]
        tools.append(alternate)

    detour = copy.deepcopy(standard["collect_secondary_context"])
    detour["provenance"] = {"generator": "geoflow-search-hard-v2"}
    tools.append(detour)

    pools = _composition_pools()
    for kinds in (*pools["train"], *pools["dev"], *pools["test"]):
        signature = _signature(kinds)
        required = ["annotated", *[EVIDENCE[kind]["flag"] for kind in kinds]]
        tools.append(
            _symbolic_tool(
                f"finish_{signature}_report",
                f"Finish {signature.replace('_', ' ')} report",
                "Assemble exactly the requested evidence composition into a typed report.",
                preconditions=[
                    *[_condition(field, "eq", True) for field in required],
                    _condition("report_ready", "neq", True),
                ],
                effects=[
                    {"field": "report_ready", "op": "set", "value": True},
                    {"field": "report_type", "op": "set", "value": signature},
                ],
                tags=["report", "compositional", *kinds],
            )
        )
    tools.append(
        _symbolic_tool(
            "quality_check_report",
            "Quality-check report",
            "Validate the assembled report against its typed release and evidence provenance.",
            preconditions=[
                _condition("report_ready", "eq", True),
                _condition("quality_checked", "neq", True),
            ],
            effects=[{"field": "quality_checked", "op": "set", "value": True}],
            tags=["report", "quality-control", "provenance"],
        )
    )
    # Finish compositions are disjoint across the three pools, so duplicates
    # indicate a benchmark-design error rather than something to silently merge.
    identifiers = [row["tool_id"] for row in tools]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Hard fixture generated duplicate tool IDs")
    for tool in tools:
        validate_tool(tool)
    return tools


def _target_values(index: int) -> dict[str, Any]:
    gene = f"HGENE{index:05d}"
    return {
        "gene_symbol": gene,
        "phenotype_gene": gene,
        "disease_id": f"MONDO:HARD{index:05d}",
        "allele_frequency": round(((index * 7) % 997 + 1) / 1_000_000, 7),
        "mouse_ortholog": f"Hgene{index:05d}",
        "pmid": f"7{index:07d}",
    }


def _wrong_values(index: int) -> dict[str, Any]:
    return {
        "gene_symbol": f"LOWCONF{index:05d}",
        "phenotype_gene": f"LOWCONF{index:05d}",
        "disease_id": f"MONDO:LOWCONF{index:05d}",
        "allele_frequency": round(((index * 13) % 997 + 1) / 10_000, 5),
        "mouse_ortholog": f"Lowconf{index:05d}",
        "pmid": f"6{index:07d}",
    }


def _evidence_arguments(
    kind: str, task_id: str, release: str, gene_symbol: str
) -> dict[str, Any]:
    if kind in {"disease", "ortholog", "literature"}:
        return {"gene_symbol": gene_symbol, "database_release": release}
    return {"record_id": task_id, "database_release": release}


def _evidence_output(kind: str, values: dict[str, Any]) -> dict[str, Any]:
    spec = EVIDENCE[kind]
    return {spec["value"]: values[spec["value"]], spec["flag"]: True}


def _query(kinds: tuple[str, ...], index: int, variant: str, assembly: str) -> str:
    evidence = ", ".join(kinds)
    templates = (
        "Build a release-pinned {evidence} dossier for {variant} on {assembly}.",
        "Resolve {variant} ({assembly}), then integrate only {evidence} evidence.",
        "Prepare a source-validated report combining {evidence} for {variant}.",
        "For {variant} on {assembly}, recover the requested {evidence} evidence and quality-check it.",
    )
    return templates[index % len(templates)].format(
        evidence=evidence, variant=variant, assembly=assembly
    )


def generate_hard_search_fixture(
    output_dir: str | Path,
    *,
    task_count: int = 240,
    seed: int = 17,
) -> dict[str, Any]:
    if task_count < 60:
        raise ValueError("hard_v2 task_count must be at least 60 for composition coverage")
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    tools = build_hard_search_fixture_tools()
    tool_ids = {row["tool_id"] for row in tools}
    pools = _composition_pools()
    counters = {split: 0 for split in pools}
    split_cycle = ("train",) * 6 + ("dev",) * 2 + ("test",) * 2
    tasks: list[dict[str, Any]] = []
    verifiers: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    for index in range(task_count):
        split = split_cycle[index % len(split_cycle)]
        pool = pools[split]
        split_sequence_index = counters[split]
        kinds = pool[split_sequence_index % len(pool)]
        template_variant = (split_sequence_index // len(pool)) % 4
        counters[split] += 1
        signature = _signature(kinds)
        release = f"synthetic-hard-snapshot-2026-09-{split}-v2"
        task_id = f"hard-{split}-{index:05d}"
        assembly = "GRCh37" if index % 3 == 0 else "GRCh38"
        variant = f"HARD_TX{index:05d}.2:c.{1000 + index}C>T"
        targets = _target_values(index)
        wrong = _wrong_values(index)
        annotation_degraded = index % 4 == 0
        degraded = {
            kind: (index + 3 * offset + seed) % (4 + offset % 2) == 0
            for offset, kind in enumerate(EVIDENCE)
        }
        unrequested = [kind for kind in EVIDENCE if kind not in kinds]
        decoys = tuple(
            unrequested[(index + offset) % len(unrequested)]
            for offset in range(min(2, len(unrequested)))
        )
        active_kinds = tuple(dict.fromkeys((*kinds, *decoys)))
        finish_tool = f"finish_{signature}_report"
        available_tools = [
            "parse_variant",
            "liftover_grch37_to_grch38",
            "verify_reference",
            "normalize_variant",
            "validate_release",
            "annotate_primary",
            "annotate_alternate",
            *[
                tool_id
                for kind in active_kinds
                for tool_id in (EVIDENCE[kind]["tool"], f"{EVIDENCE[kind]['tool']}_alternate")
            ],
            "collect_secondary_context",
            finish_tool,
            "quality_check_report",
        ]
        if not set(available_tools) <= tool_ids:
            raise RuntimeError("Hard task references an unknown tool")
        # Sequence encoders consume ``sequence_context`` (see serialize_sequence),
        # so keep the generator on that explicit contract. Derive a task-unique,
        # reproducible window instead of a four-rotation periodic pattern.
        sequence_digest = hashlib.sha256(
            f"{seed}:{task_id}:sequence-context".encode()
        ).digest()
        sequence = "".join(
            "ACGT"[(sequence_digest[offset // 4] >> (2 * (offset % 4))) & 0b11]
            for offset in range(96)
        )
        state = {
            "is_synthetic": True,
            "record_id": task_id,
            "raw_variant": variant,
            "assembly": assembly,
            "sequence_reference": "HARD_GRCh37" if assembly == "GRCh37" else "HARD_GRCh38",
            "coordinate_system": "one_based_closed",
            "database_release": release,
            "hpo_terms": [f"HP:HARD{index % 29:03d}", f"HP:HARD{(index * 7) % 29:03d}"],
            "source_status": {
                "annotation_primary": "degraded" if annotation_degraded else "available",
                **{
                    f"{kind}_primary": "degraded" if degraded[kind] else "available"
                    for kind in EVIDENCE
                },
            },
            "entities": [variant, *[f"HP:HARD{index % 29:03d}"], *kinds],
            "sequence_context": sequence,
        }
        task = {
            "task_id": task_id,
            "query": _query(kinds, index, variant, assembly),
            "category": f"composition_{signature}",
            "initial_state": state,
            "goal": {
                "conditions": [
                    _condition("report_ready", "eq", True),
                    _condition("report_type", "eq", signature),
                    _condition("quality_checked", "eq", True),
                ]
            },
            "available_tools": available_tools,
            "background_ids": ["hard-fixture-safety", "hard-source-policy"],
            "split": split,
            "split_group": f"hard-entity-{index:05d}",
            "provenance": {
                "synthetic": True,
                "research_use": "stress_validation_only",
                "source": f"procedural-search-hard-v2-{split}",
                "source_revision": release,
                "wrapper_revision": "geoflow-search-hard-v2",
                "entity_group": f"hard-entity-{index:05d}",
                "template_group": f"hard-{split}-{signature}-template-{template_variant}",
                "workflow_family": f"composition_{signature}",
                "topology_group": f"hard-{split}-{signature}-decoy-{_signature(decoys) or 'none'}",
                "requested_evidence": list(kinds),
                "decoy_evidence": list(decoys),
                "primary_degraded": {
                    "annotation": annotation_degraded,
                    **degraded,
                },
            },
        }
        validate_task(task)
        tasks.append(task)

        private_fields = ("gene_symbol", *[EVIDENCE[kind]["value"] for kind in kinds])
        verifier = {
            "task_id": task_id,
            "private_verifier": {
                "conditions": [_condition(field, "eq", targets[field]) for field in private_fields]
            },
            "provenance": {
                "synthetic": True,
                "review_status": "deterministically_generated_and_schema_checked",
                "source_revision": release,
            },
        }
        validate_verifier(verifier)
        verifiers.append(verifier)

        common_args = {"record_id": task_id, "database_release": release}
        primary_values = wrong if annotation_degraded else targets
        snapshots.extend(
            [
                _snapshot(
                    task_id,
                    split,
                    "annotate_primary",
                    common_args,
                    {"gene_symbol": primary_values["gene_symbol"], "annotated": True},
                    release,
                ),
                _snapshot(
                    task_id,
                    split,
                    "annotate_alternate",
                    common_args,
                    {"gene_symbol": targets["gene_symbol"], "annotated": True},
                    release,
                ),
                _snapshot(
                    task_id,
                    split,
                    "collect_secondary_context",
                    common_args,
                    {"secondary_context": True},
                    release,
                ),
            ]
        )
        for kind in active_kinds:
            spec = EVIDENCE[kind]
            arguments = _evidence_arguments(kind, task_id, release, targets["gene_symbol"])
            snapshots.extend(
                [
                    _snapshot(
                        task_id,
                        split,
                        spec["tool"],
                        arguments,
                        _evidence_output(kind, wrong if degraded[kind] else targets),
                        release,
                    ),
                    _snapshot(
                        task_id,
                        split,
                        f"{spec['tool']}_alternate",
                        arguments,
                        _evidence_output(kind, targets),
                        release,
                    ),
                ]
            )
            if annotation_degraded and kind in {"disease", "ortholog", "literature"}:
                wrong_arguments = _evidence_arguments(
                    kind, task_id, release, wrong["gene_symbol"]
                )
                for tool_id in (spec["tool"], f"{spec['tool']}_alternate"):
                    snapshots.append(
                        _snapshot(
                            task_id,
                            split,
                            tool_id,
                            wrong_arguments,
                            _evidence_output(kind, wrong),
                            release,
                        )
                    )

    rng.shuffle(tasks)
    order = {task["task_id"]: position for position, task in enumerate(tasks)}
    verifiers.sort(key=lambda row: order[row["task_id"]])
    snapshots.sort(
        key=lambda row: (order[row["snapshot_id"].split(":", 1)[0]], row["tool_id"])
    )
    background = [
        {
            "card_id": "hard-fixture-safety",
            "title": "Synthetic stress-fixture warning",
            "text": "Every identifier and result is synthetic and validates only agent mechanics.",
            "source": "GeoFlowAgent",
            "redistribution_status": "allowed",
        },
        {
            "card_id": "hard-source-policy",
            "title": "Source fallback policy",
            "text": (
                "For every requested evidence type, use the cheaper primary only when its "
                "typed source_status is available; otherwise select the alternate. Ignore "
                "unrequested evidence and quality-check the final report."
            ),
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
    split_counts = {
        split: sum(task["split"] == split for task in tasks) for split in pools
    }
    manifest = {
        "kind": "synthetic_search_compositional_stress_fixture",
        "difficulty": "hard_v2",
        "research_evidence": False,
        "seed": seed,
        "task_count": task_count,
        "counts": {
            "tasks": len(tasks),
            "tools": len(tools),
            "snapshots": len(snapshots),
            "verifiers": len(verifiers),
            "workflow_families": len({task["category"] for task in tasks}),
            "topologies": len({task["provenance"]["topology_group"] for task in tasks}),
            "split_tasks": split_counts,
            "unique_sequence_contexts": len(
                {task["initial_state"]["sequence_context"] for task in tasks}
            ),
        },
        "composition_protocol": {
            split: [_signature(kinds) for kinds in signatures]
            for split, signatures in pools.items()
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
            "split_disjoint_compositions": True,
            "irreversible_degraded_source_choices": True,
            "held_out_finish_tools": True,
            "nonempty_task_unique_sequence_contexts": all(
                task["initial_state"].get("sequence_context")
                for task in tasks
            )
            and len({task["initial_state"]["sequence_context"] for task in tasks})
            == len(tasks),
            "research_claim_allowed": False,
        },
    }
    write_json(output_dir / "fixture_manifest.json", manifest)
    return manifest


__all__ = ["build_hard_search_fixture_tools", "generate_hard_search_fixture"]

#!/usr/bin/env python3
"""Run standardized external frozen context encoders for one benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcc_baselines.external_frozen import (
    AuditRequired, context_inputs, load_registry, pool_model,
    preflight_model, run_context,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default=ROOT / "configs/external_frozen_models.yaml")
    parser.add_argument("--models", default="uce,transcriptformer_cell,scgpt,scfoundation,scprint2")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--gene-map", default=ROOT / "manifests/gene_map.csv")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise SystemExit(f"unknown external model(s): {unknown}")
    specs = [registry[name] for name in names]
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    inputs = context_inputs(args.config)
    audits = [preflight_model(spec, ROOT) for spec in specs]
    audit_for = {audit["model"]: audit for audit in audits}
    (artifact_root / "preflight.json").write_text(json.dumps({
        "config": str(Path(args.config).resolve()),
        "contexts": {name: str(path) for name, path in inputs.items()},
        "models": audits,
    }, indent=2, sort_keys=True))
    for audit in audits:
        status = "READY" if audit["ready"] else "UNAVAILABLE"
        print(f"[external-preflight] {status:11s} {audit['model']}: "
              f"{'; '.join(audit['errors']) if audit['errors'] else 'ok'}")
    unavailable = {audit["model"] for audit in audits if not audit["ready"]}
    if args.preflight_only:
        if unavailable and not args.allow_unavailable:
            raise SystemExit(2)
        return
    if unavailable and not args.allow_unavailable:
        raise AuditRequired(
            "AUDIT_REQUIRED: unavailable requested models: " + ", ".join(sorted(unavailable)))

    enrolled = [spec for spec in specs if spec.name not in unavailable]
    total = len(enrolled) * len(inputs)
    step = 0
    for spec in enrolled:
        for context, input_path in inputs.items():
            step += 1
            print(f"[external-matrix {step:02d}/{total:02d} {100*step/max(total,1):5.1f}%] "
                  f"{spec.name}/{context}", flush=True)
            run_context(spec, ROOT, input_path, context, artifact_root,
                        gene_map=args.gene_map, force=args.force,
                        preflight=audit_for[spec.name])
        output = artifact_root / "pooled" / f"{spec.name}_contexts.npz"
        pool_model(spec, inputs, artifact_root, output)
        print(f"[external-frozen] pooled {spec.name} -> {output}")
    (artifact_root / "enrolled_models.json").write_text(json.dumps({
        "models": [spec.name for spec in enrolled],
        "embeddings": {spec.name: str(
            artifact_root / "pooled" / f"{spec.name}_contexts.npz") for spec in enrolled},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

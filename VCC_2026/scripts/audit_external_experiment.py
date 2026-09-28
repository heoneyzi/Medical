#!/usr/bin/env python3
"""Fail-closed response-source and frozen-model role audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vcc_baselines.config import Config
from vcc_baselines.external_frozen import load_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--policy", default=ROOT / "configs/data_policy_replogle_only.yaml")
    parser.add_argument("--registry", default=ROOT / "configs/external_frozen_models.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    policy = yaml.safe_load(Path(args.policy).read_text())
    allowed_response = set(policy["explicit_perturbation_sources"]["allowed"])
    expected = {name.removeprefix("replogle_") for name in allowed_response}
    findings = []
    for filename in args.config:
        path = Path(filename).resolve()
        cfg = Config.load(path).resolve(path.parent)
        errors = []
        if cfg.experiment.get("track") != "STRICT_REPLOGLE_ONLY":
            errors.append("config is not stamped STRICT_REPLOGLE_ONLY")
        if set(cfg.gwps.sources) != expected:
            errors.append(f"response source keys {sorted(cfg.gwps.sources)} != {sorted(expected)}")
        manifest_path = Path(cfg.gwps.cache).with_suffix(Path(cfg.gwps.cache).suffix + ".manifest.json")
        if not manifest_path.is_file():
            errors.append(f"missing effect manifest {manifest_path}")
            manifest = {}
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("protocol") != "STRICT_REPLOGLE_ONLY":
                errors.append("effect library protocol is not strict Replogle-only")
            if set(manifest.get("explicit_perturbation_sources", [])) != expected:
                errors.append("effect manifest source set differs from policy")
        findings.append({"config": str(path), "ready": not errors, "errors": errors,
                         "effect_manifest": str(manifest_path)})

    registry = load_registry(args.registry)
    roles = {name: {"role": spec.role, "track": spec.track, "status": spec.status,
                    "reason": spec.reason} for name, spec in registry.items()}
    result = {"policy": str(Path(args.policy).resolve()), "configs": findings,
              "model_roles": roles, "passed": all(item["ready"] for item in findings)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

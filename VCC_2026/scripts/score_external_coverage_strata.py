#!/usr/bin/env python3
"""Score Replogle coverage strata only through the exact cell-eval2 wrapper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from vcc_baselines.config import Config
from vcc_baselines import io, vcc2026_eval


AXES = tuple(vcc2026_eval.SCORED_METRICS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset(source: Path, destination: Path, pert_col: str,
            control: str, targets: set[str]) -> Path:
    sidecar = destination.with_suffix(destination.suffix + ".subset.json")
    expected = {"source": str(source.resolve()), "source_sha256": _sha256(source),
                "pert_col": pert_col, "control": control,
                "targets": sorted(targets)}
    if destination.is_file() and sidecar.is_file():
        observed = json.loads(sidecar.read_text())
        if (all(observed.get(key) == value for key, value in expected.items())
                and observed.get("subset_sha256") == _sha256(destination)):
            return destination
    data = ad.read_h5ad(source)
    labels = data.obs[pert_col].astype(str)
    mask = ((labels == control) | labels.isin(targets)).to_numpy()
    selected = data[mask].copy()
    observed = set(selected.obs[pert_col].astype(str)) - {control}
    if observed != targets:
        raise ValueError(
            f"{source}: selected labels differ from requested stratum; "
            f"missing={sorted(targets-observed)[:10]} extra={sorted(observed-targets)[:10]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.write_h5ad(destination, compression="gzip")
    sidecar.write_text(json.dumps({**expected, "subset_sha256": _sha256(destination)},
                                  indent=2, sort_keys=True))
    return destination


def _persist(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            ["representation", "method", "stratum"], keep="last")
    frame.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--overall", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-gene-map")
    parser.add_argument("--allow-cpu-scorer", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = Config.load(config_path).resolve(config_path.parent)
    contexts = [name for name in io.list_contexts(
        cfg.data.contexts_dir, cfg.data.control_h5ad)
        if io.load_truth(cfg.data.contexts_dir, name, cfg.data.truth_h5ad) is not None]
    if len(contexts) != 1:
        raise ValueError(f"coverage-stratum runner requires one holdout context; found {contexts}")
    context = contexts[0]
    truth_source = Path(cfg.data.contexts_dir) / context / cfg.data.truth_h5ad
    coverage = pd.read_csv(args.coverage)
    required = {"target_gene", "coverage_class"}
    if not required <= set(coverage.columns):
        raise ValueError(f"coverage file lacks {sorted(required-set(coverage.columns))}")
    if coverage["target_gene"].astype(str).duplicated().any():
        raise ValueError("coverage contains duplicated target labels")
    all_targets = set(coverage["target_gene"].astype(str))
    strata = {"ALL_TARGETS": all_targets}
    strata.update({
        str(group): set(frame["target_gene"].astype(str))
        for group, frame in coverage.groupby("coverage_class", sort=True)
    })

    overall = pd.read_csv(args.overall)
    successful = overall[overall["status"] == "ok"].copy()
    out = Path(args.out)
    if out.is_file():
        prior = pd.read_csv(out).drop_duplicates(
            ["representation", "method", "stratum"], keep="last")
        rows = prior.to_dict("records")
    else:
        rows = []
    complete = {(str(row["representation"]), str(row["method"]), str(row["stratum"]))
                for row in rows if row.get("status") == "ok"}
    total = len(successful) * len(strata)
    step = 0
    details = Path(args.details).resolve()

    for _, result_row in successful.iterrows():
        representation, method = str(result_row["representation"]), str(result_row["method"])
        for stratum, targets in strata.items():
            step += 1
            tag = (representation, method, stratum)
            if tag in complete:
                print(f"[coverage {step:03d}/{total:03d}] SKIP {'/'.join(tag)}", flush=True)
                continue
            print(f"[coverage {step:03d}/{total:03d}] START {'/'.join(tag)} "
                  f"targets={len(targets)}", flush=True)
            record = {"representation": representation, "method": method,
                      "stratum": stratum, "n_targets": len(targets), "status": "ok"}
            try:
                if targets == all_targets:
                    for axis in (*AXES, "overall"):
                        record[axis] = float(result_row[f"{axis}.mean"])
                    record["artifact_reuse"] = "full-dataset exact cell-eval2 result"
                elif not targets:
                    record.update(status="excluded", error="empty coverage stratum")
                else:
                    root = details / "coverage_strata" / stratum / representation / method / context
                    full_pred = (details / "representations" / representation /
                                 "ladder" / method / context / "pred.h5ad")
                    if not full_pred.is_file():
                        raise FileNotFoundError(full_pred)
                    pred = _subset(full_pred, root / "pred.h5ad", cfg.data.pert_col,
                                   cfg.data.control_label, targets)
                    truth = _subset(
                        truth_source,
                        details / "coverage_inputs" / stratum / context / "truth.h5ad",
                        cfg.data.pert_col, cfg.data.control_label, targets)
                    metrics = vcc2026_eval.evaluate(
                        pred, truth, root / "vcc2026",
                        details / "coverage_scales" / stratum / context,
                        pert_col=cfg.data.pert_col, control=cfg.data.control_label,
                        target_gene_map_path=args.target_gene_map,
                        require_gpu=not args.allow_cpu_scorer,
                    )
                    record.update({axis: metrics[axis] for axis in (*AXES, "overall")})
                    record["artifact_reuse"] = ""
            except Exception as exc:
                record.update(status="error", error=str(exc))
            rows = [row for row in rows if (
                str(row["representation"]), str(row["method"]), str(row["stratum"])) != tag]
            rows.append(record)
            _persist(rows, out)
            print(f"[coverage {step:03d}/{total:03d}] DONE  {'/'.join(tag)} "
                  f"status={record['status']}", flush=True)

    manifest = {
        "protocol": "exact-cell-eval2-vcc2026-by-replogle-coverage",
        "official_challenge_score": False,
        "score_scope": "external-dataset-local-anchors-per-coverage-stratum",
        "strata": {name: len(targets) for name, targets in strata.items()},
        "cross_stratum_aggregation_allowed": False,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[coverage] COMPLETE out={out}")
    if any(row.get("status") == "error" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

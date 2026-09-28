#!/usr/bin/env python3
"""Build independent Replogle-only configs for the two external benchmarks."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from vcc_baselines import io
from vcc_baselines.config import Config
from vcc_baselines.replogle import (
    OFFICIAL_FILES,
    AuditRequired,
    audit_raw_bulk,
    build_native_lnfc,
    internal_cross_source_cv,
    project_effect_library,
    write_control_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "jiang24": ROOT / "data_shadow/jiang24_ifng_bxpc3_loco",
    "gse270828": ROOT / "data_shadow/gse270828_rep3_lobo_vcc2026",
}


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=SPLIT_DIR")
    name, path = value.split("=", 1)
    return name, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=parse_dataset)
    parser.add_argument("--k562", type=Path,
                        default=ROOT / "data/replogle/K562_gwps_raw_bulk_01.h5ad")
    parser.add_argument("--rpe1", type=Path,
                        default=ROOT / "data/replogle/rpe1_raw_bulk_01.h5ad")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data_experiments/replogle_only")
    parser.add_argument("--hgnc", type=Path,
                        default=ROOT / "data/mappings/hgnc_complete_set.txt")
    parser.add_argument("--force-native", action="store_true")
    args = parser.parse_args()

    datasets = dict(args.dataset or DEFAULT_DATASETS.items())
    args.out.mkdir(parents=True, exist_ok=True)
    native_dir = args.out / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    sources = {"k562": args.k562.resolve(), "rpe1": args.rpe1.resolve()}
    audit = {}
    native = {}
    for source, path in sources.items():
        observed = audit_raw_bulk(path, args.hgnc)
        official = OFFICIAL_FILES[source]
        if path.stat().st_size != official["bytes"]:
            raise AuditRequired(
                f"AUDIT_REQUIRED: {path} size {path.stat().st_size} != {official['bytes']}")
        audit[source] = {**observed, "official_figshare": official}
        native_path = native_dir / f"{source}_native_lnfc.npz"
        if args.force_native:
            native_path.unlink(missing_ok=True)
            native_path.with_suffix(native_path.suffix + ".manifest.json").unlink(missing_ok=True)
        native[source] = build_native_lnfc(path, native_path, hgnc_path=args.hgnc)
    (args.out / "replogle_data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True))

    selected = internal_cross_source_cv(native["k562"], native["rpe1"],
                                        args.out / "internal_cv")
    for dataset, split in datasets.items():
        config_path = split / "shadow.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg = Config.load(config_path).resolve(config_path.parent)
        genes = io.read_gene_list(cfg.data.gene_list)
        targets = io.read_targets(cfg.data.targets)
        dataset_dir = args.out / dataset
        library_path = dataset_dir / "effect_library.npz"
        library, coverage = project_effect_library(
            native, genes, targets, library_path,
            global_scale=selected["global_scale"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(dataset_dir / "target_coverage.csv", index=False)
        source_paths = {}
        for source in library.sources:
            source_paths[source] = str(write_control_proxy(
                library, source, dataset_dir / "source_controls" / f"{source}.h5ad").resolve())

        cfg.run_name = f"replogle_only_{dataset}"
        cfg.output_dir = str((dataset_dir / "outputs").resolve())
        cfg.gwps.sources = source_paths
        cfg.gwps.pert_col = "gene"
        cfg.gwps.control_label = "non-targeting"
        cfg.gwps.counts_space = True
        cfg.gwps.cache = str(library_path.resolve())
        cfg.predict.effect = "logfc"
        cfg.predict.generator = "pseudobulk_multinomial"
        # The selected scale was baked into the projected lnFC library exactly once.
        cfg.predict.scale = 1.0
        cfg.predict.shrink = "none"
        cfg.predict.seed = 2026
        cfg.experiment = {
            "track": "STRICT_REPLOGLE_ONLY",
            "metric_source": "external_dataset_local_anchors",
            "hyperparameter_source": "replogle_internal_cv",
            "selected_global_scale_baked_into_library": selected["global_scale"],
            "external_ground_truth_used_for_selection": False,
        }
        resolved = dataclasses.asdict(cfg)
        (dataset_dir / "shadow.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False))
        print(f"[replogle-only] {dataset}: config={dataset_dir / 'shadow.yaml'} "
              f"coverage={coverage['coverage_class'].value_counts().to_dict()}")

    print(f"[replogle-only] COMPLETE root={args.out}")


if __name__ == "__main__":
    main()

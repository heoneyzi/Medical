#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np

from geoflowagent.embeddings.cache import EmbeddingCache
from geoflowagent.utils.io import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--final-cache", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    training = EmbeddingCache(args.training_cache)
    final = EmbeddingCache(args.final_cache)
    final_rows = np.asarray([final.tool_index[value] for value in training.tool_ids])
    report = {"training_tools": len(training.tool_ids), "final_tools": len(final.tool_ids), "arrays": {}}
    manifest = final.manifest
    for relative in training.manifest["files"]:
        if not relative.split("/", 1)[-1].startswith("tools."):
            continue
        source = np.load(training.root / relative, allow_pickle=False)
        target_path = final.root / relative
        target = np.load(target_path, allow_pickle=False)
        before = target[final_rows].astype(np.float32)
        reference = source.astype(np.float32)
        target[final_rows] = source
        temporary = target_path.with_suffix(target_path.suffix + ".tmp.npy")
        np.save(temporary, target, allow_pickle=False)
        os.replace(temporary, target_path)
        manifest["files"][relative]["sha256"] = sha256_file(target_path)
        report["arrays"][relative] = {
            "max_abs_before": float(np.max(np.abs(before - reference))),
            "shared_rows_replaced": len(training.tool_ids),
            "sha256_after": manifest["files"][relative]["sha256"],
        }
    write_json(final.root / "manifest.json", manifest)
    report["final_manifest_sha256"] = sha256_file(final.root / "manifest.json")
    write_json(args.report, report)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from geoflowagent.constants import CACHE_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.data.serialize import (
    serialize_background,
    serialize_entities,
    serialize_goal,
    serialize_history,
    serialize_sequence,
    serialize_state,
    serialize_tool_contract,
    serialize_tool_description,
)
from geoflowagent.embeddings.base import FrozenEncoder, encode_nonempty
from geoflowagent.embeddings.hashing import HashEncoder
from geoflowagent.utils.io import (
    canonical_json,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_file,
    sha256_text,
    write_json,
)


def build_encoder(config: dict[str, Any]) -> FrozenEncoder:
    backend = config.get("backend", "hash")
    kwargs = {key: value for key, value in config.items() if key != "backend"}
    if backend == "hash":
        return HashEncoder(**kwargs)
    if backend == "huggingface":
        from geoflowagent.embeddings.huggingface import HuggingFaceEncoder

        return HuggingFaceEncoder(**kwargs)
    raise ValueError(f"Unsupported encoder backend={backend!r}")


def _example_text(row: dict[str, Any], field: str) -> str:
    if field == "query":
        return str(row["query"])
    if field == "goal":
        return serialize_goal(row["goal"])
    if field == "state":
        return serialize_state(row["state"])
    if field == "next_state":
        return serialize_state(row["next_state"])
    if field == "history":
        return serialize_history(row["history"])
    if field == "sequence":
        return serialize_sequence(row["state"])
    if field == "next_sequence":
        return serialize_sequence(row["next_state"])
    if field == "entities":
        return serialize_entities(row["state"])
    if field == "next_entities":
        return serialize_entities(row["next_state"])
    raise ValueError(f"Unknown example field {field!r}")


def _tool_text(row: dict[str, Any], field: str) -> str:
    if field == "description":
        return serialize_tool_description(row)
    if field == "contract":
        return serialize_tool_contract(row)
    raise ValueError(f"Unknown tool field {field!r}")


def _pair_text(row: dict[str, Any], side: str) -> str:
    direct = row.get(f"{side}_text")
    if direct is not None:
        return str(direct)
    value = row.get(side, {})
    return canonical_json(value)


def _save_array(
    root: Path,
    relative: str,
    array: np.ndarray,
    dtype: str,
    files: dict[str, Any],
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    cast = array.astype(np.float16 if dtype == "float16" else np.float32)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, cast, allow_pickle=False)
    temporary.replace(path)
    files[relative] = {
        "rows": int(cast.shape[0]),
        "dim": int(cast.shape[1]) if cast.ndim == 2 else None,
        "dtype": str(cast.dtype),
        "sha256": sha256_file(path),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _resumable_array_is_valid(
    root: Path,
    relative: str,
    files: dict[str, Any],
    *,
    rows: int,
    dim: int,
    dtype: str,
) -> bool:
    metadata = files.get(relative)
    path = root / relative
    if not isinstance(metadata, dict) or not path.exists():
        return False
    expected_dtype = "float16" if dtype == "float16" else "float32"
    if (
        metadata.get("rows") != rows
        or metadata.get("dim") != dim
        or metadata.get("dtype") != expected_dtype
        or metadata.get("sha256") != sha256_file(path)
    ):
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return array.shape == (rows, dim) and str(array.dtype) == expected_dtype


def build_embedding_cache(
    processed_dir: str | Path,
    cache_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    cache_dir = Path(cache_dir)
    config = read_yaml(config_path)
    embedding_config = config.get("embedding", config)
    batch_size = int(embedding_config.get("batch_size", 32))
    storage_dtype = str(embedding_config.get("storage_dtype", "float32"))
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    config_sha256 = sha256_text(canonical_json(embedding_config))
    source_manifest_path = processed_dir / "manifest.json"
    verify_processed_dataset(processed_dir)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    existing_manifest = cache_dir / "manifest.json"
    progress_path = cache_dir / "build_progress.json"
    progress_backup_value = config.get("paths", {}).get("embedding_progress_backup")
    progress_backup_path = (
        Path(progress_backup_value) if progress_backup_value is not None else None
    )
    if existing_manifest.exists():
        existing = EmbeddingCache(cache_dir, verify=True).manifest
        if (
            existing.get("config_sha256") != config_sha256
            or existing.get("source_manifest_sha256") != source_manifest_sha256
        ):
            raise RuntimeError(
                "Refusing to overwrite an immutable embedding cache with a different "
                "configuration or processed dataset. Choose a new cache_dir."
            )
        if progress_backup_path is not None:
            _atomic_write_json(
                progress_backup_path,
                {
                    "status": "complete",
                    "cache_dir": str(cache_dir),
                    "config_sha256": config_sha256,
                    "source_manifest_sha256": source_manifest_sha256,
                    "completed_arrays": len(existing.get("files", {})),
                    "manifest_sha256": sha256_file(existing_manifest),
                },
            )
        return existing
    examples = read_jsonl(processed_dir / "examples.jsonl")
    tools = read_jsonl(processed_dir / "tools.jsonl")
    background_path = processed_dir / "background.jsonl"
    pair_path = processed_dir / "minimal_pairs.jsonl"
    background = read_jsonl(background_path) if background_path.exists() else []
    pairs = read_jsonl(pair_path) if pair_path.exists() else []
    views = embedding_config.get("views", {})
    if not views:
        raise ValueError("No embedding.views configured")

    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = read_json(progress_path) if progress_path.exists() else None
    if progress is not None and (
        progress.get("config_sha256") != config_sha256
        or progress.get("source_manifest_sha256") != source_manifest_sha256
    ):
        raise RuntimeError(
            "Refusing to resume an embedding cache built from a different configuration "
            "or processed dataset. Choose a new cache_dir."
        )
    files: dict[str, Any] = dict(progress.get("files", {})) if progress else {}

    def persist_progress(*, status: str = "running") -> None:
        payload = {
            "status": status,
            "cache_dir": str(cache_dir),
            "config_sha256": config_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "completed_arrays": len(files),
            "files": files,
        }
        if existing_manifest.exists():
            payload["manifest_sha256"] = sha256_file(existing_manifest)
        _atomic_write_json(progress_path, payload)
        if progress_backup_path is not None:
            _atomic_write_json(progress_backup_path, payload)

    def encode_and_save(
        relative: str,
        encoder: FrozenEncoder,
        texts: list[str],
    ) -> None:
        dim = int(encoder.metadata["dim"])
        if _resumable_array_is_valid(
            cache_dir,
            relative,
            files,
            rows=len(texts),
            dim=dim,
            dtype=storage_dtype,
        ):
            return
        vectors = encode_nonempty(encoder, texts, batch_size=batch_size)
        _save_array(cache_dir, relative, vectors, storage_dtype, files)
        persist_progress()

    persist_progress()
    view_manifest: dict[str, Any] = {}
    for view_name, view_config in views.items():
        query_encoder = build_encoder(view_config["query_encoder"])
        document_config = view_config.get("document_encoder", view_config["query_encoder"])
        document_encoder = (
            query_encoder
            if document_config == view_config["query_encoder"]
            else build_encoder(document_config)
        )
        example_fields = list(view_config.get("example_fields", []))
        tool_fields = list(view_config.get("tool_fields", []))
        successor_fields = {
            "state": "next_state",
            "sequence": "next_sequence",
            "entities": "next_entities",
        }
        for field, successor in successor_fields.items():
            if field in example_fields and successor not in example_fields:
                example_fields.append(successor)
        for field in example_fields:
            texts = [_example_text(row, field) for row in examples]
            encode_and_save(
                f"{view_name}/examples.{field}.npy", query_encoder, texts
            )
        for field in tool_fields:
            texts = [_tool_text(row, field) for row in tools]
            encode_and_save(
                f"{view_name}/tools.{field}.npy", document_encoder, texts
            )
        if view_config.get("encode_background", False) and background:
            encode_and_save(
                f"{view_name}/background.text.npy",
                document_encoder,
                [serialize_background(row) for row in background],
            )
        encode_pairs = bool(view_config.get("encode_pairs", True))
        if pairs and encode_pairs:
            for side in ("left", "right"):
                encode_and_save(
                    f"{view_name}/pairs.{side}.npy",
                    query_encoder,
                    [_pair_text(row, side) for row in pairs],
                )
        query_meta = query_encoder.metadata
        document_meta = document_encoder.metadata
        if tool_fields and int(query_meta["dim"]) != int(document_meta["dim"]):
            raise ValueError(
                f"View {view_name} query/document dimensions differ: "
                f"{query_meta['dim']} vs {document_meta['dim']}"
            )
        view_manifest[view_name] = {
            "kind": view_config.get("kind", "prototype"),
            "example_fields": example_fields,
            "tool_fields": tool_fields,
            "query_encoder": query_meta,
            "document_encoder": document_meta,
            "dim": int(query_meta["dim"]),
            "encode_pairs": encode_pairs,
            "snapshot_field": view_config.get("snapshot_field"),
        }

    manifest = {
        "cache_version": CACHE_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "config_sha256": config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "counts": {
            "examples": len(examples),
            "tools": len(tools),
            "background": len(background),
            "minimal_pairs": len(pairs),
        },
        "indices": {
            "examples": [row["example_id"] for row in examples],
            "tools": [row["tool_id"] for row in tools],
            "background": [row.get("card_id", str(index)) for index, row in enumerate(background)],
            "minimal_pairs": [row.get("pair_id", str(index)) for index, row in enumerate(pairs)],
        },
        "views": view_manifest,
        "files": files,
        "rules": {
            "backbones_frozen": True,
            "exact_fields_not_replaced_by_dense_vectors": True,
            "future_observations_excluded": True,
        },
    }
    _atomic_write_json(cache_dir / "manifest.json", manifest)
    persist_progress(status="complete")
    return manifest


class EmbeddingCache:
    def __init__(self, root: str | Path, *, verify: bool = True):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version: {self.manifest.get('cache_version')}")
        if self.manifest.get("feature_spec_version") != FEATURE_SPEC_VERSION:
            raise ValueError(
                "Embedding cache feature semantics differ from this runtime: "
                f"{self.manifest.get('feature_spec_version')!r} != {FEATURE_SPEC_VERSION!r}"
            )
        self.example_ids = list(self.manifest["indices"]["examples"])
        self.tool_ids = list(self.manifest["indices"]["tools"])
        self.example_index = {value: index for index, value in enumerate(self.example_ids)}
        self.tool_index = {value: index for index, value in enumerate(self.tool_ids)}
        self._arrays: dict[str, np.ndarray] = {}
        if verify:
            self.verify()

    @property
    def content_sha256(self) -> str:
        """Bind the complete canonical manifest, including every array checksum."""

        return sha256_text(canonical_json(self.manifest))

    @property
    def evaluation_fingerprint_sha256(self) -> str:
        """Bind the frozen coordinate system and tool prototypes, not example rows.

        A checkpoint selected on a physically sealed train+dev cache may be used
        with a separately materialized test cache only when this fingerprint is
        identical.  Full view metadata binds the actually loaded encoders/dtypes;
        tool array hashes and ordered IDs bind action semantics.  Example arrays
        are intentionally excluded because evaluation examples must be new.
        """

        tool_files = {
            relative: metadata
            for relative, metadata in self.manifest["files"].items()
            if relative.split("/", 1)[-1].startswith("tools.")
        }
        payload = {
            "cache_version": self.manifest["cache_version"],
            "feature_spec_version": self.manifest["feature_spec_version"],
            "config_sha256": self.manifest["config_sha256"],
            "views": self.manifest["views"],
            "tool_ids": self.manifest["indices"]["tools"],
            "tool_files": tool_files,
        }
        return sha256_text(canonical_json(payload))

    @property
    def prototype_views(self) -> list[str]:
        return [
            name
            for name, view in self.manifest["views"].items()
            if view.get("tool_fields") and view.get("kind", "prototype") == "prototype"
        ]

    @property
    def state_only_views(self) -> list[str]:
        return [
            name
            for name, view in self.manifest["views"].items()
            if view.get("kind") == "state_only"
        ]

    def verify(self) -> None:
        for relative, metadata in self.manifest["files"].items():
            path = self.root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            if sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"Embedding cache checksum mismatch: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (int(metadata["rows"]), int(metadata["dim"]))
            if array.ndim != 2 or tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Embedding cache shape mismatch: {path}; "
                    f"expected={expected_shape}, actual={tuple(array.shape)}"
                )
            if str(array.dtype) != str(metadata["dtype"]):
                raise ValueError(
                    f"Embedding cache dtype mismatch: {path}; "
                    f"expected={metadata['dtype']}, actual={array.dtype}"
                )
            for start in range(0, len(array), 4096):
                if not np.isfinite(array[start : start + 4096]).all():
                    raise ValueError(f"Embedding cache contains non-finite values: {path}")

            prefix = relative.split("/", 1)[-1]
            if prefix.startswith("examples."):
                expected_rows = len(self.example_ids)
            elif prefix.startswith("tools."):
                expected_rows = len(self.tool_ids)
            elif prefix == "background.text.npy":
                expected_rows = len(self.manifest["indices"].get("background", []))
            elif prefix.startswith("pairs."):
                expected_rows = len(self.manifest["indices"].get("minimal_pairs", []))
            else:
                raise ValueError(f"Unrecognized embedding cache file role: {relative}")
            if len(array) != expected_rows:
                raise ValueError(
                    f"Embedding cache row/index mismatch: {path}; "
                    f"rows={len(array)}, ids={expected_rows}"
                )

    def array(self, relative: str) -> np.ndarray:
        if relative not in self._arrays:
            self._arrays[relative] = np.load(
                self.root / relative, mmap_mode="r", allow_pickle=False
            )
        return self._arrays[relative]

    def example_field(self, view: str, field: str) -> np.ndarray:
        return self.array(f"{view}/examples.{field}.npy")

    def tool_field(self, view: str, field: str) -> np.ndarray:
        return self.array(f"{view}/tools.{field}.npy")

    @staticmethod
    def _masked_mean(arrays: list[np.ndarray]) -> np.ndarray:
        if not arrays:
            raise ValueError("Cannot pool an empty list of fields")
        stacked = np.stack([np.asarray(item, dtype=np.float32) for item in arrays], axis=1)
        mask = np.linalg.norm(stacked, axis=-1, keepdims=True) > 0
        return (stacked * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)

    def example_matrix(self, view: str, *, next_state: bool = False) -> np.ndarray:
        view_meta = self.manifest["views"][view]
        if next_state:
            return np.asarray(self.example_field(view, "next_state"), dtype=np.float32)
        fields = [field for field in view_meta["example_fields"] if not field.startswith("next_")]
        return self._masked_mean([self.example_field(view, field) for field in fields])

    def context_matrix(self, view: str, examples: list[dict[str, Any]]) -> np.ndarray:
        """Pool the same model-facing fields and background cards used downstream."""

        if len(examples) != len(self.example_ids):
            raise ValueError("Example rows do not match the embedding cache index")
        fields = [
            field
            for field in self.manifest["views"][view]["example_fields"]
            if not field.startswith("next_")
        ]
        field_arrays = [
            np.asarray(self.example_field(view, field), dtype=np.float32) for field in fields
        ]
        dim = int(self.manifest["views"][view]["dim"])
        pooled = np.zeros((len(examples), dim), dtype=np.float32)
        relative = f"{view}/background.text.npy"
        cards = (
            np.asarray(self.array(relative), dtype=np.float32)
            if relative in self.manifest["files"]
            else None
        )
        background_index = {
            value: index
            for index, value in enumerate(self.manifest["indices"].get("background", []))
        }
        for row_index, example in enumerate(examples):
            pieces = [
                array[row_index] for array in field_arrays if np.linalg.norm(array[row_index]) > 0
            ]
            if cards is not None:
                selected = [
                    cards[background_index[card_id]]
                    for card_id in example.get("background_ids", [])
                    if card_id in background_index
                ]
                if selected:
                    pieces.append(np.stack(selected).mean(axis=0))
            if pieces:
                pooled[row_index] = np.stack(pieces).mean(axis=0)
        return pooled

    def tool_matrix(self, view: str) -> np.ndarray:
        fields = self.manifest["views"][view]["tool_fields"]
        return self._masked_mean([self.tool_field(view, field) for field in fields])


def verify_evaluation_cache_superset(
    training_cache_dir: str | Path,
    evaluation_cache: EmbeddingCache,
) -> dict[str, Any]:
    """Verify that a larger evaluation registry preserves trained prototypes."""

    training = EmbeddingCache(training_cache_dir, verify=True)
    for key in ("cache_version", "feature_spec_version", "config_sha256", "views"):
        if training.manifest.get(key) != evaluation_cache.manifest.get(key):
            raise ValueError(
                f"Evaluation cache superset has different {key}; frozen coordinates changed"
            )

    training_ids = list(training.tool_ids)
    evaluation_ids = list(evaluation_cache.tool_ids)
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("Evaluation cache contains duplicate tool IDs")
    missing = sorted(set(training_ids) - set(evaluation_ids))
    if missing:
        raise ValueError(f"Evaluation cache is missing training tools: {missing}")
    evaluation_index = {tool_id: index for index, tool_id in enumerate(evaluation_ids)}
    evaluation_rows = np.asarray([evaluation_index[tool_id] for tool_id in training_ids])

    checked_files: list[str] = []
    for relative, metadata in training.manifest["files"].items():
        if not relative.split("/", 1)[-1].startswith("tools."):
            continue
        evaluation_metadata = evaluation_cache.manifest["files"].get(relative)
        if evaluation_metadata is None:
            raise ValueError(f"Evaluation cache lacks tool prototype array {relative}")
        if metadata.get("dim") != evaluation_metadata.get("dim"):
            raise ValueError(f"Tool prototype width changed for {relative}")
        training_array = np.load(training.root / relative, mmap_mode="r", allow_pickle=False)
        evaluation_array = np.load(
            evaluation_cache.root / relative, mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(training_array, evaluation_array[evaluation_rows]):
            raise ValueError(
                f"Shared training tool prototypes changed in evaluation cache: {relative}"
            )
        checked_files.append(relative)

    if not checked_files:
        raise ValueError("No tool prototype arrays were available for superset verification")
    return {
        "compatible": True,
        "training_tool_count": len(training_ids),
        "evaluation_tool_count": len(evaluation_ids),
        "unseen_tool_ids": sorted(set(evaluation_ids) - set(training_ids)),
        "checked_tool_arrays": checked_files,
        "training_evaluation_fingerprint_sha256": training.evaluation_fingerprint_sha256,
        "evaluation_evaluation_fingerprint_sha256": (
            evaluation_cache.evaluation_fingerprint_sha256
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract immutable frozen embedding caches")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_embedding_cache(args.processed_dir, args.cache_dir, args.config)
    print(canonical_json(manifest["counts"]))


if __name__ == "__main__":
    main()

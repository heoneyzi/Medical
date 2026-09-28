from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from geoflowagent.constants import FEATURE_SPEC_VERSION
from geoflowagent.embeddings.cache import (
    EmbeddingCache,
    build_embedding_cache,
    verify_evaluation_cache_superset,
)
from geoflowagent.embeddings.huggingface import HuggingFaceEncoder
from geoflowagent.embeddings.runtime import RuntimeFeatureEncoder
from geoflowagent.training.features import FeatureStore
from geoflowagent.utils.io import read_json, read_jsonl, read_yaml, sha256_file, write_json


def test_hash_embedding_cache_is_reproducible_and_fieldwise(
    processed_dir: Path,
    repository_root: Path,
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config = repository_root / "configs" / "smoke.yaml"
    first = build_embedding_cache(processed_dir, first_dir, config)
    second = build_embedding_cache(processed_dir, second_dir, config)

    assert first["config_sha256"] == second["config_sha256"]
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert first["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert set(first["files"]) == set(second["files"])
    for relative in first["files"]:
        assert first["files"][relative]["sha256"] == second["files"][relative]["sha256"]
        np.testing.assert_array_equal(
            np.load(first_dir / relative, allow_pickle=False),
            np.load(second_dir / relative, allow_pickle=False),
        )

    cache = EmbeddingCache(first_dir)
    assert cache.prototype_views == ["general", "biomedical"] or set(cache.prototype_views) == {
        "general",
        "biomedical",
    }
    assert cache.state_only_views == ["sequence_aux"]
    sequence_meta = cache.manifest["views"]["sequence_aux"]
    assert sequence_meta["snapshot_field"] == "sequence"
    assert sequence_meta["example_fields"] == ["sequence", "next_sequence"]
    for view in cache.prototype_views:
        view_meta = cache.manifest["views"][view]
        assert "next_state" in view_meta["example_fields"]
        assert cache.example_matrix(view).shape == (
            first["counts"]["examples"],
            view_meta["dim"],
        )
        assert cache.example_matrix(view, next_state=True).shape == cache.example_matrix(view).shape
        assert cache.tool_matrix(view).shape == (first["counts"]["tools"], view_meta["dim"])


def test_cache_checksum_detects_file_tampering(cache_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "tampered"
    import shutil

    shutil.copytree(cache_dir, copied)
    cache = EmbeddingCache(copied)
    relative = next(iter(cache.manifest["files"]))
    path = copied / relative
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        EmbeddingCache(copied, verify=True)


def test_cache_content_fingerprint_changes_for_rebuilt_array_bytes(
    cache_dir: Path, tmp_path: Path
) -> None:
    import shutil

    copied = tmp_path / "reencoded"
    shutil.copytree(cache_dir, copied)
    original = EmbeddingCache(cache_dir)
    manifest_path = copied / "manifest.json"
    manifest = read_json(manifest_path)
    relative = next(iter(manifest["files"]))
    path = copied / relative
    array = np.load(path, allow_pickle=False)
    array[0, 0] += np.asarray(0.25, dtype=array.dtype)
    np.save(path, array, allow_pickle=False)
    manifest["files"][relative]["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    rebuilt = EmbeddingCache(copied, verify=True)

    assert rebuilt.manifest["config_sha256"] == original.manifest["config_sha256"]
    assert rebuilt.manifest["source_manifest_sha256"] == original.manifest["source_manifest_sha256"]
    assert rebuilt.content_sha256 != original.content_sha256


def test_cache_verifies_shape_and_finite_values(cache_dir: Path, tmp_path: Path) -> None:
    import shutil

    malformed_shape = tmp_path / "malformed-shape"
    shutil.copytree(cache_dir, malformed_shape)
    manifest = read_json(malformed_shape / "manifest.json")
    relative = next(iter(manifest["files"]))
    manifest["files"][relative]["dim"] += 1
    write_json(malformed_shape / "manifest.json", manifest)
    with pytest.raises(ValueError, match="shape mismatch"):
        EmbeddingCache(malformed_shape, verify=True)

    nonfinite = tmp_path / "nonfinite"
    shutil.copytree(cache_dir, nonfinite)
    manifest = read_json(nonfinite / "manifest.json")
    relative = next(iter(manifest["files"]))
    path = nonfinite / relative
    array = np.load(path, allow_pickle=False)
    array[0, 0] = np.nan
    np.save(path, array, allow_pickle=False)
    manifest["files"][relative]["sha256"] = sha256_file(path)
    write_json(nonfinite / "manifest.json", manifest)
    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingCache(nonfinite, verify=True)


def test_cache_refuses_stale_feature_semantics(cache_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "stale-feature-spec"
    import shutil

    shutil.copytree(cache_dir, copied)
    manifest_path = copied / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["feature_spec_version"] = "geoflowagent.feature-spec.stale"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="feature semantics differ"):
        EmbeddingCache(copied, verify=True)


def test_same_embedding_cache_is_verified_and_reused(
    processed_dir: Path,
    cache_dir: Path,
    repository_root: Path,
) -> None:
    before = (cache_dir / "manifest.json").stat().st_mtime_ns
    manifest = build_embedding_cache(
        processed_dir, cache_dir, repository_root / "configs/smoke.yaml"
    )
    after = (cache_dir / "manifest.json").stat().st_mtime_ns

    assert manifest["counts"]["examples"] > 0
    assert after == before


def test_processed_dataset_tampering_is_detected_before_training(
    processed_dir: Path,
    cache_dir: Path,
    tmp_path: Path,
) -> None:
    import shutil

    copied = tmp_path / "processed"
    shutil.copytree(processed_dir, copied)
    path = copied / "examples.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Processed dataset checksum mismatch"):
        FeatureStore(copied, cache_dir, structured_dim=48)


def test_cached_training_context_matches_runtime_pooling(
    processed_dir: Path,
    cache_dir: Path,
    repository_root: Path,
) -> None:
    config = repository_root / "configs" / "smoke.yaml"
    store = FeatureStore(processed_dir, cache_dir, structured_dim=48)
    runtime = RuntimeFeatureEncoder(
        config,
        store.views + store.aux_views,
        structured_dim=store.structured_dim,
    )
    tasks = {item["task_id"]: item for item in read_jsonl(processed_dir / "tasks.jsonl")}
    cards = {item["card_id"]: item for item in read_jsonl(processed_dir / "background.jsonl")}
    device = torch.device("cpu")
    for row in store.examples:
        task = tasks[row["task_id"]]
        background = [cards[card_id] for card_id in row["background_ids"]]
        online_views, online_structured = runtime.encode(
            task, row["state"], row["history"], background
        )
        cached = store.runtime_features(task, row["state"], row["history"], device)
        assert cached is not None
        cached_views, cached_structured = cached

        for view in store.views + store.aux_views:
            np.testing.assert_allclose(
                online_views[view].numpy(),
                cached_views[view].numpy(),
                atol=1e-7,
                err_msg=f"cache/runtime mismatch for {row['example_id']} view={view}",
            )
        np.testing.assert_allclose(
            online_structured.numpy(),
            cached_structured.numpy(),
            atol=1e-7,
            err_msg=f"structured cache/runtime mismatch for {row['example_id']}",
        )


def test_float16_cache_and_runtime_apply_the_same_storage_roundtrip(
    processed_dir: Path,
    repository_root: Path,
    tmp_path: Path,
) -> None:
    config = read_yaml(repository_root / "configs" / "smoke.yaml")
    config["embedding"]["storage_dtype"] = "float16"
    config_path = tmp_path / "float16.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cache_dir = tmp_path / "float16-cache"
    build_embedding_cache(processed_dir, cache_dir, config_path)
    store = FeatureStore(processed_dir, cache_dir, structured_dim=48)
    runtime = RuntimeFeatureEncoder(
        config_path,
        store.views + store.aux_views,
        structured_dim=store.structured_dim,
    )
    row = store.examples[0]
    task = next(
        item
        for item in read_jsonl(processed_dir / "tasks.jsonl")
        if item["task_id"] == row["task_id"]
    )
    cards = {item["card_id"]: item for item in read_jsonl(processed_dir / "background.jsonl")}
    background = [cards[card_id] for card_id in row["background_ids"]]

    online_views, _ = runtime.encode(task, row["state"], row["history"], background)
    cached = store.runtime_features(task, row["state"], row["history"], torch.device("cpu"))
    assert cached is not None
    cached_views, _ = cached
    for view in store.views + store.aux_views:
        torch.testing.assert_close(online_views[view], cached_views[view], rtol=0, atol=0)


def test_runtime_encoder_rejects_metadata_that_differs_from_cache(
    processed_dir: Path,
    cache_dir: Path,
    repository_root: Path,
) -> None:
    store = FeatureStore(processed_dir, cache_dir, structured_dim=48)
    expected = {
        name: {
            **metadata,
            "query_encoder": {**metadata["query_encoder"]},
            "document_encoder": {**metadata["document_encoder"]},
        }
        for name, metadata in store.cache.manifest["views"].items()
    }
    expected[store.views[0]]["query_encoder"]["loaded_dtype"] = "different-host-dtype"
    runtime = RuntimeFeatureEncoder(
        repository_root / "configs" / "smoke.yaml",
        store.views + store.aux_views,
        structured_dim=store.structured_dim,
        expected_view_metadata=expected,
    )

    with pytest.raises(RuntimeError, match="metadata differs.*loaded_dtype"):
        runtime.warmup()


def test_runtime_prefix_lookup_hits_exact_context_and_misses_new_history(
    processed_dir: Path,
    cache_dir: Path,
) -> None:
    store = FeatureStore(processed_dir, cache_dir, structured_dim=48)
    row = store.examples[0]
    task = next(
        item
        for item in read_jsonl(processed_dir / "tasks.jsonl")
        if item["task_id"] == row["task_id"]
    )

    device = torch.device("cpu")
    assert store.runtime_features(task, row["state"], row["history"], device=device) is not None
    novel_history = [*row["history"], {"tool_id": "novel_failure", "status": "failure"}]
    assert store.runtime_features(task, row["state"], novel_history, device=device) is None


def test_sequence_encoder_input_rejects_non_dna_metadata(
    processed_dir: Path,
    cache_dir: Path,
    repository_root: Path,
) -> None:
    store = FeatureStore(processed_dir, cache_dir, structured_dim=48)
    runtime = RuntimeFeatureEncoder(
        repository_root / "configs" / "smoke.yaml",
        store.views + store.aux_views,
        structured_dim=48,
    )
    row = store.examples[0]
    task = next(
        item
        for item in read_jsonl(processed_dir / "tasks.jsonl")
        if item["task_id"] == row["task_id"]
    )
    state = {**row["state"], "sequence_context": "GRCh38"}

    with pytest.raises(ValueError, match="IUPAC DNA"):
        runtime.encode(task, state, row["history"], [])


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "deadbeef"])
def test_huggingface_encoder_refuses_mutable_or_short_revisions(revision: str | None) -> None:
    with pytest.raises(ValueError, match="revision|Pin an immutable"):
        HuggingFaceEncoder("example/not-downloaded", revision=revision)


def test_last_token_pooling_handles_left_and_right_padding() -> None:
    encoder = HuggingFaceEncoder.__new__(HuggingFaceEncoder)
    encoder.pooling = "last_token"
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

    pooled = encoder._pool(hidden, mask)

    torch.testing.assert_close(pooled[0], hidden[0, 1])
    torch.testing.assert_close(pooled[1], hidden[1, 3])


def test_evaluation_fingerprint_excludes_examples_but_binds_tools() -> None:
    cache = EmbeddingCache.__new__(EmbeddingCache)
    cache.manifest = {
        "cache_version": "cache-v1",
        "feature_spec_version": "features-v1",
        "config_sha256": "embedding-config",
        "indices": {"tools": ["tool-a"], "examples": ["example-a"]},
        "views": {"general": {"dim": 4, "query_encoder": {"revision": "abc"}}},
        "files": {
            "general/examples.state.npy": {"sha256": "example-bytes"},
            "general/tools.description.npy": {"sha256": "tool-bytes"},
        },
    }
    original = cache.evaluation_fingerprint_sha256

    cache.manifest["files"]["general/examples.state.npy"]["sha256"] = "new-test-examples"
    assert cache.evaluation_fingerprint_sha256 == original

    cache.manifest["files"]["general/tools.description.npy"]["sha256"] = "changed-tool"
    assert cache.evaluation_fingerprint_sha256 != original


def test_evaluation_cache_superset_preserves_shared_tool_prototypes(
    cache_dir: Path, tmp_path: Path
) -> None:
    import shutil

    expanded = tmp_path / "expanded"
    shutil.copytree(cache_dir, expanded)
    manifest_path = expanded / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["indices"]["tools"].append("unseen-tool")
    manifest["counts"]["tools"] += 1
    for relative, metadata in manifest["files"].items():
        if not relative.split("/", 1)[-1].startswith("tools."):
            continue
        path = expanded / relative
        array = np.load(path, allow_pickle=False)
        array = np.concatenate([array, np.zeros_like(array[:1])], axis=0)
        np.save(path, array, allow_pickle=False)
        metadata["rows"] += 1
        metadata["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)

    report = verify_evaluation_cache_superset(cache_dir, EmbeddingCache(expanded))

    assert report["compatible"] is True
    assert report["unseen_tool_ids"] == ["unseen-tool"]

    relative = report["checked_tool_arrays"][0]
    path = expanded / relative
    array = np.load(path, allow_pickle=False)
    array[0, 0] += np.asarray(0.25, dtype=array.dtype)
    np.save(path, array, allow_pickle=False)
    manifest = read_json(manifest_path)
    manifest["files"][relative]["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="prototypes changed"):
        verify_evaluation_cache_superset(cache_dir, EmbeddingCache(expanded))

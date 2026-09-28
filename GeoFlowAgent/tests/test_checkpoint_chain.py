from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.evaluation.agent import _validate_checkpoint_chain
from geoflowagent.training.flow import load_flow_checkpoint
from geoflowagent.training.metric import load_metric_checkpoint
from geoflowagent.utils.io import sha256_file


def _bound_metadata(metric_path: Path) -> tuple[dict, dict, SimpleNamespace]:
    manifest = {
        "config_sha256": "cache-config",
        "source_manifest_sha256": "processed-source",
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }
    common = {
        "tool_ids": ["tool-a"],
        "cache_config_sha256": manifest["config_sha256"],
        "cache_content_sha256": "cache-content",
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "feature_spec_version": manifest["feature_spec_version"],
    }
    geometry = dict(common)
    flow = {**common, "metric_checkpoint_sha256": sha256_file(metric_path)}
    store = SimpleNamespace(
        tool_ids=["tool-a"],
        cache=SimpleNamespace(manifest=manifest, content_sha256="cache-content"),
    )
    return geometry, flow, store


def test_checkpoint_chain_binds_exact_metric_file_and_cache(tmp_path: Path) -> None:
    metric_path = tmp_path / "metric.pt"
    metric_path.write_bytes(b"exact metric checkpoint bytes")
    geometry, flow, store = _bound_metadata(metric_path)

    _validate_checkpoint_chain(metric_path, geometry, flow, store)

    metric_path.write_bytes(b"different metric checkpoint bytes")
    with pytest.raises(ValueError, match="different metric checkpoint"):
        _validate_checkpoint_chain(metric_path, geometry, flow, store)


@pytest.mark.parametrize("component", ["Metric", "Flow"])
def test_checkpoint_chain_binds_cache_array_content(component: str, tmp_path: Path) -> None:
    metric_path = tmp_path / "metric.pt"
    metric_path.write_bytes(b"metric")
    geometry, flow, store = _bound_metadata(metric_path)
    target = geometry if component == "Metric" else flow
    target["cache_content_sha256"] = "rebuilt-with-different-array-bytes"

    with pytest.raises(ValueError, match=f"{component} checkpoint.*different embedding cache"):
        _validate_checkpoint_chain(metric_path, geometry, flow, store)


@pytest.mark.parametrize("component", ["Metric", "Flow"])
def test_checkpoint_chain_binds_processed_source(component: str, tmp_path: Path) -> None:
    metric_path = tmp_path / "metric.pt"
    metric_path.write_bytes(b"metric")
    geometry, flow, store = _bound_metadata(metric_path)
    target = geometry if component == "Metric" else flow
    target["source_manifest_sha256"] = "different-source"

    with pytest.raises(ValueError, match=f"{component} checkpoint.*different processed dataset"):
        _validate_checkpoint_chain(metric_path, geometry, flow, store)


@pytest.mark.parametrize(
    ("loader", "kind"),
    [
        (load_metric_checkpoint, "functional_geometry"),
        (load_flow_checkpoint, "whole_plan_flow"),
    ],
)
def test_checkpoint_loaders_reject_wrong_version(loader, kind: str, tmp_path: Path) -> None:
    path = tmp_path / f"{kind}.pt"
    torch.save(
        {
            "checkpoint_version": "geoflowagent.checkpoint.stale",
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "kind": kind,
        },
        path,
    )

    with pytest.raises(ValueError, match="checkpoint version"):
        loader(path, torch.device("cpu"))


@pytest.mark.parametrize(
    ("loader", "kind"),
    [
        (load_metric_checkpoint, "functional_geometry"),
        (load_flow_checkpoint, "whole_plan_flow"),
    ],
)
def test_checkpoint_loaders_reject_wrong_feature_spec(loader, kind: str, tmp_path: Path) -> None:
    path = tmp_path / f"{kind}.pt"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "feature_spec_version": "geoflowagent.feature-spec.stale",
            "kind": kind,
        },
        path,
    )

    with pytest.raises(ValueError, match="feature semantics differ"):
        loader(path, torch.device("cpu"))

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoflowagent.constants import STOP_TOOL_ID
from geoflowagent.data.preprocess import verify_processed_dataset
from geoflowagent.data.serialize import runtime_context_key
from geoflowagent.data.structured import StructuredHasher
from geoflowagent.embeddings.cache import EmbeddingCache
from geoflowagent.utils.io import read_jsonl, sha256_file


class FeatureStore:
    """Joins processed examples with immutable caches by stable ID."""

    def __init__(
        self,
        processed_dir: str | Path,
        cache_dir: str | Path,
        *,
        structured_dim: int = 64,
        verify_cache: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        processed_manifest = verify_processed_dataset(self.processed_dir)
        if processed_manifest.get("dataset_kind") == "weighted_search_supervision":
            raise ValueError(
                "The demonstration FeatureStore cannot consume weighted search supervision: "
                "unreachable states and top-k paths need explicit masks/sampling. Use "
                "SearchFeatureStore with train-value; a State Flow adapter is a separate stage."
            )
        self.examples = read_jsonl(self.processed_dir / "examples.jsonl")
        self.tools = read_jsonl(self.processed_dir / "tools.jsonl")
        self.cache = EmbeddingCache(cache_dir, verify=verify_cache)
        if self.cache.manifest["source_manifest_sha256"] != sha256_file(
            self.processed_dir / "manifest.json"
        ):
            raise ValueError("Embedding cache was built from a different processed manifest")
        if processed_manifest["processed_files"].get("examples.jsonl") is None:
            raise ValueError("Processed manifest does not bind examples.jsonl")
        example_ids = [row["example_id"] for row in self.examples]
        tool_ids = [row["tool_id"] for row in self.tools]
        if example_ids != self.cache.example_ids:
            raise ValueError("Processed examples and embedding cache order/IDs differ")
        if tool_ids != self.cache.tool_ids:
            raise ValueError("Processed tools and embedding cache order/IDs differ")
        self.tool_ids = tool_ids
        self.tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
        self.views = self.cache.prototype_views
        self.aux_views = self.cache.state_only_views
        if not self.views:
            raise ValueError("At least one prototype view with tool fields is required")
        self.state_views = {
            view: torch.from_numpy(self.cache.context_matrix(view, self.examples)).float()
            for view in self.views
        }
        self.state_snapshot_views = {
            view: torch.from_numpy(
                np.array(self.cache.example_field(view, "state"), copy=True)
            ).float()
            for view in self.views
        }
        self.next_state_views = {
            view: torch.from_numpy(
                np.array(self.cache.example_matrix(view, next_state=True), copy=True)
            ).float()
            for view in self.views
        }
        self.tool_views = {
            view: torch.from_numpy(np.array(self.cache.tool_matrix(view), copy=True)).float()
            for view in self.views
        }
        self.aux_state_views = {
            view: torch.from_numpy(self.cache.context_matrix(view, self.examples)).float()
            for view in self.aux_views
        }
        self.aux_snapshot_views = {}
        self.aux_next_state_views = {}
        for view in self.aux_views:
            view_meta = self.cache.manifest["views"][view]
            fields = view_meta["example_fields"]
            current_fields = [field for field in fields if not field.startswith("next_")]
            snapshot_field = view_meta.get("snapshot_field")
            if snapshot_field is None:
                snapshot_field = "state" if "state" in current_fields else current_fields[0]
            if snapshot_field not in current_fields:
                raise ValueError(f"Invalid snapshot_field={snapshot_field!r} for view {view}")
            successor_field = f"next_{snapshot_field}"
            next_field = successor_field if successor_field in fields else snapshot_field
            self.aux_snapshot_views[view] = torch.from_numpy(
                np.array(self.cache.example_field(view, snapshot_field), copy=True)
            ).float()
            self.aux_next_state_views[view] = torch.from_numpy(
                np.array(self.cache.example_field(view, next_field), copy=True)
            ).float()
        hasher = StructuredHasher(structured_dim)
        self.structured_state = torch.from_numpy(
            np.stack([hasher.encode_state(row["state"]) for row in self.examples])
        ).float()
        self.structured_next_state = torch.from_numpy(
            np.stack([hasher.encode_state(row["next_state"]) for row in self.examples])
        ).float()
        self.structured_tools = torch.from_numpy(
            np.stack([hasher.encode_tool(row) for row in self.tools])
        ).float()
        self.structured_dim = structured_dim
        self.runtime_context_index: dict[str, int] = {}
        for index, row in enumerate(self.examples):
            key = runtime_context_key(row, row["state"], row["history"])
            previous = self.runtime_context_index.setdefault(key, index)
            if previous != index:
                previous_row = self.examples[previous]
                if previous_row["task_id"] != row["task_id"]:
                    raise ValueError(f"Runtime context hash collision for {key}")

    @property
    def view_dims(self) -> dict[str, int]:
        return {view: int(matrix.shape[1]) for view, matrix in self.state_views.items()}

    @property
    def aux_dims(self) -> dict[str, int]:
        return {view: int(matrix.shape[1]) for view, matrix in self.aux_state_views.items()}

    def indices(self, split: str, *, include_terminal: bool) -> list[int]:
        return [
            index
            for index, row in enumerate(self.examples)
            if row["split"] == split and (include_terminal or not row["terminal"])
        ]

    def batch(self, indices: Iterable[int], device: torch.device) -> dict[str, Any]:
        index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
        rows = [self.examples[int(index)] for index in index_tensor]
        batch_size = len(rows)
        tool_count = len(self.tool_ids)
        candidate_mask = torch.zeros((batch_size, tool_count), dtype=torch.bool)
        contract_mask = torch.zeros((batch_size, tool_count), dtype=torch.bool)
        valid_mask = torch.zeros((batch_size, tool_count), dtype=torch.bool)
        regret = torch.zeros((batch_size, tool_count), dtype=torch.float32)
        regret_mask = torch.zeros((batch_size, tool_count), dtype=torch.bool)
        gold = torch.full((batch_size,), -1, dtype=torch.long)
        terminal = torch.zeros((batch_size,), dtype=torch.bool)
        for batch_index, row in enumerate(rows):
            terminal[batch_index] = bool(row["terminal"])
            for tool_id in row["candidate_tools"]:
                if tool_id in self.tool_index:
                    candidate_mask[batch_index, self.tool_index[tool_id]] = True
            for tool_id in row["contract_valid_tools"]:
                if tool_id in self.tool_index:
                    contract_mask[batch_index, self.tool_index[tool_id]] = True
            for tool_id in row["valid_next_tools"]:
                if tool_id in self.tool_index:
                    valid_mask[batch_index, self.tool_index[tool_id]] = True
            if row["gold_next_tool"] in self.tool_index:
                gold[batch_index] = self.tool_index[row["gold_next_tool"]]
            for tool_id, tool_index in self.tool_index.items():
                value = row.get("action_regret", {}).get(tool_id)
                is_known = bool(row.get("action_regret_mask", {}).get(tool_id, value is not None))
                if value is not None:
                    regret[batch_index, tool_index] = float(value)
                regret_mask[batch_index, tool_index] = is_known and value is not None
        return {
            "indices": index_tensor.to(device),
            "rows": rows,
            "state_views": {
                view: matrix[index_tensor].to(device) for view, matrix in self.state_views.items()
            },
            "state_snapshot_views": {
                view: matrix[index_tensor].to(device)
                for view, matrix in self.state_snapshot_views.items()
            },
            "next_state_views": {
                view: matrix[index_tensor].to(device)
                for view, matrix in self.next_state_views.items()
            },
            "tool_views": {view: matrix.to(device) for view, matrix in self.tool_views.items()},
            "aux_views": {
                view: matrix[index_tensor].to(device)
                for view, matrix in self.aux_state_views.items()
            },
            "aux_snapshot_views": {
                view: matrix[index_tensor].to(device)
                for view, matrix in self.aux_snapshot_views.items()
            },
            "aux_next_state_views": {
                view: matrix[index_tensor].to(device)
                for view, matrix in self.aux_next_state_views.items()
            },
            "structured_state": self.structured_state[index_tensor].to(device),
            "structured_next_state": self.structured_next_state[index_tensor].to(device),
            "structured_tools": self.structured_tools.to(device),
            "candidate_mask": candidate_mask.to(device),
            "contract_mask": contract_mask.to(device),
            "valid_mask": valid_mask.to(device),
            "regret": regret.to(device),
            "regret_mask": regret_mask.to(device),
            "gold": gold.to(device),
            "terminal": terminal.to(device),
        }

    def suffix_indices(self, row: dict[str, Any]) -> list[int]:
        output = []
        for tool_id in row["gold_suffix_tool_ids"]:
            if tool_id == STOP_TOOL_ID:
                raise ValueError("STOP must not be stored inside gold_suffix_tool_ids")
            output.append(self.tool_index[tool_id])
        return output

    def runtime_features(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        """Return cached features for an exactly known visible prefix, if present."""

        index = self.runtime_context_index.get(runtime_context_key(task, state, history))
        if index is None:
            return None
        views = {
            **{
                view: matrix[index].unsqueeze(0).to(device)
                for view, matrix in self.state_views.items()
            },
            **{
                view: matrix[index].unsqueeze(0).to(device)
                for view, matrix in self.aux_state_views.items()
            },
        }
        structured = self.structured_state[index].unsqueeze(0).to(device)
        return views, structured

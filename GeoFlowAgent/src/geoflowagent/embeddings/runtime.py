from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoflowagent.data.serialize import (
    serialize_background,
    serialize_entities,
    serialize_goal,
    serialize_history,
    serialize_sequence,
    serialize_state,
)
from geoflowagent.data.structured import StructuredHasher
from geoflowagent.embeddings.base import encode_nonempty
from geoflowagent.embeddings.cache import build_encoder
from geoflowagent.utils.io import canonical_json, read_yaml


class RuntimeFeatureEncoder:
    """Encode genuinely new closed-loop states using the same frozen view config."""

    def __init__(
        self,
        config_path: str | Path,
        view_names: list[str],
        *,
        structured_dim: int,
        batch_size: int = 16,
        expected_view_metadata: dict[str, Any] | None = None,
    ) -> None:
        config = read_yaml(config_path)
        embedding = config.get("embedding", config)
        self.batch_size = int(embedding.get("batch_size", batch_size))
        self.storage_dtype = str(embedding.get("storage_dtype", "float32"))
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32")
        self.views: dict[str, dict[str, Any]] = {
            view: dict(embedding["views"][view]) for view in view_names
        }
        self.expected_view_metadata = expected_view_metadata
        if expected_view_metadata is not None:
            missing = sorted(set(view_names) - set(expected_view_metadata))
            if missing:
                raise ValueError(f"Embedding cache metadata is missing runtime views: {missing}")
        self._encoders: dict[str, dict[str, Any]] = {}
        self.structured = StructuredHasher(structured_dim)

    def warmup(self) -> None:
        """Load each frozen encoder outside per-condition latency accounting."""

        for view in self.views:
            self._encoders_for_view(view)

    def _storage_roundtrip(self, values: np.ndarray) -> np.ndarray:
        dtype = np.float16 if self.storage_dtype == "float16" else np.float32
        return np.asarray(values, dtype=dtype).astype(np.float32)

    def _encoders_for_view(self, view: str) -> dict[str, Any]:
        if view in self._encoders:
            return self._encoders[view]
        view_config = self.views[view]
        query_encoder = build_encoder(view_config["query_encoder"])
        document_config = view_config.get("document_encoder", view_config["query_encoder"])
        bundle = {
            "query": query_encoder,
            "document": (
                query_encoder
                if document_config == view_config["query_encoder"]
                else build_encoder(document_config)
            ),
        }
        self._verify_encoder_metadata(view, bundle)
        self._encoders[view] = bundle
        return bundle

    def _verify_encoder_metadata(self, view: str, bundle: dict[str, Any]) -> None:
        """Refuse to mix cache vectors with a differently loaded online encoder.

        A pinned model/config is not sufficient when ``device: auto`` changes the
        actually loaded dtype or implementation metadata on another host.  This
        check remains lazy: a fully cached run never loads a backbone merely to
        compare metadata, while the first true miss fails before producing a
        vector in an incompatible coordinate system.
        """

        if self.expected_view_metadata is None:
            return
        expected_view = self.expected_view_metadata[view]
        for role in ("query", "document"):
            expected = expected_view[f"{role}_encoder"]
            actual = bundle[role].metadata
            if canonical_json(actual) != canonical_json(expected):
                differing = sorted(
                    key
                    for key in set(expected) | set(actual)
                    if expected.get(key) != actual.get(key)
                )
                raise RuntimeError(
                    "Runtime frozen-encoder metadata differs from the encoder that built "
                    f"the immutable cache for view={view!r}, role={role!r}; "
                    f"differing_fields={differing}. Rebuild the cache on this host or use "
                    "runtime_embedding_policy=cache_only."
                )

    def encode(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        background_cards: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        background_cards = background_cards or []
        field_text = {
            "query": str(task["query"]),
            "goal": serialize_goal(task["goal"]),
            "state": serialize_state(state),
            "history": serialize_history(history),
            "sequence": serialize_sequence(state),
            "entities": serialize_entities(state),
        }
        output: dict[str, torch.Tensor] = {}
        for view, view_config in self.views.items():
            bundle = self._encoders_for_view(view)
            fields = [
                field
                for field in view_config.get("example_fields", [])
                if not field.startswith("next_")
            ]
            texts = [field_text[field] for field in fields]
            vectors = encode_nonempty(bundle["query"], texts, batch_size=self.batch_size)
            vectors = self._storage_roundtrip(vectors)
            nonzero = np.linalg.norm(vectors, axis=1) > 0
            pooled_parts = [vectors[index] for index in range(len(vectors)) if nonzero[index]]
            if view_config.get("encode_background", False) and background_cards:
                background = encode_nonempty(
                    bundle["document"],
                    [serialize_background(card) for card in background_cards],
                    batch_size=self.batch_size,
                )
                background = self._storage_roundtrip(background)
                if len(background):
                    pooled_parts.append(background.mean(axis=0))
            if pooled_parts:
                pooled = np.stack(pooled_parts).mean(axis=0)
            else:
                dim = int(bundle["query"].metadata["dim"])
                pooled = np.zeros(dim, dtype=np.float32)
            output[view] = torch.from_numpy(np.asarray(pooled, dtype=np.float32)).unsqueeze(0)
        structured = torch.from_numpy(self.structured.encode_state(state)).float().unsqueeze(0)
        return output, structured

    def encode_search(
        self,
        task: dict[str, Any],
        state: dict[str, Any],
        history: list[dict[str, Any]],
        background_cards: list[dict[str, Any]] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Encode the explicit state and goal endpoints used by value geometry.

        Prototype state views pool only query, current state, observed history, and
        declared background cards.  The goal is encoded separately, so the model
        cannot obtain a deceptively easy endpoint match by averaging the goal into
        both arguments.  State-only genomics views (for example a DNA sequence
        encoder) use their declared ``snapshot_field`` and receive a zero goal;
        this mirrors the search feature store and avoids pretending that tools or
        abstract goals are nucleotide sequences.
        """

        background_cards = background_cards or []
        field_text = {
            "query": str(task["query"]),
            "goal": serialize_goal(task["goal"]),
            "state": serialize_state(state),
            "history": serialize_history(history),
            "sequence": serialize_sequence(state),
            "entities": serialize_entities(state),
        }
        state_views: dict[str, torch.Tensor] = {}
        goal_views: dict[str, torch.Tensor] = {}
        for view, view_config in self.views.items():
            bundle = self._encoders_for_view(view)
            dim = int(bundle["query"].metadata["dim"])
            kind = str(view_config.get("kind", "prototype"))
            configured = [
                field
                for field in view_config.get("example_fields", [])
                if not field.startswith("next_")
            ]
            if kind == "state_only":
                snapshot_field = str(view_config.get("snapshot_field") or "state")
                if snapshot_field not in field_text:
                    raise ValueError(
                        f"Unsupported snapshot_field={snapshot_field!r} for runtime view {view!r}"
                    )
                state_fields = [snapshot_field]
            else:
                state_fields = [field for field in configured if field != "goal"]
            state_texts = [field_text[field] for field in state_fields]
            state_vectors = encode_nonempty(
                bundle["query"], state_texts, batch_size=self.batch_size
            )
            state_vectors = self._storage_roundtrip(state_vectors)
            pieces = [row for row in state_vectors if np.linalg.norm(row) > 0]
            if kind != "state_only" and view_config.get("encode_background", False):
                background = encode_nonempty(
                    bundle["document"],
                    [serialize_background(card) for card in background_cards],
                    batch_size=self.batch_size,
                )
                background = self._storage_roundtrip(background)
                if len(background):
                    pieces.append(background.mean(axis=0))
            pooled = (
                np.stack(pieces).mean(axis=0)
                if pieces
                else np.zeros(dim, dtype=np.float32)
            )
            state_views[view] = torch.from_numpy(
                np.asarray(pooled, dtype=np.float32)
            ).unsqueeze(0)

            if kind == "state_only" or "goal" not in configured:
                goal_vector = np.zeros(dim, dtype=np.float32)
            else:
                goal_vector = encode_nonempty(
                    bundle["query"], [field_text["goal"]], batch_size=self.batch_size
                )[0]
                goal_vector = self._storage_roundtrip(goal_vector)
            goal_views[view] = torch.from_numpy(
                np.asarray(goal_vector, dtype=np.float32)
            ).unsqueeze(0)

        structured_state = torch.from_numpy(self.structured.encode_state(state)).float().unsqueeze(0)
        structured_goal = (
            torch.from_numpy(self.structured.encode_state(task["goal"])).float().unsqueeze(0)
        )
        return state_views, goal_views, structured_state, structured_goal

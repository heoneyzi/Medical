from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from geoflowagent.utils.io import canonical_json


class StructuredHasher:
    """Small train-free vector for compatibility heads; never used as an exactness guarantee."""

    def __init__(self, dim: int = 64, salt: str = "geoflow-structured-v1") -> None:
        self.dim = int(dim)
        self.salt = salt

    def _add(self, vector: np.ndarray, feature: str) -> None:
        digest = hashlib.blake2b(f"{self.salt}:{feature}".encode(), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "little") % self.dim
        vector[index] += 1.0 if digest[8] & 1 else -1.0

    def encode_state(self, state: dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for key, value in _flatten(state):
            if key == "sequence_context":
                continue
            self._add(vector, f"state:{key}={canonical_json(value)}")
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode_tool(self, tool: dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for item in tool.get("preconditions", []):
            self._add(vector, f"pre:{canonical_json(item)}")
        for item in tool.get("effects", []):
            self._add(vector, f"effect:{canonical_json(item)}")
        for tag in tool.get("tags", []):
            self._add(vector, f"tag:{tag}")
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        output = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            output.extend(_flatten(value[key], child))
        return output
    if isinstance(value, list):
        return [(prefix, value)]
    return [(prefix, value)]

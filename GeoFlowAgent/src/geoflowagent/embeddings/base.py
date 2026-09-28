from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


class FrozenEncoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return [N, D] float32 embeddings without changing model weights."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Serializable model/tokenizer/pooling provenance."""


def encode_nonempty(
    encoder: FrozenEncoder,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode unique real text while representing a missing modality by an exact zero.

    Search-state datasets repeat task-level query and goal strings many times. A
    frozen eval-mode encoder is a pure function of its text, so exact-string
    deduplication preserves the feature definition while avoiding redundant
    backbone calls.
    """

    dim = int(encoder.metadata["dim"])
    output = np.zeros((len(texts), dim), dtype=np.float32)
    indices = [index for index, value in enumerate(texts) if value.strip()]
    if not indices:
        return output
    unique_texts: list[str] = []
    unique_by_text: dict[str, int] = {}
    inverse: list[int] = []
    for index in indices:
        text = texts[index]
        unique_index = unique_by_text.get(text)
        if unique_index is None:
            unique_index = len(unique_texts)
            unique_by_text[text] = unique_index
            unique_texts.append(text)
        inverse.append(unique_index)
    encoded = encoder.encode(unique_texts, batch_size=batch_size)
    if encoded.shape != (len(unique_texts), dim):
        raise ValueError(
            f"Encoder returned shape={encoded.shape}; expected {(len(unique_texts), dim)}"
        )
    output[indices] = encoded[np.asarray(inverse, dtype=np.int64)]
    return output

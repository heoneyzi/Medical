from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from geoflowagent.embeddings.base import FrozenEncoder, l2_normalize

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:><=+-]+|[^\s]", re.UNICODE)


class HashEncoder(FrozenEncoder):
    """Deterministic feature hashing for tests only, never for paper results."""

    def __init__(
        self,
        dim: int = 128,
        *,
        salt: str = "geoflow-smoke",
        min_ngram: int = 1,
        max_ngram: int = 2,
        normalize: bool = True,
    ) -> None:
        self.dim = int(dim)
        self.salt = salt
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.normalize = bool(normalize)
        if self.dim <= 0 or self.min_ngram <= 0 or self.max_ngram < self.min_ngram:
            raise ValueError("Invalid hash encoder dimensions or n-gram range")

    def _features(self, text: str) -> list[str]:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        output: list[str] = []
        for n in range(self.min_ngram, self.max_ngram + 1):
            output.extend(
                " ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
            )
        return output

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(f"{self.salt}:{feature}".encode(), digest_size=16).digest()
                column = int.from_bytes(digest[:8], "little") % self.dim
                sign = 1.0 if digest[8] & 1 else -1.0
                matrix[row_index, column] += sign
        return l2_normalize(matrix) if self.normalize else matrix

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "hash",
            "dim": self.dim,
            "salt": self.salt,
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "normalize": self.normalize,
            "research_use": "smoke_test_only",
        }

"""Deterministic seeds that are stable across Python processes."""
from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a uint32 seed without relying on Python's randomized ``hash``."""
    key = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)

"""Atomic, hash-addressed artifacts for long EXP3 runs.

Large activation tensors must never be expanded into the small orchestration
JSON written by :mod:`exp3.run`.  This module stores tensors/tables separately,
hashes the exact bytes, and returns a compact reference that can be sealed in
the EXP3 run contract.  Writes use a sibling temporary file followed by an
atomic rename, so a pre-empted GPU job cannot masquerade as a complete result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    kind: str
    sha256: str
    byte_size: int
    source_split: str
    source_role: str
    created_by: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactStore:
    """Small dependency-free artifact store rooted at one run directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, name: str, suffix: str) -> Path:
        if not name or name.startswith(".") or any(p in {"", ".", ".."} for p in Path(name).parts):
            raise ValueError(f"unsafe artifact name {name!r}")
        path = (self.root / name).with_suffix(suffix).resolve()
        if self.root not in path.parents:
            raise ValueError(f"artifact escapes root: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _commit(tmp: Path, path: Path) -> None:
        if path.exists():
            same = path.stat().st_size == tmp.stat().st_size \
                and _sha256(path) == _sha256(tmp)
            tmp.unlink()
            if not same:
                raise FileExistsError(
                    f"refusing to overwrite sealed artifact with new bytes: {path}")
            return
        tmp.replace(path)

    @staticmethod
    def _finish(path: Path, *, name: str, kind: str, source_split: str,
                source_role: str, created_by: str,
                metadata: Mapping[str, Any] | None = None) -> ArtifactRef:
        return ArtifactRef(
            name=name, path=str(path), kind=kind, sha256=_sha256(path),
            byte_size=path.stat().st_size, source_split=source_split,
            source_role=source_role, created_by=created_by,
            metadata=dict(metadata or {}),
        )

    def put_tensor(self, name: str, tensor: Tensor, *, source_split: str,
                   source_role: str, created_by: str,
                   metadata: Mapping[str, Any] | None = None) -> ArtifactRef:
        path = self._target(name, ".pt")
        tmp = path.with_suffix(path.suffix + ".tmp")
        value = tensor.detach().cpu().contiguous()
        torch.save(value, tmp)
        self._commit(tmp, path)
        meta = dict(metadata or {})
        meta.update(shape=list(value.shape), dtype=str(value.dtype))
        return self._finish(path, name=name, kind="tensor", source_split=source_split,
                            source_role=source_role, created_by=created_by, metadata=meta)

    def put_json(self, name: str, value: Mapping[str, Any], *, source_split: str,
                 source_role: str, created_by: str,
                 metadata: Mapping[str, Any] | None = None) -> ArtifactRef:
        path = self._target(name, ".json")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
        self._commit(tmp, path)
        return self._finish(path, name=name, kind="json", source_split=source_split,
                            source_role=source_role, created_by=created_by,
                            metadata=metadata)

    def put_jsonl(self, name: str, rows: Iterable[Mapping[str, Any]], *,
                  source_split: str, source_role: str, created_by: str,
                  metadata: Mapping[str, Any] | None = None) -> ArtifactRef:
        path = self._target(name, ".jsonl")
        tmp = path.with_suffix(path.suffix + ".tmp")
        n = 0
        with tmp.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
                n += 1
        self._commit(tmp, path)
        meta = dict(metadata or {})
        meta["n_rows"] = n
        return self._finish(path, name=name, kind="jsonl", source_split=source_split,
                            source_role=source_role, created_by=created_by, metadata=meta)

    @staticmethod
    def verify(ref: ArtifactRef) -> None:
        path = Path(ref.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != ref.byte_size or _sha256(path) != ref.sha256:
            raise RuntimeError(f"artifact changed after registration: {ref.name}")
        # Composite artifacts (notably TokenPanel) may bind additional files
        # from their hashed manifest.  Verify those bytes whenever the parent
        # reference is consumed so a swapped token tensor cannot survive on
        # the strength of an unchanged panel.json alone.
        bound_files = ref.metadata.get("bound_files", ())
        if not isinstance(bound_files, (list, tuple)):
            raise TypeError(f"artifact {ref.name!r} bound_files must be a sequence")
        for index, bound in enumerate(bound_files):
            if not isinstance(bound, Mapping):
                raise TypeError(
                    f"artifact {ref.name!r} bound file {index} is not a mapping")
            expected_keys = {"path", "sha256", "byte_size"}
            if set(bound) != expected_keys:
                raise ValueError(
                    f"artifact {ref.name!r} bound file {index} must contain "
                    f"exactly {sorted(expected_keys)}")
            child = Path(str(bound["path"])).expanduser().resolve()
            digest = str(bound["sha256"])
            size = bound["byte_size"]
            if (not child.is_file() or not isinstance(size, int) or size < 0
                    or child.stat().st_size != size or _sha256(child) != digest):
                raise RuntimeError(
                    f"bound artifact changed after registration: {ref.name}[{index}]")

    @staticmethod
    def load_tensor(ref: ArtifactRef, *, map_location: str | torch.device = "cpu") -> Tensor:
        ArtifactStore.verify(ref)
        if ref.kind != "tensor":
            raise TypeError(f"artifact {ref.name!r} is {ref.kind}, not tensor")
        try:
            value = torch.load(ref.path, map_location=map_location, weights_only=True)
        except TypeError:  # older torch used by some Evo 2 containers
            value = torch.load(ref.path, map_location=map_location)
        if not isinstance(value, Tensor):
            raise TypeError(f"tensor artifact {ref.name!r} decoded as {type(value).__name__}")
        return value

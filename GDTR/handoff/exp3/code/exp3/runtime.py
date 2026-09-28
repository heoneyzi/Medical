"""Independent, hash-sealed attestation of the model runtime used by EXP3."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class RuntimeManifest:
    provider_id: str
    checkpoint_sha256: str
    architecture_sha256: str
    exp2_code_sha256: str
    provider_source_sha256: str
    manifest_sha256: str = ""

    def compute_hash(self) -> str:
        value = asdict(self)
        value["manifest_sha256"] = ""
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("runtime provider_id is required")
        for name in (
            "checkpoint_sha256", "architecture_sha256", "exp2_code_sha256",
            "provider_source_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"runtime {name} must be a lowercase SHA-256")
        if self.manifest_sha256 != self.compute_hash():
            raise RuntimeError("runtime manifest changed after sealing")

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        temporary.replace(target)
        return target

    @staticmethod
    def load(path: str | Path) -> "RuntimeManifest":
        value = RuntimeManifest(**json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")))
        value.validate()
        return value


def create_runtime_manifest(
    *, provider_id: str, checkpoint_sha256: str,
    architecture_sha256: str, exp2_code_sha256: str,
    provider_source_sha256: str,
) -> RuntimeManifest:
    provisional = RuntimeManifest(
        provider_id=provider_id,
        checkpoint_sha256=checkpoint_sha256,
        architecture_sha256=architecture_sha256,
        exp2_code_sha256=exp2_code_sha256,
        provider_source_sha256=provider_source_sha256,
    )
    value = RuntimeManifest(
        **{**asdict(provisional), "manifest_sha256": provisional.compute_hash()})
    value.validate()
    return value


__all__ = ["RuntimeManifest", "create_runtime_manifest"]

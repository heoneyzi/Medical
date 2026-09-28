"""Hash-verified handoff from EXP2 into EXP3.

The handoff contains references, not copies of EXP2 result tables.  EXP3 may
consume a frozen basis, carrier axis, margins and architecture identity, but it
must not silently rediscover them on its locked evaluation data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import Tensor

from .foundation import FIXED_EXP2_FOUNDATION_CLAIMS


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenInput:
    name: str
    path: str
    sha256: str
    kind: str
    source_split: str
    source_role: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify(self) -> Path:
        if not self.name.strip() or not self.source_split.strip():
            raise ValueError("frozen input name and source_split are required")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError(f"{self.name}: sha256 must be a lowercase full SHA-256")
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != self.sha256:
            raise RuntimeError(f"frozen EXP2 input changed: {self.name}")
        if self.source_role not in {"discovery", "development"}:
            raise RuntimeError(
                f"EXP3 selection input {self.name!r} came from {self.source_role!r}; "
                "locked/external outcomes cannot select the mechanism")
        return path

    def load_tensor(self, *, map_location: str | torch.device = "cpu") -> Tensor:
        path = self.verify()
        if self.kind != "tensor":
            raise TypeError(f"{self.name!r} is {self.kind!r}, not a tensor")
        try:
            value = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            value = torch.load(path, map_location=map_location)
        if not isinstance(value, Tensor):
            raise TypeError(f"{self.name!r} did not decode to a tensor")
        return value


REQUIRED_ROLES = ("writer", "b29_candidate", "reencoder", "readout_gate")
EXPECTED_LAYER_ROLES = {
    "writer": 28,
    "b29_candidate": 29,
    "reencoder": 30,
    "readout_gate": 31,
}
REQUIRED_EXP2_FOUNDATIONS = FIXED_EXP2_FOUNDATION_CLAIMS
REQUIRED_SEMANTIC_INPUTS = {
    "u28": "b28_causal_subspace",
    "carrier_axis_x31": "x31_carrier_axis",
    "b30_subspace": "b30_mediation_subspace",
    "reference_directions": "late_stack_reference_directions",
}
REQUIRED_HANDOFF_MARGINS = frozenset({
    "effect", "equivalence", "rescue", "specificity",
})


@dataclass
class Exp2Handoff:
    """Minimal, auditable boundary between completed EXP2 and new EXP3."""

    handoff_id: str
    exp2_run_id: str
    checkpoint: str
    checkpoint_sha256: str
    architecture_sha256: str
    code_sha256: str
    layer_roles: dict[str, int]
    accepted_exp2_foundations: dict[str, bool]
    inputs: dict[str, FrozenInput]
    scientific_margins: dict[str, float]
    notes: str = ""
    handoff_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["handoff_sha256"] = ""
        return value

    def compute_hash(self) -> str:
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"),
                         default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate(self, *, verify_files: bool = True) -> None:
        if not self.handoff_id or not self.exp2_run_id or not self.checkpoint:
            raise ValueError("handoff_id, exp2_run_id and checkpoint are required")
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("architecture_sha256", self.architecture_sha256),
            ("code_sha256", self.code_sha256),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a lowercase full SHA-256")
        if set(self.layer_roles) != set(REQUIRED_ROLES):
            raise ValueError(
                "layer_roles must contain exactly "
                f"{sorted(REQUIRED_ROLES)}, received {sorted(self.layer_roles)}")
        if any(type(v) is not int for v in self.layer_roles.values()):
            raise TypeError("layer role indices must be literal integers")
        if len(set(self.layer_roles.values())) != len(self.layer_roles):
            raise ValueError("EXP2 role blocks must be distinct")
        if any(int(v) < 0 for v in self.layer_roles.values()):
            raise ValueError("layer role indices must be non-negative")
        observed_roles = {name: self.layer_roles[name] for name in REQUIRED_ROLES}
        if observed_roles != EXPECTED_LAYER_ROLES:
            raise RuntimeError(
                "this EXP3 program is defined for the confirmed 7B late-stack "
                f"roles {EXPECTED_LAYER_ROLES}, received {observed_roles}")
        missing_foundations = set(REQUIRED_EXP2_FOUNDATIONS) - set(
            self.accepted_exp2_foundations)
        if missing_foundations:
            raise ValueError(
                f"EXP2 foundation declaration missing {sorted(missing_foundations)}")
        rejected = sorted(
            name for name in REQUIRED_EXP2_FOUNDATIONS
            if self.accepted_exp2_foundations.get(name) is not True)
        if rejected:
            raise RuntimeError(
                "EXP3 is defined conditional on the confirmed EXP2 logic; "
                f"foundation(s) not accepted: {rejected}")
        missing_inputs = sorted(set(REQUIRED_SEMANTIC_INPUTS) - set(self.inputs))
        if missing_inputs:
            raise ValueError(
                f"EXP3 handoff lacks required semantic inputs: {missing_inputs}")
        for name, semantic_role in REQUIRED_SEMANTIC_INPUTS.items():
            value = self.inputs[name]
            if value.kind != "tensor":
                raise TypeError(f"{name} must be a tensor artifact")
            if value.metadata.get("semantic_role") != semantic_role:
                raise ValueError(
                    f"{name} must declare semantic_role={semantic_role!r}")
        if not self.scientific_margins or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v)) or float(v) < 0
                for v in self.scientific_margins.values()):
            raise ValueError("scientific margins must be finite non-negative numbers")
        missing_margins = sorted(
            REQUIRED_HANDOFF_MARGINS - set(self.scientific_margins))
        if missing_margins:
            raise ValueError(
                f"EXP2 handoff lacks required margin provenance: {missing_margins}")
        if verify_files:
            for value in self.inputs.values():
                value.verify()
            tensors = {
                name: self.inputs[name].load_tensor()
                for name in REQUIRED_SEMANTIC_INPUTS
            }
            if tensors["u28"].ndim != 2 or min(tensors["u28"].shape) == 0:
                raise ValueError("u28 must be a non-empty [width,rank] basis")
            if (tensors["b30_subspace"].ndim != 2
                    or min(tensors["b30_subspace"].shape) == 0):
                raise ValueError("b30_subspace must be a non-empty basis")
            carrier = tensors["carrier_axis_x31"]
            if carrier.ndim != 1 or not torch.isfinite(carrier).all() \
                    or float(carrier.double().norm()) == 0:
                raise ValueError("carrier_axis_x31 must be a finite non-zero vector")
            references = tensors["reference_directions"]
            if references.ndim != 2 or min(references.shape) == 0:
                raise ValueError("reference_directions must be a non-empty matrix")
        observed = self.compute_hash()
        if self.handoff_sha256 and self.handoff_sha256 != observed:
            raise RuntimeError("EXP2→EXP3 handoff changed after sealing")

    def require_inputs(self, *names: str) -> dict[str, FrozenInput]:
        """Resolve the exact sealed inputs needed by one EXP3 capability.

        The base handoff stays minimal; cube, reverse-trace, and learned-model
        providers call this method with their own preregistered requirements.
        """
        if not names or any(not isinstance(name, str) or not name.strip()
                            for name in names):
            raise ValueError("require_inputs needs one or more nonblank names")
        missing = sorted(set(names) - set(self.inputs))
        if missing:
            raise KeyError(f"EXP2 handoff lacks required frozen inputs: {missing}")
        selected = {name: self.inputs[name] for name in names}
        for value in selected.values():
            value.verify()
        return selected

    def seal(self) -> None:
        if self.handoff_sha256:
            raise RuntimeError("handoff is already sealed")
        self.validate(verify_files=True)
        self.handoff_sha256 = self.compute_hash()

    def save(self, path: str | Path) -> None:
        self.validate(verify_files=True)
        if not self.handoff_sha256:
            raise RuntimeError("seal the handoff before saving")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        temporary.replace(target)

    @staticmethod
    def load(path: str | Path, *, verify_files: bool = True) -> "Exp2Handoff":
        raw = json.loads(Path(path).expanduser().read_text())
        raw["inputs"] = {k: FrozenInput(**v) for k, v in raw["inputs"].items()}
        value = Exp2Handoff(**raw)
        value.validate(verify_files=verify_files)
        if not value.handoff_sha256:
            raise RuntimeError("EXP2→EXP3 handoff is not sealed")
        return value


def create_handoff(*, handoff_id: str, exp2_run_id: str, checkpoint: str,
                   checkpoint_sha256: str, architecture_sha256: str,
                   code_sha256: str, layer_roles: Mapping[str, int],
                   accepted_exp2_foundations: Mapping[str, bool],
                   inputs: Mapping[str, FrozenInput],
                   scientific_margins: Mapping[str, float],
                   notes: str = "") -> Exp2Handoff:
    bad_foundations = {
        str(k): v for k, v in accepted_exp2_foundations.items()
        if type(v) is not bool
    }
    if bad_foundations:
        raise TypeError(
            "accepted_exp2_foundations values must be literal booleans; "
            f"invalid keys: {sorted(bad_foundations)}")
    bad_roles = {
        str(k): v for k, v in layer_roles.items() if type(v) is not int
    }
    if bad_roles:
        raise TypeError(
            "layer_roles values must be literal integers; "
            f"invalid keys: {sorted(bad_roles)}")
    value = Exp2Handoff(
        handoff_id=handoff_id, exp2_run_id=exp2_run_id,
        checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
        architecture_sha256=architecture_sha256, code_sha256=code_sha256,
        layer_roles={str(k): v for k, v in layer_roles.items()},
        accepted_exp2_foundations={
            str(k): v for k, v in accepted_exp2_foundations.items()},
        inputs=dict(inputs),
        scientific_margins={str(k): float(v) for k, v in scientific_margins.items()},
        notes=notes,
    )
    value.seal()
    return value

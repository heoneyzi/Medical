"""Concrete token-panel format for real and rehearsal EXP3 runs.

The scientific modules accept tensors and remain independent of FASTA or
tokenizer libraries.  This file is the executable bridge: tokenize sequences
with the *currently loaded* Evo 2 tokenizer, save exact ids plus genomic
metadata, and later stream loci/pairs into every intervention panel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .artifacts import ArtifactRef
from .splits import GenomicUnit, assert_no_genomic_overlap


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _commit_immutable(temporary: Path, target: Path) -> None:
    """Atomically create a panel file without replacing different evidence."""
    if target.exists():
        same = (target.stat().st_size == temporary.stat().st_size
                and _sha256_file(target) == _sha256_file(temporary))
        temporary.unlink()
        if not same:
            raise FileExistsError(
                f"refusing to overwrite sealed token-panel artifact: {target}")
        return
    temporary.replace(target)


@dataclass(frozen=True)
class TokenUnit:
    unit_id: str
    chromosome: str
    start: int
    end: int
    target_position: int = -1
    true_token: int | None = None
    family: str = "unlabelled"
    dependency_keys: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def genomic(self) -> GenomicUnit:
        keys = self.dependency_keys or (f"locus:{self.chromosome}:{self.start}-{self.end}",)
        return GenomicUnit(self.unit_id, self.chromosome, self.start, self.end,
                           tuple(keys), self.family, self.metadata)


@dataclass(frozen=True)
class TokenPair:
    pair_id: str
    left_unit: str
    right_unit: str
    wrong_pair_unit: str | None = None
    family: str = "matched"
    dependency_keys: tuple[str, ...] = ()
    matching: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TokenPanelRef:
    """Portable, independently verifiable reference to a complete token panel."""

    name: str
    directory: str
    panel_json_sha256: str
    panel_metadata_sha256: str
    tokens_sha256: str
    tokenizer_fingerprint: str
    source_split: str
    source_role: str
    created_by: str
    unit_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)

    def verify(self) -> "TokenPanel":
        for label, digest in (
            ("panel_json_sha256", self.panel_json_sha256),
            ("panel_metadata_sha256", self.panel_metadata_sha256),
            ("tokens_sha256", self.tokens_sha256),
        ):
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"{label} is not a full lowercase SHA-256")
        if not _SHA256.fullmatch(self.tokenizer_fingerprint):
            raise ValueError("tokenizer_fingerprint is not a full lowercase SHA-256")
        if self.source_role not in {
                "discovery", "development", "locked", "external", "unspecified"}:
            raise ValueError(f"unsupported token-panel source role {self.source_role!r}")
        root = Path(self.directory).expanduser().resolve()
        panel = TokenPanel.load(
            root,
            expected_tokenizer_fingerprint=self.tokenizer_fingerprint,
            expected_panel_metadata_sha256=self.panel_metadata_sha256,
            expected_panel_json_sha256=self.panel_json_sha256,
        )
        observed = (
            panel.panel_name, panel.tokens_sha256, panel.source_split,
            panel.source_role, panel.created_by, tuple(sorted(panel.units)),
            tuple(pair.pair_id for pair in panel.pairs),
        )
        expected = (
            self.name, self.tokens_sha256, self.source_split,
            self.source_role, self.created_by, self.unit_ids, self.pair_ids,
        )
        if observed != expected:
            raise RuntimeError("token panel identity/provenance differs from its reference")
        return panel

    def artifact_ref(self) -> ArtifactRef:
        """Return the full ArtifactRef accepted by ScientificDecision."""
        panel = self.verify()
        if self.source_role == "unspecified":
            raise RuntimeError(
                "production token-panel artifacts require an explicit source_role")
        index = Path(self.directory).expanduser().resolve() / "panel.json"
        tokens = Path(self.directory).expanduser().resolve() / "tokens.pt"
        return ArtifactRef(
            name=f"token_panel/{self.name}", path=str(index), kind="token_panel",
            sha256=self.panel_json_sha256, byte_size=index.stat().st_size,
            source_split=self.source_split, source_role=self.source_role,
            created_by=self.created_by,
            metadata={
                "panel_metadata_sha256": self.panel_metadata_sha256,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "unit_ids": list(self.unit_ids), "pair_ids": list(self.pair_ids),
                "bound_files": [{
                    "path": str(tokens), "sha256": panel.tokens_sha256,
                    "byte_size": tokens.stat().st_size,
                }],
            },
        )


class TokenPanel:
    def __init__(self, units: Mapping[str, TokenUnit], ids: Mapping[str, Tensor],
                 pairs: Sequence[TokenPair] = (), *, tokenizer_fingerprint: str = "",
                 panel_name: str = "", source_split: str = "",
                 source_role: str = "", created_by: str = ""):
        self.units = dict(units)
        self.ids = {}
        for key, value in ids.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"token ids for {key!r} must be a torch.Tensor")
            if torch.is_floating_point(value) or torch.is_complex(value) or value.dtype == torch.bool:
                raise TypeError(f"token ids for {key!r} must have an integer dtype")
            self.ids[str(key)] = value.detach().to(torch.long).cpu().contiguous()
        self.pairs = tuple(pairs)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.panel_name = str(panel_name)
        self.source_split = str(source_split)
        self.source_role = str(source_role)
        self.created_by = str(created_by)
        self.panel_metadata_sha256 = ""
        self.panel_json_sha256 = ""
        self.tokens_sha256 = ""
        self.artifact_directory = ""
        self.validate()

    def validate(self) -> None:
        if set(self.units) != set(self.ids):
            raise ValueError("unit metadata and token tensors have different ids")
        for uid, unit in self.units.items():
            if uid != unit.unit_id:
                raise ValueError(f"unit key/id mismatch {uid!r}/{unit.unit_id!r}")
            unit.genomic()
            value = self.ids[uid]
            if value.ndim == 1:
                value = value.unsqueeze(0)
                self.ids[uid] = value
            if value.ndim != 2 or value.shape[0] != 1 or value.numel() == 0:
                raise ValueError(f"{uid}: input ids must be [1,length]")
            if torch.any(value < 0):
                raise ValueError(f"{uid}: token ids must be non-negative")
            if unit.true_token is not None and int(unit.true_token) < 0:
                raise ValueError(f"{uid}: true_token must be non-negative")
            pos = unit.target_position if unit.target_position >= 0 else value.shape[1] + unit.target_position
            if not 0 <= pos < value.shape[1]:
                raise IndexError(f"{uid}: target position outside token sequence")
        seen = set()
        for pair in self.pairs:
            if pair.pair_id in seen:
                raise ValueError(f"duplicate pair {pair.pair_id!r}")
            seen.add(pair.pair_id)
            if pair.left_unit == pair.right_unit:
                raise ValueError(f"pair {pair.pair_id!r} uses the same unit twice")
            if pair.wrong_pair_unit in {pair.left_unit, pair.right_unit}:
                raise ValueError(f"pair {pair.pair_id!r} has a non-independent wrong pair")
            for uid in (pair.left_unit, pair.right_unit, pair.wrong_pair_unit):
                if uid is not None and uid not in self.units:
                    raise KeyError(f"pair {pair.pair_id!r} references unknown unit {uid!r}")
            if self.ids[pair.left_unit].shape != self.ids[pair.right_unit].shape:
                raise ValueError(f"pair {pair.pair_id!r} has unequal token lengths")
            if (pair.wrong_pair_unit is not None and
                    self.ids[pair.left_unit].shape != self.ids[pair.wrong_pair_unit].shape):
                raise ValueError(f"pair {pair.pair_id!r} wrong-pair length differs")

    @staticmethod
    def _fingerprint_tokenizer(tokenizer: object) -> str:
        probes = {}
        for seq in ("A", "C", "G", "T", "ACGTN", "GT", "AG"):
            if callable(tokenizer):
                out = tokenizer(seq)
            elif hasattr(tokenizer, "tokenize"):
                out = tokenizer.tokenize(seq)
            else:
                raise TypeError("tokenizer must be callable or expose tokenize")
            if isinstance(out, Tensor):
                out = out.detach().cpu().reshape(-1).tolist()
            probes[seq] = [int(x) for x in out]
        return hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_sequences(cls, records: Iterable[tuple[TokenUnit, str]], *,
                       tokenizer: object, pairs: Sequence[TokenPair] = (),
                       panel_name: str = "", source_split: str = "",
                       source_role: str = "", created_by: str = "") -> "TokenPanel":
        units, ids = {}, {}
        for unit, sequence in records:
            if unit.unit_id in units:
                raise KeyError(f"duplicate unit {unit.unit_id!r}")
            if not sequence:
                raise ValueError(f"empty sequence for {unit.unit_id!r}")
            raw = tokenizer(sequence) if callable(tokenizer) else tokenizer.tokenize(sequence)
            tensor = raw if isinstance(raw, Tensor) else torch.tensor(raw, dtype=torch.long)
            units[unit.unit_id] = unit
            ids[unit.unit_id] = tensor.reshape(1, -1)
        return cls(units, ids, pairs,
                   tokenizer_fingerprint=cls._fingerprint_tokenizer(tokenizer),
                   panel_name=panel_name, source_split=source_split,
                   source_role=source_role, created_by=created_by)

    def assert_split_disjoint(self, other: "TokenPanel", *, buffer_bp: int = 0) -> None:
        assert_no_genomic_overlap([u.genomic() for u in self.units.values()],
                                  [u.genomic() for u in other.units.values()],
                                  buffer_bp=buffer_bp)
        shared = set(self.units) & set(other.units)
        if shared:
            raise RuntimeError(f"panels share unit ids: {sorted(shared)[:5]}")

    def loci(self, *, device: str | torch.device = "cpu") -> list[tuple]:
        return [(uid, self.ids[uid].to(device), unit.true_token,
                 {"keys": unit.dependency_keys or
                  (f"locus:{unit.chromosome}:{unit.start}-{unit.end}",),
                  "chromosome": unit.chromosome, "start": unit.start,
                  "end": unit.end, "family": unit.family, **unit.metadata})
                for uid, unit in sorted(self.units.items())]

    def matched_pairs(self, *, device: str | torch.device = "cpu") -> list[tuple]:
        out = []
        for p in self.pairs:
            keys = p.dependency_keys or (f"pair:{p.pair_id}",
                                         f"unit:{p.left_unit}", f"unit:{p.right_unit}")
            out.append((p.pair_id, self.ids[p.left_unit].to(device),
                        self.ids[p.right_unit].to(device),
                        {"keys": keys, "family": p.family,
                         "wrong_pair_unit": p.wrong_pair_unit, **p.matching}))
        return out

    def save(self, directory: str | Path, *, panel_name: str | None = None,
             source_split: str | None = None, source_role: str | None = None,
             created_by: str | None = None) -> tuple[Path, Path]:
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        tensors = root / "tokens.pt"
        index = root / "panel.json"
        tmp_t = tensors.with_suffix(".pt.tmp")
        tmp_i = index.with_suffix(".json.tmp")
        torch.save(self.ids, tmp_t)
        _commit_immutable(tmp_t, tensors)
        tensor_sha = _sha256_file(tensors)
        name = str(panel_name if panel_name is not None else
                   (self.panel_name or root.name))
        split = str(source_split if source_split is not None else
                    (self.source_split or ",".join(sorted(
                        {unit.chromosome for unit in self.units.values()}))))
        role = str(source_role if source_role is not None else
                   (self.source_role or "unspecified"))
        creator = str(created_by if created_by is not None else
                      (self.created_by or "exp3.data.TokenPanel.save"))
        if not name.strip() or not split.strip() or not creator.strip():
            raise ValueError("panel name, source split, and creator must be non-empty")
        payload = {
            "version": 2,
            "panel_name": name,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tokens_sha256": tensor_sha,
            "provenance": {
                "source_split": split,
                "source_role": role,
                "created_by": creator,
            },
            "units": {k: asdict(v) for k, v in sorted(self.units.items())},
            "pairs": [asdict(p) for p in self.pairs],
        }
        metadata_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        payload["panel_metadata_sha256"] = metadata_sha
        tmp_i.write_text(json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False),
            encoding="utf-8")
        _commit_immutable(tmp_i, index)
        self.panel_name = name
        self.source_split = split
        self.source_role = role
        self.created_by = creator
        self.panel_metadata_sha256 = metadata_sha
        self.panel_json_sha256 = _sha256_file(index)
        self.tokens_sha256 = tensor_sha
        self.artifact_directory = str(root)
        return index, tensors

    @classmethod
    def load(cls, directory: str | Path, *,
             expected_tokenizer_fingerprint: str | None = None,
             expected_panel_metadata_sha256: str | None = None,
             expected_panel_json_sha256: str | None = None) -> "TokenPanel":
        root = Path(directory).expanduser().resolve()
        index, tensors = root / "panel.json", root / "tokens.pt"
        raw = json.loads(index.read_text())
        if raw.get("version") != 2:
            raise RuntimeError("token panel is not metadata-sealed schema version 2")
        observed_metadata_sha = raw.get("panel_metadata_sha256", "")
        if not _SHA256.fullmatch(str(observed_metadata_sha)):
            raise RuntimeError("token panel has no valid metadata seal")
        metadata_payload = dict(raw)
        metadata_payload.pop("panel_metadata_sha256", None)
        computed_metadata_sha = hashlib.sha256(
            _canonical_json_bytes(metadata_payload)).hexdigest()
        if computed_metadata_sha != observed_metadata_sha:
            raise RuntimeError("panel.json metadata changed after panel creation")
        if (expected_panel_metadata_sha256 is not None and
                observed_metadata_sha != expected_panel_metadata_sha256):
            raise RuntimeError("token panel differs from the expected metadata seal")
        panel_json_sha = _sha256_file(index)
        if (expected_panel_json_sha256 is not None and
                panel_json_sha != expected_panel_json_sha256):
            raise RuntimeError("panel.json differs from the expected artifact digest")
        digest = _sha256_file(tensors)
        if digest != raw["tokens_sha256"]:
            raise RuntimeError("token tensor artifact changed after panel creation")
        if (expected_tokenizer_fingerprint is not None and
                raw["tokenizer_fingerprint"] != expected_tokenizer_fingerprint):
            raise RuntimeError("tokenizer fingerprint differs from the panel")
        try:
            ids = torch.load(tensors, map_location="cpu", weights_only=True)
        except TypeError:
            ids = torch.load(tensors, map_location="cpu")
        units = {}
        for key, value in raw["units"].items():
            value["dependency_keys"] = tuple(value.get("dependency_keys", ()))
            units[key] = TokenUnit(**value)
        pairs = []
        for value in raw.get("pairs", []):
            value["dependency_keys"] = tuple(value.get("dependency_keys", ()))
            pairs.append(TokenPair(**value))
        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError("token panel provenance is missing")
        panel = cls(
            units, ids, pairs,
            tokenizer_fingerprint=raw["tokenizer_fingerprint"],
            panel_name=str(raw.get("panel_name", "")),
            source_split=str(provenance.get("source_split", "")),
            source_role=str(provenance.get("source_role", "")),
            created_by=str(provenance.get("created_by", "")),
        )
        if not panel.panel_name or not panel.source_split or not panel.created_by:
            raise RuntimeError("token panel identity/provenance is incomplete")
        panel.panel_metadata_sha256 = str(observed_metadata_sha)
        panel.panel_json_sha256 = panel_json_sha
        panel.tokens_sha256 = digest
        panel.artifact_directory = str(root)
        return panel

    def reference(self) -> TokenPanelRef:
        """Return a sealed reference after :meth:`save` or :meth:`load`."""
        if not self.artifact_directory or not all(_SHA256.fullmatch(value) for value in (
                self.panel_metadata_sha256, self.panel_json_sha256,
                self.tokens_sha256)):
            raise RuntimeError("save or load the token panel before requesting a reference")
        return TokenPanelRef(
            name=self.panel_name, directory=self.artifact_directory,
            panel_json_sha256=self.panel_json_sha256,
            panel_metadata_sha256=self.panel_metadata_sha256,
            tokens_sha256=self.tokens_sha256,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            source_split=self.source_split, source_role=self.source_role,
            created_by=self.created_by, unit_ids=tuple(sorted(self.units)),
            pair_ids=tuple(pair.pair_id for pair in self.pairs),
        )


def load_token_panel_refs(
    directories: Sequence[str | Path],
    *,
    require_scientific_roles: bool = True,
) -> tuple[TokenPanelRef, ...]:
    """Load, verify and deduplicate panels before sealing a run contract.

    ``execute_program`` can put ``ref.as_dict()`` in ``run_contract.json`` and
    expose ``ref.artifact_ref()`` to providers.  The latter is also accepted
    directly by :class:`~exp3.adaptive.ScientificDecision`, which causes the
    same panel and its bound token bytes to be reverified on ledger save/load.
    """
    refs: list[TokenPanelRef] = []
    seen_names: set[str] = set()
    seen_directories: set[str] = set()
    for directory in directories:
        panel = TokenPanel.load(directory)
        ref = panel.reference()
        ref.verify()
        if require_scientific_roles and ref.source_role == "unspecified":
            raise RuntimeError(
                f"token panel {ref.name!r} needs an explicit scientific source role")
        if ref.name in seen_names:
            raise ValueError(f"duplicate token panel name {ref.name!r}")
        resolved = str(Path(ref.directory).resolve())
        if resolved in seen_directories:
            raise ValueError(f"duplicate token panel directory {resolved!r}")
        seen_names.add(ref.name)
        seen_directories.add(resolved)
        refs.append(ref)
    return tuple(refs)

"""Step 0 — registry, split bank, architecture manifest.

One rule governs this file: **no number used by an analysis is written here
by hand.** Everything is either (a) read off the loaded checkpoint, (b)
produced by Step 2 development calibration, or (c) a selection rule frozen
in discovery. `Margins.calibrated` gates every locked analysis.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional

BlockType = Literal["hcs", "hcm", "hcl", "attn"]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: str | Path) -> str:
    """Content hash a checkpoint/artifact file or a directory tree.

    Directory hashes include relative paths and file bytes in lexical order,
    so moving a sealed directory is harmless but renaming or changing any
    contained file is detected.  This is intentionally explicit: hashing a
    multi-GB checkpoint is done only when the caller asks for validation.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    if p.is_file():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    files = sorted(q for q in p.rglob("*") if q.is_file())
    if not files:
        raise ValueError(f"cannot hash empty checkpoint/artifact directory: {p}")
    for q in files:
        h.update(q.relative_to(p).as_posix().encode())
        h.update(b"\0")
        with q.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


@dataclass
class ArchitectureManifest:
    """Read from the LOADED config + runtime module classes + forward graph.

    The plan (Step 0) forbids inferring the 7B block map from the 1B pattern.
    `verify_against_runtime` is what actually fills this in; the expected
    values below are recorded only so a mismatch is loud.
    """
    checkpoint: str
    checkpoint_sha256: str
    vortex_commit: str
    n_layers: int
    d_model: int
    vocab_size: int
    attn_layer_idxs: tuple[int, ...]
    block_types: tuple[BlockType, ...]
    rms_form: Literal["norm_over_sqrtd_plus_eps", "sqrt_mean_sq_plus_eps"]
    rms_eps: float
    has_unembed_bias: bool
    mlp_is_bilinear: tuple[bool, ...]      # per block: g = W3[(W1 z) * (W2 z)]
    mlp_activation: tuple[str, ...]        # runtime activation name per block
    hcl_stage_names: dict[int, list[str]]  # runtime names, block -> ordered stages
    legacy_alias: dict[str, str] = field(default_factory=dict)
    dtype: str = "bfloat16"
    notes: str = ""

    # blocks the program is about; filled by verify_against_runtime
    core_blocks: tuple[int, ...] = ()
    homologous_pairs: tuple[tuple[int, int], ...] = ()
    # Hash of the architecture contract (not the weights); filled at creation
    # and verified whenever the manifest is loaded.
    architecture_sha256: str = ""
    config_sha256: str = ""
    operator_evidence: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Direct constructors used by tests/analysis remain compatible while
        # every manifest gets the correct legacy block-output convention.
        expected = {f"h{l}": f"x{l + 1}" for l in range(self.n_layers)}
        if not self.legacy_alias:
            self.legacy_alias = expected
        else:
            for key, value in expected.items():
                got = self.legacy_alias.get(key)
                if got is None:
                    self.legacy_alias[key] = value
                elif got != value:
                    raise ValueError(
                        f"legacy alias {key!r}={got!r} is off by one; block-output "
                        f"states require {key!r}={value!r}"
                    )
            extra_block_aliases = sorted(
                key for key in self.legacy_alias
                if key.startswith("h") and key[1:].isdigit() and key not in expected
            )
            if extra_block_aliases:
                raise ValueError(
                    f"legacy aliases name non-existent block outputs: {extra_block_aliases}"
                )
        self.validate_structure()
        digest = self.compute_architecture_sha256()
        if self.architecture_sha256 and self.architecture_sha256 != digest:
            raise RuntimeError(
                "architecture manifest hash mismatch; architecture fields were "
                "modified after the manifest was sealed"
            )
        self.architecture_sha256 = digest

    def validate_structure(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if len(self.block_types) != self.n_layers:
            raise ValueError(
                f"block_types has {len(self.block_types)} entries for "
                f"n_layers={self.n_layers}"
            )
        if len(self.mlp_is_bilinear) != self.n_layers:
            raise ValueError("mlp_is_bilinear must have one value per block")
        if len(self.mlp_activation) != self.n_layers:
            raise ValueError("mlp_activation must have one value per block")
        allowed = {"hcs", "hcm", "hcl", "attn"}
        bad = [(l, t) for l, t in enumerate(self.block_types) if t not in allowed]
        if bad:
            raise ValueError(f"invalid runtime operator labels: {bad}")
        expected_attn = tuple(l for l, t in enumerate(self.block_types) if t == "attn")
        if tuple(self.attn_layer_idxs) != expected_attn:
            raise ValueError(
                f"attn_layer_idxs={self.attn_layer_idxs} conflicts with runtime "
                f"block_types (expected {expected_attn})"
            )
        for l in self.hcl_stage_names:
            if not 0 <= int(l) < self.n_layers or self.block_types[int(l)] != "hcl":
                raise ValueError(f"HCL stages declared for non-HCL block {l}")
        for l in self.core_blocks:
            if not 0 <= l < self.n_layers:
                raise ValueError(f"core block {l} outside [0, {self.n_layers})")

    def _architecture_payload(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "attn_layer_idxs": list(self.attn_layer_idxs),
            "block_types": list(self.block_types),
            "rms_form": self.rms_form,
            "rms_eps": self.rms_eps,
            "has_unembed_bias": self.has_unembed_bias,
            "mlp_is_bilinear": list(self.mlp_is_bilinear),
            "mlp_activation": list(self.mlp_activation),
            "hcl_stage_names": {str(k): v for k, v in sorted(self.hcl_stage_names.items())},
            "legacy_alias": dict(sorted(self.legacy_alias.items())),
            "dtype": self.dtype,
            "core_blocks": list(self.core_blocks),
            "homologous_pairs": [list(p) for p in self.homologous_pairs],
            "config_sha256": self.config_sha256,
            "operator_evidence": {
                str(k): v for k, v in sorted(self.operator_evidence.items())
            },
        }

    def compute_architecture_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._architecture_payload()).encode()).hexdigest()

    def assert_integrity(self) -> None:
        self.validate_structure()
        got = self.compute_architecture_sha256()
        if not self.architecture_sha256 or got != self.architecture_sha256:
            raise RuntimeError(
                f"architecture hash mismatch: stored={self.architecture_sha256!r}, "
                f"observed={got!r}"
            )

    def validate_provenance(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        vortex_commit: str | None = None,
        require_pinned: bool = True,
    ) -> None:
        """Validate weight identity and loader commit before a locked run.

        Callers may skip expensive checkpoint hashing for discovery, but a
        locked analysis can set `require_pinned=True` to make blank provenance
        a hard error rather than silently trusting the checkpoint label.
        """
        self.assert_integrity()
        if require_pinned and not self.checkpoint_sha256:
            raise RuntimeError("checkpoint_sha256 is blank; locked architecture is unpinned")
        if require_pinned and not self.vortex_commit:
            raise RuntimeError("vortex_commit is blank; loader implementation is unpinned")
        if checkpoint_path is not None:
            observed = sha256_path(checkpoint_path)
            if not self.checkpoint_sha256:
                raise RuntimeError("cannot validate checkpoint: manifest has no SHA-256")
            if observed != self.checkpoint_sha256:
                raise RuntimeError(
                    f"checkpoint SHA-256 mismatch: expected {self.checkpoint_sha256}, "
                    f"observed {observed}"
                )
        if vortex_commit is not None:
            if not self.vortex_commit:
                raise RuntimeError("cannot validate loader commit: manifest commit is blank")
            if vortex_commit != self.vortex_commit:
                raise RuntimeError(
                    f"Vortex commit mismatch: expected {self.vortex_commit!r}, "
                    f"observed {vortex_commit!r}"
                )

    def assert_runtime_match(self, observed: "ArchitectureManifest") -> None:
        """Compare a newly read runtime graph to this frozen contract."""
        self.assert_integrity()
        observed.assert_integrity()
        if self.compute_architecture_sha256() != observed.compute_architecture_sha256():
            keys = sorted(
                k for k in self._architecture_payload()
                if self._architecture_payload()[k] != observed._architecture_payload()[k]
            )
            raise RuntimeError(f"loaded runtime architecture differs in fields: {keys}")

    def block_type(self, ell: int) -> BlockType:
        return self.block_types[ell]

    def assert_core_layout(self) -> None:
        """The plan's Step 0 expectation, checked -- not assumed.

        b28 HCS, b29 HCM, b30 HCL, b31 attention; and every one of those
        blocks additionally carries its own MLP update, so an operator label
        names the MIXER only.
        """
        want = {28: "hcs", 29: "hcm", 30: "hcl", 31: "attn"}
        bad = {
            l: (self.block_types[l] if l < self.n_layers else "<missing>")
            for l, t in want.items()
            if l >= self.n_layers or self.block_types[l] != t
        }
        if bad:
            raise AssertionError(
                f"core block layout differs from the plan's expectation: {bad}. "
                f"Do NOT proceed by editing this assertion -- re-derive "
                f"core_blocks/homologous_pairs from the runtime graph, then "
                f"re-derive the phase contrasts in step15."
            )

    def period_and_phase(self) -> tuple[Optional[int], dict[int, int]]:
        """Infer the attention period from attn_layer_idxs, for Step 15.

        Returns (period, {block: phase}) or (None, {}) if attention indices
        are not evenly spaced -- in which case the phase contrast is not
        defined and Step 15 must be re-specified rather than forced.
        """
        idxs = sorted(self.attn_layer_idxs)
        if len(idxs) < 2:
            return None, {}
        gaps = {idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)}
        if len(gaps) != 1:
            return None, {}
        period = gaps.pop()
        anchor = idxs[0]
        return period, {l: (l - anchor) % period for l in range(self.n_layers)}


# --------------------------------------------------------------------------
# splits  (§1.2 split bank -- a locked split is never reused)
# --------------------------------------------------------------------------


@dataclass
class Split:
    name: str
    role: Literal["discovery", "development", "locked"]
    wave: str                       # "A" | "B" | "C" | ...
    regions: list[tuple[str, int, int]]   # (chrom, start, end), half-open
    exclusion_rule: str
    checksum: str = ""
    opened: bool = False            # set True the first time it is read

    def compute_checksum(self) -> str:
        payload = json.dumps(
            {"name": self.name, "regions": self.regions, "excl": self.exclusion_rule},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class SplitBank:
    splits: dict[str, Split] = field(default_factory=dict)
    # Append-only hash chain.  `opened=True` alone cannot show when/by which
    # step a confirmatory split was consumed, and used to be lost because the
    # CLI never saved the manifest after opening it.
    open_ledger: list[dict[str, Any]] = field(default_factory=list)
    ledger_head_sha256: str = ""

    def add(self, s: Split) -> None:
        if s.name in self.splits:
            raise KeyError(f"split {s.name} already sealed")
        s.checksum = s.compute_checksum()
        self.splits[s.name] = s

    def _append_open_record(
        self, s: Split, *, for_wave: str, step: str = "", run_id: str = ""
    ) -> None:
        previous = self.ledger_head_sha256
        body: dict[str, Any] = {
            "sequence": len(self.open_ledger),
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": s.name,
            "role": s.role,
            "declared_wave": s.wave,
            "requested_wave": for_wave,
            "step": step,
            "run_id": run_id,
            "split_checksum": s.checksum,
            "previous_sha256": previous,
        }
        digest = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
        body["record_sha256"] = digest
        self.open_ledger.append(body)
        self.ledger_head_sha256 = digest

    def verify_ledger(self) -> None:
        previous = ""
        for i, record in enumerate(self.open_ledger):
            body = dict(record)
            stored = body.pop("record_sha256", "")
            if body.get("sequence") != i or body.get("previous_sha256") != previous:
                raise RuntimeError(f"split-open ledger chain is broken at record {i}")
            observed = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
            if not stored or stored != observed:
                raise RuntimeError(f"split-open ledger hash mismatch at record {i}")
            previous = stored
        if self.ledger_head_sha256 != previous:
            raise RuntimeError("split-open ledger head does not match the record chain")

    def open(
        self,
        name: str,
        *,
        for_wave: str,
        step: str = "",
        run_id: str = "",
    ) -> Split:
        self.verify_ledger()
        s = self.splits[name]
        observed_checksum = s.compute_checksum()
        if s.checksum and s.checksum != observed_checksum:
            raise RuntimeError(
                f"split {name!r} checksum mismatch: regions/exclusion rule changed "
                "after sealing"
            )
        if not s.checksum:  # backward-compatible upgrade of an old manifest
            s.checksum = observed_checksum
        prior = [r for r in self.open_ledger if r.get("split") == name]
        for record in prior:
            if record.get("role") != s.role or record.get("declared_wave") != s.wave:
                raise RuntimeError(
                    f"split {name!r} role/wave changed after its first opening"
                )
            if record.get("split_checksum") != s.checksum:
                raise RuntimeError(
                    f"split {name!r} definition differs from its open ledger"
                )
        if s.role == "locked":
            used_waves = {str(r.get("requested_wave")) for r in prior}
            if used_waves and used_waves != {for_wave}:
                raise RuntimeError(
                    f"locked split {name!r} was already opened for wave(s) "
                    f"{sorted(used_waves)}; it cannot be reused for {for_wave!r}"
                )
            if s.opened and s.wave != for_wave:
                raise RuntimeError(
                    f"locked split {name!r} was already opened for wave {s.wave!r}; "
                    f"§1.2 forbids reusing it as the confirmatory split for wave "
                    f"{for_wave!r}. Advance to the next unused split in the bank."
                )
            if s.wave != for_wave:
                raise RuntimeError(
                    f"locked split {name!r} is reserved for wave {s.wave!r}, not {for_wave!r}"
                )
        s.opened = True
        self._append_open_record(s, for_wave=for_wave, step=step, run_id=run_id)
        return s

    def assert_disjoint(self) -> None:
        seen: dict[str, list[tuple[int, int, str]]] = {}
        for s in self.splits.values():
            for (c, a, b) in s.regions:
                for (a2, b2, other) in seen.get(c, []):
                    if a < b2 and a2 < b:
                        raise AssertionError(
                            f"splits {s.name!r} and {other!r} overlap on {c}:{max(a,a2)}-{min(b,b2)}"
                        )
                seen.setdefault(c, []).append((a, b, s.name))


# --------------------------------------------------------------------------
# margins  (§1.5 -- three reference distributions, separate per-endpoint margins)
# --------------------------------------------------------------------------


@dataclass
class Margins:
    """Per-endpoint practical margins. Units differ, so they are NEVER
    combined into one number (§1.5).

    `calibrated` is False until Step 2 writes these from the development
    split. Locked analyses refuse to run otherwise.
    """
    calibrated: bool = False

    # R_num: numerical/implementation tolerance. NOT a scientific null.
    r_num_d_shape: Optional[float] = None
    r_num_log_beta: Optional[float] = None
    r_num_logit_l2: Optional[float] = None
    r_num_identity_residual: Optional[float] = None

    # practical margins
    delta_shape: Optional[float] = None
    delta_nll: Optional[float] = None
    delta_path: Optional[float] = None
    delta_acc: Optional[float] = None           # Step 6-3 accuracy saturation
    delta_spec: dict[str, float] = field(default_factory=dict)  # per control family

    # beta* identifiability / range (§1.4)
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    flat_logit_threshold: Optional[float] = None

    # Step 5 eligibility
    c_abs_max: Optional[float] = None
    phi_min: Optional[float] = None
    phi_max: Optional[float] = None
    content_theta_q25: Optional[float] = None
    content_theta_q50: Optional[float] = None
    content_theta_q75: Optional[float] = None

    # Step 6-3 dose interval for slope contrasts
    q_lo: Optional[float] = None
    q_hi: Optional[float] = None
    alpha_sat: Optional[float] = None

    # Step 2 power
    n_clusters_required: dict[str, int] = field(default_factory=dict)

    def require(self, *names: str) -> None:
        if not self.calibrated:
            raise RuntimeError(
                "margins are not calibrated: run Step 2 on the development "
                "split first. Locked analysis with hand-picked margins is "
                "exactly what §1.5 forbids."
            )
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise RuntimeError(f"margins missing (calibrate in Step 2): {missing}")

    def equivalence_or_ci_only(self, endpoint: str) -> Optional[float]:
        """Return the margin, or None meaning 'report CI upper bound only'.

        §1.5: if there is no defensible margin, do not declare equivalence.
        """
        return {"d_shape": self.delta_shape, "nll": self.delta_nll,
                "path": self.delta_path}.get(endpoint)


# --------------------------------------------------------------------------
# frozen selections  (§1.3 -- selection in discovery, evaluation in locked)
# --------------------------------------------------------------------------


@dataclass
class FrozenSelections:
    """Everything chosen in discovery, sealed before locked analysis."""
    carrier_coordinate: Optional[int] = None            # e.g. 3756
    axis_control_pool: list[int] = field(default_factory=list)
    bilinear_channels: dict[int, list[int]] = field(default_factory=dict)  # block -> channels
    bilinear_K_ladder: list[int] = field(default_factory=list)
    mediator_P_path: Optional[str] = None               # saved [d, k] orthonormal basis
    mediator_rank: Optional[int] = None
    hcl_stage_subspaces: dict[str, str] = field(default_factory=dict)      # stage -> path
    response_subspace_path: Optional[str] = None        # Step 6-4 held-out response
    donor_matching_version: Optional[str] = None
    sealed_at: Optional[str] = None
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    selection_sha256: str = ""
    hash_version: int = 1

    def _artifact_paths(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.mediator_P_path:
            out["mediator_P"] = self.mediator_P_path
        if self.response_subspace_path:
            out["response_subspace"] = self.response_subspace_path
        for stage, path in sorted(self.hcl_stage_subspaces.items()):
            out[f"hcl_stage:{stage}"] = path
        return out

    @staticmethod
    def _resolve_artifact(path: str, base_dir: str | Path | None) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        return p.resolve()

    def _refresh_artifact_hashes(self, base_dir: str | Path | None) -> None:
        hashes: dict[str, str] = {}
        for name, path in self._artifact_paths().items():
            hashes[name] = sha256_path(self._resolve_artifact(path, base_dir))
        self.artifact_sha256 = hashes

    def _selection_payload(self) -> dict[str, Any]:
        return {
            "carrier_coordinate": self.carrier_coordinate,
            "axis_control_pool": list(self.axis_control_pool),
            "bilinear_channels": {
                str(k): list(v) for k, v in sorted(self.bilinear_channels.items())
            },
            "bilinear_K_ladder": list(self.bilinear_K_ladder),
            "mediator_P_path": self.mediator_P_path,
            "mediator_rank": self.mediator_rank,
            "hcl_stage_subspaces": dict(sorted(self.hcl_stage_subspaces.items())),
            "response_subspace_path": self.response_subspace_path,
            "donor_matching_version": self.donor_matching_version,
            "sealed_at": self.sealed_at,
            "artifact_sha256": dict(sorted(self.artifact_sha256.items())),
            "hash_version": self.hash_version,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._selection_payload()).encode()).hexdigest()

    def seal(self, when: str, *, base_dir: str | Path | None = None) -> None:
        if self.sealed_at is not None:
            raise RuntimeError(f"selections already sealed at {self.sealed_at}")
        self.sealed_at = when
        self._refresh_artifact_hashes(base_dir)
        self.selection_sha256 = self.compute_hash()

    def assert_sealed(
        self,
        *,
        base_dir: str | Path | None = None,
        verify_files: bool = True,
    ) -> None:
        if self.sealed_at is None:
            raise RuntimeError(
                "selections are not sealed; §1.3 forbids choosing channels / "
                "mediators / donors on the same data used to evaluate them"
            )
        if verify_files:
            for name, path in self._artifact_paths().items():
                observed = sha256_path(self._resolve_artifact(path, base_dir))
                expected = self.artifact_sha256.get(name)
                if expected is None:
                    # Upgrade old manifests in memory without breaking an
                    # in-flight analysis; the next Manifest.save persists it.
                    self.artifact_sha256[name] = observed
                elif observed != expected:
                    raise RuntimeError(
                        f"frozen selection artifact {name!r} changed: "
                        f"expected {expected}, observed {observed}"
                    )
        observed_hash = self.compute_hash()
        if not self.selection_sha256:
            # Backward-compatible migration for already sealed manifests.
            self.selection_sha256 = observed_hash
        elif observed_hash != self.selection_sha256:
            raise RuntimeError(
                "frozen selections changed after sealing: "
                f"expected {self.selection_sha256}, observed {observed_hash}"
            )


@dataclass
class Manifest:
    arch: ArchitectureManifest
    splits: SplitBank
    margins: Margins
    selections: FrozenSelections
    run_id: str
    cache_root: str = "exp1/cache/raw_s4"
    out_root: str = "exp2/out"
    seed: int = 42
    _source_path: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.arch.assert_integrity()
        self.splits.verify_ledger()
        if self.selections.sealed_at is not None:
            self.selections.assert_sealed(base_dir=p.parent)
        payload = json.dumps(_encode(self), indent=2, sort_keys=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(p)
        self._source_path = str(p)

    def open_split(
        self,
        name: str,
        *,
        for_wave: str,
        step: str = "",
        persist: bool = True,
    ) -> Split:
        """Open a split, append the hash-chain ledger and persist immediately."""
        if persist and self._source_path is None:
            raise RuntimeError(
                "cannot persist split opening: save/load the manifest first, "
                "or call open_split(..., persist=False) and save it explicitly"
            )
        split = self.splits.open(
            name, for_wave=for_wave, step=step, run_id=self.run_id
        )
        if persist:
            self.save(self._source_path)
        return split

    @staticmethod
    def load(path: str | Path) -> "Manifest":
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text())
        split_raw = raw["splits"]
        bank = SplitBank(
            splits={k: Split(**v) for k, v in split_raw["splits"].items()},
            open_ledger=list(split_raw.get("open_ledger", [])),
            ledger_head_sha256=split_raw.get("ledger_head_sha256", ""),
        )
        selection_raw = dict(raw["selections"])
        if selection_raw.get("bilinear_channels"):
            selection_raw["bilinear_channels"] = {
                int(k): list(v) for k, v in selection_raw["bilinear_channels"].items()
            }
        man = Manifest(
            arch=ArchitectureManifest(**_tuplify(raw["arch"])),
            splits=bank,
            margins=Margins(**raw["margins"]),
            selections=FrozenSelections(**selection_raw),
            run_id=raw["run_id"],
            cache_root=raw["cache_root"],
            out_root=raw["out_root"],
            seed=raw["seed"],
        )
        man._source_path = str(source)
        man.splits.verify_ledger()
        return man


def _encode(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _encode(v) for k, v in asdict(obj).items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _tuplify(d: dict) -> dict:
    out = dict(d)
    for k in ("attn_layer_idxs", "block_types", "mlp_is_bilinear", "mlp_activation",
              "core_blocks"):
        if k in out and out[k] is not None:
            out[k] = tuple(out[k])
    if out.get("homologous_pairs"):
        out["homologous_pairs"] = tuple(tuple(p) for p in out["homologous_pairs"])
    if out.get("hcl_stage_names"):
        out["hcl_stage_names"] = {int(k): list(v) for k, v in out["hcl_stage_names"].items()}
    if out.get("operator_evidence"):
        out["operator_evidence"] = {int(k): str(v) for k, v in out["operator_evidence"].items()}
    # Migrate the known v1 off-by-one defect without changing import paths or
    # requiring current experiments to regenerate an otherwise valid file.
    if out.get("n_layers") is not None:
        n = int(out["n_layers"])
        expected = {f"h{l}": f"x{l + 1}" for l in range(n)}
        aliases = dict(out.get("legacy_alias") or {})
        old_style = aliases and all(
            aliases.get(f"h{l}") == f"x{l}" for l in range(n)
        )
        if old_style:
            aliases = expected
            # v1's architecture hash was computed under the wrong aliases (or
            # was blank); recompute it in ArchitectureManifest.__post_init__.
            out["architecture_sha256"] = ""
        out["legacy_alias"] = aliases or expected
    return out

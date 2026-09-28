"""Executable orchestration API and CLI for the EXP2 steps.

The original CLI only printed an instruction after consuming a split.  This
module now has one explicit contract: a step callback receives a frozen run
context, returns a JSON-serialisable mapping, and is executed before a result
file is written.  Existing package/module import paths are unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .manifest import Manifest, Split


STEP_NAMES = tuple(f"step{i}" for i in range(1, 11))


@dataclass(frozen=True)
class StepRunContext:
    step: str
    manifest: Manifest
    split: Split
    manifest_path: Path
    wave: str


StepCallback = Callable[[StepRunContext], Optional[Mapping[str, Any]]]
_STEP_CALLBACKS: dict[str, StepCallback] = {}


def register_step(
    step: str, callback: Optional[StepCallback] = None
) -> StepCallback | Callable[[StepCallback], StepCallback]:
    """Register a callback, directly or as ``@register_step("step3")``.

    Registration is process-local by design.  CLI invocations can instead use
    ``--callback package.module:function`` so model construction remains in
    the EXP1 environment rather than being guessed inside this package.
    """
    if step not in STEP_NAMES:
        raise KeyError(f"unknown step {step!r}; expected one of {STEP_NAMES}")

    def bind(fn: StepCallback) -> StepCallback:
        if step in _STEP_CALLBACKS and _STEP_CALLBACKS[step] is not fn:
            raise RuntimeError(f"a different callback is already registered for {step}")
        _STEP_CALLBACKS[step] = fn
        return fn

    return bind(callback) if callback is not None else bind


def unregister_step(step: str) -> None:
    """Testing/notebook helper; never called automatically during a run."""
    _STEP_CALLBACKS.pop(step, None)


def registered_steps() -> tuple[str, ...]:
    return tuple(sorted(_STEP_CALLBACKS))


def _load_callback(spec: str) -> StepCallback:
    if ":" not in spec:
        raise ValueError("callback must be written as package.module:function")
    module_name, attr = spec.rsplit(":", 1)
    if not module_name or not attr:
        raise ValueError("callback must be written as package.module:function")
    module = importlib.import_module(module_name)
    callback = getattr(module, attr)
    if not callable(callback):
        raise TypeError(f"callback {spec!r} is not callable")
    return callback


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _dump(man: Manifest, step: str, payload: dict) -> Path:
    p = Path(man.out_root) / man.run_id / f"{step}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str))
    return p


def execute_step(
    step: str,
    *,
    manifest_path: str | Path,
    split_name: str,
    wave: str,
    callback: StepCallback | None = None,
    checkpoint_path: str | Path | None = None,
    vortex_commit: str | None = None,
    require_pinned_architecture: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate gates, persist split opening, execute, and save a result.

    ``require_pinned_architecture=None`` means "required for locked splits".
    A callback exception propagates and no result JSON is emitted, while the
    already-persisted split-open ledger honestly records that the locked data
    was exposed.
    """
    if step not in STEP_NAMES:
        raise KeyError(f"unknown step {step!r}; expected one of {STEP_NAMES}")
    if len(wave) != 1 or wave not in "ABCDEFG":
        raise ValueError("wave must be one of A..G")

    manifest_path = Path(manifest_path).expanduser().resolve()
    man = Manifest.load(manifest_path)
    if split_name not in man.splits.splits:
        raise KeyError(f"unknown split {split_name!r}")
    split_decl = man.splits.splits[split_name]
    require_pinned = (
        split_decl.role == "locked"
        if require_pinned_architecture is None
        else require_pinned_architecture
    )
    man.arch.validate_provenance(
        checkpoint_path=checkpoint_path,
        vortex_commit=vortex_commit,
        require_pinned=require_pinned,
    )
    if step != "step1":
        man.margins.require("beta_min", "beta_max", "flat_logit_threshold")
    if step in ("step7", "step8", "step9", "step10"):
        man.selections.assert_sealed(base_dir=manifest_path.parent)

    selected = callback or _STEP_CALLBACKS.get(step)
    if selected is None:
        raise RuntimeError(
            f"no executable callback for {step}; call register_step({step!r}, fn) "
            "or pass --callback package.module:function"
        )

    # Opening is persisted immediately before the callback: a failed
    # confirmatory callback has still been given the locked split and must
    # remain in the audit trail.  Pure configuration errors above do not
    # consume it.
    split = man.open_split(split_name, for_wave=wave, step=step, persist=True)
    result = selected(
        StepRunContext(
            step=step,
            manifest=man,
            split=split,
            manifest_path=manifest_path,
            wave=wave,
        )
    )
    if result is None:
        result_dict: dict[str, Any] = {}
    elif isinstance(result, Mapping):
        result_dict = dict(result)
    elif is_dataclass(result):
        result_dict = asdict(result)
    else:
        raise TypeError(
            f"{step} callback returned {type(result).__name__}; expected a mapping, "
            "dataclass, or None"
        )

    payload: dict[str, Any] = {
        "run_id": man.run_id,
        "step": step,
        "wave": wave,
        "split": {
            "name": split.name,
            "role": split.role,
            "checksum": split.checksum,
        },
        "architecture_sha256": man.arch.architecture_sha256,
        "checkpoint_sha256": man.arch.checkpoint_sha256,
        "vortex_commit": man.arch.vortex_commit,
        "result": result_dict,
    }
    out = _dump(man, step, payload)
    # Persist callback-written calibrated margins or newly sealed discovery
    # selections only after a successful callback.
    man.save(manifest_path)
    return out, payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exp2")
    ap.add_argument("step", choices=STEP_NAMES)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--wave", required=True, choices=list("ABCDEFG"))
    ap.add_argument(
        "--callback",
        help="executable callback as package.module:function (or register via API)",
    )
    ap.add_argument("--checkpoint-path")
    ap.add_argument("--vortex-commit")
    ap.add_argument(
        "--allow-unpinned-architecture",
        action="store_true",
        help="discovery/debug only; locked runs are pinned by default",
    )
    args = ap.parse_args(argv)

    callback = _load_callback(args.callback) if args.callback else None
    out, payload = execute_step(
        args.step,
        manifest_path=args.manifest,
        split_name=args.split,
        wave=args.wave,
        callback=callback,
        checkpoint_path=args.checkpoint_path,
        vortex_commit=args.vortex_commit,
        require_pinned_architecture=(False if args.allow_unpinned_architecture else None),
    )
    split = payload["split"]
    print(
        f"[exp2] completed {args.step} on split {split['name']} "
        f"({split['role']}, wave {args.wave}); wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

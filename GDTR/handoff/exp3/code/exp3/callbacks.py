"""Provider bridge for executable EXP3 experiments.

The bridge deliberately dispatches *EXP3 experiment ids*, not EXP2 steps.
The user's existing Evo-2 loader can remain in its own module and expose one
of four small interfaces:

* a mapping from experiment id to callback;
* ``run_experiment(context)``;
* one method named after each experiment id; or
* a callable accepting :class:`~exp3.run.Exp3RunContext`.

Set ``EXP3_PROVIDER_FACTORY=package.module:factory`` and use
``exp3.callbacks:configured_callback`` from the CLI.  The factory is built
once per process and may accept zero arguments or the first run context.
"""
from __future__ import annotations

import importlib
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING

from .adaptive import ExperimentOutcome, RetryDirective

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from .run import Exp3RunContext


ExperimentResult = ExperimentOutcome | RetryDirective
ExperimentCallback = Callable[["Exp3RunContext"], ExperimentResult]


def _load(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("provider factory must be package.module:function")
    module_name, attribute = spec.rsplit(":", 1)
    if not module_name or not attribute:
        raise ValueError("provider factory must be package.module:function")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"provider factory {spec!r} is not callable")
    return value


def _call_factory(factory: Callable[..., Any], context: "Exp3RunContext") -> Any:
    try:
        signature = inspect.signature(factory)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        required = [object()]
    if not required:
        return factory()
    if len(required) == 1:
        return factory(context)
    raise TypeError("provider factory must accept zero arguments or Exp3RunContext")


def _validate_result(value: Any) -> ExperimentResult:
    if not isinstance(value, (ExperimentOutcome, RetryDirective)):
        raise TypeError(
            "EXP3 providers must return an explicit-validity ExperimentOutcome "
            f"or RetryDirective; received {type(value).__name__}"
        )
    return value


def dispatch(provider: Any, context: "Exp3RunContext") -> ExperimentResult:
    """Dispatch one EXP3 node without interpreting its scientific result."""

    experiment_id = context.experiment_id
    if isinstance(provider, Mapping):
        callback = provider.get(experiment_id)
        if callback is None:
            raise KeyError(f"provider mapping has no {experiment_id!r} callback")
        if not callable(callback):
            raise TypeError(f"provider entry {experiment_id!r} is not callable")
        result = callback(context)
    elif hasattr(provider, "run_experiment"):
        result = provider.run_experiment(context)
    elif hasattr(provider, experiment_id):
        result = getattr(provider, experiment_id)(context)
    elif callable(provider):
        result = provider(context)
    else:
        raise TypeError(
            "provider must be an experiment mapping, callback, run_experiment "
            "object, or expose methods named after EXP3 experiment ids"
        )
    return _validate_result(result)


_PROVIDER_CACHE: dict[tuple[str, str, str, str], Any] = {}


def configured_callback(context: "Exp3RunContext") -> ExperimentResult:
    """Environment-configured callback used by ``python -m exp3.run``."""

    spec = os.environ.get("EXP3_PROVIDER_FACTORY", "").strip()
    if not spec:
        raise RuntimeError(
            "EXP3_PROVIDER_FACTORY is unset. Point it to package.module:factory "
            "or pass a callback directly to execute_program()."
        )
    cache_key = (
        spec,
        context.handoff.handoff_sha256,
        context.config.config_sha256,
        str(context.run_dir),
    )
    if cache_key not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[cache_key] = _call_factory(_load(spec), context)
    return dispatch(_PROVIDER_CACHE[cache_key], context)


def make_callback(provider: Any) -> ExperimentCallback:
    """Bind an already loaded model/provider for notebook or API execution."""

    def callback(context: "Exp3RunContext") -> ExperimentResult:
        return dispatch(provider, context)

    def source_hash(value: Any) -> str:
        target = value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
        source = inspect.getsourcefile(target)
        if source and Path(source).is_file():
            return hashlib.sha256(Path(source).read_bytes()).hexdigest()
        code = getattr(value, "__code__", None)
        if code is None:
            raise ValueError("bound provider has no hashable implementation")
        return hashlib.sha256(code.co_code).hexdigest()

    explicit_provider_id = getattr(
        provider, "exp3_provider_id", getattr(provider, "provider_id", ""))
    if isinstance(provider, Mapping):
        contract = {
            "kind": "mapping",
            "experiment_ids": sorted(map(str, provider)),
            "callback_qualnames": {
                str(key): (
                    f"{getattr(value, '__module__', type(value).__module__)}:"
                    f"{getattr(value, '__qualname__', type(value).__qualname__)}"
                )
                for key, value in provider.items()
            },
            "callback_source_sha256": {
                str(key): source_hash(value) for key, value in provider.items()
            },
        }
    else:
        contract = {
            "kind": "provider",
            "provider_qualname": (
                f"{getattr(provider, '__module__', type(provider).__module__)}:"
                f"{getattr(provider, '__qualname__', type(provider).__qualname__)}"
            ),
            "provider_source_sha256": source_hash(provider),
        }
    if explicit_provider_id:
        contract["explicit_provider_id"] = str(explicit_provider_id)
    setattr(callback, "_exp3_provider_contract", contract)

    return callback


__all__ = [
    "ExperimentCallback", "ExperimentResult", "configured_callback", "dispatch",
    "make_callback",
]

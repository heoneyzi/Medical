from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from geoflowagent.embeddings.base import FrozenEncoder, l2_normalize
from geoflowagent.utils.reproducibility import choose_device


def _last_hidden_state(result: Any) -> torch.Tensor:
    """Accept modern ModelOutput and pinned legacy remote-model tuples."""

    hidden = getattr(result, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(result, (tuple, list)) and result and isinstance(result[0], torch.Tensor):
        return result[0]
    raise TypeError(f"Model output does not expose a last hidden-state tensor: {type(result)!r}")


class HuggingFaceEncoder(FrozenEncoder):
    """Frozen AutoModel adapter with explicit pooling and revision metadata."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        pooling: str = "mean",
        normalize: bool = True,
        max_length: int = 512,
        device: str = "auto",
        dtype: str = "float32",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = True,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        if revision in {None, "", "main"}:
            raise ValueError(
                f"Pin an immutable Hugging Face revision for {model_name!r}; mutable 'main' is refused."
            )
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)) is None:
            raise ValueError(
                f"revision for {model_name!r} must be a full 40-character commit SHA, "
                f"not {revision!r}"
            )
        if pooling not in {"mean", "cls", "last_token"}:
            raise ValueError("pooling must be mean, cls, or last_token")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install Hugging Face support with `pip install -e '.[hf]'`"
            ) from exc
        self.model_name = model_name
        self.requested_revision = revision
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = int(max_length)
        self.device = choose_device(device)
        self.requested_dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.config_overrides = dict(config_overrides or {})
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        if self.device.type == "cpu" and torch_dtype != torch.float32:
            torch_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        for key, value in self.config_overrides.items():
            if not hasattr(model_config, key):
                raise ValueError(f"Unknown config override for {model_name!r}: {key!r}")
            setattr(model_config, key, value)
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            config=model_config,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(self.device)
        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if (
            resolved_revision is not None
            and str(resolved_revision).lower() != str(revision).lower()
        ):
            raise RuntimeError(
                f"Resolved model revision {resolved_revision!r} differs from requested {revision!r}"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.tokenizer.pad_token_id is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                raise ValueError(f"Tokenizer for {model_name!r} has no usable padding token")
            self.tokenizer.pad_token = fallback
        try:
            self.loaded_dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        except StopIteration:
            self.loaded_dtype = "unknown"

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "last_token":
            token_positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
            positions = (token_positions * mask).argmax(dim=1)
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        output = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            result = self.model(**batch, return_dict=True)
            pooled = self._pool(_last_hidden_state(result), batch["attention_mask"])
            output.append(pooled.float().cpu().numpy())
        matrix = np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)
        return l2_normalize(matrix) if self.normalize else matrix

    @property
    def metadata(self) -> dict[str, Any]:
        resolved = getattr(self.model.config, "_commit_hash", None)
        dimension = getattr(self.model.config, "hidden_size", None)
        if dimension is None:
            dimension = getattr(self.model.config, "d_model", None)
        if dimension is None:
            raise AttributeError("Model config exposes neither hidden_size nor d_model")
        return {
            "backend": "huggingface",
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": resolved,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.model).__name__,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_length": self.max_length,
            "requested_dtype": self.requested_dtype,
            "loaded_dtype": self.loaded_dtype,
            "device": str(self.device),
            "padding_side": self.tokenizer.padding_side,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "config_overrides": self.config_overrides,
            "dim": int(dimension),
        }

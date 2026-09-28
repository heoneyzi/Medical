"""Model loading, block anatomy discovery, hooks, dtype ledger, weight shuffling.

Design note
-----------
Evo 2's internal module names differ between the released stacks, so nothing
here hard-codes a path.  Instead we *discover* the candidate mixer / MLP modules
inside each block and then **verify** the discovery arithmetically:

    x_in + mixer_out + mlp_out  ==  x_out     (to rounding)

That check is step 3 of the protocol and doubles as validation of the
discovery.  If it fails, the run stops and prints the module tree so the paths
can be pinned in ``configs/models.yaml`` -- it never proceeds on a guess,
because every downstream number depends on the branch decomposition being real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MIXER_ATTR_CANDIDATES = (
    "filter", "mixer", "inner_mha_cls", "attn", "attention",
    "hyena", "conv", "time_mixer", "seq_mixer",
)
MLP_ATTR_CANDIDATES = ("mlp", "ffn", "feed_forward", "channel_mixer")


def _first_attr(module, names: Sequence[str]):
    for n in names:
        if hasattr(module, n):
            child = getattr(module, n)
            if child is not None and hasattr(child, "forward"):
                return n, child
    return None, None


def _resolve(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        if part == "":
            continue
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _unwrap(out):
    """Modules may return (tensor, state) or dataclasses; take the hidden state."""
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


@dataclass
class BlockAnatomy:
    n_blocks: int
    mixer_attr: Optional[str]
    mlp_attr: Optional[str]
    blocks_path: str
    norm_path: str
    unembed_path: Optional[str]
    discovered: bool = True
    note: str = ""


@dataclass
class CaptureConfig:
    """What to keep from one forward pass."""

    block_outputs: bool = True
    block_inputs_for: Sequence[int] = ()
    branches_for: Sequence[int] = ()
    keep_device: bool = True     # set False to offload each tap to CPU (40B)


class Evo2Runner:
    """Thin wrapper that owns the model, the hooks and the tap buffers."""

    def __init__(self, spec, device: str = "cuda:0", shuffle_seed: Optional[int] = None):
        import torch

        self.torch = torch
        self.spec = spec
        self.device = device
        self.model_obj = self._load(spec, device)
        self.torch_model = self._inner_module()
        self.anatomy = self._discover()
        self.shuffle_seed = shuffle_seed
        self.shuffle_report: Optional[Dict] = None
        if shuffle_seed is not None:
            self.shuffle_report = shuffle_weights(self.torch_model, shuffle_seed)
        self._handles: List = []
        self.taps: Dict[str, "object"] = {}

    # ---------------------------------------------------------------- loading

    def _load(self, spec, device: str):
        # The rehearsal model goes through this same path on purpose: the point
        # of a dry run is to exercise discovery, hooks and extraction exactly as
        # the real model will, not a parallel code path that cannot break.
        if getattr(spec, "hf_name", "") == "mock":
            from .mockmodel import build_mock_model

            model = build_mock_model(n_blocks=spec.n_blocks, width=spec.width)
            model.to(device)
            model.eval()
            return model

        from evo2 import Evo2

        model = Evo2(spec.hf_name)
        try:
            model.model.to(device)
        except Exception:
            pass
        try:
            model.model.eval()
        except Exception:
            pass
        return model

    def _inner_module(self):
        m = self.model_obj
        for attr in ("model", "module", "backbone"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        return m

    # ------------------------------------------------------------- discovery

    def _discover(self) -> BlockAnatomy:
        spec = self.spec
        blocks = _resolve(self.torch_model, spec.block_module_path.replace("model.", "", 1)) \
            if spec.block_module_path.startswith("model.") else _resolve(self.torch_model, spec.block_module_path)
        n = len(blocks)

        mixer_attr = spec.mixer_attr
        mlp_attr = spec.mlp_attr
        if mixer_attr is None:
            mixer_attr, _ = _first_attr(blocks[0], MIXER_ATTR_CANDIDATES)
        if mlp_attr is None:
            mlp_attr, _ = _first_attr(blocks[0], MLP_ATTR_CANDIDATES)

        return BlockAnatomy(
            n_blocks=n,
            mixer_attr=mixer_attr,
            mlp_attr=mlp_attr,
            blocks_path=spec.block_module_path,
            norm_path=spec.norm_module_path,
            unembed_path=spec.unembed_module_path,
            discovered=(spec.mixer_attr is None or spec.mlp_attr is None),
        )

    @property
    def blocks(self):
        path = self.spec.block_module_path
        path = path[len("model."):] if path.startswith("model.") else path
        return _resolve(self.torch_model, path)

    @property
    def final_norm(self):
        path = self.spec.norm_module_path
        path = path[len("model."):] if path.startswith("model.") else path
        return _resolve(self.torch_model, path)

    def unembed_weight(self):
        """Return the [V, D] matrix the head reads with (tied embedding if needed)."""
        torch = self.torch
        path = self.spec.unembed_module_path
        if path:
            path = path[len("model."):] if path.startswith("model.") else path
            try:
                mod = _resolve(self.torch_model, path)
                w = getattr(mod, "weight", None)
                if w is not None:
                    return w.detach()
            except Exception:
                pass
        for name, param in self.torch_model.named_parameters():
            if name.endswith("embedding_weights") or "embed" in name and param.dim() == 2:
                return param.detach()
        raise RuntimeError("could not locate the unembedding / tied embedding weight")

    def module_tree(self, max_depth: int = 3) -> str:
        lines = []
        for name, mod in self.torch_model.named_modules():
            if name.count(".") <= max_depth:
                lines.append(f"{name or '<root>'}: {type(mod).__name__}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- hooks

    def attach(self, cfg: CaptureConfig):
        torch = self.torch
        self.detach()
        self.taps = {}

        def store(key):
            def _hook(_mod, _inp, out):
                t = _unwrap(out)
                t = t.detach()
                if not cfg.keep_device:
                    t = t.to("cpu")
                self.taps[key] = t
            return _hook

        def store_input(key):
            def _hook(_mod, inp):
                t = _unwrap(inp)
                t = t.detach()
                if not cfg.keep_device:
                    t = t.to("cpu")
                self.taps[key] = t
            return _hook

        blocks = self.blocks
        for i, blk in enumerate(blocks):
            if cfg.block_outputs:
                self._handles.append(blk.register_forward_hook(store(f"h{i}")))
            if i in cfg.block_inputs_for:
                self._handles.append(blk.register_forward_pre_hook(store_input(f"xin{i}")))
            if i in cfg.branches_for:
                if self.anatomy.mixer_attr:
                    mx = getattr(blk, self.anatomy.mixer_attr)
                    self._handles.append(mx.register_forward_hook(store(f"m{i}")))
                if self.anatomy.mlp_attr:
                    mp = getattr(blk, self.anatomy.mlp_attr)
                    self._handles.append(mp.register_forward_hook(store(f"g{i}")))

        self._handles.append(self.final_norm.register_forward_hook(store("hnorm")))
        return self

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    # --------------------------------------------------------------- forward

    def tokenize(self, seq: str):
        torch = self.torch
        ids = self.model_obj.tokenizer.tokenize(seq)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @property
    def acgt_ids(self) -> Dict[str, int]:
        """Verify the byte-level assumption instead of trusting it."""
        out = {}
        for base in "ACGT":
            ids = self.model_obj.tokenizer.tokenize(base)
            out[base] = int(ids[0])
        return out

    def forward(self, input_ids):
        torch = self.torch
        with torch.inference_mode():
            out = self.model_obj(input_ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(logits, "logits"):
            logits = logits.logits
        return logits

    # ---------------------------------------------------------------- ledger

    def dtype_ledger(self) -> Dict[str, object]:
        torch = self.torch
        params = {}
        for name, p in list(self.torch_model.named_parameters())[:20]:
            params[name] = str(p.dtype)
        taps = {k: str(v.dtype) for k, v in self.taps.items()}
        return dict(
            model=self.spec.key,
            param_dtypes_sample=params,
            tap_dtypes=taps,
            matmul_tf32=bool(getattr(torch.backends.cuda, "matmul", None)
                             and torch.backends.cuda.matmul.allow_tf32),
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            autocast_enabled=bool(torch.is_autocast_enabled()),
            torch_version=torch.__version__,
            anatomy=dict(mixer_attr=self.anatomy.mixer_attr,
                         mlp_attr=self.anatomy.mlp_attr,
                         n_blocks=self.anatomy.n_blocks,
                         discovered=self.anatomy.discovered),
            shuffle=self.shuffle_report,
        )


# --------------------------------------------------------------------------
# step 3: reconstruction fidelity
# --------------------------------------------------------------------------

def reconstruction_error(runner: "Evo2Runner", blocks: Sequence[int]) -> List[Dict[str, float]]:
    """Per-block ||x_out - (x_in + mixer + mlp)|| / ||x_out||, in float64.

    The handoff manuscript reports 2.3e-3 (bf16 rounding).  Anything materially
    larger means the discovered branch modules are not the ones being added to
    the residual stream, and the run must stop.
    """
    torch = runner.torch
    rows = []
    for i in blocks:
        need = [f"xin{i}", f"m{i}", f"g{i}", f"h{i}"]
        if not all(k in runner.taps for k in need):
            rows.append(dict(block=i, rel_err=float("nan"), status="missing_taps"))
            continue
        xin = runner.taps[f"xin{i}"].to(torch.float64)
        m = runner.taps[f"m{i}"].to(torch.float64)
        g = runner.taps[f"g{i}"].to(torch.float64)
        hout = runner.taps[f"h{i}"].to(torch.float64)
        recon = xin + m + g
        num = torch.linalg.vector_norm(hout - recon, dim=-1)
        den = torch.clamp(torch.linalg.vector_norm(hout, dim=-1), min=1e-300)
        rel = (num / den)
        rows.append(dict(
            block=int(i),
            rel_err_mean=float(rel.mean()),
            rel_err_max=float(rel.max()),
            status="ok",
        ))
    return rows


# --------------------------------------------------------------------------
# untrained control
# --------------------------------------------------------------------------

def shuffle_weights(module, seed: int) -> Dict[str, object]:
    """Shuffle entries *within* each parameter tensor.

    Architecture, parameter count and every tensor's marginal distribution are
    preserved; the learned structure is destroyed.  Matching the follow-up
    manuscript, we report the tensor and parameter counts so the control can be
    compared against its 339 tensors / 6.58e9 parameters.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    n_tensors = 0
    n_params = 0
    with torch.no_grad():
        for _, p in module.named_parameters():
            flat = p.detach().reshape(-1)
            perm = torch.randperm(flat.numel(), generator=g).to(flat.device)
            p.data.copy_(flat[perm].reshape(p.shape))
            n_tensors += 1
            n_params += flat.numel()
    return dict(seed=seed, n_tensors=n_tensors, n_params=int(n_params), mode="within_tensor")

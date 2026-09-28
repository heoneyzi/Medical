"""Backend shim so the numerical core runs identically under numpy and torch.

The extraction path runs on GPU with torch; the self-tests and any CPU
post-processing run with numpy.  Writing the formulas once and executing them
under both backends is deliberate: it is the only way the math that ships to the
server is the same math that is unit-tested here.

Every function in :mod:`exp1.metrics` takes arrays of shape ``[T, D]`` (positions
by width) and is written against this namespace only.
"""

from __future__ import annotations

from typing import Any

TINY = 1e-30


class _NumpyNS:
    name = "numpy"

    def __init__(self) -> None:
        import numpy as np

        self.np = np

    def sqrt(self, x):
        return self.np.sqrt(x)

    def clip_min(self, x, lo):
        return self.np.clip(x, lo, None)

    def sum(self, x, axis=-1):
        return self.np.sum(x, axis=axis)

    def max(self, x, axis=-1, keepdims=False):
        return self.np.max(x, axis=axis, keepdims=keepdims)

    def exp(self, x):
        return self.np.exp(x)

    def log(self, x):
        return self.np.log(x)

    def abs(self, x):
        return self.np.abs(x)

    def mean(self, x, axis=None, keepdims=False):
        return self.np.mean(x, axis=axis, keepdims=keepdims)

    def matmul(self, a, b):
        return a @ b

    def svd(self, x):
        u, s, vh = self.np.linalg.svd(x, full_matrices=False)
        return u, s, vh

    def to_float64(self, x):
        return x.astype(self.np.float64)

    def to_numpy(self, x):
        return self.np.asarray(x)

    def argmax(self, x, axis=-1):
        return self.np.argmax(x, axis=axis)

    def take_rows(self, x, idx):
        return x[idx]

    def zeros(self, shape, like=None):
        return self.np.zeros(shape, dtype=self.np.float64)


class _TorchNS:
    name = "torch"

    def __init__(self) -> None:
        import torch

        self.torch = torch

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def clip_min(self, x, lo):
        return self.torch.clamp(x, min=lo)

    def sum(self, x, axis=-1):
        return self.torch.sum(x, dim=axis)

    def max(self, x, axis=-1, keepdims=False):
        return self.torch.max(x, dim=axis, keepdim=keepdims).values

    def exp(self, x):
        return self.torch.exp(x)

    def log(self, x):
        return self.torch.log(x)

    def abs(self, x):
        return self.torch.abs(x)

    def mean(self, x, axis=None, keepdims=False):
        if axis is None:
            return self.torch.mean(x)
        return self.torch.mean(x, dim=axis, keepdim=keepdims)

    def matmul(self, a, b):
        return a @ b

    def svd(self, x):
        u, s, vh = self.torch.linalg.svd(x, full_matrices=False)
        return u, s, vh

    def to_float64(self, x):
        return x.to(self.torch.float64)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def argmax(self, x, axis=-1):
        return self.torch.argmax(x, dim=axis)

    def take_rows(self, x, idx):
        return x[idx]

    def zeros(self, shape, like=None):
        return self.torch.zeros(shape, dtype=self.torch.float64, device=like.device if like is not None else None)


def ns_for(x: Any):
    """Return the namespace matching the array type of ``x``."""
    mod = type(x).__module__
    if mod.startswith("torch"):
        return _TorchNS()
    return _NumpyNS()

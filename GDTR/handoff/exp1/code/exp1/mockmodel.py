"""A small real torch model shaped like Evo 2, for end-to-end rehearsal.

Why a model and not a mock runner
---------------------------------
The unit tests check the maths.  What they cannot check is the part that
actually breaks on a cluster at 2am: hook attachment finding the right
submodules, tap keys matching what the extractor asks for, dtypes surviving the
round trip, column names agreeing between extraction and every analysis step,
and empty or degenerate groups not raising three steps later.

So this is a genuine ``torch.nn.Module`` with Evo 2's structure -- a block list
whose blocks each add a mixer branch and a gated-MLP branch to a residual
stream, a final norm, a tied unembedding, and a byte tokenizer.  ``Evo2Runner``
wraps it through exactly the same path it uses for the real model, which means
the rehearsal exercises the real discovery, the real hooks and the real
extraction code.  Nothing in ``extract.py`` knows it is not Evo 2.

What is planted in it
---------------------
Numbers taken from the two manuscripts, so the rehearsal has a ground truth the
analysis is supposed to recover:

* a **context direction** built up smoothly over blocks 0..onset-1, whose
  strength depends on the input bases -- so a probe has something to find and
  the pre-handoff reference has something to approach;
* at the **onset** block, an MLP branch that writes a very large component along
  a single global direction ``u`` (the norm explosion: 40 -> several thousand);
* at the **rotation** block, a turn into the output frame, so that
  ``cos(h, h_norm)`` is near zero before it and clearly positive after;
* a **passthrough** final block, reproducing ``cos(h30) == cos(h31)``.

The planted values are deliberately not the manuscript's exact numbers.  A
rehearsal that reproduced them would only prove the rehearsal was written to,
and the point is to check the plumbing, not to simulate the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

MOCK_HF_NAME = "mock"


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

class MockTokenizer:
    """Byte-level, exactly like Evo 2: a base tokenises to its ASCII code."""

    def tokenize(self, seq: str) -> List[int]:
        return [ord(c) for c in seq]

    def detokenize(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------

def _make_block(width: int, kind: str, gen, kernel: int = 9):
    torch = _torch()
    nn = torch.nn

    class MockMixer(nn.Module):
        """Causal depthwise convolution + pointwise mix.

        Real position mixing, so a probe has local context to find rather than
        only the current base.  Causal, because Evo 2 is a next-token model and
        a rehearsal that leaked the future would make every probe curve a lie.
        """

        def __init__(self):
            super().__init__()
            self.dw = nn.Conv1d(width, width, kernel_size=kernel, groups=width,
                                bias=False)
            self.pw = nn.Linear(width, width, bias=False)
            self.pad = kernel - 1
            with torch.no_grad():
                self.dw.weight.mul_(0.3)
                self.pw.weight.mul_(0.05)

        def forward(self, x):
            y = x.transpose(1, 2)
            y = torch.nn.functional.pad(y, (self.pad, 0))
            y = self.dw(y).transpose(1, 2)
            return self.pw(y)

    class MockMLP(nn.Module):
        """Gated MLP.  Every planted term lives INSIDE this module.

        That is deliberate and load-bearing: the extraction hooks the branch
        modules, and the fidelity gate asserts ``x_in + mixer + mlp == h_out``.
        A planted term added in the block body instead would be invisible to the
        branch decomposition and would break the gate -- the rehearsal has to
        satisfy the same invariant the real model does.
        """

        def __init__(self):
            super().__init__()
            self.up = nn.Linear(width, 2 * width, bias=False)
            self.gate = nn.Linear(width, 2 * width, bias=False)
            self.down = nn.Linear(2 * width, width, bias=False)
            self.kind = kind
            self.gain = float(gen["gain"])
            self.register_buffer("u_dir", gen["u"].clone())
            self.register_buffer("ctx_dir", gen["ctx"].clone())
            self.register_buffer("rot_dir", gen["rot"].clone())
            with torch.no_grad():
                for m in (self.up, self.gate, self.down):
                    m.weight.mul_(0.05)

        def forward(self, x):
            base = self.down(torch.nn.functional.silu(self.up(x)) * self.gate(x))
            if self.kind == "build":
                # input-dependent growth of a context direction
                amp = x.mean(dim=-1, keepdim=True) * 0.5 + 0.1
                return base + self.gain * amp * self.ctx_dir
            if self.kind == "onset":
                # the norm explosion, along a carrier ORTHOGONAL to the output
                # frame -- which is what makes cos(h, h_norm) stay near zero
                # through the onset instead of jumping there
                return base + self.gain * self.u_dir
            if self.kind == "rotate":
                # remove the carrier, write the output-ready direction
                carrier = (x * self.u_dir).sum(dim=-1, keepdim=True) * self.u_dir
                return base - carrier + self.gain * self.rot_dir
            return base * 0.0 if self.kind == "passthrough" else base

    class MockBlock(nn.Module):
        """``h_out = x_in + mixer(x_in) + mlp(x_in)`` and nothing else."""

        def __init__(self):
            super().__init__()
            self.mixer = MockMixer()
            self.mlp = MockMLP()
            self.kind = kind

        def forward(self, x):
            if self.kind == "passthrough":
                # a residual passthrough, reproducing cos(h30) == cos(h31)
                return x + self.mixer(x) * 0.0 + self.mlp(x)
            return x + self.mixer(x) + self.mlp(x)

    return MockBlock()


def build_mock_model(n_blocks: int = 32, width: int = 64, vocab: int = 512,
                     onset: int = 28, rotation: int = 30, seed: int = 0):
    """A torch module with Evo 2's shape and a planted handoff."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    g = torch.Generator().manual_seed(seed)

    def unit(*orthogonal_to):
        v = torch.randn(width, generator=g)
        for o in orthogonal_to:
            v = v - (v @ o) * o
        return v / v.norm()

    u = unit()                 # the carrier written at the onset
    rot = unit(u)              # the output-ready direction
    ctx = unit(u, rot)         # what the pre-handoff stack builds

    class MockInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, width)
            blocks = []
            for i in range(n_blocks):
                if i == onset:
                    kind, gain = "onset", 900.0
                elif i == rotation:
                    kind, gain = "rotate", 400.0
                elif i == n_blocks - 1:
                    kind, gain = "passthrough", 0.0
                else:
                    kind, gain = "build", 0.35
                blocks.append(_make_block(width, kind,
                                          dict(u=u, ctx=ctx, rot=rot, gain=gain)))
            self.blocks = nn.ModuleList(blocks)
            self.norm = nn.LayerNorm(width, elementwise_affine=False)
            self.unembed = nn.Linear(width, vocab, bias=False)
            with torch.no_grad():
                # ACGT rows dominate the output frame, as in Evo 2
                self.unembed.weight.mul_(0.02)
                for base in "ACGT":
                    row = torch.randn(width, generator=g)
                    row = row - (row @ u) * u          # the head cannot read the carrier
                    row = row / row.norm()
                    self.unembed.weight[ord(base)] = row + 1.5 * rot

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for blk in self.blocks:
                x = blk(x)
            return self.unembed(self.norm(x))

    class MockEvo2(nn.Module):
        """The outer object, with ``.model`` and ``.tokenizer`` like evo2.Evo2."""

        def __init__(self):
            super().__init__()
            self.model = MockInner()
            self.tokenizer = MockTokenizer()

        def forward(self, input_ids):
            return self.model(input_ids)

    return MockEvo2()


def mock_spec(n_blocks: int = 32, width: int = 64, onset: int = 28,
              rotation: int = 30):
    """A :class:`~exp1.config.ModelSpec` pointing at the mock."""
    from .config import ModelSpec

    return ModelSpec(
        key="mock", hf_name=MOCK_HF_NAME, n_blocks=n_blocks, width=width,
        dtype="float32", block_module_path="model.blocks",
        norm_module_path="model.norm", unembed_module_path="model.unembed",
        mixer_attr="mixer", mlp_attr="mlp",
        notes=(f"rehearsal model: planted onset {onset}, rotation {rotation}, "
               "passthrough final block; not Evo 2 and not a simulation of it"),
    )

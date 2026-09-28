"""Small, deterministic Evo-2-shaped runtime for EXP3 integration testing.

The real experiments are intentionally checkpoint-specific.  This module is
not a biological surrogate and none of its numbers may be used as evidence.
It exists to make the *software path* executable on any CPU before an
expensive Evo 2 job is launched.

The toy graph has the exact public anatomy consumed by :class:`Evo2Runner`:

``x -> RMS_pre -> mixer -> residual -> RMS_post -> bilinear MLP -> residual``

There are exactly 32 blocks, and the late stack is deliberately planted as

``b28(HCS) preconditioner -> g28 writer -> g29 amplifier/editor``
``-> b30(HCL) re-encoder -> x31 carrier -> b31(attention) -> readout``.

The HCL block exposes the canonical internal stage modules, so component
hooks exercise the same contract as the production adapter.  All operations
are ordinary PyTorch tensor operations; no checkpoint or network is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .adapters.vortex import VortexAdapter
from .exp2_api import ArchitectureManifest, Evo2Runner


N_LAYERS = 32
DEFAULT_D_MODEL = 16
DEFAULT_VOCAB_SIZE = 12
DEFAULT_MAX_LENGTH = 64
WRITER_COORDINATES = (8, 9)
REENCODED_COORDINATES = (10, 11)
CARRIER_COORDINATE = 15


class ToyRMSNorm(nn.Module):
    """RMSNorm using the runtime form expected by the endpoint identities."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        denominator = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x / denominator


class ToyBilinearMLP(nn.Module):
    """Exact ``W3[(W1 z)*(W2 z)]`` MLP used by the accounting code."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.l1 = nn.Linear(width, hidden, bias=False)
        self.l2 = nn.Linear(width, hidden, bias=False)
        self.l3 = nn.Linear(hidden, width, bias=False)
        # VortexAdapter records ``identity`` for ``None`` and therefore marks
        # this expression as exactly bilinear rather than approximately gated.
        self.activation = None

    def forward(self, z: Tensor) -> Tensor:
        return self.l3(self.l1(z) * self.l2(z))


class _LinearMixer(nn.Module):
    """Base for lightweight HCS/HCM/attention-shaped operators."""

    operator_type: str

    def __init__(self, width: int, operator_type: str):
        super().__init__()
        self.operator_type = operator_type
        self.proj = nn.Linear(width, width, bias=False)

    def mix_context(self, z: Tensor) -> Tensor:
        return z

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(self.mix_context(z))


class ToyHCSMixer(_LinearMixer):
    def __init__(self, width: int):
        super().__init__(width, "hcs")

    def mix_context(self, z: Tensor) -> Tensor:
        # A one-token causal receptive field lets the planted b28 bilinear
        # writer express a genuine two-position motif grammar in the dry-run.
        if z.shape[-2] <= 1:
            return z
        previous = torch.cat((z[..., :1, :], z[..., :-1, :]), dim=-2)
        return 0.75 * z + 0.25 * previous


class ToyHCMMixer(_LinearMixer):
    def __init__(self, width: int):
        super().__init__(width, "hcm")

    def mix_context(self, z: Tensor) -> Tensor:
        if z.shape[-2] <= 1:
            return z
        previous = torch.cat((z[..., :1, :], z[..., :-1, :]), dim=-2)
        return 0.75 * z + 0.25 * previous


class ToyAttentionMixer(_LinearMixer):
    def __init__(self, width: int):
        super().__init__(width, "attn")

    def mix_context(self, z: Tensor) -> Tensor:
        length = z.shape[-2]
        denom = torch.arange(1, length + 1, dtype=z.dtype, device=z.device)
        shape = (1,) * (z.ndim - 2) + (length, 1)
        return z.cumsum(dim=-2) / denom.reshape(shape)


class _Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class _Ones(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)


class _Multiply(nn.Module):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return left * right


class _Add(nn.Module):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return left + right


class _Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.value.to(dtype=x.dtype, device=x.device)


class _CausalLag(nn.Module):
    """Tiny causal long path, sufficient to test lag/component hooks."""

    def __init__(self, current: float = 0.20, previous: float = 0.25):
        super().__init__()
        self.current = float(current)
        self.previous = float(previous)

    def forward(self, x: Tensor) -> Tensor:
        previous = torch.cat((torch.zeros_like(x[..., :1, :]), x[..., :-1, :]), dim=-2)
        return self.current * x + self.previous * previous


class ToyHCLMixer(nn.Module):
    """Canonical, hookable HCL dataflow with a planted b30 re-encoder."""

    operator_type = "hcl"

    def __init__(self, width: int):
        super().__init__()
        # Keep registration order equal to the mathematical dataflow.  The
        # adapter records these exact runtime names in the manifest.
        self.in_proj_x2 = _Ones()
        self.in_proj_x1 = _Identity()
        self.in_proj_v = _Ones()
        self.fir_x2 = _Identity()
        self.fir_x1 = _Identity()
        self.fir_v = _Identity()
        self.pregate_q = _Multiply()
        self.long_conv = _CausalLag()
        self.direct = _Scale(0.80)
        self.path_sum = _Add()
        self.postgate = _Multiply()
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        x2 = self.fir_x2(self.in_proj_x2(z))
        x1 = self.fir_x1(self.in_proj_x1(z))
        v = self.fir_v(self.in_proj_v(z))
        q = self.pregate_q(x1, v)
        long = self.long_conv(q)
        direct = self.direct(q)
        path = self.path_sum(long, direct)
        gated = self.postgate(x2, path)
        return self.out_proj(gated)


class ToyBlock(nn.Module):
    def __init__(self, width: int, hidden: int, kind: str):
        super().__init__()
        self.pre_norm = ToyRMSNorm(width)
        self.post_norm = ToyRMSNorm(width)
        if kind == "hcs":
            self.mixer = ToyHCSMixer(width)
        elif kind == "hcm":
            self.mixer = ToyHCMMixer(width)
        elif kind == "hcl":
            self.mixer = ToyHCLMixer(width)
        elif kind == "attn":
            self.mixer = ToyAttentionMixer(width)
        else:  # pragma: no cover - construction is internal and fixed
            raise ValueError(kind)
        self.mlp = ToyBilinearMLP(width, hidden)

    def forward(self, x: Tensor) -> Tensor:
        r = x + self.mixer(self.pre_norm(x))
        return r + self.mlp(self.post_norm(r))


def _block_types() -> tuple[str, ...]:
    # Early blocks provide sentinels/homologs.  The exact late layout is the
    # experimentally relevant part and is asserted after runtime discovery.
    early_cycle = ("hcs", "hcm", "hcl", "attn")
    kinds = [early_cycle[i % len(early_cycle)] for i in range(28)]
    kinds.extend(("hcs", "hcm", "hcl", "attn"))
    # Explicit homologs used by the research program.
    kinds[25], kinds[26], kinds[27] = "hcs", "hcm", "hcl"
    return tuple(kinds)


class ToyBackbone(nn.Module):
    def __init__(
        self,
        width: int = DEFAULT_D_MODEL,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        super().__init__()
        if width <= CARRIER_COORDINATE:
            raise ValueError(f"toy width must exceed carrier coordinate {CARRIER_COORDINATE}")
        self.width = int(width)
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.kinds = _block_types()
        self.embed = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(
            [ToyBlock(width, 2 * width, kind) for kind in self.kinds]
        )
        self.norm = ToyRMSNorm(width)
        self.unembed = nn.Linear(width, vocab_size, bias=False)
        self._plant_mechanism()

    @torch.no_grad()
    def _plant_mechanism(self) -> None:
        """Install a deterministic late-stack circuit with known ground truth."""

        # Every residual update starts at exact zero.  This leaves early
        # layers transparent and makes the late roles independently testable.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()

        # Token features: two content factors, a motif/precondition factor,
        # and one positive reference coordinate used as the bilinear gate.
        for token in range(self.vocab_size):
            phase = 2.0 * math.pi * token / self.vocab_size
            self.embed.weight[token].zero_()
            self.embed.weight[token, 0] = 0.90 * math.cos(phase)
            self.embed.weight[token, 1] = 0.85 * math.sin(phase)
            self.embed.weight[token, 2] = 0.55 if token % 3 == 0 else -0.35
            self.embed.weight[token, 3] = 1.25
            self.embed.weight[token, 4] = (token - (self.vocab_size - 1) / 2) / self.vocab_size

        self.position.weight.zero_()
        positions = torch.arange(self.max_length, dtype=self.position.weight.dtype)
        self.position.weight[:, 5] = 0.15 * torch.sin(positions / 3.0)
        self.position.weight[:, 6] = 0.10 * torch.cos(positions / 5.0)

        # A weaker homologous 25/26/27 triplet gives the scale-scan dry-run an
        # independently declared "small-model/control" map.  It is not used
        # to infer the 28/29/30 map; the two site lists are supplied
        # separately, exactly as the real 7B/1B protocol requires.
        b25 = self.blocks[25]
        assert isinstance(b25.mixer, ToyHCSMixer)
        b25.mixer.proj.weight[1, 2] = 0.18
        b25.mlp.l1.weight[0, 0] = 0.50
        b25.mlp.l2.weight[0, 1] = 0.45
        b25.mlp.l3.weight[WRITER_COORDINATES[0], 0] = 0.30
        b25.mlp.l1.weight[1, 2] = 0.40
        b25.mlp.l2.weight[1, 3] = 0.35
        b25.mlp.l3.weight[WRITER_COORDINATES[1], 1] = 0.25

        b26 = self.blocks[26]
        for channel, coord in enumerate(WRITER_COORDINATES):
            b26.mlp.l1.weight[channel, coord] = 0.45
            b26.mlp.l2.weight[channel, 3] = 0.40
            b26.mlp.l3.weight[coord, channel] = 0.30

        b27 = self.blocks[27]
        assert isinstance(b27.mixer, ToyHCLMixer)
        b27.mixer.out_proj.weight[REENCODED_COORDINATES[0], WRITER_COORDINATES[0]] = 0.12
        b27.mixer.out_proj.weight[REENCODED_COORDINATES[1], WRITER_COORDINATES[1]] = 0.12

        # m28 preconditions factor b by writing coordinate 2 into coordinate 1.
        b28 = self.blocks[28]
        assert isinstance(b28.mixer, ToyHCSMixer)
        b28.mixer.proj.weight[1, 2] = 0.80
        b28.mixer.proj.weight[0, 0] = 0.08

        # g28 is the first large content writer.  Two exact bilinear channels
        # write a two-dimensional causal subspace U28 = span(e8,e9).
        mlp28 = b28.mlp
        mlp28.l1.weight[0, 0] = 1.60
        mlp28.l2.weight[0, 1] = 1.40
        mlp28.l3.weight[WRITER_COORDINATES[0], 0] = 2.60
        mlp28.l1.weight[1, 2] = 1.35
        mlp28.l2.weight[1, 3] = 1.10
        mlp28.l3.weight[WRITER_COORDINATES[1], 1] = 2.20

        # g29 repeats the two writer directions through a positive reference
        # gate (amplifier), plus a deliberately small perpendicular editor.
        b29 = self.blocks[29]
        mlp29 = b29.mlp
        for channel, coord in enumerate(WRITER_COORDINATES):
            mlp29.l1.weight[channel, coord] = 1.25
            mlp29.l2.weight[channel, 3] = 1.00
            mlp29.l3.weight[coord, channel] = 1.15
        mlp29.l1.weight[2, 0] = 0.40
        mlp29.l2.weight[2, 2] = 0.40
        mlp29.l3.weight[12, 2] = 0.18

        # m30 converts U28 into a fresh content basis and co-writes the
        # denominator-only carrier.  g30 remains exactly zero by construction.
        b30 = self.blocks[30]
        assert isinstance(b30.mixer, ToyHCLMixer)
        W = b30.mixer.out_proj.weight
        W[REENCODED_COORDINATES[0], WRITER_COORDINATES[0]] = 1.35
        W[REENCODED_COORDINATES[1], WRITER_COORDINATES[1]] = 1.35
        W[WRITER_COORDINATES[0], WRITER_COORDINATES[0]] = -0.30
        W[WRITER_COORDINATES[1], WRITER_COORDINATES[1]] = -0.30
        W[CARRIER_COORDINATE, WRITER_COORDINATES[0]] = 1.70
        W[CARRIER_COORDINATE, WRITER_COORDINATES[1]] = -1.25

        # b31 is intentionally transparent in the planted ground truth; its
        # modules still execute, so freeze and suffix-gate hooks are tested.
        # The unembedding ignores the carrier coordinate.  Consequently x31's
        # carrier changes final RMS denominator/temperature but not content
        # logits directly, matching the scale-carrier distinction.
        for token in range(self.vocab_size):
            phase = 2.0 * math.pi * token / self.vocab_size
            self.unembed.weight[token, REENCODED_COORDINATES[0]] = 1.30 * math.cos(phase)
            self.unembed.weight[token, REENCODED_COORDINATES[1]] = 1.30 * math.sin(phase)
            self.unembed.weight[token, 0] = 0.15 * math.cos(2 * phase)
            self.unembed.weight[token, 1] = 0.15 * math.sin(2 * phase)
        self.unembed.weight[:, CARRIER_COORDINATE].zero_()

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("toy input_ids must have shape [batch, sequence]")
        if input_ids.shape[-1] > self.max_length:
            raise ValueError(
                f"sequence length {input_ids.shape[-1]} exceeds toy max_length={self.max_length}"
            )
        positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
        x = self.embed(input_ids) + self.position(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))


class ToyEvo2(nn.Module):
    """Runtime root matching the default paths in :class:`VortexAdapter`."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = ToyBackbone(**kwargs)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.backbone(input_ids)


@dataclass
class ToyRuntime:
    model: ToyEvo2
    adapter: VortexAdapter
    arch: ArchitectureManifest
    runner: Evo2Runner
    carrier_axis: Tensor
    writer_basis: Tensor
    carrier_coordinate: int = CARRIER_COORDINATE

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def ids(self, values: Sequence[int]) -> Tensor:
        return torch.tensor([list(values)], dtype=torch.long, device=self.device)


def toy_sequences(runtime: ToyRuntime | None = None, *, length: int = 12) -> dict[str, Tensor]:
    """Deterministic discovery/locked-like inputs with distinct target motifs."""

    if length < 6:
        raise ValueError("toy sequences require length >= 6")
    vocab = DEFAULT_VOCAB_SIZE if runtime is None else runtime.arch.vocab_size
    device = torch.device("cpu") if runtime is None else runtime.device
    base = torch.arange(length, dtype=torch.long, device=device) % vocab
    donor = (base * 5 + 3) % vocab
    motif_a = base.clone()
    motif_b = donor.clone()
    motif_a[-3:] = torch.tensor([0, 3, 6], dtype=torch.long, device=device) % vocab
    motif_b[-3:] = torch.tensor([2, 5, 8], dtype=torch.long, device=device) % vocab
    chr17 = (base * 7 + 1) % vocab
    return {
        "recipient": motif_a.unsqueeze(0),
        "donor": motif_b.unsqueeze(0),
        "wrong_pair": torch.flip(base, dims=(0,)).unsqueeze(0),
        "chr17": chr17.unsqueeze(0),
    }


def build_toy_runtime(
    *,
    seed: int = 7,
    width: int = DEFAULT_D_MODEL,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> ToyRuntime:
    """Construct, inspect and validate a checkpoint-free 32-block runtime."""

    # fork_rng prevents a dry-run from perturbing the caller's experiment RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ToyEvo2(width=width, vocab_size=vocab_size, max_length=max_length)
    model = model.to(device=device, dtype=dtype).eval()
    adapter = VortexAdapter(model)
    probe = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0) % vocab_size
    arch = adapter.read_architecture("toy-evo2-32", probe_ids=probe)
    arch.assert_core_layout()
    if arch.n_layers != N_LAYERS:
        raise RuntimeError(f"toy graph has {arch.n_layers} blocks, expected {N_LAYERS}")
    runner = Evo2Runner(adapter, arch)
    carrier = torch.zeros(width, dtype=dtype, device=device)
    carrier[CARRIER_COORDINATE] = 1.0
    writer = torch.zeros(width, len(WRITER_COORDINATES), dtype=dtype, device=device)
    for j, coordinate in enumerate(WRITER_COORDINATES):
        writer[coordinate, j] = 1.0
    return ToyRuntime(model, adapter, arch, runner, carrier, writer)


__all__ = [
    "CARRIER_COORDINATE",
    "DEFAULT_D_MODEL",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_VOCAB_SIZE",
    "N_LAYERS",
    "REENCODED_COORDINATES",
    "WRITER_COORDINATES",
    "ToyEvo2",
    "ToyRuntime",
    "build_toy_runtime",
    "toy_sequences",
]

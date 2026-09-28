from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


ModelFamily = Literal["prott5", "esm"]


@dataclass
class JepaPretrainConfig:
    # General
    debug_mode: bool = False
    seed: int = 42

    model_family: ModelFamily = "prott5"
    ckpt_name: str = "Rostlab/prot_t5_xl_uniref50"
    output_dir: Path = Path("models/jepa-protein")
    use_wandb: bool = False
    wandb_project: str = "cafa6"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Data
    fasta_paths: list[Path] = field(
        default_factory=lambda: [
            Path("data/Train/train_sequences.fasta"),
            Path("data/Test/testsuperset.fasta"),
        ]
    )
    max_sequences: Optional[int] = None
    min_residues: int = 32

    # Masking (span prediction)
    span_min: int = 8
    span_max: int = 128
    mask_char: str = "X"

    # Tokenization
    max_length: int = 1024
    batch_size: int = 1

    # Optimization
    num_train_steps: int = 4_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 10.0

    # JEPA bits
    predictor_hidden_mult: int = 2
    ema_decay: float = 0.999

    # Anti-collapse regularizers (optional, keep small defaults)
    var_reg: float = 0.0
    var_target: float = 1.0

    log_every: int = 50
    save_every: int = 500

    @classmethod
    def with_debug_mode(cls, **kwargs):
        kwargs["debug_mode"] = True
        kwargs.setdefault("max_sequences", 256)
        kwargs.setdefault("num_train_steps", 50)
        kwargs.setdefault("max_length", 256)
        kwargs.setdefault("span_max", 32)
        kwargs.setdefault("log_every", 1)
        kwargs.setdefault("save_every", 25)
        return cls(**kwargs)

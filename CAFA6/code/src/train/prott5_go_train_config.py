from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


# Namespaces:
# - Dataset uses: BPO/CCO/MFO (or aspects P/C/F in train_terms.tsv)
# - Task spec uses: BP/CC/MF
GoNamespace = Literal["BPO", "CCO", "MFO", "BP", "CC", "MF", "P", "C", "F"]


@dataclass(frozen=True)
class ResolvedTrainConfig:
    train_fasta: Path
    train_terms_tsv: Path
    output_dir: Path
    ckpt_name: str
    init_encoder_from: Path | None
    namespace: Optional[GoNamespace]
    min_label_count: int
    max_labels: int | None
    max_train_proteins: int | None
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    val_fraction: float
    seed: int
    do_eval: bool
    freeze_encoder: bool
    unfreeze_last_n_layers: int
    dropout: float
    logging_steps: int
    run_name: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    eval_ia_path: Path
    eval_obo_path: Path
    eval_thresholds: int
    eval_top_k: list[int]
    eval_min_score: float


@dataclass
class TrainConfig:
    debug_mode: bool = False

    seed: int = 42

    ckpt_name: str = field(
        default="Rostlab/prot_t5_xl_uniref50",
        metadata={"alias": ["--model-ckpt"]},
    )
    init_encoder_from: Path | None = field(
        default=None,
        metadata={
            "alias": ["--init-encoder-from"],
            "help": "Optional path to a JEPA checkpoint encoder.pt (or a directory containing encoder.pt).",
        },
    )
    train_fasta: Path = field(
        default=Path("data/Train/train_sequences.fasta"),
        metadata={"alias": ["--train-fasta"]},
    )
    train_terms_tsv: Path = field(
        default=Path("data/Train/train_terms.tsv"),
        metadata={"alias": ["--train-terms"]},
    )

    run_name: str = "prott5-go"
    output_dir: Path = field(default=Path("models/prott5-go"), metadata={"alias": ["--out"]})

    use_wandb: bool = True
    wandb_project: str = "cafa6"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    eval_ia_path: Path = Path("data/IA.tsv")
    eval_obo_path: Path = Path("data/Train/go-basic.obo")
    eval_thresholds: int = 101
    eval_top_k: list[int] = field(default_factory=lambda: [50, 200, 1500])
    eval_min_score: float = 0.001

    namespace: Optional[GoNamespace] = None

    min_label_count: int = 1
    max_labels: int | None = None
    max_train_proteins: int | None = None

    max_length: int = 1024

    train_batch_size: int = field(default=1, metadata={"alias": ["--batch-size"]})
    eval_batch_size: int = 1
    num_train_epochs: int = field(default=1, metadata={"alias": ["--epochs", "--num-epochs"]})
    learning_rate: float = field(default=2e-4, metadata={"alias": ["--lr"]})
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = field(default=1, metadata={"alias": ["--grad-accum"]})
    val_fraction: float = 0.01
    do_eval: bool = True
    logging_steps: int = 50

    freeze_encoder: bool = True
    unfreeze_last_n_layers: int = 0
    dropout: float = 0.1

    @classmethod
    def with_debug_mode(cls, **kwargs: Any) -> "TrainConfig":
        kwargs["debug_mode"] = True
        kwargs.setdefault("max_train_proteins", 256)
        kwargs.setdefault("max_labels", 128)
        kwargs.setdefault("min_label_count", 2)
        kwargs.setdefault("max_length", 256)
        kwargs.setdefault("logging_steps", 1)
        kwargs["run_name"] = "debug: " + kwargs.get("run_name", cls.run_name)
        return cls(**kwargs)

    def to_large_setting(self, multiplier: int = 4) -> "TrainConfig":
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
        self.eval_batch_size *= multiplier
        self.train_batch_size *= multiplier
        self.gradient_accumulation_steps = max(1, self.gradient_accumulation_steps // multiplier)
        return self

    def resolve(self) -> ResolvedTrainConfig:
        if self.debug_mode:
            if self.max_train_proteins is None:
                self.max_train_proteins = 256
            if self.max_labels is None:
                self.max_labels = 128

        return ResolvedTrainConfig(
            train_fasta=self.train_fasta,
            train_terms_tsv=self.train_terms_tsv,
            output_dir=self.output_dir,
            ckpt_name=self.ckpt_name,
            init_encoder_from=self.init_encoder_from,
            namespace=self.namespace,
            min_label_count=self.min_label_count,
            max_labels=self.max_labels,
            max_train_proteins=self.max_train_proteins,
            max_length=self.max_length,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            val_fraction=self.val_fraction,
            seed=self.seed,
            do_eval=self.do_eval,
            freeze_encoder=self.freeze_encoder,
            unfreeze_last_n_layers=self.unfreeze_last_n_layers,
            dropout=self.dropout,
            logging_steps=self.logging_steps,
            run_name=self.run_name,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_run_name=self.wandb_run_name,
            eval_ia_path=self.eval_ia_path,
            eval_obo_path=self.eval_obo_path,
            eval_thresholds=self.eval_thresholds,
            eval_top_k=list(self.eval_top_k),
            eval_min_score=self.eval_min_score,
        )

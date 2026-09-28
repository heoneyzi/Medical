from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from simple_parsing import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, EsmModel, T5EncoderModel

from ..common.fasta import read_fasta
from ..common.protein import prott5_tokenize_sequence, sanitize_sequence
from .jepa_protein_config import JepaPretrainConfig, ModelFamily


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wandb_config(cfg: JepaPretrainConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, list):
            data[key] = [str(v) if isinstance(v, Path) else v for v in value]
    return data


def _maybe_init_wandb(cfg: JepaPretrainConfig):
    if not cfg.use_wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb not installed; continue without logging", flush=True)
        return None
    run_name = cfg.wandb_run_name
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config=_wandb_config(cfg),
    )
    return wandb


def _tokenize_for_family(model_family: ModelFamily, sequence: str) -> str:
    seq = sanitize_sequence(sequence)
    if model_family == "prott5":
        return prott5_tokenize_sequence(seq)
    return seq


def _mask_span(seq: str, *, span_start: int, span_len: int, mask_char: str) -> str:
    seq = sanitize_sequence(seq)
    end = min(len(seq), span_start + span_len)
    if span_start >= end:
        return seq
    return seq[:span_start] + (mask_char * (end - span_start)) + seq[end:]


class ProteinUnlabeledDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        fasta_paths: list[Path],
        *,
        max_sequences: int | None,
        min_residues: int,
    ) -> None:
        sequences: list[str] = []
        for path in fasta_paths:
            for _, seq in tqdm(
                read_fasta(path),
                desc=f"Load {path.name}",
                unit="seq",
                leave=False,
            ):
                seq = sanitize_sequence(seq)
                if len(seq) < min_residues:
                    continue
                sequences.append(seq)
                if max_sequences is not None and len(sequences) >= max_sequences:
                    break
            if max_sequences is not None and len(sequences) >= max_sequences:
                break
        if not sequences:
            raise RuntimeError("No sequences loaded for JEPA pretraining.")
        self._sequences = sequences
        print(f"JEPA dataset sequences={len(sequences)}", flush=True)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"sequence": self._sequences[idx]}


def _build_encoder(model_family: ModelFamily, ckpt_name: str) -> nn.Module:
    if model_family == "prott5":
        return T5EncoderModel.from_pretrained(ckpt_name)
    if model_family == "esm":
        m = AutoModel.from_pretrained(ckpt_name)
        if isinstance(m, EsmModel):
            return m
        return m
    raise ValueError(f"Unknown model_family={model_family}")


def _get_hidden_size(encoder: nn.Module) -> int:
    cfg = getattr(encoder, "config", None)
    if cfg is None:
        raise ValueError("Encoder has no config; cannot infer hidden size.")
    if hasattr(cfg, "d_model"):
        return int(cfg.d_model)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise ValueError("Unsupported encoder config; cannot infer hidden size.")


def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters(), strict=True):
            p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


def _masked_residue_positions(
    *, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor
) -> list[list[int]]:
    # returns, per item, the token positions corresponding to residues
    out: list[list[int]] = []
    attn = attention_mask.bool()
    special = special_tokens_mask.bool()
    for b in range(attn.size(0)):
        positions = torch.where(attn[b] & (~special[b]))[0].tolist()
        out.append(positions)
    return out


def _masked_variance_loss(z: torch.Tensor, *, target_std: float) -> torch.Tensor:
    # z: [N, D]
    if z.numel() == 0:
        return torch.zeros((), device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    # penalize dimensions whose std is below target
    return torch.mean(torch.relu(target_std - std))


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(
    out_dir: Path,
    *,
    step: int,
    cfg: JepaPretrainConfig,
    encoder: nn.Module,
    predictor: nn.Module,
    tokenizer_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jepa_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "tokenizer_name.txt").write_text(tokenizer_name + "\n", encoding="utf-8")
    torch.save({"step": step, "state_dict": encoder.state_dict()}, out_dir / "encoder.pt")
    torch.save({"step": step, "state_dict": predictor.state_dict()}, out_dir / "predictor.pt")


def pretrain(cfg: JepaPretrainConfig) -> Path:
    _set_seed(cfg.seed)
    device = _pick_device()

    ds = ProteinUnlabeledDataset(
        [Path(p) for p in cfg.fasta_paths],
        max_sequences=cfg.max_sequences,
        min_residues=cfg.min_residues,
    )
    rng = random.Random(cfg.seed)

    if cfg.model_family == "prott5":
        try:
            import sentencepiece  # noqa: F401
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "ProtT5 tokenization requires `sentencepiece`. Install it in the project env "
                "(e.g. `./.venv/bin/python -m pip install sentencepiece`)."
            ) from e
        tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt_name)
    encoder = _build_encoder(cfg.model_family, cfg.ckpt_name)
    target_encoder = _build_encoder(cfg.model_family, cfg.ckpt_name)
    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    dim = _get_hidden_size(encoder)
    predictor = Predictor(dim, hidden_mult=cfg.predictor_hidden_mult)

    encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    mse = nn.MSELoss(reduction="mean")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        originals: list[str] = [b["sequence"] for b in batch]
        masked: list[str] = []
        spans: list[tuple[int, int]] = []
        for seq in originals:
            max_start = max(1, len(seq) - cfg.span_min)
            start = rng.randrange(0, max_start)
            span_len = rng.randrange(cfg.span_min, min(cfg.span_max, len(seq) - start) + 1)
            spans.append((start, span_len))
            masked.append(_mask_span(seq, span_start=start, span_len=span_len, mask_char=cfg.mask_char))

        tok_orig = tokenizer(
            [_tokenize_for_family(cfg.model_family, s) for s in originals],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_special_tokens_mask=True,
        )
        tok_ctx = tokenizer(
            [_tokenize_for_family(cfg.model_family, s) for s in masked],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_special_tokens_mask=True,
        )

        residue_positions = _masked_residue_positions(
            attention_mask=tok_orig["attention_mask"],
            special_tokens_mask=tok_orig["special_tokens_mask"],
        )

        # Create a token mask for the chosen residue span.
        span_token_mask = torch.zeros_like(tok_orig["attention_mask"], dtype=torch.bool)
        for b, (start, span_len) in enumerate(spans):
            pos = residue_positions[b]
            if not pos:
                continue
            start = min(start, len(pos) - 1)
            end = min(len(pos), start + span_len)
            chosen = pos[start:end]
            span_token_mask[b, chosen] = True

        return {
            "orig": tok_orig,
            "ctx": tok_ctx,
            "span_token_mask": span_token_mask,
        }

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    step = 0
    optimizer.zero_grad(set_to_none=True)

    print(
        f"JEPA start steps={cfg.num_train_steps} batch={cfg.batch_size} max_len={cfg.max_length} "
        f"device={device.type} model_family={cfg.model_family}",
        flush=True,
    )
    wandb = _maybe_init_wandb(cfg)
    if wandb is not None:
        wandb.log({"data/sequences": len(ds)}, step=0)
    progress = tqdm(total=cfg.num_train_steps, desc="JEPA", unit="step", leave=True)
    while step < cfg.num_train_steps:
        for batch in loader:
            step += 1
            orig = {k: v.to(device) for k, v in batch["orig"].items() if k != "special_tokens_mask"}
            ctx = {k: v.to(device) for k, v in batch["ctx"].items() if k != "special_tokens_mask"}
            span_mask = batch["span_token_mask"].to(device)

            encoder.train()
            predictor.train()

            with torch.no_grad():
                target_out = target_encoder(**orig)
                target_h = target_out.last_hidden_state

            ctx_out = encoder(**ctx)
            ctx_h = ctx_out.last_hidden_state
            pred_h = predictor(ctx_h)

            # Loss computed only on masked span tokens.
            idx = torch.where(span_mask)
            pred_sel = pred_h[idx]
            tgt_sel = target_h[idx]
            loss = mse(pred_sel, tgt_sel)

            if cfg.var_reg > 0:
                loss = loss + cfg.var_reg * _masked_variance_loss(pred_sel, target_std=cfg.var_target)

            (loss / cfg.gradient_accumulation_steps).backward()

            if step % cfg.gradient_accumulation_steps == 0:
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        max_norm=cfg.clip_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _ema_update(target_encoder, encoder, decay=cfg.ema_decay)

            if step % cfg.log_every == 0:
                progress.set_postfix(loss=f"{float(loss.item()):.6f}")
                print(
                    f"step={step}/{cfg.num_train_steps} loss={float(loss.item()):.6f} "
                    f"device={device.type} batch={cfg.batch_size} max_len={cfg.max_length}",
                    flush=True,
                )
                if wandb is not None:
                    wandb.log({"pretrain/loss": float(loss.item())}, step=step)
            progress.update(1)

            if step % cfg.save_every == 0 or step >= cfg.num_train_steps:
                save_checkpoint(
                    cfg.output_dir,
                    step=step,
                    cfg=cfg,
                    encoder=encoder,
                    predictor=predictor,
                    tokenizer_name=cfg.ckpt_name,
                )

            if step >= cfg.num_train_steps:
                break

    progress.close()
    save_checkpoint(
        cfg.output_dir,
        step=step,
        cfg=cfg,
        encoder=encoder,
        predictor=predictor,
        tokenizer_name=cfg.ckpt_name,
    )
    if wandb is not None:
        wandb.finish()
    return cfg.output_dir


def _parse_args(argv: list[str] | None = None) -> JepaPretrainConfig:
    parser = ArgumentParser(description="JEPA-style protein pretraining (ProtT5 or ESM).")
    parser.add_arguments(JepaPretrainConfig, dest="config")
    args = parser.parse_args(argv)
    return args.config


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    out = pretrain(cfg)
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

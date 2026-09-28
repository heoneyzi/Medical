from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from ..common.fasta import read_fasta
from ..common.go_obo import parse_go_parents, propagate_scores_to_parents
from ..common.protein import prott5_tokenize_sequence
from .prott5_go_train_config import GoNamespace, ResolvedTrainConfig, TrainConfig


__all__ = [
    "GoNamespace",
    "ResolvedTrainConfig",
    "TrainConfig",
    "ProtT5GoClassifier",
    "load_trained_model",
    "train",
]


class ProteinGoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        protein_ids: list[str],
        sequences_by_id: dict[str, str],
        label_indices_by_id: dict[str, list[int]],
    ) -> None:
        self._protein_ids = protein_ids
        self._sequences_by_id = sequences_by_id
        self._label_indices_by_id = label_indices_by_id

    def __len__(self) -> int:
        return len(self._protein_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        protein_id = self._protein_ids[idx]
        return {
            "protein_id": protein_id,
            "sequence": self._sequences_by_id[protein_id],
            "label_indices": self._label_indices_by_id[protein_id],
        }


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


def _wandb_config(cfg: ResolvedTrainConfig) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            data[k] = str(v)
        else:
            data[k] = v
    return data


def _maybe_init_wandb(cfg: ResolvedTrainConfig):
    if not cfg.use_wandb:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb not installed; continue without logging", flush=True)
        return None
    run_name = cfg.wandb_run_name or cfg.run_name
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config=_wandb_config(cfg),
    )
    return wandb


def load_terms(
    train_terms_tsv: Path, *, namespace: str | None
) -> dict[str, list[str]]:
    def normalize(ns: str) -> str:
        ns = ns.strip()
        if ns in {"P", "BPO", "BP"}:
            return "BPO"
        if ns in {"C", "CCO", "CC"}:
            return "CCO"
        if ns in {"F", "MFO", "MF"}:
            return "MFO"
        return ns

    namespace_norm = normalize(namespace) if namespace is not None else None
    terms_by_protein: dict[str, list[str]] = {}
    with train_terms_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "EntryID":
                continue
            protein_id, go_term, ns = row[0], row[1], row[2]
            ns_norm = normalize(ns)
            if namespace_norm is not None and ns_norm != namespace_norm:
                continue
            terms_by_protein.setdefault(protein_id, []).append(go_term)
    return terms_by_protein


def load_term_namespaces(train_terms_tsv: Path) -> dict[str, str]:
    def normalize(ns: str) -> str:
        ns = ns.strip()
        if ns in {"P", "BPO", "BP"}:
            return "BPO"
        if ns in {"C", "CCO", "CC"}:
            return "CCO"
        if ns in {"F", "MFO", "MF"}:
            return "MFO"
        return ns

    namespaces: dict[str, str] = {}
    with train_terms_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "EntryID":
                continue
            term = row[1]
            ns = normalize(row[2])
            namespaces[term] = ns
    return namespaces


def load_ia_weights(path: Path) -> dict[str, float]:
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            term, weight = line.split("\t", 1)
            try:
                weights[term] = float(weight)
            except ValueError:
                weights[term] = 0.0
    return weights


def _compute_fmax(
    pred_scores: torch.Tensor,
    true_labels: torch.Tensor,
    *,
    weights: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, float]:
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_t = float(thresholds[0].item()) if thresholds.numel() else 0.0

    true_w = (true_labels * weights).sum(dim=1)
    true_mask = true_w > 0
    for t in thresholds:
        pred_mask = pred_scores >= t
        pred_w = (pred_mask * weights).sum(dim=1)
        tp_w = ((pred_mask & true_labels.bool()) * weights).sum(dim=1)

        precision = torch.where(pred_w > 0, tp_w / pred_w, torch.zeros_like(tp_w))
        recall = torch.where(true_w > 0, tp_w / true_w, torch.zeros_like(tp_w))

        p = precision.mean().item()
        r = recall[true_mask].mean().item() if true_mask.any() else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_t = float(t.item())

    return {
        "fmax": best_f1,
        "precision": best_p,
        "recall": best_r,
        "threshold": best_t,
    }


def _compute_topk_metrics(
    pred_scores: torch.Tensor, true_labels: torch.Tensor, k: int
) -> dict[str, float]:
    num_labels = pred_scores.size(1)
    if num_labels == 0:
        return {"precision": 0.0, "recall": 0.0}
    k = min(k, num_labels)
    values, indices = torch.topk(pred_scores, k=k, dim=1, largest=True, sorted=True)
    true_at_k = true_labels.gather(1, indices)
    tp = true_at_k.sum(dim=1)
    precision = (tp / float(k)).mean().item()
    true_count = true_labels.sum(dim=1)
    recall = torch.where(true_count > 0, tp / true_count, torch.zeros_like(tp)).mean().item()
    return {"precision": precision, "recall": recall}


def _compute_task_metrics(
    pred_scores: torch.Tensor,
    true_labels: torch.Tensor,
    *,
    labels: list[str],
    term_namespaces: dict[str, str],
    ia_weights: dict[str, float],
    term_parents: dict[str, set[str]],
    cfg: ResolvedTrainConfig,
) -> dict[str, float]:
    true_labels = true_labels.bool()
    label2id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)
    weights = torch.tensor(
        [ia_weights.get(term, 0.0) for term in labels], dtype=torch.float32
    )

    # Propagate predictions to parents (max child score).
    pred_scores = pred_scores.clone()
    for row_idx in range(pred_scores.size(0)):
        row = pred_scores[row_idx]
        scores = {labels[j]: float(row[j].item()) for j in range(num_labels) if row[j].item() > 0}
        scores = propagate_scores_to_parents(scores, term_parents=term_parents)
        for term, score in scores.items():
            idx = label2id.get(term)
            if idx is None:
                continue
            if score > pred_scores[row_idx, idx]:
                pred_scores[row_idx, idx] = score

    thresholds = torch.linspace(0.0, 1.0, cfg.eval_thresholds)
    metrics: dict[str, float] = {}

    ns_keys = {"MFO": "mf", "BPO": "bp", "CCO": "cc"}
    indices_by_ns: dict[str, list[int]] = {k: [] for k in ns_keys}
    for idx, term in enumerate(labels):
        ns = term_namespaces.get(term)
        if ns in indices_by_ns:
            indices_by_ns[ns].append(idx)

    for ns, key in ns_keys.items():
        idxs = indices_by_ns.get(ns, [])
        if not idxs:
            continue
        pred_ns = pred_scores[:, idxs]
        true_ns = true_labels[:, idxs]
        weights_ns = weights[idxs]

        ia = _compute_fmax(pred_ns, true_ns, weights=weights_ns, thresholds=thresholds)
        uw = _compute_fmax(pred_ns, true_ns, weights=torch.ones_like(weights_ns), thresholds=thresholds)

        metrics[f"val/{key}/ia_fmax"] = ia["fmax"]
        metrics[f"val/{key}/ia_precision"] = ia["precision"]
        metrics[f"val/{key}/ia_recall"] = ia["recall"]
        metrics[f"val/{key}/ia_threshold"] = ia["threshold"]

        metrics[f"val/{key}/fmax"] = uw["fmax"]
        metrics[f"val/{key}/precision"] = uw["precision"]
        metrics[f"val/{key}/recall"] = uw["recall"]
        metrics[f"val/{key}/threshold"] = uw["threshold"]

        for k in cfg.eval_top_k:
            topk = _compute_topk_metrics(pred_ns, true_ns, k=k)
            metrics[f"val/{key}/topk_{k}_precision"] = topk["precision"]
            metrics[f"val/{key}/topk_{k}_recall"] = topk["recall"]

        pred_count = (pred_ns >= cfg.eval_min_score).sum(dim=1).float()
        metrics[f"val/{key}/coverage_mean_pred"] = pred_count.mean().item()
        metrics[f"val/{key}/coverage_frac_nonzero"] = (pred_count > 0).float().mean().item()
        metrics[f"val/{key}/true_mean"] = true_ns.sum(dim=1).float().mean().item()

    return metrics


def build_label_vocab(
    terms_by_protein: dict[str, list[str]],
    *,
    min_label_count: int,
    max_labels: int | None,
) -> list[str]:
    counts: dict[str, int] = {}
    for terms in terms_by_protein.values():
        for term in terms:
            counts[term] = counts.get(term, 0) + 1

    labels = [term for term, c in counts.items() if c >= min_label_count]
    labels.sort(key=lambda t: (-counts[t], t))
    if max_labels is not None:
        labels = labels[:max_labels]
    return labels


def build_targets(
    terms_by_protein: dict[str, list[str]],
    label2id: dict[str, int],
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for protein_id, terms in terms_by_protein.items():
        indices = sorted({label2id[t] for t in terms if t in label2id})
        if indices:
            out[protein_id] = indices
    return out


def load_sequences_for(
    fasta_path: Path, protein_ids: set[str]
) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for pid, seq in tqdm(
        read_fasta(fasta_path), desc="Load sequences", unit="seq", leave=False
    ):
        if pid in protein_ids:
            sequences[pid] = seq
    return sequences


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    mask = attention_mask.bool()
    if pad_token_id is not None:
        mask = mask & input_ids.ne(pad_token_id)
    if eos_token_id is not None:
        mask = mask & input_ids.ne(eos_token_id)

    lengths = mask.sum(dim=1).clamp(min=1)
    masked = last_hidden_state * mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.unsqueeze(-1)


class ProtT5GoClassifier(nn.Module):
    def __init__(
        self,
        model_ckpt: str,
        *,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_ckpt)
        hidden = int(self.encoder.config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(
            outputs.last_hidden_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return self.classifier(pooled)


def _configure_encoder_gradients(
    model: ProtT5GoClassifier, *, freeze: bool, unfreeze_last_n_layers: int
) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = not freeze

    if freeze:
        return

    if unfreeze_last_n_layers <= 0:
        return

    for p in model.encoder.parameters():
        p.requires_grad = False

    blocks = getattr(model.encoder, "block", None)
    if blocks is None:
        for p in model.encoder.parameters():
            p.requires_grad = True
        return

    to_unfreeze = list(blocks)[-unfreeze_last_n_layers:]
    for block in to_unfreeze:
        for p in block.parameters():
            p.requires_grad = True

    if hasattr(model.encoder, "final_layer_norm"):
        for p in model.encoder.final_layer_norm.parameters():
            p.requires_grad = True


def save_trained_model(
    output_dir: Path,
    *,
    model: ProtT5GoClassifier,
    ckpt_name: str,
    labels: list[str],
    train_config: ResolvedTrainConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                # Keep legacy key for compatibility with older runs.
                "model_ckpt": ckpt_name,
                "ckpt_name": ckpt_name,
                "num_labels": len(labels),
                "dropout": train_config.dropout,
                "namespace": train_config.namespace,
                "max_length": train_config.max_length,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config.__dict__, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    torch.save({"state_dict": model.state_dict()}, output_dir / "model.pt")


def load_trained_model(
    model_dir: str | Path, *, device: torch.device | None = None
) -> tuple[ProtT5GoClassifier, Any, list[str], dict[str, Any]]:
    model_dir = Path(model_dir)
    model_cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    labels: list[str] = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))

    ckpt_name = model_cfg.get("model_ckpt") or model_cfg["ckpt_name"]
    try:
        import sentencepiece  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ProtT5 tokenization requires `sentencepiece`. Install it in the project env "
            "(e.g. `./.venv/bin/python -m pip install sentencepiece`)."
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(ckpt_name, use_fast=False)
    model = ProtT5GoClassifier(
        ckpt_name,
        num_labels=len(labels),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])

    if device is None:
        device = _pick_device()
    model.to(device)
    model.eval()
    return model, tokenizer, labels, model_cfg


def train(cfg: TrainConfig | ResolvedTrainConfig) -> Path:
    cfg_resolved = cfg if isinstance(cfg, ResolvedTrainConfig) else cfg.resolve()

    if cfg_resolved.val_fraction < 0 or cfg_resolved.val_fraction >= 1:
        raise ValueError("val_fraction must be in [0, 1).")
    if cfg_resolved.train_batch_size < 1:
        raise ValueError("train_batch_size must be >= 1.")
    if cfg_resolved.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be >= 1.")
    if cfg_resolved.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    _set_seed(cfg_resolved.seed)
    device = _pick_device()

    terms_by_protein = load_terms(cfg_resolved.train_terms_tsv, namespace=cfg_resolved.namespace)
    labels = build_label_vocab(
        terms_by_protein,
        min_label_count=cfg_resolved.min_label_count,
        max_labels=cfg_resolved.max_labels,
    )
    label2id = {label: i for i, label in enumerate(labels)}
    label_indices_by_id = build_targets(terms_by_protein, label2id)

    protein_ids = sorted(label_indices_by_id.keys())
    sequences_by_id = load_sequences_for(cfg_resolved.train_fasta, set(protein_ids))
    protein_ids = [pid for pid in protein_ids if pid in sequences_by_id]

    if not protein_ids:
        raise RuntimeError("No training proteins found after filtering.")

    print(
        f"run={cfg_resolved.run_name} proteins={len(protein_ids)} labels={len(labels)}",
        flush=True,
    )

    rng = random.Random(cfg_resolved.seed)
    rng.shuffle(protein_ids)

    if cfg_resolved.max_train_proteins is not None:
        protein_ids = protein_ids[: cfg_resolved.max_train_proteins]

    val_fraction = cfg_resolved.val_fraction if cfg_resolved.do_eval else 0.0
    val_size = int(math.floor(len(protein_ids) * val_fraction))
    val_ids = protein_ids[:val_size]
    train_ids = protein_ids[val_size:]

    print(
        f"run={cfg_resolved.run_name} train_n={len(train_ids)} val_n={len(val_ids)}",
        flush=True,
    )

    wandb = _maybe_init_wandb(cfg_resolved)
    if wandb is not None:
        wandb.log(
            {
                "data/train_n": len(train_ids),
                "data/val_n": len(val_ids),
                "data/labels": len(labels),
            },
            step=0,
        )

    metrics_ready = False
    term_namespaces: dict[str, str] | None = None
    ia_weights: dict[str, float] | None = None
    term_parents: dict[str, set[str]] | None = None
    if wandb is not None and cfg_resolved.do_eval and len(val_ids) > 0:
        if cfg_resolved.eval_ia_path.exists() and cfg_resolved.eval_obo_path.exists():
            term_namespaces = load_term_namespaces(cfg_resolved.train_terms_tsv)
            ia_weights = load_ia_weights(cfg_resolved.eval_ia_path)
            term_parents = parse_go_parents(cfg_resolved.eval_obo_path)
            metrics_ready = True
        else:
            print(
                f"run={cfg_resolved.run_name} metrics skipped (missing eval files: "
                f"{cfg_resolved.eval_ia_path} or {cfg_resolved.eval_obo_path})",
                flush=True,
            )

    try:
        import sentencepiece  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ProtT5 tokenization requires `sentencepiece`. Install it in the project env "
            "(e.g. `./.venv/bin/python -m pip install sentencepiece`)."
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(cfg_resolved.ckpt_name, use_fast=False)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [prott5_tokenize_sequence(b["sequence"]) for b in batch]
        tokenized = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg_resolved.max_length,
        )
        y = torch.zeros((len(batch), len(labels)), dtype=torch.float32)
        for row, b in enumerate(batch):
            y[row, b["label_indices"]] = 1.0
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": y,
        }

    train_ds = ProteinGoDataset(train_ids, sequences_by_id, label_indices_by_id)
    val_ds = ProteinGoDataset(val_ids, sequences_by_id, label_indices_by_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_resolved.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_resolved.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = ProtT5GoClassifier(cfg_resolved.ckpt_name, num_labels=len(labels), dropout=cfg_resolved.dropout)
    if cfg_resolved.init_encoder_from is not None:
        init_path = Path(cfg_resolved.init_encoder_from)
        if init_path.is_dir():
            init_path = init_path / "encoder.pt"
        state = torch.load(init_path, map_location="cpu")
        encoder_state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"init_encoder_from={init_path} missing={len(missing)} unexpected={len(unexpected)}")
    _configure_encoder_gradients(
        model,
        freeze=cfg_resolved.freeze_encoder,
        unfreeze_last_n_layers=cfg_resolved.unfreeze_last_n_layers,
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg_resolved.learning_rate, weight_decay=cfg_resolved.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg_resolved.num_train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0

        print(f"run={cfg_resolved.run_name} epoch={epoch+1} start", flush=True)
        train_bar = tqdm(
            train_loader,
            desc=f"Train {epoch+1}/{cfg_resolved.num_train_epochs}",
            unit="batch",
            leave=True,
        )
        for step, batch in enumerate(train_bar):
            global_step = epoch * len(train_loader) + step + 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            loss = criterion(logits, labels_t) / cfg_resolved.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg_resolved.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            steps += 1

            if cfg_resolved.logging_steps > 0 and (step + 1) % cfg_resolved.logging_steps == 0:
                train_bar.set_postfix(loss=f"{running_loss/max(1,steps):.4f}")
                print(
                    f"run={cfg_resolved.run_name} epoch={epoch+1}/{cfg_resolved.num_train_epochs} "
                    f"step={step+1} loss={running_loss/max(1,steps):.4f}"
                    ,
                    flush=True,
                )
                if wandb is not None:
                    wandb.log(
                        {"train/loss": running_loss / max(1, steps)},
                        step=global_step,
                    )

        train_loss = running_loss / max(1, steps)

        val_loss = float("nan")
        if cfg_resolved.do_eval and len(val_ds) > 0:
            model.eval()
            total = 0.0
            n = 0
            val_preds: list[torch.Tensor] = []
            val_true: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc="Eval", unit="batch", leave=True):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_t = batch["labels"].to(device)
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
                    total += float(criterion(logits, labels_t).item())
                    n += 1
                    if metrics_ready:
                        val_preds.append(torch.sigmoid(logits).cpu())
                        val_true.append(labels_t.detach().cpu())
            val_loss = total / max(1, n)

        print(
            f"run={cfg_resolved.run_name} epoch={epoch+1}/{cfg_resolved.num_train_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"labels={len(labels)} train_n={len(train_ds)} val_n={len(val_ds)} device={device.type}",
            flush=True,
        )
        if wandb is not None:
            wandb.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                },
                step=(epoch + 1) * len(train_loader),
            )
            if metrics_ready and val_preds and val_true:
                pred_scores = torch.cat(val_preds, dim=0)
                true_labels = torch.cat(val_true, dim=0)
                task_metrics = _compute_task_metrics(
                    pred_scores,
                    true_labels,
                    labels=labels,
                    term_namespaces=term_namespaces or {},
                    ia_weights=ia_weights or {},
                    term_parents=term_parents or {},
                    cfg=cfg_resolved,
                )
                wandb.log(task_metrics, step=(epoch + 1) * len(train_loader))

    save_trained_model(
        cfg_resolved.output_dir,
        model=model,
        ckpt_name=cfg_resolved.ckpt_name,
        labels=labels,
        train_config=cfg_resolved,
    )
    if wandb is not None:
        wandb.finish()
    return cfg_resolved.output_dir


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = ArgumentParser(description="Fine-tune a ProtT5 GO-term classifier.")
    parser.add_arguments(TrainConfig, dest="config")
    args = parser.parse_args(argv)
    return args.config


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    out = train(cfg)
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

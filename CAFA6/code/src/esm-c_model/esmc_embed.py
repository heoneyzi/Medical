#!/usr/bin/env python3
"""
ESM-C embedding cache builder.

This module generates protein embeddings using ESM-C models and saves them
for downstream training tasks.

Supported pooling modes:
  - mean: Mean pooling over sequence length [D]
  - meanmax: Concatenation of mean and max pooling [2D]
  - cls: CLS token embedding [D]

Example usage:
    python -m src.esm_c_model.esmc_embed \\
        --base-path /path/to/cafa6 \\
        --model esmc_300m \\
        --modes mean,meanmax \\
        --run train \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.common.fasta import read_fasta

# ESM-C imports (SDK style)
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    ESMC_AVAILABLE = True
except ImportError:
    ESMC_AVAILABLE = False


# =========================
# Constants
# =========================

VALID_MODES = {"mean", "meanmax", "cls"}
DEFAULT_BASE_PATH = "/path/to/cafa6"
SUPPORTED_MODELS = ["esmc_300m", "esmc_600m", "esmc_1b"]


# =========================
# Path Configuration
# =========================

@dataclass(frozen=True)
class EmbedPaths:
    """Path configuration for embedding generation."""
    
    base_path: str
    
    @property
    def train_fasta(self) -> str:
        return os.path.join(self.base_path, "Train", "train_sequences.fasta")
    
    @property
    def test_fasta(self) -> str:
        return os.path.join(self.base_path, "Test", "testsuperset.fasta")
    
    @property
    def feats_dir(self) -> str:
        return os.path.join(self.base_path, "helpers", "feats")
    
    @property
    def train_ids_npy(self) -> str:
        return os.path.join(self.feats_dir, "train_ids.npy")
    
    @property
    def cache_root(self) -> str:
        return os.path.join(self.base_path, "cache", "embeds")
    
    def output_dir(self, model: str, mode: str) -> str:
        return os.path.join(self.cache_root, "esmc", model, mode)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return path."""
    os.makedirs(path, exist_ok=True)
    return path


# =========================
# FASTA Utilities
# =========================

def read_fasta_pairs(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Read FASTA file and return list of (id, sequence) pairs.
    
    Args:
        fasta_path: Path to FASTA file.
    
    Returns:
        List of (protein_id, sequence) tuples.
    """
    return list(read_fasta(fasta_path))


def sequences_for_ids(
    fasta_path: str,
    ids: np.ndarray,
    allow_missing: bool = False,
) -> List[str]:
    """
    Get sequences for specific IDs from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file.
        ids: Array of protein IDs.
        allow_missing: If False, raise error for missing IDs.
    
    Returns:
        List of sequences in the same order as ids.
    
    Raises:
        KeyError: If allow_missing is False and IDs are missing.
    """
    # Build id -> sequence mapping
    id_to_seq = {pid: seq for pid, seq in read_fasta(fasta_path)}
    
    sequences = []
    missing = []
    
    for pid in ids:
        if pid in id_to_seq:
            sequences.append(id_to_seq[pid])
        elif allow_missing:
            sequences.append("")
        else:
            missing.append(pid)
    
    if missing:
        raise KeyError(f"{len(missing)} IDs not found in FASTA. Example: {missing[:5]}")
    
    return sequences


def load_train_ids(train_ids_path: str) -> np.ndarray:
    ids = np.load(train_ids_path, allow_pickle=True)
    if ids.dtype != object:
        ids = ids.astype(object)
    return ids


# =========================
# Model loading
# =========================

def load_esmc_model(model_name: str, device: torch.device):
    """
    Load ESM-C model using the official SDK API.

    Valid model names (as of current open-source release):
      - esmc_300m
      - esmc_600m
      - esmc_1b
    
    Args:
        model_name: ESM-C model name.
        device: PyTorch device.
    
    Returns:
        Loaded ESM-C model in eval mode.
    
    Raises:
        ImportError: If ESM-C SDK is not available.
    """
    if not ESMC_AVAILABLE:
        raise ImportError("ESM-C SDK not available. Install with: pip install esm")
    
    if model_name not in SUPPORTED_MODELS:
        print(f"Warning: {model_name} not in known models {SUPPORTED_MODELS}")
    
    model = ESMC.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


# =========================
# Embedding
# =========================

@torch.no_grad()
def embed_sequence_esmc(
    model,
    seq: str,
    device: torch.device,
    mode: str,
    cls_index: int = 0,
) -> torch.Tensor:
    """
    Returns a 1D embedding tensor:
      - mean:    [D]
      - meanmax: [2D]
      - cls:     [D]

    ESM-C flow:
      sequence -> ESMProtein -> encode -> logits(return_embeddings=True)
    """

    protein = ESMProtein(sequence=seq)

    # Encode sequence
    protein_tensor = model.encode(protein)

    # Forward to get embeddings
    out = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )

    emb = out.embeddings  # [L, D] or [1, L, D]

    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)

    if emb.dim() == 3:
        emb = emb[0]  # [L, D]

    emb = emb.to(device)

    if mode == "mean":
        return emb.mean(dim=0)

    if mode == "meanmax":
        mean = emb.mean(dim=0)
        mx = emb.max(dim=0).values
        return torch.cat([mean, mx], dim=0)

    if mode == "cls":
        if not (0 <= cls_index < emb.shape[0]):
            raise IndexError(
                f"cls_index={cls_index} out of range for sequence length {emb.shape[0]}"
            )
        return emb[cls_index]

    raise ValueError(f"Unknown mode: {mode}")

def parse_modes(modes_str: str) -> List[str]:
    """
    Parse comma-separated mode string and validate.
    
    Args:
        modes_str: Comma-separated modes (e.g., "mean,meanmax,cls").
    
    Returns:
        List of unique valid modes.
    
    Raises:
        ValueError: If invalid modes are specified.
    """
    modes = [m.strip() for m in modes_str.split(",") if m.strip()]
    invalid = [m for m in modes if m not in VALID_MODES]
    
    if invalid:
        raise ValueError(f"Invalid modes: {invalid}. Allowed: {sorted(VALID_MODES)}")
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    
    return unique


@torch.no_grad()
def embed_batch_esmc_all_modes(
    model,
    seqs: List[str],
    device: torch.device,
    modes: List[str],
    cls_index: int = 0,
) -> dict:
    """
    One forward pass for a batch. Returns dict mode -> tensor [B, D or 2D].
    """
    proteins = [ESMProtein(sequence=s) for s in seqs]
    lengths = [len(s) for s in seqs]

    # --- batch encode/logits (fast path); fallback to per-seq if batch not supported ---
    try:
        protein_tensor = model.encode(proteins)
        out = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        emb = out.embeddings
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb = emb.to(device)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)  # [1, L, D]
    except Exception:
        # fallback: build [B, maxL, D] manually by per-seq calls
        embs = []
        for s in seqs:
            protein = ESMProtein(sequence=s)
            pt = model.encode(protein)
            out = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
            e = out.embeddings
            if isinstance(e, np.ndarray):
                e = torch.from_numpy(e)
            if e.dim() == 3:
                e = e[0]
            embs.append(e.to(device))
        # pad to max len
        maxL = max(x.shape[0] for x in embs)
        D = embs[0].shape[1]
        emb = torch.zeros((len(embs), maxL, D), device=device, dtype=embs[0].dtype)
        for i, e in enumerate(embs):
            emb[i, : e.shape[0], :] = e

    # emb: [B, Lpad, D]
    B, Lpad, D = emb.shape

    out_dict = {}

    # compute per-sample pooling (slice to true length to avoid padding)
    if "mean" in modes or "meanmax" in modes:
        means = []
        maxes = []
        for i, L in enumerate(lengths):
            e = emb[i, :L, :]
            means.append(e.mean(dim=0))
            if "meanmax" in modes:
                maxes.append(e.max(dim=0).values)
        means = torch.stack(means, dim=0)  # [B, D]
        if "mean" in modes:
            out_dict["mean"] = means
        if "meanmax" in modes:
            maxes = torch.stack(maxes, dim=0)  # [B, D]
            out_dict["meanmax"] = torch.cat([means, maxes], dim=1)  # [B, 2D]

    if "cls" in modes:
        cls_vecs = []
        for i, L in enumerate(lengths):
            if not (0 <= cls_index < L):
                raise IndexError(f"cls_index={cls_index} out of range for L={L}")
            cls_vecs.append(emb[i, cls_index, :])
        out_dict["cls"] = torch.stack(cls_vecs, dim=0)  # [B, D]

    return out_dict

# =========================
# Build embeddings
# =========================

def build_embeddings_multi(
    fasta_path: str,
    ids: Optional[np.ndarray],
    model_name: str,
    modes: List[str],
    device: torch.device,
    cls_index: int,
    limit: Optional[int] = None,
    batch_size: int = 8,
    sort_by_length: bool = True,
) -> Tuple[dict, np.ndarray]:
    """
    Returns:
      embeds_by_mode: dict(mode -> np.ndarray [N, D or 2D])
      out_ids: aligned ids
    """
    if ids is None:
        pairs = read_fasta_pairs(fasta_path)
        out_ids = np.array([p[0] for p in pairs], dtype=object)
        seqs = [p[1] for p in pairs]
    else:
        out_ids = ids.astype(object)
        seqs = sequences_for_ids(fasta_path, out_ids, allow_missing=False)

    if limit is not None:
        out_ids = out_ids[:limit]
        seqs = seqs[:limit]

    model = load_esmc_model(model_name, device)

    if sort_by_length:
        order = np.argsort([len(s) for s in seqs])
        seqs_sorted = [seqs[i] for i in order]
        ids_sorted = out_ids[order]
    else:
        seqs_sorted = seqs
        ids_sorted = out_ids
        order = None

    # accumulate tensors per mode
    chunks = {m: [] for m in modes}

    for i in tqdm(range(0, len(seqs_sorted), batch_size), desc=f"ESM-C embed {modes}"):
        batch_seqs = seqs_sorted[i:i+batch_size]
        batch_out = embed_batch_esmc_all_modes(
            model=model,
            seqs=batch_seqs,
            device=device,
            modes=modes,
            cls_index=cls_index,
        )
        for m in modes:
            chunks[m].append(batch_out[m].detach().cpu().to(torch.float32))

    embeds_by_mode = {m: torch.cat(chunks[m], dim=0).numpy().astype(np.float32) for m in modes}

    # unsort back
    if order is not None:
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        for m in modes:
            embeds_by_mode[m] = embeds_by_mode[m][inv]
        ids_sorted = ids_sorted[inv]

    return embeds_by_mode, ids_sorted


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-path", default=DEFAULT_BASE_PATH)

    # ⚠️ IMPORTANT: underscore, not dash
    ap.add_argument(
        "--model",
        default="esmc_300m",
        help="ESM-C model name (e.g., esmc_300m, esmc_600m, esmc_1b)",
    )

    ap.add_argument(
        "--modes",
        default="mean,meanmax,cls",
        help="Comma-separated modes to compute in one pass. Options: mean, meanmax, cls",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cls-index", type=int, default=0)

    ap.add_argument("--train-fasta", default=None)
    ap.add_argument(
        "--train-ids",
        default=None,
        help="Defaults to {base_path}/helpers/feats/train_ids.npy for alignment",
    )

    ap.add_argument(
        "--test-fasta",
        default=None,
        help="Optional: if provided, also compute test embeddings (fasta order)",
    )

    ap.add_argument(
        "--outdir",
        default=None,
        help="Override output dir. Default: {base_path}/cache/embeds/esmc/{model}/{mode}",
    )

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Micro-batch size (#proteins per forward). Increase until OOM.")
    ap.add_argument("--sort-by-length", action="store_true",
                    help="Sort sequences by length to reduce padding and speed up batching.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split dataset into N shards (for multi-GPU parallel runs).")
    ap.add_argument("--shard", type=int, default=0,
                    help="Which shard index to run (0..num_shards-1).")
    ap.add_argument("--run", default="both", choices=["train", "test", "both"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[esmc_embed] Device: {device}")

    paths = EmbedPaths(base_path=args.base_path)
    out_root = args.outdir or os.path.join(paths.cache_root, "esmc", args.model)
    ensure_dir(out_root)
    train_fasta = args.train_fasta or paths.train_fasta
    train_ids_path = args.train_ids or paths.train_ids_npy

    run_mode = args.run
    modes = parse_modes(args.modes)
    print(f"[esmc_embed] Model: {args.model}")
    print(f"[esmc_embed] Modes: {modes}")

    # TRAIN only or BOTH
    if run_mode in ("train", "both"):
        # TRAIN
        train_ids = load_train_ids(train_ids_path)
        train_embeds_by_mode, out_train_ids = build_embeddings_multi(
            fasta_path=train_fasta,
            ids=train_ids,
            model_name=args.model,
            modes=modes,
            device=device,
            cls_index=args.cls_index,
            limit=args.limit,
            batch_size=args.batch_size,
            sort_by_length=args.sort_by_length,
        )

        for m in modes:
            d = os.path.join(out_root, m)
            ensure_dir(d)
            np.save(os.path.join(d, "train_embeds.npy"), train_embeds_by_mode[m])
            np.save(os.path.join(d, "train_ids.npy"), out_train_ids, allow_pickle=True)
            print(f"[TRAIN:{m}] saved: {d}/train_embeds.npy shape={train_embeds_by_mode[m].shape}")

            # mode별 저장

    # TEST only or BOTH
    if run_mode in ("test", "both"):
        if not args.test_fasta:
            raise ValueError("--test-fasta is required when --run is test or both")
    # TEST
        test_embeds_by_mode, out_test_ids = build_embeddings_multi(
            fasta_path=args.test_fasta,
            ids=None,
            model_name=args.model,
            modes=modes,
            device=device,
            cls_index=args.cls_index,
            limit=args.limit,
            batch_size=args.batch_size,
            sort_by_length=args.sort_by_length,
        )
        for m in modes:
            d = os.path.join(out_root, m)
            ensure_dir(d)
            np.save(os.path.join(d, "test_embeds.npy"), test_embeds_by_mode[m])
            np.save(os.path.join(d, "test_ids.npy"), out_test_ids, allow_pickle=True)
            print(f"[TEST:{m}] saved: {d}/test_embeds.npy shape={test_embeds_by_mode[m].shape}")


if __name__ == "__main__":
    main()

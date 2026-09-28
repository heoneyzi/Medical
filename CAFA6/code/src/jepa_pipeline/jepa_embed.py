"""
JEPA-ProtT5 embedding generator.

Uses the pretrained JEPA encoder to generate protein embeddings
that can be used for downstream GO term prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from src.common.fasta import read_fasta
from src.common.protein import prott5_tokenize_sequence, sanitize_sequence


DEFAULT_BASE_PATH = "/path/to/cafa6"
DEFAULT_JEPA_MODEL = "models/jepa-prott5-1"


def load_jepa_encoder(
    model_dir: Path,
    device: torch.device,
) -> Tuple[T5EncoderModel, AutoTokenizer, dict]:
    """
    Load JEPA pretrained encoder.
    
    Args:
        model_dir: Path to JEPA model directory.
        device: PyTorch device.
    
    Returns:
        Tuple of (encoder, tokenizer, config).
    """
    config_path = model_dir / "jepa_config.json"
    encoder_path = model_dir / "encoder.pt"
    tokenizer_name_path = model_dir / "tokenizer_name.txt"
    
    if not config_path.exists():
        raise FileNotFoundError(f"JEPA config not found: {config_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"JEPA encoder not found: {encoder_path}")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    tokenizer_name = tokenizer_name_path.read_text().strip() if tokenizer_name_path.exists() else cfg.get("ckpt_name", "Rostlab/prot_t5_xl_uniref50")
    
    print(f"[jepa_embed] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    
    print(f"[jepa_embed] Loading encoder from: {cfg.get('ckpt_name', 'Rostlab/prot_t5_xl_uniref50')}")
    encoder = T5EncoderModel.from_pretrained(cfg.get("ckpt_name", "Rostlab/prot_t5_xl_uniref50"))
    
    print(f"[jepa_embed] Loading JEPA weights from: {encoder_path}")
    state = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(state["state_dict"])
    
    encoder.to(device)
    encoder.eval()
    
    return encoder, tokenizer, cfg


@torch.no_grad()
def embed_sequences(
    encoder: T5EncoderModel,
    tokenizer: AutoTokenizer,
    sequences: List[str],
    device: torch.device,
    mode: str = "mean",
    max_length: int = 1024,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Generate embeddings for sequences using JEPA encoder.
    
    Args:
        encoder: JEPA-pretrained T5 encoder.
        tokenizer: ProtT5 tokenizer.
        sequences: List of protein sequences.
        device: PyTorch device.
        mode: Pooling mode ('mean', 'cls', 'meanmax').
        max_length: Maximum sequence length.
        batch_size: Batch size for inference.
    
    Returns:
        Embeddings array [N, D] or [N, 2D] for meanmax.
    """
    embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="JEPA embedding"):
        batch_seqs = sequences[i:i + batch_size]
        
        # Tokenize (ProtT5 style: space-separated)
        tokenized_seqs = [prott5_tokenize_sequence(sanitize_sequence(s)) for s in batch_seqs]
        
        inputs = tokenizer(
            tokenized_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        out = encoder(**inputs)
        h = out.last_hidden_state  # [B, L, D]
        
        # Apply attention mask for proper pooling
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        h_masked = h * mask
        
        for j in range(h.shape[0]):
            seq_len = mask[j].sum().item()
            h_seq = h[j, :int(seq_len), :]  # [L_actual, D]
            
            if mode == "mean":
                emb = h_seq.mean(dim=0)
            elif mode == "cls":
                emb = h_seq[0]
            elif mode == "meanmax":
                emb = torch.cat([h_seq.mean(dim=0), h_seq.max(dim=0).values])
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            embeddings.append(emb.cpu().numpy())
    
    return np.stack(embeddings, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using JEPA-ProtT5 encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH,
                        help="Base path for CAFA6 data")
    parser.add_argument("--jepa-model", default=None,
                        help="Path to JEPA model directory")
    parser.add_argument("--train-fasta", default=None,
                        help="Training FASTA file")
    parser.add_argument("--test-fasta", default=None,
                        help="Test FASTA file")
    parser.add_argument("--train-ids", default=None,
                        help="Train IDs numpy file for alignment")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for embeddings")
    parser.add_argument("--mode", default="mean", choices=["mean", "cls", "meanmax"],
                        help="Pooling mode")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--device", default="cuda:0",
                        help="PyTorch device")
    parser.add_argument("--run", default="both", choices=["train", "test", "both"],
                        help="Which data to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sequences (for debugging)")
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.base_path)
    jepa_model_dir = Path(args.jepa_model) if args.jepa_model else base_path / DEFAULT_JEPA_MODEL
    train_fasta = Path(args.train_fasta) if args.train_fasta else base_path / "data/Train/train_sequences.fasta"
    test_fasta = Path(args.test_fasta) if args.test_fasta else base_path / "data/Test/testsuperset.fasta"
    train_ids_path = Path(args.train_ids) if args.train_ids else base_path / "helpers/feats/train_ids.npy"
    output_dir = Path(args.output_dir) if args.output_dir else base_path / "cache/embeds/jepa-prott5" / args.mode
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[jepa_embed] Device: {device}")
    print(f"[jepa_embed] Mode: {args.mode}")
    
    # Load encoder
    encoder, tokenizer, cfg = load_jepa_encoder(jepa_model_dir, device)
    
    # Process train data
    if args.run in ("train", "both"):
        print(f"\n[jepa_embed] Processing TRAIN data from: {train_fasta}")
        
        # Load train IDs for alignment if available
        if train_ids_path.exists():
            train_ids = np.load(train_ids_path, allow_pickle=True).astype(object)
            print(f"[jepa_embed] Aligning to {len(train_ids)} train IDs")
            
            # Build id -> seq mapping
            id_to_seq = {pid: seq for pid, seq in read_fasta(train_fasta)}
            sequences = [id_to_seq[pid] for pid in train_ids if pid in id_to_seq]
            out_ids = np.array([pid for pid in train_ids if pid in id_to_seq], dtype=object)
        else:
            pairs = list(read_fasta(train_fasta))
            out_ids = np.array([p[0] for p in pairs], dtype=object)
            sequences = [p[1] for p in pairs]
        
        if args.limit:
            out_ids = out_ids[:args.limit]
            sequences = sequences[:args.limit]
        
        train_embeds = embed_sequences(
            encoder, tokenizer, sequences, device,
            mode=args.mode,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        
        np.save(output_dir / "train_embeds.npy", train_embeds)
        np.save(output_dir / "train_ids.npy", out_ids, allow_pickle=True)
        print(f"[jepa_embed] Saved train embeddings: {train_embeds.shape}")
    
    # Process test data
    if args.run in ("test", "both"):
        print(f"\n[jepa_embed] Processing TEST data from: {test_fasta}")
        
        pairs = list(read_fasta(test_fasta))
        out_ids = np.array([p[0] for p in pairs], dtype=object)
        sequences = [p[1] for p in pairs]
        
        if args.limit:
            out_ids = out_ids[:args.limit]
            sequences = sequences[:args.limit]
        
        test_embeds = embed_sequences(
            encoder, tokenizer, sequences, device,
            mode=args.mode,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        
        np.save(output_dir / "test_embeds.npy", test_embeds)
        np.save(output_dir / "test_ids.npy", out_ids, allow_pickle=True)
        print(f"[jepa_embed] Saved test embeddings: {test_embeds.shape}")
    
    print(f"\n[jepa_embed] Done! Output: {output_dir}")


if __name__ == "__main__":
    main()

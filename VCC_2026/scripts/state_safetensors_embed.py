#!/usr/bin/env python3
"""Embed an H5AD with the official SE-100M safetensors release.

PyPI arc-state 0.11.1 currently searches only for Lightning ``*.ckpt`` files,
while the official SE-100M repository publishes ``model.safetensors``. This
adapter instantiates the upstream StateEmbeddingModel, derives the dataset-head
size from the checkpoint, and requires a strict state-dict match.
Run with the Python interpreter from the isolated arc-state uv tool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from state.emb.inference import Inference
from state.emb.nn.model import StateEmbeddingModel
from state.emb.utils import get_dataset_cfg, get_embedding_cfg, get_precision_config


def build_model(model_dir: Path, device: str) -> tuple[StateEmbeddingModel, object, dict]:
    cfg = OmegaConf.load(model_dir / "config.yaml")
    state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
    dataset_bias = state_dict.get("dataset_encoder.4.bias")
    if dataset_bias is not None:
        get_dataset_cfg(cfg).num_datasets = int(dataset_bias.shape[0])

    emb_cfg = get_embedding_cfg(cfg)
    model = StateEmbeddingModel(
        token_dim=emb_cfg.size, d_model=cfg.model.emsize, nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid, nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim, dropout=0.0, compiled=False,
        max_lr=cfg.optimizer.max_lr, emb_size=emb_cfg.size, cfg=cfg,
    )
    proteins = torch.load(model_dir / "protein_embeddings.pt", weights_only=False, map_location="cpu")
    if not isinstance(proteins, dict) or not proteins:
        raise TypeError("protein_embeddings.pt must contain a non-empty gene->tensor dict")
    model.pe_embedding = torch.nn.Embedding.from_pretrained(torch.vstack(list(proteins.values())))
    model.load_state_dict(state_dict, strict=True)

    dtype = get_precision_config("cuda" if device.startswith("cuda") else "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg, proteins


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-folder", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--embed-key", default="X_state")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    model, cfg, proteins = build_model(Path(a.model_folder), a.device)
    inferer = Inference(cfg=cfg, protein_embeds=proteins)
    inferer.init_from_model(model, proteins)
    embeddings = inferer.encode_adata(
        input_adata_path=a.input, output_adata_path=a.output,
        emb_key=a.embed_key, batch_size=a.batch_size,
    )
    if embeddings is None:
        raise RuntimeError("STATE returned no embeddings")
    print(f"[state-safetensors] {embeddings.shape} -> {a.output}::{a.embed_key}")


if __name__ == "__main__":
    main()

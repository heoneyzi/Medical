"""Phase 1 constants for Evo 2 7B (locked).

Source of truth for project-wide constants. Do NOT modify without
also updating PHASE1_DECISIONS.md.
"""
from __future__ import annotations
import math

# --- Model identity ---
MODEL_NAME: str = "evo2_7b"  # Evo2() constructor arg; falls back to evo2_7b_base if TE missing.
MODEL_NAME_FALLBACK: str = "evo2_7b_base"
HF_REVISION: str = "bda0089f92582d5baabf0f22d9fc85f3588f6b58"

# --- Vocabulary facts (verified by smoke test) ---
VOCAB_SIZE: int = 512
VOCAB_REAL_BIO = [65, 67, 71, 84, 78]  # A, C, G, T, N (ASCII)
LOG_VOCAB: float = math.log(VOCAB_SIZE)  # ~6.2383

# --- Tokenizer facts ---
BOS_OFFSET: int = 0  # Evo 2 tokenizer does NOT prepend BOS

# --- Block topology (32 layers, striped) ---
N_LAYERS: int = 32
HIDDEN_SIZE: int = 4096
ATTN_LAYERS = [3, 10, 17, 24, 31]
HCS_LAYERS = [0, 4, 7, 11, 14, 18, 21, 25, 28]
HCM_LAYERS = [1, 5, 8, 12, 15, 19, 22, 26, 29]
HCL_LAYERS = [2, 6, 9, 13, 16, 20, 23, 27, 30]

# --- Default DTR hyperparameters ---
GAMMA_DEFAULT: float = 0.5
RHO_DEFAULT: float = 0.85

# --- Reproducibility ---
SEED_DEFAULT: int = 42

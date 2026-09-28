"""Single source of truth for project-wide constants (Phase 0).

Values fixed by Appendix C of phase0_design.md. Do NOT change without
updating the design document.
"""
from __future__ import annotations
import math

# --- Model identity (locked) ---
MODEL_ID: str = "LongSafari/hyenadna-medium-160k-seqlen-hf"
HF_REVISION: str = "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce"

# --- Vocabulary facts (verified by smoke test) ---
# lm_head.out_features = 16 but only 12 are real tokens; mask logits[..., :12]
VOCAB_REAL: int = 12
LOG_VOCAB_REAL: float = math.log(VOCAB_REAL)  # ~2.4849

# --- Tokenizer facts ---
BOS_ID: int = 2
BOS_OFFSET: int = 1  # tokenizer prepends BOS automatically

# --- Default DTR hyperparameters (NLP DTR; calibrated in section 6.2) ---
GAMMA_DEFAULT: float = 0.5
RHO_DEFAULT: float = 0.85
L_DEFAULT: int = 8

# --- Reproducibility ---
SEED_DEFAULT: int = 42

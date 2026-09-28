"""Paths, model registry and run-wide constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(os.environ.get("EXP1_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_DIR = ROOT / "configs"
RESULTS = Path(os.environ.get("EXP1_RESULTS", ROOT / "results"))
CACHE = Path(os.environ.get("EXP1_CACHE", ROOT / "cache"))

for _p in (RESULTS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Byte-level tokenizer: A/C/G/T map to their ASCII codes.  Verified at load time
# against the model's own tokenizer, never assumed.
DEFAULT_ACGT_IDS = {"A": 65, "C": 67, "G": 71, "T": 84}

# Hard protocol constants -- changing these invalidates comparability with the
# published estimates, so they live here and nowhere else.
WINDOW_BP = 6000
SCORED_BP = 3000
MIN_N_PER_CLASS = 30
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
ONSET_THRESHOLD = 10.0

CONTEXTS = [
    "coding_exon",
    "intron",
    "five_utr",
    "three_utr",
    "splice_donor",
    "splice_acceptor",
    "intergenic",
]

# Covariates used for the adjusted columns.  Identical to the follow-up
# manuscript's list; do not extend it without re-running the unadjusted columns,
# or the two tables stop being comparable.
ADJUST_COVARIATES = [
    "entropy_final",
    "logp_true_final",
    "phylop",
    "repeat",
    "gc",
    "boundary_dist",
    "feature_len",
]


@dataclass
class ModelSpec:
    key: str
    hf_name: str
    n_blocks: int
    width: int
    context: str = ""
    dtype: str = "bfloat16"
    requires_te: bool = False
    block_module_path: str = "model.blocks"
    norm_module_path: str = "model.norm"
    unembed_module_path: str = "model.unembed"
    mixer_attr: Optional[str] = None      # pin if auto-discovery fails
    mlp_attr: Optional[str] = None
    notes: str = ""


@dataclass
class PanelSpec:
    chroms: List[str] = field(default_factory=lambda: ["chr22", "chr17"])
    n_windows: int = 400
    window_bp: int = WINDOW_BP
    scored_bp: int = SCORED_BP
    seed: int = 42
    max_n_fraction: float = 0.02
    fasta: str = ""
    gtf: str = ""
    phylop_bw: str = ""
    legacy_windows: str = ""


def load_models(path: Optional[Path] = None) -> Dict[str, ModelSpec]:
    path = path or CONFIG_DIR / "models.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    return {k: ModelSpec(key=k, **v) for k, v in raw["models"].items()}


def load_panel(path: Optional[Path] = None) -> PanelSpec:
    path = path or CONFIG_DIR / "panel.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    return PanelSpec(**raw["panel"])


def result_path(*parts: str) -> Path:
    p = RESULTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(*parts: str) -> Path:
    p = CACHE.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

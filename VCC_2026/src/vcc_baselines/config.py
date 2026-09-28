"""Configuration schema for VCC 2026 frozen baselines (v0.2, raw-counts space).

One YAML drives everything. See configs/synthetic.yaml (runnable offline) and
configs/real_example.yaml (points at the real `vcc datasets download` bundle).

2026 reality baked into the defaults (from the official `vcc` CLI):
  * predictions are RAW INTEGER COUNTS (2026 scores in counts space),
  * 18,533 genes, 400 cells per perturbation, contexts A/B/C,
  * the submission must NOT contain non-targeting control cells.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    contexts_dir: str = "data/validation"     # one sub-folder per context (A/B/C)
    control_h5ad: str = "controls.h5ad"       # that context's NT control cells
    truth_h5ad: str = "truth.h5ad"            # shadow-CV ground truth only
    targets: str = "data/validation/targets.csv"          # the panel gene IDs
    perts: str = "data/validation/pert_counts.csv"        # official pert list + n_cells
    gene_list: str = "data/validation/gene_names.csv"     # expected genes + order
    pert_col: str = "target_gene"
    control_label: str = "non-targeting"
    context_col: str = "context"              # A/B/C — the column vcc prep checks
    celltype_col: str = "celltype"            # optional passthrough


@dataclass
class GwpsCfg:
    enabled: bool = True
    sources: dict[str, str] = field(default_factory=lambda: {
        "k562": "data/gwps/K562_gwps_pseudobulk.h5ad",
        "rpe1": "data/gwps/RPE1_essential_pseudobulk.h5ad",
    })
    pert_col: str = "gene"
    control_label: str = "non-targeting"
    counts_space: bool = True     # GWPS X is raw counts? (if lognorm, set False)
    cache: str = "data/gwps/effect_library.npz"


@dataclass
class Esm2Cfg:
    enabled: bool = False
    model: str = "facebook/esm2_t33_650M_UR50D"
    fasta: str = ""
    embedding_cache: str = "data/esm2/gene_embeddings.npz"
    allow_random_fallback: bool = True
    knn_k: int = 5                # D6: ESM-2 KNN fallback for unseen genes
    device: str = "cuda"


@dataclass
class EmbedCfg:
    """Precomputed frozen context embeddings (STATE-SE / STACK / scGPT ...).

    npz mapping {context_name -> vector}. Provide these to run F6/F7 similarity
    (produced by `state emb transform` / `stack-embedding`). If absent, methods
    that ask for `similarity_space: embed` fall back to raw/pca.
    """
    context_embeddings: str = ""   # npz: keys = context names incl. gwps sources


@dataclass
class PredictCfg:
    method: str = "gwps_weighted"   # see methods.METHODS
    effect: str = "logfc"           # 'logfc' (multiplicative) or 'additive'
    generator: str = "multinomial"  # also: pseudobulk_multinomial (zero-lock free G0)
    cells_per_pert: int = 400
    include_controls: bool = False  # submission: False. local eval: set True.
    similarity_space: str = "raw"   # 'raw' | 'pca' | 'embed'
    pca_components: int = 50
    scale: float = 1.0              # D2b: logFC magnitude multiplier (shrink/expand)
    shrink: str = "none"            # D5: 'none' | 'context' | 'targetexpr' | 'both'
    tau: float = 0.10
    knn_k: int = 5
    nb_theta: float = 10.0          # NB dispersion (generator=nbinom)
    multinomial_alpha: float = 1e-3 # pooled G0 pseudocount per gene
    clip_negative: bool = True
    seed: int = 0
    # H1 ensemble (method: ensemble): list of {method, weight, ...} handled in cli


@dataclass
class SubmitCfg:
    packer: str = "vcc"             # 'vcc' (official) | 'cell-eval' | 'auto'
    gene_dim: int = 18533
    max_cell_dim: int = 400000
    max_counts_per_cell: int = 1000000
    encoding: int = 32
    require_counts: bool = True
    reject_controls: bool = True
    verify_targets: bool = True
    check_cell_counts: bool = True


@dataclass
class EvalCfg:
    profile: str = "full"           # cell-eval run profile for local scoring
    num_threads: int = 4


@dataclass
class Config:
    run_name: str = "run"
    output_dir: str = "outputs"
    data: DataCfg = field(default_factory=DataCfg)
    gwps: GwpsCfg = field(default_factory=GwpsCfg)
    esm2: Esm2Cfg = field(default_factory=Esm2Cfg)
    embed: EmbedCfg = field(default_factory=EmbedCfg)
    predict: PredictCfg = field(default_factory=PredictCfg)
    submit: SubmitCfg = field(default_factory=SubmitCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    # Optional protocol metadata for independent/audited experiments.
    experiment: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(dc, overrides: dict[str, Any]):
        if not overrides:
            return dc
        fields = {f.name: f for f in dataclasses.fields(dc)}
        for k, v in overrides.items():
            if k not in fields:
                raise KeyError(f"Unknown config key '{k}' in {type(dc).__name__}")
            cur = getattr(dc, k)
            if dataclasses.is_dataclass(cur) and isinstance(v, dict):
                setattr(dc, k, Config._merge(cur, v))
            else:
                setattr(dc, k, v)
        return dc

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls._merge(cls(), raw)

    def resolve(self, root: str | Path) -> "Config":
        root = Path(root)

        def fix(p: str) -> str:
            if not p:
                return p
            pp = Path(p)
            return str(pp if pp.is_absolute() else (root / pp))

        self.output_dir = fix(self.output_dir)
        for attr in ("contexts_dir", "targets", "perts", "gene_list"):
            setattr(self.data, attr, fix(getattr(self.data, attr)))
        self.gwps.sources = {k: fix(v) for k, v in self.gwps.sources.items()}
        self.gwps.cache = fix(self.gwps.cache)
        self.esm2.fasta = fix(self.esm2.fasta)
        self.esm2.embedding_cache = fix(self.esm2.embedding_cache)
        self.embed.context_embeddings = fix(self.embed.context_embeddings)
        return self

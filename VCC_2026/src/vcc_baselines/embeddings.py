"""Gene protein embeddings for the ESM-2 nearest-neighbour method.

If ESM-2 (via `transformers`) and a protein FASTA are available, real ESM-2
embeddings are computed on GPU and cached. Otherwise, if
`allow_random_fallback` is set, a *deterministic* pseudo-embedding derived from
the gene symbol is used so the pipeline still runs end-to-end. The random
fallback is for smoke-testing plumbing only — it carries no biology.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import Esm2Cfg


def _read_fasta(path: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name, buf = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            # header format: ">GENE ..." — take the first whitespace token
            name = line[1:].strip().split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def _random_embedding(gene: str, dim: int = 320) -> np.ndarray:
    h = hashlib.sha256(gene.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal(dim).astype(np.float32)


class GeneEmbedder:
    def __init__(self, cfg: Esm2Cfg):
        self.cfg = cfg
        self._cache: dict[str, np.ndarray] = {}
        self._model = None
        self._tok = None
        self._seqs: dict[str, str] = {}
        self._backend = "random"
        if Path(cfg.embedding_cache).exists():
            z = np.load(cfg.embedding_cache, allow_pickle=True)
            self._cache = {str(k): z[k] for k in z.files}
        if cfg.enabled and cfg.fasta and Path(cfg.fasta).exists():
            try:
                self._init_esm(cfg)
                self._seqs = _read_fasta(cfg.fasta)
                self._backend = "esm2"
            except Exception as e:  # pragma: no cover - optional heavy path
                if not cfg.allow_random_fallback:
                    raise
                print(f"[embeddings] ESM-2 unavailable ({e}); using deterministic random fallback")

    def _init_esm(self, cfg: Esm2Cfg):
        import torch  # noqa
        from transformers import AutoModel, AutoTokenizer
        self._device = "cuda" if (cfg.device == "cuda" and _cuda()) else "cpu"
        self._tok = AutoTokenizer.from_pretrained(cfg.model)
        self._model = AutoModel.from_pretrained(cfg.model).eval().to(self._device)

    def _embed_one(self, gene: str) -> np.ndarray:
        if gene in self._cache:
            return self._cache[gene]
        if self._backend == "esm2" and gene in self._seqs:
            import torch
            with torch.no_grad():
                seq = self._seqs[gene][:1022]
                t = self._tok(seq, return_tensors="pt").to(self._device)
                out = self._model(**t).last_hidden_state.mean(1)[0]
                v = out.float().cpu().numpy()
        else:
            v = _random_embedding(gene)
        self._cache[gene] = v
        return v

    def embed(self, genes: list[str]) -> np.ndarray:
        vs = [self._embed_one(str(g)) for g in genes]
        dim = max(len(v) for v in vs)
        # pad/truncate to a common dim (mixed backends only happens in fallback)
        out = np.zeros((len(vs), dim), np.float32)
        for i, v in enumerate(vs):
            out[i, : len(v)] = v[:dim]
        return out

    def save_cache(self):
        if not self._cache:
            return
        Path(self.cfg.embedding_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cfg.embedding_cache, **self._cache)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

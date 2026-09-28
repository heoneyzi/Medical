<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [CAFA6](../../README.md) › [Pipelines](../README.md) › **P3 · ESM-C embeddings**</sub>

# 🔬 P3 · ESM-C embeddings + MLP heads

> **Question —** Are frozen ESM-C protein embeddings plus a light multi-label head enough, and which pooling (mean, mean⊕max, CLS) carries the most function signal?

| | |
|---|---|
| **Status** | 🧪 exploratory (Jan 2026) — Yoonjin Cho's line (team lead) |
| **Model / data** | ESM-C 300M (EvolutionaryScale `esm` SDK), frozen; CAFA 6 training set with precomputed label matrix and cross-validation folds |
| **Compute** | single GPU; embeddings cached to disk, heads trained with AMP |
| **Headline** | early base-stage runs: fold-level micro-F1 **0.0154 – 0.0201** (unweighted, best threshold 0.4) |

## Setup

Two stages, both in [`code/src/esm-c_model/`](../../code/src/esm-c_model/):

1. **Embed** — [`esmc_embed.py`](../../code/src/esm-c_model/esmc_embed.py) runs ESM-C over every sequence and caches one vector per protein for each pooling mode: `mean` (average over residues, *D*), `meanmax` (mean ⊕ max, *2D*) or `cls` (first token, *D*).
2. **Train** — [`train_base.py`](../../code/src/esm-c_model/train_base.py) + [`trainer.py`](../../code/src/esm-c_model/trainer.py) fit a linear or MLP head (hidden 1024 → 512, dropout 0.2) per fold on precomputed folds (`folds.npy`; grouped K-fold, by taxonomy according to the design notes), with BCE computed only on each protein's positives plus **1,024 shared negative labels per batch** — a trick that avoids materializing the full protein × ~26k-label logit matrix. Out-of-fold and test logits are stored as float16 memory maps; the model with the best validation F-max per fold is kept.

The example config is [`code/configs/esmc_base_example.yaml`](../../code/configs/esmc_base_example.yaml) (`esmc_300m`, `mean`, MLP head, lr 1e-3, 5 epochs, batch 512, 5 folds). The design notes are in [`code/AGENT.md`](../../code/AGENT.md).

The plan (Yoonjin's [“Method” page](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/cafa6/06_yoonjin_method.md)) was a CAFA 5-style stack: embedding → linear GO scores → **GCN refinement over the GO DAG**. Because the GCN was slow to train, the base heads were run first; the GCN stage is not in the repository.

## Results

Screenshots of the base-stage training logs in [Yoonjin Cho's trial page](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/cafa6/05_yoonjin_code_attempt1.md) (team Notion, Jan 2026); the log line reports an unweighted micro-F1 at the best threshold of the grid (all at 0.4):

| Run (directory name) | Pooling / head | Epochs | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| `exp3_cls_mlp` | CLS token → MLP | 8 | – | 0.0201 | 0.0184 |
| `exp1_meanmax_linear_gcn` (base stage) | mean ⊕ max → linear | 6 | 0.0154 | 0.0171 | 0.0171 |

The [4th team meeting](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/meetings/04_meeting04_2026-01-29.md) (29 Jan 2026) concluded these heads "fell short of expectations" and that predicting exact GO terms from embeddings alone is very hard; the team decided to anchor predictions with a GO retriever and tune them with embeddings (**late fusion**), and to evaluate with the official IA-weighted metric instead of plain F1.

## Takeaway

- The numbers are **tiny and not comparable** to the leaderboard: unweighted micro-F1 over the full label space, early epochs, an older trainer version, and a best threshold of 0.4, the top of the current code's default grid (0.05–0.4) — the heads were far from calibrated.
- They were still informative: a frozen embedding with a flat multi-label head does not exploit the GO hierarchy, which motivated the GCN refinement and the retrieval-based late fusion discussed next.
- **Not claimed:** any leaderboard score for ESM-C models.

## Files

| File | What it is |
|---|---|
| [`src/esm-c_model/esmc_embed.py`](../../code/src/esm-c_model/esmc_embed.py) | embedding cache builder (mean / meanmax / cls) |
| [`src/esm-c_model/train_base.py`](../../code/src/esm-c_model/train_base.py) · [`trainer.py`](../../code/src/esm-c_model/trainer.py) | CV training, OOF/test logits, checkpoints |
| [`src/esm-c_model/models.py`](../../code/src/esm-c_model/models.py) · [`dataset.py`](../../code/src/esm-c_model/dataset.py) · [`metrics.py`](../../code/src/esm-c_model/metrics.py) | head, negative sampling, micro F-max |
| [`src/esm-c_model/config.py`](../../code/src/esm-c_model/config.py) · [`paths.py`](../../code/src/esm-c_model/paths.py) | YAML config dataclasses and path layout (server path → `/path/to/cafa6`) |
| [`configs/esmc_base_example.yaml`](../../code/configs/esmc_base_example.yaml) | example experiment config |

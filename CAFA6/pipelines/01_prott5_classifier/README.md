<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [CAFA6](../../README.md) › [Pipelines](../README.md) › **P1 · ProtT5 classifier**</sub>

# 🔬 P1 · ProtT5 classifier — the team's “T5” line

> **Question —** Can a ProtT5 protein language model, partly fine-tuned with a multi-label head per sub-ontology, score thousands of GO terms directly from sequence?

| | |
|---|---|
| **Status** | 📦 implemented and archived (Jan 2026) — no validation or leaderboard number was saved |
| **Model / data** | `Rostlab/prot_t5_xl_uniref50` encoder (ProtT5-XL) + masked mean-pooling + 2-layer MLP; CAFA 6 training set, 1 % random hold-out |
| **Compute** | SLURM, 1 GPU (job script targets an RTX 3090 partition), 64 GB RAM, 48 h limit per job |
| **Headline** | — (no archived score; see Takeaway) |

**Jiheon's link to this line.** The team's role split at the [3rd meeting](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/meetings/03_meeting03_2026-01-21.md) (21 Jan 2026) reads "지헌 : T5", and the [4th meeting](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/meetings/04_meeting04_2026-01-29.md) (29 Jan 2026) records "지헌: T5 임베딩을 활용한 비교 실험 진행" (a comparison experiment using T5 embeddings). The code below is the ProtT5 implementation in the team repository; upstream commits were all pushed from the maintainer's account, so git does not record who wrote which file.

## Setup

**Training** — [`code/src/train/prott5_go_train.py`](../../code/src/train/prott5_go_train.py), launched per sub-ontology by [`code/job-scripts/train.sh`](../../code/job-scripts/train.sh):

| Setting | Value (train.sh) | Why |
|---|---|---|
| Label vocabulary | GO terms with ≥ 5 training proteins, capped at the 4,096 most frequent | keep the head tractable on a long-tailed label space |
| Models | three separate models: MF, BP, CC | the metric scores each sub-ontology independently |
| Encoder | frozen except the last 2 T5 blocks + final layer norm | adapt the top of the pLM with little GPU memory |
| Pooling / head | masked mean over residues (padding and `</s>` excluded) → Linear 1024→1024 → ReLU → Dropout 0.1 → Linear → logits | |
| Loss / optimizer | `BCEWithLogitsLoss`, AdamW, lr 2e-4, batch 1, 1 epoch, max length 1,024 tokens | |
| Validation | 1 % hold-out; when W&B is on: IA-weighted and unweighted F-max per sub-ontology after parent propagation, plus top-k precision/recall (k = 50, 200, 1500) | mirrors the competition metric in the loop |

Sequences are upper-cased, rare residues U/Z/O/B map to X, and residues are space-separated for the ProtT5 tokenizer ([`code/src/common/protein.py`](../../code/src/common/protein.py)). An optional `--init-encoder-from` loads a JEPA-pre-trained encoder ([P4](../04_jepa_encoder/README.md)).

**Prediction** — [`code/src/test/prott5_go_predict.py`](../../code/src/test/prott5_go_predict.py) (or [`job-scripts/test.sh`](../../code/job-scripts/test.sh)): sigmoid scores → top 1,500 terms per model → MF/BP/CC models merged by max → **propagated to all GO ancestors** (parent = max of children) → min score 0.001, ≤ 1,500 terms per protein, scores written to 3 significant figures as the rules require. [`code/notebooks/prot-t5-pred.ipynb`](../../code/notebooks/prot-t5-pred.ipynb) drives the same modules interactively (MF model, `DO_TRAIN = False` by default).

## Results

No metric was archived for this pipeline: the repository holds no training logs or checkpoints, the W&B project is not exported, and the meeting notes record the assignment but not its outcome. The team's saved numbers belong to other lines ([P3](../03_esm_c/README.md), [P5](../05_label_space_jepa/README.md)).

## Takeaway

- A complete, rule-compliant supervised baseline: per-ontology heads, in-loop IA-weighted F-max, ontology propagation and submission formatting.
- Freezing all but the top two T5 blocks is what makes the large ProtT5-XL encoder trainable at batch size 1 on a single GPU; the flip side is a short schedule (one epoch per sub-ontology by default).
- **Not claimed:** any score for this model, or that it produced the medal-winning submission.

## Files

| File | What it is |
|---|---|
| [`src/train/prott5_go_train.py`](../../code/src/train/prott5_go_train.py) | model, dataset, label vocabulary, training loop, in-loop IA-weighted F-max |
| [`src/train/prott5_go_train_config.py`](../../code/src/train/prott5_go_train_config.py) | dataclass config with CLI aliases (`--out`, `--epochs`, `--lr`, …) and a debug preset |
| [`src/test/prott5_go_predict.py`](../../code/src/test/prott5_go_predict.py) | single- or multi-model prediction with propagation and capping |
| [`src/common/`](../../code/src/common/) | FASTA reader, GO parent parsing + propagation, sequence sanitizing |
| [`job-scripts/train.sh`](../../code/job-scripts/train.sh) · [`test.sh`](../../code/job-scripts/test.sh) · [`train-test.sh`](../../code/job-scripts/train-test.sh) | SLURM launchers (MF → BP → CC; then prediction) |
| [`notebooks/prot-t5-pred.ipynb`](../../code/notebooks/prot-t5-pred.ipynb) | notebook front-end for the two modules |

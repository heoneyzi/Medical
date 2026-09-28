<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [CAFA6](../README.md) › **Code**</sub>

# 💻 CAFA6 code snapshot

Source snapshot of **[heoneyzi/CAFA6](https://github.com/heoneyzi/CAFA6)** (commit `6157174`, 12 Sep 2026), Jiheon's fork of the team repository **[SOL1archive/CAFA6](https://github.com/SOL1archive/CAFA6)**. The Python package layout is kept intact so every `python -m src.…` command in the docs works from this folder. The fork's own README is kept as [ORIGINAL_README.md](ORIGINAL_README.md).

**Provenance.** The 16 upstream commits (20 Jan – 3 Feb 2026) were all pushed from the team maintainer's account; Jiheon's 3 fork commits (12 Sep 2026) change documentation only (`README.md`, `AGENT.md`, `docs/en/*`, `docs/ko/reproducing_kaggle_ensemble.md`, new `docs/en/release_scope.md`). A local download of the upstream repository (`CAFA6-main`) was diffed against this snapshot: all code, notebooks and scripts are byte-identical; it differs only in those documentation files and lacks `uv.lock`.

## Map

| Path | Contents | Pipeline |
|---|---|---|
| [`eda.ipynb`](eda.ipynb) | dataset EDA with saved outputs | [P0](../pipelines/00_eda/README.md) |
| [`src/common/`](src/common/) | FASTA reader, GO parent parsing / propagation / capping, sequence sanitizing | shared |
| [`src/train/`](src/train/) · [`src/test/`](src/test/) · [`notebooks/prot-t5-pred.ipynb`](notebooks/prot-t5-pred.ipynb) | ProtT5 classifier training and prediction | [P1](../pipelines/01_prott5_classifier/README.md) |
| [`notebooks/cafa-6-goa-prott5-ensemble-0-370.ipynb`](notebooks/cafa-6-goa-prott5-ensemble-0-370.ipynb) · [`scripts/make_ensemble_submission.py`](scripts/make_ensemble_submission.py) | public GOA + ProtT5 ensemble and its local port | [P2](../pipelines/02_goa_prott5_ensemble/README.md) |
| [`src/esm-c_model/`](src/esm-c_model/) · [`configs/esmc_base_example.yaml`](configs/esmc_base_example.yaml) | ESM-C embeddings + CV heads | [P3](../pipelines/03_esm_c/README.md) |
| [`src/pretrain/`](src/pretrain/) · [`src/jepa_pipeline/`](src/jepa_pipeline/) · `src/jepa_go/*jepa_encoder*` | JEPA pre-training and frozen-encoder head | [P4](../pipelines/04_jepa_encoder/README.md) |
| [`src/jepa_go/`](src/jepa_go/) (`*label_jepa*`, `ontology.py`, `taxonomy.py`) | label-space JEPA | [P5](../pipelines/05_label_space_jepa/README.md) |
| [`job-scripts/`](job-scripts/) | SLURM launchers for all lines | — |
| [`docs/`](docs/) | task spec, dataset guide, pipeline guide, ensemble guide (EN/KO), release scope | — |
| [`AGENT.md`](AGENT.md) | historical implementation notes from upstream (some examples outdated — see `docs/en/pipelines.md`) | — |
| [`pyproject.toml`](pyproject.toml) · [`uv.lock`](uv.lock) · [`.python-version`](.python-version) | Python ≥ 3.12 environment (PyTorch ≥ 2.9.1, Transformers, ESM, goatools, pronto, …) | — |

## Run

```bash
uv sync --frozen && source .venv/bin/activate
# data/Train/{train_sequences.fasta,train_terms.tsv,train_taxonomy.tsv,go-basic.obo}, data/Test/testsuperset.fasta, data/IA.tsv
python -m src.jepa_go.train_label_jepa --model_family esm --backbone_ckpt facebook/esm2_t33_650M_UR50D --output_dir models/label-jepa
python -m src.jepa_go.predict_label_jepa --model_dir models/label-jepa --test_fasta data/Test/testsuperset.fasta \
       --output submissions/label-jepa.tsv --min_score 0.001
```

Competition data, pretrained weights and checkpoints are not included. Per-pipeline commands are in each [pipeline README](../pipelines/README.md) and in [`docs/en/pipelines.md`](docs/en/pipelines.md).

## Changes made for this portfolio

| Change | Files |
|---|---|
| Server-specific absolute paths (a shared-cluster scratch directory and a personal workspace) replaced by the placeholder `/path/to/cafa6` | 14 files: all `job-scripts/*.sh`, `configs/esmc_base_example.yaml`, `src/esm-c_model/{config,esmc_embed,paths}.py`, `src/jepa_pipeline/jepa_embed.py` |
| `README.md` → `ORIGINAL_README.md`; `.git/` and `.gitignore` not copied | — |

Nothing else was edited; SLURM partition names and resource requests are the originals. The upstream README states an MIT license, but no `LICENSE` file exists in the checkout — check with the [team repository](https://github.com/SOL1archive/CAFA6) before reuse. Pretrained models (ProtT5, ESM2, ESM-C) and the Kaggle data keep their own terms.

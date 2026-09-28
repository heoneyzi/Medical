<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [PhenoFocus](../README.md) › **Code**</sub>

# 💻 The PhenoFocus MVP layer

> **What this is —** the team's own code, written on top of the released [PhenoCompass](https://github.com/genentech/phenocompass) model. The upstream package (`src/phenocompass/`, its notebooks, checkpoints and figures) is **not** vendored here; install it from its own repository. Everything in this folder is the PhenoFocus layer.

| | |
|---|---|
| **Author** | Jiheon Kang (Team Lead), July 2026 |
| **Depends on** | `phenocompass` (upstream, MIT) · torch · torch-geometric · rdkit · scikit-learn · scipy · pandas · matplotlib · pillow |
| **Runs without the model?** | Yes — `--mock` uses a synthetic, watermarked stand-in so the wiring can be smoke-tested |
| **Provenance** | server paths in the original scripts have been replaced with relative placeholders (`./final_model`, `./final_data`) for publication; nothing else was edited |

## What the layer does

PhenoCompass gives you a 12-model ensemble, a shared structure/morphology embedding space and six frozen mechanism anchor sets. It does **not** give you the product loop: *one customer hit in, structurally independent same-mechanism candidates out, with statistics attached*. That loop, its metrics and its figures are what this folder implements.

```text
a549_mvp/
├── expand_hit.py              ← E1: one hit → ranked candidates + significance + report
├── run_mvp.py                 ← E2: six-mechanism recovery + few-shot expansion sweep
├── mvp_core.py                ← the metric core (no torch): scoring, AUROC, enrichment, retrieval, Tanimoto
├── moa_mapping.py             ← free-text mechanism strings → the six anchor clusters (editable)
├── prepare_a549_compounds.py  ← build the compound table (LINCS-native or deposit fallback)
├── download_a549.py           ← fetch the public LINCS A549 consensus profiles
├── plots.py                   ← the four benchmark figures
├── mock_model.py              ← checkpoint-free synthetic stand-in (watermarks its output)
├── a549_adapter.py            ← (Direction B) CellProfiler → model feature-space alignment; unused by A
├── find_jump_features.py      ← (Direction B) locate the training feature list inside a deposit
└── ROADMAP_B_morphology.md    ← the scoped, unrun image-side cross-cell-line test
```

| Module | Role | Notes |
|---|---|---|
| `expand_hit.py` | the E1 loop | resolves a query by SMILES, id or `auto:<mechanism>`; ranks the library by cosine similarity; hypergeometric enrichment + ranking AUROC; draws the molecule grid, significance panel and Tanimoto heatmap; writes a self-contained HTML one-pager |
| `run_mvp.py` | the E2 benchmark | scores structures against all six anchor sets, evaluates mechanism recovery and a 3-shot expansion sweep, writes `summary.json` + four figures. Also contains the checkpoint-loading shim described below |
| `mvp_core.py` | metrics | deliberately free of any hard PyTorch dependency, so the numeric pipeline can be unit-tested against the mock model. Holds `evaluate_cross_line`, `evaluate_fewshot_retrieval`, `max_tanimoto_to_refs`, `pairwise_tanimoto_matrix`, ensemble averaging |
| `moa_mapping.py` | label mapping | lower-cased substring lookup from free-text mechanism annotations onto the six clusters; intentionally simple and auditable, and the run prints how many compounds matched each cluster |
| `mock_model.py` | dry run | synthetic embeddings drawn near per-mechanism prototypes, so figures look plausible; **every mock figure is watermarked "SYNTHETIC / MOCK"** and the CLI prints a warning banner |

Two engineering details worth calling out, both in `run_mvp.py`:

- **Loading the released checkpoints on a modern PyTorch.** The deposit's Lightning checkpoints were pickled by the training package under its former name and store OmegaConf objects in their hyper-parameters, so PyTorch ≥ 2.6 refuses them under `weights_only=True`. `_enable_legacy_checkpoint_loading()` installs a meta-path finder that resolves the old module path to the renamed package by class name, forces `weights_only=False` for these trusted local files, and allow-lists the OmegaConf globals.
- **Finding the real morphology input.** The shipped model config advertises a CellProfiler-sized morphology input, but the released weights expect a **1,536-dimensional DINO image embedding** — checked against the encoder's first weight matrix and the deposit's morphology matrix. This is why the image-side test (Direction B) needs raw Cell Painting images plus the DINO encoder, and why the MVP started from the structure side. `a549_adapter.py` and `find_jump_features.py` are the leftovers of the CellProfiler route, kept because Direction B reuses parts of them.

## Running it

```bash
# 1. wiring smoke test — no checkpoints, no torch needed beyond the deps above
python a549_mvp/run_mvp.py --mock --synthetic-compounds --out-dir ./out_mock

# 2. build a compound table (LINCS-native when --a549-dir is present, else deposit fallback)
python a549_mvp/prepare_a549_compounds.py --deposit ./final_data --a549-dir ./a549_data --out ./compounds.csv

# 3. the two experiments
python a549_mvp/expand_hit.py --final-model-dir ./final_model --a549-compounds ./compounds.csv \
       --hit "auto:hdac" --top-k 5 --out-dir ./expansions
python a549_mvp/run_mvp.py   --final-model-dir ./final_model --a549-compounds ./compounds.csv \
       --out-dir ./mvp_out
```

`notebooks/A549_cross_line_validation.ipynb` walks the same pipeline narratively (outputs stripped; it defaults to `MOCK = True`).

> [!NOTE]
> Naming: files and flags say `a549` because the run targeted the A549 lung-cancer line. The committed results were produced with the **deposit's own compound set** as a fallback — see the scope notes in the [project README](../README.md).

## Attribution

PhenoCompass is © 2026 Genentech, Inc., released under the MIT License; this layer calls its public API (`PhenoCompass`, `compute_struct_embeddings`, `score_against_anchors`) and does not modify it. The upstream repository, preprint and Zenodo deposit are linked from the [project README](../README.md).

---
<sub>[🏠 Portfolio](https://github.com/heoneyzi) · [PhenoFocus](../README.md)</sub>

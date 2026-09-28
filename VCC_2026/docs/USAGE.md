# Usage guide — VCC 2026 Frozen Baselines

> Original `README.md` of the code base, moved here; only links were re-rooted and the clone line was replaced by a `cd` into the project. Commands assume you are in the project root (`01_Medical/VCC_2026`). Portfolio overview: [../README.md](../README.md).

## VCC 2026 — Frozen Baselines (Replogle GWPS + raw-count generators, leaderboard-ready)

Training-free baselines for the **[Virtual Cell Challenge 2026](https://virtualcellchallenge.org/)**, implementing the v2 strategy: **Replogle GWPS matching** (D1–D6), **frozen foundation models** (STATE-SE / STACK context encoders → F6/F7, and STATE-ST/STACK direct via adapters), **raw-count generators** (G0–G2), a **no-training hybrid** (H1), and a **coverage-stratified 3-rung ladder** for metric analysis. Effect prediction and raw-count generation are deliberately **separated** so you can ablate each.

Built against the **real 2026 format** (verified from the official `vcc` CLI): predictions are **raw integer counts**, **18,533 genes**, **400 cells/perturbation**, contexts **A/B/C**, and the submission **excludes non-targeting controls**. Packaged with the official **`vcc prep`** and validated end-to-end in the sandbox (`tests/`, `scripts/quickstart.sh`).

---

## 0. What it does (two perspectives, one pipeline)

```
                 target-context NT controls  +  target gene
                             │
   ┌─────────── context representation (raw / PCA / STATE-SE / STACK) ───────────┐
   │                         │                                                   │
   ▼                         ▼                                                   ▼
 D1/D2 direct        D3/D4 similarity-weighted            F6/F7 embed-weighted (frozen FM)
 (additive/logFC)    K562↔RPE1 blend                      same GWPS effect, better context sim
                             │
                 D6 esm2_knn (unseen gene → nearest measured genes)
                             │
                    Replogle GWPS relative-fold effect  ── D2b scale ── D5 shrink
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
      G0/G1/G2 raw-count generator   state/stack direct (adapters)
                 │                       │
                 └───────────┬───────────┘  (H1 ensemble)
                             ▼
            400 raw-count cells × targets × A/B/C  →  vcc prep  →  .vcc  →  leaderboard
```

The 3-rung ladder decomposes the score: **no-effect (0-anchor) → GWPS transfer (gene-effect memory) → frozen FM (context representation)**.

---

## 1. Prerequisites & install

```bash
cd 01_Medical/VCC_2026                                    # inside a clone of this portfolio repository
python -m venv .venv && source .venv/bin/activate      # Python >= 3.11 (required)
pip install -e .                                        # installs vcc-cli + cell-eval too
sudo apt-get install -y zstd                            # system dep for vcc prep / cell-eval
# optional, only for esm2_knn with REAL ESM-2 (GPU):
pip install -r requirements-esm.txt
```

Prerequisite data (for a real run — not needed for the offline demo):

- **Challenge controls bundle** — `vcc login` (token from the web app → Account), then
  `vcc datasets list` and `vcc datasets download controls -d ./data`. Gives
  `gene_names.csv`, `pert_counts.csv`, and per-context control cells.
- **Replogle GWPS** — `bash scripts/download_gwps.sh` (K562 genome-wide, RPE1 essential
  from https://gwps.wi.mit.edu/ ). This repo's DB policy is **GWPS-only**.
- **(optional) STATE / STACK checkpoints** — from https://github.com/ArcInstitute/state
  and https://github.com/ArcInstitute/stack , for the F/direct methods.

---

## 2. 90-second offline smoke test (no downloads, proves it runs)

```bash
bash scripts/quickstart.sh
```

Runs: synthetic raw-count data → ladder → coverage → **`vcc prep` dry-run + real `.vcc`** → `vcc sample`.

Expected ladder (synthetic; the **MAE trap** in counts space):

| method        | pdisc_norm_rank ↓ | mae_pseudobulk ↓ | pearson_delta ↑ |
|:--------------|------------------:|-----------------:|----------------:|
| no_effect     | 0.50 (random)     | ~0.40            | ~0.00           |
| gwps_direct   | **0.11**          | ~0.41            | **0.33**        |
| gwps_weighted | **0.12**          | ~0.41            | **0.34**        |

Expected coverage stratification: **GWPS-covered targets pdisc ≈ 0.02**, **missing targets ≈ 0.53** (the DB reach ceiling; ESM-2 KNN lifts the missing group only with real embeddings).

And `vcc prep` reports: *"targets verified against the official list; normalization: counts-preserved"* — i.e. leaderboard-valid.

---

## 3. Real challenge run (local → leaderboard)

```bash
# a) get data
vcc login
vcc datasets download controls -d ./data          # gene_names.csv, pert_counts.csv, controls
bash scripts/download_gwps.sh ./data/gwps

# b) lay out per-context controls: data/validation/{A,B,C}/controls.h5ad
#    (split the official controls by the `context` column, or use provided per-context files)

# c) edit configs/real_example.yaml paths, then predict + package
python -m vcc_baselines predict --config configs/real_example.yaml --method gwps_weighted --dry-run
python -m vcc_baselines predict --config configs/real_example.yaml --method gwps_weighted --prep

# d) submit (one research question per submission — see §6)
vcc submit outputs/gwps_weighted/submission.prep.vcc \
   -m "D4_gwps_weighted_logfc" -d "GWPS raw-cosine weighted logFC transfer, multinomial" --wait
```

`vcc prep` enforces 18,533 genes, 400 cells/pert, raw counts, no controls, and per-context target verification against `pert_counts.csv` — the same checks the server runs.

---

## 4. Methods (`--method`) and knobs

| id | ladder | what it does |
|----|--------|--------------|
| `no_effect`     | A0 | relfold=1 → generator reproduces controls (0-anchor + pipeline sanity) |
| *(vcc sample)*  | A1 | official random dummy — `python -m vcc_baselines sample` |
| `gwps_direct`   | D1/D2 | source-averaged transfer; `predict.effect: additive`=D1, `logfc`=D2 |
| `gwps_nearest`  | D3 | copy the most-similar source's effect |
| `gwps_weighted` | D4 | similarity-weighted K562↔RPE1 blend (raw/PCA sim) |
| `gwps_weighted` + `similarity_space: embed` | F6/F7 | same, but similarity from STATE-SE / STACK embeddings |
| `esm2_knn`      | D6 | unseen gene → K nearest measured genes by ESM-2 |
| `state` / `stack` | F4/F5/F1 | frozen FM direct (precomputed prediction via `--fm-pred`) |

Knobs (`predict:` block): `effect` (logfc/additive), `generator` (multinomial/poisson/nbinom = G0/G1/G2), `scale` (D2b logFC magnitude sweep `{0.5,0.75,1,1.5,2}`), `shrink` (D5: context/targetexpr/both), `similarity_space` (raw/pca/embed), `tau`, `knn_k`, `cells_per_pert`.

**Fair comparison rule:** to compare effect methods, keep `generator`, `cells_per_pert`, and `seed` fixed; to study generators, fix the effect method and sweep `generator` only.

---

## 5. Metric analysis — exact VCC 2026 suite, ladder, coverage

```bash
python -m vcc_baselines ladder   --config CFG --methods no_effect,gwps_direct,gwps_weighted,esm2_knn
python -m vcc_baselines ladder   --config CFG --engine cell-eval2       # exact public vcc2026 six metrics
python -m vcc_baselines coverage --config CFG --method gwps_weighted     # per GWPS-coverage-group reach
```

- **Local engine** = fast approximate metrics (pdisc, MAE, pearson_delta, DE overlap) for tight loops.
- **cell-eval2 engine** = Arc's official public VCC 2026 implementation (`cell-eval2==0.16.0`, preset `vcc2026`): `pds`, `mse`, `nmae`, `fid`, `reach`, `jac`, plus their unweighted `overall`. External datasets use their own measured baseline and five-split replicate anchors, so their values are not leaderboard scores.
- **cell-eval engine** is retained only for legacy proxy diagnostics. Its PDS/MAE/MSE/PearsonDelta/DirMatch/OverlapN fingerprint is not a VCC 2026 score.
- **Worst-context** is reported alongside the mean — the number that matters for 2026 (final contexts differ from validation).
- **Coverage stratification** separates the DB approach's reach (covered genes) from the ESM-2 fallback (missing genes).

> Exact official Challenge scores still require the server's held-out truth and stamped panel/anchor bundle. The external runner uses the exact same public metric implementation but locally measured anchors and marks `official_challenge_score: false` in every audit.

---

## 6. Real-data shadow evaluation (no challenge validation)

Use `prepare-shadow` to create a leakage-resistant leave-one-context-out split: all perturbation responses from the holdout are written to a separate truth file, while prediction receives only its non-targeting controls. The primary public cell-line test is Jiang24/GSE281048 (one signaling condition, one cancer line held out); the ready iPSC→day-7-neuron split is a deliberately stronger state-shift sanity check.

```bash
# Primary cancer-cell-line LOCO benchmark (15 GB download; ~88 GB temporary expansion):
JIANG_H5AD_OUT=/large/tmp/jiang24_processed.h5ad \
  bash scripts/download_shadow_data.sh data_public jiang24
bash scripts/run_jiang24_loco.sh /large/tmp/jiang24_processed.h5ad
bash scripts/run_frozen_contexts.sh data_shadow/jiang24_ifng_bxpc3_loco

# Complementary differentiation/state-shift sanity check:
bash scripts/download_shadow_data.sh data_public tian
bash scripts/run_shadow_real.sh
```

The completed Jiang24 recipe uses raw-count IFNG data, holds out all perturbed BxPC3 cells, transfers from five other lines, and evaluates 56 shared targets over 4,017 genes. Raw nearest transfer reached public-proxy PDS 0.5794; STATE and STACK were run from their real frozen checkpoints but did not beat it on this split. The generated `manifest.json` records paths, filters, QC, matrix origin and leakage assertions. Missing frozen outputs are reported as unavailable and never silently replaced. These are public proxy scores, not the challenge validation score. See [`docs/SHADOW_VCC.md`](SHADOW_VCC.md) for exact hashes, commands, split contract, results, and metric interpretation.

For **complete response-sealed zero-shot** (no perturbation response from any batch or cell line of either evaluation dataset), run:

```bash
bash scripts/download_shadow_data.sh data_public gse270828
bash scripts/run_two_dataset_zero_shot.sh
```

This compares Jiang24 IFNG/BxPC3 with GSE270828 H23555 iPSC-derived NSCs under the same blind contract. GSE270828 perturbs HAR regulatory elements, not coding genes, so HAR→nearby-gene annotations are deliberately excluded. STATE-SE and STACK are scored only if a target-specific predictor is eligible under the contract; encoder-only, response-prompt-dependent, or training-overlap-unknown artifacts are recorded as excluded rather than disguised as predictors. See [`docs/STRICT_ZERO_SHOT.md`](STRICT_ZERO_SHOT.md).

To inspect how all six VCC 2026 components change across the existing GWPS methods and frozen STATE/STACK context representations, use the explicitly non-strict two-dataset shadow matrix. It calls `cell-eval2` directly, is resumable, and prints nested terminal progress:

```bash
bash scripts/run_two_dataset_full_matrix.sh --acknowledge-non-strict --preflight-only
bash scripts/run_two_dataset_full_matrix.sh --acknowledge-non-strict
```

See [`docs/TWO_DATASET_SIX_METRIC.md`](TWO_DATASET_SIX_METRIC.md) for the exact 22 rows, progress-monitoring commands, output files, and the strict-vs-shadow distinction.

---

## 7. Frozen foundation models

`python -m vcc_baselines fm-cmd --method stack` prints the exact external command. Run STATE/STACK yourself, save the prediction h5ad, then:

```bash
python -m vcc_baselines predict --config CFG --method stack --fm-pred stack_pred.h5ad --prep
```

For **STATE-SE / STACK as context encoders** (F6/F7 — the cleanest frozen FM baseline), export per-context embeddings (`state emb transform` / `stack-embedding`) into one npz keyed by context/source name, set `embed.context_embeddings` and `predict.similarity_space: embed`. See [`docs/STATE_STACK.md`](STATE_STACK.md) (incl. the STATE one-hot→ESM-2 fix and why SE-context beats ST-direct as a first baseline).

The independent external-model experiment adds UCE-4L, TranscriptFormer-Sapiens-cell, scGPT,
scFoundation, and scPRINT-2 without changing the STATE/STACK runner. Its default track allows only
Replogle K562/RPE1 perturbation responses, performs Replogle-only magnitude selection, and evaluates
both external datasets with the exact `cell-eval2==0.16.0` `vcc2026` six-component scorer:

```bash
# Install only the requested model environment/checkpoint, then verify it.
bash scripts/setup_external_frozen_models.sh gene-map
bash scripts/setup_external_frozen_models.sh replogle-data
bash scripts/setup_external_frozen_models.sh uce
bash scripts/run_two_dataset_external_frozen_matrix.sh --models=uce --preflight-only

# Full resumable embedding + prediction + exact six-metric run.
bash scripts/run_two_dataset_external_frozen_matrix.sh --models=uce
```

See [`docs/EXTERNAL_FROZEN_MODELS.md`](EXTERNAL_FROZEN_MODELS.md) for all model setup commands,
artifact paths, progress monitoring, strict/non-strict protocol labels, and the GSE270828 HAR
negative-control limitation.

---

## 8. Config & layout

`configs/synthetic.yaml` (offline) and `configs/real_example.yaml` (real) document every field. Key blocks: `data` (paths, columns), `gwps` (Replogle sources), `predict` (method/effect/generator/similarity/scale/shrink), `submit` (gene_dim 18533, counts, reject-controls, packer), `eval` (profile).

```
src/vcc_baselines/  config io effect_library methods generators context_embed
                    embeddings predict submit evaluate metrics_local ladder synthetic cli adapters/
configs/  synthetic.yaml real_example.yaml
tests/    test_smoke.py test_shadow.py   # raw-count, leakage, compaction, determinism
docs/     STATE_STACK.md SHADOW_VCC.md STRICT_ZERO_SHOT.md
```

## 9. Troubleshooting

- **`zstd is not installed`** → `apt-get install -y zstd` (needed by vcc prep).
- **`vcc: command not found`** → `pip install vcc-cli`.
- **`gene dimension mismatch`** → set `submit.gene_dim` to `len(gene_names.csv)` (18533 real).
- **`values are not integer counts` / require-counts** → 2026 needs raw counts; keep `predict.clip_negative: true` and a count generator (multinomial/poisson/nbinom). Don't log-normalize.
- **`submitted control cells`** → predictions must exclude controls (`predict.include_controls: false`, the default for submission; the ladder re-adds them only for local DE scoring).
- **cell-eval `full` slow** → use `eval.profile: vcc` for a quick 3-metric check.
- **Jiang24 full H5AD is too large** → use `compact-shadow` through `scripts/run_jiang24_loco.sh`; it streams the selected raw-count layer instead of asking AnnData to load every layer.
- **STATE cannot find the official SE-100M checkpoint** → use `scripts/state_safetensors_embed.py`; the wrapper strictly loads the release's `model.safetensors` and refuses partial/random weights.

MIT licensed. Wraps Arc Institute's open-source `vcc-cli`, `cell-eval`, `state`, `stack`; not affiliated with Arc.

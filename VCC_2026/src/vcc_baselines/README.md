<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [VCC_2026](../../README.md) › **src/vcc_baselines**</sub>

# 💻 `vcc_baselines` — the package

A pip-installable package (`pip install -e .` from `01_Medical/VCC_2026`) with one CLI, `python -m vcc_baselines <command>` (alias `vcc-baselines`). One YAML config drives every command; see [`../../configs/`](../../configs/README.md).

## Module map, in pipeline order

| Stage | Module | What it does |
|---|---|---|
| Config & I/O | [`config.py`](config.py) | Dataclass schema with the real 2026 defaults (raw integer counts, 18,533 genes, 400 cells per perturbation, no controls in the submission) |
| | [`io.py`](io.py) | Gene lists, target lists, gene-panel alignment, loading controls / truth, saving predictions |
| Effect library | [`effect_library.py`](effect_library.py) | Per-source relative fold (perturbed mean / control mean) for every knocked-down gene, with additive / log-fold views and coverage groups |
| | [`replogle.py`](replogle.py) | Strict Replogle-only track: audited raw-pseudobulk loading (hashes, duplicate loci), native ln-CPM fold changes, symmetric cross-line CV, projection onto a benchmark's gene panel |
| Context representation | [`context_embed.py`](context_embed.py) | Cosine similarity of raw pseudobulk, PCA or precomputed frozen embeddings, then softmax(τ) weights over source contexts |
| | [`external_frozen.py`](external_frozen.py) | Registry, preflight, command adapters and pooling for UCE, TranscriptFormer, scGPT, scFoundation, scPRINT-2; fails closed as `AUDIT_REQUIRED` instead of substituting weights |
| | [`adapters/`](adapters/__init__.py) | Turns precomputed STATE / STACK prediction files into per-gene effects that reuse the same generate → `vcc prep` path |
| | [`embeddings.py`](embeddings.py) | ESM-2 gene embeddings for the nearest-gene fallback (a deterministic random fallback exists only for offline smoke tests) |
| Effect prediction | [`methods.py`](methods.py) | The method ladder: `no_effect`, `global_mean`, `replogle_k562`, `gwps_direct`, `gwps_nearest`, `gwps_weighted`, `esm2_knn`, plus `state` / `stack` adapters |
| Count generation | [`generators.py`](generators.py) | Raw integer cells from an effect: multinomial (keeps each control cell's library size), Poisson, negative binomial, pooled pseudobulk multinomial |
| | [`predict.py`](predict.py) | Assembles one context's prediction (scale / shrink knobs, per-target cell counts, cap on total cells) |
| Submission | [`submit.py`](submit.py) | Packs with the official `vcc prep` (18,533 genes, 400 cells per target, integer counts, controls rejected, per-context target check) |
| Evaluation | [`shadow.py`](shadow.py) | Leakage-resistant benchmarks: `compact-shadow`, `prepare-shadow` (LOCO), `prepare-zero-shot` / `score-zero-shot` (response-sealed), `shadow-benchmark`, `pool-embeddings` |
| | [`vcc2026_eval.py`](vcc2026_eval.py) | Exact 2026 scoring through Arc's `cell-eval2` `vcc2026` preset with dataset-local anchors; records hashes and `official_challenge_score: false` |
| | [`evaluate.py`](evaluate.py), [`metrics_local.py`](metrics_local.py) | Legacy `cell-eval` proxy fingerprint (PDS / MAE / …) and fast local approximations for quick iteration |
| | [`ladder.py`](ladder.py) | Method ladder over contexts (mean and worst context) and coverage stratification |
| Utilities | [`synthetic.py`](synthetic.py), [`seeds.py`](seeds.py), [`cli.py`](cli.py) | Synthetic VCC-shaped data, process-stable SHA-256 seeds, the command-line interface |

## Commands

| Command | Purpose |
|---|---|
| `make-synth` | Generate synthetic raw-count data (offline demo) |
| `predict` | Build a raw-count prediction for every context; `--prep` / `--dry-run` runs `vcc prep` |
| `prep`, `sample` | Wrap `vcc prep` and `vcc sample` (Arc's random valid dummy) |
| `evaluate`, `ladder`, `coverage` | Score predictions, compare methods, split by Replogle coverage |
| `compact-shadow`, `prepare-shadow`, `shadow-benchmark`, `pool-embeddings` | Real-data leave-one-context-out benchmark ([E1](../../experiments/E1_jiang24_bxpc3_proxy/README.md)) |
| `prepare-zero-shot`, `score-zero-shot` | Response-sealed benchmark: prediction sees only `blind/`, truth lives in `sealed/` |
| `fm-cmd` | Print the external STATE / STACK command to run |

Design rule: to compare effect methods, keep the generator, cell count and seed fixed; to compare generators, keep the effect method fixed. Tests live in [`../../tests/`](../../tests/README.md).

<sub>Provenance: written for this project (MIT, see [`LICENSE`](../../LICENSE)). It wraps Arc Institute's open-source `vcc-cli`, `cell-eval`, `cell-eval2`, STATE and STACK and is not affiliated with Arc. No third-party code is vendored.</sub>

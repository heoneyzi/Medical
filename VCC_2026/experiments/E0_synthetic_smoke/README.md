<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [VCC_2026](../../README.md) › **E0 · Synthetic smoke test**</sub>

# 🔬 E0 · Offline smoke test and metric ladder (synthetic data)

> **Question —** Does the whole pipeline run end-to-end on raw counts, emit a leaderboard-valid `.vcc` file, and do the method rungs separate the way they should?

| | |
|---|---|
| **Status** | ✅ done — stored outputs from the source folder (Aug 2026); re-run for this portfolio on 28 Sep 2026 |
| **Model / data** | Synthetic VCC-shaped raw counts from `make-synth`: 400 genes, contexts A/B/C, 40 targets, 50 predicted cells per target; two synthetic reference "screens" (`k562`, `rpe1`) |
| **Compute** | CPU only, seconds; no downloads |
| **Headline** | Discrimination rank (mean over contexts, lower = better): `no_effect` **0.4964** → `gwps_direct` **0.1141** |

## Setup

`bash scripts/quickstart.sh` runs, in order: `make-synth` → `ladder` with the fast local metrics → `coverage` → `predict --dry-run` and `predict --prep` (official `vcc prep` packing) → `vcc sample` (Arc's random valid dummy). The synthetic generator plants a shared per-gene effect plus context-specific noise, so a method that transfers measured effects should discriminate targets far better than doing nothing. 28 of the 40 targets are guaranteed to be measured in both reference screens and 4 more are covered by chance, leaving 8 unmeasured.

Local metrics (`src/vcc_baselines/metrics_local.py`) are quick approximations for iteration, not challenge scores: `pdisc_norm_rank` (0 = each prediction is closest to its own truth, 0.5 = random), pseudobulk MAE, Pearson correlation of predicted vs true change from control, and top-100 changed-gene overlap. In this config `esm2_knn` uses deterministic random gene embeddings (plumbing only, no biology).

## Results

Mean over the three contexts, worst context in brackets (`results/ladder_results.csv`):

| Method | pdisc rank ↓ (worst) | MAE ↓ | Pearson Δ ↑ | DE overlap@100 ↑ |
|---|---:|---:|---:|---:|
| `no_effect` | 0.4964 (0.5096) | 0.3969 | −0.0033 | 0.2807 |
| `global_mean` | 0.5064 (0.5276) | 0.4099 | −0.0022 | 0.2807 |
| `gwps_direct` | **0.1141** (0.1199) | 0.4120 | 0.3258 | 0.3134 |
| `gwps_nearest` | 0.1160 (0.1250) | 0.4475 | 0.3021 | 0.3150 |
| `gwps_weighted` | 0.1199 (0.1321) | 0.4136 | 0.3351 | 0.3093 |
| `esm2_knn` | 0.1171 (0.1237) | 0.4207 | **0.3360** | 0.3098 |

Coverage stratification for `gwps_weighted` (`results/coverage_report.csv`): targets measured in both reference screens reach a discrimination rank of **0.0101 / 0.0333 / 0.0111** (contexts A / B / C, 32 targets), while the 8 unmeasured targets stay near random at **0.5357 / 0.5357 / 0.5179**.

**Reproducibility check (28 Sep 2026).** Re-running `make-synth` + `ladder` in a fresh Python 3.11 environment reproduced `pdisc_norm_rank`, MAE and Pearson Δ to every stored digit; `de_overlap@100` differed only in the fourth decimal (tie-breaking in the top-100 selection). All 22 unit tests also passed.

## Takeaway

- The plumbing works end-to-end in counts space, and `vcc prep` accepts the output, so the same code path produces real submissions.
- Transfer methods cut the discrimination rank from ~0.50 to ~0.11 while *raising* pseudobulk MAE (0.3969 → 0.4120): the "MAE trap" that made 2025 do-nothing baselines look strong, reproduced on toy data.
- Coverage decides reach: measured targets are nearly perfectly discriminated, unmeasured ones are not. On real data this is why the Replogle coverage of the challenge targets matters so much.
- Synthetic data proves correctness, not biology; real-data evidence is in [E1](../E1_jiang24_bxpc3_proxy/README.md)–[E3](../E3_replogle_only_strict/README.md).

## Files

| File | What it is |
|---|---|
| `results/ladder_results.csv` | Ladder table, mean and worst context per method (stored output of `python -m vcc_baselines ladder --config configs/synthetic.yaml`) |
| `results/coverage_report.csv` | Per-context metrics split into measured (`B_multi_source`) and unmeasured (`D_missing`) targets |
| [`../../configs/synthetic.yaml`](../../configs/synthetic.yaml) | Offline config used by the quickstart |
| [`../../scripts/quickstart.sh`](../../scripts/quickstart.sh) | The full smoke run, including `vcc prep` and `vcc sample` |

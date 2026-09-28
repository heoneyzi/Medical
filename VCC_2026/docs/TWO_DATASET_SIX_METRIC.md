# Two external datasets with the exact VCC 2026 metric implementation

This runner evaluates 11 method/representation rows on Jiang24 and GSE270828. It calls Arc
Institute's public `cell-eval2==0.16.0` directly with the packaged `vcc2026` preset. There is no
locally reimplemented metric and the old `cell-eval` six-axis proxy is not used.

## What is exact, and what is local

The six enrolled components are exactly:

| Challenge column | `cell-eval2` member |
|---|---|
| `pds` | `pds_cosine` |
| `mse` | `expr_mse_unbiased_capped_norm` |
| `nmae` | `de_wilcoxon_lfc_nmae` |
| `fid` | `de_wilcoxon_direction_fidelity_yield_raw` |
| `reach` | `de_wilcoxon_direction_reach_raw` |
| `jac` | `de_wilcoxon_sig_jaccard` |

`overall` is their exact unweighted mean. The `vcc2026` preset also fixes counts input, target-gene
exclusion, group-sum expression normalization, Wilcoxon DE, the 5 CPM control-only filter,
per-perturbation BH at 0.05, the direction-reach purity floor, and every scoring clamp.

For each external dataset, `cell-eval2` constructs the two scale ends from that dataset:

- 0: the generic mean-response baseline;
- 1: the mean of five seeded split-half replicate runs (`base_seed=0`).

Therefore these are exact VCC 2026 metric calculations with **external local anchors**. They are
not official Challenge leaderboard scores: the official server uses held-out Challenge truth and
its stamped panel/anchor bundles. Jiang and GSE scores must not be compared as if they shared one
anchor set. Compare method changes within each dataset.

## Matrix and split contract

Each dataset produces 11 rows:

| representation | methods |
|---|---|
| raw control pseudobulk | no_effect, global_mean, gwps_direct, gwps_nearest, gwps_weighted |
| PCA control state | gwps_nearest, gwps_weighted |
| frozen STATE-SE-100M control embedding | gwps_nearest, gwps_weighted |
| frozen STACK-Large control embedding | gwps_nearest, gwps_weighted |

STATE-SE and STACK are frozen context encoders here; the target-specific response still comes from
the response library. The splits are Jiang24 A549/HAP1/HT29/K562/MCF7 → held-out BxPC3, and
GSE270828 rep1/rep2 → held-out rep3. Because same-dataset response references are used, this matrix
is a transfer diagnostic and not complete response-sealed zero-shot. The acknowledgement flag is
mandatory.

GSE270828 perturbations are HAR identifiers rather than gene symbols. The evaluator reads the
published `GSE270828_feature_README.csv` label→target-gene mapping to retain those genes on the
output axis and apply the official exclusion rule. The response/effect estimator never consumes
the map. Its hash and evaluator-only role are recorded in the scale manifest.

## Install and preflight

```bash
cd 01_Medical/VCC_2026    # project root
uv pip install --python .venv/bin/python --torch-backend=auto \
  'cell-eval2[gpu,gpudge]==0.16.0'

bash scripts/run_two_dataset_full_matrix.sh \
  --acknowledge-non-strict --preflight-only
```

Preflight verifies both raw-count H5ADs, both frozen checkpoints, `cell-eval2 0.16.0`, gpudge,
CUDA, STATE and STACK. It consumes no Challenge submission quota.

## Run, resume, and monitor

```bash
cd 01_Medical/VCC_2026    # project root
bash scripts/run_two_dataset_full_matrix.sh --acknowledge-non-strict
```

Repeat the same command after an interruption. Frozen embeddings, completed local scales, and rows
already marked `ok` are reused after provenance checks.

From another terminal:

```bash
cd 01_Medical/VCC_2026    # project root
tail -f data_shadow/two_dataset_vcc2026_metrics/progress.tsv
tail -f data_shadow/two_dataset_vcc2026_metrics/logs/latest.log

watch -n 3 '.venv/bin/python -c "import pandas as p; from pathlib import Path; [print(x, p.read_csv(x)[[\"representation\",\"method\",\"status\"]].to_string(index=False)) for x in map(Path, [\"data_shadow/two_dataset_vcc2026_metrics/jiang24_vcc2026.csv\", \"data_shadow/two_dataset_vcc2026_metrics/gse270828_vcc2026.csv\"]) if x.exists()]"'
```

Outputs:

```text
data_shadow/two_dataset_vcc2026_metrics/jiang24_vcc2026.csv
data_shadow/two_dataset_vcc2026_metrics/gse270828_vcc2026.csv
data_shadow/two_dataset_vcc2026_metrics/comparison_wide.csv
data_shadow/two_dataset_vcc2026_metrics/metric_changes_long.csv
```

Every method directory contains the official raw aggregate, locally anchored scores, and an
`evaluation_audit.json`. Each dataset scale contains `baseline/`, `anchor/`, the exact resolved
YAML, strict input hashes, the scorer version, competition-rule digest, backend, GPU, and the
explicit marker `official_challenge_score: false`.

The earlier `data_shadow/two_dataset_six_metric/*_full.csv` files are legacy public-proxy results.
They must not be reported as VCC 2026 component scores.

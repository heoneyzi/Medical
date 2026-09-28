<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GDTR](../../README.md) › [Line C · Handoff](../README.md) › **EXP3**</sub>

# 🔬 C3 · EXP3 — Using the late-stack structure: design and dry run

> **Question —** Taking EXP2's writer → re-encoder → readout structure as fixed, can it be *used* — to trace output effects back to the input sequence, validate motif and variant candidates, explain benchmark differences, and generalise across chromosomes and checkpoints?

| | |
|---|---|
| **Status** | 🔄 designed; dry run passes on a toy runtime (`scientific_evidence: false`); no real-model results yet |
| **Model / data** | planned: Evo 2 checkpoints via the Vortex runtime, EXP2's sealed manifests and splits (read through `exp3/exp2_api.py` only) |
| **Compute** | CPU for tests and dry run; GPU runs not started |
| **Headline** | 8-phase adaptive program with pre-sealed confirmatory plans; 38/38 tests pass (re-run on CPU for this portfolio) |

## Design

| Phase | New question | Module |
|---|---|---|
| 1 | What prepares b28's write; is b29 an amplifier or a second writer; where does the scale plateau end? | `mechanism.py`, `bilinear_pairing.py` |
| 2 | Do scale, content and carrier play separable causal roles? | `causal_use.py` |
| 3 | Can output effects be traced back through b30 → b29 → b28 to input positions? | `reverse_trace.py` |
| 4 | Do candidate motifs/variants reproduce the causal chain when implanted? | `motifs.py` |
| 5 | Can a learned model predict and rescue the observed causal deltas? | `learned_transport.py` |
| 6 | Is the mechanism useful for benchmark explanation, error diagnosis, repair, steering? | `applications.py` |
| 7 | Does it hold across chromosomes, tasks, checkpoints and training time? | `generalization.py` |
| 8 | Only after raw causal results: do SAEs/transcoders compress the interpretation? | `optional_features.py` |

An adaptive router (`adaptive.py`) decides *which* analysis runs next from earlier outcomes, but every node's estimand, decision rule, thresholds, hyper-parameters, controls and data roles must be declared in a `ConfirmatoryAnalysisPlan` and sealed with SHA-256 before locked data are opened (`analysis_plan.py`, `config.py`); a changed config fails to load.

## Results (dry run)

[`code/exp3_dryrun_report.json`](code/exp3_dryrun_report.json): status `pass`; all eight phases reached in both the "common" and "alternative" scenarios; amplifier, alternative-b29, site-specific and branch-trace routes each selected once; EXP2 steps not re-executed. This checks orchestration only — it is **not** scientific evidence.

## Takeaway

- EXP3 turns the EXP2 findings into testable uses, with the analysis space fixed in advance to avoid forking-path results.
- Nothing here should be read as a result yet; it is included to show how the next stage is engineered.

## Files

| File | What it is |
|---|---|
| [`code/exp3/`](code/exp3/) | the package (phases, router, sealed configs, toy runtime, EXP2 API boundary) |
| [`code/tests/`](code/tests/) | 38 tests (protocol, analysis plan, applications, infrastructure, transport, mechanism, orchestration, reverse trace) |
| [`code/configs/`](code/configs/) | example run config and confirmatory-design template |
| [`code/README.md`](code/README.md), [`code/RUNBOOK.md`](code/RUNBOOK.md), [`code/RESEARCH_PROGRAM.md`](code/RESEARCH_PROGRAM.md) | original design documents (Korean); local paths replaced by `/path/to/…` |

Run the tests: `cd code && PYTHONPATH=../../exp2/code:. python -m pytest -q tests`.

---
<sub>[← EXP2](../exp2/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [↑ GDTR program](../../README.md)</sub>

<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GDTR](../../README.md) › [Line C · Handoff](../README.md) › **EXP3**</sub>

# 🔬 C3 · EXP3 — Using the late-stack structure: design and dry run

> **Question —** Taking EXP2's writer → re-encoder → readout structure as fixed, can it be *used* — to trace output effects back to the input sequence, validate motif and variant candidates, explain benchmark differences, and generalise across chromosomes and checkpoints?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

EXP3는 EXP2에서 확정한 늦은 층 구조를 출발점으로 삼아, 그 관점이 실제로 쓸모 있는지(출력 효과를 입력 서열까지 거꾸로 추적, 모티프·변이 검증, 벤치마크 차이 설명 등)를 검증하는 8단계 연구 설계입니다.
결과를 본 뒤 분석을 고르는 "갈림길의 함정"을 막기 위해, 가능한 모든 분기의 판정 기준·임계값·대조군을 데이터를 열기 전에 선언하고 해시로 봉인하도록 코드가 강제합니다.
시험 문제를 보기 전에 채점 기준표를 밀봉해 두는 것과 같습니다.
아직 실제 모델로 돌린 과학적 결과는 없고, 장난감 런타임으로 모든 단계가 도달 가능한지 확인한 dry run과 테스트 38개(통과)만 있습니다.

</details>

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

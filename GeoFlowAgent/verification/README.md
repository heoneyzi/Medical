<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **verification**</sub>

# ✅ Verification — recomputing the headline numbers

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

README와 안내 문서의 수치를 저장된 결과에서 다시 계산하는 스크립트입니다. `verify_hard_v2.py`는 hard-v2 test 수치를, `verify_portfolio.py`는 GeoACMG 7 시드 비교와 MedCPT 제거 실험의 신뢰구간까지 run별 파일에서 다시 계산하고, 모든 페이지에 적힌 소수 수치를 원본 값과 대조합니다. margin 배열이 필요한 두 검증기의 결과(276/276, 233/233 통과)도 함께 저장해 두었습니다. 단위 테스트는 295개 통과, 1개 선택 테스트 건너뜀입니다.

</details>

## Runs in this repository (NumPy only, a few seconds)

| Command | What it recomputes | Result |
|---|---|---|
| `python verification/verify_hard_v2.py` | hard-v2 3-seed test means, paired task-macro deltas vs the reference, State Flow means — from the per-seed JSON in `results/hard_v2/final_reports/` | PASS ([saved output](hard_v2_verified_metrics.json)) |
| `python verification/verify_portfolio.py` | GeoACMG 7-seed planning contrasts (R3c) and MedCPT ablation (R11 and the 8-run aggregate): estimates **and** seed/run- and gene-clustered BCa intervals from the per-run files; R1b and 15-run means; R7 and B0 arithmetic; cross-check of the margin-based numbers against the saved verification outputs; and a two-way check of every decimal printed in the READMEs and the Korean guide | PASS |

Gene clusters come from the task-id prefix (`GENE:uuid`), which reproduces the gene labels of the margin arrays exactly; every recomputed interval matches the stored one to within 1e-7.

## Verifiers that read the readout-margin arrays

The margin arrays (51 MB and 74 MB) are kept with the large artifacts of the local research archive; the outputs of these two verifiers are saved here.

| Script | What it recomputes | Saved result |
|---|---|---|
| [`verify_geoacmg_metrics.py`](verify_geoacmg_metrics.py) | R1–R4 of stage ④ from `pairs/R2_readout_margins.npz` | **276/276** checks PASS — [`geoacmg_verified_metrics.json`](geoacmg_verified_metrics.json) |
| [`verify_geoacmg_latest.py`](verify_geoacmg_latest.py) | 7-seed readouts, R10 and R11 of stage ⑤ from the extended margin file | **233/233** checks PASS — [`geoacmg_latest_verified_metrics.json`](geoacmg_latest_verified_metrics.json) |

## Unit tests

`PYTHONPATH=src pytest` → **295 passed, 1 skipped** (Linux CPU, Python 3.11; the same result on a fresh Windows CPU environment with Python 3.12 using `python -X utf8 -m pytest`, see [`qa_environment.txt`](qa_environment.txt), [`unit_test_output.txt`](unit_test_output.txt) and [`unit_test_validation.json`](unit_test_validation.json)). The skipped test runs when the optional generated search-smoke artifacts are present.

[`static_audit.json`](static_audit.json): 0 syntax errors and 0 high-confidence credential patterns across the text files of the research code.

## What each check establishes

These scripts recompute statistics from saved outputs; re-running the GPU experiments uses the steps in [docs/research/05_REPRODUCIBILITY.md](../docs/research/05_REPRODUCIBILITY.md). For the hard-v2 paired deltas and the R7 flow effects, the point estimates are recomputed and the intervals are the stored task-bootstrap results; R8/R9 readout intervals are the stored results of the late analyses.

---
<sub>[← GeoFlowAgent](../README.md) · [🏠 Portfolio](https://github.com/heoneyzi)</sub>

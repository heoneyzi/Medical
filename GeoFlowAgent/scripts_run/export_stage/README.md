<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GeoFlowAgent](../../README.md) › [scripts_run](../README.md) › **export_stage**</sub>

# 🧾 Finishing scripts of the main GeoACMG analysis (R1–R4, R7)

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

④단계 주요 분석(R1–R4, R7)을 마무리한 스크립트입니다. 함께 쓰인 라이브러리 모듈과 실행기는 `src/geoflowagent/geoacmg/`와 상위 폴더에 있습니다.

</details>


The scripts that finished the main GeoACMG analysis. The modules and runners they call (`adapters.py`, `flowfast_eval.py`, `stages.py`, `r1_paired_seeds.sh`, `train_flow_rootonly_v2.sh`) are in `src/geoflowagent/geoacmg/` and `../`.

| Script | What it does | Output |
|---|---|---|
| `r1_finish.py` | Orderings for the two new seed pairs (59/71) and the 5-seed paired contrast with zero energy parameters | `R1_paired_zero_param.json` |
| `r234_pass.py` | One pass over checkpoints to re-extract latent states: readout decomposition, horizon strata, cross-model control | `R2_readout_decomposition.json`, `R3_horizon_strata.json`, `R4_crossmodel_control.json`, `c5_within_model.json` |
| `validate_real.py` | Checks the batched flow sampler against the original on real dev roots and measures the speed-up | `R7_flowfast_validation.json` |
| `run_batched_eval.py` | Equivalence check on 24 dev roots (59 metrics within 1e-9), then the full 584-root dev evaluation | `R7_batched_equivalence.json`, `R7_flow_dev_root_only.json` |

All outputs are in [`results/geoacmg_final/findings/`](../../results/geoacmg_final/findings/). `$GEOWORK` marks the scratch directory.

<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **scripts**</sub>

# 🏃 `scripts/` — runners for stages ①–③

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

①–③단계(합성 파일럿 ~ hard-v2)의 실험 실행 스크립트입니다. 절대 경로는 자리표시자로 바꿨습니다.

</details>


The experiment runners used on the GPU machine — the exact commands and order of operations of stages ①–③. They call the library CLI (`python -m geoflowagent.cli …`).

| Script | Stage | What it runs |
|---|---|---|
| `run_smoke.sh`, `run_search_smoke.sh` | ①–② | CPU smoke runs on the bundled synthetic fixtures |
| `run_distance_sweep.sh` | ① | Six-family `compare-distances` sweep (smoke config) |
| `run_hard_v2_development.sh`, `continue_hard_v2_development.sh` | ③ | hard-v2 development: exact search, embeddings, geometry and input-ablation queues on two GPUs |
| `continue_state_flow_seed_replicates.sh` | ③ | State Flow seeds 29/43 (task-root evaluation) |
| `hard_v2_medcpt_followup.sh` | ③ | MedCPT-only geometry sweep, capacity curve and closed loop |
| `run_hard_v2_final_once.sh` | ③ | **One-time final test**: materialise, encode and evaluate the frozen candidates with an explicit test-access reason; it starts only after every dev evaluation has finished |
| `evaluate_final_value_checkpoints.py` | ③ | Test metrics for the frozen value checkpoints (writes `final_reports/value_test/`) |
| `normalize_final_cache_superset.py` | ③ | Registry/cache superset normalisation for the final package |


**Paths.** Machine paths are placeholders: `$GEOFLOW_RUNS` (run root), `$GEOFLOW_RESULTS` (result folder), `$GEOFLOW_RUNTIME` (Python environment), `$HF_CACHE_DIR` (Hugging Face cache), `$GEOWORK` (scratch/log directory). Set them — or the scripts' own `GEOFLOW_*` override variables — before running; see [docs/research/05_REPRODUCIBILITY.md](../docs/research/05_REPRODUCIBILITY.md) for the inputs they expect.

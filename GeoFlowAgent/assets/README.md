<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **assets**</sub>

# 🖼️ Figures

All figures are drawn by [`make_figures.py`](make_figures.py) directly from the saved result JSON in [`../results/`](../results/README.md). Regenerate them with `python assets/make_figures.py` (NumPy + matplotlib).

| Figure | Shows | Source files |
|---|---|---|
| `hero.png` | Goal completion by planning mode (hard-v2 test, ClinGen development tasks) and 7-seed cosine − Euclidean contrasts | `hard_v2/final_reports/state_flow_seed*_test.json`, `geoacmg_final/findings/R7_flow_dev_root_only.json`, `geoacmg_working_findings/R2f_*.json`, `R3c_*.json` |
| `hard_v2_test.png` | hard-v2 one-time test: state-level metrics vs the reference; State Flow modes with per-seed dots | `hard_v2/final_reports/value_test/summary.json`, `state_flow_seed*_test.json` |
| `geoacmg_dev_dissociation.png` | Raw vs learned ordering accuracy; development-split policy accuracy per geometry | `geoacmg_final/findings/R2_readout_decomposition.json`, `geoacmg_latest_checkpoints/geometry_comparison/*/seed-*/value_metrics.json` |
| `geoacmg_7seed_contrasts.png` | Seven-seed contrasts with seed- and gene-clustered CIs | `geoacmg_working_findings/R2f_readout_7seeds.json`, `R3c_planning_axis_7seeds.json` |
| `geoacmg_readout_ranking.png` | Five geometries × six readouts (best per readout outlined) | `geoacmg_working_findings/R10_readout_dependent_ranking.json` |
| `geoacmg_medcpt_ablation.png` | Effect of zeroing the MedCPT view (run- and gene-clustered CIs) | `geoacmg_working_findings/R11_medcpt_ablation.json` |

Colours: one highlight hue (blue) plus neutral greys, and blue/orange only where two series are compared; the pair passes CVD and normal-vision separation checks. Identity is also carried by labels or marker shape.

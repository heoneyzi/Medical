<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › **assets**</sub>

# 🖼️ assets

All figures were made for this portfolio with [`make_figures.py`](make_figures.py) (`python assets/make_figures.py` from the project root). Plotted values are read from the experiment CSVs; only the hero diagram is laid out by hand from the code structure.

| Figure | Content | Data source |
|---|---|---|
| [`hero.png`](hero.png) | Pipeline + evaluation schematic and the headline proxy result | `src/vcc_baselines/`; `experiments/E1_jiang24_bxpc3_proxy/results/bxpc3_cell_eval_proxy.csv` (from `docs/SHADOW_VCC.md`) |
| [`bxpc3_proxy_pds.png`](bxpc3_proxy_pds.png) | Public-proxy PDS and local discrimination rank for every BxPC3 row | E1 results |
| [`cells_per_pert_tradeoff.png`](cells_per_pert_tradeoff.png) | NMAE, Jaccard and Overall vs predicted cells per perturbation (Jiang24) | E2 `jiang24_cells_per_pert.csv` |
| [`replogle_scale_sweep.png`](replogle_scale_sweep.png) | Internal cross-line CV error vs effect scale | E3 `internal_cv/magnitude_sweep.csv` |

Colours follow a colour-blind-checked categorical palette (blue / orange), with grey for de-emphasised rows.

<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [Bu-net](../../README.md) › [Code](../README.md) › **reinforcing_material**</sub>

# `reinforcing_material/` — restored research archive + viewer

Snapshot of [heoneyzi/reinforcing_material](https://github.com/heoneyzi/reinforcing_material) at `4093e1f`. Named after the team (보강재, "reinforcing material"). In Sep 2026 Jiheon restored the 2024 notebooks from the project folder (personal paths replaced, outputs cleared) and added a runnable layer around them; his original README is kept as [ORIGINAL_README.md](ORIGINAL_README.md).

| Path | Origin | Contents |
|---|---|---|
| [`notebooks/`](notebooks/README.md) | 2024 team | 13 research notebooks + 2 scripts: model drafts, data conversion, preprocessing, losses, training |
| `models.py` | 2026, bodies copied unchanged | `DoubleConv`/`UNet` from `model_comparsion.ipynb`, `RES_Block`/`WC_Block` from `BU_net/model_modified.ipynb` |
| `app.py` · `mask_index.py` | 2026 | Streamlit viewer: ground truth vs three models' exported PNG masks; handles missing/corrupt files |
| `scripts/smoke_check.py` | 2026 | U-Net forward/backward on a synthetic 1×1×32×32 input; RES/WC block geometry (32→34, 32→18) |
| `tests/` | 2026 | viewer regression tests on tiny synthetic PNGs (`test_demo.py`, `test_app.py`) |
| [`docs/reproduction-notes.md`](docs/reproduction-notes.md) · [`docs/notebook-guide.md`](docs/notebook-guide.md) | 2026 | provenance, conflicting training configurations, known issues; cell-by-cell reading order |
| `requirements.txt` | 2026 | packages inferred from imports (no historical pins) |

```bash
python -m pip install -r requirements.txt
python scripts/smoke_check.py
python -m unittest discover -s tests -v
streamlit run app.py      # put ground_truth_1.png, model_prediction_1.png, model2_… , model3_… in test_data/
```

The seminar deck, posters, reference PDF, MRI data and prediction images were deliberately left out of the public archive (see the reproduction notes).

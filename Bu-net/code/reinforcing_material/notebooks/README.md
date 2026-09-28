<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../../README.md) › [Bu-net](../../../README.md) › [Code](../../README.md) › [reinforcing_material](../README.md) › **notebooks**</sub>

# `notebooks/` — 2024 research drafts

Historical drafts, not one pipeline — read [`../docs/notebook-guide.md`](../docs/notebook-guide.md) before running anything, and never "Run All".

| File | What it holds |
|---|---|
| `model_comparsion.ipynb` | cell 0: U-Net baseline (source of `models.py`); cells 1–4: BU-Net architecture drafts and a large model summary |
| `BU_net/Jiheon_BU_net.ipynb` | **Jiheon's** full BU-Net draft: RES blocks on all four skips + WC bottleneck |
| `BU_net/Jaeryeong_Bu_net.ipynb` | Jaeryeong Hwang's encoder/decoder-block BU-Net draft |
| `BU_net/model_modified.ipynb` | another full draft; source of `models.py`'s RES/WC blocks |
| `BU_net/RES_Block.ipynb` · `BU_net/WC_Block.ipynb` | isolated block experiments |
| `pretrain.ipynb` | N4ITK + percentile clipping + normalization |
| `make_file.py` · `set.ipynb` · `load_data.ipynb` | slice-dataset serialization and loading |
| `training_process.ipynb` · `all_combined.ipynb` | training drafts (T1 / T1ce, batch 4, 1 or 100 epochs, Adam) |
| `validation.ipynb` | loss-function draft (WCE + Dice) |
| `show.ipynb` | model summary and preprocessing visualization |
| `app_original.py` | the 2024 fixed-50-case comparison viewer (superseded by `../app.py`) |

All notebooks were saved without outputs; data paths point to a local `../data/` folder that is not included.

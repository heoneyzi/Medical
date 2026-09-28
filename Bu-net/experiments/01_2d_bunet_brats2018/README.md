<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [Bu-net](../../README.md) › [Experiments](../README.md) › **E1 · 2-D BU-Net on BraTS 2018**</sub>

# 🔬 E1 · U-Net vs U-Net + WC vs Simplified BU-Net on 2-D BraTS 2018 slices

> **Question —** Do BU-Net's wide-context (WC) and residual-extended-skip (RES) blocks still help a 2-D U-Net when the model has to be simplified for limited compute?

| | |
|---|---|
| **Status** | ✅ done (spring 2024) — presented at the 8th deep daiv. Open Seminar; qualitative comparison only |
| **Model / data** | BraTS 2018 training volumes (T1, T1ce, T2, FLAIR + segmentation), one modality per 2-D slice, 256×256; U-Net, U-Net + WC, Simplified BU-Net |
| **Compute** | not recorded (the project framed itself as working "under limited compute resources") |
| **Headline** | finer boundary detail in some Simplified BU-Net examples (seminar, qualitative); **no Dice / IoU was recorded** |

## Setup

| Step | Implementation | Source |
|---|---|---|
| Bias correction + normalization | N4ITK (Otsu mask) → clip 1st–99th percentile → z-score, per volume, label files skipped | [`preprocess/preprocess.py`](../../code/bu-net_pytorch/preprocess/preprocess.py) · [`notebooks/pretrain.ipynb`](../../code/reinforcing_material/notebooks/pretrain.ipynb) |
| Slicing | slices around each volume's midpoint (±n), resized to 256×256 (bilinear image / nearest label), label 4 → 3 | [`data_loader/data_loader.py`](../../code/bu-net_pytorch/data_loader/data_loader.py) · [`notebooks/make_file.py`](../../code/reinforcing_material/notebooks/make_file.py) |
| Models | `Unet` (31.0 M params), `BU_net` = U-Net + WC, `SimpleBUnet` = WC bottleneck + one RES block (identity + 15- and 9-wide factorized paths) on the 64×64 skip, softmax output | [`model/`](../../code/bu-net_pytorch/model/) |
| Loss | `BU_Net_Loss` = weighted cross-entropy + Dice, per-pixel weights 1 / (class pixel count) | [`model/loss.py`](../../code/bu-net_pytorch/model/loss.py) · [`loss.ipynb`](../../code/bu-net_pytorch/loss.ipynb) |
| Training | Adam; slice-level 80/20 random split; checkpoint every 5 epochs | [`train.py`](../../code/bu-net_pytorch/train.py) · [`notebooks/all_combined.ipynb`](../../code/reinforcing_material/notebooks/all_combined.ipynb) |
| Comparison | ground truth next to three models' exported masks | [`notebooks/app_original.py`](../../code/reinforcing_material/notebooks/app_original.py) → restored as [`app.py`](../../code/reinforcing_material/app.py) |

The archive records several training configurations that were never reconciled into one final run ([reproduction notes](../../code/reinforcing_material/docs/reproduction-notes.md)):

| Record | Settings |
|---|---|
| Final poster | 30 epochs, batch size 16, learning rate 0.01, momentum 0.9 |
| Final presentation, slide 8 | central slices ±3, a 6,840-image training collection |
| `training_process.ipynb` | 40 central slices, T1, loader batch size 4, 1 epoch, Adam |
| `all_combined.ipynb` | 40 central slices, T1ce, loader batch size 4, 100 epochs, Adam |
| `make_file.py` | 10 central slices, 256×256, label 4 → 3, saves only the T1 set |

## Results

The seminar deck and posters (not redistributed here — they embed MRI examples and paper figures) compared U-Net, U-Net + WC and the Simplified BU-Net **qualitatively**. They describe finer boundary detail in some Simplified BU-Net examples, and also state that limited training and architectural differences prevented reproducing the paper's results; the deck itself warns that a high first-epoch pixel accuracy is not evidence of superiority. No run manifest, training log, checkpoint or Dice/IoU table survives. The restored viewer (`streamlit run app.py`) reproduces the *comparison layout* with any exported masks, not the numbers.

## Takeaway

- The team trained the three variants far enough to show side-by-side masks at the seminar — a working prototype pipeline, not a benchmark (the archived notebook versions differ from the packaged `model/` files; see E3).
- Honest caveats found during the 2026 restoration: the random split is **slice-level**, so one patient's anatomy can sit in both train and validation; one Dice implementation broadcasts class IDs instead of one-hot masks; pixel accuracy is dominated by background. See [E3](../03_model_check/README.md) for which model definitions run as written.
- **Not claimed:** any segmentation score, or improvement over U-Net.

## Files

| File | What it is |
|---|---|
| [`code/bu-net_pytorch/`](../../code/bu-net_pytorch/README.md) | packaged PyTorch code (models, loss, metrics, loader, preprocessing, training script) |
| [`code/reinforcing_material/notebooks/`](../../code/reinforcing_material/notebooks/README.md) | 13 original research notebooks incl. name-attributed BU-Net drafts |
| [`code/reinforcing_material/docs/notebook-guide.md`](../../code/reinforcing_material/docs/notebook-guide.md) | cell-by-cell reading order and what to skip |
| [`notes/01_bu-net_paper_review/`](../../notes/01_bu-net_paper_review/README.md) | Jiheon's review of the BU-Net paper (Korean) |

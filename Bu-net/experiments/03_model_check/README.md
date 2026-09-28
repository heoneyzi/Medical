<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [Bu-net](../../README.md) › [Experiments](../README.md) › **E3 · Model check**</sub>

# 🔬 E3 · Which archived models run, and how large are they?

> **Question —** For every U-Net / BU-Net definition kept in the code, does a 256×256 slice flow through it as written, and how many parameters does it have?

| | |
|---|---|
| **Status** | ✅ done (28 Sep 2026) — added for this portfolio; synthetic inputs only |
| **Model / data** | 7 model classes from `code/bu-net_pytorch/model/` and `code/reinforcing_material/notebooks/BU_net/`; random / `meta` tensors, no MRI data |
| **Compute** | CPU, PyTorch 2.14.0, under a minute |
| **Headline** | only **U-Net (31.0 M)** and **Simplified BU-Net (97.4 M)** run end-to-end; every full-BU-Net draft fails on a shape or channel mismatch |

## Setup

[`model_check.py`](model_check.py) loads each class from its original file (skipping module-level demo lines such as `model = BUNet(n_classes=21)`), builds it on PyTorch's `meta` device — which tracks shapes without allocating memory — counts parameters and pushes a dummy 1×C×256×256 tensor through it (C = 1 for the packaged models, 3 for the notebook drafts, as their first layers expect). Models that pass are then run once on CPU with a random input to confirm a finite 4-class output.

## Results

From [`results/model_check.json`](results/model_check.json):

| Model | Source | Parameters | 256×256 forward |
|---|---|---|---|
| U-Net baseline | `bu-net_pytorch/model/Unet.py` | 31,042,564 | ✅ 1×4×256×256, softmax sums to 1 |
| U-Net + WC | `bu-net_pytorch/model/Unet_WC.py` | 92,394,508 | ❌ bottleneck `BatchNorm2d(out_channels)` expects 4 channels, gets 1,024 |
| **Simplified BU-Net** | `bu-net_pytorch/model/SimpleBUnet.py` | **97,381,892** | ✅ 1×4×256×256, softmax sums to 1 |
| Full BU-Net draft (repo) | `bu-net_pytorch/model/BUnet.py` | 171,925,636 | ❌ the WC block's BatchNorm is sized for 512 channels but receives the 1,024-channel concatenation |
| Full BU-Net draft (Jiheon) | `notebooks/BU_net/Jiheon_BU_net.ipynb` | 103,783,428 | ❌ in the WC block, the second conv of each branch expects 512 input channels but receives the first conv's 1,024 |
| Full BU-Net draft (Jaeryeong) | `notebooks/BU_net/Jaeryeong_Bu_net.ipynb` | 176,644,740 | ❌ unpadded RES branches return 2–8 px maps that cannot be concatenated with the 16 px input |
| Full BU-Net draft (`model_modified`) | `notebooks/BU_net/model_modified.ipynb` | 1,954,750,468 | ❌ off-by-one spatial size (34 vs 33) at a skip concatenation |

## Takeaway

- The **Simplified BU-Net is the variant that works**: it pads its factorized convolutions, keeps conv/BatchNorm channel counts consistent and interpolates feature maps before each concatenation — the drafts break on exactly these kinds of channel and size mismatches.
- It is **lighter than every full-BU-Net draft** (97.4 M vs 103.8–176.6 M) but **~3× the plain U-Net**, because the WC block alone (15×1 / 1×15 convolutions at 512 → 1,024 channels plus a 3×3 fusion conv) holds roughly 66 M of its parameters — "lightweight" in this project means simplified relative to BU-Net, not smaller than U-Net.
- The check says nothing about segmentation quality; it only confirms which definitions are executable and how big they are.

## Files

| File | What it is |
|---|---|
| [`model_check.py`](model_check.py) | the check (standalone; needs PyTorch ≥ 2.0) |
| [`results/model_check.json`](results/model_check.json) | per-model parameters, shape-check result, CPU-forward result, error message |

The archive's own [`scripts/smoke_check.py`](../../code/reinforcing_material/scripts/smoke_check.py) complements this: it trains nothing either, but back-propagates through the U-Net and checks the isolated RES/WC block geometry (32 → 34 and 32 → 18 px).

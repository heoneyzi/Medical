<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [Bu-net](../../README.md) › [Code](../README.md) › **bu-net_pytorch**</sub>

# `bu-net_pytorch/` — team PyTorch package (2024)

Snapshot of [heoneyzi/BU-Net_Pytorch_Implementation](https://github.com/heoneyzi/BU-Net_Pytorch_Implementation) at `4e5ab43`, Jiheon's fork of the team repository [iamnotwhale/BU-Net_Pytorch_Implementation](https://github.com/iamnotwhale/BU-Net_Pytorch_Implementation). Git history: 24 commits from 17 May to 5 Jul 2024 by **Boyoung Kwon** (models, loss/metrics, loader, preprocessing, utils, training script, Simplified BU-Net) and **Jaeryeong Hwang** (first BU-Net model, RES/WC refactor, sigmoid output, loss, requirements); 1 documentation commit by Jiheon (12 Sep 2026), kept as [ORIGINAL_README.md](ORIGINAL_README.md).

| Path | Contents |
|---|---|
| `preprocess/preprocess.py` | N4ITK bias correction (Otsu mask) → clip 1st–99th percentile → z-score; CLI `--folder`, `--save_path` (trailing slash required); skips `*_seg` files |
| `data_loader/data_loader.py` | `Custom2DBraTSDataset(data_dir, modality, n)`: central slices ±n, 256×256 resize, label 4 → 3 |
| `model/Unet.py` · `Unet_WC.py` · `SimpleBUnet.py` · `BUnet.py` | U-Net, U-Net + WC (class `BU_net`), Simplified BU-Net, full BU-Net draft (`BUNet`) |
| `model/loss.py` · `loss.ipynb` | `BU_Net_Loss` = class-weighted cross-entropy + weighted Dice |
| `model/metric.py` · `utils/utils.py` | Dice score, pixel accuracy; loss/accuracy plots |
| `model/BU-net.ipynb` | notebook version of the full BU-Net with a printed model summary |
| `train.py` · `test.py` | historical training loop (Adam, 80/20 slice split, checkpoint every 5 epochs); `test.py` is empty |
| `requirements.txt` | original pins (PyTorch 2.3.1, NumPy 2.0.0, SimpleITK 2.3.1, nibabel 5.2.1, …) |

**Preprocess** a BraTS 2018 folder of patient directories (`<id>/<id>_{t1,t1ce,t2,flair,seg}.nii.gz`), then copy the untouched `*_seg.nii.gz` files next to the outputs:

```bash
python preprocess/preprocess.py --folder /abs/path/BraTS2018/HGG/ --save_path /abs/path/brats2018-prep/
```

**Known issues** (unchanged in this snapshot; see [ORIGINAL_README.md](ORIGINAL_README.md)): `train.py` imports `UNet` and `get_loss_train`, which the package does not export, passes `num_slices=` to a loader that takes `n`, and calls `Unet_WC()` although the class in `model/Unet_WC.py` is named `BU_net`; `Unet_WC.py` sizes its bottleneck BatchNorm by `out_channels`; importing `model` builds a 21-class `BUNet` at module level. The Simplified BU-Net and U-Net definitions themselves run — see [E3](../../experiments/03_model_check/README.md).

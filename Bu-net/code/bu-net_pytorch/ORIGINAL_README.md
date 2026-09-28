# BU-Net in PyTorch

**deep daiv. 2024 봄 Medical AI · 팀 보강재**

U-Net 기반 뇌 종양 이미지 분할을 탐구한 팀 프로젝트입니다. 이 저장소는 [iamnotwhale/BU-Net_Pytorch_Implementation](https://github.com/iamnotwhale/BU-Net_Pytorch_Implementation)의 포크로, 기존 모델·전처리·학습 코드를 보존합니다.

팀: [강지헌](https://github.com/heoneyzi) · [권보영](https://github.com/iamnotwhale) · [황재령](https://github.com/Hwang-Jaeryeong)

[deep daiv.](https://deepdaiv.oopy.io/) · [보완 프로젝트 아카이브: reinforcing_material](https://github.com/heoneyzi/reinforcing_material) · [Jiheon Kang's portfolio](https://heoneyzi.github.io/)

## Repository scope

This team research repository contains U-Net/BU-Net variants, a BraTS preprocessing script, a 2D slice loader, loss/metric helpers, and a historical training entry point. This documentation update preserves the original implementation and authorship; it does not report a reproduced training result.

| Path | Contents |
|---|---|
| `preprocess/preprocess.py` | N4ITK bias correction, percentile clipping, and normalization |
| `data_loader/data_loader.py` | BraTS modality/segmentation loading and 256×256 slice resizing |
| `model/` | Existing model variants, losses, and metrics |
| `train.py` | Historical training loop and argument parser |
| `test.py` | Empty file in this source snapshot; no inference implementation |
| `requirements.txt` | Original pinned environment requirements |

## Environment and data

From the repository root, install the existing requirements in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The existing manifest pins NumPy 2.0.0 and PyTorch 2.3.1 along with the other original dependencies; compatibility has not been revalidated by this documentation update.

Obtain the BraTS 2018 training images and segmentation labels through the dataset's permitted access process. The original project references the [BraTS 2018 registration page](https://www.med.upenn.edu/sbia/brats2018/registration.html). Data and trained checkpoints are not included in this repository.

The preprocessing input directory must directly contain patient directories, such as the extracted `HGG/` or `LGG/` directory. Each patient directory uses filenames that match its directory name:

```text
patient_id/
  patient_id_t1.nii.gz
  patient_id_t2.nii.gz
  patient_id_t1ce.nii.gz
  patient_id_flair.nii.gz
  patient_id_seg.nii.gz
```

The loader expects all four modalities and the corresponding segmentation label. It extracts slices around the volume midpoint, resizes images/labels to 256×256, and maps segmentation label 4 to 3.

## Preprocessing

The actual parser accepts **`--folder` and `--save_path`**. Supply a trailing slash for both paths because the existing script also uses string concatenation when iterating patient directories:

```bash
python preprocess/preprocess.py \
  --folder "/absolute/path/to/BraTS2018/HGG/" \
  --save_path "/absolute/path/to/brats2018-preprocessed/"
```

For each non-segmentation image, the script applies N4ITK bias correction, clips the 1st–99th percentile range, and normalizes to zero mean/unit variance.

**The preprocessing script skips segmentation files and does not copy them to its output.** Before training, copy each original `patient_id_seg.nii.gz` unchanged into that patient's preprocessed output directory. The prepared dataset must retain the complete filename layout shown above. Do not apply image normalization to label volumes.

## Training CLI reference and current limitations

The actual training parser requires `--dataset_folder`. Its accepted model strings are `Unet` and `Unet_WC`; the defaults are batch size 1, 10 epochs, learning rate 0.001, model `Unet`, and device `cpu`.

```bash
python train.py \
  --dataset_folder "/absolute/path/to/brats2018-preprocessed" \
  --batch_size 1 \
  --epochs 10 \
  --lr 0.001 \
  --model Unet \
  --device cpu
```

This command records the parser's correct arguments, but the current source has integration errors that prevent a successful training run:

- `train.py` imports `UNet` and `get_loss_train` from `model`, while those symbols are not exported by the current package. The class in `model/Unet.py` is named `Unet`.
- The `Unet_WC` selection is also inconsistent with the class name `BU_net` in `model/Unet_WC.py`.
- Training constructs `Custom2DBraTSDataset(..., num_slices=5)`, but the dataset constructor takes `n` rather than `num_slices`.

These implementation issues remain unchanged in this documentation-only update. The loop is written to save `<epoch>.pth` into its working directory every five epochs, starting at epoch 0, but this is not a claim that checkpoints have been produced by this release. Its random split operates on concatenated slice/modality samples; it should not be described as a patient-held-out evaluation.

## Inference and related archive

`test.py` is empty, so `python test.py` does not run segmentation inference. No executable inference workflow is provided by that file.

The complementary [reinforcing_material archive](https://github.com/heoneyzi/reinforcing_material) provides a separate entry point to the team's project materials. This fork retains its original upstream and team attribution. No new license or model-performance claim is introduced here.

# U-Net based brain tumor segmentation

**A 2024 deep daiv. research project exploring U-Net variants for brain MRI segmentation.**

Team **보강재** presented this work at the **8th deep daiv. Open Seminar**. The project studied the paper *BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture* and compared a U-Net baseline with Wide Context and simplified Residual Extended Skip variants on BraTS 2018.

**Team:** Jiheon Kang (강지헌), Boyoung Kwon (권보영), Jaeryeong Hwang (황재령). The presentation credits the team collectively. `notebooks/BU_net/Jiheon_BU_net.ipynb` is a preserved, name-attributed implementation draft; the archive does not establish a complete division of individual responsibilities. This repository documents collaborative work and does not claim that Jiheon originated the published BU-Net architecture.

## What is included

- **13 original research notebooks:** architecture drafts, MRI data loading, preprocessing, loss experiments, training and model comparisons.
- **Original model definitions in `models.py`:** an importable U-Net baseline and isolated RES/WC blocks, extracted from the notebooks with their class bodies unchanged.
- **A working local comparison viewer:** opens user-exported ground truth and three prediction sets, handles missing/corrupt files, and accepts any number of complete cases.
- **Synthetic checks:** verify U-Net forward/backward computation, historical block dimensions, and the viewer's missing-file behavior without medical images or trained weights.

The notebooks are a research archive, with environment-specific paths replaced and saved outputs cleared. They contain unfinished experiments and conflicting configurations. The runnable entry points below are the smoke check and image viewer. A complete, validated training reproduction and a trained inference model are not provided.

## Quick start

Use a Python environment with a PyTorch build appropriate for your computer. Python 3.12 was used for the repository restoration checks.

```bash
git clone https://github.com/heoneyzi/reinforcing_material.git
cd reinforcing_material
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_check.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The source folder had no dependency lockfile. `requirements.txt` lists packages inferred from the retained imports, without inventing an original version pin. Full notebook execution is not covered by the smoke check.

## Architecture and research question

The experiments ask whether additional skip-path and bottleneck context can help a U-Net delineate brain tumors under limited compute resources.

| Component | Role in the project | Source |
| --- | --- | --- |
| U-Net | Encoder-decoder baseline with skip connections | `notebooks/model_comparsion.ipynb`, first cell |
| Wide Context (WC) | Bottleneck context from factorized large-kernel convolutions | `notebooks/BU_net/WC_Block.ipynb` and architecture drafts |
| Residual Extended Skip (RES) | Multi-scale features along skip paths | `notebooks/BU_net/RES_Block.ipynb` and architecture drafts |
| Simplified BU-Net | Resource-constrained experiment with WC and simplified RES blocks | Project presentation and multiple saved drafts |
| Loss experiments | Dice-style overlap term plus class-weighted cross entropy | `notebooks/validation.ipynb`, training notebooks |
| Preprocessing | N4ITK bias correction, percentile clipping and normalization | `notebooks/pretrain.ipynb` |

The full published BU-Net was not reproduced exactly. The saved drafts vary in channel counts, resize behavior and block design. In the isolated definitions exported to `models.py`, RES increases a 32×32 input to 34×34 because of the final convolution's padding, while WC reduces it to 18×18. These behaviors are preserved and checked, not silently corrected.

## Data and notebook execution order

**BraTS MRI data, segmentation labels, checkpoints and prediction images are not included.** Obtain BraTS 2018 through its data organizers or an authorized distribution and follow the applicable data-use conditions. Do not treat this repository as a data mirror.

The loader expects patient folders with matching NIfTI filenames:

```text
data/BraTS_2018_Train/
└── PATIENT_ID/
    ├── PATIENT_ID_t1.nii.gz
    ├── PATIENT_ID_t1ce.nii.gz
    ├── PATIENT_ID_t2.nii.gz
    ├── PATIENT_ID_flair.nii.gz
    └── PATIENT_ID_seg.nii.gz
```

The historical sources refer to these local paths relative to `notebooks/`. Some source variants use a differently named data directory; set their path variables to your actual local layout. The public labels use values 0, 1, 2 and 4, while some drafts map label 4 to 3. Review the convention before joining components.

For a precise cell-by-cell reading and adaptation sequence, including the cells that must be skipped or fixed before execution, see **[the notebook guide](docs/notebook-guide.md)**. In brief: inspect model definitions, configure preprocessing, inspect data conversion/loading, select a single architecture and loss convention, then review a training draft. The files are alternative experiments and should not all be executed sequentially with “Run All.”

## Open your own comparison images

Create `test_data/` locally and add four PNG files per case:

```text
test_data/
├── ground_truth_1.png
├── model_prediction_1.png
├── model2_prediction_1.png
└── model3_prediction_1.png
```

Use matching positive integers for additional cases. The viewer displays only complete sets and reports missing files. Map **Model 1/2/3** to your own experiment log; the historical app does not identify which architecture produced each prefix. Choose another local directory in the sidebar if needed. The viewer does not run inference or compute evaluation metrics.

## Results and limitations

The seminar materials report a qualitative comparison of U-Net, U-Net + WC, and a simplified BU-Net. They describe finer boundary detail in some simplified BU-Net examples, but also state that limited training and architectural differences prevented reproduction of the paper's results. No independently verified benchmark score is asserted here.

The poster describes 30 epochs, batch size 16, learning rate 0.01 and momentum 0.9. Saved training drafts instead contain settings such as 1 or 100 epochs, actual data loaders with batch size 4, and Adam. These are different experiment records; a single reproducible final configuration cannot be inferred from them.

The archive also contains implementation issues that matter for reuse: slice-level random splitting can mix slices from the same patient across train and validation sets, some Dice implementations use class IDs rather than one-hot masks, preprocessing lacks a constant-image normalization guard, and model input channels and loss instantiation differ across notebooks. **[Reproduction notes](docs/reproduction-notes.md)** identify these issues and the scope of the checks.

## Repository map

```text
app.py                       Local PNG comparison viewer
mask_index.py                Complete-case discovery and image loading
models.py                    Selected unchanged original model definitions
requirements.txt             Dependencies inferred from imports
notebooks/                   13 sanitized historical research notebooks
  BU_net/                    Named model drafts and RES/WC implementations
  app_original.py            Historical viewer source
  make_file.py               Historical slice dataset serialization script
scripts/smoke_check.py        Synthetic model forward/backward checks
tests/test_demo.py            Viewer file-handling regression checks
tests/test_app.py             Streamlit UI regression checks
docs/notebook-guide.md        Exact historical cell order and prerequisites
docs/reproduction-notes.md    Provenance, discrepancies and limitations
```

## 한국어 요약

2024년 deep daiv. 제8회 오픈 세미나의 **보강재 팀(강지헌·권보영·황재령)** 프로젝트입니다. BraTS 2018 뇌 MRI를 이용해 U-Net, WC 블록 추가 모델, 단순화한 BU-Net을 비교하며 의료영상 분할 구조를 탐구했습니다. 당시 연구 노트북과 모델 구현을 보존하고, 실행 안내와 사용자 이미지 비교 데모를 추가했습니다. 논문 원형을 완전히 재현한 결과나 검증된 의료 성능을 주장하지 않으며, 의료 데이터와 학습 가중치는 포함하지 않습니다.

## Attribution

The project presentation names *BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture* by Mobeen Ur Rehman, SeungBin Cho, Jee Hong Kim and Kil To Chong as its research reference. U-Net and BU-Net remain the work of their respective paper authors. The source directory also contained AIKU/CS231n-derived teaching exercises; those separate materials are not bundled here. No original software license was found in the source folder, so this restoration does not assign a new license to the historical team code or third-party material.

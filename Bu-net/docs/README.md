<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [Bu-net](../README.md) › **Docs**</sub>

# 📄 Bu-net docs

| Document | Language | What it is |
|---|---|---|
| [project_report.md](project_report.md) | Korean | Notion report on the 3-D cascaded U-Net track (BraTS 2019), 33 figures in [`assets/`](assets/) — summarized below |
| [reproduction-notes.md](../code/reinforcing_material/docs/reproduction-notes.md) | English | provenance of the 2-D archive, conflicting training records, known implementation issues |
| [notebook-guide.md](../code/reinforcing_material/docs/notebook-guide.md) | English | cell-by-cell reading order for the 2024 notebooks |
| [notes/](../notes/README.md) | Korean | Jiheon's paper reviews (BU-Net; CXR preprocessing) |

> [!NOTE]
> The report is the Notion page titled "BU-Net" in Jiheon's portfolio workspace. It speaks for "our team" but names no authors, and its code link is [Yuyeon-Kim/brain-tumor-segmentation](https://github.com/Yuyeon-Kim/brain-tumor-segmentation) (a "2024 MediAI team" repository). It uses **BraTS 2019**, whereas the 2-D BU-Net code uses **BraTS 2018**. Bookmark blocks in the page could not be exported; an attached image archive was dropped. Images wider than 1,600 px were downscaled.

## The report at a glance (English summary)

| Section | Content |
|---|---|
| 1 · Introduction | Goal: separate tumor from healthy brain tissue on MRI and delineate the three nested regions — whole tumor, tumor core, enhancing tumor. Background on gliomas (HGG vs LGG) and the four MRI modalities (T1, T1ce, T2, FLAIR) |
| 2.1 · Dataset | Kaggle BraTS 2019: 335 patients (259 HGG + 76 LGG), four modalities + one label volume each; each modality paired with the label as a separate training sample. Problems: few labelled volumes, strong class imbalance |
| 2.2 · Transforms | labels re-ordered from largest to smallest region for the cascade; augmentation by random-axis flip + 30° rotation (p = ½) and elastic deformation (σ = 2, cubic-spline smoothing); 240→160 in-plane resize with label remapping so classes 1 and 4 survive; cascade-aware random crop around the previous stage's label |
| 2.3–2.4 · Model | 3-D U-Net (encoder/decoder with skip connections) used three times in a cascade — WT, then TC inside the WT crop, then ET inside the TC crop ("Triple U-Net") — framed as three binary problems instead of one 4-class problem |
| 3.1 · Augmentation | 40 training volumes → 40 × 2 × 2 = 160 |
| 3.2 · Loss | BCE reached ≈ 0.14 while predicting all background (imbalance 160–5,230×) → switched to Dice loss |
| 3.3–3.4 · Optimizer / scheduler | SGD failed to converge → Adam; no scheduler diverged → LambdaLR and StepLR both tried |
| 3.5 · Hardware | "2070 Ti", 8 GB: batch size 2 at 128×128×32; Ubuntu 18.04.6; Python 3.7–3.9 |
| 4.1 · Results | best: Dice + Adam + StepLR; validation loss dropped sharply near epoch 30; 60 epochs took ≈ 24 h; final predictions ≈ 12 % accuracy with many false positives and negatives |
| 4.2 · Analysis | resize information loss; batch size 2; no modality information in the input; too little data; too few, possibly ill-suited hyper-parameters and loss |
| 4.3 · Plans | server GPU; modality-aware (multimodal) training; broader, more efficient hyper-parameter search |

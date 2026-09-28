<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [Bu-net](../README.md) › **Experiments**</sub>

# 🧪 Bu-net experiments

Three studies around one question — how much of BU-Net survives when compute is scarce. E1 is the team's 2-D BU-Net work, E2 the archived 3-D cascade report from the same Medical AI context, E3 a check added while building this portfolio.

| # | Experiment | Data | What came out | Evidence |
|---|---|---|---|---|
| E1 | [2-D U-Net vs U-Net + WC vs Simplified BU-Net](01_2d_bunet_brats2018/README.md) | BraTS 2018, 2-D slices | qualitative seminar comparison; no Dice recorded | seminar summary in the archive docs |
| E2 | [3-D “Triple U-Net” cascade](02_3d_cascade_brats2019/README.md) | BraTS 2019, 3-D volumes | BCE collapse → Dice; best ≈ 12 % accuracy; 60 epochs ≈ 24 h | Notion report (Korean) |
| E3 | [Model check](03_model_check/README.md) | synthetic | U-Net 31.0 M and Simplified BU-Net 97.4 M run; full drafts do not | `results/model_check.json` |

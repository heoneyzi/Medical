<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [Bu-net](../README.md) › **Code**</sub>

# 💻 Bu-net code

Two source snapshots, copied without code changes (`.git/` and `.gitignore` omitted; each original README kept as `ORIGINAL_README.md`):

| Folder | Snapshot of | Who wrote it | What to open first |
|---|---|---|---|
| [`bu-net_pytorch/`](bu-net_pytorch/README.md) | [heoneyzi/BU-Net_Pytorch_Implementation](https://github.com/heoneyzi/BU-Net_Pytorch_Implementation) @ `4e5ab43` (fork of [iamnotwhale/BU-Net_Pytorch_Implementation](https://github.com/iamnotwhale/BU-Net_Pytorch_Implementation)) | code: Boyoung Kwon, Jaeryeong Hwang (May – Jul 2024); docs: Jiheon (Sep 2026) | `model/SimpleBUnet.py`, `model/loss.py`, `preprocess/preprocess.py` |
| [`reinforcing_material/`](reinforcing_material/README.md) | [heoneyzi/reinforcing_material](https://github.com/heoneyzi/reinforcing_material) @ `4093e1f` | 2024 team notebooks (incl. name-attributed drafts); 2026 restoration, viewer, tests and docs by Jiheon | `docs/notebook-guide.md`, `models.py`, `app.py` |

Checks run on these copies for this portfolio (Sep 2026, CPU, PyTorch 2.14): `reinforcing_material/scripts/smoke_check.py` ✅, `python -m unittest tests.test_demo` ✅ (3 tests; the Streamlit UI test needs `streamlit`), and the model check in [experiments/03_model_check](../experiments/03_model_check/README.md). No MRI data, weights or result images are included in either snapshot, and neither source folder carried a license file.

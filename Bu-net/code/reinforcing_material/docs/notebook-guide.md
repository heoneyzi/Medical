# Historical notebook guide

These files preserve research drafts rather than a single finished training pipeline. Cell numbers below are **zero-based** and match the JSON cell positions in each notebook. Open Jupyter from the notebook directory so that the sanitized `../data/` paths point to the repository's ignored `data/` directory:

```bash
cd notebooks
jupyter notebook
```

Execute one selected branch in a fresh kernel. Do not combine competing definitions from multiple drafts in the same session. No notebook was executed against MRI data during restoration.

## 1. Inspect model definitions first

1. Open `model_comparsion.ipynb`. **Cell 0** defines `DoubleConv` and the U-Net baseline; it has no training side effect. Its input-channel argument can match a single MRI modality. This cell supplies the baseline in `../models.py`.
2. **Cells 1–3** contain alternative architecture definitions with hardcoded three-channel inputs and very large transposed-convolution kernels. Treat them as architecture drafts. Do not allocate these full models without reviewing their memory requirements and shape alignment.
3. **Skip cell 4** for a first inspection. It instantiates `BU_net1(4)` and requests a 64×3×256×256 model summary, which can use substantial memory.
4. `BU_net/RES_Block.ipynb` **cell 0** and `BU_net/WC_Block.ipynb` **cell 0** hold isolated block experiments. `WC_Block.ipynb` lacks its own torch imports; add the same imports used in `BU_net/model_modified.ipynb` before using it in a fresh kernel.
5. `BU_net/model_modified.ipynb` **cell 0** contains another full model draft. The RES/WC class bodies exported to `../models.py` come from this cell. `BU_net/Jiheon_BU_net.ipynb` **cell 1** and `BU_net/Jaeryeong_Bu_net.ipynb` **cell 0** preserve the name-attributed alternatives. Jiheon's first cell only imports Streamlit; the second creates a model at its end, so inspect that allocation before running it.

For a small, verified run of original model definitions, use `python scripts/smoke_check.py` from the repository root instead of executing all the architecture drafts.

## 2. Configure preprocessing and data loading

1. In `pretrain.ipynb`, **cell 0** defines bias correction, normalization and file preprocessing, then calls `preprocess_image(input_image_path, output_image_path)`. Define those two variables before running the cell. Use an output path separate from the input volume. Review the missing zero-standard-deviation guard before applying it to arbitrary volumes.
2. `set.ipynb` **cell 0** defines a slice dataset and serialization helpers, then loads files and saves a `.pth` dataset. Set `data_dir` and the destination before executing it. The saved filename in this cell does not match the per-modality names loaded in cell 1; align these names deliberately.
3. `set.ipynb` **cell 1** and `load_data.ipynb` **cell 0** are alternative copies of the serialized-dataset loader. Run only one after creating the expected per-modality files. These loaders call `torch.load`; load only local artifacts you created and review compatibility with your installed PyTorch serialization defaults.
4. `make_file.py` is a separate data-conversion draft. It creates all four modality datasets at module level, but saves only the T1 dataset at the end. It resizes to 256×256, takes 10 central slices, and maps label 4 to 3. It is not a general command-line preprocessing tool and should not be imported for its classes without moving or guarding its top-level execution first.

## 3. Review one training draft

### `training_process.ipynb`

1. **Skip cell 0.** It contains the shell command `conda install tensorflow-gpu` without notebook magic syntax. TensorFlow is not imported by the retained training code.
2. Run **cells 1–2** to load imports and define the dataset. Cell 3 is a commented alternative.
3. Set `data_dir` in **cell 4**. Run **cell 5** only after MRI files are available. Cell 6 is a comment. Run **cell 7** to build the actual batch-size-4 loader. Cells 8–9 are optional shape inspection; cell 8 prints image and label values.
4. Inspect **cell 10** (architecture), **cell 11** (loss), and **cells 12–16** (training/evaluation functions). Reconcile their channel counts, label shapes, loss inputs and constructor signatures before using them together.
5. Configure **cell 17** only after these issues are resolved. Its `epochs = 1`, Adam optimizer and nominal `batch_size = 16` do not override the loader already created in cell 7. The loss is assigned as a class rather than an instance.
6. **Cell 18** starts training and writes checkpoint files. It is not a validated run entry point in the restored archive.

### `all_combined.ipynb`

This notebook has **one code cell** containing data loading, a slice-level random train/validation split, a model, experimental loss and an epoch loop. It uses T1ce and sets 100 epochs. Split the cell into reviewed components before execution. Replace the slice-level split with patient-level grouping if evaluating generalization, and resolve the multiclass loss issues described in the reproduction notes. Do not treat its settings as the poster's final configuration.

## 4. Inspect losses and visualization separately

- `validation.ipynb` **cell 0** is a loss-function draft, not a validation report. `BU_Net_Loss.__init__` takes unused `pred` and `target` parameters, while other files use another signature. Its Dice and weighted-cross-entropy tensor conventions require correction before training. Cell 1 is empty.
- In `show.ipynb`, **skip cell 0**, which is another bare conda shell command. **Cell 1** defines a model and invokes `torchsummary`, which may allocate a large model. **Cell 2** defines local MRI paths and calls preprocessing helpers that are not defined in this notebook; load the helper definitions from `pretrain.ipynb` and set paths first.
- Use `streamlit run app.py` from the repository root for the restored viewer. `notebooks/app_original.py` records the old fixed-50-case interface and its missing-file limitations. It is retained for provenance, not used by the new app.

The original filename `model_comparsion.ipynb` is intentionally retained so existing references and provenance remain recognizable.

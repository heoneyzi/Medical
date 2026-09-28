# Provenance and reproduction notes

## Source evidence

The restored files come from the project source folder linked to `heoneyzi/reinforcing_material`. Its existing local Git metadata tracked only two placeholder files, while the substantive notebooks and scripts were untracked. This release restores those project sources without changing the originals.

The project deck titled **“U-Net 기반 뇌 종양 이미지 분할”** credits **강지헌, 권보영, 황재령** and the **8th deep daiv. Open Seminar**. The final 30-slide deck and A2/A4 posters supplied the project description. The decks and the reference PDF are not included here: their embedded figures, MRI examples and external paper content were not cleared for redistribution as part of the source restoration.

Only the source-notebook filenames provide individual code attribution. The presentation provides team-level authorship; it does not document project leadership, individual percentage contributions or measured improvements by each person.

## Preserved and added content

Historical notebook cell sources are unchanged except for replacing personal absolute paths and an example patient identifier with local placeholders. Saved outputs, execution counts, attachments and machine/host metadata were removed. The original filenames and the separately named author drafts remain intact.

`models.py` extracts the exact `DoubleConv` and `UNet` bodies from `model_comparsion.ipynb` cell 0, and the exact `RES_Block` and `WC_Block` bodies from `BU_net/model_modified.ipynb` cell 0. The surrounding imports and module documentation are new. `app.py`, `mask_index.py`, the smoke check, tests and documentation were added during restoration. The original app is retained as `notebooks/app_original.py`.

## Configuration disagreements

| Record | Observed settings or claims |
| --- | --- |
| Final poster | 30 epochs, batch size 16, learning rate 0.01, momentum 0.9 |
| Final presentation, slide 8 | Describes central slices ±3 and a 6,840-image training collection |
| `training_process.ipynb` | 40 central slices, T1 data, actual loader batch size 4, one epoch, Adam |
| `all_combined.ipynb` | 40 central slices, T1ce data, actual loader batch size 4, 100 epochs, Adam |
| `make_file.py` | 10 central slices, 256×256 resizing, label 4→3, saves only T1 at the end |

These values are records of different stages. There is no saved run manifest, complete final training log or checkpoint linking one of the drafts to the poster. Do not present a particular code file as the verified 30-epoch experiment.

## Issues to resolve before a new experiment

- **Evaluation split:** `all_combined.ipynb` randomly splits the combined slice dataset. Separate patients before slicing to avoid the same patient's anatomy appearing in both training and validation.
- **Labels and loss:** some drafts remap class 4 to 3 and others clamp class IDs. In the combined Dice code, integer class IDs are expanded across channels instead of converted to class-wise one-hot masks. Some model variants return probabilities before code that applies `log_softmax`. Define one consistent multiclass convention before training.
- **Loss constructors:** `BU_Net_Loss` signatures and call sites disagree across notebooks. A class assignment is not interchangeable with an initialized loss module.
- **Input channels and dimensions:** MRI loaders emit a single modality/channel, but several BU-Net drafts expect three channels. Large-kernel blocks and transposed convolutions also have different spatial behavior across drafts.
- **Model memory:** several historical decoders use transposed-convolution kernels up to 121×121 and the model-summary example requests a large batch. The smoke check does not instantiate these full experimental models.
- **Normalization:** `pretrain.ipynb` divides by the image standard deviation without a zero-variance guard. The clipping/normalization calculation also includes the whole image volume rather than an explicitly selected brain foreground.
- **Dataset utilities:** archive scripts assume an existing directory tree, specific local filenames and writable output folders. The serializer and loader filenames do not all match. They require adaptation before use.
- **Metric interpretation:** pixel accuracy can be dominated by background. The presentation itself flags high first-epoch accuracy and limited training as reasons to avoid interpreting it as evidence of model superiority. No verified Dice, IoU, sensitivity or clinical result is claimed by this restoration.

## Validation scope

The smoke script uses random CPU tensors. It checks a four-class U-Net forward pass, probability normalization and finite gradients, plus the unchanged RES/WC block dimensions. This demonstrates that these selected definitions execute; it does not reproduce a training run or validate the full BU-Net variants.

Viewer regression tests use tiny synthetic grayscale PNGs created in temporary directories. They check missing folders, partial comparison sets, numeric case ordering and corrupt-file handling. No medical image is required or included. Test images are discarded when each test finishes.

Restoration checks used Python 3.12.14, PyTorch 2.14.0, Streamlit 1.63.0, NumPy 2.3.5 and Pillow 12.3.0 on macOS arm64. These versions describe the new smoke/UI checks, not the historical training environment.

The data, weights, result images, large slide files and separate teaching exercises remain outside this source release. There is no model suitable for clinical use in this repository.

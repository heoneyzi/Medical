<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [CAFA6](../../README.md) › [Code](../README.md) › **src**</sub>

# `src/` — team Python package

Run modules from the parent folder (`python -m src.<package>.<module>`). Owners follow the team's role split in the meeting notes; git does not record per-file authorship.

| Package | What it contains | Pipeline |
|---|---|---|
| [`common/`](common/) | `fasta.py` reader · `go_obo.py` parent parsing, max-propagation to ancestors, top-k capping, 3-significant-figure formatting · `protein.py` sequence sanitizing (U/Z/O/B → X) and ProtT5 spacing | shared |
| [`train/`](train/) | `prott5_go_train.py` ProtT5-XL + MLP multi-label classifier with in-loop IA-weighted F-max · `prott5_go_train_config.py` CLI dataclass | [P1](../../pipelines/01_prott5_classifier/README.md) |
| [`test/`](test/) | `prott5_go_predict.py` single/multi-model prediction with propagation | [P1](../../pipelines/01_prott5_classifier/README.md) |
| [`esm-c_model/`](esm-c_model/) | ESM-C embedding cache, CV trainer with negative sampling, micro F-max | [P3](../../pipelines/03_esm_c/README.md) |
| [`pretrain/`](pretrain/) | span-masking JEPA pre-training of a ProtT5/ESM encoder | [P4](../../pipelines/04_jepa_encoder/README.md) |
| [`jepa_pipeline/`](jepa_pipeline/) | embedding export from a JEPA encoder | [P4](../../pipelines/04_jepa_encoder/README.md) |
| [`jepa_go/`](jepa_go/) | frozen-encoder GO head and label-space JEPA models, datasets, metrics, GO ontology and taxonomy utilities | [P4](../../pipelines/04_jepa_encoder/README.md) · [P5](../../pipelines/05_label_space_jepa/README.md) |

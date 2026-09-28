# TranscriptFormer-Sapiens cell-embedding smoke

- Smoke status: passed (`model_provenance.csv` status remains `verified_broad` because pretraining overlap is unknown)
- Official repository commit: `c943a89a4de9511a8ab1010715bf1cfa8827cc99`
- `tf_sapiens` directory SHA-256: `6d01c49f06bb902541243a72939cb21a1f30fea2fc2efd8787e4478b49aa2e05`
- Input: 128 raw-count Replogle K562 control pseudobulks, 4,017 symbols
- Staged input: 3,859 unique approved/alias HGNC-to-Ensembl mappings (96.07%)
- Result: 128/128 cells retained; finite 2,048-dimensional cell embeddings
- Elapsed inference time on this machine: 32.93 seconds

The official CLI was used with the model's documented unknown assay category. Block-mask
compilation was disabled through the official flag because this environment's PyTorch Inductor
could not compile the Triton kernel. The official output intentionally rewrites observation names,
so the adapter validates exact row count and order instead of pretending those generated names are
source cell IDs. The authoritative machine-readable record is
`data_experiments/model_smoke/transcriptformer/transcriptformer_cell/k562_replogle_smoke/metadata.json`.

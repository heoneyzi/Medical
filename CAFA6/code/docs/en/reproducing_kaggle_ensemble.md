# GOA + ProtT5 Ensemble: Local Script and Kaggle Notebook

The original Kaggle notebook `cafa-6-goa-prott5-ensemble-0-370.ipynb` does **not** train models. It:

1. Loads two **precomputed** prediction files:
   - `goa_submission.tsv` (GOA-derived predictions)
   - `prott5_interpro_predictions.tsv` (ProtT5 + InterPro predictions)
2. Ensembles them with fixed weights.
3. Applies ontology-aware post-processing (“GOA+ propagation”).
4. Writes `submission.tsv`.

## What is reproducible here

This repo contains a local reproduction of that logic:

- Notebook: `notebooks/cafa-6-goa-prott5-ensemble-0-370.ipynb`
- Script: `scripts/make_ensemble_submission.py`

The **script** uses the repository's local layout by default:

- Competition files: `data/Train/go-basic.obo`, `data/Test/testsuperset.fasta`
- Prediction files: `data/kaggle/goa_submission.tsv`, `data/kaggle/prott5_interpro_predictions.tsv`

The notebook retains its original Kaggle paths: `/kaggle/input/cafa-6-protein-function-prediction` and `/kaggle/input/cafa6-goa-predictions`. Edit its path cell to use it outside Kaggle. The script filters outputs to IDs present in `data/Test/testsuperset.fasta`; the original notebook does not perform that test-FASTA filtering.

ID filtering restricts the output set. It does **not** verify the release date, label provenance, or temporal validity of external GOA/model predictions. Those must be established independently for competition-compliant evaluation. The inherited `0-370` filename is not a score reproduced by this documentation update.

## What is *not* reproducible without extra resources

The upstream generation of:

- GOA predictions (requires external GOA/UniProt resources and a specific pipeline)
- ProtT5 + InterPro predictions (requires model weights / feature pipeline, potentially GPU, and external resources)

Those are separate training/inference strategies not included in the Kaggle notebook.

## Run (script)

Set up the Python 3.12+ project environment as described in the [README](../../README.md), obtain the competition files and both external prediction TSVs, then use the project interpreter:

`./.venv/bin/python scripts/make_ensemble_submission.py --out submissions/submission.tsv`

Use `--competition-data` and `--prediction-data` to override the two input roots. The repository does not distribute the prediction TSVs or the checkpoints/resources that originally generated them. No full ensemble or leaderboard evaluation was rerun during this update.

Key tunables (match Kaggle defaults):

- `--weight-goa 0.55`
- `--weight-prott5 0.45`
- `--top-k 200`
- `--min-score 0.001`
- `--neg-prop-alpha 0.7`
- `--scaling-power 0.8`
- `--max-score 0.95`

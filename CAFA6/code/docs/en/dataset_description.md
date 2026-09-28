# Dataset Description

This dataset provides Gene Ontology (GO) annotations for proteins and supporting resources for training and evaluating function prediction models. It includes curated protein sequences, experimentally supported GO term labels, taxonomy metadata, the GO graph, information accretion weights, and a test superset for inference.

For the end-to-end task definition, see [task_specification.md](task_specification.md). Download the external files from the [CAFA 6 competition data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data); this Git repository does not bundle the dataset.

## Background

The Gene Ontology (GO) is a directed acyclic graph (DAG) of functional descriptors (terms or classes). Terms are connected by relationships such as `is_a` and `part_of`. Each term belongs to one of three subontologies, defined by their root nodes:

- Molecular Function (MFO): `GO:0003674`
- Biological Process (BPO): `GO:0008150`
- Cellular Component (CCO): `GO:0005575`

A protein can be annotated with multiple terms within and across these subontologies. In this dataset, term-protein assignments are derived from experimentally supported evidence (and related high-confidence evidence), and are treated as ground truth labels. Missing annotations do not imply absence of function.

## Training Set

The training set includes proteins with annotations supported by:

- Experimental or high-throughput evidence
- Traceable author statement (TAS)
- Inferred by curator (IC)

Training sequences are drawn from UniProtKB (Swiss-Prot) release 2025_03 (18 June 2025). The set spans eukaryotes and a small set of non-eukaryotes (13 bacteria, 1 archaea). Only labeled proteins appear in the training sequence file.

## Test Superset and Test Set

The test superset contains protein sequences for which predictions should be generated. The final test set is a hidden subset of the superset that gains experimental annotations between the submission deadline and evaluation. Only those proteins with new experimental annotations are used for scoring.

## Files

Arrange the downloaded files under `data/`. This directory is ignored by Git and is absent from a fresh clone.

### Ontology

- `data/Train/go-basic.obo`
  - GO DAG structure (OBO format), release 2025-06-01.
  - Can be parsed using OBO-capable libraries (e.g., obonet in Python).

### Training Data

- `data/Train/train_sequences.fasta`
  - Protein sequences for the training set (FASTA format).
  - Headers contain UniProt accession and identifiers. Example:
    - `sp|P9WHI7|RECN_MYCT` indicates Swiss-Prot entry P9WHI7.

- `data/Train/train_terms.tsv`
  - Ground truth GO term labels for training proteins.
  - Columns:
    1. UniProt accession ID
    2. GO term ID
    3. Ontology namespace (`BPO`, `CCO`, `MFO`)

- `data/Train/train_taxonomy.tsv`
  - Species metadata for training proteins.
  - Columns:
    1. UniProt accession ID
    2. NCBI taxon ID

### Test Data

- `data/Test/testsuperset.fasta`
  - Protein sequences for prediction (FASTA format).
  - Headers include UniProt accession and taxon ID.

- `data/Test/testsuperset-taxon-list.tsv`
  - List of taxon IDs present in the test superset.

### Evaluation Weights

- `data/IA.tsv`
  - Information accretion weights for each GO term.
  - Used to compute weighted precision and recall during evaluation.

### Submission Format

- `data/sample_submission.tsv`
  - Example of the expected submission format for predictions.

## Notes and Conventions

- Proteins may have multiple GO terms within and across subontologies.
- The absence of a term does not mean the protein lacks the function; it may be unannotated.
- All training sequences are Swiss-Prot entries.
- The test set is derived from the test superset over time as new experimental annotations become available.

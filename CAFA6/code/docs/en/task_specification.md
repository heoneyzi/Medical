# Task Specification (Protein Function Prediction with GO Terms)

## Overview

Proteins are large biological molecules composed of 20 amino acids arranged in a sequence. A protein’s amino-acid sequence largely determines its structure and functional roles in cells. The goal of this task is to build a model that predicts what a protein does from its amino-acid sequence by assigning **Gene Ontology (GO)** terms.

GO terms describe three aspects of protein function:

- **Molecular Function (MF)**: biochemical activity (e.g., binding, catalysis)
- **Biological Process (BP)**: broader biological objective (e.g., DNA repair)
- **Cellular Component (CC)**: where it acts in the cell (e.g., nucleus)

Predictions are **multi-label** (each protein can have many GO terms) and the labels are **hierarchical** (GO is a DAG, so child terms imply parent terms).

## Problem Definition

Given a protein amino-acid sequence (FASTA record), produce a set of predicted GO terms with probabilities:

- Input: protein ID + amino-acid sequence
- Output: zero or more `(protein_id, go_id, probability)` associations
- Constraints: probabilities must be in `(0, 1]` (no zeros) and are typically limited to a manageable number per protein

The prediction file combines MF/BP/CC associations in a single list, but evaluation is performed separately per subontology and combined afterward.

## Prospective Evaluation Setup (Test Superset vs. Test Set)

This is a **prospective** evaluation: many proteins in the provided test data do not currently have experimentally assigned functions.

- **Test superset**: all proteins for which you must generate predictions (the full inference target list).
- **Test set (per subontology)**: a hidden subset of the test superset that gains **new experimentally validated GO annotations** between the submission deadline and the evaluation time for that subontology.

There are three test sets (MF, BP, CC). The same protein may appear in more than one test set if it gains new experimental annotations in multiple subontologies.

The training set contains proteins that already have at least one experimentally supported GO annotation in at least one subontology; some of these proteins may also appear in the test superset.

For file-level details (paths, formats), see `docs/dataset_description.md`.

## Evaluation Metric (Information-Accretion Weighted F1)

Evaluation is based on the **maximum F-measure (F1)** derived from **information-accretion (IA)** weighted precision and recall, computed independently for each subontology:

- Compute IA-weighted precision and recall across predicted vs. true annotations.
- Sweep a decision threshold over prediction scores to compute the **maximum** weighted F1 for:
  - MF test set
  - BP test set
  - CC test set
- Combine the three subontology scores as an **arithmetic mean**.

### Why weighting matters

GO is hierarchical, and general (high-level) terms are implied by specific (deep) terms. The IA weight for a term reflects how informative it is (terms that occur frequently in proteins have lower weight; rare, specific terms have higher weight; root terms have weight 0).

### Knowledge-gain subtypes

Evaluation may additionally stratify proteins by how much was known at submission time (terminology from CAFA-style evaluations):

- **No-knowledge**: no experimental annotations existed in a subontology at submission time, but some appear later.
- **Limited-knowledge**: some experimental annotations existed, additional ones appear later.
- **Partial-knowledge**: experimental annotations existed in all three subontologies, and additional ones appear later in at least one subontology.

Scores can be computed per subtype (as means across subontologies) and then combined (mean across subtypes) into the final score.

## Leaderboard Notes (Distribution Shift)

The public leaderboard may be based on a relatively small protein subset of the test superset and is intended as a rough indicator. The final evaluation set is determined by future experimental curation, so distribution shift between leaderboard proteins and final test proteins is expected. Aim to maximize generalization performance.

## Submission Format (GO Term Predictions)

Submissions are tab-separated with **no header** and one association per line:

```
<target_id>\t<GO:nnnnnnn>\t<score>
```

Rules and constraints:

- `target_id` must match the protein IDs in the test superset FASTA headers.
- `GO:nnnnnnn` must be a valid GO term ID from the provided GO version; invalid terms are excluded from evaluation.
- `score` must be in `(0, 1.000]` and should contain up to **3 significant figures**.
- A score of `0` is not allowed; omit such pairs entirely.
- If a protein ID is **not present** in the submission file, all of its predictions are assumed to be 0.
- To limit file sizes, each target should have **no more than 1500** associated terms across MF+BP+CC combined.

### Ontology propagation during evaluation

If the submitted predictions are not already propagated to the GO roots, evaluators may recursively propagate:

- Each parent term’s score is set to the **maximum** of its children’s scores.

This ensures ontology-consistent predictions for scoring.

## Optional Task: Free-Text Function Prediction

In addition to GO terms, you may optionally submit short English text lines describing protein function. This is **not** evaluated during the competition window and does not affect the GO leaderboard, but may be assessed later once sufficient human-written descriptions accumulate.

Format (in the same single submission file):

```
<target_id>\tText\t<score>\t<text_line>
```

Constraints:

- Up to **five** text lines per protein (to allow different confidence levels for different assertions).
- Text must use ASCII printable characters `33–126`, with space (ASCII `32`) as a word delimiter.
- Text lines must not contain tabs.
- Total text per protein is limited to **3000 characters** across all its lines (including spaces); longer submissions may be truncated.

### How text may be evaluated later

- Phase 1: automatic assessment with language models against human-written descriptions (e.g., in UniProt), using standard summarization-style metrics to identify best teams.
- Phase 2: human evaluation of top candidates against reference descriptions.

Two common scenarios:

- The literature exists publicly but is not yet summarized in UniProt at submission time (closer to summarization).
- Little to no public literature exists (requires functional inference from sequence and other signals).

## Participation Note (Scientific Article)

Organizers may write a scientific article describing the event after the competition. Some competitions offer an opt-in process for potential co-authorship consideration, with selection based on contribution merit.


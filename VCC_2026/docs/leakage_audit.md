# Leakage audit

The strict experiment permits only Replogle 2022 K562 and RPE1 as explicit perturbation-response
sources. UCE, TranscriptFormer, scGPT, scFoundation, scPRINT-2, STATE, and STACK may contribute
only frozen representations. Their checkpoint corpora are broad and overlap with Replogle or VCC
cannot currently be ruled out; `unknown` is retained in `model_provenance.csv`.

Manual review remains required before leaderboard submission:

- challenge external-model/data policy wording and phase date;
- each checkpoint's release date and weight license;
- documented inclusion of Replogle or other Perturb-seq corpora;
- possible VCC-specific answer contamination;
- whether model artifacts may be used in the intended competition phase.

The `shadow-existing` protocol is never leakage-free: it consumes source-context perturbation
responses from the same external evaluation dataset and requires an explicit acknowledgement.

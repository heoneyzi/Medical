<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [CAFA6](../README.md) › **Pipelines**</sub>

# 🧪 CAFA6 pipelines

Each folder documents one line of the team's work — the question it asked, how the code in [`../code`](../code/README.md) implements it, and what (if anything) was measured. Owners follow the team's role split in the archived meeting notes ([3rd meeting](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/meetings/03_meeting03_2026-01-21.md), 21 Jan · [4th meeting](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/meetings/04_meeting04_2026-01-29.md), 29 Jan 2026).

| # | Pipeline | Owner (notes) | What it does | Saved number |
|---|---|---|---|---|
| P0 | [Dataset EDA](00_eda/README.md) | team notebook | sequence, label, taxonomy, IA and GO-graph statistics | 82,404 train · 224,309 test superset · 26,125 terms |
| P1 | [ProtT5 classifier](01_prott5_classifier/README.md) | **Jiheon Kang** (“T5”) | ProtT5-XL, top-2 blocks fine-tuned, per-ontology MLP heads, parent propagation | none archived |
| P2 | [GOA + ProtT5 ensemble](02_goa_prott5_ensemble/README.md) | public notebook studied by the team | 0.55/0.45 blend of shared GOA and ProtT5+InterPro predictions + GO post-processing | 0.370 reported by the notebook's author |
| P3 | [ESM-C embeddings](03_esm_c/README.md) | Yoonjin Cho | frozen ESM-C 300M, three poolings, MLP heads with negative sampling, 5-fold CV | fold micro-F1 0.0154–0.0201 (early) |
| P4 | [JEPA encoder](04_jepa_encoder/README.md) | Subin Park | span-masking JEPA pre-training, then frozen encoder + MLP head | none archived |
| P5 | [Label-space JEPA](05_label_space_jepa/README.md) | Subin Park | predict masked GO-label embeddings + BCE, optional ontology/taxonomy features | 0.139 public LB |

Yumin Jung's GO-retriever line and the GCN refinement discussed in the meetings are not part of the code repository; their notes are in [03_Study/Genomics › CAFA 6](https://github.com/heoneyzi/Study/blob/main/Genomics/season1_functional_genomics/cafa6/README.md).

**Reading order for a newcomer:** P0 (what the data is) → P1 (a plain supervised baseline) → P2 (why curated annotations were hard to beat) → P5 (the team's most distinctive idea).

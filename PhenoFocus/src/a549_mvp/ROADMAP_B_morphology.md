# Direction B — A549 morphology cross-line test (roadmap)

This is the demonstration that directly answers the "U2OS 한계" caveat: run
**A549 morphology** through the frozen U2OS-trained morphology tower and show it
still lands on the correct MoA.

## What we confirmed

The morphology modality is **not** CellProfiler features. From the deposit:

```
final_model/model_ckpts/.../  ->  encoders.morph.module.0.weight : (256, 1536)
final_data/jump_map/dino_tvn_morphology.h5ad : (108836, 1536), columns '0'..'1535'
```

So the morphology tower consumes a **1536-dim DINO image embedding** (TVN-
normalised). 1536 = DINOv2 ViT-g/14 embedding width. The `3297` in
`configs/model/triple_gmc.yaml` is a stale template default, not the released
models.

## Pipeline to build

1. **Images.** Pull A549 Cell Painting images from the Cell Painting Gallery
   (`cpg0004-lincs`) on S3 (`s3://cellpainting-gallery/cpg0004-lincs/...`,
   `--no-sign-request`). This is the large step (hundreds of GB).
2. **DINO embeddings.** Encode each well/site with the **same** DINO backbone
   the authors used (identify it from the deposit/paper; if it is public
   DINOv2 ViT-g/14, use `torch.hub` `dinov2_vitg14`). Aggregate to well/compound
   level to match how JUMP embeddings were produced (`dino_raw_morphology.h5ad`).
3. **TVN.** Apply Typical Variation Normalisation using A549 negative controls
   (DMSO), mirroring `dino_tvn_morphology.h5ad`. Reuse
   `phenocompass.evaluation.map_preprocessing` (TVN / sphering utilities).
4. **Score.** Feed the 1536-dim A549 vectors to
   `PhenoCompass.compute_morph_embeddings(...)`, then score against the MoA
   anchors — reuse `mvp_core.score_morph_against_anchors`,
   `evaluate_cross_line`, `evaluate_fewshot_retrieval` (already written; they
   are modality-agnostic and take embeddings).

## Open items to verify first

- **Which DINO backbone / weights** produced the 1536-dim embeddings (public
  DINOv2 ViT-g/14 vs a custom checkpoint). Check the code tarball
  (`phenocompass-v0.1.0-code.tar`) and the paper methods.
- **Exact aggregation** (site→well→compound) and channel handling used for the
  JUMP embeddings, so A549 is produced identically.
- **TVN reference** (per-plate DMSO) for A549.

Everything downstream of step 3 (scoring, metrics, figures) is already
implemented in this folder and will be reused unchanged.

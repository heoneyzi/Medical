# Frozen Foundation Models: STATE & STACK (and the one-hot → ESM-2 fix)

This repo integrates Arc's frozen foundation models **as external predictors**: you
run the model once, save its predicted AnnData, and pass it via `--fm-pred`. The
adapter (`src/vcc_baselines/adapters/`) turns that prediction into per-perturbation
deltas and reuses the same predict → `cell-eval prep` → ladder pipeline.

We don't vendor Arc's checkpoints (too large / their license). Install and download
from the upstream repos:

- STATE — https://github.com/ArcInstitute/state
- STACK — https://github.com/ArcInstitute/stack
- cell-eval — https://github.com/ArcInstitute/cell-eval

---

## STACK — in-context, genuinely training-free

STACK (*"simulating cellular conditions via prompt engineering, without the need for
fine-tuning"*, Arc, Jan 2026) does **in-context learning at inference time**: it uses a
reference cell set as a "prompt" and predicts responses in a target set. This fits the
2026 setup, where each target context ships only its non-targeting controls.

```bash
stack-generation \
  --checkpoint /path/to/stack_pretrained.ckpt \
  --base-adata reference_context_with_knockdowns.h5ad \   # the "prompt": (control + measured KO) pairs
  --test-adata  target_context_controls.h5ad \            # the target line's NT controls
  --genelist    hvg_genes.pkl \
  --split-column condition \
  --output-dir  ./stack_out
```

Then convert the generation into deltas and package a submission:

```bash
python -m vcc_baselines predict --config configs/real_example.yaml \
  --method stack --fm-pred ./stack_out/generated.h5ad --prep
```

The FM prediction h5ad must be in the **challenge gene space** and carry the perturbation
label in `obs[target_gene]` with non-targeting controls labelled `non-targeting` (the
adapter computes `delta[g] = mean(cells where target_gene==g) − mean(controls)`).

**Prompt construction** for CRISPRi knockdown: put *(control, measured-knockdown)* example
cells from public CRISPRi you trust (e.g. Replogle, GSE281048) into `--base-adata`, and the
target line's controls into `--test-adata`. Validate the exact column/prompt convention
against STACK's example notebooks before trusting numbers, and check it in shadow-CV first.

> ⚠️ STACK uses the provided controls as a prompt at inference. Confirm on the **Rules**
> page that using the challenge-provided controls this way is permitted.

---

## STATE — and its one-hot perturbation limitation

STATE's State-Transition (ST) model predicts perturbation responses, but it encodes the
perturbation as a **learned lookup over a fixed vocabulary** (effectively one-hot). Inference:

```bash
state tx infer \
  --model-dir /path/to/run --checkpoint final.ckpt \
  --adata target_controls_preprocessed.h5ad \
  --pert-col target_gene --embed-key X_hvg \
  --output state_prediction.h5ad
```

Two things to get right before this is a valid CRISPRi baseline:

1. <b>Checkpoint modality.</b> The widely-shared ST checkpoint is trained on **Tahoe-100M
   (drugs)** → it predicts *chemical* responses. For VCC (CRISPRi knockdown) use a
   **CRISPRi-trained** checkpoint (e.g. Replogle/VCC), or you're transferring the wrong modality.
2. <b>Unseen genes.</b> Any 2026 target gene not in STATE's perturbation vocabulary has **no
   embedding** → no meaningful prediction. Fix it one of three ways:

### Fix 1 — Frozen ESM-2 nearest-neighbour (no retraining) ✅ implemented here

For an unseen gene `g`, substitute the nearest *seen* gene `g'` by ESM-2 protein-embedding
cosine similarity, and query STATE with `g'`. Fully training-free. This repo's **`esm2_nn`**
method does exactly this for the GWPS-matching path; the same idea maps STATE's vocabulary:

```python
z   = esm2_embed(protein_seq[g])
gp  = max(state_vocab, key=lambda v: cos(z, esm2_embed(protein_seq[v])))
# query STATE with gp instead of g
```

Enable real ESM-2 in the config:

```yaml
esm2:
  enabled: true
  model: facebook/esm2_t33_650M_UR50D
  fasta: data/esm2/proteins.fasta   # >GENE_SYMBOL\n<protein sequence>
  allow_random_fallback: false      # never allow random fallback for a real run
  device: cuda
```

(Offline, with no FASTA and `allow_random_fallback: true`, a deterministic *random*
embedding is used so the plumbing runs — it carries **no biology**; smoke tests only.)

### Fix 2 — Swap the perturbation encoder to ESM-2 + light adapter (semi-frozen)

Replace the one-hot lookup with a small projection `ESM-2(gene) → pert-embedding` and
fine-tune **only that projection (or a LoRA)** on public CRISPRi, backbone frozen. Now every
gene — seen or not — gets a continuous embedding. This is the modification many teams use;
it's no longer strictly "frozen" but is cheap and gives genuine zero-shot-over-genes.
Reference approach: *Efficient Fine-Tuning of Single-Cell Foundation Models Enables Zero-Shot
Molecular Perturbation Prediction* (arXiv:2412.13478).

### Fix 3 — Alignment (always, even frozen)

- **Gene space:** map challenge genes ↔ STATE's HVG space (`X_hvg`, ~2000 HVGs); reindex output back to the full challenge gene list (missing genes keep the control value).
- **Normalization:** match STATE's expected input normalization.
- **Controls:** feed covariate-matched non-targeting controls (never a global control).

---

## F6 / F7 — STATE-SE / STACK as a frozen CONTEXT encoder (recommended first FM baseline)

The cleanest way to use a frozen FM here is **not** to generate expression, but to
supply a better *context similarity* for GWPS transfer — isolating "does a better
cell representation improve zero-shot transfer?" (D4 raw/PCA vs F6/F7 embed, same
GWPS effect + same generator).

1. Embed each context's controls **and** each GWPS source's controls into one space. For the official SE-100M `model.safetensors`, use the strict wrapper in this repository because the current PyPI CLI searches for Lightning `*.ckpt` files:
   ```bash
   STATE_PYTHON=/path/to/state/tool/python
   "$STATE_PYTHON" scripts/state_safetensors_embed.py \
     --model-folder models/SE-100M --input contextA_controls.h5ad \
     --output A_state.h5ad --embed-key X_state --device cuda

   stack-embedding --checkpoint models/Stack-Large/bc_large.ckpt \
     --adata contextA_controls.h5ad \
     --genelist models/Stack-Large/basecount_1000per_15000max.pkl \
     --output A_stack.h5ad --device cuda
   ```
2. Pool the mean embedding per context/source into one npz keyed by name. STATE embeddings are in `obsm['X_state']`; STACK embeddings are in `X`:
   ```bash
   python -m vcc_baselines pool-embeddings \
     --input A=A_state.h5ad --input B=B_state.h5ad \
     --obsm-key X_state --out state_contexts.npz
   python -m vcc_baselines pool-embeddings \
     --input A=A_stack.h5ad --input B=B_stack.h5ad \
     --obsm-key X --out stack_contexts.npz
   ```
3. Point the config at it and switch the similarity space:
   ```yaml
   embed: { context_embeddings: data/embeds/ctx_emb.npz }
   predict: { method: gwps_weighted, similarity_space: embed }
   ```
   Now `gwps_weighted` weights K562↔RPE1 by frozen-FM similarity. Compare to
   `similarity_space: raw` / `pca` (= D4) to read off the FM's pure context value.
   The shadow benchmark requires all source and holdout keys. It reports a missing frozen artifact as unavailable and never substitutes random vectors or PCA.

For the complete, actually executed Jiang24 loop, use `bash scripts/run_frozen_contexts.sh data_shadow/jiang24_ifng_bxpc3_loco` and see `docs/SHADOW_VCC.md`.

## Always compare against the anchors

Whatever STATE/STACK produce, run them through `ladder` next to `control_mean`,
`global_mean`, and the `gwps_*` methods. If a frozen FM can't beat GWPS transfer (Rung 1),
that path isn't paying off yet — the single biggest lesson from VCC 2025.

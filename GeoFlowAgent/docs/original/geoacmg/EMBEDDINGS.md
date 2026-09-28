> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# Frozen embedding pipeline

## Why fieldwise caching

A single serialized prompt hides where information entered the system. Fieldwise arrays make it possible to remove goal, state, history, contract, background, entity, or DNA information without rerunning every encoder.

The cache builder serializes only model-visible content. Exact fields are also retained in structured JSON for masks and evaluation.

## Encoder roles

### General view

Use the final hidden state of a frozen instruction/base language model with explicit mean or last-token pooling. Qwen is not assumed to know exact assembly identity; it supplies procedural and linguistic context.

### Biomedical retrieval view

MedCPT has distinct query and article encoders. GeoFlowAgent preserves this asymmetry: example context uses the query encoder; tool documents and background cards use the article encoder.

### Entity auxiliary view

SapBERT is used for an explicit entity payload extracted from state keys such as gene, HPO, phenotype, variant, transcript, accession, and disease. It conditions the state gate and planner, but it is not automatically a tool prototype. Current and successor entity payloads are cached separately; the successor vector is a transition target, never a current-state input.

### DNA auxiliary view

DNABERT-2 receives actual nucleotide windows only. Strings such as `GRCh37` are not DNA and must not be sent to the DNA encoder. DNA vectors have no tool-side prototype in the MVP; they condition compatibility and context.
The loader accepts IUPAC nucleotide symbols and rejects other non-empty sequence text. Generic natural-language minimal pairs are not encoded by the DNA view; add a separately reviewed sequence-pair protocol when testing sequence sensitivity.
Current and successor sequence arrays are also distinct, even when the window is unchanged in a particular fixture.

## Cache integrity

`manifest.json` binds the cache to:

- the processed manifest SHA;
- exact row IDs/order;
- immutable encoder revisions;
- pooling and normalization;
- the feature serializer/missing-modality/background-pooling specification version;
- file dimensions/dtypes/checksums.

`EmbeddingCache(..., verify=True)` recomputes every array checksum and validates its actual
shape, dtype, row/index count, and finite values before training. Metric and flow checkpoints store
the same source/config/feature/tool bindings; agent evaluation also verifies that the exact metric
checkpoint bytes match the SHA recorded by the flow checkpoint.

Each small-model checkpoint additionally stores a canonical cache-content digest over the full
manifest, including every array SHA. This closes the case where the same encoder config and
processed source are re-extracted into numerically different bytes on another runtime.

The builder returns an already verified cache when both hashes match and refuses to overwrite it when either the embedding config or processed manifest differs. Prefer a directory named by a short config hash in larger experiments.

Offline training and closed-loop runtime pool fields and background cards with the same rule. A regression test enforces this; otherwise a seemingly minor weighting difference would create an avoidable train/inference shift.

The regression covers every cached trajectory prefix—including non-empty histories—and every
prototype/auxiliary view. Changing serialization or pooling requires a feature-spec/cache version
bump and a fresh cache.

## Runtime states after precomputation

The cache covers the reviewed trajectory prefixes. At evaluation time an exact hash over task/query/goal, typed state, visible history, and background IDs selects those cached features. Under `cache_only`, a fully covered finite benchmark runs the small geometry/flow modules without loading Qwen again. The default `prewarm_runtime_encoders: false` also leaves online-capable policies lazy, so a completely covered run does not load the large backbones. For a separate warm-latency comparison, set it to true and disclose that setting; otherwise the first real miss includes cold model loading in its planner latency.

On the first actual miss, the runtime loader compares its complete query/document encoder metadata
with the metadata that produced the cache. This includes resolved revision, implementation classes,
pooling, normalization, loaded dtype, and device. A host/dtype mismatch fails before a new vector is
used; it is not “fixed” by merely casting the final array to the cache storage dtype.

An erroneous action, tool failure, or injected perturbation can create a context that was never cached. Choose this behavior explicitly with `agent_evaluation.runtime_embedding_policy`:

- `cache_only`: fail on the first uncovered state; useful for coverage audits and a truly frozen offline deployment;
- `cache_or_encode`: use cache hits and run the pinned frozen encoders on misses;
- `encode`: bypass the lookup to measure forced online encoding cost.

Online field embeddings are cast through the configured cache storage dtype before pooling. Thus a
float16 research cache and an online miss do not silently use two different numeric spaces. Reports
name these events `runtime_embedding_miss_calls`: one means one planner context missed the cache,
not one Transformer layer forward.

For a no-Qwen runtime that must recover from new states, either enumerate and cache the versioned reachable state graph or train/evaluate a separate incremental distillation encoder. This repository does not silently substitute a hash vector for an unseen real state.

## Evaluation layers

### Raw representation audit

No learned projection. Compare original encoder spaces, train-only whitening, and exact-mask/no-mask conditions.

Raw contexts use the same query/goal/state/history and selected-background pooling as downstream
training. The default report slices to train/dev before geometry, CKA, probe, retrieval, and PCA.
Controlled pairs require a split and are aggregated only when that split is unsealed. Test enters
only through an explicit `--include-test` final run; unknown regret predictions report both null
`regret@1` and zero label coverage.

Ridge whitening is represented in the low-rank training subspace, algebraically avoiding a dense
`D×D` eigendecomposition for high-dimensional Qwen features. Linear CKA selects a primal or dual
formula according to the smaller intermediate. These are scalability changes, not test-fitted
approximations.

### Decoder-only comparison

Generating one fixed set of flow anchors and then swapping only decoder energy would isolate
decoder-family effects, but that diagnostic is not automated in this MVP. The implemented flow
decoder is a common cosine codebook for every upstream energy family.

### End-to-end native geometry

Train a parameter-matched projector and flow for each geometry. Interpret this as the joint effect of representation mapping, metric, and flow—not as a pure distance-function comparison.

## Minimal-pair interpretation

Input minimal pairs test controlled sensitivity, not causal effects of an action. Use causal or interventional language only when the same prefix is replayed under `do(action=a)` with actual successor outcomes.

## Recommended reporting

For every view/metric report:

- model/revision and input fields;
- raw dimension and projected dimension;
- trainable parameter count;
- fitted IDs for scaler/whitener/projector;
- invariant and sensitive pair distributions;
- set-valued retrieval and regret;
- held-out-tool performance;
- downstream flow and execution metrics.

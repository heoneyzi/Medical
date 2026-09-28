> 📜 **Original run log** — stage ① (synthetic pilot), kept as written. Machine paths are `$GEOFLOW_*` placeholders; the checkpoints and caches it lists by hash are kept with the large artifacts of the local research archive.

# Frozen Qwen-1.5B multi-view run state

Last updated: 2026-09-14 17:05 UTC

## Purpose

Validate the real frozen Qwen/MedCPT/SapBERT/DNABERT-2 embedding and downstream
geometry/flow pipeline on the replay-checked synthetic genomics fixture. These
results are engineering evidence only, not clinical or MARRVEL benchmark evidence.

## Storage layout

- Active source/config: `$GEOFLOW_PROJECT_ROOT` (stable link: `$GEOFLOW_PROJECT_ROOT`)
- Persistent state outside `/tmp`: `$GEOFLOW_STATE_DIR/GeoFlowAgent_RUN_STATE.md` and
  `$GEOFLOW_STATE_DIR/GeoFlowAgent_run_state.json`
- Persistent source/config recovery bundle outside `/tmp`:
  `$GEOFLOW_STATE_DIR/GeoFlowAgent_resume_bundle_20260914.tar.gz` (640 KiB, 138 entries),
  SHA-256 `a4a88100f2c07437a14171f0067a2f0e69f744df54afab6d5638d386ba01b519`.
  It intentionally excludes large artifacts, environments, and model caches.
- GPU environment: `$GEOFLOW_RUNTIME/venv-gpu`
- Hugging Face cache: `$HF_CACHE_DIR`
- Large artifacts: `$GEOFLOW_ARTIFACTS/frozen_smoke_qwen15_multiview`
- Stable artifact link: `$GEOFLOW_PROJECT_ROOT/artifacts/frozen_smoke_qwen15_multiview`

## Current state

- Environment: Torch 2.6.0+cu124, torchvision 0.21.0+cu124,
  Transformers 4.57.6, NumPy 2.2.6; CUDA smoke matmul PASS on RTX 3090
- Tests: PASS (95 tests after compatibility and flow-source-scale tests)
- Dataset prepare: PASS (12 tasks, 103 prefixes; train/dev/test tasks 7/2/3)
- Frozen embedding: PASS
  - 28 arrays, 3,332,966 bytes; every shape/checksum/finite-value check passed.
  - Cache content SHA-256: `d1c692e9f9977246307f647a54ce722680facbaa6db588cfdb53442af9968ab0`
  - Cache manifest SHA-256: `085fb33421932107530de3413a26b4346d3dee2f2964ccf7b84f640939a88844`
  - Config SHA-256: `579a580a3ed886c818399eb6ab6474a855c521151409e0586b54f94ddc19f572`
  - Processed manifest SHA-256: `9a30bf39d02f7cfa77a7876aeedd9fe4c39b3b42689a59cefe27a022d819f5e1`
  - Qwen 1.5B: 1536-D BF16 on cuda:0.
  - MedCPT/SapBERT: 768-D FP32 on cuda:1.
  - DNABERT-2: 768-D FP32 eager attention on cuda:1. Its unused pooler is
    randomly initialized by upstream code; cached vectors use last hidden states.
- Embedding audit: PASS (final report now includes the once-unsealed test split)
  - Final report SHA-256: `a96a48dbf7677302b38ffe18a84fe7f33816edd142d88f32fb1d26fa1955614e`
  - Qwen dev no-mask valid-hit@1: cosine `0.0667`, whitened `0.1333`;
    ridge-probe valid-set accuracy `0.6667`.
  - MedCPT dev no-mask valid-hit@1: cosine/whitened `0.1333`;
    ridge-probe valid-set accuracy `0.3333`.
  - Exact contract mask gives `1.0` valid-hit@1 for both prototype views,
    demonstrating that this fixture's exact constraints are highly informative.
  - Qwen is strongly anisotropic (state anisotropy `0.9921`, effective rank
    `3.51`); MedCPT is also anisotropic (`0.9787`, effective rank `10.25`).
  - Test no-mask cosine valid-hit@1: Qwen `0.1739`, MedCPT `0.0870`;
    ridge-probe valid-set accuracy: Qwen `0.6522`, MedCPT `0.5217`.
- Distance sweep seed 17: PASS (dev only; test sealed)
  - Comparison SHA-256: `ba57e748039cda187c184533353742b02dc112ed9c2dc4a14f3ee76911b68aee`
  - dev valid-set accuracy / regret@1: Euclidean `0.9333/0.0667`, cosine
    `0.6667/0.3333`, diagonal `0.9333/0.0667`, low-rank Mahalanobis
    `1.0000/0.0000`, bilinear `0.9333/0.0667`, Poincare `0.5333/0.4667`.
  - Provisional dev selection: `lowrank_mahalanobis` (3,925,253 parameters).
  - This is only one seed and 15 dev rows; multi-seed stability is still required.
- Selected metric training: PASS (`lowrank_mahalanobis`, best epoch 35,
  dev valid-set accuracy `1.0`, regret@1 `0.0`)
  - Checkpoint SHA-256: `269c3f5fad4cbfa56dee91132cbda5504d0ff4539ca1d1db1b70e1cd57aa281b`
- Baseline flow training: COMPLETED BUT FAILED SUCCESS CRITERION
  - best epoch 1; dev valid-first-action `0.1176`, stop-balanced accuracy
    `0.75`, normalized edit distance `0.7994`.
  - Checkpoint SHA-256: `292f8cb6d60f9589f1e401af15a744c867acbbe6b21f673a887d61ee3a29242a`
  - Diagnosis: 60 rows, batch 32, LR 1e-4 and patience 20 allowed too few
    effective updates; loss was still decreasing when early stopping fired.
- Tuned flow optimization diagnostics: PASS
  - v1 (standard noise, tool weight 0.5): dev first-action `0.4706`, edit `0.4736`.
  - v2 (noise 0.0625): first-action `0.4706`, edit `0.6394`, endpoint error
    `3.06`; scale matching alone did not improve action decoding.
  - v3 (noise 0.0625, tool weight 2): first-action `0.5882`, edit `0.5647`.
  - v4/v5 (noise 0.25, tool weight 2): selected. Dev-calibrated NFE 16 and
    STOP threshold 0.3 give first-action `0.7059`, stop balance `0.9333`, edit
    `0.3709`, suffix EM `0.2353`.
  - v6 (tool weight 4): worse (`0.6471`, edit `0.5164`).
  - Selected v5 checkpoint SHA-256:
    `0681ce0444a73bb1ef9b67f8df42de97411088da7597282493dfb0a750b1ce17`
- Selected-flow dev closed loop: PASS, but only 2 dev tasks
  - metric closed-loop task success `1.0`;
  - flow open-loop `0.0`, commit-1 `0.5`, commit-2 `0.0`, compute-matched
    blind replan `0.0`;
  - commit-1 contract-valid action rate `0.90`; no-mask `0.55`.
  - This supports a feedback benefit on the fixture, not a general conclusion.
  - Summary SHA-256: `5154f7230f9e5538987700991b46df412f7e0cc2510670d5ecbe45eff608a061`
- Multi-seed metric stability: PASS (seeds 17/41/73; test sealed)
  - Mean dev valid-set accuracy ± sample SD: bilinear `0.9111±0.0385`,
    low-rank `0.8667±0.1333`, diagonal `0.8222±0.1018`, Euclidean
    `0.8000±0.1155`, cosine `0.7556±0.1018`, Poincare `0.7556±0.2341`.
  - Bilinear also has the lowest mean regret@1 (`0.0889`).
  - Seed-17 low-rank selection was not stable; final family selection is changed
    to `bilinear` before test unsealing.
  - Seed-41/73 comparison SHA-256: `3d9345dac3c18951e86a1177ef3c25210098e6ebf631f72c79d3304a8568b6de`,
    `e000adf41ece945e7c7b27f517a588e3be03cacdfbe097b35a036a3082e3d52b`.
- Multi-seed bilinear flow stability: PASS, with weak downstream transfer
  - Seeds 17/41/73 dev first-action: `0.4118/0.5882/0.4706`
    (mean `0.4902`, sample SD `0.0899`).
  - Mean normalized edit distance `0.5620`; mean stop-balanced accuracy `0.7278`.
  - Strong metric ranking did not automatically produce a strong joint flow;
    this is a negative result that motivates decoder-only or more-data studies.
- Final synthetic test unsealing: PASS (one-time report after fixing family and
  hyperparameters)
  - Bilinear metric test valid-set accuracy for seeds 17/41/73:
    `0.9565/0.8696/0.7391` (mean `0.8551`, sample SD `0.1094`).
  - Flow test valid-first-action accuracy: `0.3846/0.4231/0.5000`
    (mean `0.4359`, sample SD `0.0588`).
  - Flow test normalized edit distance: `0.5895/0.5291/0.6178`;
    stop-balanced accuracy: `0.5000/0.9348/0.5000`.
  - Test closed-loop task success for metric / flow commit-1 / flow commit-2:
    seed 17 `1.0/1.0/0.6667`, seed 41 `1.0/0.3333/0.3333`, seed 73
    `1.0/1.0/0.6667`. Across seeds, flow commit-1 mean is `0.7778`.
  - Flow open-loop, no-contract-mask commit-1, and compute-matched blind replan
    all have mean task success `0.0`. Exact contracts and replanning are doing
    substantial work; the fixture has only 3 test tasks, so uncertainty is large.
  - Current seed 17/41/73 metric checkpoint SHA-256:
    `a0b988e8c3e10033cd480c13e239284c416bc568890251ccb3f42c0e73ed3617`,
    `bac70a598ac94f6f6be3b8cab6c27209d909c4ff7116cd2934cbc85ec3d56469`,
    `18136393046bc90983f7d69804f046c3e8abc8e0601648c280c5664cd20ec543`.
  - Current seed 17/41/73 flow checkpoint SHA-256:
    `8ec8c4a9e6decc42a24881a7f0bb68c453606bc4e527403a02f7ac9fc9eb1f4d`,
    `c65103a4253f7c7ac10d1d036d1d492a001d6283ab6280dcba7a268472641f46`,
    `2a5e270fdb6829e9b3b72863176d6e40eacd666872d794f5b25a95bf6958a5ee`.
  - Metric→flow→agent provenance-chain verification: PASS for all three seeds.
- Frozen Qwen2.5-7B backbone ablation: EMBEDDING/AUDIT PASS
  - Separate large-artifact path: `$GEOFLOW_ARTIFACTS/frozen_qwen7b_multiview`
    (stable link: `$GEOFLOW_PROJECT_ROOT/artifacts/frozen_qwen7b_multiview`).
  - Config: `configs/frozen_qwen7b_multiview.yaml`; SHA-256
    at embedding time `f8cbff3ddf2639be99d624d7d435ffa6a34314e07ca3e068508f90c3468ad0b1`;
    current downstream-matched SHA-256
    `a8e7ae9e780e7beff6990328ea7e0e65f32f84d8f1d1e665b9b51ae2b2cf3472`.
  - Qwen revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
  - Dataset prepare: PASS; processed manifest remains
    `9a30bf39d02f7cfa77a7876aeedd9fe4c39b3b42689a59cefe27a022d819f5e1`.
  - Embedding ran from `2026-09-14 16:17 UTC` to about `16:40 UTC`; the four
    pinned shards (about `14.19 GiB`) downloaded successfully, loaded on GPU 0,
    and the process exited 0.
  - Immutable cache verification: PASS; 27 arrays, 5,561,186 bytes; Qwen 3584-D.
    Cache content SHA-256:
    `964f4c67bc43bc08aa9abd5be2324933ee8cdff02a939bff5db0f45c61a8a7e9`;
    manifest SHA-256:
    `1a0c805f8fb50ea1619958ef1ab239e61dc7670f72af04b38848052f7253c21c`.
  - Dev-only embedding audit: PASS; report SHA-256
    `a2e0b3437f4564a7e6e1b9e0eb6bbada8e34d979af3a546d23905fdf0548efee`.
    Qwen no-mask cosine/whitened valid-hit@1 `0.1333/0.0667`, ridge valid-set
    accuracy `0.8000`, state anisotropy `0.9758`, effective rank `4.35`.
  - Relative to 1.5B dev audit, ridge accuracy improved `0.6667→0.8000` and
    anisotropy fell `0.9921→0.9758`; raw nearest-neighbour hit@1 remains weak.
  - Multi-seed distance sweep: PASS (seeds 17/41/73; test sealed). Mean dev
    valid-set accuracy ± sample SD: Poincare `0.9556±0.0385`, low-rank
    `0.9111±0.0385`, bilinear `0.9111±0.0770`, Euclidean `0.8444±0.0770`,
    cosine `0.8222±0.1018`, diagonal `0.8222±0.0385`.
  - Selected 7B metric family: `poincare` (mean regret@1 `0.0444`). Comparison
    SHA-256 for seeds 17/41/73: `39d8cc90f97b669d5de6b9867d313ecc5671b507ccd33f14a8aeea2c98a0324c`,
    `2091b8cf913d5b996df11489d5a8754ab133692a5f9c0d2f949e83fa4cf52a6b`,
    `694d49f49bef4f22b09e9d385e6c8106a624922c0da93521a0226fd07e5dae24`.
  - Poincare metric checkpoints trained for seeds 17/41/73; dev valid-set
    accuracy `1.0000/0.9333/0.9333`.
  - Poincare flow dev first-action `0.4706/0.4706/0.5294` (mean `0.4902`,
    sample SD `0.0340`); mean STOP balance `0.8556`, suffix EM `0.1569`, edit
    distance `0.6444`.
  - Poincare flow checkpoint SHA-256 for seeds 17/41/73:
    `d8bedc5ff1ca165b8ffc5eb073d3e1f9f9b9ae449a8fe098875b6ddbde7e87ad`,
    `e1bf4e3a1affec364bcb799b0985ad314e450cbd2413503075d50ea25930e7aa`,
    `b0d8e2ab87369f86190b890b95fb6cb76891ec074dcec3cc44d26f9c5272ab74`.
  - Interpretation so far: 7B Poincare and 1.5B bilinear have the same mean
    dev first-action (`0.4902`), but this is not an encoder-only contrast because
    the selected geometry differs.
  - Same-geometry 7B bilinear flow control: PASS. Dev first-action for seeds
    17/41/73 is `0.4706/0.6471/0.5294` (mean `0.5490`, sample SD `0.0899`);
    mean edit distance `0.5086`, STOP balance `0.6444`, suffix EM `0.1373`.
  - Relative to 1.5B bilinear, 7B bilinear improves mean first-action
    `0.4902→0.5490` and edit distance `0.5620→0.5086`, while STOP balance falls
    `0.7278→0.6444`. The sample is small and uncertainty remains high.
  - 7B bilinear flow checkpoint SHA-256 for seeds 17/41/73:
    `673ce4c4212cccb30ef6e3750e04a667175162b54e1a59b1cf0353773b6d2921`,
    `a01baac9b4e892175af5933fc1d1ce2d95c0ab937df62b39da5f44efa63c35ac`,
    `9eb3e727489d8a282847785ead82d4a257e6f0efc529e23fd22b5d261c6da97c`.
  - The Poincare metric ranks better than bilinear (`0.9556` vs `0.9111`) but
    its flow ranks worse (`0.4902` vs `0.5490` first-action; `0.6444` vs
    `0.5086` edit), independently reproducing weak metric-to-flow transfer.
  - Dev closed-loop (2 tasks/seed): metric policy succeeds `1.0` for both
    geometries. Poincare flow commit-1/commit-2 succeeds `0.0/0.0` for every
    seed; bilinear commit-1 is `0.5/1.0/0.5` (mean `0.6667`) and commit-2 is
    `0.5/0.5/0.0` (mean `0.3333`). Open-loop/no-mask/blind replan are all `0.0`.
  - Final 7B end-to-end selection before test unsealing: `bilinear`. Poincare
    remains the metric-only winner and is retained as a negative transfer result.
  - Final one-time 7B bilinear test: PASS, with no post-test tuning. Metric
    valid-set accuracy for seeds 17/41/73 is `0.8261/0.6957/0.7826` (mean
    `0.7681`, sample SD `0.0664`). Flow first-action is
    `0.3846/0.5385/0.5000` (mean `0.4744`, sample SD `0.0801`); mean edit
    distance `0.5703`.
  - Final 7B test closed-loop: metric policy `1.0` for every seed; flow commit-1
    `1.0/0.0/0.3333` (mean `0.4444`), commit-2 `0.6667/0.0/0.3333` (mean
    `0.3333`). Open-loop/no-mask/blind replan remain `0.0`.
  - Current final metric checkpoint SHA-256 for seeds 17/41/73:
    `a49f15a98996001f9bf2beed6781b1e69a3740ae59c7fd55ebcf25cdefc8a422`,
    `ec013bafac62c47e1e788903226daea7708d586c82a53061c3b8f657c7b063b1`,
    `f6179b4021906324e9e8981025fcb777b3f4a06178146b9c3b27c3a301d27549`.
  - Current final flow checkpoint SHA-256 for seeds 17/41/73:
    `778b26fc94e7a38ec2d0abd69b5a27bbc1821456282bbecfe5d9e69c10864636`,
    `c8a7fed59c117c1a66e23bc435c8b22af6cb7e5e53bbb173d6032da1746542dc`,
    `0543b66c612c84c5a72b132602fc3a21a7ad5f10f268be3ed01cd50467d6980c`.
  - Final agent report SHA-256 for seeds 17/41/73:
    `c743db6e88b4f20e25f2c299762a8893dbbbd4b5803aff05dc7436226c8de082`,
    `856680ffb3204f8d5d0c6e55bb797abc128093882380d69fb4fcc404b5dceced`,
    `51396f0c226c49b3f22b72c5720cc95ba40c97eec588344fc8ac085668acc19d`.
  - Final embedding report SHA-256:
    `eadf57a4a8442354277e03468ae3f2087bfe812bf0046ee830caab750e092235`;
    Qwen test no-mask cosine hit@1 `0.0870`, ridge valid-set accuracy `0.6522`.
  - Metric→flow→agent provenance-chain verification: PASS for all seeds.
    MedCPT (10 arrays), SapBERT (5), and DNABERT-2 (2) are byte-identical to
    the 1.5B experiment; only the Qwen view/backbone changes.
  - Final comparison: 7B improves same-geometry flow test first-action
    `0.4359→0.4744`, but metric test falls `0.8551→0.7681` and closed-loop
    commit-1 falls `0.7778→0.4444`. Larger frozen embeddings are not a uniform
    win on this fixture.

## Resume protocol

1. Confirm `/tmp` paths above still exist. If the artifact path exists, verify the
   immutable cache before doing anything else:

   ```bash
   cd $GEOFLOW_PROJECT_ROOT
   .venv-gpu/bin/python -c "from geoflowagent.embeddings.cache import EmbeddingCache; [EmbeddingCache(p, verify=True) for p in ('artifacts/frozen_smoke_qwen15_multiview/cache', 'artifacts/frozen_qwen7b_multiview/cache')]"
   ```

2. If `/tmp` was cleared, restore the source/config snapshot, recreate the isolated
   environment, and rerun `prepare` and `embed` with
   `HF_HOME=$HF_CACHE_DIR`:

   ```bash
   mkdir -p $GEOFLOW_PROJECT_ROOT
   tar -xzf $GEOFLOW_STATE_DIR/GeoFlowAgent_resume_bundle_20260914.tar.gz \
     -C $GEOFLOW_PROJECT_ROOT
   ```
3. The Qwen-1.5B and Qwen-7B synthetic engineering studies, including their
   one-time test reports, are complete. Do not tune against either test. The next
   scientifically meaningful stage is to supply typed, executable real MARRVEL
   trajectories; the present 100-row public curation queue lacks those labels.

   ```bash
   .venv-gpu/bin/geoflow finalize-marrvel --help
   ```

The machine-readable companion is `run_state.json`; update both files at every
stage boundary.

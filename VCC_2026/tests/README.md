<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [VCC_2026](../README.md) › **tests**</sub>

# ✅ tests

22 pytest checks that need no downloads, GPU or model weights. Run `pytest` from the project root. They passed on 28 Sep 2026 in a fresh Python 3.11 environment with only the core dependencies (`anndata`, `scanpy`, `numpy`, `scipy`, `pandas`, `pyyaml`, `tabulate`).

| File | What it guards |
|---|---|
| [`test_smoke.py`](test_smoke.py) (3) | End-to-end raw-count pipeline on synthetic data (non-negative integer counts, no control cells in the submission, effect transfer beats the no-effect anchor on discrimination); all three generators emit integer cells of the right shape; the ladder runs across contexts |
| [`test_shadow.py`](test_shadow.py) (13) | **Leakage barriers**: held-out responses never reach model inputs, challenge-data paths are rejected, zero-shot bundles seal every response, and scoring only runs on an existing prediction. Also covers raw-count checks beyond the first rows, sparse alignment, stable effect-library source order, embedding pooling, the exact six `cell-eval2` metric members, and target-gene map conflicts |
| [`test_external_frozen.py`](test_external_frozen.py) (6) | Frozen-encoder harness: stable seeds, deterministic sparse pooled generator without zero-locking, exact Replogle projection, command-adapter validation and reuse, preflight that fails closed on a missing runtime asset or a checksum mismatch, and rejection of non-count inputs before any model call |

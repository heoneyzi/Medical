# Reproduction guide

End-to-end reproduction of the TDiG settling-analysis pipeline. This document is the canonical reference for what runs where, in what order, with what inputs/outputs, on what hardware. Update on any pipeline change.

---

## TL;DR

| Item | Value |
|---|---|
| Total wall time (aggressive plan) | **~60–75 min** on H200 (batched + concurrent) |
| Total storage required (server side) | ~120 GB cache + 13 GB weights |
| Hardware | DigitalOcean H200 (141 GB GPU memory) |
| Pipeline scripts | 10 scripts (`scripts/01..60_*.{sh,py}`) + `run_pipeline.sh` |
| Main artifacts | 14-cell settling table + per-layer scalars + raw subset + Σ stats + variant raw |
| Execution mode | **3-phase: smoke → chr22 + cross-arch concurrent → chr17 + variants concurrent** |
| Batching | up to 16 sequences per forward (vs gDTR-PoC's batch=1) |

---

## Hardware and environment

### Server (compute)

| Item | Value |
|---|---|
| Provider | DigitalOcean GPU droplet |
| Droplet | `<gpu-host>` (revived from gDTR-PoC snapshot) |
| IP | `<server-ip>` (current; previous droplets had different IPs) |
| GPU | NVIDIA H200, 143771 MiB total |
| OS | Ubuntu (snapshot baseline) |
| venv | `/path/to/gDTR/venv/` (shared with gDTR-PoC, contains evo2 0.3.0 + torch 2.4.1+cu124) |
| Python | 3.10.12 |
| Model weights | `~/.cache/huggingface/hub/models--arcinstitute--evo2_7b_base/` (13 GB) |
| Model HF revision | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` |
| Model weights MD5 | `359ef88ccac2a62644035578de8a7db4` (1M variant; loaded variant is `evo2_7b_base`) |
| SSH config (local) | alias `<gpu-host>` in `~/.ssh/config` |

### Local (orchestration + analysis)

| Item | Value |
|---|---|
| Repo path | `/path/to/TDiG/` |
| GitHub | `https://github.com/YAICON-8th-Think-Deep-in-Genome/TDiG` (private) |
| Python | matches `pyproject.toml` (≥ 3.10) |
| Required for | post-processing (S2–S6), visualization, downstream tasks |

### Server file layout (gitignored on the local repo)

```
/path/to/gDTR/                            # existing gDTR-PoC repo, kept for infrastructure
├── venv/                              # SHARED venv, used for TDiG forwards
├── scripts/                           # reference forward scripts (16_phase1_6_chr22_forward.py etc.)
├── results/                           # existing gDTR caches (chr22_cache.h5, ...) — read-only for TDiG
└── data/                              # GRCh38 chr22/chr17 FASTAs, GENCODE v44, ClinVar

/path/to/TDiG/                            # NEW: TDiG repo clone (created in 01_environment_setup.sh)
├── ...                                # mirror of github.com/YAICON-8th-Think-Deep-in-Genome/TDiG
└── data/cache/                        # TDiG output target — see "Output manifest" below
```

---

## Data versions

All inherited from gDTR-PoC `data/DATA_VERSIONS.txt`. Locked.

| Resource | Version |
|---|---|
| Reference genome | GRCh38 primary assembly (UCSC `hg38`) |
| Gene annotation | GENCODE v44 |
| ClinVar VCF | 2026-04-18 release |
| Conservation | PhyloP 100-way |
| Regulatory | ENCODE SCREEN v3 cCRE, GTEx v8 eQTL, GWAS Catalog |

---

## Pre-flight checklist

Before running any forward, verify the server is healthy:

```bash
ssh <gpu-host> '
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader && \
  source /path/to/gDTR/venv/bin/activate && \
  python -c "import evo2; print(evo2.__version__)" && \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())" && \
  test -d ~/.cache/huggingface/hub/models--arcinstitute--evo2_7b_base && echo "weights OK" && \
  df -h /root | tail -1
'
```

Expected output:
- GPU: NVIDIA H200, 143771 MiB total
- evo2: 0.3.0
- torch: 2.4.1+cu124, CUDA available True
- weights OK
- ≥ 130 GB free on `/root`

---

## Pipeline architecture (3 phases, batched + concurrent)

The pipeline is designed around three sequential phases. **Within each phase** scripts run concurrently using CUDA streams and async I/O. **Between phases** we gate on sanity verification before committing more compute.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE A — Smoke test (5 min)                                              │
│   11_smoke_batched.py   10 chr22 windows × batch=8 vs batch=1 comparison  │
│   Decision gate:        batched output matches batch=1 within ε=1e-5      │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓ pass
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE B — chr22 + cross-arch concurrent (15–20 min)                       │
│   15_chr22_forward.py --with-crossarch                                    │
│     ├─ Stream A: Evo 2 chr22 main forward (batch=8–16)                    │
│     └─ Stream B: HyenaDNA + NT-v2 + DNABERT chr22 forward (concurrent)    │
│   Decision gate:        chr22 splice donor vs intron Cohen's d ≥ 0.20     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓ pass
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE C — chr17 + variants concurrent (35–50 min)                         │
│   16_chr17_forward.py    chr17 main forward (batch=8–16)                  │
│   18_variant_forward.py  ClinVar variants ref/alt with kv-cache reuse     │
│   These run on the same H200 — kv-cache cooperative scheduling            │
└──────────────────────────────────────────────────────────────────────────┘
```

The orchestrator is `scripts/run_pipeline.sh` which manages tmux windows, phase gating, and verification between phases.

## Pipeline scripts (execution order)

### `01_environment_setup.sh` — SERVER, one-time
Clone TDiG repo to `/path/to/TDiG/`, verify Phase 1 forward infrastructure still importable, create `/path/to/TDiG/data/cache/` directory, write `_provenance_baseline.json` (model SHA, package versions, host info).

### `05_select_subset.py` — LOCAL, one-time
Pick 100 representative chr22 windows for Tier 3 raw storage. Selection: stratify across all 7 context classes plus boundary regions. Outputs `data/subset_window_ids.json` (committed to repo).

### `10_population_stats.py` — SERVER, ~3–5 min batched
Run chr22 sanity sequences (100 sequences × 6 kb, batch=16) through Evo 2, compute and save:
- Per-layer mean h_ℓ (32 × 4096 fp32)
- Per-layer std h_ℓ
- Per-layer Σ_ℓ diagonal
- Per-layer Σ_ℓ Ledoit-Wolf shrinkage estimate (32 × 4096² fp32 ≈ 2 GB)
- Per-layer Σ_ℓ PCA-top-128 (32 × 128 × 4096 fp32 ≈ 64 MB)
- All TDiG γ thresholds (γ_dir, γ_mag, γ_M, γ_τ) via q70 calibration

**Required before PHASE A/B/C**. Locks calibration before any biological-context forward.

### `11_smoke_batched.py` — SERVER, ~5 min — **PHASE A gate**
10 chr22 windows × batch=8 batched forward. Same 10 windows also run at batch=1 for comparison. Asserts numerical equivalence (per-token D_cos / D_L2 / step_norm diff < 1e-5). If pass, batched path is verified; if fail, fall back to smaller batch or batch=1.

### `15_chr22_forward.py` — SERVER, ~15–20 min — **PHASE B**
Main chr22 batched forward. 12,978 windows × 6 kb, batch=8–16 (auto-adjusted from GPU memory probe). For each (window, layer, token) computes Tier 2 scalars; for 100 representative windows stores Tier 3 raw h_ℓ subsampled to 600 tokens.

CLI:
```
--batch-size N        explicit batch size (default: auto)
--with-crossarch      run HyenaDNA + NT-v2 + DNABERT concurrently on Stream B
--resume              skip already-completed windows
```

When `--with-crossarch` is set, this script handles step 17's work in the same run via CUDA-stream concurrency — saves ~10 min vs sequential.

### `16_chr17_forward.py` — SERVER, ~25–35 min — **PHASE C (part 1)**
chr17 batched forward with **trimmed Tier 2** (settling tables + Tier 3 subset + stats only; per-layer scalars skipped). chr17 storage: ~21 GB vs chr22's 70 GB.

Concurrent with `18_variant_forward.py` on the same GPU via CUDA streams + cooperative memory allocation.

### `17_crossarch_forward.py` — SERVER, ~10 min standalone (or 0 min if run via 15)
4-model Phase 4 replay on the same 12,978 chr22 windows: HyenaDNA-large, NT-v2 500M, DNABERT-2 117M. Same metric set but only Tier 1 settling tables (storage constraint).

Usually invoked via `15_chr22_forward.py --with-crossarch` for concurrent execution. Direct invocation is for re-runs only.

### `18_variant_forward.py` — SERVER, ~10–15 min — **PHASE C (part 2)**
ClinVar 10,910 variants × {ref, alt} forward at variant position ±5 bp. Uses **kv-cache reuse**: ref and alt share all tokens except the variant position, so kv-cache for the shared prefix is reused → 2–3x speedup vs naive ref/alt independent forwards.

Stores raw h_ℓ at variant position + scalar ||Δh_ℓ|| at ±5 positions per layer.

### `run_pipeline.sh` — SERVER, orchestrator
Bash script that:
1. Runs pre-flight checklist
2. Launches `10_population_stats.py`
3. Runs PHASE A (`11_smoke_batched.py`), checks gate
4. Runs PHASE B (`15_chr22_forward.py --with-crossarch`), checks gate
5. Launches PHASE C (`16_chr17_forward.py` + `18_variant_forward.py` in parallel tmux windows)
6. Final verification (`_verify_outputs.py --stage all`)
7. Writes `_done_full_pipeline.json` with timestamps + provenance

Total wall: **~60–75 min** if all phase gates pass.

### `20_run_gates.py` — LOCAL, ~10 min
S2 sanity gates (splice signal, entropy, motif) on all settling cells.

### `30_concordance.py` — LOCAL, ~5 min
S3 pairwise Spearman ρ + dissociation token mapping.

### `40_context_stratify.py` — LOCAL, ~5 min
S4 98-cell biological context heatmap data.

### `50_make_viz.py` — LOCAL, ~10 min
Generate V1–V9 figures.

### `60_downstream.py` — LOCAL, ~15 min
T-A splice prediction + T-B ClinVar variant classification.

### `_verify_outputs.py` — SERVER + LOCAL helper
- Schema check on every h5 output
- Splice donor vs intron Cohen's d on M1 × Ref C must match gDTR-PoC `phase1.6/gate_b.json` (-0.43 ± 0.05) — sanity that the batched + reduced-precision pipeline didn't drift
- Per-window completion count
- Provenance JSON validation

---

## Field reference (the 14 cells + their Tier 2 dependencies)

### Per-token-per-layer scalars (Tier 2, fp16 unless noted)

The pipeline stores **base scalars** rather than derived metric values, so that any metric variant (|r-1|, (r-1)², D_L2, |log r|, ...) can be re-computed without re-running forward.

| Field | Shape | Used by | Note |
|---|---|---|---|
| `norm_h_ell_raw` | (W, 32, T) | M2, M5, all `r` | $\|h_\ell\|_2$ |
| `res_norm_refA` | (W, 32, T) | M5×A, M6×A | $\|h_\ell - h_{29}\|_2$ — both raw |
| `res_norm_refB` | (W, 32, T) | M5×B, M6 derived | $\|\mathrm{RMSNorm}(h_\ell) - \mathrm{RMSNorm}(h_{29})\|_2$ — both normed |
| `res_norm_refC` | (W, 32, T) | M5×C, M6×C | $\|h_\ell - h_{\text{norm}}\|_2$ — asymmetric |
| `cos_refA` | (W, 32, T) | M1×A | $\cos(h_\ell, h_{29})$ — both raw |
| `cos_refB` | (W, 32, T) | M1×B | $\cos(\mathrm{RMSNorm}(h_\ell), \mathrm{RMSNorm}(h_{29}))$ — both normed |
| `cos_refC` | (W, 32, T) | M1×C | $\cos(h_\ell, h_{\text{norm}})$ — current gDTR |
| `step_norm` | (W, 31, T) | M3, M5 | $\|h_{\ell+1} - h_\ell\|_2$ |
| `step_cos` | (W, 30, T) | M3 curvature | $\cos(h_{\ell+1}-h_\ell, h_{\ell+2}-h_{\ell+1})$ |
| `entropy_ell` | (W, 32, T) | E8 gate, per-layer entropy | $H(\mathrm{softmax}(h_\ell W_U))$ |
| `top1_prob_ell` | (W, 32, T) | confidence proxy | max softmax probability |

W = 12,978 windows; T = 6000 tokens. Total per-field: ~5 GB. Sum: ~50 GB.

### Per-token scalars (saved once, fp32)

Constant across layers; small.

| Field | Shape | Note |
|---|---|---|
| `norm_h_29_per_token` | (W, T) | $\|h_{29}\|_2$ |
| `norm_rmsnorm_h_29_per_token` | (W, T) | $\|\mathrm{RMSNorm}(h_{29})\|_2$ |
| `norm_h_norm_per_token` | (W, T) | $\|h_{\text{norm}}\|_2$ |
| `context_label` | (W, T) | uint8 — index into 7 context classes |

### Tier 3 raw subset (100 windows)

| Field | Shape | Size | Note |
|---|---|---|---|
| `raw_h_ell` | (100, 32, 600, 4096) | 16 GB fp16 | Subsampled to 600 tokens/window (every 10th) |
| `raw_h_ell_rmsnormed` | (100, 32, 600, 4096) | 16 GB fp16 | RMSNormed version (for Ref B raw analysis) |

### Tier 4 population stats

| Field | Shape | Note |
|---|---|---|
| `per_layer_mean` | (32, 4096) fp32 | from chr22 sanity 100 seq |
| `per_layer_std` | (32, 4096) fp32 | same |
| `sigma_diagonal` | (32, 4096) fp32 | M4 fast |
| `sigma_ledoit_wolf` | (32, 4096, 4096) fp32 | M4 full (~2 GB) |
| `sigma_pca_top128` | (32, 128, 4096) fp32 | M4 reduced (~64 MB) |
| `sigma_inv_diag` | (32, 4096) fp32 | precomputed |

### Tier 5 variant raw

| Field | Shape | Size | Note |
|---|---|---|---|
| `variant_h_ell_ref` | (10910, 32, 4096) | 2.9 GB fp16 | Raw h at variant position, ref sequence |
| `variant_h_ell_alt` | (10910, 32, 4096) | 2.9 GB fp16 | Raw h at variant position, alt sequence |
| `variant_delta_norm_per_layer` | (10910, 32, 11) | 8 MB fp16 | $\|\Delta h\|_2$ at variant ± 5 positions |

---

## Output manifest (server paths → local sync paths)

```
SERVER: /path/to/TDiG/data/cache/                              LOCAL: data/cache/

├── population_stats/
│   ├── per_layer_mean.npy           (0.5 MB)
│   ├── per_layer_std.npy            (0.5 MB)
│   ├── sigma_diagonal.npy           (0.5 MB)
│   ├── sigma_ledoit_wolf.npy        (2 GB)
│   ├── sigma_pca_top128.npy         (64 MB)
│   └── gamma_calibration.json       γ_dir, γ_mag, γ_M, γ_τ (q70)
│
├── chr22/
│   ├── tier1_settling.parquet       14 cells × 78M tokens (1 GB)
│   ├── tier2_scalars.h5             11 fields × per-layer per-token (50 GB)
│   ├── tier3_raw.h5                 100 windows × 32 × 600 × 4096 + RMSNormed (32 GB)
│   ├── tier4_stats.json             chr22-specific aggregates
│   ├── window_metadata.parquet      (chr, start, end, n_tokens)
│   └── _done                        completion marker with run metadata
│
├── chr17/
│   ├── tier1_settling.parquet       (2 GB; trimmed for storage)
│   ├── tier3_raw.h5                 100 windows subset (16 GB)
│   ├── window_metadata.parquet
│   └── _done
│
├── crossarch/
│   ├── hyenadna_tier1.parquet       (smaller — 8 layers)
│   ├── nt_v2_tier1.parquet          (29 layers)
│   ├── dnabert2_tier1.parquet       (12 layers)
│   ├── per_model_summary.json
│   └── _done
│
├── variants/
│   ├── variant_raw_ref.h5           (2.9 GB)
│   ├── variant_raw_alt.h5           (2.9 GB)
│   ├── variant_delta_norms.parquet  (small)
│   ├── variant_metadata.parquet     (gene, chr, pos, ref, alt, class)
│   └── _done
│
└── _logs/
    ├── 10_population_stats.log
    ├── 15_chr22_forward.log
    ├── 16_chr17_forward.log
    ├── 17_crossarch_forward.log
    ├── 18_variant_forward.log
    └── _provenance.json              git SHAs, model SHAs, package versions, run timestamps

LOCAL SYNC                                       Pull command
data/cache/population_stats/                     rsync -avz <gpu-host>:~/TDiG/data/cache/population_stats/ data/cache/population_stats/
data/cache/chr22/                                same pattern
data/cache/chr17/                                same pattern (selective: tier1 + tier3 only)
data/cache/crossarch/                            same pattern
data/cache/variants/                             same pattern
data/cache/_logs/                                same pattern
```

Total server storage estimate:

| Component | Size |
|---|---|
| population_stats/ | 3 GB |
| chr22/ | 70 GB |
| chr17/ | 21 GB |
| crossarch/ | 20 GB |
| variants/ | 6 GB |
| logs | < 100 MB |
| Evo 2 weights (already there) | 13 GB |
| **Server total** | **~133 GB** |

Within 143 GB disk; OS + venv take residual.

---

## Resume / recovery

All forward scripts (10, 15, 16, 17, 18) implement:
- Per-window or per-batch `_done` marker writing
- On restart, skip already-completed units
- Atomic h5 writes (write to `.h5.tmp` then rename)

If a script is interrupted:
```bash
ssh <gpu-host> 'cd ~/TDiG && python scripts/15_chr22_forward.py --resume'
```

If the entire droplet is destroyed:
1. Spin up new H200 droplet from the gDTR snapshot
2. Update `~/.ssh/config` `<gpu-host>` `HostName` to new IP
3. Re-run pre-flight checklist
4. Resume with `--resume` flags

---

## Provenance and reproducibility metadata

Each forward script writes a `_provenance.json` sidecar:

```json
{
  "script": "15_chr22_forward.py",
  "run_timestamp": "2026-05-XX HH:MM:SS UTC",
  "host": "<gpu-host>",
  "ip": "<server-ip>",
  "tdig_git_sha": "<commit SHA at time of run>",
  "gdtr_git_sha": "<gDTR-PoC commit SHA>",
  "model": {
    "name": "evo2_7b_base",
    "hf_revision": "bda0089f92582d5baabf0f22d9fc85f3588f6b58",
    "weights_md5": "359ef88ccac2a62644035578de8a7db4"
  },
  "packages": {
    "torch": "2.4.1+cu124",
    "evo2": "0.3.0",
    "vtx": "1.0.8",
    "transformer_engine": "2.14.0"
  },
  "data_versions": {
    "grch38_chr22": "<hash or release date>",
    "gencode": "v44",
    "clinvar": "2026-04-18"
  },
  "seed": 42,
  "n_windows_total": 12978,
  "n_windows_completed": 12978,
  "wall_time_minutes": 85.3,
  "gpu_peak_memory_gb": 23.1
}
```

These sidecars are committed to the TDiG repo (small JSON files) so the full provenance trail lives in git history.

---

## Verification

After each forward stage, run verification:

```bash
# Server-side: check h5 files non-empty and schema match
ssh <gpu-host> 'cd ~/TDiG && python scripts/_verify_outputs.py --stage chr22'

# Local-side: spot-check 5 random windows for known-good values (re-derive M1 from chr22_cache.h5 reference)
python scripts/_verify_outputs.py --stage chr22 --against gdtr_baseline
```

Re-derivation against gDTR baseline:
- M1 × C settling depth must match Phase 1.6 c(t) within ±1 layer (floor due to fp16/fp32 differences)
- Splice donor vs intron Cohen's d (M1 × C) must match `gDTR-PoC results/phase1.6/gate_b.json` value -0.43 within ±0.05

If any verification fails, do not proceed to downstream stages.

---

## Cost summary

**Aggressive plan (batched + concurrent, target ~60–75 min wall):**

| Stage | Wall time | GPU memory peak | Disk produced |
|---|---|---|---|
| `01_environment_setup` (one-time) | < 1 min | 0 | < 1 MB |
| `05_select_subset` (local) | < 1 min | 0 | < 1 KB |
| `10_population_stats` | ~3–5 min | ~30 GB (batch=16) | 3 GB |
| `11_smoke_batched` (PHASE A) | ~5 min | ~30 GB | < 100 MB |
| `15_chr22_forward --with-crossarch` (PHASE B) | **~15–20 min** | ~50 GB (Evo 2 + 3 small models concurrent) | 70 + 20 = 90 GB |
| `16 + 18 concurrent` (PHASE C) | **~35–50 min** | ~40 GB (chr17 + variant streams) | 21 + 6 = 27 GB |
| **Server total** | **~60–75 min** | | **120 GB** |
| `20_run_gates` (local) | ~10 min | — | < 100 MB |
| `30_concordance` | ~5 min | — | < 100 MB |
| `40_context_stratify` | ~5 min | — | < 50 MB |
| `50_make_viz` | ~10 min | — | < 200 MB |
| `60_downstream` | ~15 min | — | < 100 MB |
| **Local total** | **~45 min** | | **< 500 MB** |

**Conservative fallback (sequential, batch=1, ~5.5h)**: if `11_smoke_batched` fails or aggressive plan exhibits OOM, fall back to batch=1 sequential. Same outputs, same correctness; just slower. The pipeline scripts auto-detect via `--batch-size 1 --no-concurrent` flags.

**Speedup sources (5–7× vs conservative):**
1. **Batching** (3–5×): batch=8–16 instead of batch=1 utilizes H200's 141 GB
2. **CUDA streams concurrent multi-model** (1.5×): chr22 Evo 2 + cross-arch 3-model on the same GPU
3. **kv-cache reuse for variants** (2×): ref/alt share all but variant position
4. **Async I/O**: write h5 in background while next batch computes

---

## Caveats and known limitations

- **chr17 is trimmed**: Only Tier 1 settling + Tier 3 100-window subset + Tier 4 stats. Full Tier 2 (per-layer scalars) skipped due to ~140 GB storage cost. If chr17 per-layer scalars become needed, re-run `16_chr17_forward.py` with `--full-tier2` (adds ~3h wall but requires ~21 GB more disk).
- **Phase 4 cross-arch is Tier 1 only**. No per-layer scalars stored for the 4-FM run.
- **Variant context is ±5 bp only** for stored raw h_ℓ. Full ±100 bp variant context would require ~63 GB; out of scope for this pipeline.
- **Mahalanobis M4 in two modes**: settling computed online from `sigma_inv_diag` (cheap, approximate); full Σ analysis must use Tier 3 raw subset (100 windows only).
- **Antiparallel diagnostic for M2** is run inside `20_run_gates.py`; if fraction > 5%, falls back to D_L2 audit per `docs/metric_definitions.md` M2 notes.

---

## What changes after the run

After successful completion of `15`–`18` and verification:
1. Update `PLAN.md` §7 open questions 1, 2, 4, 5, 10, 11 with their data-driven locks
2. Commit `_provenance.json` files for every stage
3. Tag the TDiG repo: `git tag v0.1-first-forward-run`
4. Proceed to `20`–`60` for analysis and visualization

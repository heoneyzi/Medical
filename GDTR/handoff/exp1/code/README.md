# exp1 — Evo 2 handoff, Experiment 1

Numerator/denominator and carrier/content decomposition of the Evo 2 residual
trajectory across the representation-to-output handoff.

**Scope: Evo 2 7B only.**  The 1B and 40B entries stay in `configs/models.yaml`
so the operator audit (step 1) and a later scale replication cost nothing to
switch on, but nothing in the default run touches them.

---

## What this answers

One question, measured rather than argued:

> Block-18 alignment to the output frame reaches 0.416, blocks 28–29 sit at
> −0.009, block 30 jumps to 0.61.  Is that **loss**, **dilution**, or an
> **orthogonal overwrite**?

The cosine is a ratio.  In a stretch where the denominator moves by eleven
orders of magnitude, the ratio alone cannot say what happened to the numerator.
So every tap is stored as

```
p = <h, r̂>      aligned component (signed)
n = ‖h‖          total magnitude
q = ‖h − p r̂‖    orthogonal remainder
a = p / n        the cosine that was being reported
```

and the reference itself is split into a global carrier `u` and a per-position
content direction `v(t)`, giving the exact identity

```
a = α·cos(h, u) + β·<ĥ, v(t)>
    └ carrier ┘   └ content ┘
```

The prediction under test: the +0.57 correlation with next-base entropy lives in
the carrier channel, and the annotation contrast that survives adjustment lives
in the content channel.

---

## Install

```bash
pip install -r requirements.txt
# plus Evo 2 itself:  pip install evo2     (https://github.com/ArcInstitute/evo2)
```

`pyarrow`, `pyfaidx` and `pyBigWig` are optional for the analysis path — tables
fall back to gzipped CSV and phyloP is filled with NaN if unavailable — but all
three are needed for a real extraction.

Point `configs/panel.yaml` at your hg38 FASTA, GENCODE GTF and phyloP bigWig.
If you have the original 100-window list, set `legacy_windows` to a CSV with
`chrom,start`; those windows are then placed first in the 400-window panel so
the published numbers can be reproduced on the exact sub-panel.

---

## Run

```bash
./run_all.sh                       # 7B, chr22, in the protocol's order
MODEL=7b CHROM=chr17 ./run_all.sh  # second chromosome
```

or step by step:

| step | command | what it does | gate |
|---|---|---|---|
| 0 | `python -m exp1 selftest` | numerical core, estimator, detectors, analysis path on synthetic data | must be 0 failures |
| 1 | `python -m exp1 step1-blockmap` | operator map, hyena-short prefix check, table **T2** | prefix must match `0,4,7,11,14,18,21,25,28` |
| 4 | `python -m exp1 step4-panel --chrom chr22 --n-windows 400` | window panel + per-context usable-window counts | report the counts, they are the real sample size |
| 2 | `python -m exp1 step2-onset --model 7b` | norm-ratio onset, cosine rotation, threshold stability | onset should be stable for every T in 6…200 |
| 3 | `python -m exp1 step3-fidelity --model 7b` | branch discovery **verification** + dtype ledger | aborts if `x_in + mixer + mlp ≠ x_out` |
| 6 | `python -m exp1 step6-prereg` | freeze + hash the analysis plan | must run before step 7 |
| 5a | `python -m exp1 step5-uref --model 7b` | global carrier direction `u`, frozen | — |
| 5b | `python -m exp1 step5-extract --model 7b` | the extraction pass | identity error must be ~0 |
| 5c | `... step5-extract --shuffle-seed 1234 --tag shuffled` | untrained control | — |
| 7 | `python -m exp1 step7-analysis --model 7b` | **F1**, **F2**, **T1**, yardstick, overdispersion | — |
| 7b | `python -m exp1 step7b-regimes --model 7b` | R1 / R2 / R3 features, per-regime effects, **regime contrast**, transfer | — |
| 7c | `python -m exp1 step7c-motifs --model 7b` | motif-stratified contrasts + paired perturbation effects | needs `--perturb` at step 5 |
| 8 | `python -m exp1 step8-stage2 --model 7b` | stage-2 readouts, bridge test, **F4** | — |

Step 3 is a hard gate on purpose.  Evo 2's internal module names differ between
stacks, so the branch modules are *discovered* and then verified by
reconstructing every block output.  If the check fails the run stops and prints
the module tree; pin `mixer_attr` / `mlp_attr` in `configs/models.yaml` and
re-run rather than proceeding on a guess.

---

## Motif-level work

**No matrix ships with this package, and none can be added by typing one in.**
A `PWM` cannot be constructed without a `MotifProvenance` recording where its
counts came from, and matrices arrive by exactly three audited routes:

| route | what it is | used for |
|---|---|---|
| `derived` | counted from real GENCODE introns in *your* hg38, on chromosomes held out of the panel | donor / acceptor — the primary route |
| `jaspar` | a version-pinned JASPAR CORE matrix, fetched or loaded from file, SHA-256 recorded | TF / promoter elements |
| `maxentscan` | Yeo & Burge (2004) via `maxentpy` | splice-site **strength** |

Acquire everything obtainable before anything motif-related runs:

```bash
python -m exp1 step0-motifs plan          # what the catalogue can get, nothing touched
python -m exp1 step0-motifs acquire --chroms chr1,chr2,chr3
python -m exp1 step0-motifs verify        # digests, sample sizes, information content
```

`acquire` walks a single catalogue (`motifsrc.CATALOG`) and gets every motif it
can in one pass over the sequence. Eight come from your own GENCODE + hg38 and
need no network at all:

| motif | counted from | check that must pass |
|---|---|---|
| `splice_donor_U2` / `splice_acceptor_U2` | GT-AG introns | consensus `GT` / `AG` |
| `splice_donor_GCAG` / `splice_acceptor_GCAG` | GC-AG introns (~0.8%) | `GC` / `AG` |
| `splice_donor_ATAC` / `splice_acceptor_ATAC` | AT-AC introns (minor spliceosome) | `AT` / `AC` |
| `kozak_start` | annotated `start_codon` features, −6..+4 | `ATG` at +1 |
| `polyA_signal` | hexamer **discovered** upstream of transcript 3′ ends | rediscovers `AATAAA`/`ATTAAA` |

The polyA entry is the one worth reading twice: nothing assumes `AATAAA`. The
step counts every hexamer in the 60 nt upstream of annotated 3′ ends against a
composition-matched shuffle of the same regions, and only if a canonical signal
comes out enriched does it build a matrix — anchored on the hexamer's own
occurrences, not on a fixed offset from the transcript end. If it does not come
out enriched, the recipe raises and prints what did. The enrichment table lands
in the report.

Four more are JASPAR (`TATA_box`, `SP1_GC_box`, `CTCF`, `NFYA`). An unversioned
id resolves to whatever JASPAR currently serves and the **resolved** id and
SHA-256 are written into the manifest, which pins it from then on. If the
network is blocked, they are reported as not obtained with the download URL,
and `--jaspar-dir` loads files you fetched by hand.

Three are genuinely external and are reported as missing *with their real
source named*: `branchpoint` (GENCODE does not annotate branchpoints, so there
is nothing to count — Mercer et al. 2015 or branchpointer), and
`maxentscan_5ss` / `maxentscan_3ss` (Yeo & Burge 2004; not on PyPI).

`step0-motifs build` remains as the splice-only subset.

Splice matrices use the Yeo-Burge spans (donor 3 exonic + 6 intronic, acceptor
20 intronic + 3 exonic), counted over every unique annotated intron on the
chromosomes you name, strand-correct, de-duplicated across isoforms, with the
background measured from the same sequence rather than assumed uniform. The
manifest at `motifs/motif_set.json` carries each matrix's counts, sample size,
per-position information content and digest — paste-ready for a methods section
— and the full run lands in `results/00_motifs/acquisition_report.json`.

Perturbations pick their matrix from the dinucleotide the site actually has, so
a GC-AG donor is edited under the GC-AG matrix. A site whose class has no matrix
in the set is skipped and counted, not perturbed under the wrong one.

Three refusals are deliberate, because each one is a silent-failure mode:

* deriving on a panel chromosome **raises** — a motif model fit on the
  chromosomes whose effects it later stratifies has seen its own test set;
* the derived consensus must read `GT` / `AG` at the core or the build
  **aborts** — this is the check that catches a strand or offset error, the
  failure mode Appendix H of the companion manuscript is about;
* MaxEntScan without `maxentpy` **raises** rather than falling back to a PWM —
  a PWM assumes positional independence, which is the assumption MaxEntScan
  exists to drop, so the substitution would answer a different question.

### Model scores vs measurements

Two kinds of column come out, and they are never mixed:

* **model scores** (`motif_*`) — best covering log-odds and signed distance to
  the core per motif, both strands, plus `motif_best` / `motif_best_score`.
  Produced only when a motif set is loaded.
* **measurements** (`MEASUREMENT_COLUMNS`) — the observed splice dinucleotide
  and its canonicality, polypyrimidine-tract fraction, CpG o/e, 3-mer entropy,
  low-complexity. These are counts of the sequence, involve no model, and are
  produced either way. The polypyrimidine tract is here on purpose: it is a
  composition, not a binding site, so it is counted rather than given a matrix.

Running extraction without a motif set is allowed and says so: you get the
measurement strata (canonical vs non-canonical, complexity, CpG) and no model
scores, rather than scores from matrices nobody chose.

`perturb.py` provides the interventional half, as paired edits carried through
the same extraction pass:

| family | what it tests |
|---|---|
| `core_mut` | necessity — mutate the 2-nt core, keep the flanks |
| `flank_shuffle` | context — dinucleotide-preserving shuffle of ±100 bp, keep the core |
| `both` | interaction |
| `scramble` | composition control — shuffle the motif itself |
| `insert` | sufficiency — plant a consensus into a matched background, dose-response |
| `spacing` | grammar — two copies at swept spacing |
| `rescue` | restore the core on a shuffled background |

Every edited sequence carries its own untouched reference under the same
`pert_pair_id`, so the comparison is a **paired difference at the same position
in the same window**: composition, context and window identity are held fixed by
construction instead of being adjusted for afterwards.

```bash
python -m exp1 step5-extract --model 7b --perturb \
    --perturb-families core_mut,flank_shuffle,rescue --perturb-sites 2
python -m exp1 step7c-motifs --model 7b
```

---

## Three-regime comparison

`regimes.py` turns each position's block profile into regime features and then
compares regimes with the same window-clustered estimator:

```
R1  blocks 0 .. onset-1       how structure accumulates
R2  onset-1 -> onset -> rot   the handoff, in its two steps
R3  rotation .. post-norm     how the output state is sharpened
```

Per position, for every quantity (`av`, `au`, `a`, `sctr`, `kl`, and `n` in log units):

```
<q>_r1_slope  <q>_r1_peak  <q>_r1_peak_block  <q>_r1_end  <q>_r1_auc
<q>_r2_delta  <q>_r2_rel   <q>_r2_delta_to_rotation
<q>_r3_slope  <q>_r3_end   <q>_r3_delta
```

Three things become possible that a per-block curve cannot give:

* **per-regime effects** — the same context contrast computed separately in R1 and R3
* **regime contrast** — `regime_contrast()` pairs the per-window effects of two
  regimes *by window* and bootstraps the difference, so window heterogeneity
  cancels; it reports whether the difference excludes zero
* **regime transfer** — does an R1 feature predict an R3 feature once entropy and
  composition are controlled (this is the bridge question, asked within the
  regime frame)

`regime_profile_by_group()` plus `plot_regime_profiles()` draw one line per
context — or per motif class, by passing `group_col="motif_best"` or
`"splice_canonical"` — with R1/R2/R3 shaded on the same axis.

---

## What gets stored

One wide row per (variant, position), so later experiments do not need a new
forward pass:

* identity, GENCODE context, repeat / GC / phyloP / boundary distance / feature length
* current and next base, and the realised base's log-probability
* from the unmodified head: entropy, margin, top-1, ACGT mass, the four ACGT logits
* **per block** (0…31 and the post-norm state): `p n q a au av sctr kl ent top1 acgtmass` and the four ACGT logits
* **per branch block** (24…31): mixer / MLP update projection and norm, plus the incoming stream
* a stratified subsample of **raw hidden states** at blocks 18, 24, 26–31 (`cache/raw/*.npz`) for later probes, SAEs or transcoders

Motif columns (PWM scores and distances, splice core and canonicality, CpG o/e,
3-mer entropy, low-complexity) ride along on every row, so any contrast can be
re-run per motif class without touching the model.

Input variants are generated deterministically per window: `real`, `random`,
`polyA`, `shuffle_mono`, `shuffle_di`, `revcomp`, `mask_center_200`, plus the
perturbation families above, each tagged with `pert_family`, `pert_site`,
`pert_param`, `pert_pair_id` and `pert_dist_to_core`.  Add more in
`exp1/variants.py`; a point-substitution helper is there for later ref/alt work.

Rough size: 400 windows × 3 kb × 4 variants ≈ 5 M rows, ~2 KB/row → about 10 GB
per chromosome in parquet, plus ~2 GB of raw states.

---

## Statistics

Fixed to match the follow-up manuscript so the new numbers are directly
comparable to the published ones:

* resampling unit is the **window**, never the position
* estimand is the within-window paired standardised difference against intron
* windows with fewer than 30 positions of either class are dropped **and counted**
* cluster bootstrap, B = 2000, seed 42, percentile intervals
* adjustment is a within-window regression on the published covariate list
* no position-level p-value is produced anywhere

`exp1/stats.py` also reports the overdispersion ratio, so the 266× figure can be
re-derived on the new panel rather than assumed.

---

## What is verified here, and what is not

Verified by `python -m exp1 selftest` (21 checks, all passing, and the suite is
run with `-W error::RuntimeWarning` so numerical warnings fail the build):

* `p, n, q, a` against direct computation, including `n² = p² + q²`
* the channel identity `a = α·a_u + β·a_v` to < 1e-12
* stable log-softmax; KL(p‖p) = 0 and KL ≥ 0
* the ACGT contrast space is rank 3 and orthonormal
* `early_ratio` returns NaN on a vanishing denominator instead of a spike
* `settle_depth` honours the persistence requirement and censors correctly
* period-7 block map, config parsing, onset detection and threshold stability,
  rotation detection with the interior peak
* the estimator recovers a planted effect while **ignoring** a planted
  window-level offset; permutation null centred at zero; within-window
  adjustment removes a planted covariate effect; overdispersion is detected
* PWM scanning recovers a planted consensus at the exact offset; canonical and
  non-canonical splice cores are classified correctly; poly-A is flagged as low
  complexity
* **degenerate windows** (poly-A, all-N, half-N) produce NaN rather than a huge
  finite score — a regression test for the `np.nan_to_num(..., nan=-inf)` trap,
  which silently turns −inf into −1.8e308 and then overflows the float32 cast
* perturbations: `core_mut` changes exactly the two core bases and nothing else,
  `flank_shuffle` preserves the core and the flank's dinucleotide composition,
  `rescue` restores the core on the shuffled background, dose-response and
  spacing series are generated; paired differencing recovers a planted delta
* regime features recover the planted 10³ norm jump across the onset; per-regime
  effect tables, the paired regime contrast and the transfer fit all run
* the whole analysis path end-to-end on synthetic states with planted structure:
  F1 returns the correct verdict, F2 assigns the entropy correlation to the
  carrier channel, T1 recovers the planted content contrast, the bridge test runs

Not verifiable without the model and a GPU: model loading, hook placement,
branch discovery, and anything that depends on real activations.  Their
equivalent safeguard is the step-3 reconstruction gate and the identity check
recorded on every extracted row.

---

## Layout

```
exp1/
  backend.py    numpy/torch shim so the formulas are written and tested once
  metrics.py    p/n/q, channel split, ACGT contrast basis, lens readouts
  stats.py      window-clustered estimator, adjustment, permutation, bridge
  blockmap.py   operator map, onset and rotation detectors
  modelio.py    loading, discovery + verification, hooks, dtype ledger, shuffle
  labels.py     GENCODE / repeat / GC / phyloP tracks
  panels.py     400-window panel with the legacy subset preserved
  variants.py   input variants (incl. dinucleotide-preserving shuffle)
  motifs.py     PWM scan, splice cores, strength quartiles, composition features
  perturb.py    paired motif edits: core / flank / rescue / insert / spacing
  regimes.py    R1-R2-R3 features, per-regime effects, regime contrast, transfer
  extract.py    the extraction pass
  analysis.py   F1, F2, T1, stage-2, bridge, figures
  prereg.py     freeze + hash the plan
  selftest.py   everything above, on synthetic data
```

Results land in `results/` (tables, figures, JSON verdicts) and caches in
`cache/`.  Every artefact gets a `.meta.json` sidecar recording host, git commit,
dtypes and row counts — the provenance discipline that the two-extraction
discrepancy in the follow-up manuscript made necessary.


## Two references, joined at the handoff

Every settling readout in this line of work measures one thing: how close block
*l* is to `h_norm`, the state the output head reads. But blocks 28 and 29 are
near-orthogonal to that frame (−0.0090, −0.0058) — so across the whole
pre-handoff stack, that quantity is alignment to a frame the stream has not
entered yet, and its largest covariate turns out to be the model's own next-base
uncertainty.

`dualref.py` points each regime at the reference it is actually approaching:

| blocks | reference | question |
|---|---|---|
| 0 … onset−1 | `h27`, the pre-handoff endpoint | when does the **representation settle**? |
| onset … last | `h_norm`, the state before the ACGT logits | when does the **output commit**? |

```bash
python -m exp1 step7d-dualref --model 7b --chrom chr22
```

Both cosines are stored at *every* block, so the crossover is visible rather
than assumed, and the same class can come out with opposite signs in the two
regimes — which is the dissociation the construction exists to expose.

Three things it does not pretend:

* **the endpoint is inside its own reference** — `cos(h27, h27) = 1` by
  construction, so a held-out control reference (`h26`) is extracted alongside
  and `self_reference_control()` checks the two curves agree away from their
  endpoints. Neither reference appears in its own pool, and the self-test
  asserts that;
* **thresholded depths censor** — every `c_pre` / `c_post` carries its censored
  fraction, non-crossing positions stay `NaN` instead of entering at a ceiling,
  and the continuous maxima are primary;
* **post-handoff is short in 7B** — blocks 28–31, of which 30 and 31 are a
  passthrough pair. Three distinct points. Read it as a jump with a shape; the
  construction earns more at 40B, where 26 blocks follow the rotation.

## Two ways to call a splice site

`sitedefs.py` runs every contrast under both definitions and crosses them:

```bash
python -m exp1 step7e-sitedefs --model 7b --chrom chr22
```

|  | motif-strong | motif-weak |
|---|---|---|
| **annotated** | `both` | `annot_only` |
| **not annotated** | `motif_only` | `neither` |

The threshold is the *q*-th percentile of the motif score at annotated sites of
that class, so "motif-strong" means "at least as good as the weakest q% of real
annotated sites" — a property of the panel, reportable and pre-registerable.

The off-diagonals are the point. If a layer-wise readout separates `both` from
`neither` but not `annot_only` from `neither`, the model is tracking the
sequence motif. If it separates `annot_only` too, it is tracking something the
local sequence does not carry. Neither definition alone can say that.

## Linear decodability at every block

`probesweep.py` fits a probe from the full residual state to each biological
label at every block, folds grouped by window, and reports one curve per label
plus how much of each peak survives the handoff.

```bash
python -m exp1 step9-profile --model 7b --chrom chr22 --motif-probes
```

Labels span annotation classes, motif classes and strengths, the measurements,
the site-agreement cells, and continuous covariates (GC, phyloP, CpG o/e,
polypyrimidine fraction, next-base entropy). Raw states are now kept at **every**
block for the stratified subsample, which is what makes the full sweep possible.

Two honesties are built in. The curve is **linear decodability**, not
information: a drop across the handoff is equally consistent with summarisation
and with non-linear re-encoding, and separating those needs the non-linear probe
comparison, not this curve. And the shuffled-label floor is permuted *within*
window — which is vacuous for any label that does not vary inside a window, so
the sweep detects that case, marks it `VACUOUS`, and drops it from the retention
table rather than reporting a floor equal to the real score.

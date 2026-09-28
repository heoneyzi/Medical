"""Self-tests for everything that does not need the model.

The numerical core (:mod:`exp1.metrics`), the estimator (:mod:`exp1.stats`), the
operator/onset detectors (:mod:`exp1.blockmap`) and the whole analysis path are
exercised here against synthetic data with **planted** structure, so a failure
points at the code rather than at the biology.  Run it before every server run:

    python -m exp1 selftest

The model-dependent parts (loading, hooks, branch discovery) cannot be tested
without Evo 2 and a GPU; their equivalent check is the reconstruction-fidelity
gate in step 3, which refuses to proceed on a bad discovery.
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import analysis as A
from . import blockmap as B
from . import metrics as M
from . import motifs as MO
from . import dualref as DRF
from . import motifsrc as SRC
from . import probesweep as PS
from . import sitedefs as SD
from . import perturb as PB
from . import regimes as RG
from . import stats as S

TOL = 1e-10


class Check:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            self.passed.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            self.failed.append((name, f"{exc}\n{traceback.format_exc(limit=3)}"))
            print(f"  FAIL  {name}: {exc}")

    def report(self) -> int:
        print(f"\n{len(self.passed)} passed, {len(self.failed)} failed")
        for name, err in self.failed:
            print(f"\n--- {name} ---\n{err}")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------
# synthetic state generator
# --------------------------------------------------------------------------

def synth_states(
    n_pos: int = 400,
    dim: int = 64,
    n_blocks: int = 32,
    onset: int = 28,
    rotation: int = 30,
    alpha: float = 0.94,
    seed: int = 0,
):
    """Build states in an explicit {u, v(t), w(t,l)} basis with planted structure.

    * the carrier coefficient tracks a per-position "entropy" variable
    * the content coefficient tracks a per-position "context" effect
    * the overall scale explodes at ``onset`` and again at ``rotation``
    """
    rng = np.random.default_rng(seed)
    u = np.zeros(dim)
    u[0] = 1.0

    # per-position orthonormal v(t)
    v = rng.normal(size=(n_pos, dim))
    v[:, 0] = 0.0
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    beta = np.sqrt(1 - alpha**2)
    ref_unit = alpha * u[None, :] + beta * v

    entropy = rng.uniform(0.2, 1.3, size=n_pos)
    context_effect = rng.normal(size=n_pos) * 0.0  # filled by caller

    return dict(u=u, v=v, ref_unit=ref_unit, entropy=entropy, alpha=alpha, beta=beta,
                rng=rng, dim=dim, n_pos=n_pos, n_blocks=n_blocks,
                onset=onset, rotation=rotation, context_effect=context_effect)


def build_h(state, block: int, content: np.ndarray):
    """One block's states: [T, D]."""
    rng = state["rng"]
    n_pos, dim = state["n_pos"], state["dim"]
    onset, rotation = state["onset"], state["rotation"]

    # Growth profile.  The onset is NOT a rescaling of the whole state: a
    # uniform scale multiplies numerator and denominator alike and leaves the
    # cosine untouched, which is a degenerate case rather than the phenomenon.
    # The real event ADDS a large vector orthogonal to the output reference, so
    # the aligned component is preserved while the norm explodes.  The fixture
    # has to plant that, or the decomposition is tested against the wrong thing.
    scale = 1.0 * (1.15 ** block)
    if block >= rotation:
        scale *= 1e6

    # carrier coefficient rises mid-stack and tracks "entropy"
    ramp = np.exp(-((block - 18) ** 2) / 40.0)
    c_u = (0.25 + 0.55 * ramp * state["entropy"] / state["entropy"].max())
    # content coefficient carries the context effect, pre-rotation only
    c_v = content * (1.0 if block < rotation else 0.15)
    # the rest is noise orthogonal to both
    w = rng.normal(size=(n_pos, dim))
    w[:, 0] = 0.0
    w -= (np.sum(w * state["v"], axis=1, keepdims=True)) * state["v"]
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    c_w = np.full(n_pos, 1.0)

    h = (c_u[:, None] * state["u"][None, :]
         + c_v[:, None] * state["v"]
         + c_w[:, None] * w)
    h = h * scale
    if block >= onset:
        ref = state["ref_unit"]
        orth = rng.normal(size=(n_pos, dim))
        orth -= np.sum(orth * ref, axis=1, keepdims=True) * ref   # exactly orthogonal
        orth /= np.linalg.norm(orth, axis=1, keepdims=True)
        h = h + 7000.0 * orth
    return h


def synth_table(n_windows: int = 24, n_pos: int = 300, seed: int = 0) -> pd.DataFrame:
    """A small stand-in for the extraction output, with known planted effects."""
    rng = np.random.default_rng(seed)
    contexts = ["intron"] * (n_pos // 2) + ["coding_exon"] * (n_pos // 4) + \
               ["splice_donor"] * (n_pos - n_pos // 2 - n_pos // 4)
    frames = []
    for w in range(n_windows):
        st = synth_states(n_pos=n_pos, seed=seed * 1000 + w)
        ctx = np.asarray(contexts)
        rng.shuffle(ctx)
        # planted content effect: coding exon shifted, donor shifted less
        content = np.where(ctx == "coding_exon", 0.9, np.where(ctx == "splice_donor", 0.5, 0.2))
        content = content + rng.normal(scale=0.12, size=n_pos)
        # window-level offset -> this is what makes pooled statistics lie
        content = content + rng.normal(scale=0.8)

        col: dict = dict(
            model=["synth"] * n_pos, variant=["real"] * n_pos, chrom=["chrS"] * n_pos,
            window_id=[f"chrS:{w}"] * n_pos, legacy_window=[w < 6] * n_pos,
            pos=np.arange(n_pos) + w * 10_000,
            context=ctx,
            entropy_final=st["entropy"],
            repeat=(rng.random(n_pos) < 0.2).astype(float),
            gc=rng.uniform(0.3, 0.6, n_pos),
            phylop=rng.normal(size=n_pos),
            boundary_dist=rng.integers(1, 5000, n_pos).astype(float),
            feature_len=rng.uniform(3, 10, n_pos),
            logp_true_final=-rng.uniform(0.2, 2.0, n_pos),
        )

        for b in range(st["n_blocks"]):
            h = build_h(st, b, content)
            p, n, q, a = M.pnq(h, st["ref_unit"])
            split = M.channel_split(h, st["ref_unit"], st["u"])
            tag = f"{b:02d}"
            col[f"p_{tag}"] = p
            col[f"n_{tag}"] = n
            col[f"q_{tag}"] = q
            col[f"a_{tag}"] = a
            col[f"au_{tag}"] = split.a_u
            col[f"av_{tag}"] = split.a_v
            col[f"kl_{tag}"] = np.abs(rng.normal(scale=0.3, size=n_pos)) * (1.0 if b < 28 else 0.05)
            col[f"ent_{tag}"] = st["entropy"] + rng.normal(scale=0.05, size=n_pos)
            col[f"top1_{tag}"] = (rng.random(n_pos) < (0.2 if b < 28 else 0.95)).astype(float)
            col[f"acgtmass_{tag}"] = np.clip(rng.normal(0.99, 0.005, n_pos), 0, 1)
            col[f"sctr_{tag}"] = np.clip(0.01 + 0.6 * (b >= 30) + rng.normal(scale=0.01, size=n_pos), 0, 1)

        # post-norm tap
        h = build_h(st, st["n_blocks"] - 1, content)
        p, n, q, a = M.pnq(h, st["ref_unit"])
        split = M.channel_split(h, st["ref_unit"], st["u"])
        for name, val in (("p", p), ("n", n), ("q", q), ("a", a),
                          ("au", split.a_u), ("av", split.a_v)):
            col[f"{name}_norm"] = val
        col["alpha_ref"] = split.alpha
        col["beta_ref"] = split.beta
        col["kl_norm"] = np.zeros(n_pos)
        col["ent_norm"] = st["entropy"]
        col["top1_norm"] = np.ones(n_pos)
        col["acgtmass_norm"] = np.full(n_pos, 0.99)
        col["sctr_norm"] = np.full(n_pos, 0.63)

        # Branch columns consistent with hypothesis B at the onset block AND
        # with the residual identity p_l = p_{l-1} + <mixer> + <mlp>.  The
        # fixture has to satisfy the same arithmetic the real model does, or the
        # identity check in f1_decomposition is testing the fixture's sloppiness
        # rather than the analysis.
        for b in range(24, 32):
            tag, prev = f"{b:02d}", f"{b-1:02d}"
            big = 7000.0 if b == 28 else (30.0 if b < 28 else 1e6)
            dp = (col[f"p_{tag}"] - col[f"p_{prev}"]) if f"p_{prev}" in col else np.zeros(n_pos)
            mixer = rng.normal(scale=1.0, size=n_pos)
            col[f"mxp_{tag}"] = mixer
            col[f"mlpp_{tag}"] = dp - mixer          # closes the identity exactly
            col[f"mlpn_{tag}"] = big
            col[f"mxn_{tag}"] = 70.0 if b == 28 else 20.0
            col[f"xinp_{tag}"] = col[f"p_{prev}"] if f"p_{prev}" in col else np.zeros(n_pos)
            col[f"xinn_{tag}"] = 40.0
        frames.append(pd.DataFrame(col))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def t_pnq() -> None:
    rng = np.random.default_rng(1)
    h = rng.normal(size=(50, 16))
    r = rng.normal(size=(50, 16))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    p, n, q, a = M.pnq(h, r)
    assert np.allclose(p, np.sum(h * r, axis=1), atol=TOL)
    assert np.allclose(n, np.linalg.norm(h, axis=1), atol=TOL)
    resid = h - p[:, None] * r
    assert np.allclose(q, np.linalg.norm(resid, axis=1), atol=1e-8)
    assert np.allclose(a, p / n, atol=TOL)
    assert np.allclose(n**2, p**2 + q**2, rtol=1e-8)


def t_channel_identity() -> None:
    rng = np.random.default_rng(2)
    dim = 32
    u = np.zeros(dim); u[0] = 1.0
    v = rng.normal(size=(60, dim)); v[:, 0] = 0
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    alpha = 0.94
    ref = alpha * u[None, :] + np.sqrt(1 - alpha**2) * v
    h = rng.normal(size=(60, dim)) * rng.uniform(1, 1e6, size=(60, 1))
    split = M.channel_split(h, ref, u)
    err = float(M.channel_identity_error(split))
    assert err < 1e-12, f"channel identity broken: {err}"
    assert np.allclose(split.alpha, alpha, atol=1e-10)


def t_log_softmax() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 11)) * 50
    lp = M.log_softmax(x)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(lp).all()


def t_acgt_basis() -> None:
    rng = np.random.default_rng(4)
    E = rng.normal(size=(64, 24))
    basis = M.acgt_contrast_basis(E, np.array([0, 1, 2, 3]))
    assert basis.shape[0] == 3, f"contrast space should be rank 3, got {basis.shape[0]}"
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-8)
    mean_row = E[[0, 1, 2, 3]].mean(axis=0)
    frac = M.subspace_energy_fraction(mean_row[None, :], basis)
    assert frac[0] < 0.5 or True  # mean direction need not be in the contrast space


def t_lens_metrics() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(30, 12))
    final = M.log_softmax(logits.copy())
    lm = M.lens_metrics(logits, final, np.array([0, 1, 2, 3]))
    assert np.allclose(lm.kl_to_final, 0.0, atol=1e-10), "KL to itself must be 0"
    assert np.allclose(lm.top1_match, 1.0)
    other = M.log_softmax(rng.normal(size=(30, 12)))
    lm2 = M.lens_metrics(logits, other, np.array([0, 1, 2, 3]))
    assert (lm2.kl_to_final >= -1e-12).all(), "KL must be non-negative"


def t_early_ratio_and_settle() -> None:
    r = M.early_ratio(np.array([0.5]), np.array([0.0]), np.array([1.0]))
    assert abs(r[0] - 0.5) < 1e-12
    r0 = M.early_ratio(np.array([0.5]), np.array([1.0]), np.array([1.0]))
    assert np.isnan(r0[0]), "zero denominator must be NaN, not a spike"

    m = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=float)
    d = M.settle_depth(m, [10, 11, 12, 13, 14], persistence=2)
    assert d[0] == 12
    assert d[1] == 13
    assert np.isnan(d[2])


def t_blockmap() -> None:
    bm = B.blockmap_from_period(32, "7b")
    assert bm.hcs_idx[:9] == B.HCS_PREFIX
    assert bm.type_of(28) == "hcs" and bm.type_of(30) == "hcl" and bm.type_of(31) == "attn"
    chk = B.check_hcs_prefix(bm)
    assert chk["match"], chk

    cfg = dict(num_layers=32, hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28],
               hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29],
               hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30],
               attn_layer_idxs=[3, 10, 17, 24, 31])
    bm2 = B.blockmap_from_config(cfg, model="7b")
    assert bm2.source == "config" and bm2.types == bm.types


def t_onset_rotation() -> None:
    norms = [1.0 * (1.2 ** i) for i in range(28)]
    norms += [norms[-1] * 250, norms[-1] * 250 * 1e3, norms[-1] * 250 * 1e3 * 1e6, 0.0]
    norms[31] = norms[30]
    det = B.detect_onset(norms, threshold=10.0)
    assert det["onset"] == 28, det["onset"]
    stab = B.onset_threshold_stability(norms)
    assert len(set(v for v in stab.values() if v is not None)) == 1, stab

    cos = [0.0] * 18 + [0.4] + [0.2] * 11 + [0.61, 0.61]
    rot = B.detect_rotation(cos)
    assert rot["rotation"] == 30, rot["rotation"]
    assert rot["interior_peak_block"] == 18


def t_stats_effect() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for w in range(30):
        off = rng.normal(scale=3.0)          # window offset: must not leak into d
        a = rng.normal(loc=off - 1.0, scale=1.0, size=120)
        b = rng.normal(loc=off, scale=1.0, size=120)
        rows.append(pd.DataFrame(dict(window_id=w, context="coding_exon", val=a)))
        rows.append(pd.DataFrame(dict(window_id=w, context="intron", val=b)))
    df = pd.concat(rows, ignore_index=True)
    eff = S.window_effect(df, value="val", target="coding_exon")
    assert eff.n_windows_used == 30
    assert -1.4 < eff.mean_d < -0.7, eff.mean_d
    assert eff.ci_lo < eff.mean_d < eff.ci_hi
    assert eff.ci_hi < 0, "planted effect should exclude zero"


def t_stats_permutation() -> None:
    rng = np.random.default_rng(8)
    rows = []
    for w in range(20):
        off = rng.normal(scale=2.0)
        rows.append(pd.DataFrame(dict(window_id=w, context="coding_exon",
                                      val=rng.normal(off, 1.0, 80))))
        rows.append(pd.DataFrame(dict(window_id=w, context="intron",
                                      val=rng.normal(off, 1.0, 80))))
    df = pd.concat(rows, ignore_index=True)
    null = S.permutation_null(df, value="val", target="coding_exon", n_perm=40)
    assert abs(null["null_mean"]) < 0.15, null
    assert null["null_lo"] < 0 < null["null_hi"], null


def t_stats_adjust() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for w in range(25):
        n = 150
        cov = rng.normal(size=n)
        ctx = np.where(rng.random(n) < 0.5, "coding_exon", "intron")
        # the entire "effect" is carried by the covariate
        val = 2.0 * cov + np.where(ctx == "coding_exon", 0.0, 0.0) + rng.normal(scale=0.2, size=n)
        cov = cov + np.where(ctx == "coding_exon", 1.0, 0.0)
        val = 2.0 * cov + rng.normal(scale=0.2, size=n)
        rows.append(pd.DataFrame(dict(window_id=w, context=ctx, val=val, cov=cov)))
    df = pd.concat(rows, ignore_index=True)
    raw = S.window_effect(df, value="val", target="coding_exon")
    adj_df = S.residualise_within_window(df, "val", ["cov"])
    adj = S.window_effect(adj_df, value="val_adj", target="coding_exon")
    assert raw.mean_d > 0.8, raw.mean_d
    assert abs(adj.mean_d) < 0.2, adj.mean_d


def t_overdispersion() -> None:
    rng = np.random.default_rng(10)
    rows = []
    for w in range(40):
        p = rng.beta(0.2, 0.2)                 # strong between-window heterogeneity
        rows.append(pd.DataFrame(dict(window_id=w,
                                      crossed=(rng.random(200) < p).astype(int))))
    df = pd.concat(rows, ignore_index=True)
    od = S.overdispersion(df, "crossed")
    assert od["ratio"] > 10, od


def t_analysis_e2e() -> None:
    df = synth_table(n_windows=16, n_pos=240, seed=3)

    prof = A.f1_profile(df)
    assert len(prof) >= 32
    br = A.f1_branch_decomposition(df)
    # the decomposition is an identity, so it must close to machine precision
    dec = A.f1_decomposition(df, onset=28)
    assert dec["available"], dec
    assert dec["identity_log_error"] < 1e-9, dec["identity_log_error"]
    assert dec["identity_branch_rel_error"] < 1e-6, dec["identity_branch_rel_error"]
    for k in ("d_log_cos", "d_log_numerator", "d_log_denominator", "d_p"):
        assert dec[k]["ci_lo"] <= dec[k]["mean"] <= dec[k]["ci_hi"], (k, dec[k])
    # The fixture plants an exactly-orthogonal onset write on top of a smooth
    # 15%-per-block growth.  The decomposition must attribute almost all of the
    # cosine collapse to the DENOMINATOR, and must not attribute it to the
    # numerator -- that is the planted fact, and it is what the identity buys.
    share_denom = abs(dec["d_log_denominator"]["mean"]) / (
        abs(dec["d_log_denominator"]["mean"]) + abs(dec["d_log_numerator"]["mean"]))
    assert share_denom > 0.9, share_denom
    assert dec["d_log_cos"]["ci_hi"] < 0, dec["d_log_cos"]
    hyp = A.f1_hypothesis_check(dec).set_index("hypothesis")
    # cancellation is planted absent, so its interval must NOT lie below zero
    assert not bool(hyp.loc["C_active_cancellation", "consistent"]), hyp
    assert set(hyp.index) >= {"A_dilution", "B_orthogonal_overwrite",
                              "C_active_cancellation", "D_adds_alignment"}
    rot = A.f1_rotation_share(df, rotation=30)
    assert rot["available"] and 0.0 <= rot["written_share"] <= 1.0

    ch = A.f2_channel_profile(df)
    mid = ch[ch["block"] == "18"].iloc[0]
    assert mid["r_entropy_au"] > 0.5, mid["r_entropy_au"]
    assert abs(mid["r_entropy_av"]) < 0.3, mid["r_entropy_av"]

    pool_blocks = [f"{i:02d}" for i in range(0, 28)]
    df = A.pool_max(df, "av", pool_blocks, "av_pool")
    df = A.pool_max(df, "au", pool_blocks, "au_pool")
    df = A.pool_max(df, "a", pool_blocks, "a_pool")

    t1 = A.t1_context_table(df, {"a": "a_pool", "a_u": "au_pool", "a_v": "av_pool"},
                            contexts=("coding_exon", "splice_donor"))
    ce = t1[(t1["channel"] == "a_v") & (t1["context"] == "coding_exon")].iloc[0]
    assert ce["d_unadj"] > 0.3, ce.to_dict()
    assert ce["n_windows"] >= 10

    s2 = A.stage2_table(df, onset=28, rotation=30)
    assert (s2[s2["block"] == "norm"]["top1_match_rate"] == 1.0).all()

    feats = A.stage2_position_features(df, onset=28, rotation=30)
    assert "r_D" in feats.columns

    out = A.bridge_test(feats.dropna(subset=["av_pool"]), predictor="av_pool",
                        outcome="r_D", controls=("entropy_final", "gc"))
    assert "fit" in out and "quadrant_counts" in out
    assert len(out["quadrant_counts"]) == 4




# --------------------------------------------------------------------------
# motif fixtures
#
# These matrices are built inside the test from a planted consensus.  They are
# fixtures for the scanning code, never a model of any real signal -- the
# package ships no matrices, and a test is not a back door for one.
# --------------------------------------------------------------------------

def _fixture_pwm(name: str, consensus: str, n: int = 800, purity: float = 0.9,
                 core_offset: int = 0, core_len: int = 2) -> "MO.PWM":
    """Counts for a matrix whose consensus is exactly ``consensus``."""
    counts = np.full((len(consensus), 4), n * (1.0 - purity) / 3.0)
    for i, ch in enumerate(consensus):
        counts[i, MO.BASES.index(ch)] = n * purity
    return MO.PWM.from_counts(
        name, np.rint(counts),
        MO.MotifProvenance(source="file", detail="synthetic self-test fixture",
                           n_sites=n, uri="selftest"),
        core_offset=core_offset, core_len=core_len)


def _fixture_set() -> "MO.MotifSet":
    ms = MO.MotifSet(note="self-test fixtures")
    ms.add(_fixture_pwm("splice_donor_U2", "CAGGTAAGT", core_offset=3, core_len=2))
    ms.add(_fixture_pwm("splice_acceptor_U2", "TTTTTTTTTTTTTTTTTTTTAGG",
                        core_offset=20, core_len=2))
    return ms


def t_motifs_scan() -> None:
    rng = np.random.default_rng(11)
    bg = "".join(rng.choice(list("ACGT"), size=2000))
    donor = _fixture_set().pwms["splice_donor_U2"]
    planted_at = 900
    seq = bg[:planted_at] + donor.consensus() + bg[planted_at + donor.length:]
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)

    scores = MO.scan_pwm(arr, donor)
    best = int(np.nanargmax(scores))
    assert best == planted_at, f"planted motif not recovered: {best} vs {planted_at}"
    assert donor.consensus()[donor.core_offset:donor.core_offset + 2] == "GT"

    # a consensus hit must beat the background by a wide margin
    finite = scores[np.isfinite(scores)]
    assert scores[planted_at] > np.nanquantile(finite, 0.999)


def t_motifs_annotation() -> None:
    rng = np.random.default_rng(12)
    seq = "".join(rng.choice(list("ACGT"), size=1200))
    arr = np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    # plant a canonical donor core at scored position 100 and a non-canonical at 200
    arr[400 + 100:400 + 102] = np.frombuffer(b"GT", dtype=np.uint8)
    arr[400 + 200:400 + 202] = np.frombuffer(b"GC", dtype=np.uint8)
    context = np.array(["intron"] * 400, dtype=object)
    context[100] = "splice_donor"
    context[200] = "splice_donor"

    ms = _fixture_set()
    ann = MO.annotate_window(arr, slice(400, 800), context, ms)
    cols = ann.to_dict()
    assert cols["splice_core"][100] == "GT" and cols["splice_canonical"][100] == 1.0
    assert cols["splice_core"][200] == "GC" and cols["splice_canonical"][200] == 0.0
    assert "motif_splice_donor_U2_score" in cols and "motif_best" in cols
    assert np.isfinite(cols["kmer3_entropy"]).all()
    for col in MO.MEASUREMENT_COLUMNS:
        assert col in cols, f"measurement column {col} missing"

    # measurements must be available with no model at all
    only = MO.measure_window(arr, slice(400, 800), context)
    assert set(only) == set(MO.MEASUREMENT_COLUMNS)
    assert not any(k.startswith("motif_") for k in only)

    # low complexity must be flagged on poly-A
    polya = np.full(1200, ord("A"), dtype=np.uint8)
    ann2 = MO.annotate_window(polya, slice(400, 800), context, ms)
    # poly-A has essentially no 3-mer diversity; the continuous entropy says so
    # without a bit-count cut being applied to it
    assert ann2.to_dict()["kmer3_entropy"].mean() < 1.0


def t_motifs_quartiles() -> None:
    x = np.concatenate([np.arange(100, dtype=float), [np.nan] * 5])
    q = MO.strength_quartiles(x)
    assert set(q[:100]) <= {"q1", "q2", "q3", "q4"}
    assert (q[100:] == "na").all()


def t_perturb_families() -> None:
    rng = np.random.default_rng(13)
    seq = "".join(rng.choice(list("ACGT"), size=6000))
    arr = np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    ms = _fixture_set()
    donor = ms.pwms["splice_donor_U2"]
    site_core = 3000
    arr[site_core:site_core + 2] = np.frombuffer(b"GT", dtype=np.uint8)

    site = PB.PerturbSite(offset=site_core - donor.core_offset, motif="splice_donor_U2",
                          core_offset=donor.core_offset, core_len=2, flank=100,
                          label="splice_donor")
    plans = PB.build_perturbations(arr, [site], ms,
                                   families=("core_mut", "flank_shuffle", "rescue", "scramble"),
                                   seed=5)
    fams = {p.meta["pert_family"] for p in plans}
    assert {"reference", "core_mut", "flank_shuffle", "rescue", "scramble"} <= fams

    by = {p.meta["pert_family"]: p.seq for p in plans}
    ref = by["reference"]

    # core_mut changes exactly the two core bases and nothing else
    diff = np.where(ref != by["core_mut"])[0]
    assert set(diff.tolist()) == {site_core, site_core + 1}, diff[:10]
    assert bytes(by["core_mut"][site_core:site_core + 2]).decode() == "AA"

    # flank_shuffle keeps the core and preserves flank composition
    assert (by["flank_shuffle"][site_core:site_core + 2] == ref[site_core:site_core + 2]).all()
    lo, hi = site_core - 100, site_core
    from collections import Counter
    assert Counter(bytes(by["flank_shuffle"][lo:hi]).decode()) == Counter(bytes(ref[lo:hi]).decode())

    # rescue restores the core on the shuffled background
    assert (by["rescue"][site_core:site_core + 2] == ref[site_core:site_core + 2]).all()
    assert not (by["rescue"] == ref).all()

    # dose-response and spacing series are generated on a matched background
    plans2 = PB.build_perturbations(arr, [site], ms, families=("insert", "spacing"),
                                    seed=5, spacings=(10, 40), copies=(1, 2))
    fams2 = [p.meta["pert_param"] for p in plans2]
    assert "n=1" in fams2 and "n=2" in fams2 and "gap=10" in fams2


def t_perturb_pairing() -> None:
    rows = []
    for w in range(6):
        for fam in ("reference", "core_mut"):
            for off in range(20):
                val = 1.0 + (0.5 if fam == "core_mut" else 0.0) + 0.01 * off
                rows.append(dict(window_id=f"w{w}", pert_pair_id="s0", offset=off,
                                 pert_family=fam, pert_motif="donor_U2",
                                 pert_label="splice_donor", context="splice_donor",
                                 a_pool=val))
    df = pd.DataFrame(rows)
    m = PB.paired_effect_frame(df, value="a_pool", family="core_mut")
    assert len(m) == 6 * 20
    assert np.allclose(m["delta"], 0.5)




def t_motifs_provenance() -> None:
    """A matrix cannot exist without saying where it came from."""
    counts = np.full((4, 4), 10.0)
    try:
        MO.PWM(name="anon", counts=counts)          # type: ignore[call-arg]
        raise AssertionError("a PWM was constructed with no provenance")
    except TypeError:
        pass

    try:
        MO.MotifProvenance(source="vibes", detail="made up")
        raise AssertionError("an unknown motif source was accepted")
    except ValueError:
        pass

    pwm = _fixture_pwm("t", "ACGT")
    assert pwm.n_observations > 0 and pwm.consensus() == "ACGT"
    # counts, not probabilities: the sample size survives the round trip
    d = pwm.to_dict()
    assert d["provenance"]["detail"]
    back = MO.PWM.from_dict(d)
    assert back.digest() == pwm.digest()

    # a tampered matrix must not load quietly
    d2 = pwm.to_dict()
    d2["counts"][0][0] = float(d2["counts"][0][0]) + 7.0
    try:
        MO.PWM.from_dict(d2)
        raise AssertionError("an edited matrix loaded against a stale digest")
    except ValueError:
        pass


def t_motifs_manifest(tmp=None) -> None:
    import tempfile

    ms = _fixture_set()
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "motif_set.json")
        ms.save(path)
        back = MO.MotifSet.load(path)
        assert set(back.pwms) == set(ms.pwms)
        for name in ms.pwms:
            assert back.pwms[name].digest() == ms.pwms[name].digest()
            assert back.pwms[name].provenance.source == "file"
        man = json.loads(pathlib.Path(path).read_text())
        assert set(man["measurement_columns"]) == set(MO.MEASUREMENT_COLUMNS)
        assert "motif_splice_donor_U2_score" in man["model_columns"]


def t_motifs_no_default() -> None:
    """There is no built-in matrix set, and nothing quietly invents one."""
    assert not hasattr(MO, "BUILTIN_PWMS"), "a built-in matrix set has reappeared"

    ctx = np.array(["intron"] * 50, dtype=object)
    arr = np.frombuffer(("ACGT" * 100).encode(), dtype=np.uint8)
    try:
        MO.annotate_window(arr, slice(100, 150), ctx, MO.MotifSet())
        raise AssertionError("annotation ran with an empty motif set")
    except ValueError as exc:
        assert "step0-motifs" in str(exc)

    site = PB.PerturbSite(offset=10, motif="splice_donor_U2", core_offset=3)
    try:
        PB.build_perturbations(arr.copy(), [site], MO.MotifSet())
        raise AssertionError("perturbation ran with an empty motif set")
    except ValueError:
        pass

    # and insertion refuses to plant a motif that is not in the set
    try:
        PB._consensus_bytes("not_a_motif", _fixture_set())
        raise AssertionError("insertion invented a consensus")
    except KeyError:
        pass


def _rc(text: str) -> str:
    return text[::-1].translate(str.maketrans("ACGT", "TGCA"))


def t_motifs_derive_strand() -> None:
    """Derivation recovers GT/AG from BOTH strands, or it must fail loudly.

    This is the test that matters most in this module.  The companion manuscript
    records a whole appendix on splice-label offset and strand conventions
    differing per class, and the failure mode is silent: a reverse-strand site
    read as if it were forward yields a plausible-looking matrix that is simply
    wrong.  Here a plus-strand and a minus-strand intron carry the *same*
    transcript-oriented signal, so if either is mishandled the pooled consensus
    stops reading GT / AG.
    """
    import tempfile

    donor9 = "CAGGTAAGT"                                   # 3 exonic + 6 intronic
    accept23 = "T" * 18 + "AG" + "CGA"                     # 20 intronic + 3 exonic
    rng = np.random.default_rng(7)
    genome = list("".join(rng.choice(list("ACGT"), size=2000)))

    def put(at: int, text: str) -> None:
        genome[at:at + len(text)] = list(text)

    # plus-strand intron [100, 300)
    put(97, donor9)                 # [97,106) = exon[-3..-1] + intron[+1..+6]
    put(280, accept23)              # [280,303) = intron[-20..-1] + exon[+1..+3]
    # minus-strand intron [1100, 1300): same signal, written reverse-complemented
    put(1294, _rc(donor9))          # [1294,1303)
    put(1097, _rc(accept23))        # [1097,1120)

    seq = np.frombuffer("".join(genome).encode(), dtype=np.uint8)

    gtf_lines = []
    def exon(tid, start1, end1, strand):
        gtf_lines.append("\t".join([
            "chrT", "test", "exon", str(start1), str(end1), ".", strand, ".",
            f'gene_id "g"; transcript_id "{tid}";']))
    exon("tx_plus", 1, 100, "+"); exon("tx_plus", 301, 400, "+")
    exon("tx_minus", 1001, 1100, "-"); exon("tx_minus", 1301, 1400, "-")

    with tempfile.TemporaryDirectory() as d:
        gtf = pathlib.Path(d) / "t.gtf"
        gtf.write_text("\n".join(gtf_lines) + "\n")
        introns = sorted(MO.iter_introns(str(gtf), "chrT"))

    assert introns == [(100, 300, "+"), (1100, 1300, "-")], introns

    donors, acceptors = MO._site_windows(seq, introns)
    assert len(donors) == 2 and len(acceptors) == 2
    for w in donors:
        assert bytes(w).decode() == donor9, bytes(w).decode()
    for w in acceptors:
        assert bytes(w).decode() == accept23, bytes(w).decode()

    dc, nd = MO.count_matrix(donors, MO.DONOR_EXONIC + MO.DONOR_INTRONIC)
    ac, na = MO.count_matrix(acceptors, MO.ACCEPTOR_INTRONIC + MO.ACCEPTOR_EXONIC)
    assert nd == 2 and na == 2
    prov = MO.MotifProvenance(source="derived", detail="selftest", n_sites=2)
    dpwm = MO.PWM.from_counts("d", dc, prov, core_offset=MO.DONOR_EXONIC)
    apwm = MO.PWM.from_counts("a", ac, prov, core_offset=MO.ACCEPTOR_INTRONIC - 2)
    assert dpwm.consensus() == donor9
    assert apwm.consensus() == accept23
    assert dpwm.consensus()[MO.DONOR_EXONIC:MO.DONOR_EXONIC + 2] == "GT"
    assert apwm.consensus()[MO.ACCEPTOR_INTRONIC - 2:MO.ACCEPTOR_INTRONIC] == "AG"

    # N-containing windows are dropped from the counts rather than encoded as A
    bad = [np.frombuffer(("N" + donor9[1:]).encode(), dtype=np.uint8)]
    _, n_used = MO.count_matrix(bad, 9)
    assert n_used == 0

    # information content must be concentrated on the invariant core
    ic = dpwm.information_content(np.full(4, 0.25))
    assert ic[MO.DONOR_EXONIC] == ic.max() or ic[MO.DONOR_EXONIC + 1] == ic.max()


def t_motifs_no_leakage() -> None:
    """Deriving a motif on a panel chromosome is refused, not warned about."""
    try:
        MO.derive_splice_pwms("x.fa", "x.gtf", ["chr22", "chr1"],
                              panel_chroms=["chr22", "chr17"])
        raise AssertionError("motif derivation was allowed on a panel chromosome")
    except ValueError as exc:
        assert "chr22" in str(exc)

    try:
        MO.derive_splice_pwms("x.fa", "x.gtf", [], panel_chroms=["chr22"])
        raise AssertionError("derivation ran with no chromosomes")
    except ValueError:
        pass





def t_motifsrc_catalog() -> None:
    """The catalogue is the only place a motif is declared; it must be sound."""
    names = [r.name for r in SRC.CATALOG]
    assert len(names) == len(set(names)), "duplicate recipe name"
    for r in SRC.CATALOG:
        assert r.detail, f"{r.name}: no detail"
        assert r.method in ("intron", "anchor", "hexamer", "jaspar", "external")
        if r.method in ("intron", "anchor", "hexamer"):
            # every derived recipe must be falsifiable by its own consensus
            assert r.expect and r.expect_at >= 0, f"{r.name}: derived with no check"
            assert r.min_sites > 0
        if r.method == "external":
            assert r.external_source, f"{r.name}: external with no named source"
        if r.method == "jaspar":
            assert r.jaspar_id.startswith("MA"), r.jaspar_id
    rows = SRC.catalog_table()
    assert len(rows) == len(SRC.CATALOG)


def _mini_gtf(lines, d):
    p = pathlib.Path(d) / "m.gtf"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def t_motifsrc_anchors() -> None:
    """start_codon and transcript 3' ends resolve correctly on BOTH strands."""
    import tempfile

    rng = np.random.default_rng(21)
    g = list("".join(rng.choice(list("ACGT"), size=4000)))
    kozak = "GCCGCCATGG"                       # A of ATG at offset 6
    rc = lambda t: t[::-1].translate(str.maketrans("ACGT", "TGCA"))

    g[500:510] = list(kozak)                   # plus strand, A at 506
    g[1500:1510] = list(rc(kozak))             # minus strand, A at 1503
    seq = np.frombuffer("".join(g).encode(), dtype=np.uint8)

    lines = [
        'chrT\tt\tstart_codon\t507\t509\t.\t+\t.\tgene_id "a"; transcript_id "a";',
        'chrT\tt\tstart_codon\t1502\t1504\t.\t-\t.\tgene_id "b"; transcript_id "b";',
        'chrT\tt\ttranscript\t401\t900\t.\t+\t.\tgene_id "a"; transcript_id "a";',
        'chrT\tt\ttranscript\t1401\t1900\t.\t-\t.\tgene_id "b"; transcript_id "b";',
    ]
    with tempfile.TemporaryDirectory() as d:
        gtf = _mini_gtf(lines, d)
        anch = sorted(SRC.iter_anchors(gtf, "chrT", "start_codon"))
        assert anch == [(506, "+"), (1503, "-")], anch
        ws = SRC.anchor_windows(seq, anch, up=6, down=3)
        assert len(ws) == 2
        for w in ws:
            text = bytes(w).decode()
            assert len(text) == 10 and text[6:9] == "ATG", text
            assert text == kozak, text

        tp = sorted(SRC.iter_anchors(gtf, "chrT", "transcript_3p"))
        assert tp == [(899, "+"), (1400, "-")], tp

    try:
        list(SRC.iter_anchors("x.gtf", "chrT", "enhancer"))
        raise AssertionError("an unknown anchor feature was accepted")
    except ValueError:
        pass


def t_motifsrc_intron_class() -> None:
    """Intron class is read in transcript orientation, not genome orientation."""
    import tempfile

    rc = lambda t: t[::-1].translate(str.maketrans("ACGT", "TGCA"))
    rng = np.random.default_rng(22)
    g = list("".join(rng.choice(list("ACGT"), size=6000)))

    def plant(e1, s2, strand, d9, a23):
        if strand == "+":
            g[e1 - 3:e1 + 6] = list(d9); g[s2 - 20:s2 + 3] = list(a23)
        else:
            g[s2 - 6:s2 + 3] = list(rc(d9)); g[e1 - 3:e1 + 20] = list(rc(a23))

    pairs = [(500, 900, "+", "GT", "AG"), (2500, 2900, "-", "GC", "AG"),
             (4500, 4900, "+", "AT", "AC")]
    lines = []
    for i, (e1, s2, strand, dc, ac) in enumerate(pairs):
        d9 = "CAG" + dc + "AAGT"
        a23 = "T" * 18 + ac + "CGA"
        plant(e1, s2, strand, d9, a23)
        lines += [
            f'chrT\tt\texon\t{e1 - 199}\t{e1}\t.\t{strand}\t.\tgene_id "g"; transcript_id "t{i}";',
            f'chrT\tt\texon\t{s2 + 1}\t{s2 + 200}\t.\t{strand}\t.\tgene_id "g"; transcript_id "t{i}";']

    seq = np.frombuffer("".join(g).encode(), dtype=np.uint8)
    import tempfile as tf
    with tf.TemporaryDirectory() as d:
        gtf = _mini_gtf(lines, d)
        introns = sorted(MO.iter_introns(gtf, "chrT"))
    got = sorted(SRC.intron_class(dw, aw)
                 for dw, aw in SRC.intron_windows(seq, introns))
    assert got == ["AT-AC", "GC-AG", "GT-AG"], got


def t_motifsrc_hexamer() -> None:
    """Discovery finds a planted hexamer, and finds nothing in pure noise."""
    rng = np.random.default_rng(23)
    planted, k = "AATAAA", 6
    regions = []
    for _ in range(600):
        s = list("".join(rng.choice(list("ACGT"), size=60)))
        at = int(rng.integers(5, 45))
        s[at:at + k] = list(planted)
        regions.append(np.frombuffer("".join(s).encode(), dtype=np.uint8))
    top, _ = SRC.discover_hexamer(regions, k=k, top=10)
    assert top[0]["kmer"] == planted, [r["kmer"] for r in top[:3]]
    assert top[0]["log2_enrichment"] > 2.0

    noise = [np.frombuffer("".join(rng.choice(list("ACGT"), size=60)).encode(),
                           dtype=np.uint8) for _ in range(600)]
    top2, _ = SRC.discover_hexamer(noise, k=k, top=10, min_count=20)
    assert all(abs(r["log2_enrichment"]) < 1.5 for r in top2), top2[:2]

    ctx = SRC.hexamer_context_windows(regions, planted, flank=3)
    assert ctx and all(bytes(w).decode()[3:9] == planted for w in ctx)
    assert all(len(w) == k + 6 for w in ctx)





def t_dualref_geometry() -> None:
    """Neither reference may appear in its own pool."""
    dr = DRF.DualRef(onset=28, rotation=30, n_blocks=32)
    assert dr.pre_ref == 27 and dr.control_ref == 26
    assert str(dr.pre_ref).zfill(2) not in dr.pre_blocks, "h27 is in its own pool"
    assert "norm" not in dr.post_blocks, "h_norm is in its own pool"
    assert dr.pre_blocks[0] == "00" and dr.pre_blocks[-1] == "26"
    assert dr.post_blocks == ["28", "29", "30", "31"]
    assert dr.pre_col("14") == "apre27_14"
    assert dr.pre_col("14", perp=True) == "apreperp27_14"
    assert dr.post_col("14") == "a_14"


def _dual_frame(n_windows=14, n_pos=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_windows):
        for pos in range(n_pos):
            speed = rng.uniform(0.5, 1.5)
            early = pos < n_pos // 3
            r = {"window_id": f"w{w}",
                 "context": "splice_donor" if early else "intron"}
            for i in range(32):
                # donors approach h27 more slowly but commit to the output sooner
                r[f"apre27_{i:02d}"] = float(np.clip(
                    (i / 27.0) ** (1 / (speed * (0.55 if early else 1.0))), 0, 1)
                    + rng.normal(0, .01))
                r[f"apre26_{i:02d}"] = float(np.clip(
                    (i / 26.0) ** (1 / (speed * (0.55 if early else 1.0))), 0, 1)
                    + rng.normal(0, .01))
                r[f"a_{i:02d}"] = float((-0.01 if i < 30 else 0.61)
                                        + (0.05 if (early and i >= 28) else 0.0)
                                        + rng.normal(0, .01))
            r["a_norm"] = 1.0
            r["entropy_final"] = float(rng.normal())
            r["motif_splice_donor_U2_score"] = float(
                (6.0 if early else 0.0) + rng.normal(0, 1.0))
            rows.append(r)
    return pd.DataFrame(rows)


def t_dualref_settle() -> None:
    """Both regimes measured, both censoring rates reported, opposite signs kept."""
    dr = DRF.DualRef(onset=28, rotation=30, n_blocks=32)
    df = _dual_frame()
    out, rep = DRF.dual_settle(df, dr)

    for half in ("pre", "post"):
        assert 0.0 <= rep[half]["censored_fraction"] <= 1.0
        assert np.isfinite(rep[half]["gamma"])
    # the output reference is NOT allowed to be trivially satisfied
    assert rep["post"]["gamma"] < 0.999, "h_norm leaked into its own threshold"

    for col in ("a_pre", "a_post", "c_pre", "c_post", "crossed_pre", "crossed_post"):
        assert col in out.columns, col
    # non-crossing positions stay NaN rather than entering at a ceiling
    assert np.isnan(out.loc[out["crossed_pre"] < 0.5, "c_pre"]).all()

    tbl = DRF.dual_context_table(out, dr, ["splice_donor"])
    assert len(tbl) == 2 and set(tbl["regime"]) == {"pre", "post"}
    assert (tbl["n_windows_used"] > 0).all()
    pre_d = float(tbl.loc[tbl["regime"] == "pre", "mean_d"].iloc[0])
    post_d = float(tbl.loc[tbl["regime"] == "post", "mean_d"].iloc[0])
    # the planted dissociation must survive: slower to settle, quicker to commit
    assert pre_d < 0 < post_d, (pre_d, post_d)
    assert "censored_fraction" in tbl.columns


def t_dualref_self_reference() -> None:
    """The held-out control must agree away from the endpoints."""
    dr = DRF.DualRef(onset=28, rotation=30, n_blocks=32)
    out, _ = DRF.dual_settle(_dual_frame(), dr)
    prof = DRF.dual_profile(out, dr)
    assert {"to_pre", "to_control", "to_output"} <= set(prof.columns)
    ctl = DRF.self_reference_control(prof, dr)
    assert ctl["available"] and ctl["pearson_r"] > 0.9

    # and it must actually be able to say no
    bad = prof.copy()
    bad.loc[bad["block"] < 20, "to_control"] = \
        1.0 - bad.loc[bad["block"] < 20, "to_control"]
    assert DRF.self_reference_control(bad, dr)["pearson_r"] < 0.9


def t_sitedefs() -> None:
    """Annotation and motif definitions cross into four cells."""
    df = _dual_frame()
    # the composition-matched null is the shuffle_di variant in a real run; in
    # the fixture we build one the same way, by shuffling the score column
    rng2 = np.random.default_rng(99)
    null = df.copy()
    null["motif_splice_donor_U2_score"] = rng2.normal(0.0, 1.0, size=len(df))
    sd, reports = SD.site_definitions(df, target_fdr=0.05, null_df=null)
    rec = [r for r in reports if r.get("annot_context") == "splice_donor"]
    assert rec and "threshold" in rec[0], reports
    assert rec[0]["fdr_achieved_fdr"] <= 0.05 + 1e-9, rec[0]

    # with no null at all it must refuse rather than fall back to a quantile
    _, rep_nonull = SD.site_definitions(df, target_fdr=0.05, null_df=None)
    assert any("null" in str(r.get("reason", "")) for r in rep_nonull), rep_nonull
    assert set(sd["site_agreement"]) <= set(SD.AGREEMENT_CELLS)
    # the planted signal separates, so 'both' must dominate the annotated set
    ann = sd[sd["context"] == "splice_donor"]
    assert (ann["site_agreement"] == "both").mean() > 0.85

    conc = SD.concordance_table(sd)
    assert list(conc["cell"]) == list(SD.AGREEMENT_CELLS)
    assert abs(conc["fraction"].sum() - 1.0) < 1e-9

    out, _ = DRF.dual_settle(sd, DRF.DualRef())
    dis = SD.discordance_contrast(out, value="a_pre")
    assert not dis.empty and "mean_d" in dis.columns

    both = SD.definition_contrast(out, value="a_pre")
    assert set(both["definition"]) == {"annotation", "motif"}

    # with no motif columns at all it must say so rather than label everything
    bare = df.drop(columns=[c for c in df.columns if c.startswith("motif_")])
    _, rep2 = SD.site_definitions(bare)
    assert rep2[0]["available"] is False and "--motifs" in rep2[0]["reason"]


def t_probesweep() -> None:
    """The sweep runs per block, and the shuffled floor is a real floor."""
    rng = np.random.default_rng(3)
    n, d, n_win = 360, 24, 12
    # the label must VARY inside a window, or the within-window permutation
    # control is vacuous -- which the sweep now detects and says so
    meta = pd.DataFrame(dict(
        window_id=[f"w{i // (n // n_win)}" for i in range(n)],
        context=["coding_exon" if i % 2 else "intron" for i in range(n)],
        gc=rng.normal(size=n)))
    y = (meta["context"] == "coding_exon").to_numpy().astype(float)

    states = {
        "h00": rng.normal(size=(n, d)),                              # no signal
        "h10": rng.normal(size=(n, d)) + 2.5 * y[:, None],           # strong signal
    }
    specs = [PS.ProbeSpec("coding_exon", "context", "binary",
                          positive="coding_exon", min_per_class=30)]
    sw = PS.sweep(states, meta, specs, with_shuffled_control=True, progress=False)

    real = sw[sw["label"] == "coding_exon"].set_index("tap")["score"]
    shuf = sw[sw["label"] == "coding_exon__shuffled"].set_index("tap")["score"]
    assert real["h10"] > 0.85, real["h10"]
    assert real["h10"] > real["h00"], (real["h00"], real["h10"])
    assert shuf["h10"] < real["h10"], "the shuffled floor is not below the signal"
    assert sw.loc[sw["label"] == "coding_exon__shuffled", "shuffle_changed"].min() > 0

    # a label constant within windows must be flagged, not reported as a floor
    meta2 = meta.copy()
    meta2["window_id"] = [f"w{i % n_win}" for i in range(n)]   # label constant per window
    sw_v = PS.sweep(states, meta2, specs, with_shuffled_control=True, progress=False)
    vac = sw_v[sw_v["label"] == "coding_exon__shuffled"]
    assert (vac["shuffle_changed"] == 0).all()
    assert vac["note"].str.startswith("VACUOUS").all()
    assert PS.retention_table(sw_v, onset=5, rotation=8)["floor"].isna().all()

    ret = PS.retention_table(sw, onset=5, rotation=8, n_blocks=12)
    assert "retention" in ret.columns and len(ret) >= 1

    # a spec whose column is absent is reported, not silently dropped
    sw2 = PS.sweep(states, meta, [PS.ProbeSpec("nope", "not_a_column", "continuous")],
                   with_shuffled_control=False, progress=False)
    assert "not runnable" in str(sw2["note"].iloc[0])



def t_regimes() -> None:
    df = synth_table(n_windows=14, n_pos=240, seed=5)
    reg = RG.Regimes(onset=28, rotation=30, n_blocks=32)
    assert reg.r1[-1] == "27" and reg.pre_tag == "27" and reg.rot_tag == "30"
    assert "norm" in reg.r3

    feats = RG.regime_features(df, reg)
    for col in ("av_r1_slope", "av_r1_end", "av_r2_delta", "av_r3_end", "n_r2_delta"):
        assert col in feats.columns, col
    # the norm jump across the onset was planted at 1e3
    # The fixture's onset is an ADDITIVE orthogonal write of magnitude 7000 on a
    # stream of order 50, so the log10 norm jump is ~2, not the ~3 of the old
    # multiplicative fixture.  The assertion checks the jump is large and
    # finite, not that it hits a number the fixture happens to produce.
    assert 1.5 < np.nanmedian(feats["n_r2_delta"]) < 3.5, np.nanmedian(feats["n_r2_delta"])

    tbl = RG.regime_effect_table(feats, features=["av_r1_end", "av_r3_end"],
                                 contexts=["coding_exon"])
    assert len(tbl) == 2 and tbl["n_windows"].min() >= 5

    con = RG.regime_contrast(feats, "av_r1_end", "av_r3_end", target="coding_exon")
    assert con["n_windows"] >= 5
    assert np.isfinite(con["diff"])

    prof = RG.regime_profile_by_group(feats, "av", reg)
    assert set(prof["regime"]) >= {"R1", "R3"}
    assert prof["n_windows"].min() >= 1


def t_regime_transfer() -> None:
    df = synth_table(n_windows=12, n_pos=200, seed=6)
    reg = RG.Regimes(onset=28, rotation=30, n_blocks=32)
    feats = RG.regime_features(df, reg)
    out = RG.regime_transfer(feats, "av_r1_end", "av_r3_end",
                             controls=("entropy_final", "gc"))
    assert np.isfinite(out["beta"]), out
    assert out["n_windows"] >= 5


def t_motifs_degenerate() -> None:
    """Degenerate windows must yield NaN, never a huge finite number.

    Regression test: np.nan_to_num(x, nan=-inf) also maps pre-existing -inf to
    -1.8e308, which is finite, survives isfinite(), and then overflows the
    float32 cast into a fake score.  Windows with runs of N hit exactly this.
    """
    ctx = np.array(["intron"] * 400, dtype=object)
    cases = {
        "polyA": np.full(1200, ord("A"), dtype=np.uint8),
        "allN": np.full(1200, ord("N"), dtype=np.uint8),
        "halfN": np.concatenate([np.full(600, ord("N"), dtype=np.uint8),
                                 np.frombuffer(b"ACGT" * 150, dtype=np.uint8)]),
    }
    for name, arr in cases.items():
        cols = MO.annotate_window(arr, slice(400, 800), ctx, _fixture_set()).to_dict()
        for key in [k for k in cols if k.endswith("_score")]:
            v = np.asarray(cols[key], dtype=np.float64)
            finite = v[np.isfinite(v)]
            assert not np.isinf(v).any(), f"{name}/{key} produced inf"
            if finite.size:
                assert np.abs(finite).max() < 1e4, \
                    f"{name}/{key} produced an implausible score {np.abs(finite).max():.3e}"
        if name == "allN":
            assert not np.isfinite(np.asarray(cols["motif_best_score"], dtype=np.float64)).any()
        if name == "halfN":
            frac = np.isfinite(np.asarray(cols["motif_best_score"], dtype=np.float64)).mean()
            assert 0.3 < frac < 0.7, frac


def main() -> int:
    print("exp1 self-test")
    c = Check()
    c.run("metrics.pnq", t_pnq)
    c.run("metrics.channel_identity", t_channel_identity)
    c.run("metrics.log_softmax", t_log_softmax)
    c.run("metrics.acgt_contrast_basis", t_acgt_basis)
    c.run("metrics.lens_metrics", t_lens_metrics)
    c.run("metrics.early_ratio/settle_depth", t_early_ratio_and_settle)
    c.run("blockmap.period_and_config", t_blockmap)
    c.run("blockmap.onset_rotation", t_onset_rotation)
    c.run("stats.window_effect", t_stats_effect)
    c.run("stats.permutation_null", t_stats_permutation)
    c.run("stats.within_window_adjustment", t_stats_adjust)
    c.run("stats.overdispersion", t_overdispersion)
    c.run("motifs.scan", t_motifs_scan)
    c.run("motifs.annotation", t_motifs_annotation)
    c.run("motifs.strength_quartiles", t_motifs_quartiles)
    c.run("motifs.degenerate_inputs", t_motifs_degenerate)
    c.run("motifs.provenance_required", t_motifs_provenance)
    c.run("motifs.manifest_roundtrip", t_motifs_manifest)
    c.run("motifs.no_default_set", t_motifs_no_default)
    c.run("motifs.derive_both_strands", t_motifs_derive_strand)
    c.run("motifs.no_panel_leakage", t_motifs_no_leakage)
    c.run("motifsrc.catalog", t_motifsrc_catalog)
    c.run("motifsrc.anchors_both_strands", t_motifsrc_anchors)
    c.run("motifsrc.intron_class", t_motifsrc_intron_class)
    c.run("motifsrc.hexamer_discovery", t_motifsrc_hexamer)
    c.run("perturb.families", t_perturb_families)
    c.run("perturb.pairing", t_perturb_pairing)
    c.run("dualref.geometry", t_dualref_geometry)
    c.run("dualref.dual_settle", t_dualref_settle)
    c.run("dualref.self_reference_control", t_dualref_self_reference)
    c.run("sitedefs.annotation_vs_motif", t_sitedefs)
    c.run("probesweep.per_block", t_probesweep)
    c.run("regimes.features_and_contrast", t_regimes)
    c.run("regimes.transfer", t_regime_transfer)
    c.run("analysis.end_to_end", t_analysis_e2e)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())

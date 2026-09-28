"""54-d LCP vs 4096-d single-layer embedding — the baseline reviewers will demand.

LCP redefined self-contained (no dependence on the final norm module):
    vel_l = ||h_l - h_{l-1}|| / ||h_{l-1}||          l = 1..L
    cos_l = cos(h_l, h_L)                             l = 0..L-1
Both computed from block outputs only -> portable across architectures.

Baselines, identical positions / folds / classifier:
    h_L 4096-d      the last usable layer  ("just use the final embedding")
    h_best 4096-d   best single layer, chosen from data (favours the baseline)
"""
import numpy as np, zipfile, sys, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP = 4000
RNG = np.random.default_rng(20260905)

def open_H(path):
    """memmap H.npy inside an uncompressed .npz."""
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shp, fo, dt = np.lib.format._read_array_header(fh, ver)
        hdr = fh.tell()
    z.fp.seek(zi.header_offset)
    import struct
    raw = z.fp.read(30)
    n, m = struct.unpack("<HH", raw[26:30])
    off = zi.header_offset + 30 + n + m + hdr
    return np.memmap(path, dtype=dt, mode="r", offset=off, shape=shp), shp

def analyse(chrom, fn):
    path = f"{R}/{fn}"
    mm, (NB, N, D) = open_H(path)
    d = np.load(path.replace("H", "H"), mmap_mode=None) if False else None
    z = np.load(path)                       # small arrays only
    lab, win = z["label"], z["window"]
    print(f"\n{'='*100}\n{chrom}   H = {NB} x {N} x {D}\n{'='*100}")

    # --- which layers survive fp16? ---
    chk = np.sort(RNG.choice(N, 1500, replace=False))
    fin = np.array([np.isfinite(np.asarray(mm[l][chk], dtype=np.float32)).all(1).mean()
                    for l in range(NB)])
    L = int(np.max(np.where(fin > 0.999)[0]))
    print("  finite fraction per layer:",
          " ".join(f"{l}:{f:.2f}" for l, f in enumerate(fin) if f < 0.999) or "all layers finite")
    print(f"  -> last usable layer L = {L}   (layers {L+1}..{NB-1} overflow fp16)")

    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = RNG.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]

        # stream layers 0..L for the selected rows only
        Hs = np.stack([np.asarray(mm[l][rows], dtype=np.float32) for l in range(L + 1)])
        nrm = np.linalg.norm(Hs, axis=-1).clip(1e-12)
        vel = np.linalg.norm(Hs[1:] - Hs[:-1], axis=-1) / nrm[:-1]        # (L, n)
        ref = Hs[L]
        cos = (Hs[:L] * ref).sum(-1) / (nrm[:L] * nrm[L]).clip(1e-12)     # (L, n)
        LCP = np.concatenate([vel.T, cos.T], axis=1)                      # (n, 2L)

        def sc(X):
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 1: X = X[:, None]
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            a = cross_val_score(p, X, y, cv=GroupKFold(5), groups=g,
                                scoring="roc_auc", n_jobs=5).mean()
            return max(a, 1 - a)

        a_lcp  = sc(LCP)
        a_last = sc(Hs[L])
        per_l  = [sc(Hs[l]) for l in range(0, L + 1, 4)]
        a_best = max(per_l); l_best = list(range(0, L + 1, 4))[int(np.argmax(per_l))]
        print(f"\n  {nm}   (n={len(rows)}, pos={y.sum()})")
        print(f"    LCP  {2*L}-d          {a_lcp:.3f}")
        print(f"    h_{L} 4096-d          {a_last:.3f}   <- 'use the final embedding'")
        print(f"    h_{l_best} 4096-d (best/4)  {a_best:.3f}   <- baseline picks its own layer")
        print(f"    ratio LCP/h_{L} = {a_lcp/a_last:.3f}   dim ratio = {4096/(2*L):.0f}x smaller")
        del Hs, vel, cos, LCP

analyse("chr22", "evo2_7b_probe_subsample.npz")
analyse("chr17", "chr17_evo2_7b_probe_subsample.npz")

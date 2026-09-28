"""The pivotal test: at EQUAL dimension, is LCP better than a linear compression
of a single layer?  PCA is fitted inside each training fold (no leakage).

If PCA-k on h_24 beats LCP-k, the compression argument for LCP fails and the
metric's value is interpretability, not efficiency.  Run this before spending
GPU on downstream tasks.
"""
import numpy as np, zipfile, struct
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP, L = 4000, 28
KS = [8, 16, 28, 56]
RNG = np.random.default_rng(20260906)

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); raw = z.fp.read(30)
    n, m = struct.unpack("<HH", raw[26:30])
    return np.memmap(path, dtype=dt, mode="r",
                     offset=zi.header_offset + 30 + n + m + hdr, shape=shp), shp

def analyse(chrom, fn):
    path = f"{R}/{fn}"; mm, (NB, N, D) = open_H(path)
    z = np.load(path); lab, win = z["label"], z["window"]
    print(f"\n{'='*104}\n{chrom}\n{'='*104}")
    hdr = f"  {'region':<17}{'dim':>5}{'LCP':>9}{'PCA(h24)':>11}{'PCA(h28)':>11}{'winner':>10}"
    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = RNG.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]

        Hs = np.stack([np.asarray(mm[l][rows], dtype=np.float32) for l in range(L + 1)])
        nrm = np.linalg.norm(Hs, axis=-1).clip(1e-12)
        vel = np.linalg.norm(Hs[1:] - Hs[:-1], axis=-1) / nrm[:-1]
        cos = (Hs[:L] * Hs[L]).sum(-1) / (nrm[:L] * nrm[L]).clip(1e-12)
        LCP_full = np.concatenate([vel.T, cos.T], axis=1)          # (n, 56)
        h24, h28 = Hs[24].copy(), Hs[28].copy(); del Hs

        def sc(X, k=None, pca=False):
            steps = [StandardScaler()]
            if pca: steps.append(PCA(n_components=k, random_state=0))
            steps.append(LogisticRegression(max_iter=3000))
            a = cross_val_score(make_pipeline(*steps), X, y, cv=GroupKFold(5),
                                groups=g, scoring="roc_auc", n_jobs=5).mean()
            return max(a, 1 - a)

        print(f"\n{hdr}" if code == list(CTX)[0] else "")
        for k in KS:
            # LCP truncated to k dims: keep evenly spaced layers from both gauges
            idx = np.linspace(0, L - 1, k // 2).astype(int)
            Xl = np.concatenate([LCP_full[:, idx], LCP_full[:, L + idx]], axis=1)
            a_l, a24, a28 = sc(Xl), sc(h24, k, True), sc(h28, k, True)
            w = max([("LCP", a_l), ("PCA24", a24), ("PCA28", a28)], key=lambda t: t[1])
            print(f"  {nm if k==KS[0] else '':<17}{k:>5}{a_l:>9.3f}{a24:>11.3f}"
                  f"{a28:>11.3f}{w[0]:>10}")
        del LCP_full, h24, h28

analyse("chr22", "evo2_7b_probe_subsample.npz")
analyse("chr17", "chr17_evo2_7b_probe_subsample.npz")

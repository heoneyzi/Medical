"""Is the trajectory redundant GIVEN a single mid-late layer?

  A  PCA_56(h24)              layer alone
  B  LCP_56                   trajectory alone
  C  [PCA_56(h24) | LCP_56]   112-d  -- does the path ADD anything?
  D  PCA_112(h24)             112-d  -- dimension-matched control for C

If C == D, the path contributes nothing beyond the layer and the gain in C is
just extra dimensions.  If C > D, the path carries something h24 does not.

Also: how much of LCP is linearly predictable from h24 (ridge R^2)?
"""
import numpy as np, zipfile, struct
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP, L, K = 4000, 28, 56
RNG = np.random.default_rng(20260906)

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); n, m = struct.unpack("<HH", z.fp.read(30)[26:30])
    return np.memmap(path, dtype=dt, mode="r",
                     offset=zi.header_offset + 30 + n + m + hdr, shape=shp), shp

class Concat:
    """PCA the layer inside the fold, then glue the (already-cheap) LCP on."""
    def __init__(s, k): s.k = k
    def fit(s, X, y=None):
        s.p = PCA(n_components=s.k, random_state=0).fit(X[:, :4096]); return s
    def transform(s, X):
        return np.hstack([s.p.transform(X[:, :4096]), X[:, 4096:]])
    def fit_transform(s, X, y=None): return s.fit(X).transform(X)
    def get_params(s, deep=True): return {"k": s.k}
    def set_params(s, **kw): s.k = kw.get("k", s.k); return s

def analyse(chrom, fn):
    path = f"{R}/{fn}"; mm, _ = open_H(path)
    z = np.load(path); lab, win = z["label"], z["window"]
    print(f"\n{'='*96}\n{chrom}\n{'='*96}")
    print(f"  {'region':<17}{'A layer':>10}{'B path':>9}{'C A+B':>9}{'D dim-ctl':>11}"
          f"{'C-D':>8}{'LCP|h24 R2':>12}")
    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = RNG.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]
        cv = GroupKFold(5)

        Hs = np.stack([np.asarray(mm[l][rows], dtype=np.float32) for l in range(L + 1)])
        nrm = np.linalg.norm(Hs, axis=-1).clip(1e-12)
        vel = np.linalg.norm(Hs[1:] - Hs[:-1], axis=-1) / nrm[:-1]
        cos = (Hs[:L] * Hs[L]).sum(-1) / (nrm[:L] * nrm[L]).clip(1e-12)
        LCP = np.concatenate([vel.T, cos.T], axis=1)
        h24 = Hs[24].copy(); del Hs

        def auc(pipe, X):
            a = cross_val_score(pipe, X, y, cv=cv, groups=g,
                                scoring="roc_auc", n_jobs=5).mean()
            return max(a, 1 - a)
        lr = lambda: LogisticRegression(max_iter=3000)
        A = auc(make_pipeline(StandardScaler(), PCA(K, random_state=0), lr()), h24)
        B = auc(make_pipeline(StandardScaler(), lr()), LCP)
        XC = np.hstack([h24, LCP])
        C = auc(make_pipeline(StandardScaler(), Concat(K), lr()), XC)
        D = auc(make_pipeline(StandardScaler(), PCA(2 * K, random_state=0), lr()), h24)

        # how much of the trajectory is linearly readable off h24 alone?
        pred = cross_val_pred = cross_val_predict(
            make_pipeline(StandardScaler(), PCA(256, random_state=0), Ridge(alpha=10.0)),
            h24, LCP, cv=cv, groups=g)
        ss_res = ((LCP - pred) ** 2).sum(0); ss_tot = ((LCP - LCP.mean(0)) ** 2).sum(0)
        r2 = float(np.median(1 - ss_res / ss_tot.clip(1e-12)))
        print(f"  {nm:<17}{A:>10.3f}{B:>9.3f}{C:>9.3f}{D:>11.3f}{C-D:>+8.3f}{r2:>12.3f}")
        del h24, LCP, XC

analyse("chr22", "evo2_7b_probe_subsample.npz")
analyse("chr17", "chr17_evo2_7b_probe_subsample.npz")

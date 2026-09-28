"""Is the post-cascade collapse real, or an artifact of PCA(256) in my probe?

If the blow-up channels monopolise variance, PCA(256) may discard the informative
directions.  Test h_norm and h30 with no PCA at all, and with a wider PCA.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 0: "intergenic", 5: "splice_donor"}
CAP = 4000

for chrom in ("chr22", "chr17"):
    d = np.load(f"{R}/{chrom}_late_fp32.npz")
    lab, win = d["label"], d["window"]
    print(f"\n{'='*112}\n{chrom}\n{'='*112}")
    print(f"  {'region':<15}{'layer':<8}{'PCA256':>10}{'PCA1024':>10}{'raw 4096':>11}"
          f"{'eff.rank':>10}{'var in top256':>15}")
    for code, nm in CTX.items():
        rng = np.random.default_rng(7)
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = rng.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]
        for key in ("h27", "h30", "hnorm"):
            X = np.nan_to_num(np.asarray(d[key][rows], dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            Xs = StandardScaler().fit_transform(X)
            ev = np.linalg.eigvalsh(np.cov(Xs, rowvar=False))[::-1].clip(0, None)
            p = ev / ev.sum()
            eff = float(np.exp(-(p * np.log(p.clip(1e-16))).sum()))
            top256 = float(p[:256].sum())
            out = []
            for k in (256, 1024, None):
                steps = [StandardScaler()]
                if k: steps.append(PCA(n_components=k, random_state=0))
                steps.append(LogisticRegression(max_iter=4000))
                a = cross_val_score(make_pipeline(*steps), X, y, cv=GroupKFold(5),
                                    groups=g, scoring="roc_auc", n_jobs=5).mean()
                out.append(max(a, 1 - a))
            print(f"  {nm if key=='h27' else '':<15}{key:<8}{out[0]:>10.3f}{out[1]:>10.3f}"
                  f"{out[2]:>11.3f}{eff:>10.1f}{top256:>15.3f}", flush=True)

"""BRCA1 saturation genome editing by layer (Evo 2 7B).

Mirrors clinvar_eval.py, with two differences forced by the data:
  * one gene, so folds are grouped by EXON rather than by gene;
  * a continuous functional score rather than a binary label.

Spearman is the primary metric because it needs no threshold.  We also report a
binary AUROC, but the split is derived from a two-component Gaussian mixture on
the scores and the implied cutoff is printed, because we do not hold Findlay's
published functional cutoffs and inventing one would not be auditable.
"""
import numpy as np, json
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

R = "/path/to/TDiG/arch_compare/results"
meta = np.load(f"{R}/brca1_meta.npz", allow_pickle=True)
score = meta["score"].astype(float)
exon  = meta["exon"].astype(int)
llr   = meta["llr"].astype(float)
keys  = [str(k) for k in meta["keys"]]
NB = len(keys) - 1
print(f"variants {len(score)} | exons {len(set(exon.tolist()))} | layers {len(keys)}", flush=True)
print(f"score: min {score.min():.3f} med {np.median(score):.3f} max {score.max():.3f}", flush=True)

gm = GaussianMixture(2, random_state=0).fit(score.reshape(-1, 1))
lo = int(np.argmin(gm.means_.ravel()))
yb = (gm.predict(score.reshape(-1, 1)) == lo).astype(int)
cut = float(np.max(score[yb == 1]))
print(f"GMM split: {yb.sum()} abnormal / {len(yb) - yb.sum()} normal | implied cutoff <= {cut:.3f}",
      flush=True)

auc = lambda a: max(a, 1 - a)
print(f"zero-shot LLR: spearman {spearmanr(llr, score).statistic:+.3f} | "
      f"AUROC {auc(roc_auc_score(yb, llr)):.3f}", flush=True)

cv = GroupKFold(5)
def evaluate(X):
    pr = make_pipeline(StandardScaler(), PCA(128, random_state=0), Ridge(alpha=1.0))
    pred = cross_val_predict(pr, X, score, cv=cv, groups=exon, n_jobs=4)
    rho = spearmanr(pred, score).statistic
    pc = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(max_iter=3000))
    a = cross_val_score(pc, X, yb, cv=cv, groups=exon, scoring="roc_auc", n_jobs=4).mean()
    return float(rho), float(auc(a))

out = {"keys": keys, "n": int(len(score)), "cutoff": cut,
       "n_abnormal": int(yb.sum()),
       "zero_shot": {"spearman": float(spearmanr(llr, score).statistic),
                     "auroc": float(auc(roc_auc_score(yb, llr)))}}
for style in ("tok", "pool"):
    A = np.load(f"{R}/brca1_dh_{style}.npy", mmap_mode="r")
    rhos, aucs = [], []
    for j, k in enumerate(keys):
        X = np.nan_to_num(np.asarray(A[:, j, :], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        r, a = evaluate(X)
        rhos.append(r); aucs.append(a)
        print(f"  {style} {k:>5}: spearman {r:+.3f} | AUROC {a:.3f}", flush=True)
    out[style] = {"spearman": rhos, "auroc": aucs}
    rr = np.abs(np.array(rhos[:NB])); aa = np.array(aucs[:NB])
    print(f"  --> {style}: rule b24 spearman {rhos[24]:+.3f} AUROC {aucs[24]:.3f} | "
          f"final {keys[-1]} spearman {rhos[-1]:+.3f} AUROC {aucs[-1]:.3f} | "
          f"oracle b{int(rr.argmax())} |rho| {rr.max():.3f} / b{int(aa.argmax())} AUROC {aa.max():.3f}",
          flush=True)
    json.dump(out, open(f"{R}/brca1_eval.json", "w"), indent=1)
print("done", flush=True)

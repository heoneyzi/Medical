"""ClinVar missense pathogenicity by layer (Evo 2 7B), two feature styles.

  dh_tok   = h_alt - h_ref at the variant position
  dh_pool  = window mean-pool difference      <- the style used in DeltaH
Both are evaluated per layer with the same readout and leave-genes-out folds, so
the comparison isolates WHERE you read, not how you featurise.

Also reported: ||dh||_2 as a single scalar per layer (the DeltaH probe), and the
model's own zero-shot log-likelihood ratio as a reference point.
"""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

R = "/path/to/TDiG/arch_compare/results"
meta = np.load(f"{R}/clinvar17_meta.npz", allow_pickle=True)
y = meta["y"].astype(int); gene = meta["gene"]; llr = meta["llr"]; keys = [str(k) for k in meta["keys"]]
NB = len(keys) - 1
print(f"variants {len(y)} | pathogenic {y.sum()} | genes {len(set(gene.tolist()))}", flush=True)
auc = lambda a: max(a, 1 - a)
print(f"zero-shot log-likelihood ratio AUROC: {auc(roc_auc_score(y, llr)):.3f}", flush=True)

cv = GroupKFold(5)
def probe(X):
    p = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(max_iter=3000))
    return auc(cross_val_score(p, X, y, cv=cv, groups=gene, scoring="roc_auc", n_jobs=4).mean())

out = {}
for style in ("tok", "pool"):
    A = np.load(f"{R}/clinvar17_dh_{style}.npy", mmap_mode="r")
    vec, scal = [], []
    for j, k in enumerate(keys):
        X = np.nan_to_num(np.asarray(A[:, j, :], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        vec.append(probe(X))
        scal.append(auc(roc_auc_score(y, np.linalg.norm(X, axis=1))))
        if j % 8 == 0: print(f"  {style} layer {k}: vector {vec[-1]:.3f} | ||dh|| {scal[-1]:.3f}", flush=True)
    out[style] = {"keys": keys, "vector": vec, "norm_scalar": scal}
    v = np.array(vec[:NB]); s = np.array(scal[:NB])
    print(f"\n  {style}: best block {int(v.argmax())} = {v.max():.3f} | last block {v[-1]:.3f} | "
          f"h_norm {vec[NB]:.3f} | pre-onset band (24-27) {v[24:28].max():.3f}")
    print(f"  {style} ||dh|| scalar: best block {int(s.argmax())} = {s.max():.3f} | h_norm {scal[NB]:.3f}")
    print("   vector curve: " + " ".join(f"{i}:{x:.3f}" for i, x in enumerate(vec)))
json.dump(out, open(f"{R}/clinvar_eval.json", "w"), indent=1)
print("\nwrote clinvar_eval.json")

"""Close the two gaps: evaluate L29-31 (never seen -- they overflowed fp16) and
h_norm (what people actually call "the embedding")."""
import numpy as np, sys, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP = 4000
BT = {24:"attn",25:"hcs",26:"hcm",27:"hcl",28:"hcs",29:"hcm",30:"hcl",31:"attn"}
RNG = np.random.default_rng(20260906)

prev = {r["chrom"]: r["auroc"] for r in json.load(open(f"{R}/layer_select.json"))}
out = {}
for chrom in sys.argv[1:]:
    d = np.load(f"{R}/{chrom}_late_fp32.npz", mmap_mode=None)
    lab, win = d["label"], d["window"]
    keys = [f"h{l}" for l in range(24, 32)] + ["hnorm"]
    print(f"\n{'='*100}\n{chrom}\n{'='*100}")
    print(f"  {'region':<17}" + "".join(f"{k:>9}" for k in keys) +
          f"{'oracle L0-28':>14}{'best late':>11}")
    out[chrom] = {}
    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = RNG.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]
        vals = []
        for k in keys:
            X = np.asarray(d[k][rows], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            p = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                              LogisticRegression(max_iter=3000))
            a = cross_val_score(p, X, y, cv=GroupKFold(5), groups=g,
                                scoring="roc_auc", n_jobs=4).mean()
            vals.append(max(a, 1 - a))
        oc = max(prev[chrom][nm])
        out[chrom][nm] = dict(zip(keys, vals))
        bi = int(np.argmax(vals))
        print(f"  {nm:<17}" + "".join(f"{v:>9.3f}" for v in vals) +
              f"{oc:>14.3f}{keys[bi]:>11}", flush=True)
json.dump(out, open(f"{R}/late_eval.json", "w"), indent=1)

"""Late-layer evaluation with error bars: R independent subsample draws, so the
rule comparisons can be read against actual sampling noise."""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R_ = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP, REPS = 4000, 5
KEYS = [f"h{l}" for l in range(24, 32)] + ["hnorm"]

res = {}
for chrom in ("chr22", "chr17"):
    d = np.load(f"{R_}/{chrom}_late_fp32.npz")
    lab, win = d["label"], d["window"]
    cache = {k: d[k] for k in KEYS}
    print(f"\n{'='*118}\n{chrom}   ({REPS} independent subsamples, mean +- sd)\n{'='*118}")
    print(f"  {'region':<17}" + "".join(f"{k:>13}" for k in KEYS))
    res[chrom] = {}
    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        acc = {k: [] for k in KEYS}
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + rep)
            rows = []
            for c in (code, 1):
                jj = np.where(lab == c)[0]
                if len(jj) > CAP: jj = rng.choice(jj, CAP, replace=False)
                rows.append(jj)
            rows = np.sort(np.concatenate(rows))
            y = (lab[rows] == code).astype(int); g = win[rows]
            for k in KEYS:
                X = np.nan_to_num(np.asarray(cache[k][rows], dtype=np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
                p = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                                  LogisticRegression(max_iter=3000))
                a = cross_val_score(p, X, y, cv=GroupKFold(5), groups=g,
                                    scoring="roc_auc", n_jobs=4).mean()
                acc[k].append(max(a, 1 - a))
        res[chrom][nm] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in acc.items()}
        print(f"  {nm:<17}" + "".join(
            f"{np.mean(acc[k]):>8.3f}±{np.std(acc[k],ddof=1):<4.3f}" for k in KEYS), flush=True)

print(f"\n{'='*118}\ngain of the best pre-cascade layer over h_norm (the conventional embedding)\n{'='*118}")
print(f"  {'':<6}{'region':<17}{'best pre-cascade':>20}{'h_norm':>16}{'gain':>16}")
for chrom, r in res.items():
    for nm, v in r.items():
        pre = {k: v[k] for k in KEYS[:4]}                    # h24..h27
        bk = max(pre, key=lambda k: pre[k][0])
        m, s = pre[bk]; hm, hs = v["hnorm"]
        g = m - hm; gs = float(np.hypot(s, hs))
        print(f"  {chrom:<6}{nm:<17}{f'{m:.3f}±{s:.3f} ({bk})':>20}"
              f"{f'{hm:.3f}±{hs:.3f}':>16}{f'{g:+.3f}±{gs:.3f}':>16}")
json.dump(res, open(f"{R_}/late_eval_reps.json", "w"), indent=1)

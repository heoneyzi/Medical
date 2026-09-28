"""Re-score every (layer, cell) WITHOUT the orientation-free convention.

analyze_layers.py line 59 records max(a, 1-a).  A reviewer objects that for a
supervised per-block probe a test AUROC below 0.5 signals a generalisation
failure rather than a sign flip, so the convention can mask problems and lifts
the null above 0.5.  This re-runs the identical protocol and records the raw
cross-validated AUROC, so we can say how often raw < 0.5 actually occurs and
whether any reported number depended on the convention.
"""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

AC = "/path/to/TDiG/arch_compare"
DIRS = [("7B", "chr22", "chr22_evo2_7b_utr"), ("7B", "chr17", "chr17_evo2_7b_utr"),
        ("7B", "chr19", "chr19_evo2_7b_utr"), ("40B", "chr22", "chr22_evo2_40b"),
        ("40B", "chr17", "chr17_evo2_40b"), ("40B", "chr19", "chr19_evo2_40b_all")]
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic",
       3: "utr5", 4: "utr3"}
CAP = 4000

out = {}
for scale, chrom, dname in DIRS:
    D = f"{AC}/results/{dname}"
    if not os.path.isdir(D):
        print(f"{dname}: MISSING", flush=True); continue
    prof = json.load(open(f"{D}/profile.json")); NB = prof["blocks"]
    has_norm = os.path.exists(f"{D}/norm.npy")
    keys = [f"b{i}" for i in range(NB)] + (["norm"] if has_norm else [])
    meta = np.load(f"{D}/meta.npz"); lab, win = meta["label"], meta["window"]
    H = {k: np.load(f"{D}/{k}.npy", mmap_mode="r") for k in keys}
    print(f"=== {scale} {chrom}: {NB} blocks, {len(lab)} rows ===", flush=True)
    rec = {}
    for code, nm in CTX.items():
        r2 = np.random.default_rng(20260916); rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = r2.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]
        if y.sum() < 200 or (1 - y).sum() < 200:
            continue
        raw = []
        for k in keys:
            X = np.nan_to_num(np.asarray(H[k][rows], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            a = cross_val_score(make_pipeline(StandardScaler(), PCA(256, random_state=0),
                                              LogisticRegression(max_iter=3000)),
                                X, y, cv=GroupKFold(5), groups=g,
                                scoring="roc_auc", n_jobs=4).mean()
            raw.append(float(a))
        rec[nm] = raw
        below = [(keys[i], round(v, 3)) for i, v in enumerate(raw) if v < 0.5]
        print(f"  {nm:<16} min raw {min(raw):.3f} | below 0.5: {len(below)} {below[:6]}", flush=True)
    out[f"{scale}_{chrom}"] = {"keys": keys, "raw_auroc": rec}
    json.dump(out, open(f"{AC}/results/raw_auroc.json", "w"), indent=1)

tot = sum(1 for v in out.values() for c in v["raw_auroc"].values() for x in c)
bad = sum(1 for v in out.values() for c in v["raw_auroc"].values() for x in c if x < 0.5)
print(f"\nTOTAL (layer, cell) pairs {tot} | raw AUROC < 0.5: {bad} ({100*bad/max(tot,1):.2f}%)", flush=True)
print("done", flush=True)

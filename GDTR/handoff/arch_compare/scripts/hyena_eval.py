"""Evaluate V and PRA on HyenaDNA with the same protocol used for Evo 2:
leave-window-out logistic AUROC and within-window paired d, per context."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
d = np.load(f"{R}/hyenadna_chr22_VPRA.npz")
V, PRA, VL, PL = d["V"], d["PRA"], d["V_layer"], d["PRA_layer"]
lab, win = d["label"], d["window"]
nl, rot = int(d["n_layers"]), int(d["rot"])
prof = json.load(open(f"{R}/hyenadna_layer_profile.json"))

CTX = {1: "intron", 2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor",
       0: "intergenic"}
print(f"HyenaDNA-medium-160k | layers {nl} | rotation into L{rot} | n {len(lab):,}")
print(f"\nper-layer mean norm : {np.round(prof['norm'],2).tolist()}")
print(f"per-layer cos-final : {np.round(prof['cos_final'],3).tolist()}")
print(f"per-layer mean vel  : {np.round(prof['vel'],3).tolist()}")
nrm = np.array(prof["norm"])
print(f"norm growth L0->last: {nrm[-1]/nrm[0]:.1f}x   max single-layer ratio "
      f"{np.max(nrm[1:]/np.maximum(nrm[:-1],1e-9)):.1f}x")

print("\n" + "=" * 82)
print("leave-window-out logistic AUROC (context vs intron)")
print("=" * 82)
print(f"  {'context':<18}{'n':>9}{'V':>9}{'PRA':>9}{'V+PRA':>9}")
rows = {}
for code, nm in CTX.items():
    if nm == "intron" or (lab == code).sum() < 500:
        continue
    m = (lab == code) | (lab == 1)
    y = (lab[m] == code).astype(int)
    g = win[m]
    out = {}
    for k, X in (("V", V[m][:, None]), ("PRA", PRA[m][:, None]),
                 ("V+PRA", np.stack([V[m], PRA[m]], 1))):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        out[k] = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=g,
                                 scoring="roc_auc").mean()
    rows[nm] = out
    print(f"  {nm:<18}{int((lab==code).sum()):>9,}{out['V']:>9.3f}"
          f"{out['PRA']:>9.3f}{out['V+PRA']:>9.3f}")

print("\n" + "=" * 82)
print("within-window paired d vs intron (window cluster bootstrap)")
print("=" * 82)
rng = np.random.default_rng(20260821)
wu = np.unique(win)
print(f"  {'context':<18}{'metric':>7}{'d':>9}{'95% CI':>22}{'windows':>9}")
for code, nm in CTX.items():
    if nm == "intron" or (lab == code).sum() < 500:
        continue
    for k, X in (("V", V), ("PRA", PRA)):
        sd = X.std()
        diffs = []
        for w in wu:
            a = X[(win == w) & (lab == code)]
            b = X[(win == w) & (lab == 1)]
            if len(a) >= 30 and len(b) >= 30:
                diffs.append(a.mean() - b.mean())
        diffs = np.array(diffs) / sd
        if len(diffs) < 8:
            continue
        bs = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                       for _ in range(1000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = "*" if lo * hi > 0 else " "
        print(f"  {nm:<18}{k:>7}{diffs.mean():>+9.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]{star}':>22}{len(diffs):>9}")

print("\n" + "=" * 82)
print("layer concentration (homogeneous stack: no block types to lock to)")
print("=" * 82)
for k, A, L in (("V", VL, rot), ("PRA", PL, rot)):
    u, c = np.unique(A, return_counts=True)
    f = c / c.sum()
    top = np.argsort(c)[::-1][:3]
    print(f"  {k}: layers used 0..{L-1}  uniform={100/L:.1f}%  "
          f"top={[(int(u[i]), f'{f[i]*100:.0f}%') for i in top]}  "
          f"max/uniform={f.max()/(1/L):.1f}x")

"""Evo 2 7B: layer-wise supervised probe vs training-free V and PRA.

Note: hidden states were dumped as fp16. From L28 on, the residual norm exceeds
the fp16 range (65504), so those layers are stored as inf and cannot be probed.
They are reported as --- ; all pre-cascade structure (L0-L27) is intact.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
HCS = {0, 4, 7, 11, 14, 18, 21, 25, 28}
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
RNG = np.random.default_rng(20260903)
CAP = 4000

sc = np.load(f"{R}/evo2_7b_scalars.npz")
vel, cos = sc["vel"].astype(np.float32), sc["cos"].astype(np.float32)
lab_a, win_a = sc["label"], sc["window"]
PRE = 27
mu, sd = vel[:, :PRE].mean(0), vel[:, :PRE].std(0).clip(1e-6)
Vz = (vel[:, :PRE] - mu) / sd
V_all, V_lay = Vz.min(1), Vz.argmin(1)
P_all, P_lay = cos[:, :30].max(1), cos[:, :30].argmax(1)
print(f"scalars n={len(lab_a):,}")
for nm, A in (("V", V_lay), ("PRA", P_lay)):
    u, c = np.unique(A, return_counts=True); f = c / c.sum()
    share = sum(f[i] for i in range(len(u)) if int(u[i]) in HCS)
    top = np.argsort(c)[::-1][:3]
    print(f"  {nm:<4} layer: HCS {share:.1%}  top {[(int(u[i]), f'{f[i]*100:.0f}%') for i in top]}")

pr = np.load(f"{R}/evo2_7b_probe_subsample.npz")
H, lab_p, win_p = pr["H"], pr["label"], pr["window"]
NL = H.shape[0]
print(f"\nprobe subsample: {H.shape}", flush=True)

idx_of = {}
for code in list(CTX) + [1]:
    ii = np.where(lab_p == code)[0]
    if len(ii) > CAP:
        ii = RNG.choice(ii, CAP, replace=False)
    idx_of[code] = np.sort(ii)

print("\n" + "=" * 104)
print("LAYER-WISE PROBE (leave-window-out AUROC) vs TRAINING-FREE READOUTS — Evo 2 7B")
print("=" * 104)
print("  layers shown: 0,2,4,...  ( --- = fp16 overflow, not probeable )")
print(f"  {'context':<16}{'n':>6}  " + "".join(f"{l:>6}" for l in range(0, NL, 2)))
res = {}
for code, nm in CTX.items():
    sel = np.concatenate([idx_of[code], idx_of[1]])
    y = (lab_p[sel] == code).astype(int); g = win_p[sel]
    aur = []
    for l in range(NL):
        X = np.asarray(H[l][sel], dtype=np.float32)
        if not np.isfinite(X).all():
            aur.append(np.nan); continue
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=200, n_jobs=-1))
        aur.append(cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=g,
                                   scoring="roc_auc").mean())
    aur = np.array(aur); res[nm] = aur
    fin = np.where(np.isfinite(aur))[0]; a2 = aur[fin]

    m2 = np.isin(lab_a, [code, 1])
    ya, ga = (lab_a[m2] == code).astype(int), win_a[m2]
    def one(x):
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return cross_val_score(p, x[m2][:, None], ya, cv=GroupKFold(5),
                               groups=ga, scoring="roc_auc").mean()
    row = "".join((f"{a:>6.3f}" if np.isfinite(a) else "   ---") for a in aur[::2])
    print(f"  {nm:<16}{len(idx_of[code]):>6}  {row}")
    print(f"     -> best {a2.max():.3f} @L{int(fin[np.argmax(a2)])} | "
          f"worst {a2.min():.3f} @L{int(fin[np.argmin(a2)])} | "
          f"range {a2.max()-a2.min():.3f} || V {one(V_all):.3f}  PRA {one(P_all):.3f}",
          flush=True)

np.savez(f"{R}/evo2_7b_probe_curves.npz", **res)
print("\n" + "=" * 104)
print("DEPTH STRUCTURE — flat like HyenaDNA, or structured?")
print("=" * 104)
for nm, a in res.items():
    b = a[1:][np.isfinite(a[1:])]
    print(f"  {nm:<16} min {b.min():.3f}  max {b.max():.3f}  "
          f"range {b.max()-b.min():.3f}  sd {b.std():.3f}   (n_layers {len(b)})")
print(f"  {'HyenaDNA 참고':<16} min 0.822  max 0.859  range 0.037  sd ~0.012   (splice_donor, L1-L9)")

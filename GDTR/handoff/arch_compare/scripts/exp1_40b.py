"""EXP1-style observational analyses at 40B, on activations already stored.

Two questions, no forward passes needed, so no GPU.

E1-5, recoverability. At 7B the informative result was that the onset is not where
information is lost: h28 recovers h27 with R^2 0.808 while h30 recovers it with 0.270,
so the bottleneck sits at the re-encoder rather than at the onset. Today's ablations
said the same thing at 40B causally, block 23's mixer being the block the output needs.
If the observational measure independently picks out block 23, two different kinds of
evidence agree, which is worth more than either alone.

E1-4, carrier. At 7B one coordinate holds 90.7% of the mean direction's energy and is
the intervention target EXP2 used. The 1B model has no such coordinate. The prediction
here is that 40B does not either, and a negative result bounds the claim honestly.
"""
import numpy as np, json, os, sys
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

R = "/path/to/TDiG/arch_compare/results"
OUT = R + "/exp1_40b.json"
TARGET = 20                      # last pre-onset block; onset is 21
SOURCES = [19, 21, 22, 23, 34, "norm"]
NPC = 256
NROW = 8000
res = {}

def load(d, b):
    name = "norm" if b == "norm" else "b" + str(b)
    return np.load(R + "/" + d + "/" + name + ".npy", mmap_mode="r")

def rows_for(d):
    m = np.load(R + "/" + d + "/meta.npz")
    return m["window"]

print("### E1-5 recoverability: can block 20 be reconstructed from later blocks? ###", flush=True)
for d in ("chr22_evo2_40b", "chr17_evo2_40b"):
    win = rows_for(d)
    n = len(win)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, min(NROW, n), replace=False))
    g = win[idx]
    tgt = np.asarray(load(d, TARGET)[idx], dtype=np.float64)
    tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
    Pt = PCA(NPC, random_state=0).fit(StandardScaler().fit_transform(tgt))
    Yt = Pt.transform(StandardScaler().fit_transform(tgt))
    print("  " + d + "  rows " + str(len(idx)) + "  target b" + str(TARGET), flush=True)
    res.setdefault(d, {})
    for s in SOURCES:
        X = np.asarray(load(d, s)[idx], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        Xp = PCA(NPC, random_state=0).fit_transform(Xs)
        num = 0.0; den = 0.0
        for tr, te in GroupKFold(5).split(Xp, Yt, g):
            pred = Ridge(alpha=1.0).fit(Xp[tr], Yt[tr]).predict(Xp[te])
            num += ((Yt[te] - pred) ** 2).sum()
            den += ((Yt[te] - Yt[tr].mean(0)) ** 2).sum()
        r2 = float(1.0 - num / max(den, 1e-30))
        key = "norm" if s == "norm" else "b" + str(s)
        res[d]["recover_" + key] = r2
        print("     " + key.rjust(5) + " -> b" + str(TARGET) + "   held-out R2 " + ("%.4f" % r2), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

print(flush=True)
print("### E1-4 carrier: does one coordinate dominate the mean direction? ###", flush=True)
us = {}
for d in ("chr22_evo2_40b", "chr17_evo2_40b"):
    H = load(d, "norm")
    n = H.shape[0]
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(n, min(NROW, n), replace=False))
    X = np.asarray(H[idx], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    u = (X / np.clip(nrm, 1e-30, None)).mean(0)
    u = u / max(np.linalg.norm(u), 1e-30)
    us[d] = u
    e = u ** 2
    order = np.argsort(e)[::-1]
    top1 = float(e[order[0]])
    top1pct = float(e[order[: max(1, len(e) // 100)]].sum())
    res.setdefault(d, {})
    res[d]["carrier_top_coord"] = int(order[0])
    res[d]["carrier_top1_share"] = top1
    res[d]["carrier_top1pct_share"] = top1pct
    print("  " + d + "   top coordinate " + str(int(order[0]))
          + "   holds " + ("%.2f%%" % (100 * top1))
          + " of |u|^2,  top 1% holds " + ("%.2f%%" % (100 * top1pct)), flush=True)

ks = list(us)
if len(ks) == 2:
    c = float(np.dot(us[ks[0]], us[ks[1]]))
    res["carrier_cos_chr22_chr17"] = c
    print("  cos(u_chr22, u_chr17) = " + ("%.4f" % c), flush=True)
    print("  7B reference: one coordinate held 90.69% and cos was 0.9989", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print(flush=True)
print("wrote " + OUT, flush=True)

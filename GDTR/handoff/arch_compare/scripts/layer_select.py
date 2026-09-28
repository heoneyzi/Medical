"""(A) Can the best read-out layer be found WITHOUT labels?

Step 1  full sweep: supervised AUROC at EVERY layer 0..28 (the earlier sweep hit
        only every 4th layer, and 24 was the sole attention block in that grid --
        so "attention is best" has never actually been tested).
Step 2  label-free statistics at every layer, computed on a random position
        sample with no annotation whatsoever.
Step 3  do any of them track the supervised curve?

Evo2-7B stripe layout (from the model's own config):
  hcs  0 4 7 11 14 18 21 25 28 | hcm 1 5 8 12 15 19 22 26 29
  hcl  2 6 9 13 16 20 23 27 30 | attn 3 10 17 24 31
"""
import numpy as np, zipfile, struct, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R = "/path/to/TDiG/arch_compare/results"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
CAP, LMAX, NSTAT = 4000, 28, 2048
BT = {}
for n, ls in [("hcs", [0,4,7,11,14,18,21,25,28]), ("hcm", [1,5,8,12,15,19,22,26,29]),
              ("hcl", [2,6,9,13,16,20,23,27,30]), ("attn", [3,10,17,24,31])]:
    for l in ls: BT[l] = n
RNG = np.random.default_rng(20260906)

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); n, m = struct.unpack("<HH", z.fp.read(30)[26:30])
    return np.memmap(path, dtype=dt, mode="r",
                     offset=zi.header_offset + 30 + n + m + hdr, shape=shp), shp

def stats(Xp, X, Xn):
    """Label-free descriptors of one layer. Xp/Xn = previous/next layer."""
    nrm = np.linalg.norm(X, axis=1)
    C = X - X.mean(0)
    G = C @ C.T / len(C)
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
    p = ev / ev.sum().clip(1e-12)
    d = {"norm": float(nrm.mean()),
         "pr": float((ev.sum() ** 2) / (ev ** 2).sum().clip(1e-12)),   # participation ratio
         "eff_rank": float(np.exp(-(p * np.log(p.clip(1e-12))).sum())), # spectral entropy rank
         "top1_var": float(p[0])}
    U = C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
    s = U[RNG.choice(len(U), 512, replace=False)]
    M = s @ s.T; iu = np.triu_indices(len(s), 1)
    d["aniso"] = float(np.abs(M[iu]).mean())                            # anisotropy
    if Xp is not None:
        dv = X - Xp
        d["vel"] = float((np.linalg.norm(dv, axis=1) / np.linalg.norm(Xp, axis=1).clip(1e-12)).mean())
        if Xn is not None:
            dn = Xn - X
            cc = (dv * dn).sum(1) / (np.linalg.norm(dv, axis=1) * np.linalg.norm(dn, axis=1)).clip(1e-12)
            d["curv"] = float(cc.mean())                                # update-to-update turn
    return d

def analyse(chrom, fn):
    path = f"{R}/{fn}"; mm, (NB, N, D) = open_H(path)
    z = np.load(path); lab, win = z["label"], z["window"]
    out = {"chrom": chrom, "auroc": {}, "stats": {}}
    print(f"\n{'#'*92}\n# {chrom}\n{'#'*92}", flush=True)

    # ---------- Step 2: label-free stats (no annotation used) ----------
    srow = np.sort(RNG.choice(N, NSTAT, replace=False))
    cache = {l: np.asarray(mm[l][srow], dtype=np.float32) for l in range(LMAX + 1)}
    S = {}
    for l in range(LMAX + 1):
        S[l] = stats(cache.get(l - 1), cache[l], cache.get(l + 1))
    del cache
    out["stats"] = S
    print("\n  label-free statistics")
    keys = ["norm", "pr", "eff_rank", "top1_var", "aniso", "vel", "curv"]
    print("   L  type " + "".join(f"{k:>12}" for k in keys))
    for l in range(LMAX + 1):
        print(f"  {l:>2}  {BT[l]:<5}" + "".join(
            f"{S[l].get(k, float('nan')):>12.4g}" for k in keys), flush=True)

    # ---------- Step 1: supervised AUROC at every layer ----------
    for code, nm in CTX.items():
        if (lab == code).sum() < 500: continue
        rows = []
        for c in (code, 1):
            jj = np.where(lab == c)[0]
            if len(jj) > CAP: jj = RNG.choice(jj, CAP, replace=False)
            rows.append(jj)
        rows = np.sort(np.concatenate(rows))
        y = (lab[rows] == code).astype(int); g = win[rows]
        curve = []
        for l in range(LMAX + 1):
            X = np.asarray(mm[l][rows], dtype=np.float32)
            p = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                              LogisticRegression(max_iter=3000))
            a = cross_val_score(p, X, y, cv=GroupKFold(5), groups=g,
                                scoring="roc_auc", n_jobs=5).mean()
            curve.append(max(a, 1 - a))
        out["auroc"][nm] = curve
        b = int(np.argmax(curve))
        print(f"\n  {nm}: best L={b} ({BT[b]}) auroc={curve[b]:.3f} | "
              f"L28(hcs)={curve[28]:.3f} | loss={curve[b]-curve[28]:+.3f}")
        print("    " + " ".join(f"{l}:{v:.3f}" for l, v in enumerate(curve)), flush=True)
    return out

res = [analyse("chr22", "evo2_7b_probe_subsample.npz"),
       analyse("chr17", "chr17_evo2_7b_probe_subsample.npz")]
json.dump(res, open(f"{R}/layer_select.json", "w"), indent=1)
print("\nwrote layer_select.json")

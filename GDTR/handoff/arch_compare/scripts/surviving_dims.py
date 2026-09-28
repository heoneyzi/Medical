"""What survives the cascade?

Part 1 — two currencies.  Per layer, decode (a) the next nucleotide and (b) the
region label from the same rows.  If the post-cascade collapse is compression
onto the output, next-base accuracy should hold up exactly where region AUROC
falls apart.

Part 2 — the variant case.  Post-cascade dh keeps ~2 dimensions.  Are they the
model's own likelihood change?  Correlate the leading PCs of dh with the
zero-shot log-likelihood ratio.
"""
import gzip, json, zipfile, struct, numpy as np, pandas as pd, time
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"; R = f"{AC}/results"
CHROM = "chr22"; NSUB = 8000

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh); shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); n, m = struct.unpack("<HH", z.fp.read(30)[26:30])
    return np.memmap(path, dtype=dt, mode="r", offset=zi.header_offset + 30 + n + m + hdr, shape=shp)

def replay_7b(ch):
    scal = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}[ch]
    w = np.load(f"{R}/{scal}")["window"]
    meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{ch}_metadata.parquet")
    st = meta.set_index("window_idx")["start"].to_dict(); en = meta.set_index("window_idx")["end"].to_dict()
    labels = np.load(f"{ROOT}/paper/data_local/{ch}_position_labels.npy")
    use = [int(x) for x in w[np.concatenate(([True], w[1:] != w[:-1]))]]
    rng = np.random.default_rng(20260903); pos = []; labs = []
    for x in use:
        a, b = int(st[x]), int(en[x]); lab = np.zeros(b - a, np.uint8); gp = a + np.arange(b - a)
        ok = (gp >= 0) & (gp < len(labels)); lab[ok] = labels[gp[ok]]
        sel = np.isin(lab, [2, 5, 6])
        for code, frac in {1: 0.035, 0: 0.035}.items():
            idx = np.where(lab == code)[0]
            if len(idx): sel[rng.choice(idx, max(1, int(len(idx) * frac)), replace=False)] = True
        o = np.where(sel)[0]; pos.append(a + o); labs.append(lab[o])
    return np.concatenate(pos), np.concatenate(labs)

print("=== Part 1: next base vs region, same rows, Evo 2 7B chr22 ===", flush=True)
late = np.load(f"{R}/{CHROM}_late_fp32.npz"); lab_all = late["label"]; win_all = late["window"]
pos, lab_chk = replay_7b(CHROM)
assert np.array_equal(lab_chk, lab_all), "replay mismatch"
seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
nxt = np.array([S[p + 1] if p + 1 < len(S) else "N" for p in pos])
base = {c: i for i, c in enumerate("ACGT")}
okb = np.isin(nxt, list("ACGT"))
rng = np.random.default_rng(5)
idx = np.where(okb)[0]
idx = np.sort(rng.choice(idx, min(NSUB, len(idx)), replace=False))
yb = np.array([base[c] for c in nxt[idx]]); g = win_all[idx]
print(f"rows {len(idx)} | next-base class balance {np.bincount(yb) / len(yb)}", flush=True)

H16 = open_H(f"{R}/evo2_7b_probe_subsample.npz")
layers = list(range(0, 24, 2)) + list(range(24, 32)) + ["norm"]
def get(l, rows):
    if l == "norm": return late["hnorm"][rows].astype(np.float64)
    if l >= 24: return late[f"h{l}"][rows].astype(np.float64)
    return np.asarray(H16[l][rows], np.float64)

acc = []
for l in layers:
    X = np.nan_to_num(get(l, idx), nan=0.0, posinf=0.0, neginf=0.0)
    p = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(max_iter=3000))
    a = cross_val_score(p, X, yb, cv=GroupKFold(5), groups=g, scoring="accuracy", n_jobs=4).mean()
    acc.append(float(a)); print(f"  layer {l}: next-base accuracy {a:.4f}", flush=True)

reg = json.load(open(f"{R}/late_eval_reps.json"))["chr22"]
print("\n  layer      next-base   region AUROC (intergenic, fp32 reps)")
for l, a in zip(layers, acc):
    k = "hnorm" if l == "norm" else f"h{l}"
    rr = reg["intergenic"].get(k, [None])[0] if isinstance(l, str) or l >= 24 else None
    print(f"  {str(l):<9}{a:.4f}     {('%.3f' % rr) if rr else '-'}")

print("\n=== Part 2: are the surviving dh dimensions the likelihood change? ===", flush=True)
meta = np.load(f"{R}/clinvar17_meta.npz", allow_pickle=True)
llr = meta["llr"].astype(np.float64); gene = meta["gene"]; keys = [str(k) for k in meta["keys"]]
out2 = {}
for style in ("tok", "pool"):
    A = np.load(f"{R}/clinvar17_dh_{style}.npy", mmap_mode="r")
    rho1, r2 = [], []
    for j, k in enumerate(keys):
        X = np.nan_to_num(np.asarray(A[:, j, :], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Z = PCA(4, random_state=0).fit_transform(StandardScaler().fit_transform(X))
        rho1.append(float(abs(spearmanr(Z[:, 0], llr).correlation)))
        pred = cross_val_predict(RidgeCV(), Z, llr, cv=GroupKFold(5), groups=gene)
        r2.append(float(1 - ((llr - pred) ** 2).sum() / ((llr - llr.mean()) ** 2).sum()))
    out2[style] = {"keys": keys, "abs_rho_pc1_llr": rho1, "r2_llr_from_4pc": r2}
    print(f"\n  {style}: |rho(PC1, LLR)| " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{v:.2f}" for k, v in zip(keys, rho1)))
    print(f"  {style}: R2(LLR | 4 PCs)  " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{v:.2f}" for k, v in zip(keys, r2)))
json.dump({"next_base": {"layers": [str(l) for l in layers], "acc": acc}, "dh_vs_llr": out2},
          open(f"{R}/surviving_dims.json", "w"), indent=1)
with open(f"{R}/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\tsurviving_dims\tchr22\tnextbase_hnorm={acc[-1]:.4f}\t"
            f"pool_r2_llr_hnorm={out2['pool']['r2_llr_from_4pc'][-1]:.3f}\n")
print("\nwrote surviving_dims.json")

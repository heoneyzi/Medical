"""Downstream 1: does layer choice matter for predicting conservation (phyloP)?

Stored rows carry no genomic offsets, so offsets are rebuilt by replaying each
extraction's label-only selection (same seed, same draw order).  The rebuilt
labels must equal the stored labels exactly, or the script stops.

Targets: phyloP100way at each stored position.  Readout per layer:
StandardScaler -> PCA(256) -> Ridge, GroupKFold(5) by window, Spearman rho.
Subsets: all rows; non-coding only (intron+intergenic, removes the
exon-is-conserved shortcut); coding exon only.
"""
import numpy as np, pandas as pd, pyBigWig, zipfile, struct, json, gzip, sys
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"; R = f"{AC}/results"
BW = pyBigWig.open(f"{AC}/tracks/hg38.phyloP100way.bw")
NSUB = 6000

def windows(ch):
    scal = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}[ch]
    w = np.load(f"{R}/{scal}")["window"]
    meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{ch}_metadata.parquet")
    st = meta.set_index("window_idx")["start"].to_dict(); en = meta.set_index("window_idx")["end"].to_dict()
    labels = np.load(f"{ROOT}/paper/data_local/{ch}_position_labels.npy")
    use = [int(x) for x in w[np.concatenate(([True], w[1:] != w[:-1]))]]
    for x in use:
        a, b = int(st[x]), int(en[x]); lab = np.zeros(b - a, np.uint8); gp = a + np.arange(b - a)
        ok = (gp >= 0) & (gp < len(labels)); lab[ok] = labels[gp[ok]]
        yield x, a, lab

def replay_7b(ch):
    rng = np.random.default_rng(20260903); pos = []; labs = []
    for w, a, lab in windows(ch):
        sel = np.isin(lab, [2, 5, 6])
        for code, frac in {1: 0.035, 0: 0.035}.items():
            idx = np.where(lab == code)[0]
            if len(idx): sel[rng.choice(idx, max(1, int(len(idx) * frac)), replace=False)] = True
        o = np.where(sel)[0]; pos.append(a + o); labs.append(lab[o])
    return np.concatenate(pos), np.concatenate(labs)

def replay_40b(ch):
    rng = np.random.default_rng(20260915); pos = []; labs = []
    FRAC = {5: 1.0, 6: 1.0, 2: 0.10, 1: 0.035, 0: 0.035}
    for w, a, lab in windows(ch):
        sel = np.zeros(len(lab), bool)
        for code, f in FRAC.items():
            idx = np.where(lab == code)[0]
            if len(idx) == 0: continue
            k = len(idx) if f >= 1 else max(1, int(len(idx) * f))
            sel[rng.choice(idx, k, replace=False)] = True
        o = np.where(sel)[0]; pos.append(a + o); labs.append(lab[o])
    return np.concatenate(pos), np.concatenate(labs)

def phylop(ch, pos):
    out = np.full(len(pos), np.nan, np.float32)
    order = np.argsort(pos); p = pos[order]
    for i in range(0, len(p), 20000):
        chunk = p[i:i + 20000]; lo, hi = int(chunk.min()), int(chunk.max()) + 1
        v = np.array(BW.values(ch, lo, hi), np.float32); out[order[i:i + 20000]] = v[chunk - lo]
    return out

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh); shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); n, m = struct.unpack("<HH", z.fp.read(30)[26:30])
    return np.memmap(path, dtype=dt, mode="r", offset=zi.header_offset + 30 + n + m + hdr, shape=shp)

def evaluate(getter, layers, y, g, lab, tag):
    rng = np.random.default_rng(3); res = {}
    subsets = {"all": np.ones(len(y), bool), "noncoding": np.isin(lab, [0, 1]), "coding_exon": lab == 2}
    for sname, mask in subsets.items():
        idx = np.where(mask & np.isfinite(y))[0]
        if len(idx) > NSUB: idx = np.sort(rng.choice(idx, NSUB, replace=False))
        yy, gg = y[idx], g[idx]; curve = []
        for l in layers:
            X = np.nan_to_num(getter(l, idx), nan=0.0, posinf=0.0, neginf=0.0)
            pred = np.zeros(len(idx))
            for tr, te in GroupKFold(5).split(X, yy, gg):
                mdl = make_pipeline(StandardScaler(), PCA(256, random_state=0), Ridge(alpha=10.0)).fit(X[tr], yy[tr])
                pred[te] = mdl.predict(X[te])
            curve.append(float(spearmanr(pred, yy).correlation))
        res[sname] = {"n": int(len(idx)), "layers": [str(l) for l in layers], "rho": curve}
        best = int(np.argmax(curve[:-1]))
        print(f"  {tag} {sname:<12} n={len(idx):<5} best {layers[best]} rho={curve[best]:.3f} | "
              f"h_norm {curve[-1]:.3f} | gain {curve[best]-curve[-1]:+.3f}", flush=True)
    return res

out = {}
for ch in ("chr22", "chr17"):
    # ---- 7B ----
    late = np.load(f"{R}/{ch}_late_fp32.npz"); pos, lab = replay_7b(ch)
    assert len(pos) == len(late["label"]) and np.array_equal(lab, late["label"]), f"7B {ch} replay mismatch"
    print(f"7B {ch}: replay matches {len(pos)} rows", flush=True)
    y = phylop(ch, pos); g = late["window"]
    H16 = open_H(f"{R}/{'evo2_7b' if ch=='chr22' else 'chr17_evo2_7b'}_probe_subsample.npz")
    layers = list(range(0, 24, 2)) + list(range(24, 32)) + ["norm"]
    get7 = lambda l, idx: (np.asarray(H16[l][idx], np.float64) if l != "norm" and l < 24
                           else late["hnorm" if l == "norm" else f"h{l}"][idx].astype(np.float64))
    out[f"7B_{ch}"] = evaluate(get7, layers, y, g, lab, f"7B {ch}")
    # ---- 40B ----
    D = f"{R}/{ch}_evo2_40b"; meta = np.load(f"{D}/meta.npz"); pos, lab = replay_40b(ch)
    assert len(pos) == len(meta["label"]) and np.array_equal(lab, meta["label"]), f"40B {ch} replay mismatch"
    print(f"40B {ch}: replay matches {len(pos)} rows", flush=True)
    y = phylop(ch, pos); g = meta["window"]
    layers = list(range(0, 17, 2)) + list(range(17, 26)) + [30, 40, 49, "norm"]
    get40 = lambda l, idx: np.load(f"{D}/{'norm' if l=='norm' else f'b{l}'}.npy", mmap_mode="r")[idx].astype(np.float64)
    out[f"40B_{ch}"] = evaluate(get40, layers, y, g, lab, f"40B {ch}")
json.dump(out, open(f"{R}/phylop_layers.json", "w"), indent=1)
print("wrote phylop_layers.json")

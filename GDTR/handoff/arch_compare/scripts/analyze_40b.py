"""Evo 2 40B: does the 7B story replicate?

  1. norm cascade: adjacent-layer norm ratio, onset at T=10
  2. per-layer linear probe AUROC (same protocol as 7B: StandardScaler ->
     PCA(256) -> logistic, GroupKFold(5) by window)
  3. per-layer effective rank + position-specific share
  4. gain of the pre-onset band over h_norm
  5. attention blocks as local maxima (40B attention: 3,10,17,24,31,35,42,49)

usage: python analyze_40b.py <chr22|chr17>
"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

CHROM = sys.argv[1]
D = f"/path/to/TDiG/arch_compare/results/{CHROM}_evo2_40b"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
ATTN = [3, 10, 17, 24, 31, 35, 42, 49]
CAP, NSTAT = 4000, 2048

prof = json.load(open(f"{D}/profile.json"))
NB = prof["blocks"]; keys = [f"b{i}" for i in range(NB)] + ["norm"]
meta = np.load(f"{D}/meta.npz"); lab, win = meta["label"], meta["window"]
H = {k: np.load(f"{D}/{k}.npy", mmap_mode="r") for k in keys}

ratio = np.array(prof["ratio"])
onset = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
print(f"{'='*96}\n{CHROM}  Evo 2 40B  |  {NB} blocks  |  rows {len(lab)}\n{'='*96}")
print("adjacent norm ratio:", " ".join(f"{i+1}:{r:.3g}" for i, r in enumerate(ratio)))
pre = ratio[: (onset - 1) if onset else len(ratio)]
print(f"cascade onset (T=10): {onset} | max pre-onset ratio {pre.max():.2f} @ {int(pre.argmax())+1}"
      + (f" | onset ratio {ratio[onset-1]:.3g} | margin {ratio[onset-1]/pre.max():.1f}x" if onset else ""))
for T in (3, 5, 10, 20, 50, 100):
    on = next((i + 1 for i, x in enumerate(ratio) if x >= T), None)
    print(f"   T={T:<4} onset {on}")

# ---- effective rank and position-specific share (label-free) ----
rng = np.random.default_rng(0)
srow = np.sort(rng.choice(len(lab), min(NSTAT, len(lab)), replace=False))
geo = {}
for k in keys:
    X = np.nan_to_num(np.asarray(H[k][srow], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0); share = float(np.linalg.norm(X - mu, axis=1).mean() / max(np.linalg.norm(X, axis=1).mean(), 1e-30))
    sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ev = np.clip(np.linalg.eigvalsh(Z @ Z.T / len(Z)), 0, None)[::-1]
    p = ev / max(ev.sum(), 1e-30)
    geo[k] = {"eff_rank": float(np.exp(-(p * np.log(np.clip(p, 1e-16, None))).sum())), "share": share}
print("\neffective rank (standardized, n=%d rows):" % len(srow))
print("  " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{geo[k]['eff_rank']:.1f}" for k in keys))
print("position-specific share:")
print("  " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{geo[k]['share']:.2f}" for k in keys))

# ---- per-layer probe ----
auroc = {}
for code, nm in CTX.items():
    r2 = np.random.default_rng(20260916)
    rows = []
    for c in (code, 1):
        jj = np.where(lab == c)[0]
        if len(jj) > CAP: jj = r2.choice(jj, CAP, replace=False)
        rows.append(jj)
    rows = np.sort(np.concatenate(rows)); y = (lab[rows] == code).astype(int); g = win[rows]
    if y.sum() < 200 or (1 - y).sum() < 200:
        print(f"skip {nm}: pos {y.sum()} neg {(1-y).sum()}"); continue
    curve = []
    for k in keys:
        X = np.nan_to_num(np.asarray(H[k][rows], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        pipe = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(max_iter=3000))
        a = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=g, scoring="roc_auc", n_jobs=4).mean()
        curve.append(float(max(a, 1 - a)))
    auroc[nm] = curve
    c = np.array(curve[:NB]); hn = curve[NB]; b = int(c.argmax())
    line = f"\n{nm} (pos {y.sum()}, neg {(1-y).sum()}): best block {b} ({c[b]:.3f}) | h_norm {hn:.3f} | last block {c[-1]:.3f}"
    if onset:
        band = c[max(0, onset - 4):onset]; bb = max(0, onset - 4) + int(band.argmax())
        line += f" | pre-onset band best {bb} ({band.max():.3f}) | gain over h_norm {band.max()-hn:+.3f}"
    print(line)
    print("  " + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(c)))
    loc = [l for l in range(1, NB - 1) if c[l] >= c[l - 1] and c[l] >= c[l + 1]]
    at = [l for l in ATTN if 1 <= l <= NB - 2]
    hit = [l for l in at if l in loc]
    print(f"  local maxima {len(loc)}/{NB-2} | attention hits {len(hit)}/{len(at)} {hit}")

json.dump({"chrom": CHROM, "onset": onset, "ratio": ratio.tolist(), "geometry": geo, "auroc": auroc},
          open(f"{D}/analysis.json", "w"), indent=1)
print("\nwrote", f"{D}/analysis.json")

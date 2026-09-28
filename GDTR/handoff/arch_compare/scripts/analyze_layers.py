"""Generic layer analysis for a directory written by extract_layers.py or
extract_40b.py.  usage: python analyze_layers.py <dir> <attn_idxs csv>"""
import sys, json, os, time, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

D = sys.argv[1]; ATTN = [int(x) for x in sys.argv[2].split(",")]
AC = "/path/to/TDiG/arch_compare"
CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic", 3: "utr5", 4: "utr3"}
CAP = 4000
prof = json.load(open(f"{D}/profile.json")); NB = prof["blocks"]
import os as _os
HAS_NORM = _os.path.exists(f"{D}/norm.npy")
keys = [f"b{i}" for i in range(NB)] + (["norm"] if HAS_NORM else [])
FINAL = "norm" if HAS_NORM else f"b{NB-1}"   # what practitioners call "the embedding"
meta = np.load(f"{D}/meta.npz"); lab, win = meta["label"], meta["window"]
H = {k: np.load(f"{D}/{k}.npy", mmap_mode="r") for k in keys}
ratio = np.array(prof["ratio"])
onset_detected = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
ABSTAIN = onset_detected is None
onset = onset_detected if not ABSTAIN else NB
if ABSTAIN:
    print("  no cascade detected (max adjacent ratio %.2f): the rule ABSTAINS; "
          "the band below is shown for reference only" % ratio.max())
print(f"{D}\n  blocks {NB} rows {len(lab)} | onset {onset} | max pre-onset ratio "
      f"{ratio[:onset-1].max():.2f} | onset ratio {(ratio[onset-1] if onset_detected is not None else float('nan')):.3g}", flush=True)

rng = np.random.default_rng(0)
srow = np.sort(rng.choice(len(lab), min(2048, len(lab)), replace=False))
geo = {}
for k in keys:
    X = np.nan_to_num(np.asarray(H[k][srow], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; Z = (X - mu) / sd
    ev = np.clip(np.linalg.eigvalsh(Z @ Z.T / len(Z)), 0, None)[::-1]; p = ev / max(ev.sum(), 1e-30)
    geo[k] = float(np.exp(-(p * np.log(np.clip(p, 1e-16, None))).sum()))
print("  eff rank: " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{geo[k]:.0f}" for k in keys), flush=True)

out = {"onset": onset, "onset_detected": onset_detected, "abstained": ABSTAIN,
       "has_norm_state": HAS_NORM,
       "ratio": ratio.tolist(), "eff_rank": geo, "auroc": {}}
for code, nm in CTX.items():
    r2 = np.random.default_rng(20260916); rows = []
    for c in (code, 1):
        jj = np.where(lab == c)[0]
        if len(jj) > CAP: jj = r2.choice(jj, CAP, replace=False)
        rows.append(jj)
    rows = np.sort(np.concatenate(rows)); y = (lab[rows] == code).astype(int); g = win[rows]
    if y.sum() < 200 or (1 - y).sum() < 200:
        print(f"  skip {nm}: pos {y.sum()} neg {(1-y).sum()}"); continue
    curve = []
    for k in keys:
        X = np.nan_to_num(np.asarray(H[k][rows], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        a = cross_val_score(make_pipeline(StandardScaler(), PCA(256, random_state=0),
                                          LogisticRegression(max_iter=3000)),
                            X, y, cv=GroupKFold(5), groups=g, scoring="roc_auc", n_jobs=4).mean()
        curve.append(float(max(a, 1 - a)))
    out["auroc"][nm] = curve
    c = np.array(curve[:NB]); hn = curve[NB] if HAS_NORM else float(c[NB - 1])
    band = c[max(0, onset - 4):onset]; bb = max(0, onset - 4) + int(band.argmax())
    rule = c[max(0, onset - 4)]
    tag = "reference band" if ABSTAIN else f"rule onset-4 b{onset-4}"
    print(f"  {nm}: oracle b{int(c.argmax())}={c.max():.3f} | band best b{bb}={band.max():.3f} | "
          f"{tag}={rule:.3f} (regret {c.max()-rule:+.3f}) | final {hn:.3f} "
          f"(gain {band.max()-hn:+.3f})", flush=True)
json.dump(out, open(f"{D}/analysis.json", "w"), indent=1)
reg = []
for nm, curve in out["auroc"].items():
    c = np.array(curve[:NB]); reg.append(c.max() - c[max(0, onset - 4)])
with open(f"{AC}/results/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\tanalyze\t{os.path.basename(D)}\tonset={onset}\t"
            f"rule_regret_mean={np.mean(reg):.4f}\trule_regret_max={np.max(reg):.4f}\n")
print("wrote", f"{D}/analysis.json")

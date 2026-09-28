"""Why does a difference feature survive the cascade?

If the cascade adds a position-independent component, it cancels in alt - ref.
Test it: per layer, the effective rank and top-1 variance share of dh, next to
the same statistics for the raw states h (which collapse to rank ~2-5).
"""
import numpy as np, json, time
R = "/path/to/TDiG/arch_compare/results"
meta = np.load(f"{R}/clinvar17_meta.npz", allow_pickle=True)
keys = [str(k) for k in meta["keys"]]; NB = len(keys) - 1
prof = json.load(open(f"{R}/chr22_evo2_40b/profile.json")) if False else None

def stats(X):
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ev = np.clip(np.linalg.eigvalsh(Z @ Z.T / len(Z)), 0, None)[::-1]
    p = ev / max(ev.sum(), 1e-30)
    raw = X - X.mean(0)
    s = np.linalg.svd(raw, compute_uv=False) ** 2
    return (float(np.exp(-(p * np.log(np.clip(p, 1e-16, None))).sum())),
            float(s[0] / s.sum()), float(np.linalg.norm(X, axis=1).mean()))

out = {}
rng = np.random.default_rng(0)
for style in ("tok", "pool"):
    A = np.load(f"{R}/clinvar17_dh_{style}.npy", mmap_mode="r")
    idx = np.sort(rng.choice(A.shape[0], min(2048, A.shape[0]), replace=False))
    er, t1, nm = [], [], []
    for j in range(len(keys)):
        e, t, n = stats(A[idx, j, :])
        er.append(e); t1.append(t); nm.append(n)
    out[style] = {"eff_rank": er, "top1": t1, "norm": nm}
    print(f"\n{style}  dh effective rank by layer")
    print("  " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{e:.0f}" for k, e in zip(keys, er)))
    print(f"  top-1 share: " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{t:.2f}" for k, t in zip(keys, t1)))
    print(f"  mean ||dh||: " + " ".join(f"{k[1:] if k!='norm' else 'N'}:{n:.3g}" for k, n in zip(keys, nm)))
json.dump(out, open(f"{R}/dh_geometry.json", "w"), indent=1)
with open(f"{R}/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\tdh_geometry\tclinvar17\ttok_effrank_b30={out['tok']['eff_rank'][30]:.0f}\t"
            f"pool_effrank_b30={out['pool']['eff_rank'][30]:.0f}\n")
print("\nwrote dh_geometry.json")

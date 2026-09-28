"""Why does the rank collapse?  Test the claim h_t ~= c(t) * u + r_t.

If one branch adds an output far larger than the stream it joins, every position
ends up near the same direction u, differing mainly in a scalar c(t).  Then a
correlation matrix over positions is dominated by the variation of that scalar,
which forces effective rank toward one.  Measured per block:

  cos_to_u      mean |cos(h_t, u)| with u the normalised mean direction
  top1_share    fraction of centred variance in the first principal component
  rank1_resid   ||H - (H u) u^T||_F / ||H||_F  (uncentred: the scalar-multiple claim)
  corr_norm_proj  corr(||h_t||, h_t . u)
"""
import numpy as np, json, time
R = "/path/to/TDiG/arch_compare/results"
rng = np.random.default_rng(0)

def stats(X):
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0); u = mu / max(np.linalg.norm(mu), 1e-30)
    nrm = np.linalg.norm(X, axis=1)
    proj = X @ u
    cos = np.abs(proj) / np.clip(nrm, 1e-30, None)
    resid = X - np.outer(proj, u)
    Xc = X - mu
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return (float(cos.mean()), float(s[0] / s.sum()),
            float(np.linalg.norm(resid) / max(np.linalg.norm(X), 1e-30)),
            float(np.corrcoef(nrm, proj)[0, 1]))

out = {}
print("=== Evo 2 7B, chr22 ===")
print(f"  {'block':<8}{'cos_to_u':>10}{'top1_share':>12}{'rank1_resid':>13}{'corr(norm,proj)':>17}")
d = np.load(f"{R}/chr22_late_fp32.npz")
idx = np.sort(rng.choice(len(d["label"]), 4096, replace=False))
for k in [f"h{i}" for i in range(24, 32)] + ["hnorm"]:
    v = stats(d[k][idx]); out[f"7B_{k}"] = v
    print(f"  {k:<8}{v[0]:>10.4f}{v[1]:>12.4f}{v[2]:>13.4f}{v[3]:>17.4f}", flush=True)

print("\n=== Evo 2 40B, chr22 ===")
print(f"  {'block':<8}{'cos_to_u':>10}{'top1_share':>12}{'rank1_resid':>13}{'corr(norm,proj)':>17}")
Dir = f"{R}/chr22_evo2_40b"
meta = np.load(f"{Dir}/meta.npz"); idx2 = np.sort(rng.choice(len(meta["label"]), 4096, replace=False))
for k in ["b17", "b19", "b20", "b21", "b22", "b25", "b30", "b49", "norm"]:
    A = np.load(f"{Dir}/{k}.npy", mmap_mode="r")
    v = stats(A[idx2]); out[f"40B_{k}"] = v
    print(f"  {k:<8}{v[0]:>10.4f}{v[1]:>12.4f}{v[2]:>13.4f}{v[3]:>17.4f}", flush=True)

json.dump(out, open(f"{R}/scalar_dominance.json", "w"), indent=1)
with open(f"{R}/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\tscalar_dominance\tchr22\t7B_hnorm_cos={out['7B_hnorm'][0]:.3f}\t"
            f"40B_b30_cos={out['40B_b30'][0]:.3f}\n")
print("\nwrote scalar_dominance.json")

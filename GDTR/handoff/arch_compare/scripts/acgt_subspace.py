"""Hypothesis: the post-cascade collapse to rank ~4 is compression onto the
output subspace.  Logits over A,C,G,T are E_b . h_norm, so everything the model
needs for next-base prediction lives in span{E_A, E_C, E_G, E_T} (4 dims).

Measure the fraction of centred variance inside that span, against
  * 200 random 4-dim subspaces (chance level),
  * a pre-cascade layer from the same model (should be near chance).
"""
import numpy as np, torch, json

R = "/path/to/TDiG/arch_compare"
ACGT = [ord(c) for c in "ACGT"]
rng = np.random.default_rng(0)

def emb_rows(ckpt):
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
    keys = [k for k in sd if "embed" in k.lower()]
    print("  embedding-like keys:", {k: tuple(sd[k].shape) for k in keys})
    out = {}
    for k in keys:
        W = sd[k]
        if W.ndim == 2 and W.shape[0] >= 256:
            out[k] = W[ACGT].detach().float().numpy().astype(np.float64)
    if len(out) == 2:
        a, b = list(out.values()); print("  tied (identical rows):", bool(np.allclose(a, b)))
    return out

def frac_in_span(X, B):
    Xc = X - X.mean(0)
    Q, _ = np.linalg.qr(B.T)                    # orthonormal basis, D x 4
    tot = (Xc ** 2).sum()
    return float(((Xc @ Q) ** 2).sum() / tot)

def rand_frac(X, k, n=200):
    Xc = X - X.mean(0); tot = (Xc ** 2).sum(); D = X.shape[1]; v = []
    for _ in range(n):
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
        v.append(((Xc @ Q) ** 2).sum() / tot)
    return float(np.mean(v)), float(np.percentile(v, 99))

def top_pc_frac(X, k=4):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(s[:k].sum() / s.sum())

res = {}
cases = {
    "evo2_7b": dict(ckpt="/path/to/TDiG/arch_compare/hf_cache/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt",
                    layers={"h27 (pre-cascade)": lambda: np.load(f"{R}/results/chr22_late_fp32.npz")["h27"],
                            "h_norm": lambda: np.load(f"{R}/results/chr22_late_fp32.npz")["hnorm"]}),
    "evo2_40b": dict(ckpt=f"{R}/hf_cache/evo2_40b.pt",
                     layers={"b19 (pre-cascade)": lambda: np.load(f"{R}/results/chr22_evo2_40b/b19.npy", mmap_mode="r"),
                             "b30 (post-cascade)": lambda: np.load(f"{R}/results/chr22_evo2_40b/b30.npy", mmap_mode="r"),
                             "h_norm": lambda: np.load(f"{R}/results/chr22_evo2_40b/norm.npy", mmap_mode="r")}),
}
for model, c in cases.items():
    print(f"\n{'='*90}\n{model}\n{'='*90}")
    E = emb_rows(c["ckpt"])
    if not E: print("  no embedding matrix found"); continue
    B = list(E.values())[-1]
    res[model] = {}
    for name, get in c["layers"].items():
        A = get(); idx = np.sort(rng.choice(len(A), min(4096, len(A)), replace=False))
        X = np.nan_to_num(np.asarray(A[idx], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        f = frac_in_span(X, B); rm, r99 = rand_frac(X, 4); t4 = top_pc_frac(X, 4)
        res[model][name] = {"acgt_span": f, "random_mean": rm, "random_p99": r99, "top4_pc": t4}
        print(f"  {name:<22} variance in ACGT span {f:.4f} | random 4-d {rm:.5f} (p99 {r99:.5f}) "
              f"| best possible 4-d (top-4 PCs) {t4:.4f} | ACGT/top4 {f/max(t4,1e-12):.3f}")
json.dump(res, open(f"{R}/results/acgt_subspace.json", "w"), indent=1)

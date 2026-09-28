"""Do the layer-wise trajectory PROFILES differ by genomic region?

Not a classification question. For each region we take the mean trajectory —
cos(h_l, h_norm) and relative velocity v_l at every layer — and ask whether the
SHAPE differs from intron, layer by layer, with window-clustered CIs.

Contrast to keep in mind: the supervised probe is nearly flat across depth for
splice sites (0.90-0.96 from L2 on). If the geometry still varies by region and
by layer, decodability and geometry are dissociated.
"""
import numpy as np

R = "/path/to/TDiG/arch_compare/results"
d = np.load(f"{R}/evo2_7b_scalars.npz")
vel, cos = d["vel"].astype(np.float32), d["cos"].astype(np.float32)
lab, win = d["label"], d["window"]
CTX = {1: "intron", 2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor",
       0: "intergenic", 3: "utr5", 4: "utr3"}
PRE = 27
wu = np.unique(win)
rng = np.random.default_rng(20260903)
B = 600

def per_window_stats(X, code):
    """per window: n, sum, sumsq for each layer -> (W, L, 3)"""
    out = np.zeros((len(wu), X.shape[1], 3), dtype=np.float64)
    for i, w in enumerate(wu):
        m = (win == w) & (lab == code)
        if not m.any():
            continue
        x = X[m].astype(np.float64)
        out[i, :, 0] = len(x); out[i, :, 1] = x.sum(0); out[i, :, 2] = (x**2).sum(0)
    return out

def d_curve(A, Bs):
    n1, s1, q1 = A[..., 0], A[..., 1], A[..., 2]
    n2, s2, q2 = Bs[..., 0], Bs[..., 1], Bs[..., 2]
    N1, S1, Q1 = n1.sum(0), s1.sum(0), q1.sum(0)
    N2, S2, Q2 = n2.sum(0), s2.sum(0), q2.sum(0)
    ok = (N1 > 2) & (N2 > 2)
    m1, m2 = np.where(ok, S1/np.maximum(N1,1), np.nan), np.where(ok, S2/np.maximum(N2,1), np.nan)
    v1 = np.maximum((Q1 - N1*m1**2)/np.maximum(N1-1,1), 0)
    v2 = np.maximum((Q2 - N2*m2**2)/np.maximum(N2-1,1), 0)
    p = np.sqrt(((N1-1)*v1 + (N2-1)*v2)/np.maximum(N1+N2-2,1))
    return np.where(p > 0, (m1-m2)/p, np.nan)

for gname, X, LMAX in (("cos(h_l, h_norm)", cos, 30), ("relative velocity", vel, PRE)):
    Xs = X[:, :LMAX]
    base = per_window_stats(Xs, 1)
    print("\n" + "=" * 100)
    print(f"PER-LAYER effect size vs intron — {gname}")
    print("=" * 100)
    hdr = "".join(f"{l:>6}" for l in range(0, LMAX, 2))
    print(f"  {'region':<17}{'n':>8}  {hdr}")
    for code, nm in CTX.items():
        if nm == "intron" or (lab == code).sum() < 2000:
            continue
        A = per_window_stats(Xs, code)
        pt = d_curve(A, base)
        boot = np.empty((B, LMAX))
        for b in range(B):
            i = rng.integers(0, len(wu), len(wu))
            boot[b] = d_curve(A[i], base[i])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        sig = (lo * hi > 0)
        row = "".join(f"{pt[l]:>+6.2f}" for l in range(0, LMAX, 2))
        print(f"  {nm:<17}{int((lab==code).sum()):>8,}  {row}")
        best = int(np.nanargmax(np.abs(pt)))
        print(f"      peak |d| {abs(pt[best]):.3f} @L{best}"
              f"  [{lo[best]:+.3f},{hi[best]:+.3f}]"
              f"  | layers with CI excluding 0: {int(sig.sum())}/{LMAX}"
              f"  | sign changes across depth: {int((np.diff(np.sign(pt[~np.isnan(pt)])) != 0).sum())}")

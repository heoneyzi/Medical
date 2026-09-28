"""Per-layer AUROC vs intron — scale-free, so comparable ACROSS layers.

Cohen's d is not comparable across layers here: within-layer spread rises with
depth for cos (0.036 at L5 -> 0.205 at L20) and tracks block type for velocity
(hcs sd 0.069 vs hcm 1.468). AUROC is invariant to any monotone transform
within a layer, which removes both confounds.
"""
import numpy as np

R = "/path/to/TDiG/arch_compare/results"
d = np.load(f"{R}/evo2_7b_scalars.npz")
cos, vel = d["cos"].astype(np.float32), d["vel"].astype(np.float32)
lab, win = d["label"], d["window"]
ATTN=[3,10,17,24,31]; HCS=[0,4,7,11,14,18,21,25,28]
HCM=[1,5,8,12,15,19,22,26,29]; HCL=[2,6,9,13,16,20,23,27,30]
BT={}
for l in ATTN: BT[l]="attn"
for l in HCS: BT[l]="hcs"
for l in HCM: BT[l]="hcm"
for l in HCL: BT[l]="hcl"
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
LMAX=27; B=400
rng=np.random.default_rng(20260903)
wu=np.unique(win)

def auroc(x, y):
    """rank-based AUROC, y in {0,1}"""
    o=np.argsort(x, kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    # average ranks for ties
    xs=x[o]; i=0
    while i < len(xs):
        j=i
        while j+1 < len(xs) and xs[j+1]==xs[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    n1=y.sum(); n0=len(y)-n1
    if n1<2 or n0<2: return np.nan
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

for gname, X in (("cos(h_l, h_norm)", cos), ("relative velocity", vel)):
    Xs=X[:,:LMAX]
    print("\n"+"="*104)
    print(f"PER-LAYER AUROC vs intron — {gname}   (0.5 = 구별 안 됨, scale-free)")
    print("="*104)
    print(f"  {'region':<17}{'n':>8}  " + "".join(f"{l:>6}" for l in range(0,LMAX,2)))
    for code,nm in CTX.items():
        if (lab==code).sum()<2000: continue
        m=(lab==code)|(lab==1)
        y=(lab[m]==code).astype(int); Xm=Xs[m]; wm=win[m]
        a=np.array([auroc(Xm[:,l], y) for l in range(LMAX)])
        # window cluster bootstrap on the peak layer
        pk=int(np.nanargmax(np.abs(a-0.5)))
        bs=[]
        for _ in range(B):
            idx=rng.choice(wu, len(wu), replace=True)
            sel=np.concatenate([np.where(wm==w)[0] for w in idx])
            bs.append(auroc(Xm[sel,pk], y[sel]))
        lo,hi=np.nanpercentile(bs,[2.5,97.5])
        row="".join(f"{v:>6.3f}" for v in a[::2])
        print(f"  {nm:<17}{int((lab==code).sum()):>8,}  {row}")
        far=np.abs(a-0.5)
        top3=np.argsort(far)[::-1][:3]
        print(f"      peak {a[pk]:.3f} @L{pk}({BT[pk]}) [{lo:.3f},{hi:.3f}]"
              f" | top3 layers {[(int(t), BT[int(t)], round(float(a[t]),3)) for t in top3]}"
              f" | spread {far.max()-far.min():.3f}")

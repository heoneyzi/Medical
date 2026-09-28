"""Does the per-position TRAJECTORY VECTOR recover what min/max summaries lose?

Candidates, all leave-window-out, same positions, same cap:
  V, PRA            single scalars (min / max over layers) — current metrics
  best single layer  the one layer that separates best
  cos[L4]-cos[L8]    a two-layer contrast, chosen from the observed profile shape
  cos vector (27-d)  the whole cosine trajectory
  vel vector (27-d)  the whole velocity trajectory
  both (54-d)        concatenated
Reference: the 4096-d supervised probe on raw hidden states = 0.957 (splice_donor).
"""
import numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

R="/path/to/TDiG/arch_compare/results"
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
LMAX=27; CAP=4000
RNG=np.random.default_rng(20260903)
PROBE={"chr22":{"coding_exon":0.859,"splice_donor":0.957,"splice_acceptor":0.970,"intergenic":0.825}}

def run(chrom, f):
    d=np.load(f"{R}/{f}")
    cos=d["cos"].astype(np.float32)[:,:LMAX]; vel=d["vel"].astype(np.float32)[:,:LMAX]
    lab,win=d["label"],d["window"]
    mu,sd=vel.mean(0),vel.std(0).clip(1e-6)
    Vz=(vel-mu)/sd
    V=Vz.min(1); P=cos.max(1)
    print("\n"+"="*112); print(f"{chrom}"); print("="*112)
    print(f"  {'region':<17}{'V':>7}{'PRA':>7}{'best 1층':>10}{'L4-L8':>8}"
          f"{'cos 27d':>10}{'vel 27d':>10}{'both 54d':>10}{'probe 4096d':>13}")
    for code,nm in CTX.items():
        if (lab==code).sum()<2000: continue
        ii=[]
        for c in (code,1):
            jj=np.where(lab==c)[0]
            if len(jj)>CAP: jj=RNG.choice(jj,CAP,replace=False)
            ii.append(jj)
        sel=np.sort(np.concatenate(ii))
        y=(lab[sel]==code).astype(int); g=win[sel]
        def sc(X):
            X=np.asarray(X,dtype=np.float32)
            if X.ndim==1: X=X[:,None]
            p=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
            a=cross_val_score(p,X,y,cv=GroupKFold(5),groups=g,scoring="roc_auc").mean()
            return max(a,1-a)          # orientation-free
        best1=max(sc(cos[sel,l]) for l in range(LMAX))
        res=[sc(V[sel]), sc(P[sel]), best1, sc(cos[sel,4]-cos[sel,8]),
             sc(cos[sel]), sc(vel[sel]), sc(np.hstack([cos[sel],vel[sel]]))]
        pr=PROBE.get(chrom,{}).get(nm)
        prs=f"{pr:.3f}" if pr else "—"
        print(f"  {nm:<17}"+"".join(f"{v:>7.3f}" if i<2 else
              (f"{v:>10.3f}" if i in (2,4,5,6) else f"{v:>8.3f}")
              for i,v in enumerate(res))+f"{prs:>13}")
        if pr:
            print(f"      → 54d가 probe의 {100*res[6]/pr:.0f}% 수준 "
                  f"| 차원 {32*4096}→54 ({32*4096/54:.0f}배 압축)")

run("chr22","evo2_7b_scalars.npz")
run("chr17","chr17_evo2_7b_scalars.npz")

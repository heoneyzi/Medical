"""Per-block variance fraction inside span{E_A,E_C,E_G,E_T}.

acgt_subspace.py measured this at a handful of depths; Figure 1 needs the whole
profile, so this sweeps every block of an existing extraction directory.  CPU
only: the hidden states are already on disk, and we mmap and subsample them.
"""
import json, os, numpy as np, torch

R="/path/to/TDiG/arch_compare"
ACGT=[ord(c) for c in "ACGT"]
rng=np.random.default_rng(0)

def emb_rows(ckpt):
    sd=torch.load(ckpt,map_location="cpu",mmap=True,weights_only=False)
    if isinstance(sd,dict) and "model" in sd and isinstance(sd["model"],dict): sd=sd["model"]
    B=None
    for k in sd:
        W=sd[k]
        if "embed" in k.lower() and hasattr(W,"ndim") and W.ndim==2 and W.shape[0]>=256:
            B=W[ACGT].detach().float().numpy().astype(np.float64)
    del sd
    return B

def frac_in_span(X,B):
    Xc=X-X.mean(0); Q,_=np.linalg.qr(B.T)
    tot=(Xc**2).sum()
    return float(((Xc@Q)**2).sum()/max(tot,1e-30))

def rand_frac(X,k=4,n=50):
    Xc=X-X.mean(0); tot=(Xc**2).sum(); D=X.shape[1]; v=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.standard_normal((D,k)))
        v.append(((Xc@Q)**2).sum()/max(tot,1e-30))
    return float(np.mean(v))

def eff_rank(X):
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1.0; Z=(X-mu)/sd
    ev=np.clip(np.linalg.eigvalsh(Z@Z.T/len(Z)),0,None)[::-1]
    p=ev/max(ev.sum(),1e-30)
    return float(np.exp(-(p*np.log(np.clip(p,1e-16,None))).sum()))

CASES={
 "evo2_7b": dict(d=f"{R}/results/chr22_evo2_7b_utr",
   ckpt=f"{R}/hf_cache/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt"),
 "evo2_40b": dict(d=f"{R}/results/chr22_evo2_40b", ckpt=f"{R}/hf_cache/evo2_40b.pt"),
}
out={}
for name,c in CASES.items():
    d=c["d"]
    if not os.path.isdir(d): print(f"{name}: no dir {d}, skipped", flush=True); continue
    prof=json.load(open(f"{d}/profile.json")); NB=prof["blocks"]
    B=emb_rows(c["ckpt"])
    if B is None: print(f"{name}: no embedding matrix, skipped", flush=True); continue
    print(f"\n=== {name} | blocks {NB} | ACGT rows {B.shape} ===", flush=True)
    keys=[f"b{i}" for i in range(NB)]+(["norm"] if os.path.exists(f"{d}/norm.npy") else [])
    n=len(np.load(f"{d}/meta.npz")["label"])
    idx=np.sort(rng.choice(n,min(4096,n),replace=False))
    rec={}
    for k in keys:
        A=np.load(f"{d}/{k}.npy",mmap_mode="r")
        X=np.nan_to_num(np.asarray(A[idx],dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
        f=frac_in_span(X,B); ch=rand_frac(X); er=eff_rank(X)
        rec[k]={"acgt_frac":f,"chance":ch,"eff_rank":er,
                "mean_norm":float(np.linalg.norm(X,axis=1).mean())}
        print(f"  {k:>5}: acgt {f:.5f} | chance {ch:.5f} | eff_rank {er:8.1f}", flush=True)
    out[name]=rec
    json.dump(out,open(f"{R}/results/acgt_per_block.json","w"),indent=1)
print("\ndone", flush=True)

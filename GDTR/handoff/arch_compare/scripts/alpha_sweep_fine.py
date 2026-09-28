"""Magnitude-rescaling intervention on the block-28 MLP branch of Evo 2 7B.

The branch DIRECTION is preserved; only its magnitude is scaled by alpha.
Per alpha, on the same chr22 panel and the same readout protocol used
elsewhere, we measure: next-base accuracy, effective rank of h_norm, the
variance fraction of h_norm inside span{E_A,E_C,E_G,E_T}, and region AUROC at
h_norm.  Block 24 is upstream of the hook and is carried as a control: its
numbers must not move with alpha.
"""
import gzip, os, json
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; ALPHAS=[0.0,0.002,0.005,0.01,0.016,0.025,0.05,0.1]
KEEP_ALL={2,5,6}; SUB={1:0.035,0:0.035}
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
CAP=4000; SKIP=512

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

seq=[]
with gzip.open(f"{AC}/{CHROM}.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq)
meta=pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
st_of=meta.set_index("window_idx")["start"].to_dict()
en_of=meta.set_index("window_idx")["end"].to_dict()
labels_full=np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")
w_all=np.load(f"{AC}/results/evo2_7b_scalars.npz")["window"]
use=[int(w) for w in w_all[np.concatenate(([True], w_all[1:]!=w_all[:-1]))]]
print(f"windows {len(use)}", flush=True)

RNG=np.random.default_rng(20260903)
sel_of={}; L=[]; W=[]
for w in use:
    st,en=int(st_of[w]),int(en_of[w]); T=en-st; s=S[st:en].upper()
    if len(s)!=T: sel_of[w]=None; continue
    gp=st+np.arange(T); lab=np.zeros(T,dtype=np.uint8)
    ok=(gp>=0)&(gp<len(labels_full)); lab[ok]=labels_full[gp[ok]]
    sel=np.isin(lab,list(KEEP_ALL))
    for code,frac in SUB.items():
        idx=np.where(lab==code)[0]
        if len(idx): sel[RNG.choice(idx,max(1,int(len(idx)*frac)),replace=False)]=True
    sel_of[w]=sel
    if sel.sum(): L.append(lab[sel]); W.append(np.full(int(sel.sum()),w,dtype=np.int32))
lab_all=np.concatenate(L); win_all=np.concatenate(W)
print(f"selected rows {len(lab_all)}", flush=True)

r2=np.random.default_rng(20260916); keep=set()
for code in CTX:
    for c in (code,1):
        jj=np.where(lab_all==c)[0]
        if len(jj)>CAP: jj=r2.choice(jj,CAP,replace=False)
        keep.update(jj.tolist())
rows=np.sort(np.array(sorted(keep))); rowset=np.zeros(len(lab_all),bool); rowset[rows]=True
print(f"evaluation rows {len(rows)}", flush=True)
off=0; keep_in_win={}
for w in use:
    sel=sel_of.get(w)
    if sel is None or not sel.sum(): keep_in_win[w]=None; continue
    n=int(sel.sum()); keep_in_win[w]=rowset[off:off+n]; off+=n
assert off==len(lab_all), (off,len(lab_all))
lab_e=lab_all[rows]; win_e=win_all[rows]

CK=f"{AC}/hf_cache/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt"
sd=torch.load(CK,map_location="cpu",mmap=True,weights_only=False)
if isinstance(sd,dict) and "model" in sd and isinstance(sd["model"],dict): sd=sd["model"]
Bm=None
for k in sd:
    if "embed" in k.lower():
        Wt=sd[k]
        if Wt.ndim==2 and Wt.shape[0]>=256:
            Bm=Wt[[ord(c) for c in "ACGT"]].detach().float().numpy().astype(np.float64)
print("ACGT embedding rows:", None if Bm is None else Bm.shape, flush=True)
del sd

from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m=Evo2("evo2_7b"); net=m.model
print("model ready", flush=True)
ACGT_ID=torch.tensor([ord(c) for c in "ACGT"],device="cuda")

caught={}
def mk(name):
    def hook(mod,inp,out): caught[name]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return hook
ALPHA=[1.0]
def scale(mod,inp,out):
    o=out[0] if isinstance(out,tuple) else out
    return o*ALPHA[0]
hs=[net.blocks[24].register_forward_hook(mk("b24")),
    net.norm.register_forward_hook(mk("norm")),
    net.blocks[28].mlp.register_forward_hook(scale)]

def eff_rank(X):
    X=np.nan_to_num(np.asarray(X,np.float64),nan=0.,posinf=0.,neginf=0.)
    r=np.random.default_rng(0)
    if len(X)>2048: X=X[np.sort(r.choice(len(X),2048,replace=False))]
    mu=X.mean(0); sd_=X.std(0); sd_[sd_==0]=1.0; Z=(X-mu)/sd_
    ev=np.clip(np.linalg.eigvalsh(Z@Z.T/len(Z)),0,None)[::-1]
    p=ev/max(ev.sum(),1e-30)
    return float(np.exp(-(p*np.log(np.clip(p,1e-16,None))).sum()))

def frac_span(X,B):
    if B is None: return None
    X=np.nan_to_num(np.asarray(X,np.float64),nan=0.,posinf=0.,neginf=0.)
    Xc=X-X.mean(0); Q,_=np.linalg.qr(B.T)
    return float(((Xc@Q)**2).sum()/max((Xc**2).sum(),1e-30))

def auroc_at(X):
    res={}
    for code,nm in CTX.items():
        mk_=(lab_e==code)|(lab_e==1)
        y=(lab_e[mk_]==code).astype(int); g=win_e[mk_]
        if y.sum()<200 or (1-y).sum()<200: continue
        Xi=np.nan_to_num(np.asarray(X[mk_],np.float64),nan=0.,posinf=0.,neginf=0.)
        a=cross_val_score(make_pipeline(StandardScaler(),PCA(256,random_state=0),
                          LogisticRegression(max_iter=3000)),Xi,y,cv=GroupKFold(5),
                          groups=g,scoring="roc_auc",n_jobs=4).mean()
        res[nm]=float(max(a,1-a))
    return res

def run(a):
    ALPHA[0]=a; Hn=[]; H24=[]; corr=0; tot=0
    for w in use:
        sel=sel_of.get(w)
        if sel is None: continue
        st,en=int(st_of[w]),int(en_of[w]); s=S[st:en].upper()
        ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long).cuda()
        caught.clear()
        with torch.no_grad(): out=m(ids)
        lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
        i0=ids[0]; y=torch.full_like(i0[1:],-1)
        for k,c in enumerate(ACGT_ID): y[i0[1:]==c]=k
        ok=(y>=0); ok[:SKIP]=False
        if bool(ok.any()):
            corr+=int((lg[:-1][:,ACGT_ID][ok].argmax(1)==y[ok]).sum()); tot+=int(ok.sum())
        kw=keep_in_win.get(w)
        if sel.sum() and kw is not None and kw.any():
            Hn.append(caught["norm"][sel].cpu().numpy()[kw].astype(np.float32))
            H24.append(caught["b24"][sel].cpu().numpy()[kw].astype(np.float32))
        torch.cuda.empty_cache()
    return np.concatenate(Hn), np.concatenate(H24), corr/max(tot,1)

out={}
for a in ALPHAS:
    Hn,H24,acc=run(a)
    rec={"alpha":a,"next_base_acc":acc,
         "mean_norm_hnorm":float(np.linalg.norm(Hn,axis=1).mean()),
         "eff_rank_hnorm":eff_rank(Hn),"acgt_frac_hnorm":frac_span(Hn,Bm),
         "auroc_hnorm":auroc_at(Hn),"eff_rank_b24":eff_rank(H24),"auroc_b24":auroc_at(H24)}
    out[str(a)]=rec
    print(f"alpha={a}: nextbase {acc:.4f} | rank(hnorm) {rec['eff_rank_hnorm']:.1f} | "
          f"acgt {rec['acgt_frac_hnorm']} | auroc(hnorm) {rec['auroc_hnorm']} | "
          f"rank(b24) {rec['eff_rank_b24']:.1f}", flush=True)
    json.dump(out,open(f"{AC}/results/alpha_sweep_fine.json","w"),indent=1)
for h in hs: h.remove()
print("done", flush=True)

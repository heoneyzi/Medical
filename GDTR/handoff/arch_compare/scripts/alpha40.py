"""40B onset-branch ablation: the alpha = 0 point only.

The 7B sweep (Appendix B) showed that removing the branch that marks the onset
does not restore the rank of the final state, because later blocks carry their
own explosions.  This checks whether the same holds at 40B, whose onset sits at
block 21 of 50.  Block 21 is not in the attention list [3,10,17,24,31,35,42,49],
so its mixer is a Hyena convolution and the gated-MLP branch is the one that
explodes, as at 7B block 28.

Must run inside the NGC container: 40B needs FP8 through Transformer Engine,
and the conda env has no TE, which would silently fall back to bf16 and give
numbers not comparable with the 40B figures in the paper.
"""
import gzip, os, json
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; NAME="evo2_40b"; ONSET=21; ALPHAS=[1.0, 0.0]
KEEP_ALL={2,5,6}; SUB={1:0.035, 0:0.035}
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
r2=np.random.default_rng(20260916); keep=set()
for code in CTX:
    for c in (code,1):
        jj=np.where(lab_all==c)[0]
        if len(jj)>CAP: jj=r2.choice(jj,CAP,replace=False)
        keep.update(jj.tolist())
rows=np.sort(np.array(sorted(keep))); rowset=np.zeros(len(lab_all),bool); rowset[rows]=True
off=0; keep_in_win={}
for w in use:
    sel=sel_of.get(w)
    if sel is None or not sel.sum(): keep_in_win[w]=None; continue
    n=int(sel.sum()); keep_in_win[w]=rowset[off:off+n]; off+=n
lab_e=lab_all[rows]; win_e=win_all[rows]
print(f"evaluation rows {len(rows)}", flush=True)

CK=f"{AC}/hf_cache/evo2_40b.pt"
Bm=None
if os.path.exists(CK):
    sd=torch.load(CK,map_location="cpu",mmap=True,weights_only=False)
    if isinstance(sd,dict) and "model" in sd and isinstance(sd["model"],dict): sd=sd["model"]
    for k in sd:
        W_=sd[k]
        if "embed" in k.lower() and hasattr(W_,"ndim") and W_.ndim==2 and W_.shape[0]>=256:
            Bm=W_[[ord(c) for c in "ACGT"]].detach().float().numpy().astype(np.float64)
    del sd
print("ACGT rows:", None if Bm is None else Bm.shape, flush=True)

import evo2
from evo2.utils import CONFIG_MAP
from evo2 import Evo2
try:
    import transformer_engine
    print("Transformer Engine", transformer_engine.__version__, flush=True)
except Exception as e:
    raise SystemExit(f"ABORT: Transformer Engine missing ({e}); 40B would fall back to bf16 "
                     "and would not be comparable with the paper's 40B numbers.")
m=Evo2(NAME); net=m.model; NB=len(net.blocks)
print(f"model ready: {NB} blocks", flush=True)
assert ONSET < NB
ACGT_ID=torch.tensor([ord(c) for c in "ACGT"],device="cuda")

caught={}
def mk(n):
    def h(mod,inp,out): caught[n]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
ALPHA=[1.0]
def scale(mod,inp,out):
    o=out[0] if isinstance(out,tuple) else out
    return o*ALPHA[0]
hs=[net.norm.register_forward_hook(mk("norm")),
    net.blocks[ONSET].mlp.register_forward_hook(scale)]

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
    ALPHA[0]=a; Hn=[]; corr=0; tot=0
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
        torch.cuda.empty_cache()
    return np.concatenate(Hn), corr/max(tot,1)

out={}
for a in ALPHAS:
    Hn,acc=run(a)
    rec={"alpha":a,"next_base_acc":acc,"eff_rank_hnorm":eff_rank(Hn),
         "acgt_frac_hnorm":frac_span(Hn,Bm),"auroc_hnorm":auroc_at(Hn)}
    out[str(a)]=rec
    print(f"alpha={a}: nextbase {acc:.4f} | rank(hnorm) {rec['eff_rank_hnorm']:.1f} | "
          f"acgt {rec['acgt_frac_hnorm']} | auroc {rec['auroc_hnorm']}", flush=True)
    json.dump(out,open(f"{AC}/results/alpha40_ablation.json","w"),indent=1)
for h in hs: h.remove()
print("done", flush=True)

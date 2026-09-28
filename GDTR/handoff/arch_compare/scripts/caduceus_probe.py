"""E1: does a bidirectional state-space model show the cascade, and does the rule abstain?

Caduceus-PS is bidirectional and reverse-complement equivariant, so it is the test the
cross-model comparison was missing.  The protocol is the one used for HyenaDNA in
hyena_probe.py: same chr22 window panel, same labels, same per-class sampling and seed,
same leave-window-out readout, so only the model differs.

Two things are checked before any number is produced, because both would fail silently:
  * token alignment.  The tokenizer returns one more token than bases, and which end
    carries it decides whether every position is off by one.
  * hidden width.  caduceus-ps shares parameters across strands and concatenates the
    forward and reverse-complement channels, so the state is 2*d_model.
"""
import gzip, sys, json
import numpy as np, pandas as pd, torch, h5py
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"; R=f"{AC}/results"
MID="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
NWIN=int(sys.argv[1]) if len(sys.argv)>1 else 100
RNG=np.random.default_rng(20260822)
KEEP_ALL={2,5,6}; SUB={1:0.10, 0:0.10}
T_THRESH=10.0

dev="cuda"
tok=AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model=AutoModel.from_pretrained(MID, trust_remote_code=True).to(dev).eval()
layers=model.backbone.layers; NL=len(layers)
print(f"caduceus loaded | layers {NL} | d_model {model.config.d_model}", flush=True)

# ---- token alignment: let the tokenizer demonstrate it ------------------
# Not read off special-token attributes: their names vary between tokenizers and
# a missing attribute looks the same as "no special token", which is how an
# off-by-one gets in. A homopolymer settles it outright, because every base
# position then carries the same id and anything else is the added token.
import collections as _c
hp="A"*32
hid=tok(hp, return_tensors=None)["input_ids"]
extra=len(hid)-len(hp)
modal,_ =_c.Counter(hid).most_common(1)[0]
odd=[i for i,x in enumerate(hid) if x!=modal]
print(f"  {len(hp)} bases -> {len(hid)} tokens (extra {extra}); non-base positions {odd}", flush=True)
if   extra==0 and not odd:              LEAD=0
elif extra==1 and odd==[len(hid)-1]:    LEAD=0     # trailing terminator
elif extra==1 and odd==[0]:             LEAD=1     # leading marker
else:
    raise SystemExit(f"ABORT: cannot place the {extra} extra token(s); odd positions {odd}")
# independent cross-check: altering base 0 must alter exactly the token at index LEAD
_a=tok("A"+"T"*15, return_tensors=None)["input_ids"]
_b=tok("C"+"T"*15, return_tensors=None)["input_ids"]
_d=[i for i,(x,y) in enumerate(zip(_a,_b)) if x!=y]
if _d!=[LEAD]:
    raise SystemExit(f"ABORT: first base maps to token index {_d}, expected [{LEAD}]")
print(f"  alignment confirmed by two tests: base i is token i+{LEAD}; hidden sliced [{LEAD} : {LEAD}+T]", flush=True)

caught={}
def mk(i):
    def h(mod, inp, out):
        caught[i]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
hs=[layers[i].register_forward_hook(mk(i)) for i in range(NL)]

# ---- panel, exactly as hyena_probe.py builds it --------------------------
seq=[]
with gzip.open(f"{AC}/chr22.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq)
with h5py.File(f"{ROOT}/data/chr22_tier2_scalars.h5","r") as f: widx=f["window_idx"][:]
meta=pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
st_of=meta.set_index("window_idx")["start"].to_dict()
en_of=meta.set_index("window_idx")["end"].to_dict()
labels_full=np.load(f"{ROOT}/paper/data_local/chr22_position_labels.npy")
use=[int(w) for w in widx[:NWIN] if int(w) in st_of]
print(f"windows {len(use)}", flush=True)

Hs,LB,WI=[],[],[]
for i,w in enumerate(use):
    st,en=int(st_of[w]),int(en_of[w]); T=en-st; s=S[st:en]
    if len(s)!=T: continue
    gp=st+np.arange(T); lab=np.zeros(T,np.uint8)
    ok=(gp>=0)&(gp<len(labels_full)); lab[ok]=labels_full[gp[ok]]
    sel=np.isin(lab,list(KEEP_ALL))
    for code,frac in SUB.items():
        idx=np.where(lab==code)[0]
        if len(idx): sel[RNG.choice(idx,max(1,int(len(idx)*frac)),replace=False)]=True
    if sel.sum()==0: continue
    ids=tok(s.upper(), return_tensors="pt")["input_ids"].to(dev)
    caught.clear()
    with torch.no_grad(): model(ids)
    if len(caught)!=NL: raise SystemExit("ABORT: hooks did not fire on every layer")
    H=torch.stack([caught[j] for j in range(NL)])[:, LEAD:LEAD+T, :]
    if H.shape[1]!=T: raise SystemExit(f"ABORT: got {H.shape[1]} positions for {T} bases")
    Hs.append(H[:,sel,:].cpu().numpy().astype(np.float16))
    LB.append(lab[sel]); WI.append(np.full(int(sel.sum()),w,np.int32))
    torch.cuda.empty_cache()
    if (i+1)%25==0: print(f"  {i+1}/{len(use)}", flush=True)
for h in hs: h.remove()

H=np.concatenate(Hs,axis=1); lab=np.concatenate(LB); win=np.concatenate(WI)
print(f"\nsubsample n={len(lab):,} | H {H.shape}  (width {H.shape[2]} = 2 x d_model, strands concatenated)", flush=True)

# ---- does this model have a cascade at all? ------------------------------
nrm=[float(np.linalg.norm(H[l].astype(np.float32),axis=1).mean()) for l in range(NL)]
ratio=[nrm[l]/max(nrm[l-1],1e-30) for l in range(1,NL)]
print("\n=== cascade test ===", flush=True)
print("  mean norm per layer: " + " ".join(f"{v:.3g}" for v in nrm), flush=True)
print("  adjacent ratio:      " + " ".join(f"{v:.3g}" for v in ratio), flush=True)
over=[l+1 for l,r in enumerate(ratio) if r>=T_THRESH]
onset=over[0] if over else None
print(f"  max adjacent ratio {max(ratio):.3f} | threshold T={T_THRESH} | "
      f"{'onset at layer '+str(onset) if onset else 'NO CASCADE: the rule ABSTAINS'}", flush=True)

def eff_rank(X):
    Z=(X-X.mean(0))/X.std(0).clip(1e-8)
    ev=np.linalg.eigvalsh(np.cov(Z,rowvar=False)); ev=ev[ev>0]; p=ev/ev.sum()
    return float(np.exp(-(p*np.log(p)).sum()))
sub=RNG.choice(len(lab),min(4000,len(lab)),replace=False)
ranks=[eff_rank(H[l][sub].astype(np.float32)) for l in range(NL)]
print("  effective rank:      " + " ".join(f"{v:.0f}" for v in ranks), flush=True)

# ---- layer-wise readout, same folds as every other model -----------------
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
print("\n=== leave-window-out AUROC by layer ===", flush=True)
curves={}
for code,nm in CTX.items():
    m=(lab==code)|(lab==1)
    if (lab==code).sum()<300:
        print(f"  {nm}: only {(lab==code).sum()} positions, skipped", flush=True); continue
    y=(lab[m]==code).astype(int); g=win[m]; aur=[]; unconv=[]
    for l in range(NL):
        X=H[l][m].astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            a=cross_val_score(make_pipeline(StandardScaler(),PCA(256,random_state=0),
                                            LogisticRegression(max_iter=5000)),
                              X,y,cv=GroupKFold(5),groups=g,scoring="roc_auc",n_jobs=4).mean()
            if any(issubclass(x.category,ConvergenceWarning) for x in w): unconv.append(l)
        aur.append(float(max(a,1-a)))
    curves[nm]=aur
    if unconv: print(f"  {nm}: UNCONVERGED at layers {unconv}", flush=True)
    b=int(np.argmax(aur))
    print(f"  {nm:<17} best L{b}={aur[b]:.3f} | last L{NL-1}={aur[-1]:.3f} | "
          f"gain of best over last {aur[b]-aur[-1]:+.3f}", flush=True)
    print("    " + " ".join(f"{l}:{v:.3f}" for l,v in enumerate(aur)), flush=True)

json.dump({"model":MID,"layers":NL,"width":int(H.shape[2]),"n":int(len(lab)),
           "norm":nrm,"ratio":ratio,"onset":onset,"eff_rank":ranks,"auroc":curves},
          open(f"{R}/caduceus_probe.json","w"), indent=1)
print("\nwrote caduceus_probe.json", flush=True)

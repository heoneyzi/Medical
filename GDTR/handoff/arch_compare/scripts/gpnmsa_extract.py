"""Evo 2 features on the GPN-MSA chr17 benchmark panel.

The reviewer asked whether our layer choice is competitive with a task-specific
method.  Joining our own panel to theirs gave only 69 usable BRCA1 variants and
a 92%-positive ClinVar subset, so instead we extract Evo 2 features for THEIR
variants and score on THEIR labels, which makes the comparison head to head.

Negatives are gnomAD Common, not ClinVar Benign; that is the contrast the
GPN-MSA benchmark uses and the paper must say so.
"""
import os, gzip, json, numpy as np, pandas as pd, torch
ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
PARQ="/path/to/home/containers/gpnmsa_variants_and_preds.parquet"
CHROM="17"; WIN=6000; PER_CLASS=1042
METHODS=["GPN-MSA","CADD","phyloP","phyloP-241-mammals","phastCons-100-vertebrates",
         "NT-500m-human-ref","NT-500m-1000g","NT-2.5b-1000g","NT","ESM-1b","SpliceAI"]

seq=[]
with gzip.open(f"{AC}/chr{CHROM}.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq).upper(); print("chr17 length", len(S), flush=True)

df=pd.read_parquet(PARQ, columns=["chrom","pos","ref","alt","label","source","consequence","Gene"]+METHODS)
d=df[df["chrom"].astype(str).isin(["17","chr17"])].copy(); d["pos"]=d["pos"].astype(int)
mis=d[d["consequence"].astype(str).str.contains("missense", na=False)]
pos_=mis[(mis["source"]=="ClinVar")&(mis["label"]=="Pathogenic")]
neg_=mis[(mis["source"]=="gnomAD")&(mis["label"]=="Common")]
rng=np.random.default_rng(20260917)
def take(x,n):
    return x.iloc[rng.permutation(len(x))[:n]] if len(x)>n else x
n=min(len(pos_),len(neg_),PER_CLASS)
panel=pd.concat([take(pos_,n).assign(y=1), take(neg_,n).assign(y=0)], ignore_index=True)
print(f"panel: {len(panel)} ({int(panel.y.sum())} pathogenic / {int((1-panel.y).sum())} common)", flush=True)

keep=[]
for r in panel.itertuples():
    p0=int(r.pos)-1
    if p0<WIN//2 or p0+WIN//2>=len(S): continue
    if len(str(r.ref))!=1 or len(str(r.alt))!=1: continue
    if S[p0]!=str(r.ref): continue                       # reference must match
    keep.append((p0,str(r.ref),str(r.alt),int(r.y),str(r.Gene),int(r.pos)))
print(f"usable after reference check: {len(keep)} of {len(panel)}", flush=True)

import evo2
from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m=Evo2("evo2_7b"); net=m.model; NB=len(net.blocks); D=4096
print(f"model ready: {NB} blocks", flush=True)
caught={}
def mk(nm):
    def h(mod,inp,out): caught[nm]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
hs=[net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(NB)]
hs.append(net.norm.register_forward_hook(mk("norm")))
keys=[f"b{i}" for i in range(NB)]+["norm"]

N=len(keep)
TOK=np.lib.format.open_memmap(f"{AC}/results/gpnmsa17_dh_tok.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
POOL=np.lib.format.open_memmap(f"{AC}/results/gpnmsa17_dh_pool.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
y=np.zeros(N,np.int8); genes=[]; posl=np.zeros(N,np.int64); llr=np.zeros(N,np.float32)
ACGT=[ord(c) for c in "ACGT"]
def fwd(s):
    ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long,device="cuda")
    caught.clear()
    with torch.no_grad(): out=m(ids)
    lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
    return {k:caught[k] for k in keys}, lg

for i,(p0,ref,alt,lab,gene,pos1) in enumerate(keep):
    a=p0-WIN//2; s=S[a:a+WIN]; c=WIN//2
    assert s[c]==ref
    Cr,lgr=fwd(s); Ca,_=fwd(s[:c]+alt+s[c+1:])
    for j,k in enumerate(keys):
        TOK[i,j]=(Ca[k][c]-Cr[k][c]).cpu().numpy()
        POOL[i,j]=(Ca[k].mean(0)-Cr[k].mean(0)).cpu().numpy()
    lp=torch.log_softmax(lgr[c-1][ACGT],dim=0)
    llr[i]=float(lp[ACGT.index(ord(alt))]-lp[ACGT.index(ord(ref))])
    y[i]=lab; genes.append(gene); posl[i]=pos1
    torch.cuda.empty_cache()
    if (i+1)%100==0: print(f"  {i+1}/{N}", flush=True)
for h in hs: h.remove()
np.savez(f"{AC}/results/gpnmsa17_meta.npz",y=y,gene=np.array(genes),pos=posl,llr=llr,keys=np.array(keys))
panel.to_parquet(f"{AC}/results/gpnmsa17_panel.parquet")
print("saved", N, "variants", flush=True)

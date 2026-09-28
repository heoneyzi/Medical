"""BRCA1 saturation genome editing by layer (Evo 2 7B).

Same machinery as clinvar_extract.py, driven by the Findlay SGE score set
(MaveDB urn:mavedb:00000097-0-2) instead of ClinVar.  For each variant we
forward the reference 6 kb window and the alt window and keep, for every layer,
  dh_tok  = h_alt - h_ref at the variant position
  dh_pool = mean_t h_alt - mean_t h_ref

BRCA1 is one gene, so the leave-genes-out folds used for ClinVar are not
available.  We store the exon index instead, so evaluation can hold out whole
exons and neighbouring variants cannot share a window across a fold boundary.
"""
import os, gzip, numpy as np, torch
ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="17"; WIN=6000

EXON_S=[43044294,43047642,43049120,43051062,43057051,43063332,43063873,43067607,
        43070927,43074330,43076487,43082403,43090943,43091434,43095845,43097243,
        43099774,43104121,43104867,43106455,43115725,43124016,43125270]
EXON_E=[43045802,43047703,43049194,43051117,43057135,43063373,43063951,43067695,
        43071238,43074521,43076614,43082575,43091032,43094860,43095922,43097289,
        43099880,43104261,43104956,43106533,43115779,43124115,43125364]

seq=[]
with gzip.open(f"{AC}/chr{CHROM}.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq).upper(); print("chr17 length", len(S), flush=True)

var=np.load(f"{AC}/results/brca1_sge_variants.npy")
print("variants from MaveDB:", len(var), flush=True)

def exon_of(p0):
    for i,(s,e) in enumerate(zip(EXON_S,EXON_E)):
        if s <= p0 < e: return i
    return -1

keep=[]
for rec in var:
    pos=int(rec["pos"])-1                       # converter wrote 1-based
    ref=str(rec["ref"]); alt=str(rec["alt"]); sc=float(rec["score"])
    if pos<WIN//2 or pos+WIN//2>=len(S): continue
    if S[pos]!=ref: continue                    # reference must match
    ex=exon_of(pos)
    if ex<0: continue
    keep.append((pos,ref,alt,sc,ex))
print(f"usable variants {len(keep)} | exons {len(set(k[4] for k in keep))}", flush=True)

import evo2
from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m=Evo2("evo2_7b"); net=m.model; NB=len(net.blocks); D=4096
print(f"model ready: {NB} blocks", flush=True)
caught={}
def mk(n):
    def h(mod,inp,out):
        caught[n]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
hs=[net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(NB)]
hs.append(net.norm.register_forward_hook(mk("norm")))
keys=[f"b{i}" for i in range(NB)]+["norm"]

N=len(keep)
TOK=np.lib.format.open_memmap(f"{AC}/results/brca1_dh_tok.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
POOL=np.lib.format.open_memmap(f"{AC}/results/brca1_dh_pool.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
score=np.zeros(N,np.float32); exon=np.zeros(N,np.int16); posl=np.zeros(N,np.int64); llr=np.zeros(N,np.float32)
ACGT=[ord(c) for c in "ACGT"]
def fwd(s):
    ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long,device="cuda")
    caught.clear()
    with torch.no_grad(): out=m(ids)
    lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
    return {k:caught[k] for k in keys}, lg

for i,(pos,ref,alt,sc,ex) in enumerate(keep):
    a=pos-WIN//2; s=S[a:a+WIN]; c=WIN//2
    assert s[c]==ref
    Cr,lgr=fwd(s)
    Ca,_ =fwd(s[:c]+alt+s[c+1:])
    for j,k in enumerate(keys):
        TOK[i,j]=(Ca[k][c]-Cr[k][c]).cpu().numpy()
        POOL[i,j]=(Ca[k].mean(0)-Cr[k].mean(0)).cpu().numpy()
    lp=torch.log_softmax(lgr[c-1][ACGT],dim=0)
    llr[i]=float(lp[ACGT.index(ord(alt))]-lp[ACGT.index(ord(ref))])
    score[i]=sc; exon[i]=ex; posl[i]=pos
    torch.cuda.empty_cache()
    if (i+1)%100==0: print(f"  {i+1}/{N}", flush=True)
for h in hs: h.remove()
np.savez(f"{AC}/results/brca1_meta.npz",score=score,exon=exon,pos=posl,llr=llr,keys=np.array(keys))
print("saved", N, "variants", flush=True)

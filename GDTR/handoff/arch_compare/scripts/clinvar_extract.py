"""Downstream 2: ClinVar pathogenicity by layer (Evo 2 7B).

Missense only, so region identity cannot stand in for pathogenicity.  For each
variant: forward the reference 6 kb window and the alt window, and keep, for
every layer, both feature styles in the literature:
  dh_tok  = h_alt - h_ref at the variant position
  dh_pool = mean_t h_alt - mean_t h_ref   (window mean-pool)
Gene symbol is kept so evaluation can hold out whole genes.
"""
import os, gzip, json, numpy as np, torch
ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="17"; WIN=6000; PER_CLASS=1500; PER_GENE=40
P={"Pathogenic","Likely_pathogenic","Pathogenic/Likely_pathogenic"}
B={"Benign","Likely_benign","Benign/Likely_benign"}
OK_REV={"criteria_provided,_single_submitter","criteria_provided,_multiple_submitters,_no_conflicts",
        "reviewed_by_expert_panel","practice_guideline"}

seq=[]
with gzip.open(f"{AC}/chr{CHROM}.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq).upper(); print("chr17 length", len(S), flush=True)

rows=[]
with gzip.open(f"{AC}/tracks/clinvar_GRCh38.vcf.gz","rt") as f:
    for line in f:
        if line[0]=="#" or not line.startswith(CHROM+"\t"): continue
        p=line.rstrip("\n").split("\t")
        if len(p[3])!=1 or len(p[4])!=1 or p[3] not in "ACGT" or p[4] not in "ACGT": continue
        info=dict(kv.split("=",1) for kv in p[7].split(";") if "=" in kv)
        sig=info.get("CLNSIG",""); y=1 if sig in P else 0 if sig in B else None
        if y is None or info.get("CLNREVSTAT","") not in OK_REV: continue
        if "missense_variant" not in info.get("MC",""): continue
        pos=int(p[1])-1
        if pos<WIN//2 or pos+WIN//2>=len(S) or S[pos]!=p[3]: continue     # reference must match
        gene=info.get("GENEINFO","?").split(":")[0].split("|")[0]
        rows.append((pos,p[3],p[4],y,gene))
print("candidate missense variants:", len(rows), flush=True)

rng=np.random.default_rng(20260916); keep=[]
for y in (1,0):
    sub=[r for r in rows if r[3]==y]
    byg={}
    for r in sub: byg.setdefault(r[4],[]).append(r)
    pool=[]
    for g,v in byg.items():
        v=[v[i] for i in rng.permutation(len(v))[:PER_GENE]]; pool+=v
    pool=[pool[i] for i in rng.permutation(len(pool))[:PER_CLASS]]
    keep+=pool
    print(f"  class {y}: {len(sub)} -> {len(pool)} after per-gene cap, genes {len(byg)}", flush=True)
rng.shuffle(keep)

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
TOK=np.lib.format.open_memmap(f"{AC}/results/clinvar17_dh_tok.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
POOL=np.lib.format.open_memmap(f"{AC}/results/clinvar17_dh_pool.npy",mode="w+",dtype=np.float32,shape=(N,NB+1,D))
y=np.zeros(N,np.int8); genes=[]; posl=np.zeros(N,np.int64); llr=np.zeros(N,np.float32)
ACGT=[ord(c) for c in "ACGT"]
def fwd(s):
    ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long,device="cuda")
    caught.clear()
    with torch.no_grad(): out=m(ids)
    lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
    return {k:caught[k] for k in keys}, lg, ids[0]

for i,(pos,ref,alt,lab,gene) in enumerate(keep):
    a=pos-WIN//2; s=S[a:a+WIN]; c=WIN//2
    assert s[c]==ref
    Cr,lgr,ids=fwd(s)
    Ca,_,_=fwd(s[:c]+alt+s[c+1:])
    for j,k in enumerate(keys):
        TOK[i,j]=(Ca[k][c]-Cr[k][c]).cpu().numpy()
        POOL[i,j]=(Ca[k].mean(0)-Cr[k].mean(0)).cpu().numpy()
    lp=torch.log_softmax(lgr[c-1][ACGT],dim=0)                    # zero-shot baseline
    llr[i]=float(lp[ACGT.index(ord(alt))]-lp[ACGT.index(ord(ref))])
    y[i]=lab; genes.append(gene); posl[i]=pos
    torch.cuda.empty_cache()
    if (i+1)%100==0: print(f"  {i+1}/{N}", flush=True)
for h in hs: h.remove()
np.savez(f"{AC}/results/clinvar17_meta.npz",y=y,gene=np.array(genes),pos=posl,llr=llr,keys=np.array(keys))
print("saved", N, "variants |", int(y.sum()), "pathogenic", flush=True)

"""Evo 2 per-position features on BEND gene_finding, chr17/19/22.

BEND ships the 9-class labels as gene_finding.hdf5, one variable-length array
per BED row, aligned position-by-position with (end - start).  That lets us
evaluate on BEND's own labels and its own train/valid/test split rather than
re-deriving annotations, which we could not have matched.

Storage forces subsampling: 1,282 transcripts x ~7.6 kb x 33 layers would be
~138 GB.  We keep every position of the four rare classes and cap the common
ones, which also gives the probe a workable class balance.
"""
import os, gzip, json, collections
import numpy as np, h5py, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
BED=f"{AC}/bend/gene_finding.bed"; H5=f"{AC}/bend/gene_finding.hdf5"
OURS=("chr17","chr19","chr22")
RARE={1,3,5,7}; CAP_COMMON=3000          # per class, across all transcripts
OUT=f"{AC}/results/bend_gf"

def load_fa(c):
    for p in (f"{AC}/{c}.fa.gz", f"{AC}/chr{c.replace('chr','')}.fa.gz"):
        if os.path.exists(p):
            s=[]
            with gzip.open(p,"rt") as fh:
                for line in fh:
                    if not line.startswith(">"): s.append(line.strip())
            return "".join(s).upper()
    return None

S={c:load_fa(c) for c in OURS}
for c,v in S.items():
    if v is None: raise SystemExit(f"ABORT: no FASTA for {c}")
    print(f"  {c}: {len(v):,} bp", flush=True)

rows=[l.rstrip("\n").split("\t") for l in open(BED)]
hdr=rows[0]; ix={k:i for i,k in enumerate(hdr)}
bed=[r for r in rows[1:] if len(r)==len(hdr)]
keep=[(i,r) for i,r in enumerate(bed) if r[ix["chromosome"]] in OURS]
print(f"transcripts on {OURS}: {len(keep)}", flush=True)

h5=h5py.File(H5,"r"); LAB=h5["labels"]

# ---- sanity: splice dinucleotides, using BEND's own per-class convention -----
# Verified empirically on this share: classes 1/3 are forward-strand donor/acceptor
# and 5/7 are reverse-strand, and the dinucleotide sits at a different offset in
# each case.  Checking only the forward seq[p:p+2] finds GT for class 1 and never
# finds AG, which is a property of the check and not of the coordinates.
COMP=str.maketrans("ACGT","TGCA")
def rc(x): return x.translate(COMP)[::-1]
# class -> (offset, expected dinucleotide, reverse-complement first?)
CONV={1:(0,"GT",False), 3:(-1,"AG",False), 5:(-1,"GT",True), 7:(0,"AG",True)}
ok=collections.Counter(); tot=collections.Counter()
for i,r in keep[:200]:
    v=np.asarray(LAB[i]).astype(int); c=r[ix["chromosome"]]; a=int(r[ix["start"]])
    seq=S[c][a:a+len(v)]
    for cls,(off,want,rev) in CONV.items():
        for p in np.where(v==cls)[0][:8]:
            j=p+off
            if j<0 or j+2>len(seq): continue
            d=seq[j:j+2]
            if rev: d=rc(d)
            tot[cls]+=1; ok[cls]+= (d==want)
lines=[f"class {c}: {ok[c]}/{tot[c]} = {want}" for c,(o,want,rv) in CONV.items()]
print("sanity (per-class convention): " + " | ".join(lines), flush=True)
worst=min((ok[c]/tot[c]) for c in CONV if tot[c])
print(f"  worst class agreement {100*worst:.1f}%", flush=True)
if worst < 0.90:
    raise SystemExit("ABORT: splice dinucleotides disagree with the known convention; coordinates are wrong.")

# ---- choose positions from labels alone, before any forward pass ----------
rng=np.random.default_rng(20260917)
per_class=collections.defaultdict(list)
for i,r in keep:
    v=np.asarray(LAB[i]).astype(int)
    for cls in np.unique(v):
        idxs=np.where(v==cls)[0]
        per_class[int(cls)].append((i,idxs))
plan=collections.defaultdict(list)
for cls,chunks in per_class.items():
    allp=[(i,int(p)) for i,idxs in chunks for p in idxs]
    if cls not in RARE and len(allp)>CAP_COMMON:
        sel=rng.choice(len(allp),CAP_COMMON,replace=False)
        allp=[allp[k] for k in sel]
    for i,p in allp: plan[i].append((p,cls))
nsel=sum(len(v) for v in plan.values())
print(f"selected positions {nsel:,} over {len(plan)} transcripts", flush=True)
print("  per class:", {c:sum(1 for i in plan for p,cc in plan[i] if cc==c) for c in sorted(per_class)}, flush=True)

import evo2
from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m=Evo2("evo2_7b"); net=m.model; NB=len(net.blocks); D=4096
print(f"model ready: {NB} blocks", flush=True)
caught={}
def mk(n):
    def h(mod,inp,out): caught[n]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
hs=[net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(NB)]
hs.append(net.norm.register_forward_hook(mk("norm")))
keys=[f"b{i}" for i in range(NB)]+["norm"]

os.makedirs(OUT, exist_ok=True)
X=np.lib.format.open_memmap(f"{OUT}/H.npy",mode="w+",dtype=np.float32,shape=(nsel,NB+1,D))
y=np.zeros(nsel,np.int8); split=np.zeros(nsel,"U5"); tid=np.zeros(nsel,"U24"); chrom=np.zeros(nsel,"U6")
w=0
for k,(i,r) in enumerate(keep):
    if i not in plan: continue
    c=r[ix["chromosome"]]; a=int(r[ix["start"]]); v=np.asarray(LAB[i]).astype(int)
    seq=S[c][a:a+len(v)]
    if len(seq)!=len(v): continue
    ids=torch.tensor([m.tokenizer.tokenize(seq)],dtype=torch.long).cuda()
    caught.clear()
    with torch.no_grad(): net(ids)
    for p,cls in plan[i]:
        if p>=caught["norm"].shape[0]: continue
        for j,kk in enumerate(keys): X[w,j]=caught[kk][p].cpu().numpy()
        y[w]=cls; split[w]=r[ix["split"]]; tid[w]=r[ix["transcript_id"]]; chrom[w]=c; w+=1
    torch.cuda.empty_cache()
    if (k+1)%100==0: print(f"  {k+1}/{len(keep)} transcripts, {w:,} rows", flush=True)
for h in hs: h.remove()
np.savez(f"{OUT}/meta.npz",y=y[:w],split=split[:w],tid=tid[:w],chrom=chrom[:w],keys=np.array(keys))
json.dump({"blocks":NB,"rows":int(w)},open(f"{OUT}/profile.json","w"))
print("saved rows", w, flush=True)

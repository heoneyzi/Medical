"""Block-28 ablations on the full 100-window chr22 panel.

cause_probe.py ran these on three windows, which reads about 0.13 high against
the ten-window and hundred-window panels used elsewhere in the paper.  This
re-measures every ablation on the same panel as alpha_sweep.py so the whole
section quotes one panel.  Only next-base accuracy is needed, so no hidden
states are collected.
"""
import gzip, os, json
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; SKIP=512

seq=[]
with gzip.open(f"{AC}/{CHROM}.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq)
meta=pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
st_of=meta.set_index("window_idx")["start"].to_dict()
en_of=meta.set_index("window_idx")["end"].to_dict()
w_all=np.load(f"{AC}/results/evo2_7b_scalars.npz")["window"]
use=[int(w) for w in w_all[np.concatenate(([True], w_all[1:]!=w_all[:-1]))]]
print(f"windows {len(use)}", flush=True)

from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m=Evo2("evo2_7b"); net=m.model
print("model ready", flush=True)
ACGT_ID=torch.tensor([ord(c) for c in "ACGT"],device="cuda")

# top-8 energy channels of the block-28 MLP, from the first panel window
grab={}
h=net.blocks[28].mlp.register_forward_hook(
    lambda mo,i,o: grab.__setitem__("o",(o[0] if isinstance(o,tuple) else o).detach()[0].float()))
st,en=int(st_of[use[0]]),int(en_of[use[0]])
with torch.no_grad(): net(torch.tensor([m.tokenizer.tokenize(S[st:en].upper())],dtype=torch.long).cuda())
h.remove()
TOP=torch.argsort((grab["o"].double()**2).mean(0),descending=True)[:8]
print("top-8 channels:", TOP.tolist(), flush=True)

def zero(mo,i,o): return torch.zeros_like(o[0] if isinstance(o,tuple) else o)
def zchan(mo,i,o):
    o=(o[0] if isinstance(o,tuple) else o).clone(); o[...,TOP]=0; return o

CONFIG={"none":None,
        "mlp28":  ("mlp", zero),
        "mix28":  ("mix", zero),
        "chan28": ("mlp", zchan)}

def run(cfg):
    hs=[]
    if cfg is not None:
        which,fn=cfg
        mod = net.blocks[28].mlp if which=="mlp" else net.blocks[28].out_filter_dense
        hs.append(mod.register_forward_hook(fn))
    corr=0; tot=0
    for w in use:
        st,en=int(st_of[w]),int(en_of[w]); s=S[st:en].upper()
        ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long).cuda()
        with torch.no_grad(): out=m(ids)
        lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
        i0=ids[0]; y=torch.full_like(i0[1:],-1)
        for k,c in enumerate(ACGT_ID): y[i0[1:]==c]=k
        ok=(y>=0); ok[:SKIP]=False
        if bool(ok.any()):
            corr+=int((lg[:-1][:,ACGT_ID][ok].argmax(1)==y[ok]).sum()); tot+=int(ok.sum())
        torch.cuda.empty_cache()
    for x in hs: x.remove()
    return corr/max(tot,1)

res={}
for name,cfg in CONFIG.items():
    res[name]=run(cfg)
    print(f"{name:>8}: next-base {res[name]:.4f}  (positions {name})", flush=True)
    json.dump(res,open(f"{AC}/results/ablate100.json","w"),indent=1)
print("done", flush=True)

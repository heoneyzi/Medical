"""40B writer x re-encoder factorial: are blocks 21 and 23 doing different things?

The 7B result this mirrors: removing the block-30 mixer leaves the state at blocks 29
and 30 intact but rotates the final state almost orthogonal to its native direction,
while removing the block-28 gated-MLP destroys the summary at block 29. One writes,
the other re-encodes. If blocks 21 and 23 at 40B were the same operation at different
strengths, a 2x2 factorial would show their effects simply adding.

Geometry is measured against the unablated run on the same positions, so both the loss
and the direction of the state are compared like for like.
"""
import gzip, os, json, time
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; NAME="evo2_40b"; SKIP=512; NPOS=200
WRITER=(21,"mlp"); REENC=(23,"out_filter_dense")      # the 7B g28 x m30 analogue
TAPS=[22,23,34]                                        # plus the final norm
OUT=f"{AC}/results/fact40.json"

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
use=[w for w in use if w in st_of]
print(f"windows {len(use)}", flush=True)

import evo2
from evo2 import Evo2
try:
    import transformer_engine
    print("Transformer Engine", transformer_engine.__version__, flush=True)
except Exception as e:
    raise SystemExit(f"ABORT: Transformer Engine missing ({e})")
m=Evo2(NAME); net=m.model; NB=len(net.blocks)
print(f"model ready: {NB} blocks", flush=True)
for b,attr in (WRITER, REENC):
    assert hasattr(net.blocks[b], attr), f"block {b} lacks {attr}"
print(f"writer {WRITER}, re-encoder {REENC}: modules present", flush=True)

ACGT_ID=torch.tensor([ord(c) for c in "ACGT"], device="cuda")
def zero_hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    return torch.zeros_like(o)

caught={}
def mk(n):
    def h(mod, inp, out):
        caught[n]=(out[0] if isinstance(out,tuple) else out).detach()[0].float()
    return h
taps=[net.blocks[b].register_forward_hook(mk(f"b{b}")) for b in TAPS]
taps.append(net.norm.register_forward_hook(mk("norm")))
KEYS=[f"b{b}" for b in TAPS]+["norm"]

rng=np.random.default_rng(7)
def run(ablate):
    hs=[]
    for b,attr in ablate:
        hs.append(getattr(net.blocks[b], attr).register_forward_hook(zero_hook))
    nll=[]; states={k:[] for k in KEYS}
    for w in use:
        st,en=int(st_of[w]),int(en_of[w]); s=S[st:en].upper()
        ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long).cuda()
        caught.clear()
        with torch.no_grad(): out=m(ids)
        lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
        i0=ids[0]; y=torch.full_like(i0[1:], -1)
        for k,c in enumerate(ACGT_ID): y[i0[1:]==c]=k
        ok=(y>=0); ok[:SKIP]=False
        nll.append(float(torch.nn.functional.cross_entropy(
            lg[:-1][:,ACGT_ID][ok], y[ok], reduction="mean")) if bool(ok.any()) else np.nan)
        T=caught[KEYS[0]].shape[0]
        idx=np.sort(rng.choice(np.arange(SKIP, T), min(NPOS, T-SKIP), replace=False))
        for k in KEYS: states[k].append(caught[k][idx].cpu().numpy().astype(np.float32))
        torch.cuda.empty_cache()
    for h in hs: h.remove()
    return np.array(nll), {k: np.concatenate(v) for k,v in states.items()}

def cos_to(a, b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    num=(a*b).sum(1); den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return float(np.mean(num/np.clip(den,1e-30,None)))

CONDS=[("native", []), ("writer_off",[WRITER]), ("reenc_off",[REENC]),
       ("both_off",[WRITER,REENC])]
res={}; t0=time.time()
nat_nll=None; nat_st=None
for name, ab in CONDS:
    nll, st = run(ab)
    if name=="native":
        nat_nll, nat_st = nll, st
        print(f"  {name:11s} NLL {np.nanmean(nll):.4f}   ({time.time()-t0:.0f}s)", flush=True)
        res[name]={"nll":float(np.nanmean(nll))}
        continue
    row={"nll":float(np.nanmean(nll)), "delta_nll":float(np.nanmean(nll-nat_nll)),
         "cos": {k: cos_to(st[k], nat_st[k]) for k in KEYS}}
    res[name]=row
    print(f"  {name:11s} NLL {row['nll']:.4f}  dNLL {row['delta_nll']:+.4f}   "
          + "  ".join(f"cos({k})={row['cos'][k]:+.4f}" for k in KEYS)
          + f"   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUT,"w"), indent=1)

if all(k in res for k in ("writer_off","reenc_off","both_off")):
    add = res["writer_off"]["delta_nll"] + res["reenc_off"]["delta_nll"]
    print(f"\nadditivity check: writer {res['writer_off']['delta_nll']:+.4f} + "
          f"re-encoder {res['reenc_off']['delta_nll']:+.4f} = {add:+.4f}   "
          f"vs both together {res['both_off']['delta_nll']:+.4f}   "
          f"interaction {res['both_off']['delta_nll']-add:+.4f}", flush=True)
json.dump(res, open(OUT,"w"), indent=1)
print("\nwrote", OUT, flush=True)

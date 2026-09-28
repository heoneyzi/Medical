"""40B radial dose: is the magnitude of a late update causally inert, as at 7B?

At 7B the dose curve saturates: scaling the block-30 mixer by 0.25x, 0.5x, 2x or 4x
moves the loss by exactly zero, because RMSNorm divides the radius away once an update
dominates its host. If the same holds at 40B, then norm is a detector for locating the
event and not the variable that controls the output, which is the paper's Claim about
direction over magnitude, tested at the second scale.

Only the branches that the causal atlas showed to matter are swept: block 21 both
branches and block 23's mixer. Block 23's gated-MLP is dead and block 34 has no output
effect, so neither can show a dose response.
"""
import gzip, os, json, time
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; NAME="evo2_40b"; SKIP=512
SWEEP=[(21,"mlp"), (21,"out_filter_dense"), (23,"out_filter_dense")]
ALPHAS=[0.25, 0.5, 2.0, 4.0]          # alpha = 0 and 1 already measured by atlas40
OUT=f"{AC}/results/radial40.json"

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
m=Evo2(NAME); net=m.model
for b,attr in SWEEP: assert hasattr(net.blocks[b], attr), f"block {b} lacks {attr}"
print("sweep targets present:", SWEEP, flush=True)

ACGT_ID=torch.tensor([ord(c) for c in "ACGT"], device="cuda")
SCALE=[1.0]
def scale_hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    return o * SCALE[0]

def nll_mean(hook_target=None, a=1.0):
    h=None
    if hook_target is not None:
        b, attr = hook_target; SCALE[0]=a
        h=getattr(net.blocks[b], attr).register_forward_hook(scale_hook)
    vals=[]
    for w in use:
        st,en=int(st_of[w]),int(en_of[w]); s=S[st:en].upper()
        ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long).cuda()
        with torch.no_grad(): out=m(ids)
        lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
        i0=ids[0]; y=torch.full_like(i0[1:], -1)
        for k,c in enumerate(ACGT_ID): y[i0[1:]==c]=k
        ok=(y>=0); ok[:SKIP]=False
        if bool(ok.any()):
            vals.append(float(torch.nn.functional.cross_entropy(
                lg[:-1][:,ACGT_ID][ok], y[ok], reduction="mean")))
        torch.cuda.empty_cache()
    if h is not None: h.remove()
    return float(np.mean(vals))

t0=time.time()
base=nll_mean()
print(f"  baseline NLL {base:.4f}   ({time.time()-t0:.0f}s)", flush=True)
res={"baseline": base, "sweep": {}}
for b,attr in SWEEP:
    key=f"b{b}.{'mlp' if attr=='mlp' else 'mix'}"
    res["sweep"][key]={}
    for a in ALPHAS:
        v=nll_mean((b,attr), a)
        res["sweep"][key][str(a)]={"nll":v, "delta":v-base}
        print(f"  {key:10s} alpha={a:<5} NLL {v:.4f}  dNLL {v-base:+.4f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT,"w"), indent=1)

print("\n=== dose response, or the absence of one ===", flush=True)
for key, d in res["sweep"].items():
    row="  ".join(f"{a}x:{d[a]['delta']:+.4f}" for a in map(str, ALPHAS))
    flat=max(abs(d[a]["delta"]) for a in map(str, ALPHAS))
    print(f"  {key:10s} {row}    largest |dNLL| {flat:.4f}", flush=True)
json.dump(res, open(OUT,"w"), indent=1)
print("\nwrote", OUT, flush=True)

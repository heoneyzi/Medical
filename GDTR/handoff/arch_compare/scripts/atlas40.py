"""40B restricted causal atlas: does the checkpoint-only staging hold up causally?

The 7B work localises a writer, an amplifier candidate and a re-encoder. A rule read
from checkpoint weights alone predicts the same roles at 40B, but that prediction has
only ever been checked against our own published onset, never against an intervention.
This job supplies the intervention.

Scope is deliberately four blocks, not fifty. Past block 23 the 40B residual stream is
numerically saturated: blocks 24-33 leave it bitwise unchanged and blocks 35-49 change
it by a relative 1e-19, so ablating them cannot move anything. Only blocks 21, 22, 23
and 34 write measurably. Each is paired with the previous block of the same operator
class as a homologous control, which separates a learned block-specific computation
from a depth or operator-class effect.

Effects are read at the logits, not in the stored state, because the state stops
responding past block 23 while the output does not.
"""
import gzip, os, json, time
import numpy as np, pandas as pd, torch

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE","1")
CHROM="chr22"; NAME="evo2_40b"; SKIP=512
TARGETS=[21,22,23,34]; CONTROLS=[18,19,20,30]      # same operator class, one cycle earlier
BRANCHES=[("mlp","mlp"), ("mix","out_filter_dense")]
OUT=f"{AC}/results/atlas40.json"; OUTNPZ=f"{AC}/results/atlas40_per_window.npz"

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
    raise SystemExit(f"ABORT: Transformer Engine missing ({e}); 40B would fall back to bf16.")
m=Evo2(NAME); net=m.model; NB=len(net.blocks)
print(f"model ready: {NB} blocks", flush=True)

# fail in seconds, not hours, if the module names are wrong
for b in TARGETS+CONTROLS:
    assert b < NB, f"block {b} out of range"
    for _, attr in BRANCHES:
        assert hasattr(net.blocks[b], attr), f"block {b} has no attribute {attr}"
print("all target modules present:", TARGETS+CONTROLS, [a for _,a in BRANCHES], flush=True)

ACGT_ID=torch.tensor([ord(c) for c in "ACGT"], device="cuda")
def zero_hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    return torch.zeros_like(o)

def nll_per_window():
    """mean next-base NLL on ACGT positions past SKIP, one value per window."""
    vals=[]
    for w in use:
        st,en=int(st_of[w]),int(en_of[w]); s=S[st:en].upper()
        ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long).cuda()
        with torch.no_grad(): out=m(ids)
        lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
        i0=ids[0]; y=torch.full_like(i0[1:], -1)
        for k,c in enumerate(ACGT_ID): y[i0[1:]==c]=k
        ok=(y>=0); ok[:SKIP]=False
        if not bool(ok.any()): vals.append(np.nan); continue
        logits4=lg[:-1][:,ACGT_ID][ok]
        nll=torch.nn.functional.cross_entropy(logits4, y[ok], reduction="mean")
        vals.append(float(nll)); torch.cuda.empty_cache()
    return np.array(vals, dtype=np.float64)

conditions=[("baseline", None, None)]
for b in TARGETS+CONTROLS:
    for tag, attr in BRANCHES:
        conditions.append((f"b{b}.{tag}", b, attr))
print(f"{len(conditions)} conditions", flush=True)

res={}; per_win={}
t0=time.time()
for i,(name, blk, attr) in enumerate(conditions):
    h=None
    if blk is not None:
        h=getattr(net.blocks[blk], attr).register_forward_hook(zero_hook)
    v=nll_per_window()
    if h is not None: h.remove()
    per_win[name]=v
    mean=float(np.nanmean(v))
    if name=="baseline":
        base=v; base_mean=mean
        print(f"  [{i+1}/{len(conditions)}] {name:12s} NLL {mean:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    else:
        d=v-base; dm=float(np.nanmean(d))
        role="target " if blk in TARGETS else "control"
        res[name]={"block":blk,"branch":attr,"role":role.strip(),
                   "nll":mean,"delta_nll":dm,
                   "delta_sd_over_windows":float(np.nanstd(d,ddof=1))}
        print(f"  [{i+1}/{len(conditions)}] {name:12s} {role} NLL {mean:.4f}  "
              f"dNLL {dm:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"baseline_nll":base_mean,"conditions":res}, open(OUT,"w"), indent=1)
    np.savez(OUTNPZ, windows=np.array(use), **per_win)

print("\n=== targets vs their homologous controls ===", flush=True)
for t,c in zip(TARGETS, CONTROLS):
    for tag,_ in BRANCHES:
        kt, kc = f"b{t}.{tag}", f"b{c}.{tag}"
        if kt in res and kc in res:
            print(f"  {tag:4s}  b{t} {res[kt]['delta_nll']:+.4f}   vs  b{c} {res[kc]['delta_nll']:+.4f}"
                  f"   difference {res[kt]['delta_nll']-res[kc]['delta_nll']:+.4f}", flush=True)
print("\nwrote", OUT, "and", OUTNPZ, flush=True)

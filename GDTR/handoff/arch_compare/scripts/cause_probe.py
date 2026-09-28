"""Where does the cascade come from?  Decompose the residual update of blocks
25-31 into the mixer branch and the MLP branch, test input dependence, read the
relevant weights, and ablate.

Block algebra (vortex):  hyena  u -> u + out_filter_dense(filter(proj_norm(u)))
                                  -> + mlp(post_norm(.))
                         attn   u -> u + mha(pre_norm(u)) -> + mlp(post_norm(.))
"""
import os, gzip, json, numpy as np, pandas as pd, torch
ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
import evo2
from evo2.utils import CONFIG_MAP
pkg=os.path.dirname(evo2.__file__); CONFIG_MAP["evo2_7b"]="configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
BLOCKS=list(range(25,32)); ATTN={3,10,17,24,31}; SKIP=512
m=Evo2("evo2_7b"); net=m.model; print("model ready", flush=True)

seq=[]
with gzip.open(f"{AC}/chr22.fa.gz","rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S="".join(seq)
meta=pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
w=np.load(f"{AC}/results/evo2_7b_scalars.npz")["window"]
use=[int(x) for x in w[np.concatenate(([True],w[1:]!=w[:-1]))]][:3]
st=meta.set_index("window_idx")["start"].to_dict(); en=meta.set_index("window_idx")["end"].to_dict()
real=[S[int(st[x]):int(en[x])].upper() for x in use]
rng=np.random.default_rng(0)
INPUTS={"real":real[0],
        "random":"".join(rng.choice(list("ACGT"),len(real[0]))),
        "polyA":"A"*len(real[0]),
        "shuffled":"".join(rng.permutation(list(real[0])))}

caught={}
def mk(tag):
    def h(mod,inp,out):
        o=out[0] if isinstance(out,tuple) else out
        caught[tag]=o.detach()[0].float()
    return h
def mk_pre(tag):
    def h(mod,inp):
        caught[tag]=(inp[0] if isinstance(inp,tuple) else inp).detach()[0].float()
    return h
hs=[]
for b in BLOCKS:
    blk=net.blocks[b]
    hs.append(blk.register_forward_pre_hook(mk_pre(f"in{b}")))
    hs.append(blk.register_forward_hook(mk(f"out{b}")))
    hs.append(blk.mlp.register_forward_hook(mk(f"mlp{b}")))
    hs.append((blk.inner_mha_cls if b in ATTN else blk.out_filter_dense).register_forward_hook(mk(f"mix{b}")))

def run(s):
    ids=torch.tensor([m.tokenizer.tokenize(s)],dtype=torch.long,device="cuda")
    caught.clear()
    with torch.no_grad(): out=m(ids)
    lg=(out[0][0] if isinstance(out[0],(tuple,list)) else out[0])[0].float()
    return {k:v.clone() for k,v in caught.items()}, lg, ids[0]

ACGT=torch.tensor([ord(c) for c in "ACGT"],device="cuda")
def nextbase(lg,ids):
    y=torch.full_like(ids[1:],-1)
    for k,c in enumerate(ACGT): y[ids[1:]==c]=k
    ok=(y>=0); ok[:SKIP]=False
    return float((lg[:-1][:,ACGT][ok].argmax(1)==y[ok]).float().mean())

print("\n=== branch decomposition (mean L2 over positions) ===")
print(f"  {'input':<10}{'blk':>4}{'type':>6}{'||u||':>11}{'||mixer||':>11}{'||mlp||':>11}{'||out||':>11}{'resid err':>11}")
res={}
for name,s in INPUTS.items():
    C,lg,ids=run(s)
    for b in BLOCKS:
        u,mx,mp,o=C[f"in{b}"],C[f"mix{b}"],C[f"mlp{b}"],C[f"out{b}"]
        err=float((o-(u+mx+mp)).norm(dim=-1).mean()/o.norm(dim=-1).mean())
        row=[float(u.norm(dim=-1).mean()),float(mx.norm(dim=-1).mean()),float(mp.norm(dim=-1).mean()),float(o.norm(dim=-1).mean()),err]
        res[f"{name}_b{b}"]=row
        print(f"  {name:<10}{b:>4}{'attn' if b in ATTN else 'hyena':>6}"+"".join(f"{v:>11.4g}" for v in row), flush=True)
    if name=="real": base_acc=nextbase(lg,ids)
print(f"\n  baseline next-base accuracy (3 windows skipped): {base_acc:.4f}")

print("\n=== which channels dominate the block-28 MLP output, per input ===")
tops={}
for name,s in INPUTS.items():
    C,_,_=run(s)
    e=(C["mlp28"].double()**2).mean(0); t=torch.argsort(e,descending=True)[:8]
    tops[name]=set(t.tolist())
    print(f"  {name:<10}top8 {t.tolist()} | top1 share {float(e[t[0]]/e.sum()):.3f}")
print("  overlap with real:", {k:len(v&tops['real']) for k,v in tops.items()})

print("\n=== weights of block 28 ===")
blk=net.blocks[28]
l3=blk.mlp.l3.weight.detach().float()            # (4096, 11264) output projection
rown=l3.norm(dim=1); post=blk.post_norm.scale.detach().float().abs()
top=sorted(tops["real"])
print(f"  mlp.l3 row norm: dominant {rown[top].mean():.4g} vs others {rown.mean():.4g} "
      f"(ratio {float(rown[top].mean()/rown.mean()):.1f}x, max row {float(rown.max()):.4g})")
print(f"  post_norm.scale: dominant {post[top].mean():.4g} vs others {post.mean():.4g}")
for b in (27,28,29):
    w=net.blocks[b].mlp.l3.weight.detach().float().norm(dim=1)
    sc=net.blocks[b].post_norm.scale.detach().float().abs()
    print(f"  block {b}: mlp.l3 row norm mean {w.mean():.4g} max {w.max():.4g} | post_norm.scale mean {sc.mean():.4g} max {sc.max():.4g}")

print("\n=== ablations on the real window ===")
def ablate(kind):
    hh=[]
    if kind=="mlp28": hh.append(net.blocks[28].mlp.register_forward_hook(lambda mo,i,o: torch.zeros_like(o[0] if isinstance(o,tuple) else o)))
    if kind=="mix28": hh.append(net.blocks[28].out_filter_dense.register_forward_hook(lambda mo,i,o: torch.zeros_like(o)))
    if kind=="chan28":
        idx=torch.tensor(top,device="cuda")
        def z(mo,i,o):
            o=o.clone(); o[...,idx]=0; return o
        hh.append(net.blocks[28].mlp.register_forward_hook(z))
    C,lg,ids=run(INPUTS["real"])
    for h in hh: h.remove()
    return C,nextbase(lg,ids)
for kind in ("none","mlp28","mix28","chan28"):
    C,acc=(run(INPUTS["real"])[0],base_acc) if kind=="none" else ablate(kind)
    norms=[float(C[f"out{b}"].norm(dim=-1).mean()) for b in BLOCKS]
    print(f"  {kind:<8} next-base {acc:.4f} | ||h|| b25-31 " + " ".join(f"{n:.3g}" for n in norms), flush=True)
for h in hs: h.remove()
json.dump(res, open(f"{AC}/results/cause_probe.json","w"), indent=1)

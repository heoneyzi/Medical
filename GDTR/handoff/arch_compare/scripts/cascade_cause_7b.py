"""What produces the norm cascade in Evo 2 7B (block 28)?

1. Branch decomposition, blocks 25-31: block output y = u + mixer + mlp.
   Checks the identity numerically, then reports which branch carries the jump.
2. Input dependence: real chr22 windows vs uniform random DNA vs shuffled real
   vs poly-A vs (ACGT)n repeats.  If the same channels explode on random DNA,
   the cascade is a property of the weights, not of genomic content.
3. Weights at the dominant channels: post_norm / pre_norm scale, MLP output
   row norms (mlp.l3), mixer output row norms and bias (out_filter_dense).
4. Ablations on real windows: zero block-28 MLP branch, zero block-28 mixer
   branch, zero only the dominant channels of block-28 MLP output.  Report the
   norm profile and next-base accuracy for each.
"""
import os, gzip, json, numpy as np, pandas as pd, torch

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2("evo2_7b"); net = m.model; B = net.blocks
DOM_PRIOR = [715, 3181, 1489, 2656, 2390, 3676, 2765, 31]      # L28 top channels found earlier
BLK = list(range(25, 32)); T = 6000; rng = np.random.default_rng(0)

seq = []
with gzip.open(f"{AC}/chr22.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
st = meta.set_index("window_idx")["start"].to_dict()
w = np.load(f"{AC}/results/evo2_7b_scalars.npz")["window"]
use = [int(x) for x in w[np.concatenate(([True], w[1:] != w[:-1]))]][:5]
real = [S[int(st[x]):int(st[x]) + T] for x in use]
inputs = {"real": real,
          "shuffled_real": ["".join(rng.permutation(list(s))) for s in real[:3]],
          "random_uniform": ["".join(rng.choice(list("ACGT"), T)) for _ in range(3)],
          "polyA": ["A" * T], "ACGT_repeat": ["ACGT" * (T // 4)]}

def mixer_mod(i): return B[i].inner_mha_cls if hasattr(B[i], "inner_mha_cls") else B[i].out_filter_dense
stats = {}
def reset(): stats.clear()
def acc(name, t):
    t = (t[0] if isinstance(t, tuple) else t).detach()[0].float()
    d = stats.setdefault(name, {"n": 0, "norm": 0.0, "chan": torch.zeros(t.shape[-1], device=t.device), "vals": None})
    d["norm"] += t.norm(dim=-1).sum().item(); d["n"] += t.shape[0]; d["chan"] += t.abs().sum(0)
    d["last"] = t
hooks = []
for i in BLK:
    hooks.append(B[i].register_forward_pre_hook(lambda mod, inp, i=i: acc(f"u{i}", inp[0])))
    hooks.append(B[i].register_forward_hook(lambda mod, inp, out, i=i: acc(f"y{i}", out)))
    hooks.append(mixer_mod(i).register_forward_hook(lambda mod, inp, out, i=i: acc(f"mix{i}", out)))
    hooks.append(B[i].mlp.register_forward_hook(lambda mod, inp, out, i=i: acc(f"mlp{i}", out)))

ACGT = torch.tensor([ord(c) for c in "ACGT"], device="cuda")
def run(s):
    ids = torch.tensor([m.tokenizer.tokenize(s)], dtype=torch.long, device="cuda")
    with torch.no_grad(): out = net(ids)
    lg = out[0] if isinstance(out, tuple) else out
    lg = lg[0].float() if lg.dim() == 3 else lg.float()
    x = lg[:-1][:, ACGT]; tgt = ids[0, 1:]; y = torch.full_like(tgt, -1)
    for k, c in enumerate(ACGT): y[tgt == c] = k
    ok = (y >= 0); ok[:512] = False
    hit = (x[ok].argmax(1) == y[ok]).float().mean().item()
    bits = torch.nn.functional.cross_entropy(x[ok], y[ok]).item() / np.log(2)
    ident = {}
    for i in BLK:
        u, yy = stats[f"u{i}"]["last"], stats[f"y{i}"]["last"]
        br = stats[f"mix{i}"]["last"] + stats[f"mlp{i}"]["last"]
        ident[i] = ((yy - u - br).norm() / (yy - u).norm().clamp(min=1e-30)).item()
    return hit, bits, ident

res = {"branch": {}, "ident": {}, "dom": {}}
print("=" * 100); print("1-2. branch norms by input type (mean per position)"); print("=" * 100)
for kind, seqs in inputs.items():
    reset(); accs = []; idents = []
    for s in seqs:
        h, b, idt = run(s); accs.append((h, b)); idents.append(idt)
    row = {}
    for i in BLK:
        f = lambda k: stats[k]["norm"] / stats[k]["n"]
        row[i] = {"u": f(f"u{i}"), "mix": f(f"mix{i}"), "mlp": f(f"mlp{i}"), "y": f(f"y{i}")}
    top28 = torch.topk(stats["y28"]["chan"], 8).indices.tolist()
    topmlp28 = torch.topk(stats["mlp28"]["chan"], 8).indices.tolist()
    topmix28 = torch.topk(stats["mix28"]["chan"], 8).indices.tolist()
    res["branch"][kind] = row
    res["dom"][kind] = {"y28": top28, "mlp28": topmlp28, "mix28": topmix28}
    worst_ident = max(max(d.values()) for d in idents)
    print(f"\n[{kind}] next-base acc {np.mean([a for a,_ in accs]):.3f} bits {np.mean([b for _,b in accs]):.3f} "
          f"| identity residual max {worst_ident:.2e}")
    print(f"  {'blk':<5}{'|u|':>12}{'|mixer|':>12}{'|mlp|':>12}{'|y|':>12}{'y/u':>9}")
    for i in BLK:
        r = row[i]; print(f"  {i:<5}{r['u']:>12.4g}{r['mix']:>12.4g}{r['mlp']:>12.4g}{r['y']:>12.4g}{r['y']/max(r['u'],1e-30):>9.1f}")
    print(f"  top channels y28 {top28} | overlap with prior {len(set(top28)&set(DOM_PRIOR))}/8")
    print(f"  top channels mlp28 {topmlp28} | mixer28 {topmix28}")

print("\n" + "=" * 100); print("3. weights at dominant channels vs all channels"); print("=" * 100)
DOM = res["dom"]["real"]["y28"]; wres = {}
for i in (27, 28, 29):
    blk = B[i]; row = {}
    def cmp(name, vec):
        v = vec.detach().float().abs().cpu().numpy(); d = v[DOM]
        row[name] = {"dom_mean": float(d.mean()), "all_median": float(np.median(v)), "dom_rank_pct": float(np.mean([(v < x).mean() for x in d]))}
        print(f"  blk{i} {name:<22} dominant mean {d.mean():.4g} | all median {np.median(v):.4g} | dominant percentile {row[name]['dom_rank_pct']:.3f}")
    cmp("pre_norm.scale", blk.pre_norm.scale)
    cmp("post_norm.scale", blk.post_norm.scale)
    cmp("mlp.l3 row norm", blk.mlp.l3.weight.float().norm(dim=1))
    if hasattr(blk, "out_filter_dense"):
        cmp("mixer out row norm", blk.out_filter_dense.weight.float().norm(dim=1))
        if blk.out_filter_dense.bias is not None: cmp("mixer out bias", blk.out_filter_dense.bias)
    wres[i] = row
res["weights"] = wres

print("\n" + "=" * 100); print("4. ablations on real windows"); print("=" * 100)
def ablate(kind):
    if kind == "none": return []
    if kind == "zero_mlp28": return [B[28].mlp.register_forward_hook(lambda mod, i, o: torch.zeros_like(o))]
    if kind == "zero_mixer28": return [B[28].out_filter_dense.register_forward_hook(lambda mod, i, o: torch.zeros_like(o))]
    if kind == "zero_dom_mlp28":
        def f(mod, i, o):
            o = o.clone(); o[..., DOM] = 0; return o
        return [B[28].mlp.register_forward_hook(f)]
res["ablation"] = {}
for kind in ("none", "zero_mlp28", "zero_mixer28", "zero_dom_mlp28"):
    ah = ablate(kind); reset(); accs = []
    for s in real: h, b, _ = run(s); accs.append((h, b))
    for h in ah: h.remove()
    prof = {i: stats[f"y{i}"]["norm"] / stats[f"y{i}"]["n"] for i in BLK}
    res["ablation"][kind] = {"acc": float(np.mean([a for a,_ in accs])), "bits": float(np.mean([b for _,b in accs])), "norm": prof}
    print(f"  {kind:<16} acc {res['ablation'][kind]['acc']:.3f} bits {res['ablation'][kind]['bits']:.3f} | |y|: " +
          " ".join(f"{i}:{prof[i]:.3g}" for i in BLK))
for h in hooks: h.remove()
json.dump(res, open(f"{AC}/results/cascade_cause_7b.json", "w"), indent=1, default=str)
print("wrote cascade_cause_7b.json")

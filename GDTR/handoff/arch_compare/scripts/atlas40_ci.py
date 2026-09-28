"""Window-cluster bootstrap for the 40B restricted causal atlas.

atlas40.py stores one NLL per window per condition. The unit of resampling is the
genomic window, not the position, because positions inside a window are not
independent. Each target is compared with the homologous control of the same
operator class one cycle earlier, and the target-minus-control difference is
bootstrapped on the same resampled windows so the two share their noise.
"""
import numpy as np, json, os, sys

AC="/path/to/TDiG/arch_compare"
NPZ=f"{AC}/results/atlas40_per_window.npz"
OUT=f"{AC}/results/atlas40_ci.json"
B=10000
TARGETS=[21,22,23,34]; CONTROLS=[18,19,20,30]; BRANCHES=["mlp","mix"]

if not os.path.exists(NPZ):
    print("ABORT: no", NPZ, "- run atlas40.py first"); sys.exit(1)
z=np.load(NPZ)
if "baseline" not in z:
    print("ABORT: baseline condition missing from", NPZ); sys.exit(1)
base=z["baseline"]; wins=z["windows"] if "windows" in z else np.arange(len(base))
ok=np.isfinite(base)
for k in z.files:
    if k != "windows": ok &= np.isfinite(z[k])
n=int(ok.sum())
print(f"windows usable in every condition: {n} of {len(base)}", flush=True)
if n < 20:
    print("ABORT: too few usable windows for a window-cluster bootstrap"); sys.exit(1)

rng=np.random.default_rng(0)
draws=rng.integers(0, n, size=(B, n))          # one resample shared by all conditions

def boot(delta):
    d=delta[ok]
    means=d[draws].mean(axis=1)
    return float(d.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5)), means

res={}; cache={}
print("\n=== delta NLL against baseline, window-cluster bootstrap B=%d ===" % B, flush=True)
for b in TARGETS+CONTROLS:
    for br in BRANCHES:
        k=f"b{b}.{br}"
        if k not in z.files: continue
        m,lo,hi,dist=boot(z[k]-base)
        cache[k]=dist
        role="target " if b in TARGETS else "control"
        star="" if (lo<=0<=hi) else "  excludes 0"
        res[k]={"block":b,"branch":br,"role":role.strip(),"delta_nll":m,"ci":[lo,hi]}
        print(f"  {k:10s} {role}  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}", flush=True)

print("\n=== target minus homologous control, paired on the same resampled windows ===", flush=True)
res["paired"]={}
for t,c in zip(TARGETS,CONTROLS):
    for br in BRANCHES:
        kt,kc=f"b{t}.{br}",f"b{c}.{br}"
        if kt not in cache or kc not in cache: continue
        diff=cache[kt]-cache[kc]
        point=res[kt]["delta_nll"]-res[kc]["delta_nll"]
        lo,hi=float(np.percentile(diff,2.5)), float(np.percentile(diff,97.5))
        star="" if (lo<=0<=hi) else "  excludes 0"
        res["paired"][f"b{t}-b{c}.{br}"]={"point":point,"ci":[lo,hi]}
        print(f"  b{t} - b{c}  {br:4s}  {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}", flush=True)

json.dump({"B":B,"n_windows":n,"conditions":res}, open(OUT,"w"), indent=1)
print("\nwrote", OUT, flush=True)

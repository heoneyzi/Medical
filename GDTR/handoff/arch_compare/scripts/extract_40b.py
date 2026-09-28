"""Evo 2 40B: per-layer hidden states for the same chr22/chr17 windows as 7B.

50 blocks x 8192 dims x fp32 is 1.6 MB per stored position, so:
  * all positions -> per-layer norms only (for the cascade profile),
  * a fixed row subsample -> full fp32 states, written straight to one
    memmapped .npy per layer (never held in RAM).
Rows are chosen from labels alone before any forward pass, so the row count is
known up front and the selection is reproducible.

usage: python extract_40b.py <chr22|chr17> [noflash]
"""
import os, sys, json, gzip, time
import numpy as np, pandas as pd, torch

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CHROM = sys.argv[1]; NOFLASH = len(sys.argv) > 2
NAME = "evo2_40b"
SCAL = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}[CHROM]
FRAC = {5: 1.0, 6: 1.0, 2: 0.10, 1: 0.035, 0: 0.035}     # donor/acceptor kept whole
RNG = np.random.default_rng(20260915)
OUT = f"{AC}/results/{CHROM}_evo2_40b"; os.makedirs(OUT, exist_ok=True)

import evo2
from evo2.utils import CONFIG_MAP
if NOFLASH:
    pkg = os.path.dirname(evo2.__file__)
    src = os.path.join(pkg, CONFIG_MAP[NAME]); dst = src.replace(".yml", "-noflash.yml")
    if not os.path.exists(dst):
        open(dst, "w").write(open(src).read().replace("use_flash_attn: True", "use_flash_attn: False"))
    CONFIG_MAP[NAME] = os.path.relpath(dst, pkg)
from evo2 import Evo2

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq)
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
st = meta.set_index("window_idx")["start"].to_dict(); en = meta.set_index("window_idx")["end"].to_dict()
labels_full = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")
w_all = np.load(f"{AC}/results/{SCAL}")["window"]
use = [int(w) for w in w_all[np.concatenate(([True], w_all[1:] != w_all[:-1]))]]

# ---- pass 1: choose rows from labels only ----
plan = []
for w in use:
    a, b = int(st[w]), int(en[w]); T = b - a
    lab = np.zeros(T, dtype=np.uint8); gp = a + np.arange(T)
    ok = (gp >= 0) & (gp < len(labels_full)); lab[ok] = labels_full[gp[ok]]
    sel = np.zeros(T, dtype=bool)
    for code, f in FRAC.items():
        idx = np.where(lab == code)[0]
        if len(idx) == 0: continue
        k = len(idx) if f >= 1 else max(1, int(len(idx) * f))
        sel[RNG.choice(idx, k, replace=False)] = True
    plan.append((w, a, b, lab, np.where(sel)[0]))
NROW = sum(len(p[4]) for p in plan)
print(f"{CHROM}: windows {len(plan)} | stored rows {NROW} | "
      f"~{NROW*51*8192*4/1e9:.0f} GB on disk", flush=True)

m = Evo2(NAME); net = m.model
NB = len(net.blocks); D = net.config.get("hidden_size", 8192) if hasattr(net, "config") else 8192
print(f"model ready: {NB} blocks, D={D}, cuda mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

caught = {}
def mk(name):
    def hook(mod, inp, out):
        caught[name] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()
    return hook
hs = [net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(NB)]
hs.append(net.norm.register_forward_hook(mk("norm")))
keys = [f"b{i}" for i in range(NB)] + ["norm"]

mm = {k: np.lib.format.open_memmap(f"{OUT}/{k}.npy", mode="w+", dtype=np.float32, shape=(NROW, D))
      for k in keys}
NORMS = {k: [] for k in keys}; LAB = []; WIN = []; row = 0; t0 = time.time()
for i, (w, a, b, lab, rows) in enumerate(plan):
    s = S[a:b].upper()
    ids = torch.tensor([m.tokenizer.tokenize(s)], dtype=torch.long, device="cuda")
    caught.clear()
    with torch.no_grad(): net(ids)
    r = torch.as_tensor(rows, device="cuda")
    for k in keys:
        h = caught[k]
        NORMS[k].append(h.norm(dim=-1).cpu().numpy().astype(np.float64))   # fp64: norms reach 1e12+
        if len(rows): mm[k][row:row+len(rows)] = h[r].cpu().numpy()
    LAB.append(lab[rows]); WIN.append(np.full(len(rows), w, dtype=np.int32)); row += len(rows)
    caught.clear(); torch.cuda.empty_cache()
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(plan)}  {time.time()-t0:.0f}s", flush=True)
for h in hs: h.remove()
for k in keys: mm[k].flush()

allnorm = {k: np.concatenate(v) for k, v in NORMS.items()}
prof = [float(allnorm[k].mean()) for k in keys]
ratio = [prof[i] / prof[i-1] for i in range(1, NB)]
np.savez(f"{OUT}/meta.npz", label=np.concatenate(LAB), window=np.concatenate(WIN),
         **{f"norm_{k}": allnorm[k].astype(np.float32 if max(prof) < 3e38 else np.float64) for k in keys})
json.dump({"blocks": NB, "hidden": D, "rows": NROW, "mean_norm": prof, "ratio": ratio,
           "finite_fraction": {k: float(np.isfinite(mm[k][:min(NROW,4000)]).all(1).mean()) for k in keys}},
          open(f"{OUT}/profile.json", "w"), indent=1)
onset = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
print("mean norm:", " ".join(f"{i}:{v:.3g}" for i, v in enumerate(prof[:NB])))
print("max adjacent ratio:", f"{max(ratio):.3g} @ block {int(np.argmax(ratio))+1}", "| onset(T=10):", onset)
print("saved", OUT, flush=True)

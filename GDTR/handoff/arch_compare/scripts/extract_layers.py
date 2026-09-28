"""Generic per-layer extraction (Evo 2 7B) writing the same directory layout as
the 40B extractor, so one analysis script serves both.

usage: python extract_layers.py <chrom>
Writes results/<chrom>_evo2_7b_all/{b0..bNB-1,norm}.npy, meta.npz, profile.json
Idempotent: exits immediately if profile.json already exists.
"""
import os, sys, gzip, json, time
import numpy as np, pandas as pd, torch
ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
CHROM = sys.argv[1]
OUT = f"{AC}/results/{CHROM}_evo2_7b_all"
if os.path.exists(f"{OUT}/profile.json"):
    print("already done:", OUT); sys.exit(0)
os.makedirs(OUT, exist_ok=True)
FRAC = {5: 1.0, 6: 1.0, 2: 0.10, 1: 0.035, 0: 0.035}
RNG = np.random.default_rng(20260916)

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
labels_full = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")

plan = []
for _, r in meta.iterrows():
    a, b = int(r["start"]), int(r["end"]); T = b - a
    lab = np.zeros(T, np.uint8); gp = a + np.arange(T)
    ok = (gp >= 0) & (gp < len(labels_full)); lab[ok] = labels_full[gp[ok]]
    sel = np.zeros(T, bool)
    for code, f in FRAC.items():
        idx = np.where(lab == code)[0]
        if len(idx) == 0: continue
        k = len(idx) if f >= 1 else max(1, int(len(idx) * f))
        sel[RNG.choice(idx, k, replace=False)] = True
    plan.append((int(r["window_idx"]), a, b, lab, np.where(sel)[0]))
NROW = sum(len(p[4]) for p in plan)
print(f"{CHROM}: windows {len(plan)} | stored rows {NROW}", flush=True)

from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2("evo2_7b"); net = m.model; NB = len(net.blocks); D = 4096
print(f"model ready: {NB} blocks", flush=True)
caught = {}
def mk(n):
    def h(mod, inp, out):
        caught[n] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()
    return h
hs = [net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(NB)]
hs.append(net.norm.register_forward_hook(mk("norm")))
keys = [f"b{i}" for i in range(NB)] + ["norm"]
mm = {k: np.lib.format.open_memmap(f"{OUT}/{k}.npy", mode="w+", dtype=np.float32, shape=(NROW, D)) for k in keys}

NORMS = {k: [] for k in keys}; LAB = []; WIN = []; row = 0; t0 = time.time()
for i, (w, a, b, lab, rows) in enumerate(plan):
    ids = torch.tensor([m.tokenizer.tokenize(S[a:b])], dtype=torch.long, device="cuda")
    caught.clear()
    with torch.no_grad(): net(ids)
    r = torch.as_tensor(rows, device="cuda")
    for k in keys:
        h = caught[k]
        NORMS[k].append(h.norm(dim=-1).cpu().numpy().astype(np.float64))
        if len(rows): mm[k][row:row + len(rows)] = h[r].cpu().numpy()
    LAB.append(lab[rows]); WIN.append(np.full(len(rows), w, np.int32)); row += len(rows)
    caught.clear(); torch.cuda.empty_cache()
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(plan)}  {time.time()-t0:.0f}s", flush=True)
for h in hs: h.remove()
for k in keys: mm[k].flush()
prof = [float(np.concatenate(NORMS[k]).mean()) for k in keys]
ratio = [prof[i] / prof[i - 1] for i in range(1, NB)]
np.savez(f"{OUT}/meta.npz", label=np.concatenate(LAB), window=np.concatenate(WIN))
json.dump({"blocks": NB, "hidden": D, "rows": NROW, "mean_norm": prof, "ratio": ratio},
          open(f"{OUT}/profile.json", "w"), indent=1)
onset = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
print("max adjacent ratio:", f"{max(ratio):.3g}", "| onset(T=10):", onset, flush=True)
with open(f"{AC}/results/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\textract\t{CHROM}\t7B\trows={NROW}\tonset={onset}\tmaxratio={max(ratio):.3g}\n")

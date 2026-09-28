"""Per-layer extraction for either Evo 2 model, any chromosome with a window
metadata parquet.  Writes the common directory layout consumed by
analyze_layers.py.

usage: python extract_any.py <evo2_7b|evo2_40b> <chrom> [utr]
  utr -> also sample 5'UTR (code 3) and 3'UTR (code 4), which earlier
         extractions dropped entirely.
Idempotent: exits if profile.json exists.
"""
import os, sys, gzip, json, time
import numpy as np, pandas as pd, torch
ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL, CHROM = sys.argv[1], sys.argv[2]
UTR = len(sys.argv) > 3 and sys.argv[3] == "utr"
TAG = f"{CHROM}_{MODEL}" + ("_utr" if UTR else "_all")
OUT = f"{AC}/results/{TAG}"
if os.path.exists(f"{OUT}/profile.json"):
    print("already done:", OUT); sys.exit(0)
os.makedirs(OUT, exist_ok=True)
FRAC = {5: 1.0, 6: 1.0, 2: 0.10, 1: 0.035, 0: 0.035}
if UTR: FRAC.update({3: 0.25, 4: 0.10})
RNG = np.random.default_rng(20260916)

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
labels_full = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")

# Use the same 100-window panel as every other extraction.  chr22/chr17 have a
# 12,978-window metadata table; the panel is the window order recorded in the
# scalars file.  chr19's table is already the panel.
SCAL = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}
if CHROM in SCAL:
    w = np.load(f"{AC}/results/{SCAL[CHROM]}")["window"]
    panel = [int(x) for x in w[np.concatenate(([True], w[1:] != w[:-1]))]]
    meta = meta.set_index("window_idx").loc[panel].reset_index()
if len(meta) > 200:
    sys.exit(f"ABORT: {len(meta)} windows selected; the panel should be ~100. "
             "Refusing to write a multi-terabyte cache.")
print(f"panel: {len(meta)} windows", flush=True)

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
cnt = {}
for _, _, _, lab, rows in plan:
    for c, n in zip(*np.unique(lab[rows], return_counts=True)): cnt[int(c)] = cnt.get(int(c), 0) + int(n)
print(f"{TAG}: windows {len(plan)} | rows {NROW} | per-class {cnt}", flush=True)

from evo2.utils import CONFIG_MAP
if MODEL == "evo2_7b": CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2(MODEL); net = m.model; NB = len(net.blocks)
D = int(net.blocks[0].pre_norm.scale.shape[0]) if hasattr(net.blocks[0], "pre_norm") else 4096
print(f"model ready: {NB} blocks, D={D}", flush=True)
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
    if (i + 1) % 20 == 0: print(f"  {i+1}/{len(plan)}  {time.time()-t0:.0f}s", flush=True)
for h in hs: h.remove()
for k in keys: mm[k].flush()
prof = [float(np.concatenate(NORMS[k]).mean()) for k in keys]
ratio = [prof[i] / prof[i - 1] for i in range(1, NB)]
np.savez(f"{OUT}/meta.npz", label=np.concatenate(LAB), window=np.concatenate(WIN))
json.dump({"blocks": NB, "hidden": D, "rows": NROW, "mean_norm": prof, "ratio": ratio},
          open(f"{OUT}/profile.json", "w"), indent=1)
onset = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
print(f"max adjacent ratio {max(ratio):.3g} | onset(T=10) {onset}", flush=True)
with open(f"{AC}/results/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\textract\t{TAG}\trows={NROW}\tonset={onset}\n")

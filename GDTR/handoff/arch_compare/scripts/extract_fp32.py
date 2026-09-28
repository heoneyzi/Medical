"""Re-extract the LATE layers in fp32, plus the final norm output.

Two gaps in the layer-selection result this closes:
  1. L29-31 overflowed fp16, so the true last attention block (L31) was never
     evaluated.  The hooks always captured float32; only the cache was fp16.
  2. h_norm -- what people actually use as "the embedding" -- was never in the
     probe cache, so the conventional baseline was block 28, a straw man.

Reproduces the original subsample exactly: same window order (taken from the
scalars file, which is `use` in row order) and the same RNG draw sequence.
"""
import gzip, os, sys, json
import numpy as np, pandas as pd, torch

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CHROM = sys.argv[1] if len(sys.argv) > 1 else "chr22"
SCAL = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}[CHROM]
KEEP = range(24, 32)                       # late blocks, incl. attention L31
RNG = np.random.default_rng(20260903)      # identical seed
KEEP_ALL = {2, 5, 6}; SUB = {1: 0.035, 0: 0.035}

from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2("evo2_7b"); net = m.model
print("model ready", flush=True)

caught = {}
def mk(name):
    def hook(mod, inp, out):
        caught[name] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()
    return hook
hs = [net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in KEEP]
hs.append(net.norm.register_forward_hook(mk("norm")))

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq)
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
st_of = meta.set_index("window_idx")["start"].to_dict()
en_of = meta.set_index("window_idx")["end"].to_dict()
labels_full = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")

w_all = np.load(f"{AC}/results/{SCAL}")["window"]
use = [int(w) for w in w_all[np.concatenate(([True], w_all[1:] != w_all[:-1]))]]
print(f"windows {len(use)} (from scalars row order)", flush=True)

PH = {k: [] for k in list(KEEP) + ["norm"]}; PL = []; PW = []
for i, w in enumerate(use):
    st, en = int(st_of[w]), int(en_of[w]); T = en - st
    s = S[st:en].upper()
    if len(s) != T: continue
    ids = torch.tensor([m.tokenizer.tokenize(s)], dtype=torch.long).cuda()
    caught.clear()
    with torch.no_grad(): net(ids)
    gp = st + np.arange(T)
    lab = np.zeros(T, dtype=np.uint8)
    ok = (gp >= 0) & (gp < len(labels_full)); lab[ok] = labels_full[gp[ok]]
    sel = np.isin(lab, list(KEEP_ALL))
    for code, frac in SUB.items():                    # same order, same draws
        idx = np.where(lab == code)[0]
        if len(idx): sel[RNG.choice(idx, max(1, int(len(idx)*frac)), replace=False)] = True
    if sel.sum():
        for k in KEEP: PH[k].append(caught[f"b{k}"][sel].cpu().numpy().astype(np.float32))
        PH["norm"].append(caught["norm"][sel].cpu().numpy().astype(np.float32))
        PL.append(lab[sel]); PW.append(np.full(int(sel.sum()), w, dtype=np.int32))
    torch.cuda.empty_cache()
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(use)}", flush=True)
for h in hs: h.remove()

out = {f"h{k}": np.concatenate(PH[k]) for k in KEEP}
out["hnorm"] = np.concatenate(PH["norm"])
out["label"] = np.concatenate(PL); out["window"] = np.concatenate(PW)
n = len(out["label"])
exp = {"chr22": 146098, "chr17": 138424}[CHROM]
print(f"rows {n} (original subsample {exp}) -> {'MATCH' if n == exp else 'MISMATCH'}")
for k in list(KEEP) + ["norm"]:
    a = out[f"h{k}" if k != "norm" else "hnorm"]
    print(f"  {k}: mean|h| = {np.linalg.norm(a, axis=1).mean():.4g} "
          f"finite = {np.isfinite(a).all(1).mean():.4f}")
np.savez(f"{AC}/results/{CHROM}_late_fp32.npz", **out)
print("saved", flush=True)

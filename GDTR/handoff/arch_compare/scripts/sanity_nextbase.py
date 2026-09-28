"""Numerical sanity check before trusting any hidden state from a model.

Evo 2 README: the 40B/20B/1B models need FP8 via Transformer Engine "for
numerical accuracy".  A mis-loaded model still runs and still emits hidden
states, so check the thing a language model is for: next-base prediction on
real chr22 sequence.  40B should match or beat 7B; if it is worse, the load is
wrong and nothing downstream can be used.

usage: python sanity_nextbase.py <evo2_7b|evo2_40b> <n_windows> [noflash]
"""
import os, sys, json, gzip, shutil, time
import numpy as np, pandas as pd, torch

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
NAME = sys.argv[1]; NWIN = int(sys.argv[2]); NOFLASH = len(sys.argv) > 3
CHROM, SKIP = "chr22", 512            # ignore the first tokens: too little context

import evo2
from evo2.utils import CONFIG_MAP
if NOFLASH:
    pkg = os.path.dirname(evo2.__file__)
    src = os.path.join(pkg, CONFIG_MAP[NAME])
    dst = src.replace(".yml", "-noflash.yml")
    if not os.path.exists(dst):
        txt = open(src).read().replace("use_flash_attn: True", "use_flash_attn: False")
        open(dst, "w").write(txt)
    CONFIG_MAP[NAME] = os.path.relpath(dst, pkg)
from evo2 import Evo2

t0 = time.time()
m = Evo2(NAME)
print(f"loaded {NAME} in {time.time()-t0:.0f}s | cuda mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq)
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
w_all = np.load(f"{AC}/results/evo2_7b_scalars.npz")["window"]
use = [int(w) for w in w_all[np.concatenate(([True], w_all[1:] != w_all[:-1]))]][:NWIN]
st = meta.set_index("window_idx")["start"].to_dict(); en = meta.set_index("window_idx")["end"].to_dict()

ACGT = torch.tensor([ord(c) for c in "ACGT"], device="cuda")
tot_n = tot_hit = 0; tot_ce = 0.0; per = []
for w in use:
    s = S[int(st[w]):int(en[w])].upper()
    keep = np.array([c in "ACGT" for c in s])
    ids = torch.tensor([m.tokenizer.tokenize(s)], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = m(ids)
    logits = out[0][0] if isinstance(out[0], (tuple, list)) else out[0]
    logits = logits[0].float()                                   # (T, V)
    lg = logits[:-1][:, ACGT]                                    # predict t+1 over A,C,G,T
    tgt = ids[0, 1:]
    y = torch.full_like(tgt, -1)
    for k, c in enumerate(ACGT): y[tgt == c] = k
    ok = (y >= 0).cpu().numpy() & keep[1:] & (np.arange(len(tgt)) >= SKIP)
    okt = torch.tensor(ok, device="cuda")
    ce = torch.nn.functional.cross_entropy(lg[okt], y[okt], reduction="sum").item()
    hit = (lg[okt].argmax(1) == y[okt]).sum().item(); n = int(ok.sum())
    tot_n += n; tot_hit += hit; tot_ce += ce
    per.append(hit / max(n, 1))
    finite = bool(torch.isfinite(logits).all())
    print(f"  window {w}: n={n} acc={hit/max(n,1):.4f} bits={ce/max(n,1)/np.log(2):.4f} finite={finite}", flush=True)

res = {"model": NAME, "noflash": NOFLASH, "windows": len(use), "positions": tot_n,
       "acc": tot_hit / tot_n, "bits_per_base": tot_ce / tot_n / np.log(2),
       "acc_window_sd": float(np.std(per))}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{AC}/results/sanity_{NAME}{'_noflash' if NOFLASH else ''}.json", "w"), indent=1)

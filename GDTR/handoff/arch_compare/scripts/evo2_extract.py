"""Evo 2 7B forward over the same chr22 windows: full layer trajectory.

Hooks每 block + the final norm, so we get
    h_emb, h_after_block_0 .. h_after_block_31, h_norm
which is exactly the trajectory the tier2 scalars were derived from, but with
the raw states retained so a supervised probe can be run alongside V and PRA.

Saves
  scalars: per-position per-layer velocity and cos-to-h_norm (fp16, all 600k pos)
  probe subsample: raw hidden states for splice sites + sampled other contexts
"""
import gzip, os, sys, json
import numpy as np, pandas as pd, torch, h5py

ROOT = "/path/to/TDiG"
AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RNG = np.random.default_rng(20260903)
KEEP_ALL = {2, 5, 6}
SUB = {1: 0.035, 0: 0.035}

from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2("evo2_7b"); net = m.model
print("model ready", flush=True)

caught = {}
def mk(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        caught[name] = o.detach()[0].float()
    return hook
hs = [net.blocks[i].register_forward_hook(mk(f"b{i}")) for i in range(len(net.blocks))]
hs.append(net.norm.register_forward_hook(mk("norm")))
NB = len(net.blocks)
print("blocks", NB, flush=True)

seq = []
with gzip.open(f"{AC}/chr22.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq)
with h5py.File(f"{ROOT}/data/chr22_tier2_scalars.h5", "r") as f:
    widx = f["window_idx"][:]
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
st_of = meta.set_index("window_idx")["start"].to_dict()
en_of = meta.set_index("window_idx")["end"].to_dict()
labels_full = np.load(f"{ROOT}/paper/data_local/chr22_position_labels.npy")
use = [int(w) for w in widx[:NWIN] if int(w) in st_of]
print("windows", len(use), flush=True)

VEL, COS, LB, WI = [], [], [], []
PH, PL, PW = [], [], []
prof_n = prof_c = prof_v = None; nprof = 0

for i, w in enumerate(use):
    st, en = int(st_of[w]), int(en_of[w]); T = en - st
    s = S[st:en].upper()
    if len(s) != T: continue
    ids = torch.tensor([m.tokenizer.tokenize(s)], dtype=torch.long).cuda()
    assert ids.shape[1] == T, f"token misalignment {ids.shape[1]} vs {T}"
    caught.clear()
    with torch.no_grad(): net(ids)
    H = torch.stack([caught[f"b{k}"] for k in range(NB)])      # (32, T, D)
    hn = caught["norm"]                                        # (T, D)
    nrm = H.norm(dim=-1)
    vel = (H[1:] - H[:-1]).norm(dim=-1) / nrm[:-1].clamp(min=1e-12)   # (31, T)
    cos = torch.nn.functional.cosine_similarity(H, hn.unsqueeze(0).expand_as(H), dim=-1)

    pn, pc, pv = nrm.mean(1).cpu().numpy(), cos.mean(1).cpu().numpy(), vel.mean(1).cpu().numpy()
    prof_n = pn if prof_n is None else prof_n + pn
    prof_c = pc if prof_c is None else prof_c + pc
    prof_v = pv if prof_v is None else prof_v + pv
    nprof += 1

    gp = st + np.arange(T)
    lab = np.zeros(T, dtype=np.uint8)
    ok = (gp >= 0) & (gp < len(labels_full))
    lab[ok] = labels_full[gp[ok]]
    VEL.append(vel.T.cpu().numpy().astype(np.float16))
    COS.append(cos.T.cpu().numpy().astype(np.float16))
    LB.append(lab); WI.append(np.full(T, w, dtype=np.int32))

    sel = np.isin(lab, list(KEEP_ALL))
    for code, frac in SUB.items():
        idx = np.where(lab == code)[0]
        if len(idx): sel[RNG.choice(idx, max(1, int(len(idx)*frac)), replace=False)] = True
    if sel.sum():
        PH.append(H[:, sel, :].cpu().numpy().astype(np.float16))
        PL.append(lab[sel]); PW.append(np.full(int(sel.sum()), w, dtype=np.int32))
    del H, hn, nrm, vel, cos
    torch.cuda.empty_cache()
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(use)}", flush=True)

for h in hs: h.remove()
json.dump({"norm": (prof_n/nprof).tolist(), "cos_norm": (prof_c/nprof).tolist(),
           "vel": (prof_v/nprof).tolist(), "n_windows": nprof, "n_blocks": NB},
          open(f"{AC}/results/evo2_7b_layer_profile.json", "w"), indent=2)
np.savez(f"{AC}/results/evo2_7b_scalars.npz",
         vel=np.concatenate(VEL), cos=np.concatenate(COS),
         label=np.concatenate(LB), window=np.concatenate(WI))
np.savez(f"{AC}/results/evo2_7b_probe_subsample.npz",
         H=np.concatenate(PH, axis=1), label=np.concatenate(PL), window=np.concatenate(PW))
print("saved. scalars", np.concatenate(LB).shape, "| probe", np.concatenate(PL).shape)

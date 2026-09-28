"""Compute V and PRA on HyenaDNA over the SAME chr22 windows used for Evo 2.

V   = min over pre-rotation layers of  z_l( ||h_{l+1}-h_l|| / ||h_l|| )
PRA = max over pre-rotation layers of  cos(h_l, h_final)

Also records the per-layer profile (norm, cos-to-final, velocity) so the
rotation layer can be detected the same automatic way as for Evo 2, and so the
trajectory shape can be compared across architectures.
"""
import gzip, json, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = "/path/to/TDiG"
OUT = f"{ROOT}/arch_compare/results"
MID = "LongSafari/hyenadna-medium-160k-seqlen-hf"
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 100

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModel.from_pretrained(MID, trust_remote_code=True).to(dev).eval()

# ---- tokeniser alignment: is the extra token prepended or appended? ----
a = tok("AAAA", return_tensors="pt")["input_ids"][0].tolist()
b = tok("AAAAC", return_tensors="pt")["input_ids"][0].tolist()
if a[:4] == b[:4]:
    OFF = 0                      # extra token(s) appended
else:
    OFF = len(a) - 4             # extra token(s) prepended
print(f"tokenizer: len('AAAA')={len(a)} len('AAAAC')={len(b)} -> offset {OFF}",
      flush=True)

# ---- sequence ----
seq = []
with gzip.open(f"{ROOT}/arch_compare/chr22.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"):
            seq.append(line.strip())
S = "".join(seq)
up = np.frombuffer(S.upper().encode(), dtype=np.uint8)
print(f"chr22 {len(S):,} bp", flush=True)

# ---- the same windows Evo 2 used ----
import h5py
with h5py.File(f"{ROOT}/data/chr22_tier2_scalars.h5", "r") as f:
    widx = f["window_idx"][:]
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
start_of = meta.set_index("window_idx")["start"].to_dict()
T_of = meta.set_index("window_idx")["end"].to_dict()
labels_full = np.load(f"{ROOT}/paper/data_local/chr22_position_labels.npy")
use = [int(w) for w in widx[:NWIN] if int(w) in start_of]
print(f"windows: {len(use)}", flush=True)

prof_n, prof_c, prof_v, nprof = None, None, None, 0
Vs, PRAs, aV, aP, LBs, WIs = [], [], [], [], [], []

for i, w in enumerate(use):
    st = int(start_of[w]); en = int(T_of[w]); T = en - st
    s = S[st:en]
    if len(s) != T or up[st:en].min() == 0:
        continue
    ids = tok(s.upper(), return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0] for h in hs]).float()          # (L+1, Ttok, D)
    H = H[:, OFF:OFF + T, :]                             # align to nucleotides
    nrm = H.norm(dim=-1)                                 # (L+1, T)
    step = (H[1:] - H[:-1]).norm(dim=-1)                 # (L, T)
    v = step / nrm[:-1].clamp(min=1e-12)                 # (L, T)
    hf = H[-1]
    cosf = torch.nn.functional.cosine_similarity(
        H, hf.unsqueeze(0).expand_as(H), dim=-1)         # (L+1, T)

    pn = nrm.mean(1).cpu().numpy(); pc = cosf.mean(1).cpu().numpy()
    pv = v.mean(1).cpu().numpy()
    prof_n = pn if prof_n is None else prof_n + pn
    prof_c = pc if prof_c is None else prof_c + pc
    prof_v = pv if prof_v is None else prof_v + pv
    nprof += 1

    Vs.append(v.cpu().numpy()); PRAs.append(cosf.cpu().numpy())
    gp = st + np.arange(T)
    lab = np.zeros(T, dtype=np.uint8)
    ok = (gp >= 0) & (gp < len(labels_full))
    lab[ok] = labels_full[gp[ok]]
    LBs.append(lab); WIs.append(np.full(T, w, dtype=np.int32))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(use)}", flush=True)

prof = dict(norm=(prof_n / nprof).tolist(), cos_final=(prof_c / nprof).tolist(),
            vel=(prof_v / nprof).tolist(), n_windows=nprof)
json.dump(prof, open(f"{OUT}/hyenadna_layer_profile.json", "w"), indent=2)

# rotation layer = largest single-layer jump in mean cos-to-final
pc = np.array(prof["cos_final"])
jump = np.diff(pc)
rot = int(np.argmax(jump)) + 1
LMAX = rot                      # use layers strictly before the rotation
print(f"\nmean cos-to-final per layer: {np.round(pc,3).tolist()}")
print(f"largest jump into layer {rot} (+{jump.max():.3f}) -> using layers 0..{LMAX-1}")

V = np.concatenate([x[:, :] for x in Vs], axis=1)        # (L, Ntot)
PRA = np.concatenate(PRAs, axis=1)                       # (L+1, Ntot)
lab = np.concatenate(LBs); win = np.concatenate(WIs)
Lv = min(LMAX, V.shape[0])
mu = V[:Lv].mean(1, keepdims=True); sd = V[:Lv].std(1, keepdims=True).clip(1e-12)
Vz = (V[:Lv] - mu) / sd
v_val, v_arg = Vz.min(0), Vz.argmin(0)
p_val, p_arg = PRA[:LMAX].max(0), PRA[:LMAX].argmax(0)

np.savez_compressed(f"{OUT}/hyenadna_chr22_VPRA.npz",
                    V=v_val.astype(np.float32), V_layer=v_arg.astype(np.int8),
                    PRA=p_val.astype(np.float32), PRA_layer=p_arg.astype(np.int8),
                    label=lab, window=win, n_layers=V.shape[0], rot=rot)
print(f"\nsaved {OUT}/hyenadna_chr22_VPRA.npz   n={len(lab):,}")

"""Diagnostic: does HyenaDNA encode context at all?

Layer-wise supervised probe on the raw hidden states, leave-window-out CV,
compared head-to-head with the training-free V and PRA on the SAME positions.

  probe fails too      -> the model does not carry the signal (metric is fine)
  probe works, V fails -> the signal is there but these readouts miss it
"""
import gzip, sys
import numpy as np
import pandas as pd
import torch, h5py
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT = "/path/to/TDiG"
MID = "LongSafari/hyenadna-medium-160k-seqlen-hf"
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RNG = np.random.default_rng(20260822)
KEEP_ALL = {2, 5, 6}          # exon, donor, acceptor: keep every position
SUB = {1: 0.10, 0: 0.10}      # intron, intergenic: subsample

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModel.from_pretrained(MID, trust_remote_code=True).to(dev).eval()

seq = []
with gzip.open(f"{ROOT}/arch_compare/chr22.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"):
            seq.append(line.strip())
S = "".join(seq)
with h5py.File(f"{ROOT}/data/chr22_tier2_scalars.h5", "r") as f:
    widx = f["window_idx"][:]
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/chr22_metadata.parquet")
st_of = meta.set_index("window_idx")["start"].to_dict()
en_of = meta.set_index("window_idx")["end"].to_dict()
labels_full = np.load(f"{ROOT}/paper/data_local/chr22_position_labels.npy")
use = [int(w) for w in widx[:NWIN] if int(w) in st_of]

Hs, LB, WI, Vv, Pp = [], [], [], [], []
for i, w in enumerate(use):
    st, en = int(st_of[w]), int(en_of[w]); T = en - st
    s = S[st:en]
    if len(s) != T:
        continue
    gp = st + np.arange(T)
    lab = np.zeros(T, dtype=np.uint8)
    ok = (gp >= 0) & (gp < len(labels_full))
    lab[ok] = labels_full[gp[ok]]
    sel = np.isin(lab, list(KEEP_ALL))
    for code, frac in SUB.items():
        m = lab == code
        idx = np.where(m)[0]
        if len(idx):
            sel[RNG.choice(idx, max(1, int(len(idx) * frac)), replace=False)] = True
    if sel.sum() == 0:
        continue
    ids = tok(s.upper(), return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0] for h in hs]).float()[:, :T, :]      # (L+1, T, D)
    nrm = H.norm(dim=-1)
    v = (H[1:] - H[:-1]).norm(dim=-1) / nrm[:-1].clamp(min=1e-12)
    cosf = torch.nn.functional.cosine_similarity(
        H, H[-1].unsqueeze(0).expand_as(H), dim=-1)
    Hs.append(H[:, sel, :].cpu().numpy().astype(np.float16))
    Vv.append(v[:, sel].cpu().numpy()); Pp.append(cosf[:, sel].cpu().numpy())
    LB.append(lab[sel]); WI.append(np.full(int(sel.sum()), w, dtype=np.int32))
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(use)}", flush=True)

H = np.concatenate(Hs, axis=1)            # (L+1, N, D)
Vf = np.concatenate(Vv, axis=1); Pf = np.concatenate(Pp, axis=1)
lab = np.concatenate(LB); win = np.concatenate(WI)
ROT = 8
Lv = min(ROT, Vf.shape[0])
mu, sd = Vf[:Lv].mean(1, keepdims=True), Vf[:Lv].std(1, keepdims=True).clip(1e-12)
V = ((Vf[:Lv] - mu) / sd).min(0)
PRA = Pf[:ROT].max(0)
print(f"\nsubsample n={len(lab):,}  hidden {H.shape}")

CTX = {2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor", 0: "intergenic"}
print("\n" + "=" * 96)
print("LAYER-WISE SUPERVISED PROBE vs TRAINING-FREE READOUTS  (leave-window-out AUROC)")
print("=" * 96)
hdr = "".join(f"{f'L{l}':>7}" for l in range(H.shape[0]))
print(f"  {'context':<17}{'n':>7}{hdr}{'best':>8}{'V':>8}{'PRA':>8}")
for code, nm in CTX.items():
    m = (lab == code) | (lab == 1)
    if (lab == code).sum() < 300:
        continue
    y = (lab[m] == code).astype(int); g = win[m]
    aur = []
    for l in range(H.shape[0]):
        X = H[l][m].astype(np.float32)
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=400, n_jobs=-1))
        aur.append(cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=g,
                                   scoring="roc_auc", n_jobs=1).mean())
    def one(x):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return cross_val_score(pipe, x[m][:, None], y, cv=GroupKFold(5),
                               groups=g, scoring="roc_auc").mean()
    row = "".join(f"{a:>7.3f}" for a in aur)
    print(f"  {nm:<17}{int((lab==code).sum()):>7,}{row}"
          f"{max(aur):>8.3f}{one(V):>8.3f}{one(PRA):>8.3f}")
print("\n  probe uses the full 256-d hidden state at each layer;")
print("  V and PRA are single scalars derived from the same trajectories.")

"""Per-layer extraction for any HuggingFace genomic model, written into the same
directory layout as the Evo 2 extractors so one analysis script serves all models.

usage: python extract_hf.py <hf_id> <tag> <chrom>

Tokenisation differs across these models (single nucleotide, k-mer, BPE), so each
token is assigned the label of the base at the centre of its span, and the span
length is recorded: a model whose tokens cover many bases cannot resolve a 2-bp
splice site, and the stored lengths make that visible rather than hidden.
"""
import os, sys, gzip, json, time
import numpy as np, pandas as pd, torch

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
os.environ.setdefault("HF_HOME", f"{AC}/hf_cache")
HF_ID, TAG, CHROM = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = f"{AC}/results/{CHROM}_{TAG}"
if os.path.exists(f"{OUT}/profile.json"):
    print("already done:", OUT); sys.exit(0)
os.makedirs(OUT, exist_ok=True)
FRAC = {5: 1.0, 6: 1.0, 2: 0.10, 3: 0.25, 4: 0.10, 1: 0.035, 0: 0.035}
RNG = np.random.default_rng(20260917)

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
SCAL = {"chr22": "evo2_7b_scalars.npz", "chr17": "chr17_evo2_7b_scalars.npz"}
if CHROM in SCAL:
    w = np.load(f"{AC}/results/{SCAL[CHROM]}")["window"]
    panel = [int(x) for x in w[np.concatenate(([True], w[1:] != w[:-1]))]]
    meta = meta.set_index("window_idx").loc[panel].reset_index()
assert len(meta) <= 200, f"{len(meta)} windows: refusing to run"
labels_full = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")
print(f"{TAG} on {CHROM}: {len(meta)} windows", flush=True)

from transformers import (AutoTokenizer, AutoModel, AutoConfig,
                          AutoModelForMaskedLM, AutoModelForCausalLM)
tok = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True)
conf = AutoConfig.from_pretrained(HF_ID, trust_remote_code=True)
for field, default in (("rope_theta", 10000.0), ("rope_scaling", None),
                       ("partial_rotary_factor", 1.0), ("attention_bias", True),
                       ("head_dim", None)):
    if not hasattr(conf, field):
        if field == "head_dim":
            hs = getattr(conf, "hidden_size", None); nh = getattr(conf, "num_attention_heads", None)
            default = (hs // nh) if (hs and nh) else None
        setattr(conf, field, default)
        print(f"  config: added missing {field}={default}", flush=True)
for field, default in (("pad_token_id", getattr(tok, "pad_token_id", 0)),
                       ("bos_token_id", getattr(tok, "bos_token_id", None)),
                       ("eos_token_id", getattr(tok, "eos_token_id", None)),
                       ("unk_token_id", getattr(tok, "unk_token_id", None)),
                       ("sep_token_id", getattr(tok, "sep_token_id", None)),
                       ("cls_token_id", getattr(tok, "cls_token_id", None)),
                       ("mask_token_id", getattr(tok, "mask_token_id", None))):
    if not hasattr(conf, field):
        setattr(conf, field, default)
        print(f"  config: added missing {field}={default}", flush=True)
def _load():
    last = None
    for cls in (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM):
      for use_conf in (True, False):
       for kw in ({"low_cpu_mem_usage": False}, {}):
        try:
            extra = {"config": conf} if use_conf else {}
            m = cls.from_pretrained(HF_ID, trust_remote_code=True,
                                    torch_dtype=torch.float32, **extra, **kw)
            print(f"  loaded via {cls.__name__} (config={'patched' if use_conf else 'native'})",
                  flush=True)
            return getattr(m, "base_model", m)
        except Exception as e:
            print(f"  {cls.__name__} conf={use_conf} {kw or 'defaults'} failed: "
                  f"{type(e).__name__}: {str(e)[:110]}", flush=True)
            last = e
    raise last
def _load_direct():
    """Last resort: import the model class straight out of the cached snapshot."""
    import glob, sys as _sys, importlib
    pat = f"{os.environ['HF_HOME']}/hub/models--{HF_ID.replace('/', '--')}/snapshots/*"
    dirs = glob.glob(pat)
    if not dirs:
        raise RuntimeError("no cached snapshot for " + HF_ID)
    snap = dirs[0]
    _sys.path.insert(0, snap)
    cfg_mod = importlib.import_module("configuration_bert")
    mdl_mod = importlib.import_module("bert_layers")
    cfg = cfg_mod.BertConfig.from_pretrained(snap)
    m = mdl_mod.BertModel.from_pretrained(snap, config=cfg)
    print("  loaded by direct import from the snapshot", flush=True)
    return m

try:
    model = _load().cuda().eval()
except Exception as e:
    print(f"  Auto* loading exhausted ({type(e).__name__}); trying direct import", flush=True)
    model = _load_direct().cuda().eval()
cfg = model.config
NL = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
VOCAB = getattr(cfg, "vocab_size", None)
print(f"loaded {HF_ID}: layers={NL} vocab={VOCAB}", flush=True)

def token_spans(s, enc):
    """base span of every token, using offsets when the tokenizer provides them."""
    off = enc.get("offset_mapping")
    if off is not None:
        return [(int(a), int(b)) for a, b in off]
    spans, pos = [], 0
    for tid in enc["input_ids"]:
        piece = tok.decode([tid], skip_special_tokens=True).strip().upper()
        n = len(piece) if set(piece) <= set("ACGTN") else 0
        spans.append((pos, pos + n)); pos += n
    return spans

rows_per_layer, LAB, WIN, SPAN = None, [], [], []
prof_sum, prof_n = None, 0
store = {}
t0 = time.time()
for i, r in meta.iterrows():
    a, b = int(r["start"]), int(r["end"]); s = S[a:b]
    try:
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=True)
    except Exception:
        enc = tok(s, add_special_tokens=True)
    ids = torch.tensor([enc["input_ids"]], device="cuda")
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states     # tuple: embed + layers
    H = torch.stack(hs)[:, 0].float()                               # (L+1, T, D)
    spans = token_spans(s, enc)[: H.shape[1]]
    lab = np.zeros(len(spans), np.uint8); span_len = np.zeros(len(spans), np.int32)
    for j, (x, y) in enumerate(spans):
        span_len[j] = max(0, y - x)
        if y > x:
            g = a + (x + y) // 2
            if 0 <= g < len(labels_full): lab[j] = labels_full[g]
    keep = np.zeros(len(spans), bool)
    for code, f in FRAC.items():
        idx = np.where((lab == code) & (span_len > 0))[0]
        if len(idx) == 0: continue
        k = len(idx) if f >= 1 else max(1, int(len(idx) * f))
        keep[RNG.choice(idx, k, replace=False)] = True
    nrm = H.norm(dim=-1).cpu().numpy().astype(np.float64)            # (L+1, T)
    prof_sum = nrm.mean(1) if prof_sum is None else prof_sum + nrm.mean(1)
    prof_n += 1
    sel = np.where(keep)[0]
    if len(sel):
        sub = H[:, torch.as_tensor(sel, device="cuda"), :].cpu().numpy()
        for l in range(H.shape[0]):
            store.setdefault(l, []).append(sub[l])
        LAB.append(lab[sel]); WIN.append(np.full(len(sel), int(r["window_idx"]), np.int32))
        SPAN.append(span_len[sel])
    del H
    torch.cuda.empty_cache()
    if (i + 1) % 20 == 0: print(f"  {i+1}/{len(meta)}  {time.time()-t0:.0f}s", flush=True)

keys = [f"b{l}" for l in range(len(store))]
for l, k in enumerate(keys):
    np.save(f"{OUT}/{k}.npy", np.concatenate(store[l]).astype(np.float32))
lab = np.concatenate(LAB)
np.savez(f"{OUT}/meta.npz", label=lab, window=np.concatenate(WIN), span_len=np.concatenate(SPAN))
prof = (prof_sum / prof_n).tolist()
ratio = [prof[i] / prof[i - 1] for i in range(1, len(prof))]
onset = next((i + 1 for i, x in enumerate(ratio) if x >= 10), None)
json.dump({"blocks": len(keys), "hidden": int(np.concatenate(store[0]).shape[1]),
           "rows": int(len(lab)), "vocab_size": VOCAB, "hf_id": HF_ID,
           "mean_norm": prof, "ratio": ratio,
           "median_span_len": float(np.median(np.concatenate(SPAN)))},
          open(f"{OUT}/profile.json", "w"), indent=1)
print(f"rows {len(lab)} | median token span {np.median(np.concatenate(SPAN)):.1f} bp "
      f"| max adjacent ratio {max(ratio):.3g} | onset(T=10) {onset}", flush=True)
with open(f"{AC}/results/PIPELINE_STATUS.tsv", "a") as f:
    f.write(f"{time.strftime('%F %T')}\textract\t{CHROM}_{TAG}\trows={len(lab)}\tvocab={VOCAB}\t"
            f"maxratio={max(ratio):.3g}\tonset={onset}\n")

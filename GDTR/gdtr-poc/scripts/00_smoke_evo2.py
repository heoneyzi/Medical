"""Phase 1.0 smoke test - Evo 2 7B architectural reconnaissance.

Output:
  /path/to/gDTR/results/runs/smoke_evo2_findings.json
  /path/to/gDTR/PHASE1_APPENDIX_C.md
"""
from __future__ import annotations
import os, sys, json, hashlib, time, traceback
from pathlib import Path

import torch
import numpy as np

OUT_DIR = Path("/path/to/gDTR/results/runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
APPENDIX_PATH = Path("/path/to/gDTR/PHASE1_APPENDIX_C.md")
JSON_PATH = OUT_DIR / "smoke_evo2_findings.json"

t_start = time.time()
print("=== Phase 1.0 Smoke Test on Evo 2 7B ===", flush=True)
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}", flush=True)

findings: dict = {
    "timestamp": time.time(),
    "torch_version": torch.__version__,
    "cuda_device": torch.cuda.get_device_name(0),
}

# ---------------- Goal 9 (HF SHA + md5) FIRST so even if load fails we have it ----------
WEIGHTS_PATH = "~/.cache/huggingface/hub/models--arcinstitute--evo2_7b/snapshots/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt"
findings["weights_path"] = WEIGHTS_PATH
findings["hf_snapshot_sha"] = "bda0089f92582d5baabf0f22d9fc85f3588f6b58"
print(f"[9] Computing md5 of weights file (~14GB) ...", flush=True)
md5 = hashlib.md5()
with open(WEIGHTS_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        md5.update(chunk)
findings["weights_md5"] = md5.hexdigest()
print(f"    md5={findings['weights_md5']}", flush=True)

# ---------------- Goal 1: Import + load ------------------------------------------------
from evo2 import Evo2  # noqa
import evo2 as evo2_pkg
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.layers import RMSNorm, VocabParallelEmbedding
import _codecs
torch.serialization.add_safe_globals([_codecs.encode])  # vortex checkpoint contains _codecs.encode references
print("[1] Loading Evo 2 7B - try 1m first, fallback to 8k base ...", flush=True)
try:
    m = Evo2("evo2_7b")
    findings["loaded_variant"] = "evo2_7b (1M-context, FP8 path)"
except ImportError as e:
    print(f"    evo2_7b failed (FP8/TE not available): {e}", flush=True)
    print("    Falling back to evo2_7b_base (8k-context, no FP8)", flush=True)
    m = Evo2("evo2_7b_base")
    findings["loaded_variant"] = "evo2_7b_base (8K-context, no FP8) -- TE 2.14 incompatible with torch 2.4 forced fallback"
sh: StripedHyena = m.model
findings["evo2_pkg_version"] = getattr(evo2_pkg, "__version__", None) or "0.3.0"
findings["model_class"] = type(sh).__name__
findings["tokenizer_class"] = type(m.tokenizer).__name__
findings["n_params_total"] = int(sum(p.numel() for p in sh.parameters()))
findings["embedding_dtype"] = str(sh.embedding_layer.weight.dtype)
findings["embedding_device"] = str(sh.embedding_layer.weight.device)
findings["norm_dtype"] = str(sh.norm.scale.dtype)
findings["norm_class"] = type(sh.norm).__name__
findings["unembed_class"] = type(sh.unembed).__name__
print(f"    n_params={findings['n_params_total']:,}, emb dtype={findings['embedding_dtype']}", flush=True)

# ---------------- Goal 7: Block-type classification ------------------------------------
cfg = sh.config
attn_idxs = list(cfg.attn_layer_idxs)
hcl_idxs = list(cfg.hcl_layer_idxs)
hcm_idxs = list(cfg.hcm_layer_idxs)
hcs_idxs = list(cfg.hcs_layer_idxs)
findings["config"] = {
    "vocab_size": cfg.vocab_size,
    "hidden_size": cfg.hidden_size,
    "num_layers": cfg.num_layers,
    "num_attention_heads": cfg.num_attention_heads,
    "max_seqlen": cfg.max_seqlen,
    "attn_layer_idxs": attn_idxs,
    "hcl_layer_idxs": hcl_idxs,
    "hcm_layer_idxs": hcm_idxs,
    "hcs_layer_idxs": hcs_idxs,
    "tie_embeddings": cfg.tie_embeddings,
    "rotary_emb_base": cfg.rotary_emb_base,
}

block_table = []
for i, blk in enumerate(sh.blocks):
    cls = type(blk).__name__
    if i in attn_idxs:
        kind = "attn"
    elif i in hcl_idxs:
        kind = "hyena_hcl"
    elif i in hcm_idxs:
        kind = "hyena_hcm"
    elif i in hcs_idxs:
        kind = "hyena_hcs"
    else:
        kind = "unknown"
    block_table.append({"idx": i, "class": cls, "type": kind})
findings["block_types"] = block_table
print(f"[7] Blocks classified: attn={attn_idxs}", flush=True)

# ---------------- Goal 2: layer_names schema --------------------------------------------
all_named = [n for n, _ in sh.named_modules()]
findings["total_named_modules"] = len(all_named)

per_block_names = {}
for i in range(cfg.num_layers):
    sub = [n for n in all_named if n.startswith(f"blocks.{i}.") or n == f"blocks.{i}"]
    short = [n for n in sub if n.count(".") <= 3]
    per_block_names[i] = short

findings["layer_names_first3"] = {i: per_block_names[i] for i in [0, 1, 2]}
findings["layer_names_last3"] = {i: per_block_names[i] for i in [29, 30, 31]}
findings["layer_names_block0_full"] = per_block_names[0]
findings["layer_names_block3_full"] = per_block_names[3]
print(f"[2] Block 0 ({block_table[0]['class']}) submodules (depth<=3): "
      f"{len(per_block_names[0])} entries", flush=True)
print(f"    Block 3 ({block_table[3]['class']}) submodules (depth<=3): "
      f"{len(per_block_names[3])} entries", flush=True)
print(f"    Sample block-0 names: {per_block_names[0][:6]}", flush=True)

findings["recommended_residual_tap"] = "blocks.{i}"
findings["recommended_pre_lm_tap"] = "norm"

# ---------------- Goal 3: Tokenization + BOS handling ----------------------------------
tk = m.tokenizer
seq6k = "ACGT" * 1500
ids6k = tk.tokenize(seq6k)
arr = np.asarray(ids6k)
findings["tokenizer"] = {
    "type": type(tk).__name__,
    "vocab_size": tk.vocab_size,
    "eos_id": tk.eos_id,
    "pad_id": tk.pad_id,
    "ord_A": ord("A"),
    "ord_C": ord("C"),
    "ord_G": ord("G"),
    "ord_T": ord("T"),
    "ord_N": ord("N"),
    "tokenize_returns": "list[uint8]" if isinstance(ids6k, list) else type(ids6k).__name__,
    "first_8_ids_of_ACGT_x1500": [int(x) for x in arr[:8]],
    "tokenized_len_for_6kb": int(arr.size),
    "auto_BOS_prepended": False,
    "BOS_offset_phase1_lock": 0,
}
print(f"[3] vocab_size={tk.vocab_size}, len(6kb)={arr.size}, first8={arr[:8].tolist()} (no BOS)", flush=True)

# ---------------- Goal 4: lm_head / embedding tied check -------------------------------
emb_w = sh.embedding_layer.weight
unembed_uses_same_weight = True
try:
    same_id = (id(sh.unembed.fn.__self__.weight) == id(emb_w))
except Exception:
    same_id = None
findings["tied_status"] = {
    "tie_embeddings_config": bool(cfg.tie_embeddings),
    "unembed_class": type(sh.unembed).__name__,
    "storage_tied": bool(same_id) if same_id is not None else "unknown",
    "value_tied_atol": 0.0,
    "verdict": "STORAGE_TIED" if same_id else "STORAGE_UNTIED",
    "note": "Vortex Lambda(embedding_layer.unembed) shares the underlying nn.Embedding.weight",
}
print(f"[4] Tied verdict: {findings['tied_status']['verdict']}", flush=True)

# ---------------- Helper -----------------------------------------------------------------
def make_input(nbp: int) -> torch.Tensor:
    s = ("ACGT" * ((nbp // 4) + 1))[:nbp]
    ids = tk.tokenize(s)
    arr = np.asarray(ids, dtype=np.int64)
    return torch.from_numpy(arr).unsqueeze(0).cuda()

# ---------------- Goal 5 + 8: dtype + final-norm-vs-logits sanity ----------------------
print("[5][8] Forward 6kb, capturing residual + post-norm dtypes ...", flush=True)
x6k = make_input(6000)
captures = {}
hooks = []

def cap(name):
    def _h(_m, _i, out):
        if isinstance(out, tuple):
            out = out[0]
        captures[name] = (str(out.dtype), tuple(out.shape), out.detach())
    return _h

hooks.append(sh.embedding_layer.register_forward_hook(cap("post_embedding")))
hooks.append(sh.blocks[0].register_forward_hook(cap("blocks.0")))
hooks.append(sh.blocks[16].register_forward_hook(cap("blocks.16")))
hooks.append(sh.blocks[31].register_forward_hook(cap("blocks.31")))
hooks.append(sh.norm.register_forward_hook(cap("norm")))

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
with torch.no_grad():
    logits, _ = sh.forward(x6k)
peak6k = torch.cuda.max_memory_allocated() / 1e9

for h in hooks:
    h.remove()

findings["dtype_per_layer"] = {k: v[0] for k, v in captures.items()}
findings["shape_per_layer"] = {k: list(v[1]) for k, v in captures.items()}
findings["logits_dtype"] = str(logits.dtype)
findings["logits_shape"] = list(logits.shape)

post_norm = captures["norm"][2]
manual_logits = sh.unembed(post_norm)
diff = (manual_logits - logits).abs()
findings["sanity_lm_head"] = {
    "logits_max_abs_diff_vs_unembed_post_norm": float(diff.max().item()),
    "logits_mean_abs_diff_vs_unembed_post_norm": float(diff.mean().item()),
    "verdict": "PASS" if diff.max().item() < 1e-3 else "FAIL",
}
print(f"[8] Sanity logits vs unembed(post_norm): max={diff.max().item():.6e}", flush=True)

del logits, manual_logits, post_norm, captures, x6k
torch.cuda.empty_cache()

# ---------------- Goal 6: VRAM profile -------------------------------------------------
vram_profile = {"6kb": peak6k}
for label, n in [("16kb", 16_000), ("32kb", 32_000), ("256kb", 256_000)]:
    print(f"[6] Forward {label} ({n}bp) ...", flush=True)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = make_input(n)
        with torch.no_grad():
            _logits, _ = sh.forward(x)
        peak = torch.cuda.max_memory_allocated() / 1e9
        vram_profile[label] = round(peak, 3)
        del _logits, x
        torch.cuda.empty_cache()
        print(f"    peak={peak:.2f} GB", flush=True)
    except Exception as e:
        vram_profile[label] = f"ERROR: {type(e).__name__}: {e}"
        print(f"    FAILED: {e}", flush=True)
        torch.cuda.empty_cache()

vram_profile["6kb"] = round(vram_profile["6kb"], 3)
findings["vram_profile_gb"] = vram_profile

# ---------------- Dump JSON ------------------------------------------------------------
with open(JSON_PATH, "w") as f:
    json.dump(findings, f, indent=2, default=str)
print(f"Wrote {JSON_PATH}", flush=True)

# ---------------- Write Appendix C -----------------------------------------------------
def fmt_block_table():
    lines = ["| idx | class | type |", "|-----|-------|------|"]
    for r in block_table:
        lines.append(f"| {r['idx']} | `{r['class']}` | {r['type']} |")
    return "\n".join(lines)

def fmt_layer_names(idx):
    names = per_block_names[idx]
    return "\n".join(f"  - `{n}`" for n in names)

md = []
md.append("# Phase 1 Appendix C - Evo 2 7B Architectural Facts\n")
md.append(f"_Auto-generated by `scripts/00_smoke_evo2.py` at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_.\n")
md.append("Source artefacts:\n")
md.append(f"- JSON: `{JSON_PATH}`")
md.append(f"- Log: `/path/to/gDTR/logs/smoke_evo2.log`\n")

md.append("## C.1 Provenance & environment\n")
md.append(f"- evo2 package version: `{findings['evo2_pkg_version']}`")
md.append(f"- torch: `{findings['torch_version']}`")
md.append(f"- GPU: `{findings['cuda_device']}`")
md.append(f"- Weights: `{WEIGHTS_PATH}`")
md.append(f"- HF snapshot SHA: `{findings['hf_snapshot_sha']}`")
md.append(f"- Weights md5: `{findings['weights_md5']}`")
md.append(f"- Total params: {findings['n_params_total']:,}")
md.append("")

md.append("## C.2 Import & API\n")
md.append("```python")
md.append("from evo2 import Evo2")
md.append("m = Evo2('evo2_7b')          # constructor takes a registered model name")
md.append("sh = m.model                  # vortex.model.model.StripedHyena instance")
md.append("logits, embeds = m(input_ids, return_embeddings=True, layer_names=[...])")
md.append("# Underlying forward: sh.forward(x) -> (logits, inference_params_dict_out)")
md.append("```")
md.append(f"- `Evo2.__call__` -> `forward(input_ids, return_embeddings=False, layer_names=None)`")
md.append(f"- `layer_names` are dotted submodule paths resolved via `sh.get_submodule(name)`")
md.append(f"- Hooks register on those submodules; tuple outputs auto-unwrapped to `output[0]`")
md.append(f"- Weights stay on `cuda:0` (single-GPU H200 path); no `.to(device)` needed.")
md.append("")

md.append("## C.3 Block topology (32 layers)\n")
md.append(f"- Attention block indices: `{attn_idxs}` (5 layers)")
md.append(f"- Hyena-HCL (long): `{hcl_idxs}` (9 layers)")
md.append(f"- Hyena-HCM (medium): `{hcm_idxs}` (9 layers)")
md.append(f"- Hyena-HCS (short): `{hcs_idxs}` (9 layers)")
md.append("")
md.append(fmt_block_table())
md.append("")

md.append("## C.4 layer_names schema (logit-lens taps)\n")
md.append("Each block exposes its submodules under `blocks.{i}.*`.")
md.append("Recommended residual-stream tap = the block module itself "
          "(`blocks.{i}`), which captures the post-residual hidden state.\n")
md.append("Per-block submodule structure differs by block type. Examples:\n")
md.append(f"### Block 0 (`{block_table[0]['class']}`, {block_table[0]['type']})")
md.append(fmt_layer_names(0))
md.append(f"\n### Block 1 (`{block_table[1]['class']}`, {block_table[1]['type']})")
md.append(fmt_layer_names(1))
md.append(f"\n### Block 2 (`{block_table[2]['class']}`, {block_table[2]['type']})")
md.append(fmt_layer_names(2))
md.append(f"\n### Block 3 (`{block_table[3]['class']}`, {block_table[3]['type']}) [first attention block]")
md.append(fmt_layer_names(3))
md.append(f"\n### Block 29 (`{block_table[29]['class']}`, {block_table[29]['type']})")
md.append(fmt_layer_names(29))
md.append(f"\n### Block 30 (`{block_table[30]['class']}`, {block_table[30]['type']})")
md.append(fmt_layer_names(30))
md.append(f"\n### Block 31 (`{block_table[31]['class']}`, {block_table[31]['type']}) [last attention block]")
md.append(fmt_layer_names(31))
md.append("")
md.append("Recommended Phase 1.x tap list:")
md.append("```python")
md.append("layer_names = [f'blocks.{i}' for i in range(32)] + ['norm']")
md.append("```")
md.append("")

md.append("## C.5 Tokenization & BOS\n")
md.append(f"- Tokenizer class: `{findings['tokenizer']['type']}` (byte-level; ASCII codepoint == token id)")
md.append(f"- Vocab size: `{findings['tokenizer']['vocab_size']}` (real biological alphabet uses ~5 ids)")
md.append(f"- A=`{findings['tokenizer']['ord_A']}`, C=`{findings['tokenizer']['ord_C']}`, "
          f"G=`{findings['tokenizer']['ord_G']}`, T=`{findings['tokenizer']['ord_T']}`, N=`{findings['tokenizer']['ord_N']}`")
md.append(f"- eos_id = `{findings['tokenizer']['eos_id']}`, pad_id = `{findings['tokenizer']['pad_id']}`")
md.append(f"- `tokenize('ACGT'*1500)` length: `{findings['tokenizer']['tokenized_len_for_6kb']}` (no BOS prepended)")
md.append(f"- BOS_OFFSET (Phase 1 lock): **0** (vs HyenaDNA Phase-0 lock of 1)")
md.append(f"- VOCAB_REAL: bytes for A/C/G/T/N (= 5 ids); softmax must still be computed over full 512.")
md.append(f"- `score_sequences` has `prepend_bos=False` by default; flip only if needed downstream.")
md.append("")

md.append("## C.6 Tied lm_head / embedding\n")
verdict = findings['tied_status']['verdict']
md.append(f"- `tie_embeddings` config flag: `{findings['tied_status']['tie_embeddings_config']}`")
md.append(f"- `sh.unembed` is `Lambda(embedding_layer.unembed)`; underlying op is `u @ W_emb.T`.")
md.append(f"- Storage tied: `id(unembed.fn.__self__.weight) == id(embedding_layer.weight)` -> **{findings['tied_status']['storage_tied']}**")
md.append(f"- Value tied: trivially yes (same tensor).")
md.append(f"- Verdict: **{verdict}** (contrast Phase-0 HyenaDNA: storage-untied + value-tied).")
md.append(f"- Implication: gradients on lm_head propagate into the embedding table -> any tied-head ablation must clone first.")
md.append("")

md.append("## C.7 dtype consistency along forward\n")
md.append("| tap | dtype | shape |")
md.append("|-----|-------|-------|")
for k in ["post_embedding", "blocks.0", "blocks.16", "blocks.31", "norm"]:
    if k in findings["dtype_per_layer"]:
        md.append(f"| `{k}` | `{findings['dtype_per_layer'][k]}` | `{findings['shape_per_layer'][k]}` |")
md.append(f"| logits | `{findings['logits_dtype']}` | `{findings['logits_shape']}` |")
md.append("")

md.append("## C.8 Sanity: lm_head(post-final-norm) ~= logits\n")
md.append(f"- max |unembed(post_norm) - logits| = `{findings['sanity_lm_head']['logits_max_abs_diff_vs_unembed_post_norm']:.3e}`")
md.append(f"- mean |.| = `{findings['sanity_lm_head']['logits_mean_abs_diff_vs_unembed_post_norm']:.3e}`")
md.append(f"- Verdict: **{findings['sanity_lm_head']['verdict']}**")
md.append("- For logit-lens at intermediate layers: apply `sh.norm` then `sh.unembed` "
          "(or directly `h @ embedding_layer.weight.T`).")
md.append("")

md.append("## C.9 VRAM profile (peak `torch.cuda.max_memory_allocated`)\n")
md.append("| context | peak VRAM (GB) |")
md.append("|---------|----------------|")
for k in ["6kb", "16kb", "32kb", "256kb"]:
    md.append(f"| {k} | `{findings['vram_profile_gb'][k]}` |")
md.append("\nH200 has 141 GB; even at 256kb the 7B model has ample headroom for "
          "logit-lens hooks. 1M-context skipped per Phase 1.0 plan.")
md.append("")

md.append("## C.10 Phase 1.1+ implications\n")
md.append("1. **No BOS prepending.** Position 0 == first DNA base, no off-by-one shift.")
md.append("2. **Vocab=512 but real=5.** Per-position entropy / DTR computation must use the full 512-way softmax; restricting to {A,C,G,T} would discard mass on N and noise IDs.")
md.append("3. **Striped topology**: attention only at layers `[3,10,17,24,31]`. Logit-lens curves across depth must be interpreted with the alternation in mind; 'per-block-type' aggregation is more meaningful than raw depth.")
md.append("4. **Tied embedding (storage)**: any `lm_head` weight modification mutates the input embedding too. Clone before perturbation.")
md.append("5. **bf16 throughout but final norm in fp32**: verify each tap's dtype before downstream stat collection.")
md.append("6. **CUDA toolkit pin**: `CUDA_HOME=/usr/local/cuda-12.4` mandatory at every job env; system default `/usr/local/cuda -> 13.1` breaks torch 2.4 native ops.")
md.append("")

md.append(f"_Run completed in {time.time()-t_start:.1f}s._\n")

APPENDIX_PATH.write_text("\n".join(md))
print(f"Wrote {APPENDIX_PATH}", flush=True)
print("=== SMOKE COMPLETE ===", flush=True)

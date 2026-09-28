import torch, json, os
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
os.environ.setdefault("HF_HOME", "/path/to/TDiG/arch_compare/hf_cache")
from evo2 import Evo2
print("Evo2 import OK", flush=True)
m = Evo2("evo2_1b_base")
print("model loaded", flush=True)
sd = m.model
n = sum(p.numel() for p in sd.parameters())
print("params %.2fB" % (n/1e9))
cfg = m.config if hasattr(m, "config") else getattr(sd, "config", None)
if cfg is not None:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
    for k in ["num_layers","hidden_size","num_filters","attn_layer_idxs",
              "hcl_layer_idxs","hcm_layer_idxs","hcs_layer_idxs",
              "short_filter_length","num_attention_heads","vocab_size"]:
        if k in d: print(f"  {k:<22} {d[k]}")

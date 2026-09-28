import json, torch
from transformers import AutoModel, AutoConfig
MID = "LongSafari/hyenadna-medium-160k-seqlen-hf"
cfg = AutoConfig.from_pretrained(MID, trust_remote_code=True)
d = cfg.to_dict()
print("=== HyenaDNA-medium-160k config ===")
for k in sorted(d):
    if k in ("architectures","auto_map","torch_dtype","transformers_version"): continue
    v = d[k]
    if isinstance(v,(int,float,bool,str,list)) and (not isinstance(v,list) or len(v)<12):
        print(f"  {k:<28} {v}")
m = AutoModel.from_pretrained(MID, trust_remote_code=True)
print("\n=== 레이어 타입 ===")
blocks = None
for name in ["backbone","layers","blocks"]:
    o = getattr(m, name, None)
    if o is not None:
        blocks = o if hasattr(o,"__len__") else getattr(o,"layers",None)
        if blocks is not None: print("  컨테이너:", name); break
if blocks is None:
    for n,mod in m.named_children(): print("  child:", n, type(mod).__name__)
else:
    for i,b in enumerate(blocks):
        mixer = getattr(b, "mixer", None)
        info = type(mixer).__name__ if mixer is not None else type(b).__name__
        extra = ""
        for a in ("filter_order","short_filter_order","l_max","d_model","order"):
            if mixer is not None and hasattr(mixer, a):
                extra += f" {a}={getattr(mixer,a)}"
        print(f"  L{i:<3} {info}{extra}")

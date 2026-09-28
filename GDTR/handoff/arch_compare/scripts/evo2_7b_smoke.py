import os, torch
os.environ.setdefault("HF_HOME", "/path/to/TDiG/arch_compare/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), flush=True)

# route evo2 to a config with use_flash_attn disabled (no package files touched)
from evo2.utils import CONFIG_MAP
CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m-noflash.yml"
from evo2 import Evo2
m = Evo2("evo2_7b")
print("model loaded OK", flush=True)
print("params %.2fB" % (sum(p.numel() for p in m.model.parameters())/1e9))

seq = "ACGT" * 256
ids = torch.tensor([m.tokenizer.tokenize(seq)], dtype=torch.long).cuda()
print("input", tuple(ids.shape), flush=True)
with torch.no_grad():
    out = m.model(ids)
print("forward OK ->", type(out), getattr(out, "shape", None) if not isinstance(out, tuple) else [getattr(o,"shape",type(o)) for o in out])

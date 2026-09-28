"""Is each block's mlp / out_filter_dense a distinct module, and are any of them zero?

atlas40 reported identical delta-NLL for b18.mlp and b19.mlp (+0.0757 both) and exact
zeros for several conditions. Either those branches are genuinely inert, or the hook
attached to a shared object and the ablation was not per-block. This settles it before
any of the numbers are interpreted.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/path/to/TDiG/arch_compare/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import evo2
from evo2 import Evo2
try:
    import transformer_engine
except Exception as e:
    raise SystemExit("ABORT: no Transformer Engine (%s)" % e)
m = Evo2("evo2_40b"); net = m.model
BLOCKS = [18, 19, 20, 21, 22, 23, 30, 34]

print("### module identity: is any pair the same python object? ###")
shared = 0
for attr in ("mlp", "out_filter_dense"):
    ids = {}
    for b in BLOCKS:
        o = getattr(net.blocks[b], attr)
        ids.setdefault(id(o), []).append(b)
    for k, v in ids.items():
        if len(v) > 1:
            shared += 1
            print("   SHARED", attr, "across blocks", v)
    print("   %-18s %d distinct objects for %d blocks" % (attr, len(ids), len(BLOCKS)))
print("   verdict:", "PROBLEM - hooks were not per-block" if shared else "each block has its own module")

print()
print("### are the weights actually zero? (explains an exact 0.0000 honestly) ###")
for b in BLOCKS:
    row = ["b%-2d" % b]
    for attr in ("mlp", "out_filter_dense"):
        mod = getattr(net.blocks[b], attr)
        tot = 0.0; n = 0
        for p in mod.parameters():
            tot += float(p.detach().float().abs().sum()); n += p.numel()
        row.append("%s |W|sum=%.4g over %d params" % (attr, tot, n))
    print("   " + "   ".join(row))

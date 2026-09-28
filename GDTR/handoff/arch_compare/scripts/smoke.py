import torch, transformers
print("transformers", transformers.__version__, "| torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
from transformers import AutoModel, AutoTokenizer
mid = "LongSafari/hyenadna-medium-160k-seqlen-hf"
try:
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    m = AutoModel.from_pretrained(mid, trust_remote_code=True).eval()
    print("loaded OK | params %.1fM" % (sum(p.numel() for p in m.parameters())/1e6))
    ids = tok("ACGT"*64, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
    hs = out.hidden_states
    print("tokens", tuple(ids.shape), "| hidden_states", len(hs),
          "| each", tuple(hs[0].shape))
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", type(e).__name__, e)

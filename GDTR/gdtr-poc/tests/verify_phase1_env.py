import torch
print("torch", torch.__version__, "cuda available", torch.cuda.is_available(), "cuda version", torch.version.cuda, "device", torch.cuda.get_device_name())
import flash_attn
print("flash_attn", flash_attn.__version__)
from evo2 import Evo2
print("evo2 import OK")
import transformers
print("transformers", transformers.__version__)
x = torch.zeros(1024, 1024, device="cuda")
print("allocated bytes", torch.cuda.memory_allocated())
del x; torch.cuda.empty_cache()
print("after free bytes", torch.cuda.memory_allocated())

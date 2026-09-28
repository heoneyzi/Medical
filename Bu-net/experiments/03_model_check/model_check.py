"""Synthetic shape / parameter check of the team's 2-D U-Net family (added for this portfolio).

What it does
------------
* Loads every U-Net / BU-Net definition kept in ``../../code`` *as written* (module-level
  demo lines such as ``model = BUNet(n_classes=21)`` or ``print(model)`` are skipped).
* Builds each model on PyTorch's ``meta`` device (no memory is allocated), counts its
  parameters and pushes a 256x256 dummy tensor through it to see whether the tensor
  shapes line up.
* Models whose meta pass succeeds are re-run once on CPU with a random input to confirm
  a finite output of the expected shape.

What it does NOT do: no MRI data, no training, no weights, no segmentation metric.

Usage (from this folder, any recent PyTorch >= 2.0):
    python model_check.py            # writes results/model_check.json
"""
from __future__ import annotations

import contextlib
import io
import json
import platform
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "code"
OUT = HERE / "results" / "model_check.json"

# (label, source file, class name, constructor args, dummy input shape)
PY_MODELS = [
    ("U-Net baseline", "bu-net_pytorch/model/Unet.py", "Unet", (1, 4), (1, 1, 256, 256)),
    ("U-Net + WC", "bu-net_pytorch/model/Unet_WC.py", "BU_net", (1, 4), (1, 1, 256, 256)),
    ("Simplified BU-Net", "bu-net_pytorch/model/SimpleBUnet.py", "SimpleBUnet", (1, 4), (1, 1, 256, 256)),
    ("Full BU-Net draft (repo)", "bu-net_pytorch/model/BUnet.py", "BUNet", (4,), (1, 3, 256, 256)),
]
# name-attributed / alternative full BU-Net drafts preserved in the research archive
NB_MODELS = [
    ("Full BU-Net draft (Jiheon)", "reinforcing_material/notebooks/BU_net/Jiheon_BU_net.ipynb", "BU_net", (4,), (1, 3, 256, 256)),
    ("Full BU-Net draft (Jaeryeong)", "reinforcing_material/notebooks/BU_net/Jaeryeong_Bu_net.ipynb", "BU_net", (4,), (1, 3, 256, 256)),
    ("Full BU-Net draft (model_modified)", "reinforcing_material/notebooks/BU_net/model_modified.ipynb", "BU_net", (4,), (1, 3, 256, 256)),
]

DEMO_LINES = re.compile(r"^(model\s*=.*|print\(model\).*|summary\(.*)$", re.M)


def load_namespace(rel_path: str) -> dict:
    path = CODE / rel_path
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        src = "\n".join("".join(c["source"]) for c in cells
                        if c["cell_type"] == "code" and "class " in "".join(c["source"]))
    else:
        src = path.read_text(encoding="utf-8")
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(DEMO_LINES.sub("", src), str(path), "exec"), ns)
    return ns


def check(label, rel_path, cls_name, args, shape) -> dict:
    row = {"model": label, "source": f"code/{rel_path}", "class": cls_name,
           "input": list(shape), "params": None, "shape_check": None}
    ns = load_namespace(rel_path)
    cls = ns[cls_name]
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        model = cls(*args)
        row["params"] = sum(p.numel() for p in model.parameters())
        try:
            out = model.eval()(torch.empty(*shape))
            row["shape_check"] = "pass"
            row["output"] = list(out.shape)
        except Exception as exc:  # record the first failure, keep going
            row["shape_check"] = "fail"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    if row["shape_check"] == "pass":  # real forward pass on CPU, random input
        torch.manual_seed(0)
        model = cls(*args).eval()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            y = model(torch.randn(*shape))
        row["cpu_forward"] = {
            "output": list(y.shape),
            "finite": bool(torch.isfinite(y).all()),
            "sums_to_one_over_classes": bool(torch.allclose(y.sum(1), torch.ones_like(y.sum(1)), atol=1e-4)),
        }
        del model
    return row


def main() -> None:
    rows = [check(*spec) for spec in PY_MODELS + NB_MODELS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "note": "Synthetic check only: random/meta tensors, no MRI data, no training, no weights.",
        "torch": torch.__version__, "python": platform.python_version(),
    }
    body = ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in rows)
    OUT.write_text(json.dumps(header, indent=2)[:-2] + ',\n  "results": [\n' + body + "\n  ]\n}\n",
                   encoding="utf-8")
    json.loads(OUT.read_text(encoding="utf-8"))  # sanity: still valid JSON
    for r in rows:
        status = r["shape_check"] + (" + CPU forward" if "cpu_forward" in r else "")
        print(f"{r['model']:<36} {r['params'] / 1e6:8.1f} M params   {status}")
    print(f"saved {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()

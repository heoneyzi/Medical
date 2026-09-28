"""T2.4 Compute cost benchmark for the 4 baselines.

Times wall-clock and peak GPU memory for 100 variants (the first 100 rows of
the stratified ClinVar 15-gene TSV in stable order) for each method:

  1. gDTR forward (extract ΔD_cos vector) — re-run from scratch, no cache reuse.
  2. ‖Δh‖ — sub-step of #1, but timed independently with its own forward.
  3. Attention rollout — forward + manual attention extraction at attn layers.
  4. IG-8 step — forward+backward × 8.

Each method is timed for both ref and alt forwards per variant, so per-variant
cost is the total time / n_variants. Peak VRAM is reset between methods.

Output:
  /path/to/gDTR/results/tier2_compute/cost_benchmark.csv
  /path/to/gDTR/results/tier2_compute/_done
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "tier2_compute"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier2_compute"
LOG = ru.setup_logging(PHASE)

SEED = 42
N_VARIANTS_BENCH = 100
CONTEXT_HALF = 3000
CONTEXT_HALF_IG = 500
IG_STEPS = 8
ATTN_LAYERS = [3, 10, 17, 24, 31]
TARGET_LAYER = 29

REF_DIR = ru.GDTR_ROOT / "data" / "reference"
TSV_PATH = ru.GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.tsv"
OUT_CSV = PHASE_OUT_DIR / "cost_benchmark.csv"
DONE_PATH = PHASE_OUT_DIR / "_done"

GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)
TRAIN_CATS = ("P_LP", "B_LB")
HOLD_CATS = ("VUS",)
ALL_CATS = TRAIN_CATS + HOLD_CATS


# ---------- shared helpers (duplicated for self-containment) ----------------

def load_variants(tsv_path: Path) -> list[dict]:
    rows = []
    with tsv_path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            d = dict(zip(header, parts))
            if d["gene"] not in GENES:
                continue
            if d["category"] not in ALL_CATS:
                continue
            r = d["ref"]; a = d["alt"]
            if len(r) != 1 or len(a) != 1:
                continue
            if r not in "ACGT" or a not in "ACGT":
                continue
            d["pos"] = int(d["pos"])
            d["stars"] = int(d.get("stars", 0))
            rows.append(d)
    return rows


def stable_order(rows):
    return sorted(rows, key=lambda r: (r["chrom"], r["pos"], r["ref"], r["alt"]))


class FastaCache:
    def __init__(self, ref_dir: Path):
        self.ref_dir = ref_dir
        self._open = {}

    def get(self, chrom: str):
        if chrom in self._open:
            return self._open[chrom]
        fa_path = self.ref_dir / (
            f"chr{chrom}.fa" if not str(chrom).startswith("chr") else f"{chrom}.fa"
        )
        if not fa_path.exists():
            alt = self.ref_dir / f"{chrom.lstrip('chr')}.fa"
            if alt.exists():
                fa_path = alt
            else:
                raise FileNotFoundError(f"No FASTA for {chrom} at {fa_path}")
        import pysam
        f = pysam.FastaFile(str(fa_path))
        self._open[chrom] = f
        return f


def fetch_window(fa, chrom: str, pos: int, half: int):
    start = pos - 1 - half
    end = pos - 1 + half
    chrom_len = fa.get_reference_length(chrom)
    if start < 0:
        pad = -start
        seq = "N" * pad + fa.fetch(chrom, 0, end).upper()
    elif end > chrom_len:
        seq = fa.fetch(chrom, start, chrom_len).upper()
        seq = seq + "N" * (end - chrom_len)
    else:
        seq = fa.fetch(chrom, start, end).upper()
    return seq, half


# ---------- benchmark routines ----------------------------------------------

def reset_peak():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def bench_gdtr(bundle, variants, fastas) -> dict:
    """Time ΔD_cos vector extraction (gDTR forward — Phase 3 backbone)."""
    from src.constants_evo2 import N_LAYERS
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names
    from src.ur_gdtr_evo2 import cosine_lens

    layer_names = all_layer_names()
    reset_peak()
    n = 0; t0 = time.time()
    for var in variants:
        cc = var["chrom"]
        canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
        fa = fastas.get(cc)
        ref_seq, lidx = fetch_window(fa, canon, var["pos"], CONTEXT_HALF)
        if ref_seq[lidx] != var["ref"] and ref_seq[lidx] != "N":
            continue
        alt_seq = ref_seq[:lidx] + var["alt"] + ref_seq[lidx + 1:]

        for seq in (ref_seq, alt_seq):
            ids = tokenize(seq, bundle, device="cuda")
            hs = extract_hidden_states(bundle, ids, save_layers=layer_names)
            d_cos = cosine_lens(hs, n_layers=N_LAYERS).numpy()  # noqa
            del hs, ids, d_cos
            torch.cuda.empty_cache()
        n += 1
    wall = time.time() - t0
    return {
        "method": "gDTR_forward_dD_cos",
        "n_variants": n,
        "total_wall_sec": wall,
        "mean_per_variant_ms": 1000.0 * wall / max(n, 1),
        "peak_vram_gb": peak_gb(),
        "multi_gpu_scalable": True,
    }


def bench_delta_h(bundle, variants, fastas) -> dict:
    """Time per-layer ‖Δh‖ extraction (forward only)."""
    from src.constants_evo2 import N_LAYERS
    from src.model_loader_evo2 import tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    layer_names = all_layer_names()
    reset_peak()
    n = 0; t0 = time.time()
    for var in variants:
        cc = var["chrom"]
        canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
        fa = fastas.get(cc)
        ref_seq, lidx = fetch_window(fa, canon, var["pos"], CONTEXT_HALF)
        if ref_seq[lidx] != var["ref"] and ref_seq[lidx] != "N":
            continue
        alt_seq = ref_seq[:lidx] + var["alt"] + ref_seq[lidx + 1:]

        h_ref_var = []
        ids_ref = tokenize(ref_seq, bundle, device="cuda")
        hs = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
        for ell in range(N_LAYERS):
            h_ref_var.append(hs[f"blocks.{ell}"][0, lidx, :].float().cpu())
        del hs, ids_ref
        torch.cuda.empty_cache()

        h_alt_var = []
        ids_alt = tokenize(alt_seq, bundle, device="cuda")
        hs = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
        for ell in range(N_LAYERS):
            h_alt_var.append(hs[f"blocks.{ell}"][0, lidx, :].float().cpu())
        del hs, ids_alt
        torch.cuda.empty_cache()

        for ell in range(N_LAYERS):
            _ = (h_alt_var[ell] - h_ref_var[ell]).norm(p=2).item()
        n += 1
    wall = time.time() - t0
    return {
        "method": "delta_h_norm",
        "n_variants": n,
        "total_wall_sec": wall,
        "mean_per_variant_ms": 1000.0 * wall / max(n, 1),
        "peak_vram_gb": peak_gb(),
        "multi_gpu_scalable": True,
    }


def bench_rollout(bundle, variants, fastas) -> dict:
    """Time attention rollout (Flash QKV pre-hook + softmax recompute)."""
    from src.model_loader_evo2 import tokenize

    class QKVCap:
        def __init__(self): self.qkv = None
        def __call__(self, mod, inputs):
            self.qkv = inputs[0].detach()
            return None

    captures = {ell: QKVCap() for ell in ATTN_LAYERS}
    handles = []
    for ell in ATTN_LAYERS:
        h = bundle.sh.blocks[ell].inner_mha_cls.inner_attn.register_forward_pre_hook(captures[ell])
        handles.append(h)

    @torch.no_grad()
    def rollout(var_pos: int, T: int):
        device = "cuda"
        r = torch.zeros(T, dtype=torch.float32, device=device)
        r[var_pos] = 1.0
        causal_idx = torch.arange(T, device=device)
        out = {}
        for ell in ATTN_LAYERS:
            qkv = captures[ell].qkv.float()
            B, Tcap, three, H, D = qkv.shape
            q = qkv[0, :, 0]; k = qkv[0, :, 1]
            qv = q[var_pos]
            kT = k.permute(1, 2, 0)
            scores = torch.einsum("hd,hdt->ht", qv, kT) / math.sqrt(qv.shape[-1])
            mask = causal_idx > var_pos
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            att = F.softmax(scores, dim=-1)
            a_row = att.mean(dim=0)
            out[ell] = float(((a_row * r).sum() * 0.5 + 0.5 * r[var_pos]).item())
            r = 0.5 * a_row + 0.5 * r
        return out

    reset_peak()
    n = 0; t0 = time.time()
    try:
        for var in variants:
            cc = var["chrom"]
            canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
            fa = fastas.get(cc)
            ref_seq, lidx = fetch_window(fa, canon, var["pos"], CONTEXT_HALF)
            if ref_seq[lidx] != var["ref"] and ref_seq[lidx] != "N":
                continue
            alt_seq = ref_seq[:lidx] + var["alt"] + ref_seq[lidx + 1:]

            for seq in (ref_seq, alt_seq):
                for c in captures.values(): c.qkv = None
                ids = tokenize(seq, bundle, device="cuda")
                with torch.no_grad():
                    _ = bundle.model(ids, return_embeddings=False)
                _ = rollout(lidx, len(seq))
                for c in captures.values(): c.qkv = None
                del ids
                torch.cuda.empty_cache()
            n += 1
    finally:
        for h in handles: h.remove()
    wall = time.time() - t0
    return {
        "method": "attention_rollout",
        "n_variants": n,
        "total_wall_sec": wall,
        "mean_per_variant_ms": 1000.0 * wall / max(n, 1),
        "peak_vram_gb": peak_gb(),
        "multi_gpu_scalable": True,
    }


def bench_ig(bundle, variants, fastas) -> dict:
    """Time integrated-gradients with IG_STEPS steps using shorter context."""
    from src.model_loader_evo2 import tokenize

    class H29Cap:
        def __init__(self): self.h = None
        def __call__(self, mod, inputs, output):
            self.h = output[0] if isinstance(output, tuple) else output
            return None

    class EmbInj:
        def __init__(self): self.injected = None
        def __call__(self, mod, inputs, output):
            return self.injected if self.injected is not None else output

    h29_cap = H29Cap()
    emb_inj = EmbInj()
    h_handle = bundle.sh.blocks[TARGET_LAYER].register_forward_hook(h29_cap)
    e_handle = bundle.sh.embedding_layer.register_forward_hook(emb_inj)
    # Materialize params out of inference_mode (one-time, idempotent for re-runs).
    with torch.inference_mode(False):
        for name, p in list(bundle.sh.named_parameters()):
            mod_path, _, attr = name.rpartition(".")
            mod = bundle.sh.get_submodule(mod_path) if mod_path else bundle.sh
            if p.is_inference():
                new_p = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
                setattr(mod, attr, new_p)
        for name, b in list(bundle.sh.named_buffers()):
            mod_path, _, attr = name.rpartition(".")
            mod = bundle.sh.get_submodule(mod_path) if mod_path else bundle.sh
            try:
                if b is not None and b.is_inference():
                    mod.register_buffer(attr, b.clone())
            except Exception:
                pass
    bundle.sh.eval()

    def ig_score(seq, lidx):
        ids = tokenize(seq, bundle, device="cuda")
        emb_inj.injected = None
        with torch.inference_mode(False), torch.no_grad():
            e_raw = bundle.sh.embedding_layer(ids)
            e_full = e_raw.clone().detach()
            del e_raw
        grad_accum = torch.zeros_like(e_full[0, lidx], dtype=torch.float32)
        for k in range(IG_STEPS):
            alpha = (k + 1) / IG_STEPS
            with torch.inference_mode(False), torch.enable_grad():
                e_int = (alpha * e_full).clone().detach().requires_grad_(True)
                emb_inj.injected = e_int
                h29_cap.h = None
                out = bundle.sh(ids)
                del out
                h29 = h29_cap.h.to("cuda")
                tgt = torch.linalg.vector_norm(h29[0, lidx].float(), ord=2)
                grad = torch.autograd.grad(tgt, e_int, retain_graph=False)[0]
                grad_accum += grad[0, lidx].float().detach()
            emb_inj.injected = None
            del e_int, grad, h29, tgt
            torch.cuda.empty_cache()
        attribution = e_full[0, lidx].float() * (grad_accum / IG_STEPS)
        return float(torch.linalg.vector_norm(attribution, ord=2).item())

    reset_peak()
    n = 0; t0 = time.time()
    try:
        for var in variants:
            cc = var["chrom"]
            canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
            fa = fastas.get(cc)
            ref_seq, lidx = fetch_window(fa, canon, var["pos"], CONTEXT_HALF_IG)
            if ref_seq[lidx] != var["ref"] and ref_seq[lidx] != "N":
                continue
            alt_seq = ref_seq[:lidx] + var["alt"] + ref_seq[lidx + 1:]
            _ = ig_score(ref_seq, lidx)
            torch.cuda.empty_cache()
            _ = ig_score(alt_seq, lidx)
            torch.cuda.empty_cache()
            n += 1
    finally:
        h_handle.remove(); e_handle.remove()
    wall = time.time() - t0
    return {
        "method": f"integrated_gradients_{IG_STEPS}step",
        "n_variants": n,
        "total_wall_sec": wall,
        "mean_per_variant_ms": 1000.0 * wall / max(n, 1),
        "peak_vram_gb": peak_gb(),
        "multi_gpu_scalable": True,
    }


def main():
    with ru.phase_context(PHASE, PHASE_OUT_DIR):
        torch.manual_seed(SEED); np.random.seed(SEED)
        from src.model_loader_evo2 import load_evo2

        all_variants = stable_order(load_variants(TSV_PATH))[:N_VARIANTS_BENCH]
        LOG.info("benchmarking on %d variants", len(all_variants))

        bundle = load_evo2()
        fastas = FastaCache(REF_DIR)

        rows = []
        # Order: gDTR -> Δh -> rollout -> IG.
        LOG.info("running gDTR benchmark ...")
        rows.append(bench_gdtr(bundle, all_variants, fastas))
        LOG.info("gDTR: %.2f ms/var, peak %.1f GB", rows[-1]["mean_per_variant_ms"], rows[-1]["peak_vram_gb"])

        LOG.info("running ‖Δh‖ benchmark ...")
        rows.append(bench_delta_h(bundle, all_variants, fastas))
        LOG.info("Δh: %.2f ms/var, peak %.1f GB", rows[-1]["mean_per_variant_ms"], rows[-1]["peak_vram_gb"])

        LOG.info("running rollout benchmark ...")
        rows.append(bench_rollout(bundle, all_variants, fastas))
        LOG.info("rollout: %.2f ms/var, peak %.1f GB", rows[-1]["mean_per_variant_ms"], rows[-1]["peak_vram_gb"])

        LOG.info("running IG benchmark ...")
        rows.append(bench_ig(bundle, all_variants, fastas))
        LOG.info("IG: %.2f ms/var, peak %.1f GB", rows[-1]["mean_per_variant_ms"], rows[-1]["peak_vram_gb"])

        cols = ["method", "n_variants", "total_wall_sec", "mean_per_variant_ms",
                "peak_vram_gb", "multi_gpu_scalable"]
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        DONE_PATH.write_text(json.dumps({"ok": True, "n_methods": len(rows)}, indent=2))
        ru.write_done(PHASE, PHASE_OUT_DIR, {"ok": True, "rows": rows})
        LOG.info("benchmark done — wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()

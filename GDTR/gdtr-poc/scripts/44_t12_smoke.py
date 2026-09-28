"""Smoke test for T1.2 baselines: run each method on first 3 variants.

Validates that:
  - delta_h forward+capture works (32-d feature non-NaN finite).
  - rollout QKV-pre-hook works (5-d feature non-NaN finite).
  - IG forward+backward through Evo 2 works at CONTEXT_HALF_IG=500.

Prints summary; exits non-zero on failure. Does NOT write _done markers.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

LOG = ru.setup_logging("tier1_smoke")

SEED = 42
ATTN_LAYERS = [3, 10, 17, 24, 31]
TARGET_LAYER = 29
IG_STEPS = 8

CONTEXT_HALF_FULL = 3000
CONTEXT_HALF_IG = 500


def main():
    from src.constants_evo2 import N_LAYERS
    from src.model_loader_evo2 import load_evo2, tokenize
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names

    torch.manual_seed(SEED); np.random.seed(SEED)
    bundle = load_evo2()
    LOG.info("model loaded: %s", bundle.loaded_variant)

    # Three synthetic variant sequences (no FASTA needed for smoke).
    rng = np.random.default_rng(SEED)
    bases = "ACGT"

    def synth_seq(half: int, var_pos_offset: int = 0):
        L = 2 * half
        seq = "".join(rng.choice(list(bases), size=L))
        local_idx = half + var_pos_offset
        return seq, local_idx

    # --- Test 1: delta_h ---
    LOG.info("=== Test 1: delta_h (CONTEXT=%d) ===", CONTEXT_HALF_FULL)
    layer_names = all_layer_names()
    t0 = time.time()
    for trial in range(3):
        ref_seq, lidx = synth_seq(CONTEXT_HALF_FULL)
        ref_at = ref_seq[lidx]
        new_base = (set("ACGT") - {ref_at}).pop()
        alt_seq = ref_seq[:lidx] + new_base + ref_seq[lidx + 1:]

        ids_ref = tokenize(ref_seq, bundle, device="cuda")
        hs = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
        h_ref = [hs[f"blocks.{ell}"][0, lidx, :].float().cpu() for ell in range(N_LAYERS)]
        del hs, ids_ref; torch.cuda.empty_cache()

        ids_alt = tokenize(alt_seq, bundle, device="cuda")
        hs = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
        h_alt = [hs[f"blocks.{ell}"][0, lidx, :].float().cpu() for ell in range(N_LAYERS)]
        del hs, ids_alt; torch.cuda.empty_cache()

        deltas = [(h_alt[ell] - h_ref[ell]).norm(p=2).item() for ell in range(N_LAYERS)]
        assert all(np.isfinite(deltas)), f"NaN in deltas trial {trial}"
        LOG.info("  trial %d: ‖Δh‖ range %.3f .. %.3f  (sum=%.3f)",
                 trial, min(deltas), max(deltas), sum(deltas))
    LOG.info("delta_h smoke OK in %.1fs", time.time() - t0)

    # --- Test 2: rollout ---
    LOG.info("=== Test 2: rollout (CONTEXT=%d) ===", CONTEXT_HALF_FULL)

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

    t0 = time.time()
    try:
        for trial in range(3):
            ref_seq, lidx = synth_seq(CONTEXT_HALF_FULL)
            for c in captures.values(): c.qkv = None
            ids = tokenize(ref_seq, bundle, device="cuda")
            with torch.no_grad():
                _ = bundle.model(ids, return_embeddings=False)
            ro = rollout(lidx, len(ref_seq))
            for c in captures.values(): c.qkv = None
            torch.cuda.empty_cache()
            assert all(np.isfinite(list(ro.values()))), f"NaN in rollout trial {trial}"
            LOG.info("  trial %d: rollout layers %s",
                     trial, {k: round(v, 4) for k, v in ro.items()})
    finally:
        for h in handles: h.remove()
    LOG.info("rollout smoke OK in %.1fs", time.time() - t0)

    # --- Test 3: IG ---
    LOG.info("=== Test 3: IG (CONTEXT=%d, steps=%d) ===", CONTEXT_HALF_IG, IG_STEPS)

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
    # Materialize all params out of inference_mode by replacing with clones.
    with torch.inference_mode(False):
        for name, p in list(bundle.sh.named_parameters()):
            mod_path, _, attr = name.rpartition(".")
            mod = bundle.sh.get_submodule(mod_path) if mod_path else bundle.sh
            new_p = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
            setattr(mod, attr, new_p)
        # Also bufffers
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
        # Get embedding outside inference_mode and clone to a normal tensor.
        with torch.inference_mode(False), torch.no_grad():
            e_raw = bundle.sh.embedding_layer(ids)
            e_full = e_raw.clone().detach()  # break inference_mode lineage
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

    t0 = time.time()
    try:
        for trial in range(3):
            ref_seq, lidx = synth_seq(CONTEXT_HALF_IG)
            ref_at = ref_seq[lidx]
            new_base = (set("ACGT") - {ref_at}).pop()
            alt_seq = ref_seq[:lidx] + new_base + ref_seq[lidx + 1:]
            t_var = time.time()
            ig_ref = ig_score(ref_seq, lidx)
            torch.cuda.empty_cache()
            ig_alt = ig_score(alt_seq, lidx)
            torch.cuda.empty_cache()
            assert np.isfinite(ig_ref) and np.isfinite(ig_alt)
            LOG.info("  trial %d: IG_ref=%.4g IG_alt=%.4g delta=%.4g (%.1fs)",
                     trial, ig_ref, ig_alt, ig_alt - ig_ref, time.time() - t_var)
    finally:
        h_handle.remove(); e_handle.remove()
    LOG.info("IG smoke OK in %.1fs", time.time() - t0)

    LOG.info("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("smoke FAILED")
        sys.exit(1)

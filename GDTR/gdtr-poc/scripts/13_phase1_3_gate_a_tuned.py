"""Phase 1.3 — Gate A_evo with tuned lens applied at L30, L31.

Recompute D_jsd at layers 30 and 31 using the tuned-lens projections
A_30 h_30 and A_31 h_31, then check tuned-recovery (M2_jsd >= 0.85).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase1.3"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        from src.constants_evo2 import N_LAYERS, VOCAB_SIZE, LOG_VOCAB
        from src.model_loader_evo2 import load_evo2
        from src.tuned_lens import load_tuned_lens

        traces = np.load(ru.GDTR_ROOT / "results" / "phase1.1" / "lens_traces.npz", allow_pickle=True)
        D_jsd_untuned = traces["D_jsd"]  # [N, L, T]
        N, L, T = D_jsd_untuned.shape

        taps = np.load(ru.GDTR_ROOT / "results" / "phase1.1" / "hidden_taps.npz", allow_pickle=True)
        hidden_30 = torch.from_numpy(taps["hidden_30"]).to(torch.float32)
        hidden_31 = torch.from_numpy(taps["hidden_31"]).to(torch.float32)
        norm_tap = torch.from_numpy(taps["norm_tap"]).to(torch.bfloat16)

        bundle = load_evo2()
        emb_clone = bundle.embedding_weight.detach().clone()[:VOCAB_SIZE, :].cuda()

        # Recompute reference logits from norm_tap (matches out.logits per smoke)
        with torch.no_grad():
            target_logits = torch.empty((N, T, VOCAB_SIZE), dtype=torch.float32, device="cuda")
            for s in range(0, N, 8):
                e = min(s + 8, N)
                target_logits[s:e] = F.linear(norm_tap[s:e].cuda(), emb_clone).float()
            log_p_final = F.log_softmax(target_logits, dim=-1)
            p_final = log_p_final.exp()

        # Apply tuned lenses
        A_30 = load_tuned_lens(str(ru.GDTR_ROOT / "results" / "phase1.2" / "A_30.pt"))
        A_31 = load_tuned_lens(str(ru.GDTR_ROOT / "results" / "phase1.2" / "A_31.pt"))

        D_jsd_tuned = D_jsd_untuned.copy()
        for layer_idx, h_layer, A in [(30, hidden_30, A_30), (31, hidden_31, A_31)]:
            with torch.no_grad():
                jsd_l_all = np.zeros((N, T), dtype=np.float32)
                for i in range(N):
                    h = h_layer[i:i+1].cuda().clone().detach()  # plain tensor (Bug-2)
                    y = A(h)  # [1,T,H]
                    # Manual RMSNorm with cloned scale (avoid inference-tensor)
                    y_cast = y.to(bundle.embedding_weight.dtype)
                    scale_c = bundle.norm.scale.detach().clone().to(y_cast.dtype)
                    Hd = y_cast.shape[-1]
                    rms = y_cast.norm(2, dim=-1, keepdim=True) * (Hd ** -0.5) + bundle.norm.eps
                    h_n = (y_cast / rms) * scale_c
                    logits_l = F.linear(h_n, emb_clone).float()  # [1,T,V]
                    log_p_l = F.log_softmax(logits_l, dim=-1)
                    p_l = log_p_l.exp()
                    p_f = p_final[i:i+1]
                    log_p_f = log_p_final[i:i+1]
                    m = 0.5 * (p_l + p_f)
                    log_m = (m + 1e-30).log()
                    kl_l = (p_l * (log_p_l - log_m)).sum(dim=-1)
                    kl_f = (p_f * (log_p_f - log_m)).sum(dim=-1)
                    jsd = 0.5 * (kl_l + kl_f) / LOG_VOCAB
                    jsd_l_all[i] = jsd[0].cpu().numpy()
                D_jsd_tuned[:, layer_idx, :] = jsd_l_all
                LOG.info("recomputed tuned-lens D_jsd at layer %d", layer_idx)

        # Force final layer to 0 by convention
        D_jsd_tuned[:, -1, :] = 0.0

        # Per-layer M2_jsd untuned vs tuned
        def m2_running_min(D_per_seq):
            # fraction (over seqs, positions) where running_min == raw at each layer
            mons = []
            for i in range(D_per_seq.shape[0]):
                rmin = np.minimum.accumulate(D_per_seq[i], axis=0)
                eq = (rmin == D_per_seq[i])
                mons.append(eq.mean(axis=1))  # [L]
            return np.stack(mons).mean(axis=0)

        m2_untuned_per_layer = m2_running_min(D_jsd_untuned)
        m2_tuned_per_layer = m2_running_min(D_jsd_tuned)

        # Per block-type aggregation
        from src.block_type import ATTN_LAYERS, HCS_LAYERS, HCM_LAYERS, HCL_LAYERS
        per_block_untuned = {
            "attn": float(m2_untuned_per_layer[ATTN_LAYERS].mean()),
            "hcs": float(m2_untuned_per_layer[HCS_LAYERS].mean()),
            "hcm": float(m2_untuned_per_layer[HCM_LAYERS].mean()),
            "hcl": float(m2_untuned_per_layer[HCL_LAYERS].mean()),
        }
        per_block_tuned = {
            "attn": float(m2_tuned_per_layer[ATTN_LAYERS].mean()),
            "hcs": float(m2_tuned_per_layer[HCS_LAYERS].mean()),
            "hcm": float(m2_tuned_per_layer[HCM_LAYERS].mean()),
            "hcl": float(m2_tuned_per_layer[HCL_LAYERS].mean()),
        }

        attn_pass = per_block_tuned["attn"] >= 0.85
        hyena_pass = all(per_block_tuned[k] >= 0.85 for k in ["hcs", "hcm", "hcl"])
        verdict = {
            "attn_M2_pass_after_tuned": bool(attn_pass),
            "hyena_M2_pass_after_tuned": bool(hyena_pass),
            "overall_pass": bool(attn_pass and hyena_pass),
            "tuned_recovered": bool(attn_pass and hyena_pass),
        }

        m2_un_d = {str(i): float(v) for i,v in enumerate(m2_untuned_per_layer)}
        m2_tu_d = {str(i): float(v) for i,v in enumerate(m2_tuned_per_layer)}
        out = {
            "phase": PHASE,
            "M2_jsd_untuned": m2_un_d,
            "M2_jsd_tuned": m2_tu_d,
            "per_block_M2_jsd_untuned": per_block_untuned,
            "per_block_M2_jsd_tuned": per_block_tuned,
            "M2_per_layer_untuned": m2_untuned_per_layer.tolist(),
            "M2_per_layer_tuned": m2_tuned_per_layer.tolist(),
            "verdict": verdict,
        }
        (PHASE_DIR / "gate_a_tuned.json").write_text(json.dumps(out, indent=2))

        # Figure F3
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(N_LAYERS), m2_untuned_per_layer, "-o", label="untuned")
            ax.plot(range(N_LAYERS), m2_tuned_per_layer, "-s", label="tuned (L30, L31)")
            ax.axhline(0.85, ls="--", color="red", label="0.85 threshold")
            ax.set_xlabel("layer"); ax.set_ylabel("M2_jsd (running-min monotone fraction)")
            ax.set_title("Phase 1.3 Gate A_evo (tuned recovery)")
            ax.legend()
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F3_tuned_recovery.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("figure save failed: %s", e)

        ru.write_done(PHASE, PHASE_DIR, {"verdict": verdict})
        LOG.info("Phase 1.3 done. verdict=%s", verdict)


if __name__ == "__main__":
    main()

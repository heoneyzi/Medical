"""Phase 1.2 — Train tuned lenses A_30 and A_31 on cached hidden states.

Uses cached hidden_30, hidden_31, norm_tap from Phase 1.1.
Targets: final logits computed via unembed(norm_tap)[..., :512]. We CLONE
the embedding weight before any unembed-pathway op (storage tied warning).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase1.2"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        from src.constants_evo2 import HIDDEN_SIZE, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2
        from src.tuned_lens import train_tuned_lens, save_tuned_lens

        # ---- Load cached taps ----
        taps = np.load(ru.GDTR_ROOT / "results" / "phase1.1" / "hidden_taps.npz", allow_pickle=True)
        hidden_30 = torch.from_numpy(taps["hidden_30"]).to(torch.float32)  # [N,T,H]
        hidden_31 = torch.from_numpy(taps["hidden_31"]).to(torch.float32)
        norm_tap = torch.from_numpy(taps["norm_tap"]).to(torch.bfloat16)
        N, T, H = hidden_30.shape
        LOG.info("loaded hidden taps: N=%d T=%d H=%d", N, T, H)

        # ---- Load model (need norm + embedding weight) ----
        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)

        # ---- Compute target logits = unembed(norm_tap)[..., :V] using a CLONE ----
        emb_clone = bundle.embedding_weight.detach().clone()[:VOCAB_SIZE, :].cuda()  # [V,H]
        with torch.no_grad():
            target_logits = torch.empty((N, T, VOCAB_SIZE), dtype=torch.float32, device="cuda")
            chunk = 8
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                nt = norm_tap[s:e].cuda()  # [b,T,H] bf16
                # unembed = h @ W_emb.T using clone -> bypass tied storage
                logits = torch.nn.functional.linear(nt, emb_clone)  # [b,T,V]
                target_logits[s:e] = logits.float()
                del nt, logits
        LOG.info("target_logits shape=%s", tuple(target_logits.shape))

        target_logits_cpu = target_logits.cpu()  # keep on CPU; train_tuned_lens moves to GPU
        del target_logits, emb_clone
        torch.cuda.empty_cache()

        # ---- Train A_30 then A_31 ----
        results = {}
        for layer_idx, h_layer in [(30, hidden_30), (31, hidden_31)]:
            LOG.info("training tuned lens A_%d ...", layer_idx)
            t0 = time.time()
            lens, res = train_tuned_lens(
                hidden_layer=h_layer,
                target_logits=target_logits_cpu,
                norm_module=bundle.norm,
                embedding_weight=bundle.embedding_weight,
                layer_idx=layer_idx,
                epochs=15, lr=1e-3, batch_size=4, seed=42,
                device="cuda", dtype=torch.float32,
            )
            ckpt = PHASE_DIR / f"A_{layer_idx}.pt"
            save_tuned_lens(lens, str(ckpt))
            elapsed = time.time() - t0
            results[layer_idx] = {
                "epochs": res.epochs,
                "loss_curve": res.loss_curve,
                "final_loss": res.final_loss,
                "elapsed_s": round(elapsed, 1),
                "ckpt_path": str(ckpt),
            }
            LOG.info("A_%d trained in %.1fs final_loss=%.4e", layer_idx, elapsed, res.final_loss)
            del lens
            torch.cuda.empty_cache()

        (PHASE_DIR / "training_curve.json").write_text(json.dumps(results, indent=2, default=float))

        # ---- Plot loss curves ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for L, r in results.items():
                ax.plot(r["loss_curve"], label=f"A_{L}")
            ax.set_xlabel("step"); ax.set_ylabel("MSE loss"); ax.set_yscale("log"); ax.legend()
            ax.set_title("Phase 1.2 tuned lens training")
            fig.tight_layout()
            for ext in ("png",):
                fig.savefig(PHASE_DIR / f"F_tuned_loss.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("loss curve plot failed: %s", e)

        ru.write_done(PHASE, PHASE_DIR, {
            "final_loss_30": results[30]["final_loss"],
            "final_loss_31": results[31]["final_loss"],
        })
        LOG.info("Phase 1.2 done.")


if __name__ == "__main__":
    main()

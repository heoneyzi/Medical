"""Phase 1 follow-up — Train tuned lenses A_L for L in {15, 20, 25, 28}.

The Phase 1.2 lenses at L=30 / L=31 are degenerate (h30 == h31 == norm input
because Evo 2's last attention block has an idle/no-op residual contribution),
so MSE is exactly 0 from initialization (A=I) and there is nothing to "tune".

This follow-up trains tuned lenses at EARLIER layers where the JSD lens
actually diverges from out.logits, to test whether the "tuned lens recovers
linear decodability" story still holds — and at which depth it begins.

Targets:
  L=15 (hcm) — middle of network
  L=20 (hcl) — late-middle
  L=25 (hcs) — late
  L=28 (hcs) — last hyena before attn at L=31

Reuses src/tuned_lens.py infrastructure exactly (clones tied embedding,
clones inference-tensor norm.scale via _frozen_unembed; identity-init
affine A, MSE on logits, Adam @ lr=1e-3, 15 epochs, batch=4, seed=42).
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

PHASE = "phase1.followup"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)

TARGET_LAYERS = [15, 20, 25, 28]
SEED = 42


def parse_fasta(path: Path):
    """Yield (seq_id, seq) for each FASTA record."""
    seqs = []
    cur_id, cur = None, []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur)))
                cur = []
                cur_id = line[1:].split()[0]
            else:
                cur.append(line)
        if cur_id is not None:
            seqs.append((cur_id, "".join(cur)))
    return seqs


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.constants_evo2 import HIDDEN_SIZE, N_LAYERS, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import extract_hidden_states
        from src.tuned_lens import train_tuned_lens, save_tuned_lens

        # ---- Load 100 sanity sequences ----
        fa_path = ru.GDTR_ROOT / "data" / "baselines" / "phase1_sanity_seqs.fa"
        seqs = parse_fasta(fa_path)
        N = len(seqs)
        T = len(seqs[0][1])
        LOG.info("loaded %d sanity sequences (T=%d) from %s", N, T, fa_path)

        # ---- Load model ----
        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)

        # ---- Forward all 100 seqs, capture blocks.{15,20,25,28} + norm ----
        save_layers = [f"blocks.{L}" for L in TARGET_LAYERS] + ["norm"]
        H = HIDDEN_SIZE

        hidden_per_L = {L: np.zeros((N, T, H), dtype=np.float32) for L in TARGET_LAYERS}
        norm_taps = np.zeros((N, T, H), dtype=np.float32)

        t0 = time.time()
        for i, (sid, seq) in enumerate(seqs):
            input_ids = tokenize(seq, bundle, device="cuda")
            hs = extract_hidden_states(bundle, input_ids, save_layers=save_layers)
            for L in TARGET_LAYERS:
                hidden_per_L[L][i] = hs[f"blocks.{L}"].squeeze(0).to(torch.float32).cpu().numpy()
            norm_taps[i] = hs["norm"].squeeze(0).to(torch.float32).cpu().numpy()
            del hs
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                LOG.info("forward %d/%d (%.1fs elapsed)", i + 1, N, time.time() - t0)
        LOG.info("forward complete in %.1fs", time.time() - t0)

        # ---- Compute target logits = unembed(norm_tap)[..., :V] using cloned weight ----
        emb_clone = bundle.embedding_weight.detach().clone()[:VOCAB_SIZE, :].cuda()
        norm_taps_t = torch.from_numpy(norm_taps).to(torch.bfloat16)
        with torch.no_grad():
            target_logits = torch.empty((N, T, VOCAB_SIZE), dtype=torch.float32, device="cuda")
            chunk = 8
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                nt = norm_taps_t[s:e].cuda()
                logits = torch.nn.functional.linear(nt, emb_clone)
                target_logits[s:e] = logits.float()
                del nt, logits
        LOG.info("target_logits shape=%s", tuple(target_logits.shape))
        target_logits_cpu = target_logits.cpu()
        del target_logits, emb_clone, norm_taps_t
        torch.cuda.empty_cache()

        # ---- Compute INITIAL loss (A=I) for each L using src.tuned_lens helpers ----
        from src.tuned_lens import _frozen_unembed
        emb_clone2 = bundle.embedding_weight.detach().clone()[:VOCAB_SIZE, :].cuda().contiguous()

        initial_losses = {}
        for L in TARGET_LAYERS:
            h_L = torch.from_numpy(hidden_per_L[L]).to(torch.float32).cuda()
            tgt = target_logits_cpu.cuda()
            with torch.no_grad():
                pred = _frozen_unembed(h_L, bundle.norm, emb_clone2).float()
                loss = torch.nn.functional.mse_loss(pred, tgt).item()
            initial_losses[L] = loss
            LOG.info("L=%d initial MSE (A=I): %.6e", L, loss)
            del h_L, tgt, pred
            torch.cuda.empty_cache()
        del emb_clone2
        torch.cuda.empty_cache()

        # ---- Train A_L for each target layer ----
        results = {}
        for L in TARGET_LAYERS:
            LOG.info("training tuned lens A_%d ...", L)
            t1 = time.time()
            h_layer = torch.from_numpy(hidden_per_L[L]).to(torch.float32)
            lens, res = train_tuned_lens(
                hidden_layer=h_layer,
                target_logits=target_logits_cpu,
                norm_module=bundle.norm,
                embedding_weight=bundle.embedding_weight,
                layer_idx=L,
                epochs=15, lr=1e-3, batch_size=4, seed=SEED,
                device="cuda", dtype=torch.float32,
            )
            ckpt = PHASE_DIR / f"A_{L}.pt"
            save_tuned_lens(lens, str(ckpt))
            elapsed = time.time() - t1
            ratio = (res.final_loss / initial_losses[L]) if initial_losses[L] > 0 else float("inf")
            results[L] = {
                "epochs": res.epochs,
                "loss_curve": res.loss_curve,
                "initial_loss_identity": initial_losses[L],
                "final_loss": res.final_loss,
                "ratio_final_over_initial": ratio,
                "elapsed_s": round(elapsed, 1),
                "ckpt_path": str(ckpt),
            }
            LOG.info(
                "A_%d trained in %.1fs initial=%.4e final=%.4e ratio=%.3f",
                L, elapsed, initial_losses[L], res.final_loss, ratio,
            )
            del lens, h_layer
            torch.cuda.empty_cache()

        (PHASE_DIR / "training_curve.json").write_text(json.dumps(results, indent=2, default=float))

        # ---- Plot loss curves ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for L, r in results.items():
                ax.plot(r["loss_curve"], label=f"A_{L}  ({r['initial_loss_identity']:.2e}->{r['final_loss']:.2e})")
            ax.set_xlabel("step")
            ax.set_ylabel("MSE loss (logits)")
            ax.set_yscale("log")
            ax.legend(fontsize=9)
            ax.set_title("Phase 1 follow-up: tuned lens at earlier layers")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F_followup_recovery.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("loss curve plot failed: %s", e)

        # ---- Verdict ----
        recoveries = {
            L: (initial_losses[L] - results[L]["final_loss"]) / max(initial_losses[L], 1e-30)
            for L in TARGET_LAYERS
        }
        best_L = max(recoveries, key=lambda k: recoveries[k])
        any_real_recovery = any(r > 0.5 for r in recoveries.values())

        verdict = {
            "target_layers": TARGET_LAYERS,
            "initial_losses": {str(L): initial_losses[L] for L in TARGET_LAYERS},
            "final_losses": {str(L): results[L]["final_loss"] for L in TARGET_LAYERS},
            "fractional_recovery": {str(L): recoveries[L] for L in TARGET_LAYERS},
            "best_layer": best_L,
            "any_real_recovery_gt_50pct": any_real_recovery,
            "interpretation": (
                f"Largest tuned recovery at L={best_L} "
                f"(fractional drop {recoveries[best_L]:.3f}). "
                + ("Tuned-lens-at-earlier-layer story HOLDS — at least one L showed >50% MSE drop."
                   if any_real_recovery
                   else "No layer showed >50% MSE drop — Evo2 may not benefit from a single linear lens at any depth.")
            ),
        }
        (PHASE_DIR / "gate_a_followup.json").write_text(json.dumps(verdict, indent=2, default=float))
        LOG.info("verdict: %s", verdict["interpretation"])

        ru.write_done(PHASE, PHASE_DIR, {
            "best_layer": best_L,
            "any_real_recovery_gt_50pct": any_real_recovery,
            "fractional_recovery": {str(L): recoveries[L] for L in TARGET_LAYERS},
        })
        LOG.info("Phase 1 follow-up done.")


if __name__ == "__main__":
    main()

"""Phase 1 follow-up FULL — Train tuned lenses A_L for ALL 32 layers (l = 0..31).

Phase 1.2 covered L=30/31 (degenerate, loss=0). Phase 1.followup covered
L=15/20/25/28 (98%+ recovery). This script fills in EVERY layer to map
the full recovery landscape -> identify peak divergence layer + best
canonical "deep-thinking" tap for Phase 2/3.

Strategy (memory-aware on 144 GB H200 + 232 GB RAM):
  1. Forward each of 100 sanity sequences ONCE.
  2. Stash all 33 layer taps (blocks.0..31 + norm) in CPU bf16
     (32 layers x 100 x 6000 x 4096 x 2B ~ 157 GB; fits in 232 GB RAM).
  3. Compute target_logits once (norm tap -> unembed).
  4. For each l in 0..31: move h_l to GPU as fp32 (~9.8 GB), train A_l
     (15 epochs, lr=1e-3, eye-init, MSE on logits via src.tuned_lens),
     record initial/final loss, save A_l.pt, free GPU memory.
  5. Emit recovery_curve.json + recovery_landscape plots.

Reuses src/tuned_lens.py:train_tuned_lens() and _frozen_unembed() — does
NOT re-implement.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "phase1.followup_full"
PHASE_DIR = ru.GDTR_ROOT / "results" / "phase1.followup_full"
LOG = ru.setup_logging(PHASE)

ALL_LAYERS = list(range(32))
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
        from src.tuned_lens import (
            train_tuned_lens, save_tuned_lens, _frozen_unembed,
        )
        from src.block_type import block_type

        # ---- Load 100 sanity sequences ----
        fa_path = ru.GDTR_ROOT / "data" / "baselines" / "phase1_sanity_seqs.fa"
        seqs = parse_fasta(fa_path)
        N = len(seqs)
        T = len(seqs[0][1])
        H = HIDDEN_SIZE
        LOG.info("loaded %d sanity sequences (T=%d) from %s", N, T, fa_path)

        # ---- Load model ----
        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)

        # ---- Pre-allocate per-layer CPU bf16 buffers (~157 GB total) ----
        save_layers = [f"blocks.{L}" for L in ALL_LAYERS] + ["norm"]
        LOG.info("allocating CPU bf16 buffers: %d layers x %d x %d x %d (~%.1f GB)",
                 len(save_layers), N, T, H,
                 len(save_layers) * N * T * H * 2 / 1e9)
        hidden_per_L = {
            L: torch.empty((N, T, H), dtype=torch.bfloat16, pin_memory=False)
            for L in ALL_LAYERS
        }
        norm_taps = torch.empty((N, T, H), dtype=torch.bfloat16, pin_memory=False)

        # ---- Forward all 100 seqs ONCE, capture all 32 + norm taps ----
        t0 = time.time()
        for i, (sid, seq) in enumerate(seqs):
            input_ids = tokenize(seq, bundle, device="cuda")
            hs = extract_hidden_states(bundle, input_ids, save_layers=save_layers)
            for L in ALL_LAYERS:
                # extract_hidden_states returns bf16 GPU; copy to CPU bf16.
                hidden_per_L[L][i].copy_(hs[f"blocks.{L}"].squeeze(0).to(torch.bfloat16).cpu())
            norm_taps[i].copy_(hs["norm"].squeeze(0).to(torch.bfloat16).cpu())
            del hs, input_ids
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                LOG.info("forward %d/%d (%.1fs elapsed)", i + 1, N, time.time() - t0)
        LOG.info("forward complete in %.1fs", time.time() - t0)

        # ---- Compute target_logits (frozen unembed of norm tap) ----
        emb_clone = bundle.embedding_weight.detach().clone()[:VOCAB_SIZE, :].cuda().contiguous()
        target_logits_cpu = torch.empty((N, T, VOCAB_SIZE), dtype=torch.float32)
        chunk = 8
        with torch.no_grad():
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                nt = norm_taps[s:e].cuda().to(emb_clone.dtype)
                logits = torch.nn.functional.linear(nt, emb_clone).float()
                target_logits_cpu[s:e] = logits.cpu()
                del nt, logits
        LOG.info("target_logits shape=%s", tuple(target_logits_cpu.shape))
        # norm_taps no longer needed
        del norm_taps
        gc.collect()

        # ---- Per-layer initial loss (A=I) and tuned-lens training ----
        results = {}
        all_initial = {}
        all_final = {}
        all_recovery = {}

        for L in ALL_LAYERS:
            t1 = time.time()
            btype = block_type(L)

            # Move h_L to GPU as fp32 (~9.8 GB)
            h_L_cpu = hidden_per_L[L]
            h_L = h_L_cpu.to(torch.float32)  # CPU fp32 (~9.8 GB) — train_tuned_lens will move to GPU

            # Initial loss with A=I via _frozen_unembed
            with torch.no_grad():
                h_gpu = h_L.cuda()
                pred0 = _frozen_unembed(h_gpu, bundle.norm, emb_clone).float()
                tgt_gpu = target_logits_cpu.cuda()
                init_loss = torch.nn.functional.mse_loss(pred0, tgt_gpu).item()
                del pred0, h_gpu, tgt_gpu
                torch.cuda.empty_cache()
            all_initial[L] = init_loss
            LOG.info("L=%02d (%s) initial MSE (A=I): %.6e", L, btype, init_loss)

            # Train tuned lens (skip degenerate layers L>=30 — initial≈0)
            if init_loss < 1e-8:
                LOG.info("L=%02d degenerate (init<1e-8) — skipping training", L)
                final_loss = init_loss
                loss_curve = [init_loss]
                ckpt_path_str = ""
                degenerate = True
            else:
                lens, res = train_tuned_lens(
                    hidden_layer=h_L,
                    target_logits=target_logits_cpu,
                    norm_module=bundle.norm,
                    embedding_weight=bundle.embedding_weight,
                    layer_idx=L,
                    epochs=15, lr=1e-3, batch_size=4, seed=SEED,
                    device="cuda", dtype=torch.float32,
                )
                ckpt = PHASE_DIR / f"A_{L}.pt"
                save_tuned_lens(lens, str(ckpt))
                final_loss = res.final_loss
                loss_curve = res.loss_curve
                ckpt_path_str = str(ckpt)
                degenerate = False
                del lens, res
            all_final[L] = final_loss
            recovery = (init_loss - final_loss) / max(init_loss, 1e-30) if init_loss > 0 else 0.0
            all_recovery[L] = recovery
            elapsed = time.time() - t1
            results[str(L)] = {
                "layer": L,
                "block_type": btype,
                "initial_loss_identity": float(init_loss),
                "final_loss": float(final_loss),
                "recovery_pct": float(recovery),
                "loss_curve": [float(x) for x in loss_curve],
                "elapsed_s": round(elapsed, 1),
                "ckpt_path": ckpt_path_str,
                "degenerate": bool(degenerate),
            }
            LOG.info(
                "A_%02d (%s) trained in %.1fs init=%.4e final=%.4e recovery=%.4f",
                L, btype, elapsed, init_loss, final_loss, recovery,
            )

            # Free per-layer memory immediately
            del h_L, h_L_cpu
            del hidden_per_L[L]
            gc.collect()
            torch.cuda.empty_cache()

        del emb_clone, target_logits_cpu
        gc.collect(); torch.cuda.empty_cache()

        # ---- Persist recovery_curve.json ----
        summary = {
            "n_layers": N_LAYERS,
            "n_seqs": N,
            "T": T,
            "seed": SEED,
            "epochs": 15,
            "lr": 1e-3,
            "per_layer": results,
        }
        (PHASE_DIR / "recovery_curve.json").write_text(
            json.dumps(summary, indent=2, default=float)
        )
        LOG.info("wrote recovery_curve.json")

        # ---- Plot 1: log-scale initial vs final loss landscape ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            xs = list(range(N_LAYERS))
            init_arr = np.array([all_initial[L] for L in xs])
            final_arr = np.array([all_final[L] for L in xs])
            btypes = [block_type(L) for L in xs]
            color_map = {"attn": "#d62728", "hcs": "#1f77b4",
                         "hcm": "#2ca02c", "hcl": "#9467bd"}
            colors = [color_map[bt] for bt in btypes]

            # log scale needs >0 — clamp
            init_plot = np.maximum(init_arr, 1e-12)
            final_plot = np.maximum(final_arr, 1e-12)

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(xs, init_plot, "-", color="gray", alpha=0.5,
                    linewidth=1, label="initial (A=I)")
            ax.plot(xs, final_plot, "-", color="black", alpha=0.5,
                    linewidth=1, label="final (tuned A)")
            ax.scatter(xs, init_plot, s=50, c=colors, marker="o",
                       edgecolors="gray", linewidths=0.5, zorder=3)
            ax.scatter(xs, final_plot, s=40, c=colors, marker="^",
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.set_yscale("log")
            ax.set_xlabel("layer L")
            ax.set_ylabel("MSE (logits)")
            ax.set_title(f"Phase 1.followup_full: tuned-lens recovery, ALL 32 layers (n={N})")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(xs)
            ax.set_xticklabels(xs, fontsize=8)

            # legends
            btype_handles = [Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=c, markersize=8, label=bt)
                              for bt, c in color_map.items()]
            shape_handles = [
                Line2D([0], [0], marker="o", color="gray", markersize=8,
                       linestyle="-", label="initial (A=I)"),
                Line2D([0], [0], marker="^", color="black", markersize=8,
                       linestyle="-", label="final (tuned A)"),
            ]
            leg1 = ax.legend(handles=btype_handles, loc="lower left", title="block_type")
            ax.add_artist(leg1)
            ax.legend(handles=shape_handles, loc="upper right")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F_recovery_landscape.{ext}", dpi=150)
            plt.close(fig)
            LOG.info("wrote F_recovery_landscape.{pdf,png}")
        except Exception as e:
            LOG.warning("landscape plot failed: %s", e)

        # ---- Plot 2: bar plot of recovery_pct ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = list(range(N_LAYERS))
            rec_arr = np.array([all_recovery[L] for L in xs])
            btypes = [block_type(L) for L in xs]
            color_map = {"attn": "#d62728", "hcs": "#1f77b4",
                         "hcm": "#2ca02c", "hcl": "#9467bd"}
            colors = [color_map[bt] for bt in btypes]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(xs, rec_arr, color=colors, edgecolor="black", linewidth=0.4)
            ax.axhline(0.9, color="black", linestyle="--", linewidth=0.7,
                       alpha=0.5, label="0.90 threshold")
            ax.set_xlabel("layer L")
            ax.set_ylabel("recovery_pct = 1 - final/initial")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Phase 1.followup_full: tuned-lens recovery_pct, ALL 32 layers")
            ax.set_xticks(xs)
            ax.set_xticklabels(xs, fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(loc="lower left")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F_recovery_pct.{ext}", dpi=150)
            plt.close(fig)
            LOG.info("wrote F_recovery_pct.{pdf,png}")
        except Exception as e:
            LOG.warning("recovery_pct plot failed: %s", e)

        # ---- Verdict / canonical pick ----
        non_degenerate = [L for L in ALL_LAYERS if not results[str(L)]["degenerate"]]
        peak_divergence_L = max(non_degenerate, key=lambda L: all_initial[L])
        worst_recovery_L = min(non_degenerate, key=lambda L: all_recovery[L])
        # Canonical "deep-thinking" tap = best recovery among LATE layers
        # excluding degenerate L=30,31. Prefer late, near-final.
        late_candidates = [L for L in non_degenerate if 20 <= L <= 29]
        if late_candidates:
            canonical_L = max(late_candidates, key=lambda L: all_recovery[L])
        else:
            canonical_L = max(non_degenerate, key=lambda L: all_recovery[L])

        verdict = {
            "n_layers": N_LAYERS,
            "peak_divergence_layer": peak_divergence_L,
            "peak_divergence_initial_loss": float(all_initial[peak_divergence_L]),
            "worst_recovery_layer": worst_recovery_L,
            "worst_recovery_pct": float(all_recovery[worst_recovery_L]),
            "canonical_deep_thinking_layer": canonical_L,
            "canonical_recovery_pct": float(all_recovery[canonical_L]),
            "canonical_initial_loss": float(all_initial[canonical_L]),
            "degenerate_layers": [L for L in ALL_LAYERS if results[str(L)]["degenerate"]],
            "n_layers_recovery_gt_0p90": int(sum(1 for L in ALL_LAYERS if all_recovery[L] > 0.90)),
        }
        (PHASE_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
        LOG.info("verdict: %s", verdict)

        ru.write_done(PHASE, PHASE_DIR, verdict)
        LOG.info("Phase 1 followup_full done.")


if __name__ == "__main__":
    main()

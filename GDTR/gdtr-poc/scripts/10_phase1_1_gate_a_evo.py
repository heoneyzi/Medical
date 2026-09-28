"""Phase 1.1 — Gate A_evo (untuned lens) on 100 sanity sequences.

Per seq: forward Evo 2, capture residual stream + post-norm, compute
D_jsd[32, T], D_cos[32, T], top1[32, T]. Saves cached hidden_30/hidden_31
for downstream tuned-lens training (Phase 1.2).
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

PHASE = "phase1.1"
PHASE_DIR = ru.GDTR_ROOT / "results" / PHASE
LOG = ru.setup_logging(PHASE)


def parse_fasta(path: Path):
    """Yield (seq_id, seq, region_label). region_label inferred from header."""
    seqs = []
    cur_id, cur_label, cur = None, None, []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur), cur_label))
                cur = []
                # header e.g. >sanity_gc_001 ; >sanity_shuf_001
                cur_id = line[1:].split()[0]
                if "shuf" in cur_id:
                    cur_label = "sanity_shuf"
                elif "gc" in cur_id:
                    cur_label = "sanity_gc"
                else:
                    cur_label = "sanity_other"
            else:
                cur.append(line)
        if cur_id is not None:
            seqs.append((cur_id, "".join(cur), cur_label))
    return seqs


def main() -> None:
    with ru.phase_context(PHASE, PHASE_DIR):
        seed = 42
        torch.manual_seed(seed); np.random.seed(seed)

        from src.constants_evo2 import N_LAYERS, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2, tokenize
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, top1_predictions, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        fa_path = ru.GDTR_ROOT / "data" / "baselines" / "phase1_sanity_seqs.fa"
        seqs = parse_fasta(fa_path)
        LOG.info("loaded %d sanity sequences from %s", len(seqs), fa_path)

        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)

        layer_names = all_layer_names()  # blocks.0..31 + norm

        N = len(seqs)
        T = len(seqs[0][1])  # 6000
        D_jsd = np.zeros((N, N_LAYERS, T), dtype=np.float32)
        D_cos = np.zeros((N, N_LAYERS, T), dtype=np.float32)
        top1 = np.zeros((N, N_LAYERS, T), dtype=np.int32)
        hidden_30 = np.zeros((N, T, 4096), dtype=np.float32)  # BUG-4 FIX: fp32 (no overflow)
        hidden_31 = np.zeros((N, T, 4096), dtype=np.float32)
        norm_taps = np.zeros((N, T, 4096), dtype=np.float32)  # for tuned-lens target
        regions = []
        seq_ids = []

        t0 = time.time()
        for i, (sid, seq, region) in enumerate(seqs):
            input_ids = tokenize(seq, bundle, device="cuda")
            hs = extract_hidden_states(bundle, input_ids, save_layers=layer_names)
            # JSD lens
            d_jsd = jsd_lens(hs, bundle, n_layers=N_LAYERS).numpy()
            d_cos = cosine_lens(hs, n_layers=N_LAYERS).numpy()
            t1 = top1_predictions(hs, bundle, n_layers=N_LAYERS).numpy()
            D_jsd[i] = d_jsd
            D_cos[i] = d_cos
            top1[i] = t1.astype(np.int32)
            hidden_30[i] = hs["blocks.30"].squeeze(0).to(torch.float32).cpu().numpy()
            hidden_31[i] = hs["blocks.31"].squeeze(0).to(torch.float32).cpu().numpy()
            norm_taps[i] = hs["norm"].squeeze(0).to(torch.float32).cpu().numpy()
            regions.append(region)
            seq_ids.append(sid)
            del hs
            torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                LOG.info("seq %d/%d (%.1fs elapsed)", i + 1, N, time.time() - t0)

        regions = np.array(regions); seq_ids = np.array(seq_ids)

        # ---- M-stat computation ----
        # M2_jsd: at the FINAL block (n_layers-1) D_jsd ~ 0 by construction; we use
        # the per-layer max-final JSD as monotonicity surrogate (running-min monotone).
        # Phase 0 definition: M2 = fraction of positions where running_min(D)[:, t] is
        # itself non-increasing -> equivalent to running_min(D) == D pointwise.
        from phase0_src.src.gdtr import jsd_running_min_monotonic, top1_monotonic_after_first_match

        m1_per_seq = []
        m2_jsd_per_seq = []
        m2_cos_per_seq = []
        for i in range(N):
            t1_t = torch.from_numpy(top1[i].astype(np.int64))
            m1 = top1_monotonic_after_first_match(t1_t).float().mean().item()
            m1_per_seq.append(m1)
            m2j = jsd_running_min_monotonic(torch.from_numpy(D_jsd[i])).float().mean().item()
            m2c = jsd_running_min_monotonic(torch.from_numpy(D_cos[i])).float().mean().item()
            m2_jsd_per_seq.append(m2j); m2_cos_per_seq.append(m2c)
        m1 = float(np.mean(m1_per_seq))
        m2_jsd = float(np.mean(m2_jsd_per_seq))
        m2_cos = float(np.mean(m2_cos_per_seq))

        # Per-block-type M-stats: for each block type, compute fraction-of-positions
        # whose D_jsd value at that block is <= the running min so far (i.e. block
        # contributes to monotone descent).
        from src.block_type import block_type, ATTN_LAYERS, HCS_LAYERS, HCM_LAYERS, HCL_LAYERS
        per_block_m2_jsd = {}
        per_block_m2_cos = {}
        # Define: M2 per block group = mean over (seqs, positions, layers in group) of indicator
        # that running_min(D)[layer] equals D[layer] at that position.
        for bt_name, idxs in [("attn", ATTN_LAYERS), ("hcs", HCS_LAYERS),
                              ("hcm", HCM_LAYERS), ("hcl", HCL_LAYERS)]:
            # running_min monotone equality across all (seq, layer in group, t)
            vals_jsd = []
            vals_cos = []
            for i in range(N):
                rmin_j = np.minimum.accumulate(D_jsd[i], axis=0)
                rmin_c = np.minimum.accumulate(D_cos[i], axis=0)
                eq_j = (rmin_j[idxs] == D_jsd[i][idxs]).mean()
                eq_c = (rmin_c[idxs] == D_cos[i][idxs]).mean()
                vals_jsd.append(eq_j); vals_cos.append(eq_c)
            per_block_m2_jsd[bt_name] = float(np.mean(vals_jsd))
            per_block_m2_cos[bt_name] = float(np.mean(vals_cos))

        # M2_global per Phase 1 spec: UR M2 (cosine) >= 0.50
        m2_ur_global = m2_cos

        # Bootstrap CIs on M-stats
        from phase0_src.src.stats import bootstrap_ci
        ci_m1 = bootstrap_ci(m1_per_seq, n_boot=1000, ci=0.95, seed=seed)
        ci_m2_jsd = bootstrap_ci(m2_jsd_per_seq, n_boot=1000, ci=0.95, seed=seed)
        ci_m2_cos = bootstrap_ci(m2_cos_per_seq, n_boot=1000, ci=0.95, seed=seed)

        # ---- Verdict ----
        attn_pass = all(per_block_m2_jsd[k] >= 0.85 for k in ["attn"])
        hyena_pass = all(per_block_m2_jsd[k] >= 0.85 for k in ["hcs", "hcm", "hcl"])
        ur_pass = m2_ur_global >= 0.50
        verdict_untuned = {
            "attn_M2_jsd_pass": bool(attn_pass),
            "hyena_M2_jsd_pass": bool(hyena_pass),
            "ur_M2_global_pass": bool(ur_pass),
            "overall_pass": bool(attn_pass and hyena_pass and ur_pass),
            "tuned_recovery_required_for": [
                k for k in ["hcs", "hcm", "hcl"] if per_block_m2_jsd[k] < 0.85
            ] + ([k for k in ["attn"] if per_block_m2_jsd["attn"] < 0.85]),
        }

        # ---- Save artefacts ----
        np.savez_compressed(
            PHASE_DIR / "lens_traces.npz",
            D_jsd=D_jsd, D_cos=D_cos, top1=top1,
            regions=regions, seq_ids=seq_ids,
        )
        np.savez(  # uncompressed (compressed was 67min)
            PHASE_DIR / "hidden_taps.npz",
            hidden_30=hidden_30, hidden_31=hidden_31, norm_tap=norm_taps,
            regions=regions, seq_ids=seq_ids,
        )
        result = {
            "phase": PHASE,
            "n_seqs": N, "T": T, "n_layers": N_LAYERS,
            "loaded_variant": bundle.loaded_variant,
            "M1": m1, "M1_ci95": ci_m1,
            "M2_jsd_overall": m2_jsd, "M2_jsd_ci95": ci_m2_jsd,
            "M2_cos_overall": m2_cos, "M2_cos_ci95": ci_m2_cos,
            "per_block_M2_jsd": per_block_m2_jsd,
            "per_block_M2_cos": per_block_m2_cos,
            "M2_ur_global": m2_ur_global,
            "verdict_untuned": verdict_untuned,
            "elapsed_s": round(time.time() - t0, 2),
        }
        (PHASE_DIR / "gate_a_evo_untuned.json").write_text(json.dumps(result, indent=2))

        # ---- Figure F2 ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            mean_jsd = D_jsd.mean(axis=(0, 2))
            mean_cos = D_cos.mean(axis=(0, 2))
            axes[0].plot(range(N_LAYERS), mean_jsd, "-o")
            axes[0].set_title("Mean D_jsd vs layer (sanity)")
            axes[0].set_xlabel("layer"); axes[0].set_ylabel("D_jsd / log V")
            axes[1].plot(range(N_LAYERS), mean_cos, "-o", color="C1")
            axes[1].set_title("Mean D_cos vs layer (sanity)")
            axes[1].set_xlabel("layer"); axes[1].set_ylabel("1 - cos(h_l, h_31)")
            for ax in axes:
                ax.axvspan(2.5, 3.5, alpha=0.1, color="red")
                ax.axvspan(9.5, 10.5, alpha=0.1, color="red")
                ax.axvspan(16.5, 17.5, alpha=0.1, color="red")
                ax.axvspan(23.5, 24.5, alpha=0.1, color="red")
                ax.axvspan(30.5, 31.5, alpha=0.1, color="red")
            fig.suptitle(f"Phase 1.1 Gate A_evo (untuned) — {bundle.loaded_variant}")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(PHASE_DIR / f"F2_evo_sanity.{ext}", dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning("figure save failed: %s", e)

        ru.write_done(PHASE, PHASE_DIR, {"verdict": verdict_untuned, "M2_jsd": m2_jsd, "M2_cos": m2_cos})
        LOG.info("Phase 1.1 done. verdict=%s", verdict_untuned)


if __name__ == "__main__":
    main()

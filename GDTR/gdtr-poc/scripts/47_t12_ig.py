"""T1.2 Baseline C: Integrated Gradients on h_29 norm at variant position.

For each (ref, alt) sequence, IG attributes a scalar target — ‖h_29(var_pos)‖₂
— back to the input embedding at the variant token, integrated over a path
from a zero-embedding baseline to the actual embedding.

Computational note: IG requires forward + backward through Evo 2 7B once per
step. With Flash Attention's heavy memory footprint and 7B weights, full
context CONTEXT_HALF=3000 is intractable for 10,910 variants. We use a
shorter context CONTEXT_HALF_IG=500 (T=1000) for the bulk run; the variant
position is centered identically. Document substitution in the cost
benchmark — full-context IG is reserved for case studies (T1.3 path).

IG_score = ||  ∂‖h_29(var_pos)‖₂ / ∂e[var_pos]  averaged over k steps  ||₂
       (we report the L2 norm of the integrated gradient at the variant token)

Per variant we compute IG_ref and IG_alt and store delta = IG_alt − IG_ref.

Output (resumable):
  /path/to/gDTR/results/tier1_baselines/ig_features.csv
  /path/to/gDTR/results/tier1_baselines/_ig_checkpoint.json
  /path/to/gDTR/results/tier1_baselines/_ig_done
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import _runner_utils as ru
ru.add_repo_paths()
ru.patch_safe_globals()

PHASE = "tier1_baselines_ig"
PHASE_OUT_DIR = ru.GDTR_ROOT / "results" / "tier1_baselines"
LOG = ru.setup_logging(PHASE)

SEED = 42
CONTEXT_HALF_IG = 500   # IG-specific shorter context (memory + time)
IG_STEPS = 8
TARGET_LAYER = 29       # canonical "deep-thinking" layer
CHECKPOINT_EVERY = 250

GENES = (
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "APC",
    "MLH1", "MSH2", "PTEN", "RB1", "VHL", "ATM", "PALB2",
)
TRAIN_CATS = ("P_LP", "B_LB")
HOLD_CATS = ("VUS",)
ALL_CATS = TRAIN_CATS + HOLD_CATS

REF_DIR = ru.GDTR_ROOT / "data" / "reference"
TSV_PATH = ru.GDTR_ROOT / "data" / "variants" / "clinvar_15gene_stratified.tsv"
CSV_PATH = PHASE_OUT_DIR / "ig_features.csv"
CKPT_PATH = PHASE_OUT_DIR / "_ig_checkpoint.json"
DONE_NAME = "ig"


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
            ref = d["ref"]; alt = d["alt"]
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                continue
            d["pos"] = int(d["pos"])
            d["stars"] = int(d.get("stars", 0))
            rows.append(d)
    return rows


def stable_order(rows: list[dict]) -> list[dict]:
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


def fetch_window(fa, chrom: str, pos: int, half: int) -> tuple[str, int]:
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
    local_idx = half
    return seq, local_idx


# ---------- Hidden-state hook for h_29 ----------

class H29Capture:
    """Forward hook on blocks.29 that retains the output tensor (with grad)."""

    def __init__(self):
        self.h = None

    def __call__(self, module, inputs, output):
        # Output may be a tuple in some Evo 2 blocks; keep first element if so.
        if isinstance(output, tuple):
            self.h = output[0]
        else:
            self.h = output
        return None


# ---------- Embedding interpolation via embedding_layer hook ----------

class EmbeddingInjector:
    """Replace embedding_layer's output with a pre-computed tensor.

    We capture the original embedding once, then on subsequent forwards we
    return alpha * embedding instead of looking up tokens. The injected tensor
    can carry requires_grad=True so backward flows.
    """

    def __init__(self):
        self.injected = None  # tensor [1, T, H] requires_grad=True (or False)

    def __call__(self, module, inputs, output):
        if self.injected is not None:
            return self.injected
        return output


def compute_ig_at_var(
    bundle,
    seq: str,
    local_idx: int,
    n_steps: int,
    h29_cap: "H29Capture",
    emb_inj: "EmbeddingInjector",
) -> float:
    """Compute IG attribution L2 norm at variant position for ‖h_29(var_pos)‖.

    Returns scalar IG score (float).
    """
    device = "cuda"
    from src.model_loader_evo2 import tokenize
    ids = tokenize(seq, bundle, device=device)  # [1, T]

    # Step 1: get the actual embedding (escape inference_mode).
    emb_inj.injected = None
    with torch.inference_mode(False), torch.no_grad():
        e_raw = bundle.sh.embedding_layer(ids)
        e_full = e_raw.clone().detach()
        del e_raw

    grad_accum = torch.zeros_like(e_full[0, local_idx], dtype=torch.float32)

    for k in range(n_steps):
        alpha = (k + 1) / n_steps  # left-Riemann
        with torch.inference_mode(False), torch.enable_grad():
            e_interp = (alpha * e_full).clone().detach().requires_grad_(True)
            emb_inj.injected = e_interp
            h29_cap.h = None
            out = bundle.sh(ids)
            del out
            h29 = h29_cap.h
            if h29 is None:
                raise RuntimeError("h29 not captured")
            h29 = h29.to(device)
            h_at_var = h29[0, local_idx].float()
            target = torch.linalg.vector_norm(h_at_var, ord=2)
            grad = torch.autograd.grad(
                target, e_interp, retain_graph=False, create_graph=False,
            )[0]
            grad_at_var = grad[0, local_idx].float().detach()
            grad_accum += grad_at_var
        emb_inj.injected = None
        del e_interp, grad, h29, h_at_var, target
        torch.cuda.empty_cache()

    grad_avg = grad_accum / float(n_steps)
    e_at_var = e_full[0, local_idx].float()
    attribution = e_at_var * grad_avg
    score = float(torch.linalg.vector_norm(attribution, ord=2).item())
    return score


def feature_columns() -> list[str]:
    return [
        "chrom", "pos", "ref", "alt", "gene", "category", "stars",
        "ig_score_ref", "ig_score_alt", "ig_score_delta",
        "ig_n_steps", "ig_target_layer", "ig_context_half",
    ]


def main() -> None:
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name=DONE_NAME):
        t_start = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)

        from src.model_loader_evo2 import load_evo2

        all_variants = stable_order(load_variants(TSV_PATH))
        LOG.info("total variants: %d", len(all_variants))

        start_idx = 0
        if CKPT_PATH.exists():
            ckpt = json.loads(CKPT_PATH.read_text())
            start_idx = int(ckpt.get("next_idx", 0))
            LOG.info("resuming from idx=%d", start_idx)
        cols = feature_columns()
        write_header = not CSV_PATH.exists() or start_idx == 0
        if start_idx == 0 and CSV_PATH.exists():
            CSV_PATH.unlink()
            write_header = True

        bundle = load_evo2()
        LOG.info("model loaded: %s", bundle.loaded_variant)

        # Set model to eval but keep parameters requires_grad=False to save mem.
        for p in bundle.sh.parameters():
            p.requires_grad_(False)
        bundle.sh.eval()

        h29_cap = H29Capture()
        emb_inj = EmbeddingInjector()
        h_handle = bundle.sh.blocks[TARGET_LAYER].register_forward_hook(h29_cap)
        e_handle = bundle.sh.embedding_layer.register_forward_hook(emb_inj)
        # Materialize all params out of inference_mode (clone breaks lineage).
        with torch.inference_mode(False):
            for name, p in list(bundle.sh.named_parameters()):
                mod_path, _, attr = name.rpartition(".")
                mod = bundle.sh.get_submodule(mod_path) if mod_path else bundle.sh
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

        fastas = FastaCache(REF_DIR)
        n_err = 0; n_nan = 0; n_done = 0

        f_csv = CSV_PATH.open("a", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=cols)
        if write_header:
            writer.writeheader(); f_csv.flush()

        try:
            for i in range(start_idx, len(all_variants)):
                var = all_variants[i]
                try:
                    cc = var["chrom"]
                    canon = f"chr{cc}" if not str(cc).startswith("chr") else cc
                    fa = fastas.get(cc)
                    ref_seq, local_idx = fetch_window(fa, canon, var["pos"], CONTEXT_HALF_IG)
                    if len(ref_seq) < 100:
                        n_err += 1; continue
                    ref_at = ref_seq[local_idx] if local_idx < len(ref_seq) else "N"
                    if ref_at != var["ref"] and ref_at != "N":
                        n_err += 1; continue
                    alt_seq = ref_seq[:local_idx] + var["alt"] + ref_seq[local_idx + 1:]

                    ig_ref = compute_ig_at_var(
                        bundle, ref_seq, local_idx, IG_STEPS, h29_cap, emb_inj,
                    )
                    torch.cuda.empty_cache()
                    ig_alt = compute_ig_at_var(
                        bundle, alt_seq, local_idx, IG_STEPS, h29_cap, emb_inj,
                    )
                    torch.cuda.empty_cache()

                    if not (np.isfinite(ig_ref) and np.isfinite(ig_alt)):
                        n_nan += 1; continue

                    row = {
                        "chrom": var["chrom"], "pos": var["pos"],
                        "ref": var["ref"], "alt": var["alt"],
                        "gene": var["gene"], "category": var["category"],
                        "stars": var["stars"],
                        "ig_score_ref": float(ig_ref),
                        "ig_score_alt": float(ig_alt),
                        "ig_score_delta": float(ig_alt - ig_ref),
                        "ig_n_steps": IG_STEPS,
                        "ig_target_layer": TARGET_LAYER,
                        "ig_context_half": CONTEXT_HALF_IG,
                    }
                    writer.writerow(row)
                    n_done += 1

                    if (i + 1) % 10 == 0:
                        f_csv.flush()
                        elapsed = time.time() - t_start
                        rate = max(n_done, 1) / max(elapsed, 1e-6)
                        eta = (len(all_variants) - i - 1) / max(rate, 1e-9)
                        LOG.info(
                            "var %d/%d  done=%d  rate=%.2f/s  ETA=%.1f min  err=%d  nan=%d",
                            i + 1, len(all_variants), n_done, rate, eta / 60, n_err, n_nan,
                        )
                    if (i + 1) % CHECKPOINT_EVERY == 0:
                        f_csv.flush()
                        CKPT_PATH.write_text(json.dumps({"next_idx": i + 1}))
                        LOG.info("checkpoint at idx=%d", i + 1)
                except Exception as e:
                    LOG.exception("var %d failed: %s", i, e)
                    n_err += 1
                    torch.cuda.empty_cache()
                    continue
        finally:
            try: h_handle.remove()
            except Exception: pass
            try: e_handle.remove()
            except Exception: pass
            f_csv.flush(); f_csv.close()

        CKPT_PATH.write_text(json.dumps({"next_idx": len(all_variants)}))
        out = {
            "phase": PHASE,
            "config": {
                "seed": SEED, "context_half_ig": CONTEXT_HALF_IG,
                "ig_steps": IG_STEPS, "target_layer": TARGET_LAYER,
                "model_variant": bundle.loaded_variant,
            },
            "n_total_variants": int(len(all_variants)),
            "n_processed_variants": int(n_done),
            "n_errors": int(n_err),
            "n_nans": int(n_nan),
            "wall_time_sec": float(time.time() - t_start),
        }
        ru.write_done(PHASE, PHASE_OUT_DIR, out, step_name=DONE_NAME)
        LOG.info("ig done in %.1f min  done=%d err=%d nan=%d",
                 (time.time() - t_start) / 60, n_done, n_err, n_nan)


if __name__ == "__main__":
    main()

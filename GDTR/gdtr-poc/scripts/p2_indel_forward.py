"""P2-INDEL-2 — frameshift / inframe indel forward (Plan §3.3.3, GPU PRIORITY 1).

Modeled on 31_phase3_main.py but ALLOWS indels (the SNV-only filter at
line 64 of 31_phase3_main.py is REMOVED).

For each indel:
  * forward ref window of length 6001 bp (CONTEXT_HALF=3000)
  * forward alt window of length (6000 + (len(alt) - len(ref))) — +1 for ins,
    −1 for del at the simplest boundary
  * to be robust to insertion / deletion shift, max|ΔD_cos| is computed over
    a ±5-token window around the variant position rather than at the exact
    token. Same for max|ΔD_jsd|.

R6 mitigation: at start, forward the same sequence twice (a no-op edit,
e.g., AAA → AAA) and assert max_abs_dD < 1e-3.

Output: ~/gDTR/results/p2_indel/variants_features_indel.csv
        with the 78 SNV columns + mc_class, indel_len, frame_class, frame_position.
        ~/gDTR/results/p2_indel/_status.json
        ~/gDTR/results/p2_indel/_done_forward
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GDTR_ROOT = Path(os.environ.get("GDTR_ROOT", "/path/to/gDTR")).expanduser()
GAMMA_COS = 0.39663
SEED = 42
CONTEXT_HALF = 3000
TOKEN_WINDOW = 5  # ±5 tokens around variant position


def setup_logging() -> logging.Logger:
    log = logging.getLogger("p2_indel_forward")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    return log


def fetch_window(fa, chrom_canonical: str, pos: int, ref: str):
    """Return (ref_seq, local_idx, chrom_len). For an indel at 1-based pos:
       The variant token is at local_idx = CONTEXT_HALF.
    """
    start = pos - 1 - CONTEXT_HALF
    end = pos - 1 + CONTEXT_HALF
    chrom_len = fa.get_reference_length(chrom_canonical)
    if start < 0:
        seq = "N" * (-start) + fa.fetch(chrom_canonical, 0, end).upper()
    elif end > chrom_len:
        seq = fa.fetch(chrom_canonical, start, chrom_len).upper()
        seq = seq + "N" * (end - chrom_len)
    else:
        seq = fa.fetch(chrom_canonical, start, end).upper()
    return seq, CONTEXT_HALF, chrom_len


def make_alt_seq(ref_seq: str, local_idx: int, ref: str, alt: str) -> str:
    """Apply the variant edit at local_idx.

    For an SNV: replace 1 base.
    For a deletion: ref="ACGT" alt="A" → drop 3 bases from local_idx+1.
    For an insertion: ref="A" alt="AGGG" → insert "GGG" after local_idx.
    """
    if len(ref) == 1 and len(alt) == 1:
        return ref_seq[:local_idx] + alt + ref_seq[local_idx + 1:]
    # left-anchored representation in VCF: first base is shared
    # ref="ACGT" alt="A" -> deletion of "CGT"  starts at local_idx+1
    # ref="A"    alt="AGGG" -> insertion of "GGG" after local_idx
    common_prefix = 0
    while (common_prefix < len(ref) and common_prefix < len(alt)
           and ref[common_prefix] == alt[common_prefix]):
        common_prefix += 1
    # variant starts at local_idx + common_prefix
    edit_pos = local_idx + common_prefix
    suffix_len_ref = len(ref) - common_prefix
    insertion = alt[common_prefix:]
    return ref_seq[:edit_pos] + insertion + ref_seq[edit_pos + suffix_len_ref:]


def settling_depth_interp_np(D: np.ndarray, gamma: float) -> tuple[float, bool]:
    L = D.shape[0]
    rmin = np.minimum.accumulate(D)
    below = rmin <= gamma
    if not below.any():
        return float(L), True
    first = int(below.argmax())
    if first == 0:
        return 1.0, False
    rm_above = rmin[first - 1]; rm_below = rmin[first]
    denom = rm_above - rm_below
    if denom < 1e-12:
        return float(first + 1), False
    frac = (rm_above - gamma) / denom
    frac = max(0.0, min(1.0, frac))
    return float(first) + frac, False


def feature_columns(n_layers: int) -> list[str]:
    cols = [
        "chrom", "pos", "ref", "alt", "gene", "category", "stars",
        "max_abs_dD", "signed_argmax", "argmax_layer", "dc_interp",
        "evo2_ref_loglik", "evo2_alt_loglik", "evo2_delta_loglik",
    ]
    for ell in range(n_layers):
        cols.append(f"dD_jsd_{ell}")
    for ell in range(n_layers):
        cols.append(f"dD_cos_{ell}")
    cols += ["mc_class", "indel_len", "frame_class", "frame_position"]
    return cols


def load_panel(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            d = dict(zip(header, line.rstrip("\n").split("\t")))
            try:
                d["pos"] = int(d["pos"])
                d["stars"] = int(d.get("stars", 0))
                d["indel_len"] = int(d.get("indel_len", 0))
            except ValueError:
                continue
            rows.append(d)
    return rows


def sanity_no_op(load_evo2_fn, tokenize_fn, extract_fn, jsd_fn, cosine_fn,
                 layer_names_fn, log: logging.Logger) -> bool:
    """R6: forward a sequence twice with no edit; assert max_abs_dD < 1e-3."""
    import torch  # noqa: F401
    seq = "A" * 4096
    bundle = load_evo2_fn()
    layer_names = layer_names_fn()
    n_layers = 32
    ids = tokenize_fn(seq, bundle, device="cuda")
    hs1 = extract_fn(bundle, ids, save_layers=layer_names)
    d1_jsd = jsd_fn(hs1, bundle, n_layers=n_layers).numpy()
    d1_cos = cosine_fn(hs1, n_layers=n_layers).numpy()
    del hs1
    ids2 = tokenize_fn(seq, bundle, device="cuda")
    hs2 = extract_fn(bundle, ids2, save_layers=layer_names)
    d2_jsd = jsd_fn(hs2, bundle, n_layers=n_layers).numpy()
    d2_cos = cosine_fn(hs2, n_layers=n_layers).numpy()
    del hs2
    diff_cos = float(np.abs(d2_cos - d1_cos).max())
    diff_jsd = float(np.abs(d2_jsd - d1_jsd).max())
    log.info("R6 sanity (no-op): max|Δcos|=%.3e max|Δjsd|=%.3e", diff_cos, diff_jsd)
    return (diff_cos < 1e-3) and (diff_jsd < 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run on first 10 indels")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-sanity", action="store_true",
                        help="skip the R6 no-op sanity check")
    args = parser.parse_args()

    log = setup_logging()
    out_dir = GDTR_ROOT / "results" / "p2_indel"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "_done_forward"
    if done_marker.exists() and not args.force:
        log.info("p2_indel_forward: _done_forward exists; skipping")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    status = {"step": "p2_indel_forward", "status": "RUNNING",
              "started_at": started_at, "smoke": args.smoke}
    (out_dir / "_status.json").write_text(json.dumps(status, indent=2))

    try:
        # Repo paths so we can import src.* modules from /path/to/gDTR/scripts.
        sys.path.insert(0, str(GDTR_ROOT))
        sys.path.insert(0, str(GDTR_ROOT / "scripts"))
        try:
            import _runner_utils as ru  # noqa: F401
            ru.add_repo_paths()
            ru.patch_safe_globals()
        except Exception:
            pass

        import torch
        import torch.nn.functional as F

        from src.constants_evo2 import N_LAYERS, VOCAB_SIZE
        from src.model_loader_evo2 import load_evo2, tokenize, lm_head_apply
        from src.logit_lens_evo2 import (
            extract_hidden_states, jsd_lens, all_layer_names,
        )
        from src.ur_gdtr_evo2 import cosine_lens

        torch.manual_seed(SEED); np.random.seed(SEED)

        panel_path = out_dir / "frameshift_panel.tsv"
        if not panel_path.exists():
            raise FileNotFoundError(f"missing {panel_path} (run p2_indel_select.py first)")
        rows = load_panel(panel_path)
        if args.smoke:
            rows = rows[:10]
        log.info("loaded %d panel rows", len(rows))

        if not args.no_sanity:
            ok = sanity_no_op(load_evo2, tokenize, extract_hidden_states,
                              jsd_lens, cosine_lens, all_layer_names, log)
            if not ok:
                raise RuntimeError("R6 no-op sanity check FAILED — refusing to proceed")
        else:
            log.info("R6 sanity skipped via --no-sanity")

        bundle = load_evo2()
        log.info("model: %s  N_LAYERS=%d", bundle.loaded_variant, N_LAYERS)
        layer_names = all_layer_names()

        tok = bundle.tokenizer
        token_id = {ch: int(np.asarray(tok.tokenize(ch), dtype=np.int64)[0])
                    for ch in "ACGT"}

        import pysam
        ref_dir = GDTR_ROOT / "data" / "reference"

        out_csv = out_dir / "variants_features_indel.csv"
        cols = feature_columns(N_LAYERS)
        f_csv = out_csv.open("w", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=cols)
        writer.writeheader()

        n_done = 0
        n_err = 0
        n_indels_seen = 0  # informational (Plan I9)
        opened_fa: dict[str, "pysam.FastaFile"] = {}

        try:
            for vi, var in enumerate(rows):
                try:
                    chrom = str(var["chrom"]).lstrip("chr")
                    canon = f"chr{chrom}"
                    fa = opened_fa.get(chrom)
                    if fa is None:
                        fa_path = ref_dir / f"chr{chrom}.fa"
                        if not fa_path.exists():
                            fa_path = ref_dir / f"{chrom}.fa"
                        if not fa_path.exists():
                            raise FileNotFoundError(f"no FASTA for chr{chrom}")
                        fa = pysam.FastaFile(str(fa_path))
                        opened_fa[chrom] = fa

                    ref = var["ref"]; alt = var["alt"]
                    is_indel = (len(ref) != 1 or len(alt) != 1)
                    if is_indel:
                        n_indels_seen += 1
                    log.info("[%d/%d] %s:%d %s>%s indel=%s",
                             vi + 1, len(rows), chrom, var["pos"], ref, alt, is_indel)

                    ref_seq, local_idx, _ = fetch_window(fa, canon, var["pos"], ref)
                    if len(ref_seq) < 100:
                        n_err += 1; continue
                    ref_at = ref_seq[local_idx] if local_idx < len(ref_seq) else "N"
                    if ref_at != ref[0] and ref_at != "N":
                        log.warning("ref base mismatch at %s:%d  fa=%s clinvar=%s — skip",
                                    chrom, var["pos"], ref_at, ref)
                        n_err += 1; continue

                    alt_seq = make_alt_seq(ref_seq, local_idx, ref, alt)
                    indel_len = len(alt) - len(ref)

                    # Forward ref
                    ids_ref = tokenize(ref_seq, bundle, device="cuda")
                    hs_ref = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
                    d_jsd_ref = jsd_lens(hs_ref, bundle, n_layers=N_LAYERS).numpy()
                    d_cos_ref = cosine_lens(hs_ref, n_layers=N_LAYERS).numpy()
                    h_norm_ref = hs_ref["norm"]
                    with torch.no_grad():
                        logits_ref = lm_head_apply(h_norm_ref, bundle).float()
                    pred_pos = max(0, local_idx - 1)
                    log_p_ref = F.log_softmax(logits_ref[0, pred_pos, :VOCAB_SIZE],
                                               dim=-1).cpu().numpy()
                    ref_char = ref[0]
                    ref_loglik = float(log_p_ref[token_id[ref_char]]) \
                        if ref_char in token_id else float("nan")
                    del hs_ref, ids_ref, logits_ref, h_norm_ref
                    torch.cuda.empty_cache()

                    # Forward alt
                    ids_alt = tokenize(alt_seq, bundle, device="cuda")
                    hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                    d_jsd_alt = jsd_lens(hs_alt, bundle, n_layers=N_LAYERS).numpy()
                    d_cos_alt = cosine_lens(hs_alt, n_layers=N_LAYERS).numpy()
                    h_norm_alt = hs_alt["norm"]
                    with torch.no_grad():
                        logits_alt = lm_head_apply(h_norm_alt, bundle).float()
                    log_p_alt = F.log_softmax(logits_alt[0, pred_pos, :VOCAB_SIZE],
                                               dim=-1).cpu().numpy()
                    alt_char = alt[0]
                    alt_loglik = float(log_p_alt[token_id[alt_char]]) \
                        if alt_char in token_id else float("nan")
                    del hs_alt, ids_alt, logits_alt, h_norm_alt
                    torch.cuda.empty_cache()

                    # Robust window: pick the column index with maximum |ΔD_jsd|
                    # over local_idx ± TOKEN_WINDOW, intersected with the alt
                    # sequence's available columns.
                    T_ref = d_jsd_ref.shape[1]
                    T_alt = d_jsd_alt.shape[1]
                    win_lo = max(0, local_idx - TOKEN_WINDOW)
                    win_hi = min(T_ref, T_alt, local_idx + TOKEN_WINDOW + 1)
                    if win_hi <= win_lo:
                        n_err += 1; continue
                    sub_jsd_ref = d_jsd_ref[:, win_lo:win_hi]
                    sub_jsd_alt = d_jsd_alt[:, win_lo:win_hi]
                    sub_cos_ref = d_cos_ref[:, win_lo:win_hi]
                    sub_cos_alt = d_cos_alt[:, win_lo:win_hi]

                    if not (np.isfinite(sub_jsd_ref).all() and np.isfinite(sub_jsd_alt).all()
                            and np.isfinite(sub_cos_ref).all() and np.isfinite(sub_cos_alt).all()):
                        n_err += 1; continue

                    dD_jsd_2d = sub_jsd_alt - sub_jsd_ref  # [L, W]
                    dD_cos_2d = sub_cos_alt - sub_cos_ref  # [L, W]
                    abs_dJ_2d = np.abs(dD_jsd_2d)
                    # pick col with max |ΔD_jsd| summed over layers (or just .max())
                    col_max = abs_dJ_2d.max(axis=0)
                    best_col = int(col_max.argmax())
                    dD_jsd = dD_jsd_2d[:, best_col]
                    dD_cos = dD_cos_2d[:, best_col]
                    abs_dJ = np.abs(dD_jsd)
                    arg = int(abs_dJ.argmax())
                    signed_argmax = float(dD_jsd[arg])
                    max_abs = float(abs_dJ.max())
                    c_ref, _ = settling_depth_interp_np(sub_cos_ref[:, best_col], GAMMA_COS)
                    c_alt, _ = settling_depth_interp_np(sub_cos_alt[:, best_col], GAMMA_COS)
                    dc_interp = float(c_alt - c_ref)
                    frame_position = (indel_len % 3) if indel_len else 0

                    row = {
                        "chrom": chrom, "pos": var["pos"],
                        "ref": ref, "alt": alt,
                        "gene": var.get("gene", ""),
                        "category": var.get("category", ""),
                        "stars": var.get("stars", 0),
                        "max_abs_dD": max_abs,
                        "signed_argmax": signed_argmax,
                        "argmax_layer": arg + 1,
                        "dc_interp": dc_interp,
                        "evo2_ref_loglik": ref_loglik,
                        "evo2_alt_loglik": alt_loglik,
                        "evo2_delta_loglik": alt_loglik - ref_loglik
                            if (np.isfinite(alt_loglik) and np.isfinite(ref_loglik))
                            else float("nan"),
                        "mc_class": var.get("mc_class", ""),
                        "indel_len": abs(indel_len),
                        "frame_class": var.get("frame_class", ""),
                        "frame_position": frame_position,
                    }
                    for ell in range(N_LAYERS):
                        row[f"dD_jsd_{ell}"] = float(dD_jsd[ell])
                    for ell in range(N_LAYERS):
                        row[f"dD_cos_{ell}"] = float(dD_cos[ell])
                    writer.writerow(row)
                    f_csv.flush()
                    n_done += 1
                except Exception as e:  # noqa: BLE001
                    log.exception("var %d failed: %s", vi, e)
                    n_err += 1
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
        finally:
            f_csv.close()

        log.info("p2_indel_forward: done=%d err=%d (indels seen=%d)",
                 n_done, n_err, n_indels_seen)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_forward",
            "status": "PASS" if n_done > 0 else "FAIL",
            "reason": f"forwarded {n_done} variants ({n_err} errors)",
            "metrics": {"n_done": n_done, "n_err": n_err,
                        "n_indels_seen": n_indels_seen},
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0 if n_done > 0 else 1,
            "wall_sec": time.time() - t0,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        if n_done > 0:
            done_marker.write_text(json.dumps({"ok": True, "n_done": n_done}, indent=2))
        log.info("p2_indel_forward DONE wall=%.1fs", time.time() - t0)

    except Exception as e:  # noqa: BLE001
        log.exception("p2_indel_forward FAILED: %s", e)
        finished_at = datetime.now(timezone.utc).isoformat()
        status = {
            "step": "p2_indel_forward",
            "status": "FAIL",
            "reason": f"{type(e).__name__}: {e}",
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 1,
        }
        (out_dir / "_status.json").write_text(json.dumps(status, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Phase 1.1 sanity sequences: 50 GC-matched + 50 dinuc-shuffled, 6kb each.

Reuses Phase 0 controls.py (uShuffle, GC sampler, intergenic chr17 extractor).
Output: /path/to/gDTR/data/baselines/phase1_sanity_seqs.{fa,json}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse Phase 0 reference code rather than rewriting.
sys.path.insert(0, "/path/to/gDTR-phase0/src")
from controls import (  # type: ignore  # noqa: E402
    dinuc_shuffle,
    extract_intergenic_chr17,
    gc_content,
)

OUT_FA = Path("/path/to/gDTR/data/baselines/phase1_sanity_seqs.fa")
OUT_JSON = Path("/path/to/gDTR/data/baselines/phase1_sanity_seqs.json")
CHR17_FA = "/path/to/gDTR/data/reference/chr17.fa"

SEED = 42
N_PER_GROUP = 50
LENGTH = 6000


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    OUT_FA.parent.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] sampling {N_PER_GROUP} GC-matched intergenic chr17 windows...")
    gc_seqs, gc_coords = extract_intergenic_chr17(
        fasta_path=CHR17_FA, length=LENGTH, n=N_PER_GROUP, seed=SEED,
    )
    if len(gc_seqs) != N_PER_GROUP:
        raise RuntimeError(f"intergenic sampler returned {len(gc_seqs)}/{N_PER_GROUP}")

    print(f"[phase1] dinuc-shuffling each of {N_PER_GROUP} seqs (uShuffle)...")
    shuf_seqs = []
    for i, s in enumerate(gc_seqs):
        shuf_seqs.append(dinuc_shuffle(s, n_shuffles=1, seed=SEED + i)[0])

    # Sanity: confirm ACGT only, length matches
    for label, group in (("gc_match", gc_seqs), ("dinuc_shuf", shuf_seqs)):
        for i, s in enumerate(group):
            if len(s) != LENGTH:
                raise RuntimeError(f"{label}_{i} wrong length: {len(s)}")
            if any(c not in "ACGT" for c in s):
                raise RuntimeError(f"{label}_{i} non-ACGT character present")

    print(f"[phase1] writing FASTA -> {OUT_FA}")
    with open(OUT_FA, "w") as f:
        for i, s in enumerate(gc_seqs):
            f.write(f">gc_match_{i}\n{s}\n")
        for i, s in enumerate(shuf_seqs):
            f.write(f">dinuc_shuf_{i}\n{s}\n")

    fa_md5 = md5_file(OUT_FA)

    meta = {
        "generation_seed": SEED,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "length_bp": LENGTH,
        "n_per_group": N_PER_GROUP,
        "fasta_path": str(OUT_FA),
        "fasta_md5": fa_md5,
        "fasta_size_bytes": OUT_FA.stat().st_size,
        "source_reference": CHR17_FA,
        "sequences": [],
    }
    for i, (s, coord) in enumerate(zip(gc_seqs, gc_coords)):
        meta["sequences"].append({
            "header": f"gc_match_{i}",
            "group": "gc_match",
            "length": len(s),
            "gc_content": round(gc_content(s), 6),
            "source_chrom": coord[0],
            "source_start": int(coord[1]),
            "source_end": int(coord[2]),
        })
    for i, s in enumerate(shuf_seqs):
        meta["sequences"].append({
            "header": f"dinuc_shuf_{i}",
            "group": "dinuc_shuf",
            "length": len(s),
            "gc_content": round(gc_content(s), 6),
            "shuffle_seed": SEED + i,
            "parent_header": f"gc_match_{i}",
        })

    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    gc_avg = sum(gc_content(s) for s in gc_seqs) / len(gc_seqs)
    shuf_gc_avg = sum(gc_content(s) for s in shuf_seqs) / len(shuf_seqs)
    print(f"[phase1] done in {time.time()-t0:.1f}s  "
          f"FASTA_MD5={fa_md5}  gc_match_avgGC={gc_avg:.4f}  "
          f"dinuc_shuf_avgGC={shuf_gc_avg:.4f}")


if __name__ == "__main__":
    main()

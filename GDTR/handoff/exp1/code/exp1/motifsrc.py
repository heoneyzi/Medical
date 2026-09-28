"""Motif acquisition: get everything that can be gotten, once, with provenance.

Why this module exists
----------------------
:mod:`exp1.motifs` says what a motif model *is* and refuses to invent one.  This
module is the other half: a declarative catalogue of every motif this project
can legitimately obtain, and one command that goes and obtains them.

The catalogue splits three ways, and the split is the point:

**Derivable** -- counted from the user's own GENCODE + hg38, no network.  Seven
of the nine biological signals this project cares about are in this class, which
is a better position than it first looks: a matrix counted from a hundred
thousand annotated sites in the same assembly the experiment runs on is more
defensible than a published matrix built on someone else's data, and it is
reproducible from two files a reviewer can download.

**Fetchable** -- real JASPAR CORE matrices, either from the official API or from
a flat file the user downloaded, always recorded with a SHA-256.  Nothing about
them is reconstructed locally.

**External** -- things that genuinely cannot be derived or fetched blind, above
all the branchpoint (GENCODE does not annotate branchpoints, so there is nothing
to count) and MaxEntScan (not distributed on PyPI).  These are *reported as
missing with their real source named*, never silently replaced.

Every derived recipe carries a consensus assertion -- the donor must come out
GT, the Kozak window must contain ATG at the annotated offset, the polyA
discovery must rediscover AATAAA.  A recipe with enough sites that fails its
assertion raises, because that is a bug in the extraction geometry, not a
property of the data.  A recipe with too few sites is skipped and reported,
because that is a property of the data and not a bug.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .motifs import (ACCEPTOR_EXONIC, ACCEPTOR_INTRONIC, BASES, DONOR_EXONIC,
                     DONOR_INTRONIC, COMPLEMENT, MotifProvenance, MotifSet, PWM,
                     count_matrix, encode, iter_introns, revcomp)


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    """One motif this project knows how to obtain, and how to check it."""

    name: str
    method: str                    # intron | anchor | hexamer | jaspar | external
    detail: str
    # -- validation, applied to the recovered consensus -------------------
    expect_at: int = -1            # index into the consensus
    expect: Tuple[str, ...] = ()   # any one of these must match there
    core_offset: int = 0
    core_len: int = 2
    min_sites: int = 500
    # -- method: intron ---------------------------------------------------
    intron_class: str = ""         # "GT-AG" | "GC-AG" | "AT-AC"
    end: str = ""                  # "donor" | "acceptor"
    # -- method: anchor ---------------------------------------------------
    feature: str = ""              # "start_codon" | "transcript_3p"
    up: int = 0
    down: int = 0
    # -- method: hexamer --------------------------------------------------
    region_up: int = 0
    candidates: Tuple[str, ...] = ()
    flank: int = 3
    # -- method: jaspar / external ----------------------------------------
    jaspar_id: str = ""
    external_source: str = ""


#: Everything obtainable, in one place.  Adding a motif means adding a row here
#: and nothing else; there is no second list to keep in sync.
CATALOG: Tuple[Recipe, ...] = (
    # ---- major spliceosome, the dominant class -------------------------
    Recipe("splice_donor_U2", "intron",
           "5' splice site of GT-AG introns (U2-type), 3 exonic + 6 intronic nt",
           intron_class="GT-AG", end="donor",
           expect_at=DONOR_EXONIC, expect=("GT",),
           core_offset=DONOR_EXONIC, core_len=2, min_sites=500),
    Recipe("splice_acceptor_U2", "intron",
           "3' splice site of GT-AG introns (U2-type), 20 intronic + 3 exonic nt",
           intron_class="GT-AG", end="acceptor",
           expect_at=ACCEPTOR_INTRONIC - 2, expect=("AG",),
           core_offset=ACCEPTOR_INTRONIC - 2, core_len=2, min_sites=500),

    # ---- GC-AG: ~0.8% of introns, a real and separately regulated class --
    Recipe("splice_donor_GCAG", "intron",
           "5' splice site of GC-AG introns, a minor but genuine U2 subclass",
           intron_class="GC-AG", end="donor",
           expect_at=DONOR_EXONIC, expect=("GC",),
           core_offset=DONOR_EXONIC, core_len=2, min_sites=200),
    Recipe("splice_acceptor_GCAG", "intron",
           "3' splice site of GC-AG introns",
           intron_class="GC-AG", end="acceptor",
           expect_at=ACCEPTOR_INTRONIC - 2, expect=("AG",),
           core_offset=ACCEPTOR_INTRONIC - 2, core_len=2, min_sites=200),

    # ---- AT-AC: predominantly U12-type, rare, different machinery -------
    Recipe("splice_donor_ATAC", "intron",
           "5' splice site of AT-AC introns, predominantly minor-spliceosome (U12) type",
           intron_class="AT-AC", end="donor",
           expect_at=DONOR_EXONIC, expect=("AT",),
           core_offset=DONOR_EXONIC, core_len=2, min_sites=40),
    Recipe("splice_acceptor_ATAC", "intron",
           "3' splice site of AT-AC introns",
           intron_class="AT-AC", end="acceptor",
           expect_at=ACCEPTOR_INTRONIC - 2, expect=("AC",),
           core_offset=ACCEPTOR_INTRONIC - 2, core_len=2, min_sites=40),

    # ---- translation start: anchored on an annotated feature ------------
    Recipe("kozak_start", "anchor",
           "Kozak context around annotated start codons, -6..+4 with ATG at +1",
           feature="start_codon", up=6, down=3,
           expect_at=6, expect=("ATG",),
           core_offset=6, core_len=3, min_sites=500),

    # ---- 3' end: discovered, not assumed --------------------------------
    Recipe("polyA_signal", "hexamer",
           "cleavage/polyadenylation hexamer, discovered by enrichment upstream "
           "of annotated transcript 3' ends and then counted in context",
           feature="transcript_3p", region_up=60,
           candidates=("AATAAA", "ATTAAA"), flank=3,
           expect_at=3, expect=("AATAAA", "ATTAAA"),
           core_offset=3, core_len=6, min_sites=500),

    # ---- fetchable ------------------------------------------------------
    Recipe("TATA_box", "jaspar", "TBP / TATA box", jaspar_id="MA0108",
           core_offset=0, core_len=6),
    Recipe("SP1_GC_box", "jaspar", "SP1 / GC box", jaspar_id="MA0079",
           core_offset=0, core_len=6),
    Recipe("CTCF", "jaspar", "CTCF insulator motif", jaspar_id="MA0139",
           core_offset=0, core_len=6),
    Recipe("NFYA", "jaspar", "NF-Y / CCAAT box", jaspar_id="MA0060",
           core_offset=0, core_len=5),

    # ---- genuinely external: named, never substituted -------------------
    Recipe("branchpoint", "external",
           "intronic branchpoint adenine; GENCODE does not annotate branchpoints, "
           "so there is nothing to count and no matrix is invented",
           external_source="Mercer et al. 2015 experimental branchpoint catalogue, "
                           "or branchpointer predictions (Signal et al. 2018)"),
    Recipe("maxentscan_5ss", "external",
           "MaxEntScan 5' splice-site strength; models position dependencies that "
           "a PWM cannot, so it is not approximated by one",
           external_source="Yeo & Burge (2004) MaxEntScan distribution "
                           "(me2x5); not available on PyPI"),
    Recipe("maxentscan_3ss", "external",
           "MaxEntScan 3' splice-site strength",
           external_source="Yeo & Burge (2004) MaxEntScan distribution (me2x3acc)"),
)


def catalog_table() -> List[Dict[str, object]]:
    """The catalogue as rows, for `step0-motifs plan`."""
    rows = []
    for r in CATALOG:
        how = {
            "intron": f"count {r.intron_class} intron {r.end}s in your GENCODE",
            "anchor": f"count windows around GENCODE {r.feature}",
            "hexamer": f"discover + count hexamer upstream of {r.feature}",
            "jaspar": f"fetch JASPAR {r.jaspar_id} (or load a downloaded file)",
            "external": "you must supply it",
        }[r.method]
        rows.append(dict(motif=r.name, method=r.method, network="yes" if r.method == "jaspar" else "no",
                         min_sites=r.min_sites if r.method in ("intron", "anchor", "hexamer") else "",
                         check=("consensus " + "/".join(r.expect) + f" at {r.expect_at}")
                               if r.expect else "", how=how,
                         source=r.external_source or ""))
    return rows


# --------------------------------------------------------------------------
# GTF anchors
# --------------------------------------------------------------------------

def iter_anchors(gtf_path: str, chrom: str, feature: str) -> Iterable[Tuple[int, str]]:
    """Yield (anchor position, strand) for a GTF feature, transcript-oriented.

    ``start_codon``    the A of the ATG
    ``transcript_3p``  the last transcribed base of the transcript
    """
    from .labels import _iter_gtf

    if feature == "start_codon":
        seen = set()
        for feat, s, e, strand, tid, _ in _iter_gtf(gtf_path, chrom):
            if feat != "start_codon" or (e - s) != 3:
                continue                      # split start codons are skipped
            pos = s if strand == "+" else e - 1
            key = (pos, strand)
            if key not in seen:
                seen.add(key)
                yield key
        return

    if feature == "transcript_3p":
        seen = set()
        for feat, s, e, strand, tid, _ in _iter_gtf(gtf_path, chrom):
            if feat != "transcript" or e <= s:
                continue
            pos = e - 1 if strand == "+" else s
            key = (pos, strand)
            if key not in seen:
                seen.add(key)
                yield key
        return

    raise ValueError(f"unknown anchor feature {feature!r}")


def anchor_windows(seq: np.ndarray, anchors: Iterable[Tuple[int, str]],
                   up: int, down: int) -> List[np.ndarray]:
    """Transcript-oriented windows of length ``up + down + 1`` around anchors."""
    out: List[np.ndarray] = []
    n = len(seq)
    width = up + down + 1
    for pos, strand in anchors:
        if strand == "+":
            lo, hi = pos - up, pos + down + 1
            if lo < 0 or hi > n:
                continue
            w = seq[lo:hi]
        else:
            lo, hi = pos - down, pos + up + 1
            if lo < 0 or hi > n:
                continue
            w = revcomp(seq[lo:hi])
        if len(w) == width:
            out.append(w)
    return out


def intron_windows(seq: np.ndarray, introns: Sequence[Tuple[int, int, str]]
                   ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Paired (donor, acceptor) windows per intron, transcript-oriented.

    Paired rather than two independent lists, because the intron's *class* is a
    joint property of its two ends -- an intron whose donor window falls off the
    chromosome must not contribute its acceptor to a class it was never assigned.
    """
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    n = len(seq)
    dw, aw = DONOR_EXONIC + DONOR_INTRONIC, ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC
    for s, e, strand in introns:
        if strand == "+":
            d0, d1 = s - DONOR_EXONIC, s + DONOR_INTRONIC
            a0, a1 = e - ACCEPTOR_INTRONIC, e + ACCEPTOR_EXONIC
            if d0 < 0 or a1 > n:
                continue
            d, a = seq[d0:d1], seq[a0:a1]
        else:
            d0, d1 = e - DONOR_INTRONIC, e + DONOR_EXONIC
            a0, a1 = s - ACCEPTOR_EXONIC, s + ACCEPTOR_INTRONIC
            if a0 < 0 or d1 > n:
                continue
            d, a = revcomp(seq[d0:d1]), revcomp(seq[a0:a1])
        if len(d) == dw and len(a) == aw:
            out.append((d, a))
    return out


def intron_class(donor_w: np.ndarray, acceptor_w: np.ndarray) -> str:
    """``GT-AG`` etc., read off the two windows in transcript orientation."""
    d = bytes(donor_w[DONOR_EXONIC:DONOR_EXONIC + 2]).decode(errors="replace")
    a = bytes(acceptor_w[ACCEPTOR_INTRONIC - 2:ACCEPTOR_INTRONIC]).decode(errors="replace")
    return f"{d}-{a}"


# --------------------------------------------------------------------------
# hexamer discovery
# --------------------------------------------------------------------------

def discover_hexamer(regions: Sequence[np.ndarray], k: int = 6,
                     top: int = 10, seed: int = 42, min_count: int = 50
                     ) -> Tuple[List[Dict[str, object]], np.random.Generator]:
    """Rank k-mers by enrichment over a composition-matched shuffle.

    The background is the same regions with their bases permuted, so a hexamer
    that is merely a consequence of the AT-richness of 3' ends does not come out
    enriched.  This is what makes the polyA signal *discovered* rather than
    assumed: if AATAAA is not in the top of this list, the recipe refuses.
    """
    rng = np.random.default_rng(seed)
    obs: Counter = Counter()
    exp: Counter = Counter()
    for w in regions:
        text = bytes(w).decode(errors="replace")
        if "N" in text:
            continue
        for i in range(len(text) - k + 1):
            obs[text[i:i + k]] += 1
        sh = list(text)
        rng.shuffle(sh)
        sh = "".join(sh)
        for i in range(len(sh) - k + 1):
            exp[sh[i:i + k]] += 1

    rows = []
    for kmer, o in obs.items():
        if o < min_count:
            continue
        e = exp.get(kmer, 0) + 1.0
        rows.append(dict(kmer=kmer, observed=int(o), expected=round(float(e), 1),
                         log2_enrichment=round(float(np.log2(o / e)), 3)))
    rows.sort(key=lambda r: r["log2_enrichment"], reverse=True)
    return rows[:top], rng


def hexamer_context_windows(regions: Sequence[np.ndarray], hexamer: str,
                            flank: int) -> List[np.ndarray]:
    """Windows centred on the first occurrence of ``hexamer`` in each region."""
    k = len(hexamer)
    out: List[np.ndarray] = []
    for w in regions:
        text = bytes(w).decode(errors="replace")
        at = text.find(hexamer)
        if at < 0:
            continue
        lo, hi = at - flank, at + k + flank
        if lo < 0 or hi > len(w):
            continue
        out.append(w[lo:hi])
    return out


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def _check_consensus(pwm: PWM, r: Recipe) -> None:
    if not r.expect or r.expect_at < 0:
        return
    width = max(len(x) for x in r.expect)
    got = pwm.consensus()[r.expect_at:r.expect_at + width]
    if got not in r.expect:
        raise AssertionError(
            f"{r.name}: recovered consensus reads {got!r} at position {r.expect_at}, "
            f"expected one of {r.expect}. The window geometry or strand handling is "
            "wrong -- this matrix must not be used."
        )


def acquire(
    fasta: str,
    gtf: str,
    chroms: Sequence[str],
    panel_chroms: Sequence[str] = (),
    recipes: Sequence[Recipe] = CATALOG,
    jaspar_dir: Optional[str] = None,
    allow_network: bool = True,
    out_dir: Optional[str] = None,
) -> Tuple[MotifSet, Dict[str, object]]:
    """Obtain every motif in ``recipes`` that can be obtained, in one pass.

    Returns the motif set and a report naming what was obtained, what was
    skipped and why, and what the user has to supply by hand.  Nothing is
    approximated: a recipe either produces a matrix from real counts, or it
    appears in the report as missing.
    """
    overlap = sorted(set(chroms) & set(panel_chroms))
    if overlap:
        raise ValueError(
            f"refusing to derive motifs on {overlap}: those carry the analysis "
            "panel, so a motif model fit there would have seen its own test set."
        )
    if not chroms:
        raise ValueError("no chromosomes given to derive motifs from")

    from pyfaidx import Fasta

    fa = Fasta(fasta, as_raw=True, sequence_always_upper=True)

    wanted_classes = {r.intron_class for r in recipes if r.method == "intron"}
    anchor_feats = {r.feature for r in recipes if r.method == "anchor"}
    hex_feats = {r.feature for r in recipes if r.method == "hexamer"}

    by_class: Dict[str, Dict[str, List[np.ndarray]]] = {
        c: dict(donor=[], acceptor=[]) for c in wanted_classes}
    by_anchor: Dict[Tuple[str, int, int], List[np.ndarray]] = {}
    by_region: Dict[Tuple[str, int], List[np.ndarray]] = {}
    base_counts = np.zeros(4, dtype=np.int64)
    per_chrom: Dict[str, Dict[str, int]] = {}
    class_counts: Counter = Counter()

    for chrom in chroms:
        seq = np.frombuffer(str(fa[chrom][:]).upper().encode(), dtype=np.uint8)
        idx = encode(seq)
        for i in range(4):
            base_counts[i] += int((idx == i).sum())

        introns = list(iter_introns(gtf, chrom))
        pairs = intron_windows(seq, introns)
        for d, a in pairs:
            cls = intron_class(d, a)
            class_counts[cls] += 1
            if cls in by_class:
                by_class[cls]["donor"].append(d)
                by_class[cls]["acceptor"].append(a)

        n_anch = 0
        for r in recipes:
            if r.method == "anchor" and r.feature in anchor_feats:
                key = (r.feature, r.up, r.down)
                if key not in by_anchor:
                    by_anchor[key] = []
                ws = anchor_windows(seq, iter_anchors(gtf, chrom, r.feature),
                                    r.up, r.down)
                by_anchor[key] += ws
                n_anch += len(ws)
            elif r.method == "hexamer" and r.feature in hex_feats:
                key = (r.feature, r.region_up)
                if key not in by_region:
                    by_region[key] = []
                by_region[key] += anchor_windows(
                    seq, iter_anchors(gtf, chrom, r.feature), r.region_up, 0)

        per_chrom[chrom] = dict(introns=len(introns), intron_windows=len(pairs),
                                anchor_windows=n_anch)

    background = (base_counts + 1.0) / (base_counts + 1.0).sum()
    ref = (f"GENCODE={Path(gtf).name}; assembly={Path(fasta).name}; "
           f"chroms={','.join(chroms)}")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ms = MotifSet(background=background,
                  note=f"acquired from {','.join(chroms)} ({len(recipes)} recipes attempted)")
    obtained: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    external: List[Dict[str, object]] = []
    discovery: Dict[str, object] = {}

    def _finish(r: Recipe, windows: Sequence[np.ndarray], width: int,
                extra: str = "") -> None:
        counts, used = count_matrix(windows, width)
        if used < r.min_sites:
            skipped.append(dict(motif=r.name, reason=(
                f"only {used} usable sites (need {r.min_sites}); "
                "this is a property of the annotation, not an error")))
            return
        pwm = PWM.from_counts(
            r.name, counts,
            MotifProvenance(source="derived", detail=r.detail + extra,
                            n_sites=used, uri=str(gtf), reference=ref,
                            built_at=stamp),
            core_offset=r.core_offset, core_len=r.core_len)
        _check_consensus(pwm, r)
        ms.add(pwm)
        ic = pwm.information_content(background)
        obtained.append(dict(motif=r.name, method=r.method, n_sites=used,
                             consensus=pwm.consensus(),
                             total_ic_bits=round(float(ic.sum()), 3),
                             ic_bits=[round(float(x), 3) for x in ic]))

    for r in recipes:
        if r.method == "intron":
            pool = by_class.get(r.intron_class, {}).get(r.end, [])
            width = (DONOR_EXONIC + DONOR_INTRONIC) if r.end == "donor" \
                else (ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC)
            _finish(r, pool, width)

        elif r.method == "anchor":
            pool = by_anchor.get((r.feature, r.up, r.down), [])
            _finish(r, pool, r.up + r.down + 1)

        elif r.method == "hexamer":
            regions = by_region.get((r.feature, r.region_up), [])
            if len(regions) < r.min_sites:
                skipped.append(dict(motif=r.name,
                                    reason=f"only {len(regions)} regions (need {r.min_sites})"))
                continue
            sample = regions if len(regions) <= 20000 else regions[:20000]
            top, _ = discover_hexamer(sample, k=len(r.candidates[0]) if r.candidates else 6)
            discovery[r.name] = top
            rank = next((i for i, row in enumerate(top, 1)
                         if row["kmer"] in r.candidates), None)
            found = top[rank - 1]["kmer"] if rank else None
            if found is None:
                raise AssertionError(
                    f"{r.name}: none of {r.candidates} is enriched upstream of "
                    f"{r.feature}. Top hexamers were "
                    f"{[row['kmer'] for row in top[:5]]}. Either the anchor is wrong "
                    "or this annotation does not support the recipe -- no matrix built."
                )
            ctx = hexamer_context_windows(regions, found, r.flank)
            _finish(r, ctx, len(found) + 2 * r.flank,
                    extra=(f"; hexamer {found} discovered by enrichment "
                           f"(rank {rank}/{len(top)}, log2 "
                           f"{top[rank - 1]['log2_enrichment']}), matrix counted on "
                           "its occurrences rather than on a fixed offset"))

        elif r.method == "jaspar":
            rec = _acquire_jaspar(r, ms, jaspar_dir, allow_network, out_dir)
            (obtained if rec.get("ok") else skipped).append(
                {k: v for k, v in rec.items() if k != "ok"})

        elif r.method == "external":
            external.append(dict(motif=r.name, detail=r.detail,
                                 source=r.external_source))

    report = dict(
        chroms=list(chroms), panel_chroms=list(panel_chroms),
        per_chrom=per_chrom,
        intron_classes=dict(sorted(class_counts.items(),
                                   key=lambda kv: -kv[1])[:8]),
        background=dict(zip(BASES, background.round(4).tolist())),
        obtained=obtained, skipped=skipped, requires_external=external,
        hexamer_discovery=discovery,
        n_obtained=len(ms),
    )
    return ms, report


def _acquire_jaspar(r: Recipe, ms: MotifSet, jaspar_dir: Optional[str],
                    allow_network: bool, out_dir: Optional[str]) -> Dict[str, object]:
    """A JASPAR matrix from a local file if present, else from the API."""
    from .motifs import fetch_jaspar, load_jaspar

    # 1. a file the user already downloaded, matched by matrix id
    if jaspar_dir:
        d = Path(jaspar_dir)
        hits = sorted(d.glob(f"{r.jaspar_id}*.jaspar")) + sorted(d.glob(f"{r.jaspar_id}*.pfm"))
        if not hits:
            for flat in sorted(d.glob("*.txt")) + sorted(d.glob("*.jaspar")):
                try:
                    for pwm in load_jaspar(str(flat)):
                        if pwm.name.startswith(r.jaspar_id):
                            pwm.name = r.name
                            pwm.core_offset = min(r.core_offset, pwm.length - 1)
                            pwm.core_len = r.core_len
                            ms.add(pwm)
                            return dict(ok=True, motif=r.name, method="jaspar",
                                        n_sites=int(pwm.n_observations),
                                        consensus=pwm.consensus(),
                                        source=f"{flat.name} ({r.jaspar_id})")
                except Exception:
                    continue
        for h in hits:
            pwm = load_jaspar(str(h))[0]
            pwm.name = r.name
            pwm.core_offset = min(r.core_offset, pwm.length - 1)
            pwm.core_len = r.core_len
            ms.add(pwm)
            return dict(ok=True, motif=r.name, method="jaspar",
                        n_sites=int(pwm.n_observations),
                        consensus=pwm.consensus(), source=h.name)

    # 2. the official API
    if not allow_network:
        return dict(ok=False, motif=r.name,
                    reason=f"network disabled and no local file for {r.jaspar_id}")
    try:
        got = fetch_jaspar([r.jaspar_id], out_dir or "motifs/jaspar")
        mid, rec = next(iter(got.items()))
        pwm = load_jaspar(rec["path"])[0]
        pwm.name = r.name
        pwm.core_offset = min(r.core_offset, pwm.length - 1)
        pwm.core_len = r.core_len
        ms.add(pwm)
        return dict(ok=True, motif=r.name, method="jaspar",
                    n_sites=int(pwm.n_observations), consensus=pwm.consensus(),
                    source=f"{rec['resolved_id']} sha256={rec['sha256'][:12]}")
    except Exception as exc:
        return dict(ok=False, motif=r.name, reason=(
            f"{type(exc).__name__}: {exc}. Download "
            f"{r.jaspar_id} from https://jaspar.elixir.no/ on a machine with "
            "access and re-run with --jaspar-dir pointing at it."))
